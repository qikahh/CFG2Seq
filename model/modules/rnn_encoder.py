from typing import Dict, List

import torch
import numpy
from omegaconf import DictConfig
from torch import nn
from torch._C import ListType, device
from torch.nn.functional import batch_norm
from model.modules.attention import LocalAttention
from torch.nn.modules.sparse import Embedding

from utils.training import cut_encoded_contexts
from datamodule.data_classes import FROM_TOKEN, TO_TOKEN, PATH_NODES
from utils.vocabulary import Vocabulary, SOS, PAD, UNK, EOS, CLS, SEP

from .positional_encoder import PositionalEncoding

# 参考 Reinforcement-Learning-Guided Source Code Summarization via Hierarchical Attention，运用双层的Atten-LSTM
# 先LSTM生成每个statement自己的特征表示向量，再按照路径顺序更新路径上的state特征
# 最后WeightSum聚合对应于同一个state的特征
class RNNEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        embedding: Embedding,
        vocabulary: Vocabulary,
        n_tokens: int,
        token_pad_id: int,
    ):
        super().__init__()
        self._vocabulary = vocabulary
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self._negative_value = True

        self.pad_id = token_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1
        self.embedding_size = config.embedding_size
        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_size = config.rnn_size

        self.token_embedding = embedding
        self.dropout_rnn = nn.Dropout(config.rnn_dropout)
        self.token_lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.token_linear = nn.Linear(config.rnn_size*self.num_directions, config.embedding_size)
        self.token_norm = nn.LayerNorm(config.embedding_size)
        self.token_tanh = nn.Tanh()
        self.token_weight = LocalAttention(config.embedding_size)

        self.state_lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.unit_linear = nn.Linear(config.rnn_size*self.num_directions, config.embedding_size)    
        self.unit_norm = nn.LayerNorm(config.embedding_size)
        self.unit_tanh = nn.Tanh()
        self.unit_weight = LocalAttention(config.embedding_size)
        
    def _cut_batch(self, features:torch.Tensor, num_per_batch: List[int]):
        batch_size = len(num_per_batch)
        max_context_len = max(num_per_batch)
        batched_features = features.new_zeros((max_context_len, batch_size, features.shape[-1]))
        attention_mask = (features == 1).new_ones((max_context_len, batch_size))
        cum_sums = numpy.cumsum(num_per_batch)
        start_of_segments = numpy.append([0], cum_sums[:-1])
        slices = [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
        for i, (cur_slice, cur_size) in enumerate(zip(slices, num_per_batch)):
            batched_features[:cur_size, i] = features[cur_slice]
            attention_mask[:cur_size, i] = False

        return batched_features, attention_mask

    def _token_encoder(self, units: torch.Tensor) -> torch.Tensor:
        # [unit_length; total units; embedding size]
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(units == self.pad_id, dim=0)
            first_pad_pos[~is_contain_pad_id] = units.shape[0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        embedding_tokens = self.token_embedding(units)
        states_sort = embedding_tokens[:,sort_indices]
        packed_path_states = nn.utils.rnn.pack_padded_sequence(states_sort, sorted_path_lengths)
        # h_init = embedding_tokens.new_ones(self.rnn_num_layers*self.num_directions, units.shape[1], self.rnn_size).randn()
        # c_init = embedding_tokens.new_ones(self.rnn_num_layers*self.num_directions, units.shape[1], self.rnn_size).randn()
        # [unit length; total units; rnn size]
        output, (_, _) = self.token_lstm(packed_path_states)
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        output = output[:,reverse_sort_indices]
        token_mask = (units == self.pad_id)[:output.shape[0]]
        output = self.token_linear(output)

        output_q = self.token_norm(output)
        output_q = self.token_tanh(output_q)
        order_weight = self.token_weight(output_q, token_mask)

        output = torch.matmul(order_weight.permute(1,2,0), output.permute(1,0,2)).squeeze()
        return output
        
    def _unit_encoder(self, features, mask):
        # [path_length; batch size; embedding size]
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(mask == True, dim=0)
            first_pad_pos[~is_contain_pad_id] = features.shape[0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        states_sort = features[:,sort_indices]
        packed_path_states = nn.utils.rnn.pack_padded_sequence(states_sort, sorted_path_lengths)
        # [unit length; total units; embedding size]
        output, (_, _) = self.state_lstm(packed_path_states)
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        output = output[:,reverse_sort_indices]
        output = self.unit_linear(output)

        #output_q = self.unit_tanh(self.unit_norm(output))
        #order_weight = self.unit_weight(output_q, mask)
        #output = torch.matmul(order_weight.permute(1,2,0), output.permute(1,0,2)).squeeze()
        return output

    def forward(
        self, 
        units: torch.Tensor,
        unit_per_data: List[int], 
        paths: torch.Tensor, 
        path_per_data: List[int]
    ) -> torch.Tensor:
        # [total units; embedding_size]
        encoded_tokens = self._token_encoder(units)
        # [max path length; batch_size, embedding_size]
        batched_features, feature_mask = self._cut_batch(encoded_tokens, unit_per_data)
        # [max path length; batch_size, embedding_size]
        batched_features = self._unit_encoder(batched_features, feature_mask)
        return batched_features, feature_mask
