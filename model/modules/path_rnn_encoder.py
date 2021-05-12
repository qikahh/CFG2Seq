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
class PathRNNEncoder(nn.Module):
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

        self.max_path_length = config.max_path_length
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
        

        self.path_lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.order_lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.unit_linear = nn.Linear(config.rnn_size*self.num_directions, config.embedding_size)
        
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

    def _unit_to_path(
        self,
        unit_features: torch.Tensor, #[unit num; embedding size]
        units_per_data: List[int],
        paths: torch.Tensor, #[path length; path num]
        path_per_data: List[int],
    ):
        with torch.no_grad():
            batch_size = len(units_per_data)
            # unit_path = paths.new_zeros()
            paths_begin = 0
            units_begin = 0
            is_contain_end_id, first_end_pos = torch.max(paths == -1, dim=0)
            first_end_pos[~is_contain_end_id] = paths.shape[0]
            max_length = min(max(first_end_pos),self.max_path_length)
            # 将路径中unit填充入路径
            paths_begin = 0
            units_begin = 0
        # #[path length; path num; embedding size]
        all_expaths = unit_features.new_zeros((max_length, paths.size(1), unit_features.size(-1)))
        expaths_mask = (paths==0).new_ones((max_length, paths.size(1)))
        zero_unit = unit_features.new_full((1, unit_features.size(-1)),self.pad_id)
        for i in range(batch_size):
            paths_num = path_per_data[i]
            batch_paths = paths[:,paths_begin:paths_begin+paths_num]
            units_num = units_per_data[i]
            batch_units = unit_features[units_begin:units_begin+units_num]
            for j in range(paths_num):  
                backward_path = batch_paths[:first_end_pos[paths_begin+j],j]
                path_context = [batch_units[unit] if (unit < units_num) else zero_unit[0]
                                     for unit in backward_path[:first_end_pos[paths_begin+j]]]
                path_context = torch.stack(path_context,dim = 0)
                path_length = min(path_context.size(0), max_length)
                path_context = path_context[:path_length]
                path_mask = (backward_path < 0) + (backward_path > units_num)
                path_mask = path_mask[:path_length]
                all_expaths[:path_context.size(0),paths_begin+j] = path_context
                expaths_mask[:path_mask.size(0),paths_begin+j] = path_mask
            paths_begin += paths_num
            units_begin += units_num

        return all_expaths, expaths_mask

    def _splice_expaths(
        self, 
        expaths: torch.Tensor, # []
        expath_mask: torch.Tensor, 
        old_order: torch.Tensor,
        old_order_mask: torch.Tensor,
        paths: torch.Tensor,
        paths_per_label: List[int],
        units_per_label: List[int],
    ):
        is_contain_end_id, first_end_pos = torch.max(paths == -1, dim=0)
        first_end_pos[~is_contain_end_id] = paths.shape[0]
        batch_size = len(paths_per_label)
        embedding_size = expaths.size(-1)
        max_order_length = max(paths_per_label)*expaths.size(0) + old_order.size(0)
        batched_order = expaths.new_zeros((max_order_length, batch_size, expaths.size(-1)))
        batched_mask = expath_mask.new_ones((max_order_length, batch_size))
        path_begin = 0
        unit_begin = 0
        for batch in range(batch_size):
            batch_expaths = expaths[:,path_begin:path_begin+paths_per_label[batch]].permute(1,0,2).reshape((-1,embedding_size))
            batch_expaths_mask = expath_mask[:,path_begin:path_begin+paths_per_label[batch]].permute(1,0).reshape((-1,1))
            batch_expaths = torch.cat([old_order[:,batch],batch_expaths],dim=0)
            batch_expaths_mask = torch.cat([old_order_mask[:,batch],batch_expaths_mask[:,0]],dim=0)
            batched_order[:batch_expaths.size(0),batch] = batch_expaths
            batched_mask[:batch_expaths_mask.size(0),batch] = batch_expaths_mask
            path_begin += paths_per_label[batch]
            unit_begin += units_per_label[batch]
            torch.cuda.empty_cache()

        return batched_order, batched_mask

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
        
    def _order_encoder(self, features, mask):
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
        output, (_, _) = self.order_lstm(packed_path_states)
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        output = output[:,reverse_sort_indices]
        output = self.unit_linear(output)
        mask = mask[:max(first_pad_pos)]
        return output, mask

    def _path_encoder(self, features, mask):
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
        output, (_, _) = self.path_lstm(packed_path_states)
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        output = output[:,reverse_sort_indices]
        output = self.unit_linear(output)
        mask = mask[:max(first_pad_pos)]
        return output, mask

    def forward(
        self, 
        units: torch.Tensor,
        unit_per_data: List[int], 
        paths: torch.Tensor, 
        path_per_data: List[int]
    ) -> torch.Tensor:
        # [total units; embedding_size]
        encoded_tokens = self._token_encoder(units)
        # [max path length; total paths; embedding_size] [max path length; total paths]
        expath_features, expath_mask = self._unit_to_path(encoded_tokens, unit_per_data, paths, path_per_data)
        # [max path length; total paths, embedding_size]
        path_length = min(expath_features.size(0), self.max_path_length)
        expath_features = expath_features[:path_length]
        expath_mask = expath_mask[:path_length]
        expath_features, expath_mask = self._path_encoder(expath_features, expath_mask)

         # [max path length; batch_size, embedding_size]
        batched_features, feature_mask = self._cut_batch(encoded_tokens, unit_per_data)
        # [max path length; batch_size, embedding_size]
        path_length = min(batched_features.size(0), self.max_path_length)
        batched_features = batched_features[:path_length]
        feature_mask = feature_mask[:path_length]
        batched_features, feature_mask = self._path_encoder(batched_features, feature_mask)
        
        # [concat length; batch_size, embedding_size] [concat length; batch_size]
        batched_features, feature_mask = self._splice_expaths(expaths=expath_features, expath_mask=expath_mask
                                                    , old_order=batched_features, old_order_mask=feature_mask
                                                    , paths=paths, paths_per_label=path_per_data, units_per_label=unit_per_data)
        return batched_features, feature_mask
