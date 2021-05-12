from model.modules.attention import LocalAttention
from typing import Dict, List

import torch
import numpy
from omegaconf import DictConfig
from torch import nn
from torch._C import ListType, device
from torch.nn.functional import batch_norm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.sparse import Embedding

from utils.training import cut_encoded_contexts
from datamodule.data_classes import FROM_TOKEN, TO_TOKEN, PATH_NODES
from utils.vocabulary import Vocabulary, SOS, PAD, UNK, EOS, CLS, SEP

from .positional_encoder import PositionalEncoding


class PathNodeEncoder(nn.Module):
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
        self._negative_value = -1e9

        self.pad_id = token_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1

        self.token_embedding = embedding

        self.dropout_rnn = nn.Dropout(config.rnn_dropout)
        self.unit_lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.concat_layer = nn.Linear(self.num_directions*config.rnn_size, config.embedding_size, bias=False)

        self.path_layers = config.path_layers
        self.final_layers = config.final_layers
        self.attn_heads = config.attn_heads
        self.ffn_dim = config.ffn_dim
        self.dropout = config.dropout
        self.hidden = config.embedding_size

        self.positionalencoding = PositionalEncoding(d_model=self.hidden)
        self.token_weight = LocalAttention(self.hidden)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads,
                                                                     dim_feedforward=self.ffn_dim, dropout=self.dropout)
        self.expath_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.path_layers)
        self.order_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.path_layers)
        self.final_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.final_layers)

    # splices multiple execution paths together as input to TransformerDecoder
    def _splice_expaths(
        self, 
        expaths: torch.Tensor, 
        mask: torch.Tensor, 
        old_order: torch.Tensor,
        old_order_mask: torch.Tensor,
        paths: torch.Tensor,
        paths_per_label: List[int],
        units_per_label: List[int],
        unit_length: int
    ):
        is_contain_end_id, first_end_pos = torch.max(paths == -1, dim=0)
        first_end_pos[~is_contain_end_id] = paths.shape[0]
        batch_size = len(paths_per_label)
        max_order_length = max(units_per_label)*unit_length
        batched_order = expaths.new_zeros((max_order_length, batch_size, expaths.shape[-1]))
        batched_mask = mask.new_ones((max_order_length, batch_size))
        path_begin = 0
        unit_begin = 0
        for batch in range(batch_size):
            num_orders = torch.ones(units_per_label[batch], dtype=int)
            batch_expaths = expaths[:,path_begin:path_begin+paths_per_label[batch]]
            batch_paths = paths[:,path_begin:path_begin+paths_per_label[batch]]
            batch_mask = mask[:,path_begin:path_begin+paths_per_label[batch]]
            for i in range(paths_per_label[batch]):
                for j, unit_index in enumerate(batch_paths[:first_end_pos[path_begin+i],i]):
                    if (unit_index >= units_per_label[batch]) or batch_mask[j,i]:
                        continue
                    num_orders[unit_index] += 1
            # 按顺序排列每个unit的多个特征表示
            order = expaths.new_zeros((unit_length, max(units_per_label), torch.max(num_orders).item(), expaths.size(-1)))
            add_order = expaths.new_zeros((unit_length, max(units_per_label), expaths.size(-1)))
            order_mask = mask.new_ones((unit_length, max(units_per_label), max(num_orders)))
            add_mask = mask.new_ones((unit_length, max(units_per_label)))
            order_index = torch.zeros(max(units_per_label), dtype=int)
            for i in range(paths_per_label[batch]):
                for j, unit_index in enumerate(batch_paths[:first_end_pos[path_begin+i],i]):
                    if (unit_index >= units_per_label[batch]) or batch_mask[j*unit_length,i]:
                        continue
                    order[:,unit_index, order_index[unit_index]] = batch_expaths[j*unit_length:(j+1)*unit_length,i]
                    order_mask[:,unit_index, order_index[unit_index]] = batch_mask[j*unit_length:(j+1)*unit_length,i]
                    order_index[unit_index] += 1 
            for i in range(units_per_label[batch]):
                order[:,i,order_index[i]] = old_order[i*unit_length:(i+1)*unit_length, batch]
                order_mask[:,i,order_index[i]] = old_order_mask[i*unit_length:(i+1)*unit_length, batch]
                order_weight = self.token_weight(order[:,i], order_mask[:,i])
                add_order[:,i] = torch.matmul(order_weight.permute(0,2,1), order[:,i]).squeeze()
                add_mask[:,i] = (torch.sum((order_mask[:,i]==False), dim=1) == 0)
            batched_order[:,batch] = torch.cat([add_order[:,i] for i in range(order.size(1))], dim=0)
            batched_mask[:,batch] = torch.cat([add_mask[:,i] for i in range(order_mask.size(1))], dim=0)
            path_begin += paths_per_label[batch]
            unit_begin += units_per_label[batch]

        return batched_order, batched_mask


        
    def _cut_batch(self, features:torch.Tensor, num_per_batch: List[int], unmask_value: bool = False):
        batch_size = len(num_per_batch)
        max_context_len = max(num_per_batch)
        batched_features = features.new_zeros((max_context_len, batch_size, features.shape[-1]))
        attention_mask = (features==1).new_ones((max_context_len, batch_size))
        cum_sums = numpy.cumsum(num_per_batch)
        start_of_segments = numpy.append([0], cum_sums[:-1])
        slices = [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
        for i, (cur_slice, cur_size) in enumerate(zip(slices, num_per_batch)):
            batched_features[:cur_size, i] = features[cur_slice]
            attention_mask[:cur_size, i] = unmask_value

        return batched_features, attention_mask

    def _token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)

    def _unit_embedding(self, units: torch.Tensor) -> torch.Tensor:
        # [total units; unit_length; embedding size]
        encoded_tokens = self._token_embedding(units)
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(units == self.pad_id, dim=0)
            first_pad_pos[~is_contain_pad_id] = units.shape[0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        unit_embeddings = encoded_tokens[:,sort_indices]

        packed_path_units = nn.utils.rnn.pack_padded_sequence(unit_embeddings, sorted_path_lengths)

        # [num layers * num directions; total units; rnn size]
        _, (h_t, _) = self.unit_lstm(packed_path_units)
        # [total units; rnn size * num directions]
        encoded_units = h_t[-self.num_directions :].transpose(0, 1).reshape(h_t.shape[1], -1)
        encoded_units = self.dropout_rnn(encoded_units)

        encoded_units = encoded_units[reverse_sort_indices]
        return encoded_units
        
    def _unit_to_path(
        self,
        unit_features: torch.Tensor,
        units_per_data: torch.Tensor,
        paths: torch.Tensor, 
        path_per_data: List[int],
    ):
        is_contain_end_id, first_end_pos = torch.max(paths == -1, dim=0)
        first_end_pos[~is_contain_end_id] = paths.shape[0]
        batch_size = len(units_per_data)
        max_length = max(first_end_pos)
        zero_feature = unit_features.new_zeros([1,unit_features.shape[-1]])
        all_expaths = unit_features.new_zeros((max_length, paths.size(1)))
        expaths_mask = (paths==0).new_ones((max_length, paths.size(1)))
        # unit_path = paths.new_zeros()
        paths_begin = 0
        units_begin = 0
        
        for i in range(batch_size):
            paths_num = path_per_data[i]
            batch_paths = paths[:,paths_begin:paths_begin+paths_num]
            paths_begin += paths_num

            units_num = units_per_data[i]
            batch_units = unit_features[units_begin:units_begin+units_num]
            units_begin += units_num
            for j in range(paths_num): 
                backward_path = batch_paths[:,j] 
                path_context = [batch_units[unit] if (unit < units_num) else zero_feature[0]
                                     for unit in backward_path[:first_end_pos[j]]]
                path_context = torch.cat(path_context, dim=0)
                path_mask = (backward_path > -1)
                all_expaths[:path_context.size(0),paths_begin+j] = path_context
                expaths_mask[:path_mask.size(0),paths_begin+j] = path_mask
            paths_begin += paths_num
            units_begin += units_num

        return all_expaths, expaths_mask

    def forward(
        self, 
        units: torch.Tensor,
        unit_per_data: List[int], 
        paths: torch.Tensor, 
        path_per_data: List[int]
    ) -> torch.Tensor:

        # [unit length, total units; rnn size * num directions]
        encoded_units = self._unit_embedding(units)
        # [total units; embedding_size]
        encoded_units = self.concat_layer(encoded_units)

        # [max path length; batch size; embedding_size], [max path length; batch size]
        expaths_context, expaths_mask = self._unit_to_path(encoded_units, unit_per_data, paths, path_per_data)
        expaths_context = self.positionalencoding(expaths_context)
        # [max path length; totul paths, embedding_size]
        expaths = self.expath_encoder(src=expaths_context, src_key_padding_mask = expaths_mask.T)

        # 原始代码序列trans
        # [max_order_num*unit_length, total units; embeding size]
        batched_order, order_mask = self._cut_batch(encoded_units, unit_per_data)
        batched_order = self.positionalencoding(batched_order)
        # [max path length; totul paths, embedding_size]
        batched_order = self.expath_encoder(src=batched_order, src_key_padding_mask=order_mask.T)

        final_order, final_order_mask = self._splice_expaths(expaths, expaths_mask, batched_order, order_mask,
                                                                     paths, path_per_data, unit_per_data, 1)
        final_order_addpos = self.positionalencoding(final_order)
        final_output = self.final_encoder(src=final_order_addpos, src_key_padding_mask= final_order_mask.T)

        return final_output, final_order_mask
