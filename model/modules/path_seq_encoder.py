from model.modules.attention import LocalAttention
from typing import Dict, List

import torch
import numpy
from omegaconf import DictConfig
from torch import nn
from torch._C import ListType, device
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.sparse import Embedding

from utils.vocabulary import Vocabulary, SEP

from .positional_encoder import PositionalEncoding


class PathSeqEncoder(nn.Module):
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
        self._negative_value = -1e9

        self.pad_id = token_pad_id

        self.token_embedding = embedding

        self.path_layers = config.path_layers
        self.final_layers = config.final_layers
        self.attn_heads = config.attn_heads
        self.ffn_dim = config.ffn_dim
        self.dropout = config.dropout
        self.hidden = config.embedding_size
        self.share_weight = config.share_weight

        self.positionalencoding = PositionalEncoding(d_model=self.hidden)
        self.token_weight = LocalAttention(self.hidden)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads,
                                                                     dim_feedforward=self.ffn_dim, dropout=self.dropout)
        self.expath_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.path_layers)
        
        self.order_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.path_layers)
        #    self.order_encoder = self.expath_encoder
        self.atten_linear = nn.Linear(self.hidden, self.hidden)
        self.atten_norm = nn.LayerNorm(self.hidden)
        self.norm = nn.LayerNorm(self.hidden)

        if self.final_layers > 0:
            self.final_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.final_layers)

    def _cut_batch(self, features:torch.Tensor, num_per_batch: List[int], dim = 3, mask_value: float = -1e9):
        '''
        with torch.no_grad():
            max_length = 0
            length = []
            is_contain_pad_id, first_pad_pos = torch.max(features == self.pad_id, dim=0)
            first_pad_pos[~is_contain_pad_id] = features.shape[0]  # if no pad token use len+1 position
            cum_sums = numpy.cumsum(num_per_batch)
            start_of_segments = numpy.append([0], cum_sums[:-1])
            slices = [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
            for i, (start, end) in enumerate(zip(start_of_segments, cum_sums)):
                lengths = [first_pad_pos[i] for i in range(start, end)]
                length.append(sum(lengths))
                max_length = max(max_length,sum(lengths))
        '''
        with torch.no_grad():
            batch_size = len(num_per_batch)
            max_num = max(num_per_batch)
            max_length = max_num*features.size(0)
            cum_sums = numpy.cumsum(num_per_batch)
            start_of_segments = numpy.append([0], cum_sums[:-1])
        batched_features = features.new_zeros((max_length, batch_size))
        batched_mask = (features==1).new_ones((max_length, batch_size))
        for i, (start, end) in enumerate(zip(start_of_segments, cum_sums)):
            length = features.size(0)*num_per_batch[i]
            batched_features[:length, i] = torch.cat([features[:, j] for j in range(start, end)], dim = 0)
            batched_mask[:length, i] = torch.cat([features[:, j]==self.pad_id for j in range(start, end)], dim = 0)

        return batched_features, batched_mask

    def _unit_to_path(
        self,
        unit_features: torch.Tensor,
        units_per_data: List[int],
        paths: torch.Tensor, 
        path_per_data: List[int],
    ):
        with torch.no_grad():
            batch_size = len(units_per_data)
            # unit_path = paths.new_zeros()
            paths_begin = 0
            units_begin = 0
            is_contain_pad_id, first_pad_pos = torch.max(unit_features == self.pad_id, dim=0)
            first_pad_pos[~is_contain_pad_id] = unit_features.shape[0]
            is_contain_end_id, first_end_pos = torch.max(paths == -1, dim=0)
            first_end_pos[~is_contain_end_id] = paths.shape[0]
            max_length = max(first_end_pos)*unit_features.size(0)
            # 将路径中unit填充入路径
            paths_begin = 0
            units_begin = 0
        all_expaths = unit_features.new_zeros((max_length, paths.size(1)))
        expaths_mask = (paths==0).new_ones((max_length, paths.size(1)))
        zero_unit = unit_features.new_full((unit_features.size(0),1),self.pad_id)
        for i in range(batch_size):
            paths_num = path_per_data[i]
            batch_paths = paths[:,paths_begin:paths_begin+paths_num]
            units_num = units_per_data[i]
            batch_units = unit_features[:,units_begin:units_begin+units_num]
            for j in range(paths_num):  
                backward_path = batch_paths[:,j]
                path_context = [batch_units[:,unit] if (unit < units_num) else zero_unit[:,0]
                                     for unit in backward_path[:first_end_pos[paths_begin+j]]]
                path_context = torch.cat(path_context, dim=0)
                path_mask = (path_context == self.pad_id)
                all_expaths[:path_context.size(0),paths_begin+j] = path_context
                expaths_mask[:path_mask.size(0),paths_begin+j] = path_mask
            paths_begin += paths_num
            units_begin += units_num

        return all_expaths, expaths_mask

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
        unit_length: int,
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
            with torch.no_grad():
                for i in range(paths_per_label[batch]):
                    for j, unit_index in enumerate(batch_paths[:first_end_pos[path_begin+i],i]):
                        if (unit_index >= units_per_label[batch]) or batch_mask[j*unit_length,i]:
                            continue
                        num_orders[unit_index] += 1
            # 按顺序排列每个unit的多个特征表示
            order = expaths.new_zeros((unit_length, max(units_per_label), torch.max(num_orders).item(), expaths.size(-1)))
            add_order = expaths.new_zeros((unit_length, max(units_per_label), expaths.size(-1)))
            order_mask = mask.new_ones((unit_length, max(units_per_label), max(num_orders)))
            add_mask = mask.new_ones((unit_length, max(units_per_label)))
            with torch.no_grad():
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
                if order[:,i].size(1) > 2:
                    a = 1
                order_query = self.atten_linear(order[:,i])
                order_query = self.atten_norm(order_query)
                order_query = torch.tanh(order_query)
                order_weight = self.token_weight(order_query, order_mask[:,i])
                add_order[:,i] = torch.matmul(order_weight.permute(0,2,1), order[:,i]).squeeze()
                add_mask[:,i] = (torch.sum((order_mask[:,i]==False), dim=1) == 0)
            batched_order[:,batch] = torch.cat([add_order[:,i] for i in range(order.size(1))], dim=0)
            batched_mask[:,batch] = torch.cat([add_mask[:,i] for i in range(order_mask.size(1))], dim=0)
            path_begin += paths_per_label[batch]
            unit_begin += units_per_label[batch]
            torch.cuda.empty_cache()

        return batched_order, batched_mask

    def _token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)


    def forward(
        self, 
        units: torch.Tensor,
        unit_per_data: List[int], 
        paths: torch.Tensor, 
        path_per_data: List[int]
    ) -> torch.Tensor:
        sum_unit = sum(unit_per_data)
        if units.size(1) != sum(unit_per_data):
            raise ValueError(f"units_num wrong!")
        sum_path = sum(path_per_data)
        if paths.size(1) != sum(path_per_data):
            raise ValueError(f"paths_num wrong!")

        # 每个执行路径单独trans
        # [max_path_length*unit_length; totul_paths]
        expaths_context, expaths_mask = self._unit_to_path(units, unit_per_data, paths, path_per_data)
        # [max_path_length*unit_length, total units; embeding size]
        expaths_context_embeded = self.token_embedding(expaths_context)
        expaths_context_addpos = self.positionalencoding(expaths_context_embeded)
        # [max path length; totul paths, embedding_size]
        expaths = self.expath_encoder(src=expaths_context_addpos, src_key_padding_mask = expaths_mask.T)

        # 原始代码序列trans
        # [max_order_num*unit_length; batch_size]
        batched_order, order_mask = self._cut_batch(units, unit_per_data)
        # [max_order_num*unit_length, total units; embeding size]
        batched_order_embeded = self.token_embedding(batched_order)
        batched_order_addpos = self.positionalencoding(batched_order_embeded)
        # [max path length; totul paths, embedding_size]
        only_order = self.order_encoder(src=batched_order_addpos, src_key_padding_mask=order_mask.T)

        # 执行路径组织成原始代码序列顺序trans
        # [max order length; batch_size, embedding_size]
        final_order, final_order_mask = self._splice_expaths(expaths, expaths_mask, only_order, order_mask,
                                                                     paths, path_per_data, unit_per_data, units.size(0))
        # final_order = self.norm(final_order)
        
        if self.final_layers > 0:
            final_order_addpos = self.positionalencoding(final_order)
            final_order = self.final_encoder(src=final_order_addpos, src_key_padding_mask=final_order_mask.T)

        return final_order, final_order_mask
