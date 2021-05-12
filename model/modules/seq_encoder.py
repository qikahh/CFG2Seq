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


class SeqEncoder(nn.Module):
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

        self.positionalencoding = PositionalEncoding(d_model=self.hidden)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads,
                                                                     dim_feedforward=self.ffn_dim, dropout=self.dropout)

        self.order_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.path_layers)

        if self.final_layers > 0:
            self.final_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.final_layers)

    def _cut_batch(self, features:torch.Tensor, num_per_batch: List[int], dim = 3, mask_value: float = -1e9):
        batch_size = len(num_per_batch)
        max_num = max(num_per_batch)
        max_length = max_num*features.size(0)
        
        batched_features = features.new_zeros((max_length, batch_size))
        batched_mask = (features==1).new_ones((max_length, batch_size))
        cum_sums = numpy.cumsum(num_per_batch)
        start_of_segments = numpy.append([0], cum_sums[:-1])
        slices = [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]
        for i, (start, end) in enumerate(zip(start_of_segments, cum_sums)):
            length = features.size(0)*num_per_batch[i]
            batched_features[:length, i] = torch.cat([features[:, j] for j in range(start, end)], dim = 0)
            batched_mask[:length, i] = torch.cat([features[:, j]==self.pad_id for j in range(start, end)], dim = 0)

        return batched_features, batched_mask


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

        # 原始代码序列trans
        # [max_order_num*unit_length; batch_size]
        batched_order, order_mask = self._cut_batch(units, unit_per_data)
        # [max_order_num*unit_length, total units; embeding size]
        batched_order = self._token_embedding(batched_order)
        batched_order = self.positionalencoding(batched_order)
        # [max path length; totul paths, embedding_size]
        batched_order = self.order_encoder(src=batched_order, src_key_padding_mask=order_mask.T)

        if self.final_layers > 0:
            final_order_addpos = self.positionalencoding(batched_order)
            batched_order = self.final_encoder(src=final_order_addpos, src_key_padding_mask=order_mask.T)

        return batched_order, order_mask
