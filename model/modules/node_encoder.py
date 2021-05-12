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

class NodeEncoder(nn.Module):
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

        self.n_layers = config.n_layers
        self.attn_heads = config.attn_heads
        self.ffn_dim = config.ffn_dim
        self.dropout = config.dropout
        self.hidden = config.embedding_size

        self.positionalencoding = PositionalEncoding(d_model=self.hidden)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.hidden, nhead=self.attn_heads,
                                                                     dim_feedforward=self.ffn_dim, dropout=self.dropout)
        self.node_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                          num_layers=self.n_layers)

        
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
        batched_features, attention_mask = self._cut_batch(encoded_units, unit_per_data)
        batched_features = self.positionalencoding(batched_features)
        # [max path length; batch_size, embedding_size]
        batched_features = self.node_encoder(src=batched_features, src_key_padding_mask = attention_mask.T)
        return batched_features, attention_mask
