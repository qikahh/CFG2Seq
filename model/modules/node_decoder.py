from typing import List, Tuple
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.sparse import Embedding

from utils.vocabulary import Vocabulary
from .positional_encoder import PositionalEncoding


class NodeDecoder(nn.Module):

    _negative_value = -1e9

    def __init__(
        self,
        config: DictConfig,
        embedding: Embedding,
        vocabulary: Vocabulary, 
        input_size: int,
        out_size: int,
    ):
        super().__init__()
        self._vocabulary = vocabulary
        self.token_embedding = embedding
        self.out_size = out_size
        self.teacher_forcing = config.teacher_forcing

        self.positionalencoding = PositionalEncoding(d_model=input_size)

        self.transformer_decoder_layer = TransformerDecoderLayer(d_model=input_size, nhead=config.attn_heads,
                                                                    dim_feedforward=config.ffn_dim, dropout=config.dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer=self.transformer_decoder_layer,
                                                          num_layers=config.n_layers)

        self.norm = nn.LayerNorm(input_size)
        self.projection_layer = nn.Linear(input_size, self.out_size, bias=False)
        self.projection_softmax = nn.Softmax(dim = 1)

    def forward(
        self,
        batched_features: torch.Tensor,
        batched_mask: torch.Tensor,
        output_length: int,
        target_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode given paths into sequence

        :param encoded_paths: [total num; max path legth; encoder size]
        :param contexts_per_label: [n1, n2, ..., nk] sum = total num
        :param output_length: length of output sequence
        :param target_sequence: [sequence length+1; batch size]
        :return:
        """

        batch_size = batched_features.shape[1]
        # [target len; batch size; vocab size]  
        output = batched_features.new_zeros((output_length, batch_size, self.out_size))
        # [batch size]
        init_input = target_embedding[0].unsqueeze(0)
        current_input = init_input
        use_teacher_forcing = torch.rand(1)
        for step in range(output_length):
            current_output = self.decoder_step(
                current_input, batched_features, batched_mask
            )
            output[step] = current_output
            if target_embedding is not None and use_teacher_forcing <= self.teacher_forcing:
                current_input = target_embedding[:step+2]
            else:
                current_seq = output[:step+1].argmax(dim=-1)
                current_input = self.token_embedding(current_seq)
                current_input = torch.cat([init_input,current_input], 0)

        return output

    def decoder_step(
        self,
        input_tokens: torch.Tensor,  # [step; batch size; decoder size]
        batched_expath: torch.Tensor,  # [context size; batch size; decoder size]
        expath_mask: torch.Tensor,  # [context size; batch size]
    ) -> torch.Tensor:

        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        input_tokens = self.positionalencoding(input_tokens)
        trans_output = self.transformer_decoder(tgt=input_tokens, memory=batched_expath, memory_key_padding_mask=expath_mask.T)
        next_output = trans_output[-1].squeeze()

        # [batch size; vocab size]
        output = self.projection_layer(next_output)

        return output
