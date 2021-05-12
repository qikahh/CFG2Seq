from model.modules.attention import LocalAttention, LuongAttention
from typing import List, Tuple
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.sparse import Embedding

from utils.vocabulary import Vocabulary
from .positional_encoder import PositionalEncoding


class RNNDecoder(nn.Module):

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

        self.rnn_num_layers = config.rnn_num_layers
        self.rnn_size = input_size

        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=config.rnn_num_layers,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )
        self.atten_weight = LuongAttention(input_size)

        self.concat_layer = nn.Linear(input_size * 2, input_size, bias=False)
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
        init_input = target_embedding[0]
        current_input = init_input
        use_teacher_forcing = torch.rand(1)
        initial_state = (
            torch.cat([ctx_batch.mean(0).unsqueeze(0) for ctx_batch in batched_features.permute(1,0,2)])
            .unsqueeze(0)
            .repeat(self.rnn_num_layers, 1, 1)
        )
        h_prev, c_prev = initial_state, initial_state
        for step in range(output_length):
            current_output, (h_prev, c_prev) = self.decoder_step(
                current_input, batched_features, batched_mask, h_prev, c_prev
            )
            output[step] = current_output
            if target_embedding is not None and use_teacher_forcing <= self.teacher_forcing:
                current_input = target_embedding[step+1]
            else:
                current_input = output[step].argmax(dim=-1)
                current_input = self.token_embedding(current_input)

        return output

    def decoder_step(
        self,
        input_tokens: torch.Tensor,  # [batch size; decoder size]
        batched_feature: torch.Tensor,  # [path_length; batch size; decoder size]
        feature_mask: torch.Tensor,  # [context size; batch size]
        h_prev: torch.Tensor,  # [n layers; batch size; decoder size]
        c_prev: torch.Tensor,  # [n layers; batch size; decoder size]
    ) -> torch.Tensor:
        batched_feature = batched_feature.permute(1,0,2)
        input_tokens = input_tokens.unsqueeze(0)
        # hidden -- [n layers; batch size; decoder size]
        rnn_output, (h_prev, c_prev) = self.LSTM(input_tokens, (h_prev, c_prev))
        next_atten_weight = self.atten_weight(h_prev[-1],batched_feature,feature_mask).unsqueeze(1)
        next_atten = torch.matmul(next_atten_weight, batched_feature).squeeze()
        next_output = torch.cat([rnn_output.squeeze(), next_atten], dim=1)
        # [batch size; vocab size]
        concat = self.concat_layer(next_output)
        concat = self.norm(concat)
        concat = torch.tanh(concat)
        output = self.projection_layer(concat)

        return output, (h_prev, c_prev)
