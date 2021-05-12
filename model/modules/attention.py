import torch
import torch.nn.functional as F
from torch import nn


class LuongAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, units, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
        """Calculate attention weights

        :param hidden: [batch size; embedding size]
        :param encoder_outputs: [batch size; seq len; embedding size]
        :param mask: [batch size; seq len]
        :return: [batch size; seq len]
        """
        mask = mask.T
        batch_size, seq_len = mask.shape
        # [batch size; units]
        attended_hidden = self.attn(hidden)
        # [batch size; seq len]
        score = torch.bmm(encoder_outputs, attended_hidden.view(batch_size, -1, 1)).squeeze(-1)
        mask_atten = score.new_ones(score.size())*mask_value
        score = torch.where((mask==False), score, mask_atten)

        # [batch size; seq len]
        weights = F.softmax(score, dim=1)
        return weights


class LocalAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
        """Calculate attention weights

        :param encoder_outputs: [seq len; batch size; units]
        :param mask: [seq len; batch size]
        :return: [seq len; batch size; 1]
        """
        # [seq len; batch size; 1]
        attended_encoder_outputs = self.attn(encoder_outputs)
        mask_atten = attended_encoder_outputs.new_ones(attended_encoder_outputs.size())*mask_value
        attended = torch.where((mask.unsqueeze(2)==False), attended_encoder_outputs, mask_atten)

        # [seq len; batch size; 1]
        weights = torch.softmax(attended, dim=1)

        return weights
