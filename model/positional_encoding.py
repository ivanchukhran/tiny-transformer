import torch
from torch import nn

from .model_config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout_rate)
        block_size = config.block_size
        n_embd = config.n_embd
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        # self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
