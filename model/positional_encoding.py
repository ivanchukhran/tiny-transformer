import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dims, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pe = torch.zeros(max_len, dims)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dims, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
