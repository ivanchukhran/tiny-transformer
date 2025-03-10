import math

import torch
from torch import nn
from torch.nn import functional as F

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    pass


class AttentionConfig(BaseConfig):
    num_heads: int
    num_embeddings: int
    block_size: int


class CasualSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super(CasualSelfAttention, self).__init__()
        assert config.num_embeddings % config.num_heads == 0, (
            "Attention dimensions must be divisible by the number of heads."
        )
        self.num_heads = config.num_heads
        self.num_embeddings = config.num_embeddings
        self.attention = nn.Linear(self.num_embeddings, 3 * self.num_embeddings)
        self.projection = nn.Linear(self.num_embeddings, self.num_embeddings)
        block_size = config.block_size
        self.bias = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch size, sequence length, number of embeddings
        qkv = self.attention(x)
        q, k, v = qkv.split(self.num_embeddings, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (1 * math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # reassemble the heads to the original shape
        y = self.projection(y)
        return y


if __name__ == "__main__":
    x = torch.randn(1, 10, 512)
    config = AttentionConfig(num_heads=8, num_embeddings=512, block_size=10)
    model = CasualSelfAttention(config)
    y = model(x)
    print(y.size())
