import math

import torch
from torch import nn
from torch.nn import functional as F

from .model_config import ModelConfig


class CasualSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CasualSelfAttention, self).__init__()
        assert config.n_embd % config.n_heads == 0, (
            "Attention dimensions must be divisible by the number of heads."
        )
        self.num_heads = config.n_heads
        self.num_embeddings = config.n_embd
        self.attention = nn.Linear(self.num_embeddings, 3 * self.num_embeddings)
        self.projection = nn.Linear(self.num_embeddings, self.num_embeddings)
        block_size = config.block_size
        self.bias = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, target sequence length, number of embeddings
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



class CasualCrossAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CasualCrossAttention, self).__init__()
        assert config.n_embd % config.n_heads == 0, (
            "Attention dimensions must be divisible by the number of heads."
        )
        self.num_heads = config.n_heads
        self.num_embeddings = config.n_embd
        self.q_proj = nn.Linear(self.num_embeddings, self.num_embeddings)
        self.k_proj = nn.Linear(self.num_embeddings, self.num_embeddings)
        self.v_proj = nn.Linear(self.num_embeddings, self.num_embeddings)
        self.projection = nn.Linear(self.num_embeddings, self.num_embeddings)

        block_size = config.block_size
        self.bias = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch size, target sequence length, number of embeddings
        B, S, C = context.size() # batch size, source sequence length, number of embeddings
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, S, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, S, self.num_heads, C // self.num_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (1 * math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :S] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )
        y = self.projection(y)
        return y

if __name__ == "__main__":
    x = torch.randn(1, 10, 512)
    config = ModelConfig(n_heads=8, n_embd=512, block_size=10)
    model = CasualSelfAttention(config)
    y = model(x)
    print(y.size())
