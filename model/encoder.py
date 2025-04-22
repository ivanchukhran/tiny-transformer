import torch
from torch import nn

from .attention import CasualSelfAttention
from .feed_forward import FeedForward
from .model_config import ModelConfig


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(EncoderLayer, self).__init__()
        self.self_attn = CasualSelfAttention(config)
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x
