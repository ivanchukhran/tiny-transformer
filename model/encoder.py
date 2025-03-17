import torch
from torch import nn

from .model_config import AttentionConfig
from .casual_self_attention import CasualSelfAttention
from .mlp import MLP


class EncoderLayer(nn.Module):
    def __init__(self, config: AttentionConfig):
        super(EncoderLayer, self).__init__()
        self.self_attn = CasualSelfAttention(config)
        self.norm_1 = nn.LayerNorm(config.num_embeddings)
        self.norm_2 = nn.LayerNorm(config.num_embeddings)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x
