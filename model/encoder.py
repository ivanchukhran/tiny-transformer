from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from pydantic import BaseModel, ConfigDict

from .mlp import MLP
from .model_config import ModelConfig
from .casual_self_attention import CasualSelfAttention


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(EncoderLayer, self).__init__()
        self.self_attn = CasualSelfAttention(config)
        self.norm_1 = nn.LayerNorm(config.num_embeddings)
        self.norm_2 = nn.LayerNorm(config.num_embeddings)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.self_attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x
