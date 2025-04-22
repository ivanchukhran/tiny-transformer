import torch
from torch import nn

from .attention import CasualCrossAttention, CasualSelfAttention
from .feed_forward import FeedForward
from .model_config import ModelConfig


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(DecoderLayer, self).__init__()
        self.self_attention = CasualSelfAttention(config)
        self.cross_attenintion = CasualCrossAttention(config)
        self.mlp = FeedForward(config)
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.norm_3 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        x = self.norm_1(x)
        x = x + self.dropout(self.self_attention(x))

        x = self.norm_2(x)
        x = x + self.dropout(self.cross_attenintion(x, encoder_output))

        x = self.norm_3(x)
        x = x + self.dropout(self.mlp(x))

        return x
