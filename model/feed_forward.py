import torch
from torch import nn

from .model_config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.activation = config.activation

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
