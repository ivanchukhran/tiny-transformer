import torch
from torch import nn

from .model_config import AttentionConfig

class MLP(nn.Module):
    def __init__(self, config: AttentionConfig):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.num_embeddings, 4 * config.num_embeddings)
        self.fc2 = nn.Linear(4 * config.num_embeddings, config.num_embeddings)
        self.activation = config.activation

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
