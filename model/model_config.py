from typing import Callable

from pydantic import BaseModel, ConfigDict

from torch.nn import functional as F


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    num_layers: int = 3
    num_heads: int = 6
    num_embeddings: int = 512
    block_size: int = 128
    vocab_size: int = 2048
    dropout_rate: float = 0.1
    activation: Callable = F.gelu
    norm_first: bool = False
