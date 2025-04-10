from typing import Callable

from pydantic import BaseModel, ConfigDict

from torch.nn import functional as F


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    n_layers: int = 3
    n_heads: int = 8
    n_embd: int = 512
    block_size: int = 256
    vocab_size: int = 2048
    dropout_rate: float = 0.1
    activation: Callable = F.gelu
    norm_first: bool = False
