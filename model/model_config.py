from typing import Callable

from pydantic import BaseModel, ConfigDict

from torch.nn import functional as F


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    pass


class AttentionConfig(BaseConfig):
    num_heads: int = 6
    num_embeddings: int = 512
    block_size: int = 10
    dropout_rate: float = 0.1
    activation: Callable = F.gelu
    norm_first: bool = False
