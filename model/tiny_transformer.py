import math
from typing import Callable

import tiktoken
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import nn


class ModelConfig(BaseModel):
    r"""
    Configuration for the Tiny Transformer model.

    Attributes:
        n_layers (int) = 3: Number of layers in the model.
        n_heads (int) = 8: Number of attention heads.
        embed_dim (int) = 512: Embedding dimension.
        block_size (int) = 256: Maximum sequence length.
        vocab_size (int) = 2048: Vocabulary size.
        dropout (float) = 0.1: Dropout rate.
        activation (Callable[[torch.Tensor], torch.Tensor]) = F.gelu: Activation function.
        norm_first (bool) = False: Whether to apply normalization before or after attention.
    """

    model_config = ConfigDict(extra="allow")
    n_layers: int = 3
    n_heads: int = 8
    embed_dim: int = 512
    block_size: int = 256
    vocab_size: int = 50257
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu
    norm_first: bool = False


class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(config.block_size, config.embed_dim)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.embed_dim, 2).float()
            * (-math.log(10000.0) / config.embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]  # type: ignore


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        is_causal: bool = False,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.n_heads
        self.head_dims = self.embed_dim // self.num_heads
        assert self.head_dims * self.num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.is_causal = is_causal
        if self.is_causal:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        self.dropout = nn.Dropout(config.dropout)

        # Separate projections for q, k, v
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        # attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # If key/value not provided, use x for self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        B, T, C = query.shape
        _, S, _ = key.shape  # S is source sequence length

        q = (
            self.q_proj(query)
            .view(B, T, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(B, S, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(B, S, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dims)

        if self.is_causal:
            attn = attn.masked_fill(self.mask[:, :T, :T] == 0, float("-inf"))  # pyright: ignore

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = self.out_proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.linear2 = nn.Linear(4 * config.embed_dim, config.embed_dim)
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(config)  #
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.feedforward = FeedForward(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm_first = config.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x += self.dropout1(self.self_attn(self.norm1(x)))
            x += self.feedforward(self.norm2(x))
        else:
            x += self.norm1(x + self.self_attn(x))
            x += self.norm2(x + self.feedforward(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.norm_first = config.norm_first
        self.self_attn = MultiheadAttention(config, is_causal=True)
        self.multihead_attn = MultiheadAttention(config)
        self.feedforward = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.norm3 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        r"""Pass the inputs trough the decoder layer.

        Args:
            x (torch.Tensor): The target sequence.
            context (torch.Tensor): The context sequence from the last encoder layer.

        Returns:
            torch.Tensor: The output sequence.

        """
        if self.norm_first:
            x += self.self_attn(self.norm1(x))
            x += self.multihead_attn(self.norm2(x), key=context, value=context)
            x += self.feedforward(self.norm3(x))
        else:
            x = self.norm1(x + self.self_attn(x))
            x = self.norm2(x + self.multihead_attn(x, key=context, value=context))
            x = self.norm3(x + self.feedforward(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        for layer in self.layers:
            x = layer(x, context)
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super(TinyTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print(x.shape, target.shape)
        print("embedding dim:", self.token_embedding.embedding_dim)
        src_emb = self.token_embedding(x.to(torch.long)) * math.sqrt(
            self.token_embedding.embedding_dim
        )
        src_emb = self.positional_encoding(src_emb)
        context = self.encoder(src_emb)

        target_emb = self.token_embedding(target) * math.sqrt(
            self.token_embedding.embedding_dim
        )
        target_emb = self.positional_encoding(target_emb)
        output = self.decoder(target_emb, context)
        logits = self.output_layer(output)
        return logits


def main():
    torch.manual_seed(42)

    max_length = 10
    embed_dim = 512
    n_heads = 8

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    config = ModelConfig(
        vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads, n_layers=1
    )

    # Create model
    pos_encoding = PositionalEncoding(config)
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
    my_encoder_layer = TransformerEncoderLayer(config)
    my_decoder_layer = TransformerDecoderLayer(config)

    # Create float tensors for PyTorch's built-in transformer layers
    src_float = torch.rand((32, config.block_size, embed_dim))
    tgt_float = torch.rand((32, config.block_size, embed_dim))

    # Test PyTorch's layers
    out = encoder_layer(src_float)
    decoder_out = decoder_layer(tgt_float, out)

    # Create integer token tensors for our implementation
    src_tokens = torch.randint(
        0, vocab_size, (32, config.block_size)
    )  # [batch, seq_len]
    tgt_tokens = torch.randint(
        0, vocab_size, (32, config.block_size)
    )  # [batch, seq_len]

    # Test our implementation
    my_out = my_encoder_layer(src_float)  # Still uses float input
    my_decoder_out = my_decoder_layer(tgt_float, my_out)  # Still uses float input

    # Shapes should match for the float tensor versions
    assert out.shape == my_out.shape, "Shapes should be equal"
    assert decoder_out.shape == my_decoder_out.shape, "Shapes should be equal"

    # Test the full transformer with token IDs
    transformer = TinyTransformer(config)
    transformer_output = transformer(src_tokens, tgt_tokens)
    print("transformer output:", transformer_output.shape)


if __name__ == "__main__":
    main()
