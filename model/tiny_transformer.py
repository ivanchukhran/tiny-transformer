import torch
from torch import nn
import torch.nn.functional as F


from .decoder import DecoderLayer

from .model_config import ModelConfig
from .encoder import EncoderLayer


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TinyTransformer, self).__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Embedding(config.block_size, config.n_embd)
        # self.positional_encoding = PositionalEncoding(config)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )

        # self.encoder_embedding = nn.Embedding(config.n_embd, config.n_embd)
        # self.decoder_embedding = nn.Embedding(config.n_embd, config.n_embd)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape

        # print("x.shape", x.shape, "target shape", targets.shape)

        pos_embd = self.positional_encoding(
            torch.arange(T, device=x.device, dtype=torch.long)
        )
        tok_embd = self.token_embedding(x)
        x = tok_embd + pos_embd

        # encoder_output = self.encoder_embedding(x.long())
        encoder_output = x
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        loss = None

        if targets is None:
            return encoder_output, loss

        # decoder_output = self.positional_encoding(targets)
        # decoder_output = self.decoder_embedding(decoder_output)
        decoder_output = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        logits = self.fc_out(decoder_output)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
