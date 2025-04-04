import torch
from torch import nn

from model.decoder import DecoderLayer

from .model_config import ModelConfig
from .encoder import EncoderLayer


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TinyTransformer, self).__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.encoder_embedding = nn.Embedding(config.num_embeddings, config.num_embeddings)
        self.decoder_embedding = nn.Embedding(config.num_embeddings, config.num_embeddings)
        self.fc_out = nn.Linear(config.num_embeddings, config.num_embeddings)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        encoder_output = self.encoder_embedding(x)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        if targets is None:
            return encoder_output

        decoder_output = self.decoder_embedding(targets)
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        output = self.fc_out(decoder_output)
        return output
