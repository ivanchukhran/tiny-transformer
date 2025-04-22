import torch
import torch.nn.functional as F
from torch import nn

from .decoder import DecoderLayer
from .encoder import EncoderLayer
from .model_config import ModelConfig


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TinyTransformer, self).__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Embedding(config.block_size, config.n_embd)

        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape

        pos_embd = self.positional_encoding(torch.arange(T, device=x.device, dtype=torch.long))
        tok_embd = self.token_embedding(x)
        x = tok_embd + pos_embd

        encoder_output = x
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        loss = None

        if targets is None:
            return encoder_output, loss

        decoder_output = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        logits = self.fc_out(decoder_output)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_length: int, temperature: float = 0.1, top_k: int | None = None):
        for _ in range(max_length):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # print("logits", logits)
            logits = logits[:, -1, :] / temperature
            # print("scaled logits", logits)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
