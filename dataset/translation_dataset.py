import json

import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, path: str, tokenizer: Encoding, max_length=128):
        assert path.endswith(".jsonl"), "Path must end with .jsonl"
        self.data: list[dict] = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.eot_token = "<|endoftext|>"
        self.eot_token_id = self.tokenizer.encode(
            self.eot_token, allowed_special={self.eot_token}
        )[0]  #

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        example = self.data[index]
        en = example["en"]
        uk = example["uk"]

        src_tokens = self.tokenizer.encode(en)
        tgt_tokens = self.tokenizer.encode(uk)

        src_tokens = src_tokens[: self.max_length - 1]
        tgt_tokens = tgt_tokens[: self.max_length - 1]

        src_tokens.append(self.eot_token_id)
        tgt_tokens.append(self.eot_token_id)

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
        }

    def collate_fn(self, batch):
        batch_size = len(batch)
        pad_token_id = 0

        src_padded = torch.full(
            (batch_size, self.max_length), pad_token_id, dtype=torch.long
        )
        tgt_padded = torch.full(
            (batch_size, self.max_length), pad_token_id, dtype=torch.long
        )

        for i, item in enumerate(batch):
            src = item["src"]
            tgt = item["tgt"]
            src_padded[i, : src.size(0)] = src
            tgt_padded[i, : tgt.size(0)] = tgt

        return {
            "src": src_padded.long(),
            "tgt": tgt_padded.long(),
        }


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TranslationDataset("data/train.jsonl", tokenizer)
    print(len(dataset))
    print(dataset[0])
    res = dataset.collate_fn([dataset[0], dataset[1]])
    print(res)
