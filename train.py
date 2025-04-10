import os
from time import time

from tqdm import tqdm

import tiktoken

from pydantic import BaseModel, ConfigDict

import torch
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader

from dataset.translation_dataset import TranslationDataset
from model.model_config import ModelConfig
from model.tiny_transformer import TinyTransformer


class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: TinyTransformer
    optimizer: Optimizer
    train_loader: DataLoader
    test_loader: DataLoader
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    batch_size = 8
    epochs = 10
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.max_token_value + 1
    print("Tokenizer loaded. Max token vaule", vocab_size)
    model_config = ModelConfig(vocab_size=vocab_size)

    print("Loading model with configuration:")
    print(model_config)

    model: TinyTransformer = TinyTransformer(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    train_filepath = "data/train.jsonl"
    test_filepath = "data/test.jsonl"

    print(f"Loading datasets from {train_filepath} and {test_filepath}")

    train_dataset, test_dataset = (
        TranslationDataset(
            train_filepath, max_length=model_config.block_size, tokenizer=tokenizer
        ),
        TranslationDataset(
            test_filepath, max_length=model_config.block_size, tokenizer=tokenizer
        ),
    )
    collate_fn = train_dataset.collate_fn
    train_loader, test_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        ),
    )

    epoch = 0
    for epoch in range(epochs):
        model.train()
        start_time = time()
        train_loss, val_loss = 0.0, 0.0
        for batch in tqdm(train_loader, desc="Training Batches"):
            optimizer.zero_grad()
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            output, loss = model(src, tgt)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation Batches"):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)

                output, loss = model(src, tgt)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        end_time = time()
        elapsed_time = end_time - start_time
        print(
            f"Epoch {epoch + 1}/{epochs} completed in {elapsed_time:.2f} seconds. Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}"
        )
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    # Save the model
    checkpoint_file = f"checkpoints/tiny_transformer_{epoch}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
        },
        checkpoint_file,
    )
    print("Model saved as tiny_transformer.pth")


if __name__ == "__main__":
    main()
