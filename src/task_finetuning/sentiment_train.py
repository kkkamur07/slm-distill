from typing import List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from src.evals.sentiment_eval import compute_sentiment_accuracy


def train_sentiment_model(
    model: nn.Module,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    dev_texts: Optional[List[str]],
    dev_labels: Optional[List[int]],
    device: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
) -> Dict[str, Any]:
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for i in tqdm(range(0, len(train_texts), batch_size), desc=f"Epoch {epoch}"):
            batch_texts = train_texts[i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(**enc, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_texts) // batch_size)
        dev_metrics = None

        if dev_texts is not None and dev_labels is not None and len(dev_texts) > 0:
            dev_metrics = compute_sentiment_accuracy(
                model,
                tokenizer,
                dev_texts,
                dev_labels,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

    return {"model": model, "history": history}
