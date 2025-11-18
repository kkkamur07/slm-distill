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
    eval_on_dev: bool = True,
) -> Dict[str, Any]:
    """
    Inputs:
      - model: already loaded classifier (e.g. XLMRobertaForSequenceClassification)
      - tokenizer: matching tokenizer
      - train_texts / train_labels: training data
      - dev_texts / dev_labels: dev data (can be None)
      - device: 'cuda' or 'cpu'

    Returns:
      - dict with:
          "model": the trained model,
          "history": list of {"epoch": int, "train_loss": float, "dev_metrics": dict | None}
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    n_train = len(train_texts)
    history: List[Dict[str, Any]] = []

    print(f"\nStarting SENTIMENT training on {n_train} examples...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"Epoch {epoch}/{num_epochs}",
        ):
            batch_indices = idxs[start : start + batch_size]
            if not batch_indices:
                continue

            batch_texts = [train_texts[i] for i in batch_indices]
            batch_y = [train_labels[i] for i in batch_indices]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor(batch_y, dtype=torch.long, device=device)

            outputs = model(**enc, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch}: train loss = {avg_loss:.4f}")

        dev_metrics = None
        if eval_on_dev and dev_texts is not None and dev_labels is not None:
            dev_metrics = compute_sentiment_accuracy(
                model,
                tokenizer,
                dev_texts,
                dev_labels,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            print(
                f"Dev accuracy: {dev_metrics['accuracy']:.4f} | "
                f"macro-F1: {dev_metrics['macro_f1']:.4f} | "
                f"micro-F1: {dev_metrics['micro_f1']:.4f} | "
                f"precision (macro): {dev_metrics['precision']:.4f} | "
                f"recall (macro): {dev_metrics['recall']:.4f}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

    return {"model": model, "history": history}