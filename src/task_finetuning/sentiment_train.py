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
    weight_decay: float = 0.01,
    early_stopping_patience: Optional[int] = None,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    """
    Fine-tune a sentiment classifier with optional weight decay + early stopping.

    Returns:
      {
        "model": trained model (restored to best dev checkpoint if early stopping),
        "history": [
          {"epoch": int, "train_loss": float, "dev_metrics": dict | None},
          ...
        ],
      }
    """
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    n_train = len(train_texts)
    history: List[Dict[str, Any]] = []

    print(f"\n[Sentiment] Starting training on {n_train} examples...")

    best_dev_acc: Optional[float] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"[Sentiment] Epoch {epoch}/{num_epochs}",
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
        print(f"\n[Sentiment] Epoch {epoch}: train loss = {avg_loss:.4f}")

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
                f"[Sentiment] Dev accuracy: {dev_metrics['accuracy']:.4f} | "
                f"macro-F1: {dev_metrics['macro_f1']:.4f} | "
                f"micro-F1: {dev_metrics['micro_f1']:.4f} | "
                f"precision (macro): {dev_metrics['precision']:.4f} | "
                f"recall (macro): {dev_metrics['recall']:.4f}"
            )

            if early_stopping_patience is not None:
                curr_acc = dev_metrics["accuracy"]
                if best_dev_acc is None or curr_acc > best_dev_acc + min_delta:
                    best_dev_acc = curr_acc
                    best_state_dict = {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

        if (
            early_stopping_patience is not None
            and best_dev_acc is not None
            and no_improve >= early_stopping_patience
        ):
            print(
                f"[Sentiment] Early stopping after epoch {epoch} "
                f"(no dev improvement for {early_stopping_patience} epochs)."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    return {"model": model, "history": history}

