from typing import List, Optional, Dict, Any

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification

from src.evals.sentiment_eval import compute_sentiment_accuracy


def create_sentiment_classifier(
    base_model_name: str,
    num_labels: int,
    dropout: float,
    subfolder: str | None = None,
) -> nn.Module:
    """Create an XLM-RoBERTa classifier with custom dropout and label count."""
    cfg_kwargs: Dict[str, Any] = {}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(base_model_name, **cfg_kwargs)
    config.num_labels = num_labels
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout

    model = XLMRobertaForSequenceClassification.from_pretrained(
        base_model_name,
        config=config,
        **cfg_kwargs,
    )
    return model


def train_sentiment_model(
    model: nn.Module,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    dev_texts: Optional[List[str]],
    dev_labels: Optional[List[int]],
    device: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    weight_decay: float,
    early_stopping_patience: Optional[int],
    min_delta: float,
    eval_on_dev: bool = True,
) -> Dict[str, Any]:
    """Train a sentiment classifier with optional dev-based early stopping.

    Returns a dict with:
      - "model": the model with best weights (if dev used) or last epoch
      - "history": per-epoch loss and dev metrics
      - "batch_history": per-batch training loss (for later visualisation)
    """
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    n_train = len(train_texts)
    history: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    global_step = 0

    print(f"\n[Sentiment] Starting training on {n_train} examples...")

    best_dev_acc: Optional[float] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    has_dev = eval_on_dev and dev_texts is not None and dev_labels is not None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"[Sentiment] Epoch {epoch}/{num_epochs}",
        ):
            optimizer.zero_grad()

            batch_indices = idxs[start : start + batch_size]

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

            loss_val = float(loss.item())
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            batch_history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "batch_loss": loss_val,
                }
            )

        avg_loss = epoch_loss / max(num_batches, 1)
        dev_metrics: Optional[Dict[str, Any]] = None

        if has_dev:
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
            has_dev
            and early_stopping_patience is not None
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

    return {
        "model": model,
        "history": history,
        "batch_history": batch_history,
    }
