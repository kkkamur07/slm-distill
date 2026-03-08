import copy
from typing import List, Optional, Dict, Any

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification, PreTrainedModel

from src.evals.sentiment_eval import compute_sentiment_accuracy


def _load_base_weights(model: nn.Module, base_model: PreTrainedModel) -> None:
    """Load compatible weights from a base model, ignoring mismatched heads."""
    target_state = model.state_dict()
    base_state = base_model.state_dict()
    prefix = getattr(model, "base_model_prefix", "")

    adapted_state = {}
    for key, tensor in base_state.items():
        candidates = [key]
        if prefix and not key.startswith(f"{prefix}."):
            candidates.append(f"{prefix}.{key}")

        for cand in candidates:
            if cand in target_state and target_state[cand].shape == tensor.shape:
                adapted_state[cand] = tensor
                break

    model.load_state_dict(adapted_state, strict=False)


def create_sentiment_classifier(
    base_model_name: str | None,
    num_labels: int,
    dropout: float,
    subfolder: str | None = None,
    base_model: PreTrainedModel | None = None,
) -> nn.Module:
    """Create an XLM-RoBERTa classifier with custom dropout and label count."""
    cfg_kwargs: Dict[str, Any] = {}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder

    if base_model is not None:
        config = copy.deepcopy(base_model.config)
    elif base_model_name is not None:
        config = AutoConfig.from_pretrained(base_model_name, **cfg_kwargs)
    else:
        raise ValueError("Provide either base_model or base_model_name.")

    config.num_labels = num_labels
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout

    if base_model is not None:
        model = XLMRobertaForSequenceClassification(config)
        _load_base_weights(model, base_model)
    else:
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
    best_dev_metrics: Optional[Dict[str, Any]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
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

        print(f"[Sentiment] Epoch {epoch}: train loss = {avg_loss:.4f}")

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

            curr_acc = dev_metrics["accuracy"]
            if best_dev_acc is None or curr_acc > best_dev_acc + min_delta:
                best_dev_acc = curr_acc
                best_dev_metrics = dict(dev_metrics)
                best_state_dict = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                no_improve = 0
            elif early_stopping_patience is not None:
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
        if best_epoch is not None:
            print(
                f"[Sentiment] Restored best dev checkpoint from epoch {best_epoch} "
                f"(accuracy={best_dev_acc:.4f})"
            )

    return {
        "model": model,
        "history": history,
        "batch_history": batch_history,
        "best_dev_metrics": best_dev_metrics,
        "best_epoch": best_epoch,
    }


def train_sentiment_model_score(
    model: nn.Module,
    tokenizer,
    train_texts: List[str],
    train_labels: List[int],
    device: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    weight_decay: float,
    baseline: bool = True,
    lambda_ce: float = 0.9,
) -> Dict[str, Any]:
    """Hybrid CE + policy-gradient training with reward = correctness (0/1).

    Loss: lambda_ce * CE + (1 - lambda_ce) * REINFORCE(policy loss).
    The RL part uses a self-critical baseline (greedy reward) if enabled.
    """
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    n_train = len(train_texts)
    history: List[Dict[str, Any]] = []
    print(f"\n[Sentiment][Score] Starting score-based training on {n_train} examples...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"[Sentiment][Score] Epoch {epoch}/{num_epochs}",
        ):
            batch_indices = idxs[start : start + batch_size]
            if not batch_indices:
                continue

            batch_texts = [train_texts[i] for i in batch_indices]
            batch_y = torch.tensor(
                [train_labels[i] for i in batch_indices], dtype=torch.long, device=device
            )

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            optimizer.zero_grad()
            logits = model(**enc).logits  # (bs, num_labels)
            log_probs = torch.log_softmax(logits, dim=-1)

            # sample labels
            samples = torch.distributions.Categorical(logits=logits).sample()
            sampled_logp = log_probs.gather(1, samples.view(-1, 1)).squeeze(1)

            reward = (samples == batch_y).float()
            if baseline:
                with torch.no_grad():
                    greedy = logits.argmax(dim=-1)
                    base = (greedy == batch_y).float()
            else:
                base = torch.zeros_like(reward)

            advantage = reward - base
            rl_loss = -(advantage * sampled_logp).mean()

            ce_loss = torch.nn.functional.cross_entropy(
                logits,
                batch_y,
            )

            loss = lambda_ce * ce_loss + (1.0 - lambda_ce) * rl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
            }
        )
        print(f"[Sentiment][Score] Epoch {epoch}: hybrid loss = {avg_loss:.4f}")

    return {"model": model, "history": history}
