import copy
from typing import List, Dict, Any, Optional
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import (
    AutoConfig,
    XLMRobertaForSequenceClassification,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from src.evals.nli_eval import compute_nli_accuracy


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


def create_nli_classifier(
    base_model_name: str | None,
    num_labels: int,
    dropout: float | None,
    subfolder: str | None = None,
    base_model: PreTrainedModel | None = None,
    **model_kwargs,
):
    """
    Args:
        base_model_name: HF model id or local path.
        num_labels: number of NLI labels.
        dropout: optional dropout rate for the classification head & encoder.
        subfolder: if weights/config live in a HF subfolder (e.g. "model").
        **model_kwargs: forwarded to .from_pretrained (e.g. token=...).

    Returns:
        XLMRobertaForSequenceClassification instance.
    """
    # Only pass subfolder if it's not None
    config_kwargs = dict(model_kwargs)
    if subfolder is not None:
        config_kwargs["subfolder"] = subfolder

    if base_model is not None:
        config = copy.deepcopy(base_model.config)
    elif base_model_name is not None:
        config = AutoConfig.from_pretrained(
            base_model_name,
            **config_kwargs,
        )
    else:
        raise ValueError("Provide either base_model or base_model_name.")

    config.num_labels = num_labels

    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout

    model_kwargs2 = dict(model_kwargs)
    if subfolder is not None:
        model_kwargs2["subfolder"] = subfolder

    if base_model is not None:
        model = XLMRobertaForSequenceClassification(config)
        _load_base_weights(model, base_model)
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs2,
        )

    return model


def train_nli_model(
    model: nn.Module,
    tokenizer,
    train_premises: List[str],
    train_hypotheses: List[str],
    train_labels: List[int],
    dev_premises: Optional[List[str]],
    dev_hypotheses: Optional[List[str]],
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
    warmup_steps: int = 0,
) -> Dict[str, Any]:
    """
    Inputs:
      - model: already loaded, e.g. create_nli_classifier(...)
      - tokenizer: matching tokenizer (already loaded)
      - *_premises / *_hypotheses / *_labels: lists of strings/ints
      - device: "cuda" or "cpu"

    Returns:
      - dict with:
          "model": the trained model (restored to best dev checkpoint if early stopping),
          "history": list of {"epoch": int, "train_loss": float, "dev_metrics": dict | None}
    """
    def _optimizer_for(model: nn.Module):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "layernorm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return AdamW(groups, lr=learning_rate, weight_decay=0.0)

    model.to(device)
    optimizer = _optimizer_for(model)
    total_steps = (len(train_premises) + batch_size - 1) // batch_size * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(warmup_steps), 0),
        num_training_steps=max(total_steps, 1),
    )

    n_train = len(train_premises)
    history: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    global_step = 0

    print(f"\nStarting NLI training on {n_train} examples...")

    # early stopping state
    best_dev_acc: Optional[float] = None
    best_dev_metrics: Optional[Dict[str, Any]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # simple shuffle
        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"Epoch {epoch}/{num_epochs}",
        ):
            optimizer.zero_grad()

            batch_indices = idxs[start : start + batch_size]
            if not batch_indices:
                continue

            batch_p = [train_premises[i] for i in batch_indices]
            batch_h = [train_hypotheses[i] for i in batch_indices]
            batch_y = [train_labels[i] for i in batch_indices]

            enc = tokenizer(
                batch_p,
                batch_h,
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
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
          
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
        print(f"\nEpoch {epoch}: train loss = {avg_loss:.4f}")

        dev_metrics = None
        if (
            eval_on_dev
            and dev_premises is not None
            and dev_hypotheses is not None
            and dev_labels is not None
        ):
            dev_metrics = compute_nli_accuracy(
                model,
                tokenizer,
                dev_premises,
                dev_hypotheses,
                dev_labels,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            print(
                f"[NLI] Dev accuracy: {dev_metrics['accuracy']:.4f} | "
                f"macro-F1: {dev_metrics['macro_f1']:.4f} | "
                f"micro-F1: {dev_metrics['micro_f1']:.4f} | "
                f"precision (macro): {dev_metrics['precision']:.4f} | "
                f"recall (macro): {dev_metrics['recall']:.4f}"
            )

            # if early_stopping is int (tracks whether progress was achieved)
            curr_acc = dev_metrics["accuracy"]
            if best_dev_acc is None or curr_acc > best_dev_acc + min_delta:
                best_dev_acc = curr_acc
                best_dev_metrics = dict(dev_metrics)
                # store on CPU to avoid GPU memory growth
                best_state_dict = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                no_improve = 0
            elif early_stopping_patience is not None:
                no_improve += 1

        # record history for this epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

        # ---- check early stopping condition after each epoch ----
        if early_stopping_patience is not None and best_dev_acc is not None:
            if no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered after epoch {epoch} "
                    f"(no dev improvement for {early_stopping_patience} epochs)."
                )
                break

    # restore best dev checkpoint if we tracked one
    if best_state_dict is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state_dict.items()}
        )
        if best_epoch is not None:
            print(
                f"[NLI] Restored best dev checkpoint from epoch {best_epoch} "
                f"(accuracy={best_dev_acc:.4f})"
            )

    return {
        "model": model,
        "history": history,
        "batch_history": batch_history,
        "best_dev_metrics": best_dev_metrics,
        "best_epoch": best_epoch,
    }
