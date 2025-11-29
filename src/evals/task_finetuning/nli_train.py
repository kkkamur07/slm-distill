from typing import List, Dict, Any, Optional
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification
from src.evals.nli_eval import compute_nli_accuracy


def create_nli_classifier(
    base_model_name: str,
    num_labels: int,
    dropout: float | None,
    subfolder: str | None = None,
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

    config = AutoConfig.from_pretrained(
        base_model_name,
        **config_kwargs,
    )

    config.num_labels = num_labels

    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout

    model_kwargs2 = dict(model_kwargs)
    if subfolder is not None:
        model_kwargs2["subfolder"] = subfolder

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
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # === DEBUG: track whether classifier weights actually change ===
    tracked_param_name = "classifier.out_proj.weight"
    tracked_prev = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == tracked_param_name:
                tracked_prev = param.detach().clone()
                print(
                    f"[DEBUG] Tracking parameter: {tracked_param_name}, "
                    f"mean abs value = {tracked_prev.abs().mean().item():.6e}"
                )
                break
        if tracked_prev is None:
            print(f"[DEBUG] Could not find parameter {tracked_param_name} to track.")

    n_train = len(train_premises)
    history: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    global_step = 0

    print(f"\nStarting NLI training on {n_train} examples...")

    # early stopping state
    best_dev_acc: Optional[float] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
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
            if early_stopping_patience is not None:
                curr_acc = dev_metrics["accuracy"]
                if best_dev_acc is None or curr_acc > best_dev_acc + min_delta:
                    best_dev_acc = curr_acc
                    # store on CPU to avoid GPU memory growth
                    best_state_dict = {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1

        # === DEBUG: check how much the tracked weights changed this epoch ===
        if tracked_prev is not None:
            with torch.no_grad():
                current = None
                for name, param in model.named_parameters():
                    if name == tracked_param_name:
                        current = param.detach().clone()
                        break
                if current is not None:
                    diff = (current - tracked_prev).abs().mean().item()
                    print(
                        f"[DEBUG] Mean abs change in {tracked_param_name} "
                        f"since last check: {diff:.6e}"
                    )
                    tracked_prev = current

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

    return {
        "model": model,
        "history": history,
        "batch_history": batch_history,
    }
