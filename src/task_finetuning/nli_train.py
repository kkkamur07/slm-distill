from typing import List, Dict, Any, Optional
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from src.evals.nli_eval import compute_nli_accuracy

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
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
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
          "model": the trained model,
          "history": list of {"epoch": int, "train_loss": float, "dev_metrics": dict | None}
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # === DEBUG: track whether classifier weights actually change ===
    tracked_param_name = "classifier.out_proj.weight"
    tracked_prev = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == tracked_param_name:
                tracked_prev = param.detach().clone()
                print(f"[DEBUG] Tracking parameter: {tracked_param_name}, "
                      f"mean abs value = {tracked_prev.abs().mean().item():.6e}")
                break
        if tracked_prev is None:
            print(f"[DEBUG] Could not find parameter {tracked_param_name} to track.")

    n_train = len(train_premises)
    history: List[Dict[str, Any]] = []

    print(f"\nStarting NLI training on {n_train} examples...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()  # simple shuffle

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"Epoch {epoch}/{num_epochs}",
        ):
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
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch}: train loss = {avg_loss:.4f}")

        dev_metrics = None
        if eval_on_dev and dev_premises is not None and dev_labels is not None:
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
                f"Dev accuracy: {dev_metrics['accuracy']:.4f} | "
                f"macro-F1: {dev_metrics['macro_f1']:.4f} | "
                f"micro-F1: {dev_metrics['micro_f1']:.4f} | "
                f"precision (macro): {dev_metrics['precision']:.4f} | "
                f"recall (macro): {dev_metrics['recall']:.4f}"
            )

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
                    
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

    return {"model": model, "history": history}