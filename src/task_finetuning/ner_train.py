from typing import List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from src.evals.NER_eval import compute_ner_accuracy


def train_ner_model(
    model: nn.Module,
    tokenizer,
    train_sentences: List[List[str]],
    train_labels: List[List[int]],
    dev_sentences: Optional[List[List[str]]],
    dev_labels: Optional[List[List[int]]],
    device: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    ignore_index: int = -100,
    eval_on_dev: bool = True,
    weight_decay: float = 0.01,
    early_stopping_patience: Optional[int] = None,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    """
    Train NER model with optional weight decay and early stopping.

    Args:
        model: XLMRobertaForTokenClassification (teacher or student).
        tokenizer: matching tokenizer (fast, with word_ids).
        train_sentences: list of tokenized sentences, each a list of strings.
        train_labels: list of label sequences, each a list of ints.
        dev_sentences/dev_labels: dev set (or None).
        device: 'cuda' or 'cpu'.

    Returns:
        {
            "model": model (restored to best dev checkpoint if early stopping),
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

    n_train = len(train_sentences)
    history: List[Dict[str, Any]] = []

    print(f"\nStarting NER training on {n_train} sentences...")

    # early stopping state
    best_dev_acc: Optional[float] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"Epoch {epoch}/{num_epochs}",
        ):
            batch_idx = indices[start : start + batch_size]
            if not batch_idx:
                continue

            batch_sents = [train_sentences[i] for i in batch_idx]
            batch_label_seqs = [train_labels[i] for i in batch_idx]

            enc = tokenizer(
                batch_sents,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = enc.to(device)

            # align word-level labels to token-level labels
            aligned = []
            for b_idx, sent_labels in enumerate(batch_label_seqs):
                word_ids = enc.word_ids(batch_index=b_idx)
                cur_labels = []
                prev_wid = None
                for wid in word_ids:
                    if wid is None:
                        cur_labels.append(ignore_index)
                    elif wid != prev_wid:
                        # first subword: use the word label
                        cur_labels.append(int(sent_labels[wid]))
                        prev_wid = wid
                    else:
                        # subsequent subword: ignore in loss
                        cur_labels.append(ignore_index)
                aligned.append(cur_labels)

            label_tensor = torch.tensor(aligned, dtype=torch.long, device=device)

            outputs = model(**enc, labels=label_tensor)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch}: train loss = {avg_loss:.4f}")

        dev_metrics = None
        if eval_on_dev and dev_sentences is not None and dev_labels is not None:
            dev_metrics = compute_ner_accuracy(
                model,
                tokenizer,
                dev_sentences,
                dev_labels,
                device,
                batch_size=batch_size,
                max_length=max_length,
                ignore_index=ignore_index,
            )
            print(
                f"Dev accuracy: {dev_metrics['accuracy']:.4f} | "
                f"macro-F1: {dev_metrics['macro_f1']:.4f} | "
                f"micro-F1: {dev_metrics['micro_f1']:.4f} | "
                f"precision (macro): {dev_metrics['precision']:.4f} | "
                f"recall (macro): {dev_metrics['recall']:.4f}"
            )

            # early stopping bookkeeping
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

        # check early stopping condition
        if (
            early_stopping_patience is not None
            and best_dev_acc is not None
            and no_improve >= early_stopping_patience
        ):
            print(
                f"[NER] Early stopping after epoch {epoch} "
                f"(no dev improvement for {early_stopping_patience} epochs)."
            )
            break

    # restore best dev checkpoint if we tracked one
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    return {"model": model, "history": history}