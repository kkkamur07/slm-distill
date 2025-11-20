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
) -> Dict[str, Any]:
    """
    Inputs:
      - model: XLMRobertaForTokenClassification (teacher or student)
      - tokenizer: matching tokenizer (fast, with word_ids)
      - train_sentences / train_labels: word-level tokens and label ids
      - dev_sentences / dev_labels: dev set (can be None)
      - device: "cuda" or "cpu"

    Returns:
      - {"model": model, "history": [...]}
    """
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    n_train = len(train_sentences)
    history: List[Dict[str, Any]] = []

    print(f"\nStarting NER training on {n_train} sentences...")

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
                        # first subword of this word
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

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "dev_metrics": dev_metrics,
            }
        )

    return {"model": model, "history": history}