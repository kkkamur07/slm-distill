"""Sentiment analysis evaluation for distilled models.

This module evaluates models on sentiment classification:
- compute_sentiment_accuracy: Sentiment classification metrics
  - Accuracy for positive/negative/neutral sentiment
  - Macro and micro F1 scores
  - Precision and recall

Used to assess whether the distilled model maintains sentiment understanding
capabilities in the target language.

Example:
    >>> from src.evals.sentiment_eval import compute_sentiment_accuracy
    >>> 
    >>> texts = ["यह फिल्म बहुत अच्छी है!", "बुरी फिल्म थी।", ...]
    >>> labels = [1, 0, ...]  # 0=negative, 1=positive
    >>> 
    >>> metrics = compute_sentiment_accuracy(
    ...     model=sentiment_model,
    ...     tokenizer=tokenizer,
    ...     texts=texts,
    ...     labels=labels,
    ...     device="cuda",
    ...     batch_size=32,
    ...     max_length=128
    ... )
    >>> 
    >>> print(f"Sentiment Accuracy: {metrics['accuracy']:.2%}")
    >>> print(f"Macro F1: {metrics['macro_f1']:.4f}")
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@torch.no_grad()
def compute_sentiment_accuracy(
    model,
    tokenizer,
    texts,
    labels,
    device: str,
    batch_size: int,
    max_length: int,
):
    model.to(device)
    model.eval()
    preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment eval"):
        batch = texts[i : i + batch_size]
        if not batch:
            continue

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        preds.extend(batch_preds)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "precision": float(prec),
        "recall": float(rec),
    }

