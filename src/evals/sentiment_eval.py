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
    batch_size: int = 32,
    max_length: int = 128,
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
