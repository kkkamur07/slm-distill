import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@torch.no_grad()
def compute_nli_accuracy(
    model,
    tokenizer,
    premises,
    hypotheses,
    labels,
    device,
    batch_size: int,
    max_length: int,
):
    """
    Args:
        model: XLMRobertaForSequenceClassification (teacher or student).
        tokenizer: Matching tokenizer.
        premises: list of premise strings.
        hypotheses: list of hypothesis strings.
        labels: list of integer label ids (same ordering as premises/hypotheses).
        device: torch.device or "cuda"/"cpu" string.
        batch_size: Batch size.
        max_length: Max sequence length.

    Returns:
        dict with accuracy, macro_f1, micro_f1, precision, recall.
    """
    model.eval()
    preds = []

    for i in tqdm(range(0, len(premises), batch_size), desc="NLI inference"):
        batch_p = premises[i : i + batch_size]
        batch_h = hypotheses[i : i + batch_size]
        if not batch_p:
            continue

        enc = tokenizer(
            batch_p,
            batch_h,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc)
        logits = outputs.logits  # (B, num_labels)
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        preds.extend(batch_preds)

    y_true = labels
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "precision": float(prec),
        "recall": float(rec),
    }
