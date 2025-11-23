import torch
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_sentiment_classifier(
    base_model_name: str,
    num_labels: int = 3,
    dropout: float = 0.0,
    subfolder: str | None = None,
):
    cfg_kwargs = {}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(base_model_name, **cfg_kwargs)
    config.num_labels = num_labels
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = dropout

    model = XLMRobertaForSequenceClassification.from_pretrained(
        base_model_name,
        config=config,
        ignore_mismatched_sizes=True,
        **cfg_kwargs,
    )
    return model


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
