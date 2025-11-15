import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@torch.no_grad()
def _get_sentence_embeddings(model, tokenizer, texts, device, batch_size: int = 32, max_length: int = 128):
    """
    Return [CLS] token embeddings for a list of texts.

    Args:
        model: A sequence classification (or encoder) model.
        tokenizer: Matching tokenizer.
        texts: List of strings.
        device: torch.device or "cuda"/"cpu" string.
        batch_size: Batch size for encoding.
        max_length: Max sequence length.

    Returns:
        Tensor of shape (len(texts), hidden_dim) on CPU.
    """
    model.eval()
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding sentences"):
        batch_texts = texts[i : i + batch_size]
        if not batch_texts:
            continue

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc, output_hidden_states=True)
        # Last hidden state: (B, L, H); XLM-R uses first token <s> as sentence representation
        last_hidden = outputs.hidden_states[-1]
        cls_emb = last_hidden[:, 0, :]  # (B, H)

        all_embs.append(cls_emb.cpu())

    if not all_embs:
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.empty(0, hidden_size)

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def compute_sentiment_accuracy(
    model,
    tokenizer,
    texts,
    labels,
    device,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    Compute sentiment classification metrics on a list of (text, label).

    Args:
        model: XLMRobertaForSequenceClassification (teacher or student).
        tokenizer: Matching tokenizer.
        texts: List of review strings.
        labels: List of integer label ids (same ordering as texts).
        device: torch.device or "cuda"/"cpu" string.
        batch_size: Batch size.
        max_length: Max sequence length.

    Returns:
        dict with accuracy, macro_f1, micro_f1, precision, recall.
    """
    model.eval()
    preds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment inference"):
        batch_texts = texts[i : i + batch_size]
        if not batch_texts:
            continue

        enc = tokenizer(
            batch_texts,
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


@torch.no_grad()
def compute_sentiment_embedding_similarity(
    student,
    teacher,
    tokenizer,
    texts,
    device,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    Compare student vs teacher sentence embeddings via cosine similarity.

    We:
      * encode each review with both models
      * take [CLS] from the last hidden layer as sentence embedding
      * L2-normalize and compute cosine similarity per example
      * return the mean similarity across all examples
    """
    teacher_embs = _get_sentence_embeddings(
        teacher, tokenizer, texts, device, batch_size=batch_size, max_length=max_length
    )  # (N, H_t)
    student_embs = _get_sentence_embeddings(
        student, tokenizer, texts, device, batch_size=batch_size, max_length=max_length
    )  # (N, H_s)

    if teacher_embs.size(0) == 0 or student_embs.size(0) == 0:
        return {"similarity": 0.0}

    # Align hidden dims if different (defensive)
    if teacher_embs.size(1) != student_embs.size(1):
        d = min(teacher_embs.size(1), student_embs.size(1))
        teacher_embs = teacher_embs[:, :d]
        student_embs = student_embs[:, :d]

    teacher_norm = F.normalize(teacher_embs, dim=-1)
    student_norm = F.normalize(student_embs, dim=-1)

    cos = (teacher_norm * student_norm).sum(dim=-1)  # (N,)
    mean_sim = cos.mean().item()

    return {"similarity": float(mean_sim)}


def create_sentiment_classifier(
    base_model_name: str,
    num_labels: int,
    label2id: dict | None = None,
    id2label: dict | None = None,
    dropout: float | None = None,
    subfolder: str | None = None,
    **model_kwargs,
):
    """
    Create an XLM-R sentiment classifier on top of a pretrained base.

    Works both for:
      - plain HF models like "FacebookAI/xlm-roberta-base"
      - students stored in a subfolder (e.g. subfolder="model")

    Args:
        base_model_name: HF model id or local path.
        num_labels: number of sentiment classes.
        label2id: optional mapping from string labels to ids.
        id2label: optional mapping from ids to string labels.
        dropout: optional dropout rate for the classification head & encoder.
        subfolder: if weights/config live in a HF subfolder (e.g. "model").
        **model_kwargs: forwarded to .from_pretrained (e.g. token=...).

    Returns:
        XLMRobertaForSequenceClassification instance.
    """
    # ---- Build kwargs for config / model without passing subfolder=None ----
    config_kwargs = dict(model_kwargs)
    if subfolder is not None:
        config_kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(
        base_model_name,
        **config_kwargs,
    )

    config.num_labels = num_labels
    if label2id is not None:
        config.label2id = label2id
    if id2label is not None:
        config.id2label = id2label

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
