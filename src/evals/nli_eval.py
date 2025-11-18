import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@torch.no_grad()
def _get_pair_embeddings(
    model,
    tokenizer,
    premises,
    hypotheses,
    device,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    Args:
        model: A sequence classification (or encoder) model.
        tokenizer: Matching tokenizer.
        premises: List of premise strings.
        hypotheses: List of hypothesis strings (same length as premises).
        device: torch.device or "cuda"/"cpu" string.
        batch_size: Batch size for encoding.
        max_length: Max sequence length.

    Returns:
        Tensor of shape (len(premises), hidden_dim) on CPU.
    """
    model.eval()
    all_embs = []

    for i in tqdm(range(0, len(premises), batch_size), desc="Encoding NLI pairs"):
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

        outputs = model(**enc, output_hidden_states=True)
        # Last hidden state: (B, L, H)
        last_hidden = outputs.hidden_states[-1]
        cls_emb = last_hidden[:, 0, :]  # (B, H)

        all_embs.append(cls_emb.cpu())

    if not all_embs:
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.empty(0, hidden_size)

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def compute_nli_accuracy(
    model,
    tokenizer,
    premises,
    hypotheses,
    labels,
    device,
    batch_size: int = 32,
    max_length: int = 128,
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


@torch.no_grad()
def compute_nli_embedding_similarity(
    student,
    teacher,
    tokenizer,
    premises,
    hypotheses,
    device,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    Comparing teacher - student embeddings (L2 used)
    """
    teacher_embs = _get_pair_embeddings(
        teacher,
        tokenizer,
        premises,
        hypotheses,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )  # (N, H_t)
    student_embs = _get_pair_embeddings(
        student,
        tokenizer,
        premises,
        hypotheses,
        device,
        batch_size=batch_size,
        max_length=max_length,
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


def create_nli_classifier(
    base_model_name: str,
    num_labels: int,
    dropout: float | None = None,
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
