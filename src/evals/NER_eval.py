from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForTokenClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_ner_tagger(
    base_model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dropout: float = 0.1,
    subfolder: str | None = None,
):
    """
    Create an XLM-RoBERTa-based NER tagger.
    """
    cfg_kwargs = {}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(base_model_name, **cfg_kwargs)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = dropout

    model = XLMRobertaForTokenClassification.from_pretrained(
        base_model_name,
        config=config,
        **cfg_kwargs,
    )
    return model


@torch.no_grad()
def compute_ner_accuracy(
    model,
    tokenizer,
    sentences: List[List[str]],
    labels: List[List[int]],
    device: str,
    batch_size: int = 32,
    max_length: int = 128,
    ignore_index: int = -100,
):
    """
    Token-level evaluation for NER.
    """
    model.to(device)
    model.eval()

    y_true_all = []
    y_pred_all = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="NER eval"):
        batch_sents = sentences[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        if not batch_sents:
            continue

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
        for b_idx, sent_labels in enumerate(batch_labels):
            word_ids = enc.word_ids(batch_index=b_idx)
            cur_labels = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    cur_labels.append(ignore_index)
                elif wid != prev_wid:
                    cur_labels.append(int(sent_labels[wid]))
                    prev_wid = wid
                else:
                    cur_labels.append(ignore_index)
            aligned.append(cur_labels)

        label_tensor = torch.tensor(aligned, dtype=torch.long, device=device)

        outputs = model(**enc)
        logits = outputs.logits  # (B, L, C)
        preds = torch.argmax(logits, dim=-1)

        # flatten and ignore padding / subwords
        for b in range(preds.size(0)):
            for t in range(preds.size(1)):
                true_id = int(label_tensor[b, t].item())
                if true_id == ignore_index:
                    continue
                pred_id = int(preds[b, t].item())
                y_true_all.append(true_id)
                y_pred_all.append(pred_id)

    if not y_true_all:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    micro_f1 = f1_score(y_true_all, y_pred_all, average="micro")
    prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "precision": float(prec),
        "recall": float(rec),
    }
