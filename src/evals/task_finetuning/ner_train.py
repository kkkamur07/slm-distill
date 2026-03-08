import copy
from typing import List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForTokenClassification, PreTrainedModel
from src.evals.ner_eval import compute_ner_accuracy


def _load_base_weights(model: nn.Module, base_model: PreTrainedModel) -> None:
    """Load compatible weights from a base model, ignoring mismatched heads."""
    target_state = model.state_dict()
    base_state = base_model.state_dict()
    prefix = getattr(model, "base_model_prefix", "")

    adapted_state = {}
    for key, tensor in base_state.items():
        candidates = [key]
        if prefix and not key.startswith(f"{prefix}."):
            candidates.append(f"{prefix}.{key}")

        for cand in candidates:
            if cand in target_state and target_state[cand].shape == tensor.shape:
                adapted_state[cand] = tensor
                break

    model.load_state_dict(adapted_state, strict=False)


def create_ner_tagger(
    base_model_name: str | None,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    dropout: float = 0.1,
    subfolder: str | None = None,
    base_model: PreTrainedModel | None = None,
):
    """
    Create an XLM-RoBERTa-based NER tagger.
    """
    cfg_kwargs = {}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder

    if base_model is not None:
        config = copy.deepcopy(base_model.config)
    elif base_model_name is not None:
        config = AutoConfig.from_pretrained(base_model_name, **cfg_kwargs)
    else:
        raise ValueError("Provide either base_model or base_model_name.")

    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = dropout

    if base_model is not None:
        model = XLMRobertaForTokenClassification(config)
        _load_base_weights(model, base_model)
    else:
        model = XLMRobertaForTokenClassification.from_pretrained(
            base_model_name,
            config=config,
            **cfg_kwargs,
        )
    return model


def train_ner_model(
    model: nn.Module,
    tokenizer,
    train_sentences: List[List[str]],
    train_labels: List[List[int]],
    dev_sentences: Optional[List[List[str]]],
    dev_labels: Optional[List[List[int]]],
    device: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    weight_decay: float,
    min_delta: float,
    early_stopping_patience: Optional[int],
    ignore_index: int = -100,
    eval_on_dev: bool = True,
    warmup_steps: int = 0,
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
    def _optimizer_for(model: nn.Module):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "layernorm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return AdamW(groups, lr=learning_rate, weight_decay=0.0)

    model.to(device)
    optimizer = _optimizer_for(model)
    total_steps = (len(train_sentences) + batch_size - 1) // batch_size * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(warmup_steps), 0),
        num_training_steps=max(total_steps, 1),
    )

    n_train = len(train_sentences)
    history: List[Dict[str, Any]] = []

    print(f"\nStarting NER training on {n_train} sentences...")

    # early stopping state
    best_dev_acc: Optional[float] = None
    best_dev_metrics: Optional[Dict[str, Any]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
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
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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
            curr_acc = dev_metrics["accuracy"]
            if best_dev_acc is None or curr_acc > best_dev_acc + min_delta:
                best_dev_acc = curr_acc
                best_dev_metrics = dict(dev_metrics)
                best_state_dict = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                no_improve = 0
            elif early_stopping_patience is not None:
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
        if best_epoch is not None:
            print(
                f"[NER] Restored best dev checkpoint from epoch {best_epoch} "
                f"(accuracy={best_dev_acc:.4f})"
            )

    return {
        "model": model,
        "history": history,
        "best_dev_metrics": best_dev_metrics,
        "best_epoch": best_epoch,
    }
