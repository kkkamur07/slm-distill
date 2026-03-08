import copy
from typing import List, Dict, Any, Optional

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, XLMRobertaForSequenceClassification, PreTrainedModel

from src.evals.nli_eval import compute_nli_accuracy


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


def create_nli_classifier(
    base_model_name: str | None,
    num_labels: int,
    dropout: float | None,
    subfolder: str | None = None,
    base_model: PreTrainedModel | None = None,
    **model_kwargs,
):
    """Create classifier from backbone or HF id, with optional base weights."""
    config_kwargs = dict(model_kwargs)
    if subfolder is not None:
        config_kwargs["subfolder"] = subfolder

    if base_model is not None:
        config = copy.deepcopy(base_model.config)
    elif base_model_name is not None:
        config = AutoConfig.from_pretrained(base_model_name, **config_kwargs)
    else:
        raise ValueError("Provide either base_model or base_model_name.")

    config.num_labels = num_labels
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout

    model_kwargs2 = dict(model_kwargs)
    if subfolder is not None:
        model_kwargs2["subfolder"] = subfolder

    if base_model is not None:
        model = XLMRobertaForSequenceClassification(config)
        _load_base_weights(model, base_model)
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs2,
        )
    return model


def train_nli_model_score(
    model: nn.Module,
    tokenizer,
    train_premises: List[str],
    train_hypotheses: List[str],
    train_labels: List[int],
    device: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    weight_decay: float,
    lambda_ce: float = 0.9,
    baseline: bool = True,
) -> Dict[str, Any]:
    """Hybrid CE + policy gradient training for NLI."""
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    n_train = len(train_premises)
    history: List[Dict[str, Any]] = []
    print(f"\n[NLI][Score] Starting score-based training on {n_train} examples...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        idxs = torch.randperm(n_train).tolist()

        for start in tqdm(
            range(0, n_train, batch_size),
            desc=f"[NLI][Score] Epoch {epoch}/{num_epochs}",
        ):
            batch_idx = idxs[start : start + batch_size]
            if not batch_idx:
                continue

            batch_p = [train_premises[i] for i in batch_idx]
            batch_h = [train_hypotheses[i] for i in batch_idx]
            batch_y = torch.tensor(
                [train_labels[i] for i in batch_idx], dtype=torch.long, device=device
            )

            enc = tokenizer(
                batch_p,
                batch_h,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            optimizer.zero_grad()
            logits = model(**enc).logits  # (bs, num_labels)
            log_probs = torch.log_softmax(logits, dim=-1)

            samples = torch.distributions.Categorical(logits=logits).sample()
            sampled_logp = log_probs.gather(1, samples.view(-1, 1)).squeeze(1)
            reward = (samples == batch_y).float()
            if baseline:
                with torch.no_grad():
                    greedy = logits.argmax(dim=-1)
                    base = (greedy == batch_y).float()
            else:
                base = torch.zeros_like(reward)
            advantage = reward - base

            rl_loss = -(advantage * sampled_logp).mean()
            ce_loss = torch.nn.functional.cross_entropy(logits, batch_y)
            loss = lambda_ce * ce_loss + (1.0 - lambda_ce) * rl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"[NLI][Score] Epoch {epoch}: hybrid loss = {avg_loss:.4f}")

    return {"model": model, "history": history}
