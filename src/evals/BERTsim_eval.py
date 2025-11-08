import torch
from tqdm import tqdm
from bert_score import score as bert_score_fn

# COMET is optional; functions below will raise a helpful error if not installed
try:
    from comet import load_from_checkpoint
    _has_comet = True
except Exception:
    _has_comet = False


@torch.no_grad()
def _generate_mlm(model, tokenizer, loader, device):
    """Generate simple greedy predictions (MLM) and return list[str]."""
    model.eval()
    outputs = []
    for batch in tqdm(loader, desc="Generating (MLM)"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(out.logits, dim=-1)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        outputs.extend([t.strip() for t in decoded])
    return outputs


def _decode_labels(labels_tensor, tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lbls = labels_tensor.clone()
    lbls[lbls == -100] = pad_id
    return [t.strip() for t in tokenizer.batch_decode(lbls, skip_special_tokens=True)]


def compute_bertscore_ground_truth(model, tokenizer, eval_loader, device, lang=None, rescale=True):
    """Compute BERTScore (returns F1 float in [0,1]) comparing model -> ground-truth labels."""
    hyps = _generate_mlm(model, tokenizer, eval_loader, device)
    refs = []
    for batch in tqdm(eval_loader, desc="Preparing refs for BERTScore"):
        refs.extend(_decode_labels(batch["labels"].to(device), tokenizer))
    # bert_score returns (P, R, F) as tensors
    P, R, F = bert_score_fn(hyps, refs, lang=lang, rescale_with_baseline=rescale)
    return float(F.mean().item())

