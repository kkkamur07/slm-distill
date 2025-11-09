import torch
from tqdm import tqdm
from bert_score import score as bert_score_fn

@torch.no_grad()
def _mlm_fill_strings(model, tokenizer, loader, device):
    """
    For each example, fill masked positions with model predictions and
    also reconstruct the ground-truth full sentence by inserting labels.
    Returns (hyps, refs) as lists of strings aligned 1:1.
    """
    model.eval()
    hyps, refs = [], []

    for batch in tqdm(loader, desc="Filling masks"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # labels: original token ids at masked pos, -100 elsewhere
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is None:
            # Best effort: infer 0 on pads if tokenizer has pad id; else fall back to ones.
            if tokenizer.pad_token_id is not None:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1)

        mask_pos = labels != -100  # positions to replace

        # Build hypothesis by filling predicted tokens into the original input
        hyp_ids = input_ids.clone()
        hyp_ids[mask_pos] = preds[mask_pos]

        # Build reference by filling ground-truth tokens into the original input
        ref_ids = input_ids.clone()
        ref_ids[mask_pos] = labels[mask_pos]

        hyps.extend([s.strip() for s in tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)])
        refs.extend([s.strip() for s in tokenizer.batch_decode(ref_ids, skip_special_tokens=True)])

    return hyps, refs

def _bert_score(hyps, refs, lang="en", rescale=True):
    # Avoid baseline rescaling without a language
    use_rescale = rescale and (lang is not None)
    P, R, F = bert_score_fn(hyps, refs, lang=lang, rescale_with_baseline=use_rescale)
    return float(F.mean().item())

def compute_bertscore_ground_truth(model, tokenizer, eval_loader, device, lang="en", rescale=True):
    """
    BERTScore between model-filled sentences and the ground-truth (original) sentences.
    """
    hyps, refs = _mlm_fill_strings(model, tokenizer, eval_loader, device)
    return _bert_score(hyps, refs, lang=lang, rescale=rescale)

@torch.no_grad()
def compute_bertscore_student_teacher(student, teacher, tokenizer, eval_loader, device, lang="en", rescale=True):
    """
    BERTScore between student-filled sentences and teacher-filled sentences.
    Runs both models in a single pass to keep alignment.
    """
    student.eval(); teacher.eval()
    hyps, refs = [], []

    for batch in tqdm(eval_loader, desc="Student–Teacher fill"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is None:
            if tokenizer.pad_token_id is not None:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        mask_pos = labels != -100

        # Student fill
        s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        s_preds = torch.argmax(s_logits, dim=-1)
        s_ids = input_ids.clone()
        s_ids[mask_pos] = s_preds[mask_pos]

        # Teacher fill
        t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        t_preds = torch.argmax(t_logits, dim=-1)
        t_ids = input_ids.clone()
        t_ids[mask_pos] = t_preds[mask_pos]

        hyps.extend([t.strip() for t in tokenizer.batch_decode(s_ids, skip_special_tokens=True)])
        refs.extend([t.strip() for t in tokenizer.batch_decode(t_ids, skip_special_tokens=True)])

    return _bert_score(hyps, refs, lang=lang, rescale=rescale)