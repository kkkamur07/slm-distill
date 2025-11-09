import torch
from tqdm import tqdm
from sacrebleu.metrics import CHRF

@torch.no_grad()
def _mlm_fill_strings(model, tokenizer, loader, device):
    """
    Returns two aligned lists: (hyps, refs)
    hyps: inputs with masked positions filled by model predictions
    refs: inputs with masked positions filled by ground-truth labels
    """
    model.eval()
    hyps, refs = [], []

    for batch in tqdm(loader, desc="Filling masks"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # ground-truth tokens at masked pos; -100 elsewhere

        attn = batch.get("attention_mask")
        if attn is None:
            pad = tokenizer.pad_token_id
            attn = (input_ids != pad).long() if pad is not None else torch.ones_like(input_ids)
        attn = attn.to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        preds = torch.argmax(logits, dim=-1)

        mask_pos = (labels != -100)

        hyp_ids = input_ids.clone()
        hyp_ids[mask_pos] = preds[mask_pos]

        ref_ids = input_ids.clone()
        ref_ids[mask_pos] = labels[mask_pos]

        hyps.extend([s.strip() for s in tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)])
        refs.extend([s.strip() for s in tokenizer.batch_decode(ref_ids, skip_special_tokens=True)])

    return hyps, refs

def compute_chrf_ground_truth(model, tokenizer, eval_loader, device, word_order=2):
    """
    chrF/chrF++ between model-filled and ground-truth-filled sentences.
    Returns float in [0,1].
    """
    hyps, refs = _mlm_fill_strings(model, tokenizer, eval_loader, device)
    chrf = CHRF(word_order=word_order)  # word_order=2 => chrF++
    score = chrf.corpus_score(hyps, [refs]).score
    return score / 100.0

@torch.no_grad()
def compute_chrf_student_teacher(student, teacher, tokenizer, eval_loader, device, word_order=2):
    """
    chrF/chrF++ between student-filled and teacher-filled sentences.
    Runs both in one pass for alignment. Returns float in [0,1].
    """
    student.eval(); teacher.eval()

    hyps, refs = [], []
    for batch in tqdm(eval_loader, desc="Student–Teacher chrF"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attn = batch.get("attention_mask")
        if attn is None:
            pad = tokenizer.pad_token_id
            attn = (input_ids != pad).long() if pad is not None else torch.ones_like(input_ids)
        attn = attn.to(device)

        mask_pos = (labels != -100)

        s_logits = student(input_ids=input_ids, attention_mask=attn).logits
        t_logits = teacher(input_ids=input_ids, attention_mask=attn).logits
        s_pred = torch.argmax(s_logits, dim=-1)
        t_pred = torch.argmax(t_logits, dim=-1)

        s_ids = input_ids.clone(); s_ids[mask_pos] = s_pred[mask_pos]
        t_ids = input_ids.clone(); t_ids[mask_pos] = t_pred[mask_pos]

        hyps.extend([t.strip() for t in tokenizer.batch_decode(s_ids, skip_special_tokens=True)])
        refs.extend([t.strip() for t in tokenizer.batch_decode(t_ids, skip_special_tokens=True)])

    chrf = CHRF(word_order=word_order)
    score = chrf.corpus_score(hyps, [refs]).score
    return score / 100.0

