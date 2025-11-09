import torch
from tqdm import tqdm
from transformers import pipeline
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, Subset


import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU

@torch.no_grad()
def _mlm_fill_strings(model, tokenizer, loader, device):
    """
    Returns two aligned lists: (hyps, refs_gt)
    - hyps: input with masked positions filled by model predictions
    - refs_gt: input with masked positions filled by ground-truth labels
    """
    model.eval()
    hyps, refs = [], []

    for batch in tqdm(loader, desc="Filling masks for BLEU"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # ground-truth ids at masked pos; -100 elsewhere

        attn = batch.get("attention_mask", None)
        if attn is None:
            if tokenizer.pad_token_id is not None:
                attn = (input_ids != tokenizer.pad_token_id).long()
            else:
                attn = torch.ones_like(input_ids)
        attn = attn.to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        preds = torch.argmax(logits, dim=-1)

        mask_pos = labels != -100

        hyp_ids = input_ids.clone()
        hyp_ids[mask_pos] = preds[mask_pos]

        ref_ids = input_ids.clone()
        ref_ids[mask_pos] = labels[mask_pos]

        hyps.extend([s.strip() for s in tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)])
        refs.extend([s.strip() for s in tokenizer.batch_decode(ref_ids, skip_special_tokens=True)])

    return hyps, refs

def compute_bleu_ground_truth(model, tokenizer, eval_loader, device, lowercase=True):
    """
    BLEU between model-filled sentences and ground-truth-filled sentences.
    Returns float in [0, 1].
    """
    hyps, refs = _mlm_fill_strings(model, tokenizer, eval_loader, device)
    assert len(hyps) == len(refs) and len(hyps) > 0, "Empty or misaligned data."

    bleu = BLEU(lowercase=lowercase)  # sacrebleu default tokenization=13a
    score = bleu.corpus_score(hyps, [refs]).score
    return score / 100.0

@torch.no_grad()
def compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device, lowercase=True):
    """
    BLEU between student-filled and teacher-filled sentences.
    Runs both models in a single pass to keep exact alignment.
    """
    student.eval(); teacher.eval()
    hyps, refs = [], []

    for batch in tqdm(eval_loader, desc="Student–Teacher BLEU"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attn = batch.get("attention_mask", None)
        if attn is None:
            if tokenizer.pad_token_id is not None:
                attn = (input_ids != tokenizer.pad_token_id).long()
            else:
                attn = torch.ones_like(input_ids)
        attn = attn.to(device)

        mask_pos = labels != -100

        s_logits = student(input_ids=input_ids, attention_mask=attn).logits
        s_pred = torch.argmax(s_logits, dim=-1)
        s_ids = input_ids.clone()
        s_ids[mask_pos] = s_pred[mask_pos]

        t_logits = teacher(input_ids=input_ids, attention_mask=attn).logits
        t_pred = torch.argmax(t_logits, dim=-1)
        t_ids = input_ids.clone()
        t_ids[mask_pos] = t_pred[mask_pos]

        hyps.extend([t.strip() for t in tokenizer.batch_decode(s_ids, skip_special_tokens=True)])
        refs.extend([t.strip() for t in tokenizer.batch_decode(t_ids, skip_special_tokens=True)])

    assert len(hyps) == len(refs) and len(hyps) > 0, "Empty or misaligned data."

    bleu = BLEU(lowercase=lowercase)
    score = bleu.corpus_score(hyps, [refs]).score
    return score / 100.0


# # Get BLEU scores this is simply for me to keep track on how to implement them, I'll delete all below soon
# ground_truth_bleu = compute_bleu_ground_truth(student, tokenizer, eval_loader, device)
# teacher_bleu = compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device)

# print(f"Ground Truth BLEU: {ground_truth_bleu:.4f}")
# print(f"Teacher BLEU: {teacher_bleu:.4f}")

"""
| **Purpose**                         | **Comparison**                   | **Interpretation**                                                                                                                |
| ----------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Evaluate distillation quality**   | **Student vs Teacher**           | How close the student’s *generated outputs* are to the teacher’s outputs. Measures imitation fidelity.                            |
| **Evaluate real-world performance** | **Student vs Human (reference)** | How well the student performs the actual *task* (e.g. translation, summarization). Measures true quality, independent of teacher. |
| **Check teacher quality baseline**  | **Teacher vs Human (reference)** | Confirms the teacher’s outputs are indeed high-quality, so it’s worth mimicking.                                                  |

Next tests : Next token prediction, perplexity and BLEU + CHRF, semantic distance, BERT similarity, Logit Alignment (or the relevant Energy)
task specific MMLU (student - teacher comparisons) Indic glue (11 indic languages) or Indic MMLU- Pro or Indic gen benches

order: BLEU + CHRF, Next token prediction, MMLU
"""