import torch
from tqdm import tqdm

@torch.no_grad()
def compute_next_token_accuracy(model, tokenizer, eval_loader, device):
    """
    Simple next-token (end-of-sequence) prediction accuracy for MLM models.
    For each sequence we mask the final token and ask the model to predict it.
    Returns accuracy in [0.0, 1.0].
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    total = 0
    correct = 0

    for batch in tqdm(eval_loader, desc="NTP eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)

        # ground-truth is the last token in each sequence
        labels = input_ids[:, -1].clone()
        # mask last token
        masked = input_ids.clone()
        masked[:, -1] = mask_id

        out = model(input_ids=masked, attention_mask=attention_mask)
        logits = out.logits  # (B, seq_len, V)
        preds = torch.argmax(logits[:, -1, :], dim=-1)  # predictions for final position

        valid = attention_mask[:, -1] == 1  # only consider sequences with last token present
        if valid.any():
            total += int(valid.sum().item())
            correct += int((preds[valid] == labels[valid]).sum().item())

    return (correct / total) if total > 0 else 0.0


@torch.no_grad()
def compare_student_teacher_next_token(student, teacher, tokenizer, eval_loader, device):
    """
    Computes:
      - student_accuracy: student vs ground-truth (same scheme as compute_next_token_accuracy)
      - teacher_accuracy: teacher vs ground-truth
      - agreement: fraction where student prediction == teacher prediction
    Returns dict of floats in [0.0, 1.0].
    """
    student.eval()
    teacher.eval()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    total = 0
    student_correct = 0
    teacher_correct = 0
    agreement_count = 0

    for batch in tqdm(eval_loader, desc="NTP student-teacher"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)

        labels = input_ids[:, -1].clone()
        masked = input_ids.clone()
        masked[:, -1] = mask_id

        out_s = student(input_ids=masked, attention_mask=attention_mask)
        out_t = teacher(input_ids=masked, attention_mask=attention_mask)

        preds_s = torch.argmax(out_s.logits[:, -1, :], dim=-1)
        preds_t = torch.argmax(out_t.logits[:, -1, :], dim=-1)

        valid = attention_mask[:, -1] == 1
        if not valid.any():
            continue

        total += int(valid.sum().item())
        student_correct += int((preds_s[valid] == labels[valid]).sum().item())
        teacher_correct += int((preds_t[valid] == labels[valid]).sum().item())
        agreement_count += int((preds_s[valid] == preds_t[valid]).sum().item())

    if total == 0:
        return {"student_accuracy": 0.0, "teacher_accuracy": 0.0, "agreement": 0.0}

    return {
        "student_accuracy": student_correct / total,
        "teacher_accuracy": teacher_correct / total,
        "agreement": agreement_count / total,
    }