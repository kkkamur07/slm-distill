import torch
from tqdm import tqdm


@torch.no_grad()
def compute_masked_token_accuracy(model, tokenizer, eval_loader, device):
    """
    Compute accuracy on masked positions only.
    Returns float in [0,1].
    """
    model.eval()
    correct, total = 0, 0

    for batch in tqdm(eval_loader, desc="Masked-token accuracy"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pad = tokenizer.pad_token_id
            attention_mask = (input_ids != pad).long() if pad is not None else torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1)

        mask_pos = labels != -100
        total += int(mask_pos.sum().item())
        correct += int((preds[mask_pos] == labels[mask_pos]).sum().item())

    return (correct / total) if total > 0 else 0.0


@torch.no_grad()
def compare_student_teacher_masked_token_agreement(student, teacher, tokenizer, eval_loader, device):
    """
    Agreement between student and teacher predictions on masked tokens.
    Returns dict {'agreement': float in [0,1], 'total': int}.
    """
    student.eval()
    teacher.eval()
    total, agree = 0, 0

    for batch in tqdm(eval_loader, desc="Student–Teacher masked agreement"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pad = tokenizer.pad_token_id
            attention_mask = (input_ids != pad).long() if pad is not None else torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        s_pred = torch.argmax(s_logits, dim=-1)
        t_pred = torch.argmax(t_logits, dim=-1)

        mask_pos = labels != -100
        n = int(mask_pos.sum().item())
        if n == 0:
            continue
        total += n
        agree += int((s_pred[mask_pos] == t_pred[mask_pos]).sum().item())

    return {"agreement": (agree / total) if total > 0 else 0.0, "total": total}
