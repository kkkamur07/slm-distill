import math
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def compute_masked_token_accuracy(model, tokenizer, eval_loader, device):
    model.to(device)
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
def compare_student_teacher_masked_token_agreement(
    student, teacher, tokenizer, eval_loader, device
):
    student.to(device)
    teacher.to(device)
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


@torch.no_grad()
def compute_masked_token_perplexity(model, tokenizer, eval_loader, device):
    model.to(device)
    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(eval_loader, desc="Masked-token perplexity"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pad = tokenizer.pad_token_id
            attention_mask = (input_ids != pad).long() if pad is not None else torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        total_loss += loss.item()
        total_tokens += int((labels != -100).sum().item())

    if total_tokens == 0:
        return {"loss": 0.0, "perplexity": float("inf"), "tokens": 0}

    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)

    return {"loss": float(mean_loss), "perplexity": float(ppl), "tokens": total_tokens}


def masked_token_kl(student, teacher, loader, device):
    student.to(device)
    teacher.to(device)
    student.eval()
    teacher.eval()
    total_kl, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            mask = labels != -100
            if not mask.any():
                continue

            t_logits = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits[mask]
            s_logits = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits[mask]

            t_logp = F.log_softmax(t_logits, dim=-1)
            s_logp = F.log_softmax(s_logits, dim=-1)
            t_p = t_logp.exp()
            kl = (t_p * (t_logp - s_logp)).sum(-1)

            total_kl += kl.sum().item()
            total_tokens += int(mask.sum().item())

    kl_mean = total_kl / total_tokens if total_tokens > 0 else float("nan")
    return kl_mean, total_tokens
