import math
import torch
from tqdm import tqdm


@torch.no_grad()
def compute_masked_token_accuracy(model, tokenizer, eval_loader, device):
    """
    Masked-token accuracy on an MLM-style dataset.

    Assumes each batch from eval_loader has:
      - "input_ids": (B, L)
      - "labels": (B, L), with -100 at non-masked positions
      - optional "attention_mask"
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0

    for batch in tqdm(eval_loader, desc="Masked-token accuracy"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pad = tokenizer.pad_token_id
            if pad is not None:
                attention_mask = (input_ids != pad).long()
            else:
                attention_mask = torch.ones_like(input_ids)
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
    """
    Agreement between student and teacher predictions on masked tokens.

    Returns:
        {"agreement": float in [0,1], "total": int}
    """
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
            if pad is not None:
                attention_mask = (input_ids != pad).long()
            else:
                attention_mask = torch.ones_like(input_ids)
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
    """
    Masked-token perplexity for an MLM.

    Computes the average cross-entropy over masked positions (labels != -100)
    and returns both the mean loss and exp(mean_loss) as perplexity.

    Returns:
        {
            "loss": float,
            "perplexity": float,
            "tokens": int,   # number of masked positions
        }
    """
    model.to(device)
    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",  # we'll normalize by #masked tokens ourselves
    )

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(eval_loader, desc="Masked-token perplexity"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pad = tokenizer.pad_token_id
            if pad is not None:
                attention_mask = (input_ids != pad).long()
            else:
                attention_mask = torch.ones_like(input_ids)

        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # einops - check that out. 
        # logits.view(-1) - logits.rearrange('B L -> L B')
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

    return {
        "loss": float(mean_loss),
        "perplexity": float(ppl),
        "tokens": total_tokens,
    }


### also expand on perplexity. KL-div is already fixed (as is CE loss)
### either project for the similarity or remove it 
#### evals: FOlders( tasks; Metrics) 
### tasks MLP, sentiment, nli, ner evaluations
### metrics perplexity, F1, acc, etc


#### Folder Pipeline: import everything relevant pass through one big file
#### Run the Pipeline and pass out all metrics in the end 