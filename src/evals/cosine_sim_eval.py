import torch
import torch.nn.functional as F
from tqdm import tqdm
from bert_score import score as bert_score_fn


@torch.no_grad()
def _mlm_fill_strings(model, tokenizer, loader, device):
    """
    For each example, fill masked positions with model predictions and
    reconstruct the ground-truth text by inserting labels.

    Returns:
        hyps: list[str]  - text with model predictions at masked positions
        refs: list[str]  - text with ground-truth tokens at masked positions

    Assumptions:
      - batch["input_ids"]: (B, L)
      - batch["labels"]: (B, L), with -100 at non-masked positions,
        and true token IDs at masked positions.
      - optionally batch["attention_mask"].
    """
    model.eval()
    hyps, refs = [], []

    for batch in tqdm(loader, desc="Filling masks"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get(
            "attention_mask",
            torch.ones_like(input_ids, device=device),
        ).to(device)

        # Model predictions
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits  # (B, L, vocab)
        preds = torch.argmax(logits, dim=-1)  # (B, L)

        mask_pos = labels != -100  # True only at masked tokens

        # Hypothesis: fill masked positions with predicted tokens
        hyp_ids = input_ids.clone()
        hyp_ids[mask_pos] = preds[mask_pos]

        # Reference: fill masked positions with ground-truth labels
        ref_ids = input_ids.clone()
        ref_ids[mask_pos] = labels[mask_pos]

        hyp_texts = tokenizer.batch_decode(
            hyp_ids, skip_special_tokens=True
        )
        ref_texts = tokenizer.batch_decode(
            ref_ids, skip_special_tokens=True
        )

        hyps.extend(t.strip() for t in hyp_texts)
        refs.extend(t.strip() for t in ref_texts)

    return hyps, refs


@torch.no_grad()
def compute_cosine_similarity_ground_truth(model, tokenizer, eval_loader, device):
    """
    Cosine similarity between:
      - the model's *LM-head-transformed hidden states* at masked positions, and
      - the *decoder/token vectors* of the ground-truth tokens at those positions.

    This matches more closely how MLM predictions are actually made:
      hidden -> lm_head.dense -> activation -> lm_head.layer_norm -> decoder.weight

    Only masked tokens (where labels != -100) are used.
    """
    model.eval()
    total_sim = 0.0
    count = 0

    for batch in tqdm(eval_loader, desc="Cosine sim (ground truth)"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get(
            "attention_mask",
            torch.ones_like(input_ids, device=device),
        ).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, H)

        # Masked positions only (MLM evaluation)
        mask_pos = labels != -100
        if mask_pos.sum().item() == 0:
            continue

        # Hidden states at masked positions
        hidden_masked = hidden_states[mask_pos]  # (N_masked, H_hidden)

        # Ground-truth token ids at masked positions
        masked_labels = labels[mask_pos]  # (N_masked,)

        # ---- LM-head transformation of hidden states ----
        # This follows the standard Roberta/XLM-R LM head:
        # dense -> activation -> layer_norm -> decoder
        lm_head = model.lm_head

        z = lm_head.dense(hidden_masked)
        # Some LM heads expose .activation, others assume GELU
        activation = getattr(lm_head, "activation", torch.nn.GELU())
        z = activation(z)
        z = lm_head.layer_norm(z)  # (N_masked, H_head)

        # ---- Ground-truth token vectors from decoder matrix ----
        # Typically tied to the embedding matrix
        decoder_weight = lm_head.decoder.weight  # (vocab_size, H_dec)
        label_vecs = decoder_weight[masked_labels]  # (N_masked, H_dec)

        # If dimensions differ, project to a common min dim (very defensive)
        if z.shape[-1] != label_vecs.shape[-1]:
            min_dim = min(z.shape[-1], label_vecs.shape[-1])
            z = z[:, :min_dim]
            label_vecs = label_vecs[:, :min_dim]

        sim = F.cosine_similarity(z, label_vecs, dim=-1)
        total_sim += sim.sum().item()
        count += sim.numel()

    return (total_sim / count) if count > 0 else 0.0


@torch.no_grad()
def compute_cosine_similarity_student_teacher(student, teacher, tokenizer, eval_loader, device):
    """
    Cosine similarity between student and teacher *hidden states* at masked positions.

    Only masked tokens (where labels != -100) are used.

    Returns:
        float: mean cosine similarity over all masked tokens.

    This is a representation-level distillation metric: do the student
    and teacher encoders produce similar contextual representations
    where the model is actually asked to predict (the masked tokens)?
    """
    student.eval()
    teacher.eval()

    total_sim = 0.0
    count = 0

    for batch in tqdm(eval_loader, desc="Cosine sim (student vs teacher)"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get(
            "attention_mask",
            torch.ones_like(input_ids, device=device),
        ).to(device)

        s_out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        t_out = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        s_hidden = s_out.hidden_states[-1]  # (B, L, H_s)
        t_hidden = t_out.hidden_states[-1]  # (B, L, H_t)

        # Masked positions only
        mask_pos = labels != -100
        if mask_pos.sum().item() == 0:
            continue

        s_masked = s_hidden[mask_pos]  # (N_masked, H_s)
        t_masked = t_hidden[mask_pos]  # (N_masked, H_t)

        # If student and teacher hidden sizes differ, align to min dim
        if s_masked.shape[-1] != t_masked.shape[-1]:
            min_dim = min(s_masked.shape[-1], t_masked.shape[-1])
            s_masked = s_masked[:, :min_dim]
            t_masked = t_masked[:, :min_dim]

        sim = F.cosine_similarity(s_masked, t_masked, dim=-1)
        total_sim += sim.sum().item()
        count += sim.numel()

    return (total_sim / count) if count > 0 else 0.0
