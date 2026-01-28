"""Distillation loss functions for knowledge transfer.

This module implements various loss functions for distillation:
- distillation_loss: Combined KL divergence and cross-entropy loss
- compute_score_matching_loss: Score matching based distillation loss

The losses support:
- Temperature scaling for soft targets
- Alpha balancing between distillation and task losses
- Masked token prediction (ignoring padding)
- Projection layers for score matching

These losses enable effective knowledge transfer from teacher to student models.

Example:
    >>> from src.training.loss import distillation_loss
    >>> import torch
    >>> 
    >>> student_logits = torch.randn(2, 128, 250002)  # (batch, seq, vocab)
    >>> teacher_logits = torch.randn(2, 128, 250002)
    >>> labels = torch.randint(0, 250002, (2, 128))
    >>> 
    >>> loss, kd_loss, ce_loss = distillation_loss(
    ...     student_logits=student_logits,
    ...     teacher_logits=teacher_logits,
    ...     labels=labels,
    ...     temperature=2.0,
    ...     alpha=0.5
    ... )
    >>> print(f"Total loss: {loss.item():.4f}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float
) -> Tuple[torch.Tensor, float, float]:
    
    batch_size, seq_len, vocab_size = student_logits.shape
    student_flat = student_logits.view(-1, vocab_size)  # (batch*seq, vocab)
    teacher_flat = teacher_logits.view(-1, vocab_size)  # (batch*seq, vocab)
    labels_flat = labels.view(-1)  # (batch*seq,)
    
    mask = labels_flat != -100  # (batch*seq,) boolean mask
    
    if mask.sum() > 0:
        
        student_masked = student_flat[mask]  # (num_masked, vocab)
        teacher_masked = teacher_flat[mask]  # (num_masked, vocab)
        
        # Apply temperature scaling
        soft_teacher = F.softmax(teacher_masked.detach() / temperature, dim=-1)
        soft_student = F.log_softmax(student_masked / temperature, dim=-1)
        
        # Compute KL divergence
        loss_kd = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)
        
    else:
        loss_kd = torch.tensor(0.0, device=student_logits.device)
    
    # Loss C
    loss_ce = F.cross_entropy(
        student_flat,
        labels_flat,
        ignore_index=-100,
        reduction='mean'
    )
    
    # Combined loss
    total_loss = alpha * loss_kd + (1 - alpha) * loss_ce
    
    return total_loss, loss_kd.item(), loss_ce.item()


"""
Score matching loss : 
We are going to match the gradients of the log probabilities with respect to the embeddings. 
Why embeddings ? 

1. Because they are continuous and differentiable, making gradient computation feasible.
2. They capture rich semantic information about the inputs, which is beneficial for distillation.
3. Tokens are discrete and non-differentiable, making direct gradient computation impossible.

The intuition : 
1. We are going to match the log probability distribution of the teacher and student model. 
    1.1 This indicates that the student model is learning to produce similar output distributions as the teacher model.

This is score matching over the labels. 
"""


def compute_score_matching_loss(
    teacher_model,
    student_model,
    score_projection,
    input_ids,
    attention_mask,
    labels,
    temperature,
    alpha
):
    mask = labels != -100
    
    if mask.sum() == 0:
        device = input_ids.device
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return zero_loss, 0.0, 0.0
    
    valid_count = mask.float().sum()
    
    # Score matching loss won't support flash attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False
    ):
        
        # Compute teacher scores
        teacher_emb = teacher_model.model.get_input_embeddings()(input_ids.to(teacher_model.device))
        teacher_emb_grad = teacher_emb.detach().requires_grad_(True)
        
        teacher_outputs = teacher_model.model(
            inputs_embeds=teacher_emb_grad,
            attention_mask=attention_mask.to(teacher_model.device),
            return_dict=True
        )
        teacher_logits = teacher_outputs.logits
        
        teacher_log_Z = torch.logsumexp(teacher_logits, dim=-1) 
        teacher_objective = (teacher_log_Z * mask.float()).sum() / valid_count # Only calculate for the masked positions
        
        teacher_score = torch.autograd.grad(
            outputs=teacher_objective,
            inputs=teacher_emb_grad,
            retain_graph=False,
            create_graph=False
        )[0].detach() 
        
        teacher_score_projected = score_projection(teacher_score.to(student_model.device)) 
        
        # Compute student scores
        student_emb = student_model.model.get_input_embeddings()(input_ids.to(student_model.device))
        student_emb_grad = student_emb.detach().requires_grad_(True)
        
        student_outputs = student_model.model(
            inputs_embeds=student_emb_grad,
            attention_mask=attention_mask.to(student_model.device),
            return_dict=True
        )
        student_logits = student_outputs.logits
        
        student_log_Z = torch.logsumexp(student_logits, dim=-1)
        student_objective = (student_log_Z * mask.float().to(student_model.device)).sum() / valid_count
        
        student_score = torch.autograd.grad(
            outputs=student_objective,
            inputs=student_emb_grad,
            retain_graph=True,
            create_graph=True  
        )[0] 
   
    
    mask_expanded = mask.unsqueeze(-1).float().to(student_model.device)
    
    # MSE Loss
    score_diff = ((student_score - teacher_score_projected) ** 2) * mask_expanded.to(student_model.device)
    score_loss = score_diff.sum() / (mask_expanded.sum() * student_score.shape[-1])
    score_loss = score_loss * 100000 # Stabilization factor
    
    ## <-- KL Divergence Loss -->
    
    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.detach().view(-1, teacher_logits.size(-1))
    labels_flat = labels.view(-1)
    
    mask_flat_kd = labels_flat != -100
    
    if mask_flat_kd.sum() > 0:
        soft_teacher = F.softmax(teacher_flat[mask_flat_kd] / temperature, dim=-1)
        soft_student = F.log_softmax(student_flat[mask_flat_kd] / temperature, dim=-1)
        
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    else:
        kd_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    
    total_loss = alpha * score_loss + (1 - alpha) * kd_loss
    
    return total_loss, kd_loss.item(), score_loss.item()


