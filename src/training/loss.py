"""Distillation loss functions"""

import torch
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
    
    # Mask for valid tokens only (where label != -100)
    mask = labels_flat != -100  # (batch*seq,) boolean mask
    
    # KD Loss - only compute on masked tokens
    if mask.sum() > 0:
        # Select only masked positions
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
        # No masked tokens in this batch (shouldn't happen)
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