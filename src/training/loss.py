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
 
    # Soft targets (knowledge distillation)
    soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    loss_kd = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets (ground truth)
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Combined loss
    total_loss = alpha * loss_kd + (1 - alpha) * loss_ce
    
    return total_loss, loss_kd.item(), loss_ce.item()