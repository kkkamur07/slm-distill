"""Student model definition for knowledge distillation.

This module defines the StudentModel class, a compressed version of XLM-RoBERTa
designed for monolingual tasks. The model:
- Uses fewer layers and smaller hidden dimensions than the teacher
- Supports gradient checkpointing for memory efficiency
- Compatible with masked language modeling
- Typically 8-20x smaller than the teacher model

The architecture is configurable via parameters to allow experimentation with
different compression ratios and model sizes.

Example:
    >>> from src.models.student_model import StudentModel
    >>> import torch
    >>> 
    >>> # Create a small student model (33M parameters)
    >>> student = StudentModel(
    ...     vocab_size=250002,
    ...     hidden_size=256,
    ...     num_hidden_layers=6,
    ...     num_attention_heads=8,
    ...     intermediate_size=1024,
    ...     device=torch.device("cuda")
    ... )
    >>> 
    >>> # Forward pass
    >>> input_ids = torch.randint(0, 250002, (2, 128)).cuda()
    >>> logits = student(input_ids=input_ids, return_logits=True)
    >>> print(logits.shape)  # torch.Size([2, 128, 250002])
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaForMaskedLM, XLMRobertaConfig


class StudentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 250002,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        device: torch.device = torch.device("cpu"),
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.device = device
        
        config = XLMRobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        
        self.config = config
        self.model = XLMRobertaForMaskedLM(config).to(self.device)
        
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                labels=None, return_logits=True):

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.logits if return_logits else outputs
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self):
        return self.config
