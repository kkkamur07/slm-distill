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
        max_position_embeddings: int = 514,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        device: torch.device = torch.device("cpu"),
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # Validate
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        
        self.device = device
        
        # XLM-RoBERTa Config
        config = XLMRobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=1,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=1e-5,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        
        self.config = config
        self.model = XLMRobertaForMaskedLM(config).to(self.device)
        
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_logits=True
    ):
        input_ids = input_ids.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        if labels is not None:
            labels = labels.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
        )
        
        if return_logits:
            return outputs.logits  # [batch_size, seq_len, vocab_size]
        else:
            return outputs
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self):
        return self.config  
    
