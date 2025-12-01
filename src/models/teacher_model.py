import torch
import torch.nn as nn
from transformers import XLMRobertaForMaskedLM


class TeacherModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        
        self.device = device
        self.model = XLMRobertaForMaskedLM.from_pretrained(model_path).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, input_ids, attention_mask=None, return_logits=True):
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                output_hidden_states=True,
            )
            
        if return_logits:
            return outputs.logits  # [batch_size, seq_len, vocab_size]
        else:
            return outputs
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)