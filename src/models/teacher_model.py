import torch
import torch.nn as nn
from transformers import XLMRobertaForMaskedLM


class TeacherModel(nn.Module):
    def __init__(self, model_path: str, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.model = XLMRobertaForMaskedLM.from_pretrained(model_path).to(device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                labels=None, return_logits=True):
        """Just pass through to the underlying model"""
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        with torch.no_grad():
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
    
    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
