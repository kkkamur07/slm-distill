"""Masked token accuracy evaluation"""

import torch
from tqdm import tqdm

def evaluate(model, eval_loader, device):
    
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass, in teacher we don't need the labels as we are not calculating the loss here, what to do here ? 
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_logits=True
            )
            
            # Loss and perplexity
            mask = (labels != -100)
            num_tokens = mask.sum().item()
            
            logits_flat = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = torch.nn.functional.cross_entropy(
                logits_flat,
                labels,
                ignore_index=-100,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # Accuracy
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {
        'perplexity': perplexity,
        'masked_accuracy': accuracy,
    }