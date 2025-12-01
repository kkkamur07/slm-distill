"""Masked token accuracy evaluation"""

import torch
from tqdm import tqdm

def evaluate(model, eval_loader, device):
    
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Loss and perplexity
            mask = (labels != -100)
            num_tokens = mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Accuracy
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_masked += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return {
        # 'eval_loss': avg_loss,
        'perplexity': perplexity,
        'masked_accuracy': accuracy,
        # 'total_masked_tokens': total_masked,
        # 'correct_predictions': total_correct
    }
