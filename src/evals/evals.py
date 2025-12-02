import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, eval_loader, device):

    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get logits (no labels passed to model)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_logits=True
            )
            
            mask = (labels != -100)  # Shape: (batch, seq_len)
            num_tokens = mask.sum().item()
            
            logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq_len, vocab_size)
            labels_flat = labels.view(-1)  # (batch*seq_len,)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += num_tokens
            
            predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
            correct = (predictions == labels) & mask  # (batch, seq_len)
            total_correct += correct.sum().item()
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'perplexity': perplexity,
        'masked_accuracy': accuracy,
    }