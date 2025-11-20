"""Evaluation utilities"""

import torch
from tqdm import tqdm
from transformers import pipeline
import random
from torch.utils.data import DataLoader, Subset


def evaluate_model(model, eval_loader, device):
    """Calculate perplexity on evaluation set"""
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Count non-masked tokens
            num_tokens = (labels != -100).sum().item()
            
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


def test_predictions(model, tokenizer, device):
    """Test model on example sentences"""
    
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    
    test_sentences = [
        "मैं [MASK] जा रहा हूं",
        "यह एक [MASK] किताब है",
        "भारत की [MASK] दिल्ली है",
        "मुझे [MASK] पसंद है",
        "वह [MASK] खाता है"
    ]
    
    print()
    for sentence in test_sentences:
        print(f"Input: {sentence}")
        predictions = fill_mask(sentence, top_k=3)
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred['token_str']:15s} (score: {pred['score']:.4f})")
        print()

