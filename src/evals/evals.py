"""Evaluation utilities"""

import torch
from tqdm import tqdm
from transformers import pipeline
from sacrebleu.metrics import BLEU
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


@torch.no_grad()
def _generate(model, tokenizer, loader, device, max_new_tokens=128):
    """Generate text using the model"""
    model.eval()
    outputs = []
    
    for batch in tqdm(loader, desc="Generating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id
        )
        
        decoded = tokenizer.batch_decode(gen[:, input_ids.size(1):], 
                                       skip_special_tokens=True, 
                                       clean_up_tokenization_spaces=True)
        outputs.extend(t.strip() for t in decoded)
    
    return outputs


def _decode_labels(labels_tensor, tokenizer):
    """Decode label IDs to text"""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lbls = labels_tensor.clone()
    lbls[lbls == -100] = pad_id
    return [t.strip() for t in tokenizer.batch_decode(lbls, skip_special_tokens=True)]


def compute_bleu_ground_truth(model, tokenizer, eval_loader, device, max_new_tokens=128):
    """Evaluate model output vs ground-truth labels"""
    # Generate outputs
    hyps = _generate(model, tokenizer, eval_loader, device, max_new_tokens)
    
    # Get references
    refs = []
    for batch in tqdm(eval_loader, desc="Preparing references"):
        refs.extend(_decode_labels(batch["labels"], tokenizer))
    
    # Calculate BLEU - ground truth
    bleu = BLEU(lowercase=True)
    score = bleu.corpus_score(hyps, [refs]).score * 0.01
    
    return score


def compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device, max_new_tokens=128):
    """Compare student output vs teacher output using BLEU"""
    # Generate from both models
    teacher_refs = _generate(teacher, tokenizer, eval_loader, device, max_new_tokens)
    student_hyps = _generate(student, tokenizer, eval_loader, device, max_new_tokens)
    
    # Calculate BLEU - compared to teacher
    bleu = BLEU(lowercase=True)
    score = bleu.corpus_score(student_hyps, [teacher_refs]).score * 0.01
    
    return score
