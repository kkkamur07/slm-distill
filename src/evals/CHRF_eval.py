import torch
from tqdm import tqdm
from transformers import pipeline
from sacrebleu.metrics import CHRF
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def _generate_mlm(model, tokenizer, loader, device, max_new_tokens=128):
    """Generate text using masked language model"""
    model.eval()
    outputs = []
    
    for batch in tqdm(loader, desc="Generating (MLM)"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Get predictions
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(torch.softmax(output.logits, dim=-1), dim=-1)
        
        # Decode predictions
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        outputs.extend([text.strip() for text in decoded])
    
    return outputs

def _decode_labels(labels_tensor, tokenizer):
    """Decode label IDs to text"""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lbls = labels_tensor.clone()
    lbls[lbls == -100] = pad_id
    return [t.strip() for t in tokenizer.batch_decode(lbls, skip_special_tokens=True)]

def compute_chrf_ground_truth(model, tokenizer, eval_loader, device, max_new_tokens=128):
    """Evaluate model output vs ground-truth labels using chrF"""
    model.eval()
    
    # Generate outputs
    hyps = _generate_mlm(model, tokenizer, eval_loader, device, max_new_tokens)
    
    # Get references
    refs = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Preparing references"):
            labels = batch["labels"].to(device)
            refs.extend(_decode_labels(labels, tokenizer))
    
    # Calculate chrF score
    chrf = CHRF(word_order=2)  # Using word_order=2 for chrF++
    score = chrf.corpus_score(hyps, [refs]).score
    
    return score / 100.0  # Normalize to 0-1 range

def compute_chrf_student_teacher(student, teacher, tokenizer, eval_loader, device, max_new_tokens=128):
    """Compare student output vs teacher output using chrF"""
    # Generate from both models
    teacher_refs = _generate_mlm(teacher, tokenizer, eval_loader, device, max_new_tokens)
    student_hyps = _generate_mlm(student, tokenizer, eval_loader, device, max_new_tokens)
    
    # Calculate chrF score
    chrf = CHRF(word_order=2)  # Using word_order=2 for chrF++
    score = chrf.corpus_score(student_hyps, [teacher_refs]).score
    
    return score / 100.0  # Normalize to 0-1 range



