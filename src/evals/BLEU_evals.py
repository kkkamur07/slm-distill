import torch
from tqdm import tqdm
from transformers import pipeline
from sacrebleu.metrics import BLEU
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
        predictions = torch.argmax(torch.softmax(output.logits, dim=-1), dim=-1) # softmax by itself throws an error, argmax works (alone or as a wrapper)
        
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

def compute_bleu_ground_truth(model, tokenizer, eval_loader, device, max_new_tokens=128):
    """Evaluate model output vs ground-truth labels"""
    model.eval()  # Ensure model is in eval mode
    
    # Generate outputs
    hyps = _generate_mlm(model, tokenizer, eval_loader, device, max_new_tokens)
    
    # Get references
    refs = []
    with torch.no_grad():  # Add no_grad context
        for batch in tqdm(eval_loader, desc="Preparing references"):
            labels = batch["labels"].to(device)
            refs.extend(_decode_labels(labels, tokenizer))
    
    # Calculate BLEU score
    bleu = BLEU(lowercase=True)
    score = bleu.corpus_score(hyps, [refs]).score
    
    return score / 100.0  # Normalize to 0-1 range

def compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device, max_new_tokens=128):
    """Compare student output vs teacher output using BLEU"""
    # Generate from both models
    teacher_refs = _generate_mlm(teacher, tokenizer, eval_loader, device, max_new_tokens)
    student_hyps = _generate_mlm(student, tokenizer, eval_loader, device, max_new_tokens)
    
    # Calculate BLEU score : Here is the problem. 
    bleu = BLEU(lowercase=True)
    score = bleu.corpus_score(student_hyps, [teacher_refs]).score
    
    return score / 100.0  # Normalize to 0-1 range


# # Get BLEU scores this is simply for me to keep track on how to implement them, I'll delete all below soon
# ground_truth_bleu = compute_bleu_ground_truth(student, tokenizer, eval_loader, device)
# teacher_bleu = compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device)

# print(f"Ground Truth BLEU: {ground_truth_bleu:.4f}")
# print(f"Teacher BLEU: {teacher_bleu:.4f}")

"""
| **Purpose**                         | **Comparison**                   | **Interpretation**                                                                                                                |
| ----------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Evaluate distillation quality**   | **Student vs Teacher**           | How close the student’s *generated outputs* are to the teacher’s outputs. Measures imitation fidelity.                            |
| **Evaluate real-world performance** | **Student vs Human (reference)** | How well the student performs the actual *task* (e.g. translation, summarization). Measures true quality, independent of teacher. |
| **Check teacher quality baseline**  | **Teacher vs Human (reference)** | Confirms the teacher’s outputs are indeed high-quality, so it’s worth mimicking.                                                  |

Next tests : Next token prediction, perplexity and BLEU + CHRF, semantic distance, BERT similarity, Logit Alignment (or the relevant Energy)
task specific MMLU (student - teacher comparisons) Indic glue (11 indic languages) or Indic MMLU- Pro or Indic gen benches

order: BLEU + CHRF, Next token prediction, MMLU
"""