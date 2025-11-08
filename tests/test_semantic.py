import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from src.evals.BERTsim_COMET_eval import (
    compute_bertscore_ground_truth, 
    compute_bertscore_student_teacher,
    compute_comet_ground_truth,
    compute_comet_student_teacher
)
from src.data.data import prepare_datasets

def main():
    print("="*60)
    print("TESTING SEMANTIC EVALUATIONS (BERTScore & COMET)")
    print("="*60)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"
    
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_name).to(device)
    
    student_cfg = AutoConfig.from_pretrained(student_name, subfolder="model")
    student = AutoModelForMaskedLM.from_pretrained(
        student_name,
        subfolder="model",
        config=student_cfg,
        token=True,
    ).to(device)
    
    # Just to be safe
    student.resize_token_embeddings(len(tokenizer))
    
    # Load and prepare datasets
    print("\n[2] Loading and preparing dataset...")
    train_loader, eval_loader = prepare_datasets(
        data_path='data/hin/data-99.parquet',
        tokenizer=tokenizer,
        batch_size=12,
        max_length=256,
        train_split=0.995
    )
    
    # BERTScore evaluations
    print("\n[3] Testing BERTScore calculations...")
    try:
        teacher_bertscore = compute_bertscore_ground_truth(
            teacher, tokenizer, eval_loader, device, lang="hi"
        )
        print(f"Teacher BERTScore: {teacher_bertscore:.4f}")
        
        student_bertscore = compute_bertscore_ground_truth(
            student, tokenizer, eval_loader, device, lang="hi"
        )
        print(f"Student BERTScore: {student_bertscore:.4f}")
        
        student_teacher_bertscore = compute_bertscore_student_teacher(
            student, teacher, tokenizer, eval_loader, device, lang="hi"
        )
        print(f"Student-Teacher BERTScore: {student_teacher_bertscore:.4f}")
        print("BERTScore evaluations passed!")
    except Exception as e:
        print(f"BERTScore test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # COMET evaluations
    print("\n[4] Testing COMET calculations...")
    try:
        teacher_comet = compute_comet_ground_truth(
            teacher, tokenizer, eval_loader, device
        )
        print(f"Teacher COMET score: {teacher_comet:.4f}")
        
        student_comet = compute_comet_ground_truth(
            student, tokenizer, eval_loader, device
        )
        print(f"Student COMET score: {student_comet:.4f}")
        
        student_teacher_comet = compute_comet_student_teacher(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"Student-Teacher COMET score: {student_teacher_comet:.4f}")
        print("COMET evaluations passed!")
    except Exception as e:
        print(f"COMET test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()