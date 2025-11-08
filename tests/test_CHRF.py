import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from src.evals.CHRF import compute_chrf_ground_truth, compute_chrf_student_teacher
from src.data.data import prepare_datasets

def main():
    print("="*60)
    print("TESTING CHRF SCORE EVALUATIONS")
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
    
    print("\n[3] Testing CHRF calculations teacher...")
    try:
        ground_truth_chrf_teacher = compute_chrf_ground_truth(
            teacher, tokenizer, eval_loader, device
        )
        print(f"Ground truth CHRF score: {ground_truth_chrf_teacher:.4f}")
        print("CHRF evaluation test for teacher passed!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())

    print("\n[4] Testing CHRF calculations student...")
    try:
        ground_truth_chrf_student = compute_chrf_ground_truth(
            student, tokenizer, eval_loader, device
        )
        print(f"Ground truth CHRF score: {ground_truth_chrf_student:.4f}")
        print("CHRF evaluation test for student passed!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("\n[5] Testing CHRF calculations student to teacher...")
    try:
        student_teacher_chrf = compute_chrf_student_teacher(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"student-teacher similarity CHRF score: {student_teacher_chrf:.4f}")
        print("CHRF evaluation test for student-teacher similarity passed!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()