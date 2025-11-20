import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from src.evals.MTP_eval import compute_masked_token_accuracy, compare_student_teacher_masked_token_agreement
from src.data.data import prepare_datasets

def main():
    print("="*60)
    print("TESTING MASKED TOKEN PREDICTION")
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
    
    assert student.get_input_embeddings().num_embeddings == len(tokenizer)      
    # Load and prepare datasets
    print("\n[2] Loading and preparing dataset...")
    train_loader, eval_loader = prepare_datasets(
        data_path='data/hin/data-99.parquet',
        tokenizer=tokenizer,
        batch_size=12,
        max_length=256,
        train_split=0.995
    )
    
    print("\n[3] Testing MTP calculations for teacher...")
    try:
        teacher_accuracy = compute_masked_token_accuracy(
            teacher, tokenizer, eval_loader, device
        )
        print(f"✓ Teacher MTP accuracy: {teacher_accuracy:.4f}")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())

    print("\n[4] Testing MTP calculations for student...")
    try:
        student_accuracy = compute_masked_token_accuracy(
            student, tokenizer, eval_loader, device
        )
        print(f"✓ Student MTP accuracy: {student_accuracy:.4f}")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("\n[5] Testing student-teacher masked token agreement...")
    try:
        agreement_results = compare_student_teacher_masked_token_agreement(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"✓ Student-Teacher agreement: {agreement_results['agreement']:.4f}")
        print(f"  (evaluated on {agreement_results['total']} masked tokens)")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()