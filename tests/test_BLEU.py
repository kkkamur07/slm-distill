import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from src.evals.BLEU_evals import compute_bleu_ground_truth, compute_bleu_student_teacher
from src.data.data import prepare_datasets

def main():
    print("="*60)
    print("TESTING BLEU SCORE EVALUATIONS")
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
        token=True,  # if repo is gated/private
    ).to(device)

    # Just to be safe
    student.resize_token_embeddings(len(tokenizer))


    # print(f"\n[1] Loading models on {device}...")
    # tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast = True)
    # teacher = AutoModelForMaskedLM.from_pretrained(teacher_name).to(device)
    # cfg = AutoConfig.from_pretrained(teacher_name)
    # student = AutoModelForMaskedLM.from_pretrained(
    #     student_name,
    #     config=cfg,
    #     # token=True,
    # ).to(device)

    
    # Load and prepare datasets - returns (train_loader, eval_loader)
    print("\n[2] Loading and preparing dataset...")
    train_loader, eval_loader = prepare_datasets(
        data_path='data/hin/data-99.parquet',
        tokenizer=tokenizer,
        batch_size=12,
        max_length=256,
        train_split=0.995
    )
    #eval_loader.dataset = eval_loader.dataset.select(range(100))
    # Test BLEU calculations
    print("\n[3] Testing BLEU calculations teacher...")
    try:
        # Use eval_loader for BLEU calculation
        ground_truth_bleu_teacher = compute_bleu_ground_truth(
            teacher, tokenizer, eval_loader, device
        )
        print(f"Ground truth BLEU score: {ground_truth_bleu_teacher:.4f}")
        print("BLEU evaluation test for teacher passed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        print(traceback.format_exc())

    print("\n[4] Testing BLEU calculations student...")
    try:
        # Use eval_loader for BLEU calculation
        ground_truth_bleu_student = compute_bleu_ground_truth(
            student, tokenizer, eval_loader, device
        )
        print(f"Ground truth BLEU score: {ground_truth_bleu_student:.4f}")
        print("BLEU evaluation test for student passed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        print(traceback.format_exc())
    
    print("\n[5] Testing BLEU calculations student to teacher...")
    try:
        # Use eval_loader for BLEU calculation
        student_teacher_bleu = compute_bleu_student_teacher(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"student-teacher similarity BLEU score: {student_teacher_bleu:.4f}")
        print("BLEU evaluation test for student-teacher similarity passed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        # Print full traceback for debugging
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()



    #student = AutoModelForMaskedLM.from_pretrained(student_name).to(device)
    #eval_dataset = dataset["test"]  

    # Tokenization function
    # def tokenize_function(examples):
    #     return tokenizer(
    #         examples["text"],
    #         truncation=True,
    #         padding="max_length",
    #         max_length=128
    #     )

    # # Tokenize the dataset
    # print("\n[2] Tokenizing dataset...")
    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # eval_loader = DataLoader(tokenized_dataset, batch_size=8)

    # Test BLEU calculations
    # print("\n[3] Testing BLEU calculations...")
    # try:
    #     ground_truth_bleu = compute_bleu_ground_truth(
    #         teacher, tokenizer, dataset, device
    #     )
    #     print(f"Ground truth BLEU score: {ground_truth_bleu:.4f}")
        
        # teacher_bleu = compute_bleu_student_teacher(
        #     student, teacher, tokenizer, eval_loader, device
        # )
        # print(f"Teacher-student BLEU score: {teacher_bleu:.4f}")