import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from torch.utils.data import Subset, DataLoader

from src.evals.BLEU_evals import compute_bleu_ground_truth, compute_bleu_student_teacher
from src.data.data import prepare_datasets


def main():
    print("=" * 60)
    print("TESTING BLEU SCORE EVALUATIONS")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1) Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_name).to(device)

    # Load student model
    student_cfg = AutoConfig.from_pretrained(student_name, subfolder="model")
    student = AutoModelForMaskedLM.from_pretrained(
        student_name,
        subfolder="model",
        config=student_cfg,
        token=True,  # remove if not needed for your repository
    ).to(device)

    # Ensure embedding size matches tokenizer
    student.resize_token_embeddings(len(tokenizer))

    # 2) Load dataset and dataloaders
    print("\n[2] Loading and preparing dataset...")
    train_loader, eval_loader = prepare_datasets(
        data_path="data/hin/data-99.parquet",
        tokenizer=tokenizer,
        batch_size=12,
        max_length=256,
        train_split=0.995,
    )

    # Optional: restrict evaluation size for quick tests
    # subset_size = 128
    # eval_loader = DataLoader(
    #     Subset(eval_loader.dataset, list(range(min(subset_size, len(eval_loader.dataset))))),
    #     batch_size=12,
    #     shuffle=False,
    # )

    # 3) BLEU evaluation for teacher vs ground truth
    print("\n[3] Evaluating BLEU vs ground truth (teacher)...")
    try:
        bleu_teacher = compute_bleu_ground_truth(teacher, tokenizer, eval_loader, device)
        print(f"Ground-truth BLEU (teacher): {bleu_teacher:.4f}")
        print("BLEU evaluation for teacher completed successfully.")
    except Exception as e:
        print("BLEU evaluation for teacher failed.")
        import traceback
        print(traceback.format_exc())

    # 4) BLEU evaluation for student vs ground truth
    print("\n[4] Evaluating BLEU vs ground truth (student)...")
    try:
        bleu_student = compute_bleu_ground_truth(student, tokenizer, eval_loader, device)
        print(f"Ground-truth BLEU (student): {bleu_student:.4f}")
        print("BLEU evaluation for student completed successfully.")
    except Exception as e:
        print("BLEU evaluation for student failed.")
        import traceback
        print(traceback.format_exc())

    # 5) BLEU evaluation for student vs teacher outputs
    print("\n[5] Evaluating BLEU similarity between student and teacher...")
    try:
        bleu_student_teacher = compute_bleu_student_teacher(student, teacher, tokenizer, eval_loader, device)
        print(f"Student–Teacher BLEU similarity: {bleu_student_teacher:.4f}")
        print("BLEU evaluation for student–teacher similarity completed successfully.")
    except Exception as e:
        print("BLEU evaluation for student–teacher similarity failed.")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
