import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from torch.utils.data import Subset, DataLoader

from src.evals.NTP_eval import (
    compute_masked_token_accuracy,
    compare_student_teacher_masked_token_agreement
)
from src.data.data import prepare_datasets


def main():
    print("=" * 60)
    print("TESTING MASKED-TOKEN / NTP-STYLE EVALUATIONS")
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

    # Load student (adjust subfolder/token if not needed for your repo)
    student_cfg = AutoConfig.from_pretrained(student_name, subfolder="model")
    student = AutoModelForMaskedLM.from_pretrained(
        student_name,
        subfolder="model",
        config=student_cfg,
        token=True,  # remove if not required
    ).to(device)

    # Ensure embeddings match tokenizer vocab
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

    # Optional: speed up with a subset for smoke tests
    # subset_size = 128
    # eval_loader = DataLoader(
    #     Subset(eval_loader.dataset, list(range(min(subset_size, len(eval_loader.dataset))))),
    #     batch_size=12,
    #     shuffle=False,
    # )

    # 3) Masked-token accuracy (teacher)
    print("\n[3] Evaluating masked-token accuracy (teacher)...")
    try:
        teacher_acc = compute_masked_token_accuracy(teacher, tokenizer, eval_loader, device)
        print(f"Teacher masked-token accuracy: {teacher_acc:.4f}")
        print("Masked-token accuracy evaluation for teacher completed successfully.")
    except Exception:
        print("Masked-token accuracy evaluation for teacher failed.")
        import traceback
        print(traceback.format_exc())

    # 4) Masked-token accuracy (student)
    print("\n[4] Evaluating masked-token accuracy (student)...")
    try:
        student_acc = compute_masked_token_accuracy(student, tokenizer, eval_loader, device)
        print(f"Student masked-token accuracy: {student_acc:.4f}")
        print("Masked-token accuracy evaluation for student completed successfully.")
    except Exception:
        print("Masked-token accuracy evaluation for student failed.")
        import traceback
        print(traceback.format_exc())

    # 5) Student–Teacher agreement on masked tokens
    print("\n[5] Evaluating student–teacher agreement on masked tokens...")
    try:
        agreement = compare_student_teacher_masked_token_agreement(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"Student–Teacher agreement: {agreement['agreement']:.4f}")
        print(f"Total masked positions compared: {agreement['total']}")
        print("Student–teacher masked-token agreement evaluation completed successfully.")
    except Exception:
        print("Student–teacher masked-token agreement evaluation failed.")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
