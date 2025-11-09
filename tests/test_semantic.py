import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from torch.utils.data import Subset, DataLoader

from src.evals.BERTsim_eval import (
    compute_bertscore_ground_truth,
    compute_bertscore_student_teacher,
)
from src.data.data import prepare_datasets


def main():
    print("=" * 60)
    print("TESTING SEMANTIC EVALUATIONS (BERTScore)")
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

    # Load student; keep subfolder/token only if your repo needs them
    student_cfg = AutoConfig.from_pretrained(student_name, subfolder="model")
    student = AutoModelForMaskedLM.from_pretrained(
        student_name,
        subfolder="model",
        config=student_cfg,
        token=True,  # remove if not required
    ).to(device)

    # Ensure embedding size matches tokenizer vocab
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

    # 3) BERTScore evaluations
    # Note: for rescale_with_baseline=True, lang must be specified. Using Hindi ("hi").
    print("\n[3] Evaluating BERTScore vs ground truth (teacher/student) and student–teacher similarity...")
    try:
        teacher_bs = compute_bertscore_ground_truth(
            teacher, tokenizer, eval_loader, device, lang="hi", rescale=True
        )
        print(f"Ground-truth BERTScore (teacher): {teacher_bs:.4f}")

        student_bs = compute_bertscore_ground_truth(
            student, tokenizer, eval_loader, device, lang="hi", rescale=True
        )
        print(f"Ground-truth BERTScore (student): {student_bs:.4f}")

        st_bs = compute_bertscore_student_teacher(
            student, teacher, tokenizer, eval_loader, device, lang="hi", rescale=True
        )
        print(f"Student–Teacher BERTScore similarity: {st_bs:.4f}")
        print("BERTScore evaluations completed successfully.")
    except Exception:
        print("BERTScore evaluations failed.")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
