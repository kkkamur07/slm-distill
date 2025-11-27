import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import (
    create_nli_classifier,
    compute_nli_accuracy,
)
from src.task_finetuning.nli_data import load_nli_split


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # Test split (only)
    test_path = "data/hin/xnli_hi_test.json"
    test_prem, test_hyp, test_labels, raw_to_id = load_nli_split(test_path)
    num_labels = len(raw_to_id)

    teacher = create_nli_classifier(
        teacher_name,
        num_labels=num_labels,
        dropout=0.0,
    ).to(device)

    student = create_nli_classifier(
        student_name,
        num_labels=num_labels,
        dropout=0.0,
        subfolder="model",
    ).to(device)

    print("\n[1] Evaluating TEACHER on NLI test set...")
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )
    print("Teacher:", teacher_metrics)

    print("\n[2] Evaluating STUDENT on NLI test set...")
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )
    print("Student:", student_metrics)

    print("\nNLI evaluations completed.\n")


if __name__ == "__main__":
    main()