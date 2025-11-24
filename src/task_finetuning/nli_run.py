# src/task_finetuning/nli_run.py

import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import create_nli_classifier, compute_nli_accuracy
from src.task_finetuning.nli_data import load_nli_split
from src.task_finetuning.nli_train import train_nli_model


def run_nli(
    train_path: str = "data/hin/xnli_hi_train.json",
    dev_path: str = "data/hin/xnli_hi_dev.json",
    test_path: str = "data/hin/xnli_hi_test.json",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str | None = "model",  # None if not needed
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str | None = None,
):
    """
    Fine-tune teacher and student on Hindi XNLI and evaluate on test.

    Returns:
        {
            "teacher": {"metrics": {...}},
            "student": {"metrics": {...}},
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NLI] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Load splits with a shared label mapping
    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(train_path)
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(dev_path, raw_to_id=raw_to_id)
    test_prem, test_hyp, test_labels, _ = load_nli_split(test_path, raw_to_id=raw_to_id)
    num_labels = len(raw_to_id)

    print(
        f"[NLI] train={len(train_prem)}, dev={len(dev_prem)}, "
        f"test={len(test_prem)}, num_labels={num_labels}"
    )

    # Models (starting from HF checkpoints / KD checkpoints)
    teacher = create_nli_classifier(
        teacher_model_name,
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)

    student = create_nli_classifier(
        student_model_name,
        num_labels=num_labels,
        dropout=0.1,
        subfolder=student_subfolder,
    ).to(device)

    # Fine-tune teacher
    print("[NLI] Fine-tuning TEACHER...")
    teacher_res = train_nli_model(
        model=teacher,
        tokenizer=tokenizer,
        train_premises=train_prem,
        train_hypotheses=train_hyp,
        train_labels=train_labels,
        dev_premises=dev_prem,
        dev_hypotheses=dev_hyp,
        dev_labels=dev_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    teacher = teacher_res["model"]

    # Fine-tune student
    print("[NLI] Fine-tuning STUDENT...")
    student_res = train_nli_model(
        model=student,
        tokenizer=tokenizer,
        train_premises=train_prem,
        train_hypotheses=train_hyp,
        train_labels=train_labels,
        dev_premises=dev_prem,
        dev_hypotheses=dev_hyp,
        dev_labels=dev_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    student = student_res["model"]

    # Evaluate on test
    print("[NLI] Evaluating TEACHER on test...")
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NLI] TEACHER metrics:", teacher_metrics)

    print("[NLI] Evaluating STUDENT on test...")
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NLI] STUDENT metrics:", student_metrics)

    return {
        "teacher": {"metrics": teacher_metrics},
        "student": {"metrics": student_metrics},
    }


if __name__ == "__main__":
    run_nli()
