import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import (
    create_nli_classifier,
    compute_nli_accuracy,
)
from src.task_finetuning.nli_data import load_nli_split
from src.task_finetuning.nli_train import train_nli_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_path = "data/hin/xnli_hi_train.json"
    dev_path = "data/hin/xnli_hi_dev.json"
    test_path = "data/hin/xnli_hi_test.json"

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # Load splits with a shared label mapping
    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(train_path)
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(
        dev_path, raw_to_id=raw_to_id
    )
    test_prem, test_hyp, test_labels, _ = load_nli_split(
        test_path, raw_to_id=raw_to_id
    )

    num_labels = len(raw_to_id)
    print(
        f"Loaded NLI splits: "
        f"train={len(train_prem)}, dev={len(dev_prem)}, test={len(test_prem)}, "
        f"num_labels={num_labels}"
    )

    # Create models
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

    # Fine-tune teacher
    print("\n[1] Fine-tuning TEACHER on NLI train/dev...")
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
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    teacher = teacher_res["model"]

    # Fine-tune student
    print("\n[2] Fine-tuning STUDENT on NLI train/dev...")
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
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    student = student_res["model"]

    # Evaluate on test set
    print("\n[3] Evaluating TEACHER on NLI test set...")
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )

    print("\n[4] Evaluating STUDENT on NLI test set...")
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )

    print(
        f"\nFine-tuned teacher — acc: {teacher_metrics['accuracy']:.4f}, "
        f"macro-F1: {teacher_metrics['macro_f1']:.4f}"
    )
    print(
        f"Fine-tuned student — acc: {student_metrics['accuracy']:.4f}, "
        f"macro-F1: {student_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
