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
    student_subfolder: str | None = "model",
    num_epochs: int = 5,
    batch_size: int = 16,
    base_learning_rate: float = 2e-5,
    lr_grid: list[float] | None = None,
    max_length: int = 128,
    weight_decay: float = 0.01,
    early_stopping_patience: int | None = 2,
    min_delta: float = 0.0,
    device: str | None = None,
):
    """
    Fine-tune teacher and student on Hindi XNLI and evaluate on test.

    - Teacher: trained once with base_learning_rate.
    - Student: LR tuned over lr_grid (if provided), using dev accuracy.
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

    # ---------- TEACHER: single LR ----------
    teacher = create_nli_classifier(
        teacher_model_name,
        num_labels=num_labels,
        dropout=0.0,
    ).to(device)

    print("\n[NLI] Fine-tuning TEACHER...")
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
        learning_rate=base_learning_rate,
        max_length=max_length,
        eval_on_dev=True,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )
    teacher = teacher_res["model"]

    # ---------- STUDENT: LR tuning ----------
    if lr_grid is None:
        lr_grid = [base_learning_rate]

    best_student_acc = -1.0
    best_student_model = None
    best_lr = None

    for lr in lr_grid:
        print(f"\n[NLI] Fine-tuning STUDENT with lr={lr:.1e}...")
        student = create_nli_classifier(
            student_model_name,
            num_labels=num_labels,
            dropout=0.0,
            subfolder=student_subfolder,
        ).to(device)

        res = train_nli_model(
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
            learning_rate=lr,
            max_length=max_length,
            eval_on_dev=True,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
        )
        student = res["model"]
        last_dev_metrics = res["history"][-1]["dev_metrics"]
        dev_acc = last_dev_metrics["accuracy"] if last_dev_metrics is not None else 0.0

        print(f"[NLI] STUDENT lr={lr:.1e} dev accuracy={dev_acc:.4f}")

        if dev_acc > best_student_acc:
            best_student_acc = dev_acc
            best_student_model = student
            best_lr = lr

    if best_student_model is None:
        raise RuntimeError("Student training failed for all learning rates.")
    student = best_student_model
    print(f"\n[NLI] Best STUDENT lr={best_lr:.1e} with dev acc={best_student_acc:.4f}")

    # ---------- Evaluate both on test set ----------
    print("\n[NLI] Evaluating TEACHER on test...")
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Evaluating STUDENT on test...")
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Results:")
    print("  Teacher:", teacher_metrics)
    print("  Student:", student_metrics)

    return {
        "teacher": {"metrics": teacher_metrics},
        "student": {"metrics": student_metrics, "best_lr": best_lr},
    }
