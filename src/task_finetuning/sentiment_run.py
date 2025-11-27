import torch
from transformers import AutoTokenizer

from src.evals.sentiment_eval import (
    create_sentiment_classifier,
    compute_sentiment_accuracy,
)
from src.task_finetuning.sentiment_data import load_sentiment_csv
from src.task_finetuning.sentiment_train import train_sentiment_model


def run_sentiment(
    train_path: str = "data/hin/sentiment_hi_train.csv",
    val_path: str = "data/hin/sentiment_hi_val.csv",
    test_path: str = "data/hin/sentiment_hi_test.csv",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str | None = "model",  # None if not needed
    num_labels: int = 3,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,  # kept for backwards compat, not used if lr_grid is given
    max_length: int = 128,
    device: str | None = None,
    weight_decay: float = 0.01,
    early_stopping_patience: int | None = 2,
    min_delta: float = 0.0,
    lr_grid: list[float] | None = None,
    dropout: float = 0.0,
):
    """
    Fine-tune teacher and student on Hindi sentiment and evaluate on test.

    Adds:
      - weight decay
      - early stopping on dev accuracy
      - LR grid search for BOTH teacher and student (dev accuracy as criterion).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sentiment] Using device: {device}")

    if lr_grid is None:
        lr_grid = [2e-5, 5e-5, 1e-4]

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Data
    train_texts, train_labels = load_sentiment_csv(train_path)
    val_texts, val_labels = load_sentiment_csv(val_path)
    test_texts, test_labels = load_sentiment_csv(test_path)

    print(
        f"[Sentiment] train={len(train_texts)}, val={len(val_texts)}, "
        f"test={len(test_texts)}, num_labels={num_labels}"
    )

    # helper: LR grid search for one model
    def grid_search_model(base_model_name: str, subfolder: str | None = None):
        best_acc = -1.0
        best_model = None
        best_lr = None

        for lr in lr_grid:
            print(f"\n[Sentiment] Fine-tuning '{base_model_name}' with lr={lr:.1e}...")
            model = create_sentiment_classifier(
                base_model_name=base_model_name,
                num_labels=num_labels,
                dropout=dropout,
                subfolder=subfolder,
            ).to(device)

            res = train_sentiment_model(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                train_labels=train_labels,
                dev_texts=val_texts,
                dev_labels=val_labels,
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
            model = res["model"]
            last_dev = res["history"][-1]["dev_metrics"]
            dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
            print(f"[Sentiment] {base_model_name}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr

        if best_model is None:
            raise RuntimeError(f"[Sentiment] Grid search failed for {base_model_name}")

        print(
            f"[Sentiment] Best lr for {base_model_name}: {best_lr:.1e} "
            f"(dev acc={best_acc:.4f})"
        )
        return best_model, best_lr

    # Teacher grid search
    teacher, best_lr_teacher = grid_search_model(
        base_model_name=teacher_model_name,
        subfolder=None,
    )

    # Student grid search
    student, best_lr_student = grid_search_model(
        base_model_name=student_model_name,
        subfolder=student_subfolder,
    )

    # Final test evaluation
    print("\n[Sentiment] Evaluating TEACHER on test...")
    teacher_metrics = compute_sentiment_accuracy(
        teacher,
        tokenizer,
        test_texts,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[Sentiment] TEACHER metrics:", teacher_metrics)

    print("\n[Sentiment] Evaluating STUDENT on test...")
    student_metrics = compute_sentiment_accuracy(
        student,
        tokenizer,
        test_texts,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[Sentiment] STUDENT metrics:", student_metrics)

    return {
        "teacher": {"metrics": teacher_metrics, "best_lr": best_lr_teacher},
        "student": {"metrics": student_metrics, "best_lr": best_lr_student},
    }


if __name__ == "__main__":
    run_sentiment()