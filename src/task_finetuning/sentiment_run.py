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
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str | None = None,
):
    """
    Fine-tune teacher and student on Hindi sentiment and evaluate on test.

    Returns:
        {
            "teacher": {"metrics": {...}},
            "student": {"metrics": {...}},
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sentiment] Using device: {device}")

    # 1) Tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # 2) Data
    train_texts, train_labels = load_sentiment_csv(train_path)
    val_texts, val_labels = load_sentiment_csv(val_path)
    test_texts, test_labels = load_sentiment_csv(test_path)

    print(
        f"[Sentiment] train={len(train_texts)}, val={len(val_texts)}, "
        f"test={len(test_texts)}, num_labels={num_labels}"
    )

    # 3) Models (starting from KD checkpoints / HF models)
    teacher = create_sentiment_classifier(
        teacher_model_name,
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)

    student = create_sentiment_classifier(
        student_model_name,
        num_labels=num_labels,
        dropout=0.1,
        subfolder=student_subfolder,
    ).to(device)

    # 4) Fine-tune teacher
    print("[Sentiment] Fine-tuning TEACHER...")
    teacher_res = train_sentiment_model(
        model=teacher,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=val_texts,
        dev_labels=val_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    teacher = teacher_res["model"]

    # 5) Fine-tune student
    print("[Sentiment] Fine-tuning STUDENT...")
    student_res = train_sentiment_model(
        model=student,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=val_texts,
        dev_labels=val_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    student = student_res["model"]

    # 6) Evaluate on test
    print("[Sentiment] Evaluating TEACHER on test...")
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

    print("[Sentiment] Evaluating STUDENT on test...")
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
        "teacher": {"metrics": teacher_metrics},
        "student": {"metrics": student_metrics},
    }


if __name__ == "__main__":
    # Simple CLI run with defaults
    run_sentiment()
