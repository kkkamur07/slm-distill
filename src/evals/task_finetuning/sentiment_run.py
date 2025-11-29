import torch
from transformers import AutoTokenizer

from src.evals.sentiment_eval import compute_sentiment_accuracy
from src.evals.task_finetuning.sentiment_data import load_sentiment_csv
from src.evals.task_finetuning.sentiment_train import (
    create_sentiment_classifier,
    train_sentiment_model,
)


def run_sentiment(
    num_labels: int,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    device: str | None,
    weight_decay: float,
    early_stopping_patience: int | None,
    lr_grid: list[float] | None,
    dropout: float,
    min_delta: float = 1e-5,
    train_path: str = "data/hin/sentiment_hi_train.csv",
    val_path: str = "data/hin/sentiment_hi_val.csv",
    test_path: str = "data/hin/sentiment_hi_test.csv",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str = "model",
):
    if not lr_grid:
        raise ValueError("lr_grid must be a non-empty list of learning rates.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sentiment] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    # Load datasets with a shared label mapping and sanity-check num_labels.
    train_texts, train_labels, label2id = load_sentiment_csv(
        train_path, num_labels=num_labels
    )
    val_texts, val_labels, _ = load_sentiment_csv(
        val_path, label2id=label2id, num_labels=num_labels
    )
    test_texts, test_labels, _ = load_sentiment_csv(
        test_path, label2id=label2id, num_labels=num_labels
    )

    print(
        f"[Sentiment] train={len(train_texts)}, val={len(val_texts)}, "
        f"test={len(test_texts)}, num_labels={num_labels}"
    )

    def grid_search_model(base_model_name: str, subfolder: str | None = None):
        best_acc = -1.0
        best_model = None
        best_lr = None
        best_history = None
        best_batch_history = None

        for lr in lr_grid:
            print(
                f"\n[Sentiment] Fine-tuning '{base_model_name}' with lr={lr:.1e}..."
            )
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
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
                eval_on_dev=True,
            )

            model_trained = res["model"]
            history = res["history"]
            batch_history = res.get("batch_history")

            last_dev = history[-1]["dev_metrics"]
            dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
            print(f"[Sentiment] lr={lr:.1e} -> dev accuracy={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model_trained
                best_lr = lr
                best_history = history
                best_batch_history = batch_history

        if best_model is None:
            raise RuntimeError("No model was trained; check lr_grid configuration.")

        print(
            f"[Sentiment] Best lr for '{base_model_name}' is {best_lr:.1e} "
            f"with dev accuracy={best_acc:.4f}"
        )
        return best_model, best_lr, best_history, best_batch_history

    # Teacher model grid search
    teacher, best_lr_teacher, teacher_history, teacher_batch_history = grid_search_model(
        base_model_name=teacher_model_name,
        subfolder=None,
    )

    # Student model grid search
    student, best_lr_student, student_history, student_batch_history = grid_search_model(
        base_model_name=student_model_name,
        subfolder=student_subfolder,
    )

    # Final evaluation on test set for the best teacher model
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

    # And for the best student model
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
        "teacher": {
            "metrics": teacher_metrics,
            "best_lr": best_lr_teacher,
            "history": teacher_history,
            "batch_history": teacher_batch_history,
        },
        "student": {
            "metrics": student_metrics,
            "best_lr": best_lr_student,
            "history": student_history,
            "batch_history": student_batch_history,
        },
    }
