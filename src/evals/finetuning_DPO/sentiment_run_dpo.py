import copy

import torch
from transformers import AutoTokenizer, PreTrainedModel

from src.evals.sentiment_eval import compute_sentiment_accuracy
from src.evals.task_finetuning.sentiment_data import load_sentiment_csv
from src.evals.finetuning_DPO.sentiment_train_dpo import (
    create_sentiment_classifier,
    train_sentiment_model,
    train_sentiment_model_score,
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
    student_ce_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_ce_subfolder: str = "model",
    student_score_model_name: str | None = None,
    teacher_model: PreTrainedModel | None = None,
    student_ce_model: PreTrainedModel | None = None,
    student_score_model: PreTrainedModel | None = None,
    tokenizer=None,
):
    if not lr_grid:
        raise ValueError("lr_grid must be a non-empty list of learning rates.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sentiment] Using device: {device}")

    if tokenizer is None:
        tokenizer_source = teacher_model_name
        if tokenizer_source is None and teacher_model is not None:
            tokenizer_source = getattr(teacher_model.config, "_name_or_path", None)
        if tokenizer_source is None:
            raise ValueError("Provide a tokenizer or a teacher model/name.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

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

    teacher_template = create_sentiment_classifier(
        base_model_name=teacher_model_name,
        base_model=teacher_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=None,
    )
    student_ce_template = create_sentiment_classifier(
        base_model_name=student_ce_model_name,
        base_model=student_ce_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_sentiment_classifier(
        base_model_name=student_score_model_name,
        base_model=student_score_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=None,
    )

    def grid_search_model(model_template, label: str):
        best_acc = -1.0
        best_model = None
        best_lr = None
        best_history = None
        best_batch_history = None
        best_dev_summary = None

        for lr in lr_grid:
            print(
                f"\n[Sentiment] Fine-tuning '{label}' with lr={lr:.1e}..."
            )
            model = copy.deepcopy(model_template)

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
            best_dev_metrics = res.get("best_dev_metrics")
            best_epoch = res.get("best_epoch")

            if best_dev_metrics is not None:
                dev_acc = best_dev_metrics["accuracy"]
                print(
                    f"[Sentiment] lr={lr:.1e} best dev (epoch {best_epoch}): "
                    f"acc={dev_acc:.4f}, macro-F1={best_dev_metrics['macro_f1']:.4f}, "
                    f"micro-F1={best_dev_metrics['micro_f1']:.4f}, "
                    f"precision={best_dev_metrics['precision']:.4f}, "
                    f"recall={best_dev_metrics['recall']:.4f}"
                )
            else:
                last_dev = history[-1]["dev_metrics"]
                dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
                print(f"[Sentiment] lr={lr:.1e} -> dev accuracy={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model_trained
                best_lr = lr
                best_history = history
                best_batch_history = batch_history
                best_dev_summary = best_dev_metrics or last_dev

        if best_model is None:
            raise RuntimeError("No model was trained; check lr_grid configuration.")

        print(
            f"[Sentiment] Best lr for '{label}' is {best_lr:.1e} "
            f"with dev accuracy={best_acc:.4f}"
        )
        if best_dev_summary is not None:
            print(f"[Sentiment] Best dev metrics for '{label}': {best_dev_summary}")
        return best_model, best_lr, best_history, best_batch_history, best_dev_summary

    def grid_search_score(model_template, label: str):
        best_acc = -1.0
        best_model = None
        best_lr = None
        best_history = None
        for lr in lr_grid:
            print(f"\n[Sentiment][Score] Fine-tuning '{label}' with lr={lr:.1e}...")
            model = copy.deepcopy(model_template)
            res = train_sentiment_model_score(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                train_labels=train_labels,
                device=device,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                weight_decay=weight_decay,
                lambda_ce=0.9,
            )
            trained = res["model"]
            history = res["history"]

            # evaluate on val to pick best lr
            dev_metrics = compute_sentiment_accuracy(
                trained,
                tokenizer,
                val_texts,
                val_labels,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
            dev_acc = dev_metrics["accuracy"]
            print(
                f"[Sentiment][Score] lr={lr:.1e} -> dev accuracy={dev_acc:.4f} "
                f"(macro-F1={dev_metrics['macro_f1']:.4f})"
            )
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = trained
                best_lr = lr
                best_history = history
                best_dev = dev_metrics
        if best_model is None:
            raise RuntimeError(f"No score-based model trained for {label}")
        print(
            f"[Sentiment][Score] Best lr for '{label}' is {best_lr:.1e} "
            f"with dev accuracy={best_acc:.4f}"
        )
        return best_model, best_lr, best_history, None, best_dev

    # Teacher model grid search (score-based)
    (
        teacher,
        best_lr_teacher,
        teacher_history,
        teacher_batch_history,
        teacher_best_dev,
    ) = grid_search_score(
        teacher_template,
        "teacher",
    )

    # Student CE model grid search (score-based)
    (
        student_ce,
        best_lr_student_ce,
        student_ce_history,
        student_ce_batch_history,
        student_ce_best_dev,
    ) = grid_search_score(
        student_ce_template,
        "student_ce",
    )

    # Student SCORE model grid search (score-based)
    (
        student_score,
        best_lr_student_score,
        student_score_history,
        student_score_batch_history,
        student_score_best_dev,
    ) = grid_search_score(
        student_score_template,
        "student_score",
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

    # And for the best CE student model
    student_ce_metrics = compute_sentiment_accuracy(
        student_ce,
        tokenizer,
        test_texts,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[Sentiment] STUDENT_CE metrics:", student_ce_metrics)

    # And for the best score-based student model
    student_score_metrics = compute_sentiment_accuracy(
        student_score,
        tokenizer,
        test_texts,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[Sentiment] STUDENT_SCORE metrics:", student_score_metrics)

    return {
        "teacher": {
            "metrics": teacher_metrics,
            "best_lr": best_lr_teacher,
            "history": teacher_history,
            "batch_history": teacher_batch_history,
            "best_dev_metrics": teacher_best_dev,
        },
        "student_ce": {
            "metrics": student_ce_metrics,
            "best_lr": best_lr_student_ce,
            "history": student_ce_history,
            "batch_history": student_ce_batch_history,
            "best_dev_metrics": student_ce_best_dev,
        },
        "student_score": {
            "metrics": student_score_metrics,
            "best_lr": best_lr_student_score,
            "history": student_score_history,
            "batch_history": student_score_batch_history,
            "best_dev_metrics": student_score_best_dev,
        },
    }
