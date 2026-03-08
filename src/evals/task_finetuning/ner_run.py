import copy

import torch
from transformers import AutoTokenizer, PreTrainedModel

from src.evals.ner_eval import compute_ner_accuracy
from src.evals.task_finetuning.ner_data import load_wikiann_split
from src.evals.task_finetuning.ner_train import create_ner_tagger, train_ner_model


def run_ner(
    batch_size: int,
    max_length: int,
    device: str | None,
    weight_decay: float,
    dropout: float = 0.1,
    num_epochs: int | None = None,
    teacher_num_epochs: int | None = None,
    student_num_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    teacher_lr_grid: list[float] | None = None,
    student_lr_grid: list[float] | None = None,
    lr_grid: list[float] | None = None,
    teacher_patience: int | None = None,
    student_patience: int | None = None,
    warmup_ratio: float = 0.6,
    min_delta: float = 1e-5, ## early stopping
    train_path: str = "data/hin/train-00000-of-00001.parquet",
    dev_path: str = "data/hin/validation-00000-of-00001.parquet",
    test_path: str = "data/hin/test-00000-of-00001.parquet",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_ce_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_ce_subfolder: str | None = "model",  # None if not needed
    student_score_model_name: str | None = None,
    teacher_model: PreTrainedModel | None = None,
    student_ce_model: PreTrainedModel | None = None,
    student_score_model: PreTrainedModel | None = None,
    tokenizer=None,
):
    """
    Fine-tune teacher and student on WikiANN Hindi NER and evaluate on test.

    Adds:
      - weight decay for AdamW
      - early stopping on dev accuracy
      - learning-rate grid search for BOTH teacher and student

    Returns:
        {
            "teacher": {"metrics": {...}, "best_lr": float},
            "student": {"metrics": {...}, "best_lr": float},
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NER] Using device: {device}")

    # LR grids (teacher vs students)
    if teacher_lr_grid is None:
        teacher_lr_grid = lr_grid
    if student_lr_grid is None:
        student_lr_grid = lr_grid
    if teacher_lr_grid is None or len(teacher_lr_grid) == 0:
        raise ValueError("teacher_lr_grid for NER not provided")
    if student_lr_grid is None or len(student_lr_grid) == 0:
        raise ValueError("student_lr_grid for NER not provided")
    base_epochs = num_epochs or teacher_num_epochs or student_num_epochs
    if base_epochs is None:
        raise ValueError("Provide num_epochs or per-model epochs.")
    teacher_epochs = teacher_num_epochs or base_epochs
    student_epochs = student_num_epochs or base_epochs
    base_patience = early_stopping_patience
    teacher_pat = teacher_patience if teacher_patience is not None else base_patience
    student_pat = student_patience if student_patience is not None else base_patience
    if tokenizer is None:
        tokenizer_source = teacher_model_name
        if tokenizer_source is None and teacher_model is not None:
            tokenizer_source = getattr(teacher_model.config, "_name_or_path", None)
        if tokenizer_source is None:
            raise ValueError("Provide a tokenizer or a teacher model/name.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    # Load dataset splits
    train_sentences, train_labels = load_wikiann_split(train_path)
    dev_sentences, dev_labels = load_wikiann_split(dev_path)
    test_sentences, test_labels = load_wikiann_split(test_path)

    # Infer label space from train labels
    all_ids = {lid for seq in train_labels for lid in seq}
    num_labels = max(all_ids) + 1
    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    print(
        f"[NER] train={len(train_sentences)}, dev={len(dev_sentences)}, "
        f"test={len(test_sentences)}, num_labels={num_labels}"
    )

    teacher_template = create_ner_tagger(
        base_model_name=teacher_model_name,
        base_model=teacher_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=dropout,
        subfolder=None,
    )
    student_ce_template = create_ner_tagger(
        base_model_name=student_ce_model_name,
        base_model=student_ce_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=dropout,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_ner_tagger(
        base_model_name=student_score_model_name,
        base_model=student_score_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=dropout,
        subfolder=None,
    )

    def grid_search_model(
        model_template,
        label: str,
        lr_grid: list[float],
        epochs: int,
        patience: int | None,
    ):
        best_acc = -1.0
        best_model = None
        best_lr = None
        best_history = None
        best_dev_summary = None

        for lr in lr_grid:
            print(f"\n[NER] Fine-tuning '{label}' with lr={lr:.1e}...")
            model = copy.deepcopy(model_template)

            res = train_ner_model(
                model=model,
                tokenizer=tokenizer,
                train_sentences=train_sentences,
                train_labels=train_labels,
                dev_sentences=dev_sentences,
                dev_labels=dev_labels,
                device=device,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                ignore_index=-100,
                eval_on_dev=True,
                weight_decay=weight_decay,
                early_stopping_patience=patience,
                min_delta=min_delta,
                warmup_steps=int(((len(train_sentences) + batch_size - 1) // batch_size * epochs) * warmup_ratio),
            )
            model = res["model"]
            history = res["history"]
            best_dev_metrics = res.get("best_dev_metrics")
            best_epoch = res.get("best_epoch")
            if best_dev_metrics is not None:
                dev_acc = best_dev_metrics["accuracy"]
                print(
                    f"[NER] {label}, lr={lr:.1e} best dev (epoch {best_epoch}): "
                    f"acc={dev_acc:.4f}, macro-F1={best_dev_metrics['macro_f1']:.4f}, "
                    f"micro-F1={best_dev_metrics['micro_f1']:.4f}, "
                    f"precision={best_dev_metrics['precision']:.4f}, "
                    f"recall={best_dev_metrics['recall']:.4f}"
                )
            else:
                last_dev = history[-1]["dev_metrics"]
                dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
                print(f"[NER] {label}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr
                best_history = history
                best_dev_summary = best_dev_metrics if best_dev_metrics else last_dev

        if best_model is None:
            raise RuntimeError(f"[NER] Grid search failed for {label}")

        print(
            f"[NER] Best lr for {label}: {best_lr:.1e} "
            f"(dev acc={best_acc:.4f})"
        )
        if best_dev_summary is not None:
            print(f"[NER] Best dev metrics for '{label}': {best_dev_summary}")
        return best_model, best_lr, best_history, best_dev_summary

    # Teacher grid search
    (
        teacher,
        best_lr_teacher,
        teacher_history,
        teacher_best_dev,
    ) = grid_search_model(
        teacher_template,
        "teacher",
        teacher_lr_grid,
        teacher_epochs,
        teacher_pat,
    )

    # Student CE grid search
    (
        student_ce,
        best_lr_student_ce,
        student_ce_history,
        student_ce_best_dev,
    ) = grid_search_model(
        student_ce_template,
        "student_ce",
        student_lr_grid,
        student_epochs,
        student_pat,
    )

    # Student SCORE grid search
    (
        student_score,
        best_lr_student_score,
        student_score_history,
        student_score_best_dev,
    ) = grid_search_model(
        student_score_template,
        "student_score",
        student_lr_grid,
        student_epochs,
        student_pat,
    )

    # Final test evaluation
    print("[NER] Evaluating TEACHER on test...")
    teacher_metrics = compute_ner_accuracy(
        teacher,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NER] TEACHER metrics:", teacher_metrics)

    print("[NER] Evaluating STUDENT_CE on test...")
    student_ce_metrics = compute_ner_accuracy(
        student_ce,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NER] STUDENT_CE metrics:", student_ce_metrics)

    print("[NER] Evaluating STUDENT_SCORE on test...")
    student_score_metrics = compute_ner_accuracy(
        student_score,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NER] STUDENT_SCORE metrics:", student_score_metrics)

    return {
        "teacher": {
            "metrics": teacher_metrics,
            "best_lr": best_lr_teacher,
            "history": teacher_history,
            "best_dev_metrics": teacher_best_dev,
        },
        "student_ce": {
            "metrics": student_ce_metrics,
            "best_lr": best_lr_student_ce,
            "history": student_ce_history,
            "best_dev_metrics": student_ce_best_dev,
        },
        "student_score": {
            "metrics": student_score_metrics,
            "best_lr": best_lr_student_score,
            "history": student_score_history,
            "best_dev_metrics": student_score_best_dev,
        },
    }
