import copy

import torch
from transformers import AutoTokenizer, PreTrainedModel

from src.evals.nli_eval import compute_nli_accuracy
from src.evals.task_finetuning.nli_data import load_nli_split
from src.evals.task_finetuning.nli_train import create_nli_classifier, train_nli_model


def run_nli(
    batch_size: int,
    max_length: int,
    device: str | None,
    weight_decay: float,
    dropout,
    num_epochs: int | None = None,
    teacher_num_epochs: int | None = None,
    student_num_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    teacher_lr_grid: list[float] | None = None,
    student_lr_grid: list[float] | None = None,
    lr_grid: list[float] | None = None,
    teacher_patience: int | None = None,
    student_patience: int | None = None,
    warmup_ratio: float = 0.1,
    min_delta: float = 1e-5,
    train_path: str = "data/hin/xnli_hi_train.json",
    dev_path: str = "data/hin/xnli_hi_dev.json",
    test_path: str = "data/hin/xnli_hi_test.json",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_ce_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_ce_subfolder: str | None = "model",
    student_score_model_name: str | None = None,
    teacher_model: PreTrainedModel | None = None,
    student_ce_model: PreTrainedModel | None = None,
    student_score_model: PreTrainedModel | None = None,
    tokenizer=None,
):
    """
    NLI fine-tuning with separate LR grids for teacher and students.
    Uses dev accuracy to pick the best LR.
    """
    if teacher_lr_grid is None:
        teacher_lr_grid = lr_grid
    if student_lr_grid is None:
        student_lr_grid = lr_grid
    if teacher_lr_grid is None or len(teacher_lr_grid) == 0:
        raise ValueError("teacher_lr_grid must be a non-empty list of learning rates.")
    if student_lr_grid is None or len(student_lr_grid) == 0:
        raise ValueError("student_lr_grid must be a non-empty list of learning rates.")
    base_epochs = num_epochs or teacher_num_epochs or student_num_epochs
    if base_epochs is None:
        raise ValueError("Provide num_epochs or per-model epochs.")
    teacher_epochs = teacher_num_epochs or base_epochs
    student_epochs = student_num_epochs or base_epochs
    base_patience = early_stopping_patience
    teacher_pat = teacher_patience if teacher_patience is not None else base_patience
    student_pat = student_patience if student_patience is not None else base_patience

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NLI] Using device: {device}")

    if tokenizer is None:
        tokenizer_source = teacher_model_name
        if tokenizer_source is None and teacher_model is not None:
            tokenizer_source = getattr(teacher_model.config, "_name_or_path", None)
        if tokenizer_source is None:
            raise ValueError("Provide a tokenizer or a teacher model/name.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    # Load splits with a shared label mapping
    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(train_path)
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(dev_path, raw_to_id=raw_to_id)
    test_prem, test_hyp, test_labels, _ = load_nli_split(test_path, raw_to_id=raw_to_id)
    num_labels = len(raw_to_id)

    print(
        f"[NLI] train={len(train_prem)}, dev={len(dev_prem)}, "
        f"test={len(test_prem)}, num_labels={num_labels}"
    )

    teacher_template = create_nli_classifier(
        base_model_name=teacher_model_name,
        base_model=teacher_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=None,
    )
    student_ce_template = create_nli_classifier(
        base_model_name=student_ce_model_name,
        base_model=student_ce_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_nli_classifier(
        base_model_name=student_score_model_name,
        base_model=student_score_model,
        num_labels=num_labels,
        dropout=dropout,
        subfolder=None,
    )

    # ---- helper: grid search for one HF model id ----
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
        best_batch_history = None
        best_dev_summary = None

        for lr in lr_grid:
            print(f"\n[NLI] Fine-tuning '{label}' with lr={lr:.1e}...")
            model = copy.deepcopy(model_template)

            res = train_nli_model(
                model=model,
                tokenizer=tokenizer,
                train_premises=train_prem,
                train_hypotheses=train_hyp,
                train_labels=train_labels,
                dev_premises=dev_prem,
                dev_hypotheses=dev_hyp,
                dev_labels=dev_labels,
                device=device,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                eval_on_dev=True,
                weight_decay=weight_decay,
                early_stopping_patience=patience,
                min_delta=min_delta,
                warmup_steps=int(((len(train_prem) + batch_size - 1) // batch_size * epochs) * warmup_ratio),
            )
            model = res["model"]
            history = res["history"]
            batch_history = res.get("batch_history")
            best_dev_metrics = res.get("best_dev_metrics")
            best_epoch = res.get("best_epoch")

            if best_dev_metrics is not None:
                dev_acc = best_dev_metrics["accuracy"]
                print(
                    f"[NLI] {label}, lr={lr:.1e} best dev (epoch {best_epoch}): "
                    f"acc={dev_acc:.4f}, macro-F1={best_dev_metrics['macro_f1']:.4f}, "
                    f"micro-F1={best_dev_metrics['micro_f1']:.4f}, "
                    f"precision={best_dev_metrics['precision']:.4f}, "
                    f"recall={best_dev_metrics['recall']:.4f}"
                )
            else:
                last_dev = res["history"][-1]["dev_metrics"]
                dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
                print(f"[NLI] {label}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            # update best regardless of whether best_dev_metrics existed
            dev_summary = best_dev_metrics if best_dev_metrics is not None else last_dev
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr
                best_history = history
                best_batch_history = batch_history
                best_dev_summary = dev_summary

            print(
                f"[NLI] Best lr for {label}: {best_lr:.1e} "
                f"(dev acc={best_acc:.4f})"
            )
            
            del model, res, history, batch_history
            torch.cuda.empty_cache()
            
            if best_dev_summary is not None:
                print(f"[NLI] Best dev metrics for '{label}': {best_dev_summary}")
        return best_model, best_lr, best_history, best_batch_history, best_dev_summary

    # Teacher grid search
    (
        teacher,
        best_lr_teacher,
        teacher_history,
        teacher_batch_history,
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
        student_ce_batch_history,
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
        student_score_batch_history,
        student_score_best_dev,
    ) = grid_search_model(
        student_score_template,
        "student_score",
        student_lr_grid,
        student_epochs,
        student_pat,
    )

    # test evaluations
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

    print("\n[NLI] Evaluating STUDENT_CE on test...")
    student_ce_metrics = compute_nli_accuracy(
        student_ce,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Evaluating STUDENT_SCORE on test...")
    student_score_metrics = compute_nli_accuracy(
        student_score,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Final test results:")
    print("  Teacher:", teacher_metrics, f"(best lr={best_lr_teacher:.1e})")
    print("  Student_CE:", student_ce_metrics, f"(best lr={best_lr_student_ce:.1e})")
    print(
        "  Student_SCORE:",
        student_score_metrics,
        f"(best lr={best_lr_student_score:.1e})",
    )

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
