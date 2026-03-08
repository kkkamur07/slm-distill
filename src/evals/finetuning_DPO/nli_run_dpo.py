import copy

import torch
from transformers import AutoTokenizer, PreTrainedModel

from src.evals.nli_eval import compute_nli_accuracy
from src.evals.task_finetuning.nli_data import load_nli_split
from src.evals.finetuning_DPO.nli_train_dpo import (
    create_nli_classifier,
    train_nli_model_score,
)


def run_nli(
    num_epochs: int,
    batch_size: int,
    lr_grid: list[float] | None,
    max_length: int,
    dropout,
    weight_decay: float,
    device: str | None,
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
    train_limit: int | None = None,
    dev_limit: int | None = None,
    test_limit: int | None = None,
):
    if lr_grid is None or len(lr_grid) == 0:
        raise ValueError("lr_grid must be non-empty.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NLI][Score] Using device: {device}")

    if tokenizer is None:
        tok_src = teacher_model_name or getattr(teacher_model.config, "_name_or_path", None)
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(train_path)
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(dev_path, raw_to_id=raw_to_id)
    test_prem, test_hyp, test_labels, _ = load_nli_split(test_path, raw_to_id=raw_to_id)

    def _trim(prem, hyp, labels, limit):
        if limit is None:
            return prem, hyp, labels
        return prem[:limit], hyp[:limit], labels[:limit]

    train_prem, train_hyp, train_labels = _trim(train_prem, train_hyp, train_labels, train_limit)
    dev_prem, dev_hyp, dev_labels = _trim(dev_prem, dev_hyp, dev_labels, dev_limit)
    test_prem, test_hyp, test_labels = _trim(test_prem, test_hyp, test_labels, test_limit)
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

    def grid_search_score(model_template, label: str):
        best_acc = -1.0
        best_model = None
        best_lr = None
        best_history = None
        best_dev = None

        for lr in lr_grid:
            print(f"\n[NLI][Score] Fine-tuning '{label}' with lr={lr:.1e}...")
            model = copy.deepcopy(model_template)
            res = train_nli_model_score(
                model=model,
                tokenizer=tokenizer,
                train_premises=train_prem,
                train_hypotheses=train_hyp,
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

            dev_metrics = compute_nli_accuracy(
                trained,
                tokenizer,
                dev_prem,
                dev_hyp,
                dev_labels,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            dev_acc = dev_metrics["accuracy"]
            print(
                f"[NLI][Score] lr={lr:.1e} -> dev accuracy={dev_acc:.4f} "
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
            f"[NLI][Score] Best lr for '{label}' is {best_lr:.1e} "
            f"with dev accuracy={best_acc:.4f}"
        )
        return best_model, best_lr, best_history, None, best_dev

    teacher, best_lr_teacher, teacher_history, _, teacher_best_dev = grid_search_score(
        teacher_template, "teacher"
    )
    student_ce, best_lr_student_ce, student_ce_history, _, student_ce_best_dev = grid_search_score(
        student_ce_template, "student_ce"
    )
    student_score, best_lr_student_score, student_score_history, _, student_score_best_dev = grid_search_score(
        student_score_template, "student_score"
    )

    print("\n[NLI] Evaluating on test...")
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

    print("[NLI] TEACHER metrics:", teacher_metrics)
    print("[NLI] STUDENT_CE metrics:", student_ce_metrics)
    print("[NLI] STUDENT_SCORE metrics:", student_score_metrics)

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
