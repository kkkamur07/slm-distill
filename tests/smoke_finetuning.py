"""
Quick smoke test for the finetuning pipeline.

Runs tiny trainings (2 epochs, small subsets, 2 lrs) so it finishes fast while
exercising model loading, grid search, and metric reporting.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Ensure local src/ is on the path when running as `python scripts/smoke_finetuning.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evals.sentiment_eval import compute_sentiment_accuracy
from src.evals.task_finetuning.sentiment_data import load_sentiment_csv
from src.evals.task_finetuning.sentiment_train import (
    create_sentiment_classifier,
    train_sentiment_model,
)

from src.evals.nli_eval import compute_nli_accuracy
from src.evals.task_finetuning.nli_data import load_nli_split
from src.evals.task_finetuning.nli_train import (
    create_nli_classifier,
    train_nli_model,
)

from src.evals.ner_eval import compute_ner_accuracy
from src.evals.task_finetuning.ner_data import load_wikiann_split
from src.evals.task_finetuning.ner_train import (
    create_ner_tagger,
    train_ner_model,
)


def _take(xs, n):
    return xs[:n] if n is not None else xs


def _load_score_base(
    backbone_name: str | None,
    weight_path: Path,
    subfolder: str | None = None,
    allow_missing: bool = False,
    fallback_config=None,
):
    if not weight_path.exists():
        if allow_missing:
            print(
                f"[Smoke] Student score weights not found at {weight_path}; "
                "skipping student_score."
            )
            return None
        raise FileNotFoundError(f"Student score weights not found at {weight_path}")

    kwargs = {}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder

    base = None
    if backbone_name is not None:
        try:
            base = AutoModel.from_pretrained(backbone_name, **kwargs)
        except OSError as e:
            if fallback_config is None:
                raise
            print(
                f"[Smoke] Could not load backbone '{backbone_name}' from_pretrained "
                f"({e}); falling back to config initialisation."
            )
    if base is None:
        if fallback_config is None:
            raise RuntimeError("No backbone or fallback config provided for score model.")
        base = AutoModel.from_config(copy.deepcopy(fallback_config))

    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    base.load_state_dict(state, strict=False)
    return base


def smoke_sentiment(
    device: str,
    teacher_lr_grid: list[float],
    student_lr_grid: list[float],
    teacher_num_epochs: int,
    student_num_epochs: int,
    batch_size: int,
    max_length: int,
    train_limit: int,
    dev_limit: int,
    test_limit: int,
    teacher_model_name: str,
    student_ce_model_name: str,
    student_ce_subfolder: str,
    student_ce_path: Path,
    student_score_path: Path,
    student_score_backbone: str | None,
):
    print("\n=== Sentiment smoke ===")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if not student_ce_path.exists():
        raise FileNotFoundError(f"Student CE weights not found at {student_ce_path}")
    ce_state = torch.load(student_ce_path, map_location="cpu", weights_only=False)
    student_ce_base.load_state_dict(ce_state, strict=False)
    student_score_base = _load_score_base(
        backbone_name=student_score_backbone or student_ce_model_name,
        weight_path=student_score_path,
        fallback_config=student_ce_base.config,
    )

    train_texts, train_labels, label2id = load_sentiment_csv(
        "data/hin/sentiment_hi_train.csv", num_labels=3
    )
    dev_texts, dev_labels, _ = load_sentiment_csv(
        "data/hin/sentiment_hi_val.csv", label2id=label2id, num_labels=3
    )
    test_texts, test_labels, _ = load_sentiment_csv(
        "data/hin/sentiment_hi_test.csv", label2id=label2id, num_labels=3
    )

    train_texts, train_labels = _take(train_texts, train_limit), _take(
        train_labels, train_limit
    )
    dev_texts, dev_labels = _take(dev_texts, dev_limit), _take(dev_labels, dev_limit)
    test_texts, test_labels = _take(test_texts, test_limit), _take(
        test_labels, test_limit
    )

    teacher_template = create_sentiment_classifier(
        base_model_name=teacher_model_name,
        base_model=teacher_base,
        num_labels=3,
        dropout=0.1,
    )
    student_ce_template = create_sentiment_classifier(
        base_model_name=student_ce_model_name,
        base_model=student_ce_base,
        num_labels=3,
        dropout=0.1,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_sentiment_classifier(
        base_model_name=None,
        base_model=student_score_base,
        num_labels=3,
        dropout=0.1,
    )

    def grid(model_template, label, lr_grid, epochs):
        best_acc = -1.0
        best_model = None
        for lr in lr_grid:
            print(f"[Sentiment] {label} lr={lr}")
            model = copy.deepcopy(model_template)
            res = train_sentiment_model(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                train_labels=train_labels,
                dev_texts=dev_texts,
                dev_labels=dev_labels,
                device=device,
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                weight_decay=0.0,
                early_stopping_patience=1,
                min_delta=0.0,
                eval_on_dev=True,
            )
            metrics = res.get("best_dev_metrics") or res["history"][-1]["dev_metrics"]
            print(f"[Sentiment] {label} lr={lr} dev metrics: {metrics}")
            if metrics and metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_model = res["model"]
        return best_model

    teacher = grid(teacher_template, "teacher", teacher_lr_grid, teacher_num_epochs)
    student_ce = grid(
        student_ce_template, "student_ce", student_lr_grid, student_num_epochs
    )
    student_score = grid(
        student_score_template, "student_score", student_lr_grid, student_num_epochs
    )

    print("[Sentiment] Evaluating test...")
    teacher_metrics = compute_sentiment_accuracy(
        teacher,
        tokenizer,
        test_texts,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    student_ce_metrics = compute_sentiment_accuracy(
        student_ce,
        tokenizer,
        test_texts,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    student_score_metrics = compute_sentiment_accuracy(
        student_score,
        tokenizer,
        test_texts,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("Teacher test:", teacher_metrics)
    print("Student_CE test:", student_ce_metrics)
    print("Student_SCORE test:", student_score_metrics)


def smoke_nli(
    device: str,
    teacher_lr_grid: list[float],
    student_lr_grid: list[float],
    teacher_num_epochs: int,
    student_num_epochs: int,
    batch_size: int,
    max_length: int,
    train_limit: int,
    dev_limit: int,
    test_limit: int,
    teacher_model_name: str,
    student_ce_model_name: str,
    student_ce_subfolder: str,
    student_ce_path: Path,
    student_score_path: Path,
    student_score_backbone: str | None,
):
    print("\n=== NLI smoke ===")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if not student_ce_path.exists():
        raise FileNotFoundError(f"Student CE weights not found at {student_ce_path}")
    ce_state = torch.load(student_ce_path, map_location="cpu", weights_only=False)
    student_ce_base.load_state_dict(ce_state, strict=False)
    student_score_base = _load_score_base(
        backbone_name=student_score_backbone or student_ce_model_name,
        weight_path=student_score_path,
        fallback_config=student_ce_base.config,
    )

    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(
        "data/hin/xnli_hi_train.json"
    )
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(
        "data/hin/xnli_hi_dev.json", raw_to_id=raw_to_id
    )
    test_prem, test_hyp, test_labels, _ = load_nli_split(
        "data/hin/xnli_hi_test.json", raw_to_id=raw_to_id
    )

    train_prem, train_hyp, train_labels = (
        _take(train_prem, train_limit),
        _take(train_hyp, train_limit),
        _take(train_labels, train_limit),
    )
    dev_prem, dev_hyp, dev_labels = (
        _take(dev_prem, dev_limit),
        _take(dev_hyp, dev_limit),
        _take(dev_labels, dev_limit),
    )
    test_prem, test_hyp, test_labels = (
        _take(test_prem, test_limit),
        _take(test_hyp, test_limit),
        _take(test_labels, test_limit),
    )

    num_labels = len(raw_to_id)

    teacher_template = create_nli_classifier(
        base_model_name=teacher_model_name,
        base_model=teacher_base,
        num_labels=num_labels,
        dropout=0.1,
    )
    student_ce_template = create_nli_classifier(
        base_model_name=student_ce_model_name,
        base_model=student_ce_base,
        num_labels=num_labels,
        dropout=0.1,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_nli_classifier(
        base_model_name=None,
        base_model=student_score_base,
        num_labels=num_labels,
        dropout=0.1,
    )

    def grid(model_template, label, lr_grid, epochs):
        best_acc = -1.0
        best_model = None
        for lr in lr_grid:
            print(f"[NLI] {label} lr={lr}")
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
                weight_decay=0.0,
                early_stopping_patience=1,
                min_delta=0.0,
                eval_on_dev=True,
            )
            metrics = res.get("best_dev_metrics") or res["history"][-1]["dev_metrics"]
            print(f"[NLI] {label} lr={lr} dev metrics: {metrics}")
            if metrics and metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_model = res["model"]
        return best_model

    teacher = grid(teacher_template, "teacher", teacher_lr_grid, teacher_num_epochs)
    student_ce = grid(
        student_ce_template, "student_ce", student_lr_grid, student_num_epochs
    )
    student_score = grid(
        student_score_template, "student_score", student_lr_grid, student_num_epochs
    )

    print("[NLI] Evaluating test...")
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
    print("Teacher test:", teacher_metrics)
    print("Student_CE test:", student_ce_metrics)
    print("Student_SCORE test:", student_score_metrics)


def smoke_ner(
    device: str,
    teacher_lr_grid: list[float],
    student_lr_grid: list[float],
    teacher_num_epochs: int,
    student_num_epochs: int,
    batch_size: int,
    max_length: int,
    train_limit: int,
    dev_limit: int,
    test_limit: int,
    teacher_model_name: str,
    student_ce_model_name: str,
    student_ce_subfolder: str,
    student_ce_path: Path,
    student_score_path: Path,
    student_score_backbone: str | None,
):
    print("\n=== NER smoke ===")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if not student_ce_path.exists():
        raise FileNotFoundError(f"Student CE weights not found at {student_ce_path}")
    ce_state = torch.load(student_ce_path, map_location="cpu", weights_only=False)
    student_ce_base.load_state_dict(ce_state, strict=False)
    student_score_base = _load_score_base(
        backbone_name=student_score_backbone or student_ce_model_name,
        weight_path=student_score_path,
        fallback_config=student_ce_base.config,
    )

    train_sentences, train_labels = load_wikiann_split(
        "data/hin/train-00000-of-00001.parquet"
    )
    dev_sentences, dev_labels = load_wikiann_split(
        "data/hin/validation-00000-of-00001.parquet"
    )
    test_sentences, test_labels = load_wikiann_split(
        "data/hin/test-00000-of-00001.parquet"
    )

    train_sentences, train_labels = _take(train_sentences, train_limit), _take(
        train_labels, train_limit
    )
    dev_sentences, dev_labels = _take(dev_sentences, dev_limit), _take(
        dev_labels, dev_limit
    )
    test_sentences, test_labels = _take(test_sentences, test_limit), _take(
        test_labels, test_limit
    )

    all_ids = {lid for seq in train_labels for lid in seq}
    num_labels = max(all_ids) + 1
    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    teacher_template = create_ner_tagger(
        base_model_name=teacher_model_name,
        base_model=teacher_base,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
    )
    student_ce_template = create_ner_tagger(
        base_model_name=student_ce_model_name,
        base_model=student_ce_base,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
        subfolder=student_ce_subfolder,
    )
    student_score_template = create_ner_tagger(
        base_model_name=None,
        base_model=student_score_base,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
    )

    def grid(model_template, label, lr_grid, epochs):
        best_acc = -1.0
        best_model = None
        for lr in lr_grid:
            print(f"[NER] {label} lr={lr}")
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
                weight_decay=0.0,
                min_delta=0.0,
                early_stopping_patience=1,
                ignore_index=-100,
                eval_on_dev=True,
            )
            metrics = res.get("best_dev_metrics") or res["history"][-1]["dev_metrics"]
            print(f"[NER] {label} lr={lr} dev metrics: {metrics}")
            if metrics and metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_model = res["model"]
        return best_model

    teacher = grid(teacher_template, "teacher", teacher_lr_grid, teacher_num_epochs)
    student_ce = grid(
        student_ce_template, "student_ce", student_lr_grid, student_num_epochs
    )
    student_score = grid(
        student_score_template, "student_score", student_lr_grid, student_num_epochs
    )

    print("[NER] Evaluating test...")
    teacher_metrics = compute_ner_accuracy(
        teacher,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    student_ce_metrics = compute_ner_accuracy(
        student_ce,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    student_score_metrics = compute_ner_accuracy(
        student_score,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("Teacher test:", teacher_metrics)
    print("Student_CE test:", student_ce_metrics)
    print("Student_SCORE test:", student_score_metrics)


def main():
    parser = argparse.ArgumentParser(description="Fast smoke test for finetuning.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sentiment"],
        choices=["sentiment", "nli", "ner"],
        help="Which tasks to run (default: sentiment).",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (auto if unset).")
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--dev-limit", type=int, default=12)
    parser.add_argument("--test-limit", type=int, default=12)
    parser.add_argument("--teacher-num-epochs", type=int, default=2)
    parser.add_argument("--student-num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--teacher-lr-grid", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--student-lr-grid", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument(
        "--teacher-model",
        default="FacebookAI/xlm-roberta-base",
        help="HF id for teacher backbone.",
    )
    parser.add_argument(
        "--student-ce-model",
        default="kkkamur07/hindi-xlm-roberta-33M",
        help="HF id for student backbone.",
    )
    parser.add_argument(
        "--student-ce-subfolder",
        default="model",
        help="HF subfolder for student weights (if needed).",
    )
    parser.add_argument(
        "--student-ce-path",
        default="checkpoints/models/model_best_ce.pt",
        help="Path to CE student weights (.pt/.bin).",
    )
    parser.add_argument(
        "--student-score-path",
        default="checkpoints/models/model_best_score.pt",
        help="Path to score-based student .pt weights.",
    )
    parser.add_argument(
        "--student-score-backbone",
        default=None,
        help="Backbone to load before applying score weights (defaults to student_ce model).",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Smoke] Using device: {device}")

    for task in args.tasks:
        if task == "sentiment":
            smoke_sentiment(
                device=device,
                teacher_lr_grid=args.teacher_lr_grid,
                student_lr_grid=args.student_lr_grid,
                teacher_num_epochs=args.teacher_num_epochs,
                student_num_epochs=args.student_num_epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                train_limit=args.train_limit,
                dev_limit=args.dev_limit,
                test_limit=args.test_limit,
                teacher_model_name=args.teacher_model,
                student_ce_model_name=args.student_ce_model,
                student_ce_subfolder=args.student_ce_subfolder,
                student_ce_path=Path(args.student_ce_path),
                student_score_path=Path(args.student_score_path),
                student_score_backbone=args.student_score_backbone,
            )
        elif task == "nli":
            smoke_nli(
                device=device,
                teacher_lr_grid=args.teacher_lr_grid,
                student_lr_grid=args.student_lr_grid,
                teacher_num_epochs=args.teacher_num_epochs,
                student_num_epochs=args.student_num_epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                train_limit=args.train_limit,
                dev_limit=args.dev_limit,
                test_limit=args.test_limit,
                teacher_model_name=args.teacher_model,
                student_ce_model_name=args.student_ce_model,
                student_ce_subfolder=args.student_ce_subfolder,
                student_ce_path=Path(args.student_ce_path),
                student_score_path=Path(args.student_score_path),
                student_score_backbone=args.student_score_backbone,
            )
        elif task == "ner":
            smoke_ner(
                device=device,
                teacher_lr_grid=args.teacher_lr_grid,
                student_lr_grid=args.student_lr_grid,
                teacher_num_epochs=args.teacher_num_epochs,
                student_num_epochs=args.student_num_epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                train_limit=args.train_limit,
                dev_limit=args.dev_limit,
                test_limit=args.test_limit,
                teacher_model_name=args.teacher_model,
                student_ce_model_name=args.student_ce_model,
                student_ce_subfolder=args.student_ce_subfolder,
                student_ce_path=Path(args.student_ce_path),
                student_score_path=Path(args.student_score_path),
                student_score_backbone=args.student_score_backbone,
            )


if __name__ == "__main__":
    main()
