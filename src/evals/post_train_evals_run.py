"""Post-training evaluation orchestrator.

This module coordinates comprehensive post-training evaluations:
- run_mtp_eval: Runs all masked token prediction evaluations
  - Student vs teacher accuracy comparison
  - Agreement rates between models
  - Perplexity measurements
  - KL divergence analysis

Provides a unified interface for running multiple evaluation metrics
and saving results to JSON for analysis and comparison.

Example:
    >>> from src.evals.post_train_evals_run import run_mtp_eval
    >>> 
    >>> results = run_mtp_eval(
    ...     teacher=teacher_model,
    ...     student=student_model,
    ...     tokenizer=tokenizer,
    ...     data_path="eval_data.parquet",
    ...     seed=42,
    ...     max_length=128,
    ...     batch_size=32,
    ...     log_path="outputs/eval_results.json",
    ...     device="cuda"
    ... )
    >>> 
    >>> print(f"Teacher Accuracy: {results['teacher_accuracy']:.2%}")
    >>> print(f"Student Accuracy: {results['student_accuracy']:.2%}")
    >>> print(f"Agreement: {results['agreement']:.2%}")
    >>> print(f"KL Divergence: {results['kl_div']:.4f}")
"""

import json, random
from pathlib import Path

import numpy as np
import torch

from src.evals.mtp_perplexity_eval import (
    compute_masked_token_accuracy,
    compare_student_teacher_masked_token_agreement,
    compute_masked_token_perplexity,
    masked_token_kl,
)
from src.data.eval_prepare import prepare_datasets


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_mtp_eval(
    teacher,
    student,
    tokenizer,
    data_path: str,
    seed: int = 42,
    max_length: int = 128,
    batch_size: int = 32,
    log_path: str = "mtp_eval_results.json",
    device: str | None = None,
):
    _set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)

    loader = prepare_datasets(
        tokenizer=tokenizer,
        data_path=data_path,
        max_length=max_length,
        batch_size=batch_size,
    )

    t_acc = compute_masked_token_accuracy(teacher, tokenizer, loader, device)
    s_acc = compute_masked_token_accuracy(student, tokenizer, loader, device)
    agree = compare_student_teacher_masked_token_agreement(
        student, teacher, tokenizer, loader, device
    )
    t_res = compute_masked_token_perplexity(teacher, tokenizer, loader, device)
    s_res = compute_masked_token_perplexity(student, tokenizer, loader, device)
    kl, kl_tokens = masked_token_kl(student, teacher, loader, device)

    results = {
        "seed": seed,
        "teacher": getattr(teacher.config, "_name_or_path", "teacher"),
        "student": getattr(student.config, "_name_or_path", "student"),
        "teacher_accuracy": t_acc,
        "student_accuracy": s_acc,
        "agreement": agree["agreement"],
        "agreement_positions": agree["total"],
        "teacher_loss": t_res["loss"],
        "teacher_perplexity": t_res["perplexity"],
        "teacher_tokens": t_res["tokens"],
        "student_loss": s_res["loss"],
        "student_perplexity": s_res["perplexity"],
        "student_tokens": s_res["tokens"],
        "kl_teacher_student": kl,
        "kl_tokens": kl_tokens,
    }

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
