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
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.evals.mtp_perplexity_eval import (
    compute_masked_token_accuracy,
    compare_student_teacher_masked_token_agreement,
    compute_masked_token_perplexity,
    masked_token_kl,
)
from src.data.eval_prepare import prepare_datasets



def main():
    """Run masked-token evals for teacher, student_ce, and student_score (finetuning-style)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model_name = "FacebookAI/xlm-roberta-base"
    student_ce_model_name = "kkkamur07/hindi-xlm-roberta-33M"
    student_ce_subfolder = "model"
    student_score_state = Path("checkpoints/score_based_student/model_best.pt")

    print("Loading teacher and student models")
    teacher_base = AutoModelForMaskedLM.from_pretrained(teacher_model_name)
    student_ce_base = AutoModelForMaskedLM.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if student_score_state.exists():
        student_score_base = AutoModelForMaskedLM.from_config(
            copy.deepcopy(student_ce_base.config)
        )
        state = torch.load(student_score_state, map_location="cpu", weights_only=False)
        student_score_base.load_state_dict(state, strict=False)
        # Reuse the CE head to avoid random initialisation mismatches.
        if hasattr(student_score_base, "lm_head") and hasattr(student_ce_base, "lm_head"):
            student_score_base.lm_head.load_state_dict(
                student_ce_base.lm_head.state_dict(), strict=False
            )
        print(f"[Finetuning] Loaded student_score weights from {student_score_state}")
    else:
        raise FileNotFoundError(
            f"Expected student_score weights at {student_score_state} but not found."
        )
    shared_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher_base.to(device)
    student_ce_base.to(device)
    student_score_base.to(device)

    loader = prepare_datasets(
        tokenizer=shared_tokenizer,
        data_path="data/hin/data-99.parquet",
        max_length=126,
        batch_size=16,
    )

    print("calculating model mtps:")
    print("calculating teacher mtp:")
    results_teacher = compute_masked_token_accuracy(teacher_base, shared_tokenizer, loader, device)
    print(f"Teacher masked accuracy: {results_teacher:.4f}")
    print("calculating student_ce mtp:")
    results_student_ce = compute_masked_token_accuracy(student_ce_base, shared_tokenizer, loader, device)
    print(f"Student_CE masked accuracy: {results_student_ce:.4f}")
    print("calculating student_score mtp:")
    results_student_score = compute_masked_token_accuracy(student_score_base, shared_tokenizer, loader, device)
    print(f"Student_SCORE masked accuracy: {results_student_score:.4f}")


    print("calculating student - teacher mtp agreement:")
    print("calculating student_ce - teacher agreement:")
    agreement_student_ce = compare_student_teacher_masked_token_agreement(student_ce_base, teacher_base, shared_tokenizer, loader, device)
    print("calculating student_score - teacher agreement:")
    agreement_student_score = compare_student_teacher_masked_token_agreement(student_score_base, teacher_base, shared_tokenizer, loader, device)

    print(f"Student_CE agreement: {agreement_student_ce['agreement']:.4f}")
    print(f"Student_SCORE agreement: {agreement_student_score['agreement']:.4f}")

    print("calculating perplexities of base models")
    print("calculating teacher perlexity")
    teacher_perplexity = compute_masked_token_perplexity(teacher_base, shared_tokenizer, loader, device)
    print("calculating student_ce perlexity")
    student_ce_perplexity = compute_masked_token_perplexity(student_ce_base, shared_tokenizer, loader, device)
    print("calculating student_score perlexity")
    student_score_perplexity = compute_masked_token_perplexity(student_score_base, shared_tokenizer, loader, device)

    print(f"Teacher perplexity: {teacher_perplexity['perplexity']:.2f}")
    print(f"Student_CE perplexity: {student_ce_perplexity['perplexity']:.2f}")
    print(f"Student_SCORE perplexity: {student_score_perplexity['perplexity']:.2f}")


    print("calculating KL - metrics")
    print("calculating teacher - student_ce KL")
    kl_student_ce, kl_student_ce_tokens = masked_token_kl(student_ce_base, teacher_base, loader, device)
    print("calculating teacher - student_score KL")
    kl_student_score, kl_student_score_tokens = masked_token_kl(student_score_base, teacher_base, loader, device)

    print(f"Student_CE KL: {kl_student_ce:.4f} over {kl_student_ce_tokens} tokens")
    print(f"Student_SCORE KL: {kl_student_score:.4f} over {kl_student_score_tokens} tokens")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mtp_eval_{timestamp}.json"

    combined = {
        "teacher": {
            "name": teacher_model_name,
            "MTP": results_teacher,
            "Perplexity": teacher_perplexity,
        },
        "student_ce": {
            "name": student_ce_model_name,
            "MTP": results_student_ce,
            "MTP_agreement": agreement_student_ce,
            "Perplexity": student_ce_perplexity,
            "KL-divergence": kl_student_ce,
        },
        "student_score": {
            "name": str(student_score_state),
            "MTP": results_student_score,
            "MTP_agreement": agreement_student_score,
            "Perplexity": student_score_perplexity,
            "KL-divergence": kl_student_score,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"saved results as json here: {out_path}")


if __name__ == "__main__":
    main()
