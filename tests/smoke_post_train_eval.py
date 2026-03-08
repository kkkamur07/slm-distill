"""
Tiny smoke test for post-train masked-token evals (fixed config, 2-3 batches).
Run with: uv run -m tests.smoke_post_train_eval
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Ensure local src/ is on path when run as `python -m tests.smoke_post_train_eval`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.eval_prepare import prepare_datasets
from src.evals.mtp_perplexity_eval import (
    compute_masked_token_accuracy,
    compare_student_teacher_masked_token_agreement,
    compute_masked_token_perplexity,
    masked_token_kl,
)


def _load_score_student(base_config, weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Score student weights not found at {weights_path}")
    model = AutoModelForMaskedLM.from_config(copy.deepcopy(base_config))
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_id = "FacebookAI/xlm-roberta-base"
    student_ce_id = "kkkamur07/hindi-xlm-roberta-33M"
    student_ce_subfolder = "model"
    student_score_path = Path("checkpoints/score_based_student/model_best.pt")
    data_path = "data/hin/data-99.parquet"
    max_length = 64
    batch_size = 4
    max_batches = 3

    print(f"[Smoke] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_id, use_fast=True)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_id).to(device)
    ce_kwargs = {"subfolder": student_ce_subfolder} if student_ce_subfolder else {}
    student_ce = AutoModelForMaskedLM.from_pretrained(student_ce_id, **ce_kwargs).to(device)
    student_score = _load_score_student(
        base_config=student_ce.config,
        weights_path=student_score_path,
    ).to(device)
    if hasattr(student_score, "lm_head") and hasattr(student_ce, "lm_head"):
        student_score.lm_head.load_state_dict(student_ce.lm_head.state_dict(), strict=False)

    print("[Smoke] Preparing data loader (limited batches)...")
    loader_trimmed = []
    for i, batch in enumerate(
        prepare_datasets(
            tokenizer=tokenizer,
            data_path=data_path,
            max_length=max_length,
            batch_size=batch_size,
        )
    ):
        if i >= max_batches:
            break
        loader_trimmed.append(batch)
    if not loader_trimmed:
        print("[Smoke] No batches loaded; check data path.")
        return

    print("[Smoke] Masked-token accuracy...")
    acc_teacher = compute_masked_token_accuracy(teacher, tokenizer, loader_trimmed, device)
    acc_ce = compute_masked_token_accuracy(student_ce, tokenizer, loader_trimmed, device)
    acc_score = compute_masked_token_accuracy(student_score, tokenizer, loader_trimmed, device)
    print(f"  Teacher acc: {acc_teacher:.4f}")
    print(f"  Student_CE acc: {acc_ce:.4f}")
    print(f"  Student_SCORE acc: {acc_score:.4f}")

    print("[Smoke] Student-teacher agreement...")
    agree_ce = compare_student_teacher_masked_token_agreement(
        student_ce, teacher, tokenizer, loader_trimmed, device
    )
    agree_score = compare_student_teacher_masked_token_agreement(
        student_score, teacher, tokenizer, loader_trimmed, device
    )
    print(f"  Student_CE agreement: {agree_ce['agreement']:.4f} (tokens={agree_ce['total']})")
    print(f"  Student_SCORE agreement: {agree_score['agreement']:.4f} (tokens={agree_score['total']})")

    print("[Smoke] Perplexities...")
    ppl_teacher = compute_masked_token_perplexity(teacher, tokenizer, loader_trimmed, device)
    ppl_ce = compute_masked_token_perplexity(student_ce, tokenizer, loader_trimmed, device)
    ppl_score = compute_masked_token_perplexity(student_score, tokenizer, loader_trimmed, device)
    print(f"  Teacher ppl: {ppl_teacher['perplexity']:.2f}")
    print(f"  Student_CE ppl: {ppl_ce['perplexity']:.2f}")
    print(f"  Student_SCORE ppl: {ppl_score['perplexity']:.2f}")

    print("[Smoke] KL divergence...")
    kl_ce, kl_ce_tokens = masked_token_kl(student_ce, teacher, loader_trimmed, device)
    kl_score, kl_score_tokens = masked_token_kl(student_score, teacher, loader_trimmed, device)
    print(f"  Student_CE KL: {kl_ce:.4f} (tokens={kl_ce_tokens})")
    print(f"  Student_SCORE KL: {kl_score:.4f} (tokens={kl_score_tokens})")


if __name__ == "__main__":
    main()
