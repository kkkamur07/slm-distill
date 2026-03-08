"""
Smoke test for NLI score-based finetuning (DPO variant).
Runs on a handful of batches to verify the pipeline. Sentiment is intentionally
omitted/commented to keep runtime low.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import torch
from transformers import AutoModel, AutoTokenizer

# Ensure local src/ on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evals.finetuning_DPO.nli_run_dpo import run_nli


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model_name = "FacebookAI/xlm-roberta-base"
    student_ce_model_name = "kkkamur07/hindi-xlm-roberta-33M"
    student_ce_subfolder = "model"
    student_score_state = Path("checkpoints/score_based_student/model_best.pt")

    lr_grid = [3e-4]
    num_epochs = 1
    batch_size = 4
    max_length = 64

    print(f"[Smoke DPO NLI] Using device: {device}")
    print("[Smoke DPO NLI] Loading backbones...")
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if not student_score_state.exists():
        raise FileNotFoundError(
            f"Missing score weights at {student_score_state}; place them or adjust the path."
        )
    student_score_base = AutoModel.from_config(copy.deepcopy(student_ce_base.config))
    student_score_base.load_state_dict(
        torch.load(student_score_state, map_location="cpu", weights_only=False),
        strict=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Sentiment smoke intentionally skipped for speed.
    # To enable later, call run_sentiment here similar to the sentiment smoke test.

    results = run_nli(
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        lr_grid=lr_grid,
        dropout=0.1,
        weight_decay=0.0,
        device=device,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=tokenizer,
        train_limit=8,
        dev_limit=4,
        test_limit=4,
    )

    print("\n[Smoke DPO NLI] Results (truncated):")
    print({k: {"metrics": v["metrics"], "best_lr": v["best_lr"]} for k, v in results.items()})


if __name__ == "__main__":
    main()
