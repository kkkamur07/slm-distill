import copy
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.evals.finetuning_DPO.sentiment_run_dpo import run_sentiment
from src.evals.finetuning_DPO.nli_run_dpo import run_nli


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Finetuning] Using device: {device}")

    teacher_model_name = "FacebookAI/xlm-roberta-base"
    student_ce_model_name = "kkkamur07/hindi-xlm-roberta-33M"
    student_ce_subfolder = "model"
    student_score_state = Path("checkpoints/score_based_student/model_best.pt")

    print("[Finetuning] Loading teacher and student backbones once...")
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if student_score_state.exists():
        student_score_base = AutoModel.from_config(
            copy.deepcopy(student_ce_base.config)
        )
        state = torch.load(student_score_state, map_location="cpu", weights_only=False)
        student_score_base.load_state_dict(state, strict=False)
        print(f"[Finetuning] Loaded student_score weights from {student_score_state}")
    else:
        raise FileNotFoundError(
            f"Expected student_score weights at {student_score_state} but not found."
        )
    shared_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    sentiment_results = run_sentiment(
        num_labels=3,
        num_epochs=3,
        batch_size=32,
        max_length=128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
        device=device,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=shared_tokenizer,
    )

    nli_results = run_nli(
        num_epochs=8,
        batch_size=32,
        max_length=128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        device=device,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=shared_tokenizer,
    )

    print("\n=== SUMMARY (test metrics) ===")
    print("Sentiment:", sentiment_results)
    print("NLI:", nli_results)

    # Persist training information (including per-batch loss where available)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Sentiment
    sent_path = results_dir / f"sentiment_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved Sentiment results to {sent_path}")

    nli_path = results_dir / f"nli_{timestamp}.json"
    with nli_path.open("w", encoding="utf-8") as f:
        json.dump(nli_results, f, indent=2)
    print(f"[Finetuning] Saved NLI results to {nli_path}")



if __name__ == "__main__":
    main()
