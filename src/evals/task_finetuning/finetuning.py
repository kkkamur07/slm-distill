import copy
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.evals.task_finetuning.sentiment_run import run_sentiment
from src.evals.task_finetuning.nli_run import run_nli
from src.evals.task_finetuning.ner_run import run_ner


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Finetuning] Using device: {device}")

    teacher_model_name = "FacebookAI/xlm-roberta-base"
    student_ce_model_name = "kkkamur07/hindi-xlm-roberta-33M"
    student_ce_subfolder = "model"
    student_ce_state = Path("checkpoints/models/model_best_ce.pt")
    student_score_state = Path("checkpoints/models/model_best_score.pt")

    print("[Finetuning] Loading teacher and student backbones once...")
    teacher_base = AutoModel.from_pretrained(teacher_model_name)
    student_ce_base = AutoModel.from_pretrained(
        student_ce_model_name, subfolder=student_ce_subfolder
    )
    if not student_ce_state.exists():
        raise FileNotFoundError(
            f"Expected student_ce weights at {student_ce_state} but not found."
        )
    ce_state = torch.load(student_ce_state, map_location="cpu", weights_only=False)
    student_ce_base.load_state_dict(ce_state, strict=False)
    print(f"[Finetuning] Loaded student_ce weights from {student_ce_state}")

    if not student_score_state.exists():
        raise FileNotFoundError(
            f"Expected student_score weights at {student_score_state} but not found."
        )
    student_score_base = AutoModel.from_config(copy.deepcopy(student_ce_base.config))
    score_state = torch.load(student_score_state, map_location="cpu", weights_only=False)
    student_score_base.load_state_dict(score_state, strict=False)
    print(f"[Finetuning] Loaded student_score weights from {student_score_state}")

    shared_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    teacher_lr_grid = [8e-6, 1e-5, 3e-5, 5e-5]
    student_lr_grid = [1e-5, 4e-5, 7e-5, 1e-4, 4e-4, 7e-4]
    

    sentiment_results = run_sentiment(
        num_labels=3,
        teacher_num_epochs=8,
        student_num_epochs=16,
        batch_size=24,
        max_length=128,
        teacher_lr_grid=teacher_lr_grid,
        student_lr_grid=student_lr_grid,
        dropout=0.1,
        weight_decay=0.01,
        teacher_patience=2,
        student_patience=3,
        label_smoothing=0.00,
        warmup_steps=20,
        device=device,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=shared_tokenizer,
    )

    ner_results = run_ner(
        teacher_num_epochs=6,
        student_num_epochs=12,
        batch_size=24,
        max_length=128,
        device=device,
        teacher_lr_grid=teacher_lr_grid,
        student_lr_grid=student_lr_grid,
        dropout=0.1,
        weight_decay=0.01,
        teacher_patience=2,
        student_patience=3,
        warmup_ratio=0.06,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=shared_tokenizer,
    )

    nli_results = run_nli(
        teacher_num_epochs=4,
        student_num_epochs=8,
        batch_size = 24,
        max_length = 128,
        teacher_lr_grid=teacher_lr_grid,
        student_lr_grid=student_lr_grid,
        dropout=0.1,
        weight_decay=0.01,
        teacher_patience=2,
        student_patience=3,
        warmup_ratio=0.1,
        device=device,
        student_ce_subfolder=student_ce_subfolder,
        teacher_model=copy.deepcopy(teacher_base),
        student_ce_model=copy.deepcopy(student_ce_base),
        student_score_model=copy.deepcopy(student_score_base),
        tokenizer=shared_tokenizer,
    )

    def _compact(res: dict):
        return {
            k: {
                "metrics": v.get("metrics"),
                "best_lr": v.get("best_lr"),
                "best_dev_metrics": v.get("best_dev_metrics"),
            }
            for k, v in res.items()
        }

    print("\n=== SUMMARY (test metrics) ===")
    print("Sentiment:", _compact(sentiment_results))
    print("NER:", _compact(ner_results))
    print("NLI:", _compact(nli_results))

    # Persist training information (including per-batch loss where available)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Sentiment
    sent_path = results_dir / f"sentiment_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved Sentiment results to {sent_path}")

    # NER 
    ner_path = results_dir / f"ner_{timestamp}.json"
    with ner_path.open("w", encoding="utf-8") as f:
        json.dump(ner_results, f, indent=2)
    print(f"[Finetuning] Saved NER results to {ner_path}")

    # NLI
    nli_path = results_dir / f"nli_{timestamp}.json"
    with nli_path.open("w", encoding="utf-8") as f:
        json.dump(nli_results, f, indent=2)
    print(f"[Finetuning] Saved NLI results to {nli_path}")


if __name__ == "__main__":
    main()
