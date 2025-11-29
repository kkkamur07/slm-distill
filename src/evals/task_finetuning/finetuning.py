import json
import time
from pathlib import Path

import torch

from src.evals.task_finetuning.sentiment_run import run_sentiment
from src.evals.task_finetuning.nli_run import run_nli
from src.evals.task_finetuning.ner_run import run_ner


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Finetuning] Using device: {device}")

    sentiment_results = run_sentiment(
        num_labels=3,
        num_epochs=8,
        batch_size=24,
        max_length=128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
        device=device,
    )

    nli_results = run_nli(
        num_epochs = 8,
        batch_size = 32,
        max_length = 128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
        device=device,
    )

    ner_results = run_ner(
        num_epochs=8,
        batch_size=24,
        max_length=128,
        device=device,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
    )

    print("\n=== SUMMARY (test metrics) ===")
    print("Sentiment:", sentiment_results)
    print("NLI:", nli_results)
    print("NER:", ner_results)

    # Persist sentiment and NLI training information (including per-batch loss where available)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Sentiment
    sent_path = results_dir / f"sentiment_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved Sentiment results to {sent_path}")

    # NLI
    nli_path = results_dir / f"nli_{timestamp}.json"
    with nli_path.open("w", encoding="utf-8") as f:
        json.dump(nli_results, f, indent=2)
    print(f"[Finetuning] Saved NLI results to {nli_path}")

    # NER 
    ner_path = results_dir / f"ner_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved NER results to {sent_path}")



if __name__ == "__main__":
    main()
