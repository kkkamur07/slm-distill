# finetuning.py

import torch
from src.task_finetuning.sentiment_run import run_sentiment
from src.task_finetuning.nli_run import run_nli
from src.task_finetuning.ner_run import run_ner


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Finetuning] Using device: {device}")

    sentiment_results = run_sentiment(device=device)
    nli_results = run_nli(device=device)
    ner_results = run_ner(device=device)

    print("\n=== SUMMARY (test metrics) ===")
    print("Sentiment:", sentiment_results)
    print("NLI:", nli_results)
    print("NER:", ner_results)


if __name__ == "__main__":
    main()
