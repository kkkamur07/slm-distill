# test_sentiment.py
import torch
from transformers import AutoTokenizer
from src.evals.sentiment_eval import (
    compute_sentiment_accuracy,
    create_sentiment_classifier,
)
from src.task_finetuning.sentiment_data import load_sentiment_csv


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # test split only
    texts, labels = load_sentiment_csv("data/hin/sentiment_hi_test.csv")

    teacher = create_sentiment_classifier(teacher_name, num_labels=3).to(device)
    student = create_sentiment_classifier(
        student_name, num_labels=3, subfolder="model"
    ).to(device)

    teacher_metrics = compute_sentiment_accuracy(
        teacher, tokenizer, texts, labels, device
    )
    student_metrics = compute_sentiment_accuracy(
        student, tokenizer, texts, labels, device
    )

    print("Teacher:", teacher_metrics)
    print("Student:", student_metrics)


if __name__ == "__main__":
    main()
