# test_sentiment_pipeline.py
import os
import torch
from transformers import AutoTokenizer
from src.evals.sentiment_eval import (
    create_sentiment_classifier,
    compute_sentiment_accuracy,
)
from src.task_finetuning.sentiment_data import load_sentiment_csv
from src.task_finetuning.sentiment_train import train_sentiment_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # fixed splits from Hindi CSVs
    train_texts, train_labels = load_sentiment_csv("data/hin/sentiment_hi_train.csv")
    val_texts, val_labels = load_sentiment_csv("data/hin/sentiment_hi_val.csv")
    test_texts, test_labels = load_sentiment_csv("data/hin/sentiment_hi_test.csv")

    teacher = create_sentiment_classifier(teacher_name, num_labels=3).to(device)
    student = create_sentiment_classifier(
        student_name, num_labels=3, subfolder="model"
    ).to(device)

    teacher_res = train_sentiment_model(
        teacher,
        tokenizer,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        device=device,
    )
    teacher = teacher_res["model"]

    student_res = train_sentiment_model(
        student,
        tokenizer,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        device=device,
    )
    student = student_res["model"]

    teacher_metrics = compute_sentiment_accuracy(
        teacher, tokenizer, test_texts, test_labels, device
    )
    student_metrics = compute_sentiment_accuracy(
        student, tokenizer, test_texts, test_labels, device
    )

    print("Teacher:", teacher_metrics)
    print("Student:", student_metrics)

    os.makedirs("checkpoints", exist_ok=True)
    teacher_out = "checkpoints/sentiment_teacher_hi"
    student_out = "checkpoints/sentiment_student_hi"

    teacher.save_pretrained(teacher_out)
    tokenizer.save_pretrained(teacher_out)
    student.save_pretrained(student_out)


if __name__ == "__main__":
    main()
