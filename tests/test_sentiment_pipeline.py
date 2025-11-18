import os
import torch
from transformers import AutoTokenizer
from src.evals.sentiment_eval import (
    create_sentiment_classifier,
    compute_sentiment_accuracy,
    compute_sentiment_embedding_similarity,
)
from src.task_finetuning.sentiment_data import (
    load_sentiment_data,
    split_sentiment_data,
)
from src.task_finetuning.sentiment_train import train_sentiment_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # models (parallel to your NLI pipeline)
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    # tokenizer (shared, from teacher)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # ---- data loading ----
    texts, labels, label2id, id2label = load_sentiment_data(
        "data/hin/sentiment_metadata.tsv"
    )
    num_labels = len(label2id)
    print(f"Loaded {len(texts)} sentiment examples with labels: {label2id}")

    # --- debug: small subset like for NLI ---
    DEBUG_MAX = None  # e.g. 40 for a quick smoke test
    if DEBUG_MAX is not None and len(texts) > DEBUG_MAX:
        texts = texts[:DEBUG_MAX]
        labels = labels[:DEBUG_MAX]
        print(f"[DEBUG] Using only first {DEBUG_MAX} examples.")

    # split into train/dev/test
    (
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        test_texts,
        test_labels,
    ) = split_sentiment_data(texts, labels, 0.8, 0.1, 0.1, seed=42)

    print(
        f"Train: {len(train_texts)} | "
        f"Dev: {len(dev_texts)} | "
        f"Test: {len(test_texts)}"
    )

    # model
    teacher = create_sentiment_classifier(
        teacher_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.0, ### for now not used
    ).to(device)

    student = create_sentiment_classifier(
        student_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.0,
        subfolder="model",
    ).to(device)

    # finetuning 
    print("\nFine-tuning TEACHER on sentiment train/dev...")
    teacher_result = train_sentiment_model(
        model=teacher,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    teacher = teacher_result["model"]

    print("\nFine-tuning STUDENT on sentiment train/dev...")
    student_result = train_sentiment_model(
        model=student,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    student = student_result["model"]

    # evaluations
    print("\nFinal TEACHER performance on TEST:")
    teacher_metrics = compute_sentiment_accuracy(
        teacher,
        tokenizer,
        test_texts,
        test_labels,
        device,
    )
    print(
        f"  acc: {teacher_metrics['accuracy']:.4f}, "
        f"macro-F1: {teacher_metrics['macro_f1']:.4f}, "
        f"micro-F1: {teacher_metrics['micro_f1']:.4f}, "
        f"prec: {teacher_metrics['precision']:.4f}, "
        f"recall: {teacher_metrics['recall']:.4f}"
    )

    print("\nFinal STUDENT performance on TEST:")
    student_metrics = compute_sentiment_accuracy(
        student,
        tokenizer,
        test_texts,
        test_labels,
        device,
    )
    print(
        f"  acc: {student_metrics['accuracy']:.4f}, "
        f"macro-F1: {student_metrics['macro_f1']:.4f}, "
        f"micro-F1: {student_metrics['micro_f1']:.4f}, "
        f"prec: {student_metrics['precision']:.4f}, "
        f"recall: {student_metrics['recall']:.4f}"
    )

    # embedding sim / CLS
    print("\nTeacher–student CLS embedding similarity on TEST:")
    sim = compute_sentiment_embedding_similarity(
        student,
        teacher,
        tokenizer,
        test_texts,
        device,
    )
    print(f"  CLS cosine similarity: {sim['similarity']:.4f}")

    # saving changes:
    out_root = "checkpoints"
    os.makedirs(out_root, exist_ok=True)

    teacher_out = os.path.join(out_root, "sentiment_teacher_hi")
    student_out = os.path.join(out_root, "sentiment_student_hi")

    print(f"\nSaving fine-tuned TEACHER to '{teacher_out}'...")
    teacher.save_pretrained(teacher_out)
    tokenizer.save_pretrained(teacher_out)

    print(f"Saving fine-tuned STUDENT to '{student_out}'...")
    student.save_pretrained(student_out)

    print("\nSentiment pipeline completed.\n")


if __name__ == "__main__":
    main()
