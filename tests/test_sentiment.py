import torch
from transformers import AutoTokenizer
import pandas as pd
from src.evals.sentiment_eval import (
    compute_sentiment_accuracy,
    compute_sentiment_embedding_similarity,
    create_sentiment_classifier,
)


def load_sentiment_data(data_path: str = "data/hin/sentiment_metadata.tsv"):
    """
    Data is in a TSV file for now 
    Returns:
        texts: list of review strings
        labels: list of integer ids
        label2id: mapping from label string -> id
        id2label: mapping from id -> label string
    """
    df = pd.read_csv(data_path, sep="\t")

    # Drop rows with missing labels 
    df = df.dropna(subset=["LABEL"])

    texts = df["REVIEW"].astype(str).tolist()
    label_strings = df["LABEL"].astype(str).tolist()

    # Build consistent label mapping
    unique_labels = sorted(set(label_strings))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    labels = [label2id[lbl] for lbl in label_strings]

    return texts, labels, label2id, id2label


def main():
    print("=" * 60)
    print("TESTING SENTIMENT EVALUATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Base model ids for teacher and student
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    print("\n[2] Loading sentiment data...")
    try:
        texts, labels, label2id, id2label = load_sentiment_data(
            "data/hin/sentiment_metadata.tsv"
        )
        num_labels = len(label2id)
        print(f"✓ Loaded {len(texts)} examples with {num_labels} labels: {label2id}")
    except Exception as e:
        print(f"Failed to load sentiment data: {e}")
        import traceback
        print(traceback.format_exc())
        return

    print("\n[3] Creating sentiment classifiers (teacher & student)...")
    try:
        
        teacher = create_sentiment_classifier(
            teacher_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            dropout=0.1,
        ).to(device)

        
        student = create_sentiment_classifier(
            student_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            dropout=0.1,
            subfolder="model",
        ).to(device)

        print("✓ Models created successfully.")
    except Exception as e:
        print(
            f"Failed to create sentiment classifiers (teacher='{teacher_name}', "
            f"student='{student_name}'): {e}"
        )
        import traceback
        print(traceback.format_exc())
        return

    print("\n[4] Evaluating teacher sentiment accuracy...")
    try:
        teacher_metrics = compute_sentiment_accuracy(
            teacher, tokenizer, texts, labels, device
        )
        print(f"✓ Teacher accuracy:   {teacher_metrics['accuracy']:.4f}")
        print(f"  macro-F1:           {teacher_metrics['macro_f1']:.4f}")
        print(f"  micro-F1:           {teacher_metrics['micro_f1']:.4f}")
        print(f"  precision (macro):  {teacher_metrics['precision']:.4f}")
        print(f"  recall (macro):     {teacher_metrics['recall']:.4f}")
    except Exception as e:
        print(f"Teacher sentiment evaluation failed: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n[5] Evaluating student sentiment accuracy...")
    try:
        student_metrics = compute_sentiment_accuracy(
            student, tokenizer, texts, labels, device
        )
        print(f"✓ Student accuracy:   {student_metrics['accuracy']:.4f}")
        print(f"  macro-F1:           {student_metrics['macro_f1']:.4f}")
        print(f"  micro-F1:           {student_metrics['micro_f1']:.4f}")
        print(f"  precision (macro):  {student_metrics['precision']:.4f}")
        print(f"  recall (macro):     {student_metrics['recall']:.4f}")
    except Exception as e:
        print(f"Student sentiment evaluation failed: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n[6] Testing student-teacher embedding similarity (CLS)...")
    try:
        sim = compute_sentiment_embedding_similarity(
            student, teacher, tokenizer, texts, device
        )
        print(f"✓ Embedding similarity (CLS representations): {sim['similarity']:.4f}")
    except Exception as e:
        print(f"Embedding similarity evaluation failed: {e}")
        import traceback
        print(traceback.format_exc())

    print("\nSentiment evaluations completed.\n")


if __name__ == "__main__":
    main()
