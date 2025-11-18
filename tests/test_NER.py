import torch
import pandas as pd
from transformers import AutoTokenizer
from src.evals.NER_eval import (
    create_ner_tagger,
    compute_ner_accuracy,
    compute_ner_embedding_similarity,
)


def load_wikiann_hi_test(
    path: str = "data/hin/test-00000-of-00001.parquet",
):
    """
    Load WikiANN Hindi test data from a parquet file.
    columns:
      - 'tokens': list[str]
      - 'ner_tags': list[int]
    """
    df = pd.read_parquet(path)

    sentences = [list(map(str, toks)) for toks in df["tokens"].tolist()]
    labels = [[int(x) for x in tags] for tags in df["ner_tags"].tolist()]

    
    all_ids = {lid for seq in labels for lid in seq}
    max_id = max(all_ids)
    num_labels = max_id + 1

    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    return sentences, labels, label2id, id2label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    print("\n[2] Loading WikiANN Hindi test split...")
    sentences, labels, label2id, id2label = load_wikiann_hi_test(
        "data/hin/test-00000-of-00001.parquet"
    )
    num_labels = len(label2id)
    print(f"✓ Loaded {len(sentences)} sentences, num_labels = {num_labels}")

    # testing on my laptop
    DEBUG_MAX = 100  
    if DEBUG_MAX is not None and len(sentences) > DEBUG_MAX:
        sentences = sentences[:DEBUG_MAX]
        labels = labels[:DEBUG_MAX]
        print(f"[DEBUG] Using only first {DEBUG_MAX} examples.")

    print("\n[3] Creating NER teacher & student models...")
    teacher = create_ner_tagger(
        teacher_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
    ).to(device)

    student = create_ner_tagger(
        student_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
        subfolder="model",
    ).to(device)

    print("✓ Models loaded.")

    print("\n[4] Evaluating TEACHER on NER test set...")
    teacher_metrics = compute_ner_accuracy(
        teacher,
        tokenizer,
        sentences,
        labels,
        device,
        batch_size=32,
        max_length=128,
    )
    print(
        f"TEACHER — acc: {teacher_metrics['accuracy']:.4f}, "
        f"macro-F1: {teacher_metrics['macro_f1']:.4f}, "
        f"micro-F1: {teacher_metrics['micro_f1']:.4f}, "
        f"prec: {teacher_metrics['precision']:.4f}, "
        f"recall: {teacher_metrics['recall']:.4f}"
    )

    print("\n[5] Evaluating STUDENT on NER test set...")
    student_metrics = compute_ner_accuracy(
        student,
        tokenizer,
        sentences,
        labels,
        device,
        batch_size=32,
        max_length=128,
    )
    print(
        f"STUDENT — acc: {student_metrics['accuracy']:.4f}, "
        f"macro-F1: {student_metrics['macro_f1']:.4f}, "
        f"micro-F1: {student_metrics['micro_f1']:.4f}, "
        f"prec: {student_metrics['precision']:.4f}, "
        f"recall: {student_metrics['recall']:.4f}"
    )

    print("\n[6] Teacher student embedding similarity (word-level)...")
    sim = compute_ner_embedding_similarity(
        student,
        teacher,
        tokenizer,
        sentences,
        device,
        batch_size=32,
        max_length=128,
    )
    print(f"CLS-ish word embedding similarity: {sim['similarity']:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
