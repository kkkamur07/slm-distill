import torch
import pandas as pd
from transformers import AutoTokenizer
from src.evals.NER_eval import create_ner_tagger, compute_ner_accuracy


def load_wikiann_hi_test(path: str = "data/hin/test-00000-of-00001.parquet"):
    df = pd.read_parquet(path)

    sentences = [list(map(str, toks)) for toks in df["tokens"].tolist()]
    labels = [[int(x) for x in tags] for tags in df["ner_tags"].tolist()]

    # infer label space as 0..max_id
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
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    print("\n[1] Loading WikiANN Hindi NER test data...")
    sentences, labels, label2id, id2label = load_wikiann_hi_test()
    num_labels = len(label2id)
    print(f"✓ Loaded {len(sentences)} sentences, num_labels={num_labels}")

    # optional debug subset
    DEBUG_MAX = None  # e.g. 200 for quick smoke test
    if DEBUG_MAX is not None and len(sentences) > DEBUG_MAX:
        sentences = sentences[:DEBUG_MAX]
        labels = labels[:DEBUG_MAX]
        print(f"[DEBUG] Using first {DEBUG_MAX} examples.")

    print("\n[2] Creating NER teacher & student models...")
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

    print("✓ Models created.")

    print("\n[3] Evaluating TEACHER on NER test set...")
    teacher_metrics = compute_ner_accuracy(
        teacher, tokenizer, sentences, labels, device
    )
    print("Teacher:", teacher_metrics)

    print("\n[4] Evaluating STUDENT on NER test set...")
    student_metrics = compute_ner_accuracy(
        student, tokenizer, sentences, labels, device
    )
    print("Student:", student_metrics)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
