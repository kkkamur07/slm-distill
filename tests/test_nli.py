import json
import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import (
    compute_nli_accuracy,
    compute_nli_embedding_similarity,
    create_nli_classifier,
)


def load_nli_data(path: str = "data/hin/xnli_hi_test.json"):
    """
    Data is a JSON
    Returns:
        premises: list[str]
        hypotheses: list[str]
        labels: list[int] (contiguous ids 0..K-1)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "test" in data:
        records = data["test"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Unexpected JSON structure in {path}. "
            "Expected dict with key 'test' or a list."
        )

    premises = []
    hypotheses = []
    raw_labels = []

    for ex in records:
        if "premise" not in ex or "hypothesis" not in ex or "label" not in ex:
            raise ValueError(f"Missing keys in example: {ex}")
        premises.append(str(ex["premise"]))
        hypotheses.append(str(ex["hypothesis"]))
        raw_labels.append(int(ex["label"]))

    # Mapping raw label ids to contiguous 0..K-1 ids 
    unique_raw = sorted(set(raw_labels))
    raw_to_id = {raw: idx for idx, raw in enumerate(unique_raw)}
    labels = [raw_to_id[r] for r in raw_labels]

    return premises, hypotheses, labels, raw_to_id


def main():
    print("=" * 60)
    print("TESTING NLI EVALUATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    print("\n[2] Loading NLI data...")
    try:
        premises, hypotheses, labels, raw_to_id = load_nli_data(
            "data/hin/xnli_hi_test.json"
        )
        num_labels = len(raw_to_id)
        print(
            f"Loaded {len(premises)} examples with {num_labels} labels "
            f"(raw->id mapping: {raw_to_id})"
        )
    except Exception as e:
        print(f"Failed to load NLI data: {e}")
        import traceback

        print(traceback.format_exc())
        return

    print("\n[3] Creating NLI classifiers (teacher & student)...")
    try:
       
        teacher = create_nli_classifier(
            teacher_name,
            num_labels=num_labels,
            dropout=0.1,
        ).to(device)

        student = create_nli_classifier(
            student_name,
            num_labels=num_labels,
            dropout=0.1,
            subfolder="model",
        ).to(device)

        print("Models created successfully.")
    except Exception as e:
        print(
            f"Failed to create NLI classifiers (teacher='{teacher_name}', "
            f"student='{student_name}'): {e}"
        )
        import traceback

        print(traceback.format_exc())
        return

    print("\n[4] Evaluating teacher NLI accuracy...")
    try:
        teacher_metrics = compute_nli_accuracy(
            teacher,
            tokenizer,
            premises,
            hypotheses,
            labels,
            device,
        )
        print(f"  Teacher accuracy:   {teacher_metrics['accuracy']:.4f}")
        print(f"  macro-F1:           {teacher_metrics['macro_f1']:.4f}")
        print(f"  micro-F1:           {teacher_metrics['micro_f1']:.4f}")
        print(f"  precision (macro):  {teacher_metrics['precision']:.4f}")
        print(f"  recall (macro):     {teacher_metrics['recall']:.4f}")
    except Exception as e:
        print(f"Teacher NLI evaluation failed: {e}")
        import traceback

        print(traceback.format_exc())

    print("\n[5] Evaluating student NLI accuracy...")
    try:
        student_metrics = compute_nli_accuracy(
            student,
            tokenizer,
            premises,
            hypotheses,
            labels,
            device,
        )
        print(f"✓ Student accuracy:   {student_metrics['accuracy']:.4f}")
        print(f"  macro-F1:           {student_metrics['macro_f1']:.4f}")
        print(f"  micro-F1:           {student_metrics['micro_f1']:.4f}")
        print(f"  precision (macro):  {student_metrics['precision']:.4f}")
        print(f"  recall (macro):     {student_metrics['recall']:.4f}")
    except Exception as e:
        print(f"Student NLI evaluation failed: {e}")
        import traceback

        print(traceback.format_exc())

    print("\n[6] Testing student-teacher CLS embedding similarity on NLI pairs...")
    try:
        sim = compute_nli_embedding_similarity(
            student,
            teacher,
            tokenizer,
            premises,
            hypotheses,
            device,
        )
        print(f"Embedding similarity (CLS representations): {sim['similarity']:.4f}")
    except Exception as e:
        print(f"Embedding similarity evaluation failed: {e}")
        import traceback

        print(traceback.format_exc())

    print("\nNLI evaluations completed.\n")


if __name__ == "__main__":
    main()
