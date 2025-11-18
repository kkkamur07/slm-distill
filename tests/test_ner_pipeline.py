import os
import torch
from transformers import AutoTokenizer
from src.evals.NER_eval import (
    create_ner_tagger,
    compute_ner_accuracy,
    compute_ner_embedding_similarity,
)
from src.task_finetuning.ner_data import load_wikiann_split
from src.task_finetuning.ner_train import train_ner_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # model names (same as your evaluation-only test_NER)
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    print("\n[2] Loading WikiANN Hindi splits...")
    train_sentences, train_labels = load_wikiann_split(
        "data/hin/train-00000-of-00001.parquet"
    )
    dev_sentences, dev_labels = load_wikiann_split(
        "data/hin/validation-00000-of-00001.parquet"
    )
    test_sentences, test_labels = load_wikiann_split(
        "data/hin/test-00000-of-00001.parquet"
    )

    # infer label space from train split
    all_ids = {lid for seq in train_labels for lid in seq}
    max_id = max(all_ids)
    num_labels = max_id + 1

    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    print(
        f"Train: {len(train_sentences)} | "
        f"Dev: {len(dev_sentences)} | "
        f"Test: {len(test_sentences)} | "
        f"num_labels = {num_labels}"
    )

    # Debug subset (for laptop testing)
    DEBUG_MAX = 100  # e.g. 100 for quick runs
    if DEBUG_MAX is not None:
        if len(train_sentences) > DEBUG_MAX:
            train_sentences = train_sentences[:DEBUG_MAX]
            train_labels = train_labels[:DEBUG_MAX]
        if len(dev_sentences) > DEBUG_MAX:
            dev_sentences = dev_sentences[:DEBUG_MAX]
            dev_labels = dev_labels[:DEBUG_MAX]
        if len(test_sentences) > DEBUG_MAX:
            test_sentences = test_sentences[:DEBUG_MAX]
            test_labels = test_labels[:DEBUG_MAX]
        print(
            f"[DEBUG] Using at most {DEBUG_MAX} sentences per split "
            f"for train/dev/test."
        )

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
        subfolder="model",  # matching your other tasks' student
    ).to(device)

    print("Models created.")

    # Finetuning 
    print("\n[5] Fine-tuning TEACHER on train/dev...")
    teacher_result = train_ner_model(
        model=teacher,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    teacher = teacher_result["model"]

    print("\n[6] Fine-tuning STUDENT on train/dev...")
    student_result = train_ner_model(
        model=student,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
    )
    student = student_result["model"]

    # Final evaluation on test
    print("\n[7] Final TEACHER performance on TEST:")
    teacher_metrics = compute_ner_accuracy(
        teacher,
        tokenizer,
        test_sentences,
        test_labels,
        device,
        batch_size=32,
        max_length=128,
    )
    print(
        f"  acc: {teacher_metrics['accuracy']:.4f}, "
        f"macro-F1: {teacher_metrics['macro_f1']:.4f}, "
        f"micro-F1: {teacher_metrics['micro_f1']:.4f}, "
        f"prec: {teacher_metrics['precision']:.4f}, "
        f"recall: {teacher_metrics['recall']:.4f}"
    )

    print("\n[8] Final STUDENT performance on TEST:")
    student_metrics = compute_ner_accuracy(
        student,
        tokenizer,
        test_sentences,
        test_labels,
        device,
        batch_size=32,
        max_length=128,
    )
    print(
        f"  acc: {student_metrics['accuracy']:.4f}, "
        f"macro-F1: {student_metrics['macro_f1']:.4f}, "
        f"micro-F1: {student_metrics['micro_f1']:.4f}, "
        f"prec: {student_metrics['precision']:.4f}, "
        f"recall: {student_metrics['recall']:.4f}"
    )

    # Teacher–student similarity
    print("\n[9] Teacher–student word-level embedding similarity on TEST:")
    sim = compute_ner_embedding_similarity(
        student,
        teacher,
        tokenizer,
        test_sentences,
        device,
        batch_size=32,
        max_length=128,
    )
    print(f"  similarity: {sim['similarity']:.4f}")

    # saving
    out_root = "checkpoints"
    os.makedirs(out_root, exist_ok=True)

    teacher_out = os.path.join(out_root, "ner_teacher_hi")
    student_out = os.path.join(out_root, "ner_student_hi")

    print(f"\n[10] Saving fine-tuned TEACHER to '{teacher_out}'...")
    teacher.save_pretrained(teacher_out)
    tokenizer.save_pretrained(teacher_out)

    print(f"Saving fine-tuned STUDENT to '{student_out}'...")
    student.save_pretrained(student_out)

    print("\nNER fine-tuning pipeline completed.\n")


if __name__ == "__main__":
    main()