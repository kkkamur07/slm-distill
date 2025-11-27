import os
import torch
from transformers import AutoTokenizer
from src.evals.NER_eval import create_ner_tagger, compute_ner_accuracy
from src.task_finetuning.ner_data import load_wikiann_split
from src.task_finetuning.ner_train import train_ner_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    train_path = "data/hin/train-00000-of-00001.parquet"
    dev_path = "data/hin/validation-00000-of-00001.parquet"
    test_path = "data/hin/test-00000-of-00001.parquet"

    print("\n[1] Loading WikiANN Hindi splits...")
    train_sentences, train_labels = load_wikiann_split(train_path)
    dev_sentences, dev_labels = load_wikiann_split(dev_path)
    test_sentences, test_labels = load_wikiann_split(test_path)

    # infer label space 0..max_id
    all_ids = {lid for seq in train_labels for lid in seq}
    max_id = max(all_ids)
    num_labels = max_id + 1
    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    print(
        f"Train={len(train_sentences)}, Dev={len(dev_sentences)}, "
        f"Test={len(test_sentences)}, num_labels={num_labels}"
    )

    # optional debug subsampling
    DEBUG_MAX = None  # e.g. 200 for quick run
    if DEBUG_MAX is not None:
        train_sentences = train_sentences[:DEBUG_MAX]
        train_labels = train_labels[:DEBUG_MAX]
        dev_sentences = dev_sentences[:DEBUG_MAX]
        dev_labels = dev_labels[:DEBUG_MAX]
        test_sentences = test_sentences[:DEBUG_MAX]
        test_labels = test_labels[:DEBUG_MAX]
        print(f"[DEBUG] Using first {DEBUG_MAX} examples per split.")

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

    print("Models created.")

    print("\n[3] Fine-tuning TEACHER on train/dev...")
    teacher_res = train_ner_model(
        model=teacher,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
    )
    teacher = teacher_res["model"]

    print("\n[4] Fine-tuning STUDENT on train/dev...")
    student_res = train_ner_model(
        model=student,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
    )
    student = student_res["model"]

    print("\n[5] Evaluating TEACHER on test set...")
    teacher_metrics = compute_ner_accuracy(
        teacher, tokenizer, test_sentences, test_labels, device
    )
    print("Teacher:", teacher_metrics)

    print("\n[6] Evaluating STUDENT on test set...")
    student_metrics = compute_ner_accuracy(
        student, tokenizer, test_sentences, test_labels, device
    )
    print("Student:", student_metrics)

    out_root = "checkpoints"
    os.makedirs(out_root, exist_ok=True)
    teacher_out = os.path.join(out_root, "ner_teacher_hi")
    student_out = os.path.join(out_root, "ner_student_hi")

    print(f"\n[7] Saving TEACHER to '{teacher_out}'...")
    teacher.save_pretrained(teacher_out)
    tokenizer.save_pretrained(teacher_out)

    print(f"Saving STUDENT to '{student_out}'...")
    student.save_pretrained(student_out)

    print("\nNER fine-tuning pipeline completed.\n")


if __name__ == "__main__":
    main()
