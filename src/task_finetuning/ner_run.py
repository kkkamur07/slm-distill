import torch
from transformers import AutoTokenizer

from src.evals.NER_eval import create_ner_tagger, compute_ner_accuracy
from src.task_finetuning.ner_data import load_wikiann_split
from src.task_finetuning.ner_train import train_ner_model


def run_ner(
    train_path: str = "data/hin/train-00000-of-00001.parquet",
    dev_path: str = "data/hin/validation-00000-of-00001.parquet",
    test_path: str = "data/hin/test-00000-of-00001.parquet",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str | None = "model",  # None if not needed
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    device: str | None = None,
):
    """
    Fine-tune teacher and student on WikiANN Hindi NER and evaluate on test.

    Returns:
        {
            "teacher": {"metrics": {...}},
            "student": {"metrics": {...}},
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NER] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Data
    train_sentences, train_labels = load_wikiann_split(train_path)
    dev_sentences, dev_labels = load_wikiann_split(dev_path)
    test_sentences, test_labels = load_wikiann_split(test_path)

    # Infer label space from train labels
    all_ids = {lid for seq in train_labels for lid in seq}
    num_labels = max(all_ids) + 1
    label2id = {str(i): i for i in range(num_labels)}
    id2label = {i: str(i) for i in range(num_labels)}

    print(
        f"[NER] train={len(train_sentences)}, dev={len(dev_sentences)}, "
        f"test={len(test_sentences)}, num_labels={num_labels}"
    )

    # Models (starting from HF / KD checkpoints)
    teacher = create_ner_tagger(
        teacher_model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
    ).to(device)

    student = create_ner_tagger(
        student_model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout=0.1,
        subfolder=student_subfolder,
    ).to(device)

    # Fine-tune teacher
    print("[NER] Fine-tuning TEACHER...")
    teacher_res = train_ner_model(
        model=teacher,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    teacher = teacher_res["model"]

    # Fine-tune student
    print("[NER] Fine-tuning STUDENT...")
    student_res = train_ner_model(
        model=student,
        tokenizer=tokenizer,
        train_sentences=train_sentences,
        train_labels=train_labels,
        dev_sentences=dev_sentences,
        dev_labels=dev_labels,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    student = student_res["model"]

    # Evaluate on test
    print("[NER] Evaluating TEACHER on test...")
    teacher_metrics = compute_ner_accuracy(
        teacher,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NER] TEACHER metrics:", teacher_metrics)

    print("[NER] Evaluating STUDENT on test...")
    student_metrics = compute_ner_accuracy(
        student,
        tokenizer,
        test_sentences,
        test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("[NER] STUDENT metrics:", student_metrics)

    return {
        "teacher": {"metrics": teacher_metrics},
        "student": {"metrics": student_metrics},
    }


if __name__ == "__main__":
    run_ner()
