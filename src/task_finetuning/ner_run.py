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
    learning_rate: float = 2e-5,  # kept for backward compatibility
    max_length: int = 128,
    device: str | None = None,
    weight_decay: float = 0.01,
    early_stopping_patience: int | None = 2,
    min_delta: float = 0.0,
    lr_grid: list[float] | None = None,
):
    """
    Fine-tune teacher and student on WikiANN Hindi NER and evaluate on test.

    Adds:
      - weight decay for AdamW
      - early stopping on dev accuracy
      - learning-rate grid search for BOTH teacher and student

    Returns:
        {
            "teacher": {"metrics": {...}, "best_lr": float},
            "student": {"metrics": {...}, "best_lr": float},
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NER] Using device: {device}")

    # LR grid default, similar to NLI and sentiment
    if lr_grid is None:
        lr_grid = [2e-5, 5e-5, 1e-4]

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Load dataset splits
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

    def grid_search_model(base_model_name: str, subfolder: str | None = None):
        best_acc = -1.0
        best_model = None
        best_lr = None

        for lr in lr_grid:
            print(f"\n[NER] Fine-tuning '{base_model_name}' with lr={lr:.1e}...")
            model = create_ner_tagger(
                base_model_name=base_model_name,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                dropout=0.1,  # fixed dropout as requested
                subfolder=subfolder,
            ).to(device)

            res = train_ner_model(
                model=model,
                tokenizer=tokenizer,
                train_sentences=train_sentences,
                train_labels=train_labels,
                dev_sentences=dev_sentences,
                dev_labels=dev_labels,
                device=device,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                ignore_index=-100,
                eval_on_dev=True,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
            )
            model = res["model"]
            last_dev = res["history"][-1]["dev_metrics"]
            dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
            print(f"[NER] {base_model_name}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr

        if best_model is None:
            raise RuntimeError(f"[NER] Grid search failed for {base_model_name}")

        print(
            f"[NER] Best lr for {base_model_name}: {best_lr:.1e} "
            f"(dev acc={best_acc:.4f})"
        )
        return best_model, best_lr

    # Teacher grid search
    teacher, best_lr_teacher = grid_search_model(
        base_model_name=teacher_model_name,
        subfolder=None,
    )

    # Student grid search
    student, best_lr_student = grid_search_model(
        base_model_name=student_model_name,
        subfolder=student_subfolder,
    )

    # Final test evaluation
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
        "teacher": {"metrics": teacher_metrics, "best_lr": best_lr_teacher},
        "student": {"metrics": student_metrics, "best_lr": best_lr_student},
    }


if __name__ == "__main__":
    run_ner()
