import torch
from transformers import AutoTokenizer

from src.evals.ner_eval import compute_ner_accuracy
from src.evals.task_finetuning.ner_data import load_wikiann_split
from src.evals.task_finetuning.ner_train import create_ner_tagger, train_ner_model


def run_ner(
    num_epochs: int,
    batch_size: int,
    max_length: int,
    device: str | None,
    weight_decay: float,
    early_stopping_patience: int | None,
    lr_grid: list[float] | None,
    min_delta: float = 1e-5, ## early stopping
    train_path: str = "data/hin/train-00000-of-00001.parquet",
    dev_path: str = "data/hin/validation-00000-of-00001.parquet",
    test_path: str = "data/hin/test-00000-of-00001.parquet",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str | None = "model",  # None if not needed
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
        raise ValueError("lr grid for NER not provided")
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
        best_history = None

        for lr in lr_grid:
            print(f"\n[NER] Fine-tuning '{base_model_name}' with lr={lr:.1e}...")
            model = create_ner_tagger(
                base_model_name=base_model_name,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                dropout=0.1,
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
            history = res["history"]
            last_dev = history[-1]["dev_metrics"]
            dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
            print(f"[NER] {base_model_name}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr
                best_history = history

        if best_model is None:
            raise RuntimeError(f"[NER] Grid search failed for {base_model_name}")

        print(
            f"[NER] Best lr for {base_model_name}: {best_lr:.1e} "
            f"(dev acc={best_acc:.4f})"
        )
        return best_model, best_lr, best_history

    # Teacher grid search
    teacher, best_lr_teacher, teacher_history = grid_search_model(
        base_model_name=teacher_model_name,
        subfolder=None,
    )

    # Student grid search
    student, best_lr_student, student_history = grid_search_model(
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
        "teacher": {
            "metrics": teacher_metrics,
            "best_lr": best_lr_teacher,
            "history": teacher_history,
        },
        "student": {
            "metrics": student_metrics,
            "best_lr": best_lr_student,
            "history": student_history,
        },
    }
