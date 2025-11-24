import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import create_nli_classifier, compute_nli_accuracy
from src.task_finetuning.nli_data import load_nli_split
from src.task_finetuning.nli_train import train_nli_model


def run_nli(
    train_path: str = "data/hin/xnli_hi_train.json",
    dev_path: str = "data/hin/xnli_hi_dev.json",
    test_path: str = "data/hin/xnli_hi_test.json",
    teacher_model_name: str = "FacebookAI/xlm-roberta-base",
    student_model_name: str = "kkkamur07/hindi-xlm-roberta-33M",
    student_subfolder: str | None = "model",
    num_epochs: int = 5,
    batch_size: int = 32,
    lr_grid: list[float] | None = None,
    max_length: int = 128,
    dropout = 0.1,
    weight_decay: float = 0.01,
    early_stopping_patience: int | None = 2,
    min_delta: float = 0.0,
    device: str | None = None,
):
    """
    NLI fine-tuning with LR grid search for BOTH teacher and student.
    Uses dev accuracy to pick the best LR.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[NLI] Using device: {device}")

    if lr_grid is None:
        lr_grid = [2e-5, 5e-5, 1e-4]

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=True)

    # Load splits with a shared label mapping
    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(train_path)
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(dev_path, raw_to_id=raw_to_id)
    test_prem, test_hyp, test_labels, _ = load_nli_split(test_path, raw_to_id=raw_to_id)
    num_labels = len(raw_to_id)

    print(
        f"[NLI] train={len(train_prem)}, dev={len(dev_prem)}, "
        f"test={len(test_prem)}, num_labels={num_labels}"
    )

    # ---- helper: grid search for one HF model id ----
    def grid_search_model(base_model_name: str, subfolder: str | None = None):
        best_acc = -1.0
        best_model = None
        best_lr = None

        for lr in lr_grid:
            print(f"\n[NLI] Fine-tuning '{base_model_name}' with lr={lr:.1e}...")
            model = create_nli_classifier(
                base_model_name,
                num_labels=num_labels,
                dropout=dropout,
                subfolder=subfolder,
            ).to(device)

            res = train_nli_model(
                model=model,
                tokenizer=tokenizer,
                train_premises=train_prem,
                train_hypotheses=train_hyp,
                train_labels=train_labels,
                dev_premises=dev_prem,
                dev_hypotheses=dev_hyp,
                dev_labels=dev_labels,
                device=device,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                max_length=max_length,
                eval_on_dev=True,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                min_delta=min_delta,
            )
            model = res["model"]
            last_dev = res["history"][-1]["dev_metrics"]
            dev_acc = last_dev["accuracy"] if last_dev is not None else 0.0
            print(f"[NLI] {base_model_name}, lr={lr:.1e} dev acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model = model
                best_lr = lr

        if best_model is None:
            raise RuntimeError(f"Grid search failed for model {base_model_name}")

        print(
            f"[NLI] Best lr for {base_model_name}: {best_lr:.1e} "
            f"(dev acc={best_acc:.4f})"
        )
        return best_model, best_lr

    # ---- teacher grid search ----
    teacher, best_lr_teacher = grid_search_model(
        base_model_name=teacher_model_name,
        subfolder=None,
    )

    # ---- student grid search ----
    student, best_lr_student = grid_search_model(
        base_model_name=student_model_name,
        subfolder=student_subfolder,
    )

    # ---- test eval ----
    print("\n[NLI] Evaluating TEACHER on test...")
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Evaluating STUDENT on test...")
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\n[NLI] Final test results:")
    print("  Teacher:", teacher_metrics, f"(best lr={best_lr_teacher:.1e})")
    print("  Student:", student_metrics, f"(best lr={best_lr_student:.1e})")

    return {
        "teacher": {"metrics": teacher_metrics, "best_lr": best_lr_teacher},
        "student": {"metrics": student_metrics, "best_lr": best_lr_student},
    }