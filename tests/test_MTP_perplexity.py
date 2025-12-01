import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.evals.mtp_perplexity_eval import (
    compute_masked_token_accuracy,
    compare_student_teacher_masked_token_agreement,
    compute_masked_token_perplexity,
)

from src.data.nativeSLM import prepare_datasets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # prepare_datasets returns a tuple, not a dict.
    # We only need the eval_loader here.
    train_loader, eval_loader, *rest = prepare_datasets(
        tokenizer=tokenizer,
        data_path="data/hin/data-99.parquet",
        max_length=128,
        batch_size=32,
        train_split=0.001,
    )

    teacher = AutoModelForMaskedLM.from_pretrained(teacher_name).to(device)
    student = AutoModelForMaskedLM.from_pretrained(
        student_name, subfolder="model"
    ).to(device)

    # ---- Accuracy ----
    print("\n[Masked-token accuracy] TEACHER:")
    teacher_acc = compute_masked_token_accuracy(
        teacher, tokenizer, eval_loader, device
    )
    print(f"  accuracy = {teacher_acc:.4f}")

    print("\n[Masked-token accuracy] STUDENT:")
    student_acc = compute_masked_token_accuracy(
        student, tokenizer, eval_loader, device
    )
    print(f"  accuracy = {student_acc:.4f}")

    # ---- Agreement ----
    print("\n[Student–Teacher agreement on masked tokens]:")
    agreement = compare_student_teacher_masked_token_agreement(
        student, teacher, tokenizer, eval_loader, device
    )
    print(
        f"  agreement = {agreement['agreement']:.4f} "
        f"(over {agreement['total']} masked positions)"
    )

    # ---- Perplexity ----
    print("\n[Masked-token perplexity] TEACHER:")
    t_res = compute_masked_token_perplexity(
        teacher, tokenizer, eval_loader, device
    )
    print(
        f"  loss = {t_res['loss']:.4f}, "
        f"perplexity = {t_res['perplexity']:.4f}, "
        f"tokens = {t_res['tokens']}"
    )

    print("\n[Masked-token perplexity] STUDENT:")
    s_res = compute_masked_token_perplexity(
        student, tokenizer, eval_loader, device
    )
    print(
        f"  loss = {s_res['loss']:.4f}, "
        f"perplexity = {s_res['perplexity']:.4f}, "
        f"tokens = {s_res['tokens']}"
    )

    print("\nMTP + perplexity evaluation completed.\n")


if __name__ == "__main__":
    main()
