import torch
from transformers import AutoTokenizer

from src.evals.nli_eval import (
    create_nli_classifier,
    compute_nli_accuracy,
    compute_nli_embedding_similarity,
)
from src.task_finetuning.nli_data import load_nli_split
from src.task_finetuning.nli_train import train_nli_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_path = "data/hin/xnli_hi_train.json"
    dev_path = "data/hin/xnli_hi_dev.json"
    test_path = "data/hin/xnli_hi_test.json"

    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    # 1. Tokenizer (single, from teacher)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)

    # 2. Data
    train_prem, train_hyp, train_labels, raw_to_id = load_nli_split(
        train_path, split_key="train"
    )
    dev_prem, dev_hyp, dev_labels, _ = load_nli_split(
        dev_path, split_key="dev", raw_to_id=raw_to_id
    )
    test_prem, test_hyp, test_labels, _ = load_nli_split(
        test_path, split_key="test", raw_to_id=raw_to_id
    )

    ### testing on my laptop only here:
    train_prem, train_hyp, train_labels = train_prem[:10], train_hyp[:10], train_labels[:10]
    dev_prem,   dev_hyp,   dev_labels   = dev_prem[:5],   dev_hyp[:5],   dev_labels[:5]
    test_prem,  test_hyp,  test_labels  = test_prem[:5],  test_hyp[:5],  test_labels[:5]


    num_labels = len(raw_to_id)
    print(
        f"Train: {len(train_prem)}, Dev: {len(dev_prem)}, Test: {len(test_prem)}, "
        f"num_labels = {num_labels}, mapping = {raw_to_id}"
    )

    # 3. create nli classifier
    teacher = create_nli_classifier(
        teacher_name,
        num_labels=num_labels,
        dropout=0.0, #can be included
    ).to(device)

    student = create_nli_classifier(
        student_name,
        num_labels=num_labels,
        dropout=0.0, 
        subfolder="model",
    ).to(device)


    # Fine-tune models
    teacher_result = train_nli_model(
        model=teacher,
        tokenizer=tokenizer,
        train_premises=train_prem,
        train_hypotheses=train_hyp,
        train_labels=train_labels,
        dev_premises=dev_prem,
        dev_hypotheses=dev_hyp,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
    )
    teacher = teacher_result["model"]

    student_result = train_nli_model(
        model=student,
        tokenizer=tokenizer,
        train_premises=train_prem,
        train_hypotheses=train_hyp,
        train_labels=train_labels,
        dev_premises=dev_prem,
        dev_hypotheses=dev_hyp,
        dev_labels=dev_labels,
        device=device,
        num_epochs=3,
    )
    student = student_result["model"]

    #Evaluation
    teacher_metrics = compute_nli_accuracy(
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )
    student_metrics = compute_nli_accuracy(
        student,
        tokenizer,
        test_prem,
        test_hyp,
        test_labels,
        device,
    )

    print(
        f"Fine-tuned teacher — acc: {teacher_metrics['accuracy']:.4f}, "
        f"macro-F1: {teacher_metrics['macro_f1']:.4f}"
    )
    print(
        f"Fine-tuned student — acc: {student_metrics['accuracy']:.4f}, "
        f"macro-F1: {student_metrics['macro_f1']:.4f}"
    )

    # embedding sim
    sim = compute_nli_embedding_similarity(
        student,
        teacher,
        tokenizer,
        test_prem,
        test_hyp,
        device,
    )
    print(f"Teacher–student CLS similarity on test: {sim['similarity']:.4f}")


if __name__ == "__main__":
    main()