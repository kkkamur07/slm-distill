import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, XLMRobertaForMaskedLM
from torch.utils.data import Subset, DataLoader

from src.evals.cosine_sim_eval import (
    compute_cosine_similarity_ground_truth,
    compute_cosine_similarity_student_teacher,
)
from src.data.data import prepare_datasets



def main():
    print("=" * 60)
    print("TESTING SEMANTIC EVALUATIONS (Cosine Similarity)")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1) Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_name = "FacebookAI/xlm-roberta-base"
    student_name = "kkkamur07/hindi-xlm-roberta-33M"

    print("\n[1] Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = XLMRobertaForMaskedLM.from_pretrained(teacher_name).to(device)
    
    student_cfg = AutoConfig.from_pretrained(student_name, subfolder="model")
    student = XLMRobertaForMaskedLM.from_pretrained(
        student_name,
        subfolder="model",
        config=student_cfg,
        token=True,
    ).to(device)

    # 2) Data 
    print("\n[2] Loading and preparing dataset...")
    train_loader, eval_loader = prepare_datasets(
        data_path="data/hin/data-99.parquet",
        tokenizer=tokenizer,
        batch_size=12,
        max_length=256,
        train_split=0.995,
    )

    print("\n[3] Evaluating Cosine Similarity...")
    try:
        teacher_cs = compute_cosine_similarity_ground_truth(
            teacher, tokenizer, eval_loader, device
        )
        print(f"Cosine Similarity (teacher vs ground-truth): {teacher_cs:.4f}")

        student_cs = compute_cosine_similarity_ground_truth(
            student, tokenizer, eval_loader, device
        )
        print(f"Cosine Similarity (student vs ground-truth): {student_cs:.4f}")

        st_cs = compute_cosine_similarity_student_teacher(
            student, teacher, tokenizer, eval_loader, device
        )
        print(f"Cosine Similarity (student vs teacher): {st_cs:.4f}")
        print("Cosine similarity evaluations completed successfully.")
    except Exception:
        print("Cosine similarity evaluations failed.")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
