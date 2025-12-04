import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.data.nativeSLM import NativeSLMData
from src.training.trainer import DistillationTrainer
from src.training.logging import TrainingLogger
from src.evals.task_finetuning.sentiment_run import run_sentiment
from src.evals.task_finetuning.nli_run import run_nli
from src.evals.task_finetuning.ner_run import run_ner

from torch.utils.data import DataLoader
from torch.optim import AdamW


@hydra.main(version_base=None, config_path="/home/krrish/Desktop/Programming/slm-distill/configs", config_name="config")
def main(cfg: DictConfig):
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=cfg.paths.log_dir,
        experiment_name=cfg.names.experiment_name
    )
    
    # Print config
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision('high') 
        
    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")  

    logger.info("🔧 Loading models...")
    
    # Load teacher model & tokenizer
    teacher = TeacherModel(model_path=cfg.paths.teacher_model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.teacher_model_path, use_fast = True)
    
    logger.info(f"Teacher model loaded from {cfg.paths.teacher_model_path} with {teacher.get_num_parameters():,} parameters of which {teacher.get_num_trainable_parameters():,} are trainable")
    
    # Create student model
    student = StudentModel(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    logger.info(f"Student model created with parameters:{student.get_num_parameters():,} total, {student.get_trainable_parameters():,} trainable")

    student = torch.compile(student)
    
    # Load the datasets
    logger.info("Loading training dataset")
    
    train_dataset = NativeSLMData(
        data_path=cfg.data.dataset_path, 
        train_split=cfg.data.train_split, 
        tokenizer=tokenizer, 
        train=True,
        cache_dir=cfg.paths.cache_dir,
        max_length=cfg.model.max_sequence_length,
    )
    
    logger.info("Loading validation dataset")
    
    val_dataset = NativeSLMData(
        data_path=cfg.data.dataset_path, 
        train_split=cfg.data.train_split, 
        tokenizer=tokenizer, 
        train=False,
        cache_dir=cfg.paths.cache_dir,
        max_length=cfg.model.max_sequence_length,
    )
    
    logger.info(f"Datasets loaded: {len(train_dataset):,} training samples, {len(val_dataset):,} validation samples")

    # Creating samplers, collators and data loaders
    logger.info("Creating Datacollators & loaders")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.training.mlm_probability,
    )
    
    # Creating data loaders
    logger.info("Creating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None
    )
    
    logger.info(f"DataLoaders created")
    
    # Optimizer
    optimizer = AdamW(
        student.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        logger=logger
    )
    
    trainer.train()
    
    # Close logger
    logger.close()
    
    print("Training complete!")

    ### Finetuning part 
    print("Finetuning start")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Finetuning] Using device: {device}")

    sentiment_results = run_sentiment(
        num_labels=3,
        num_epochs=8,
        batch_size=16,
        max_length=128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
        device=device,
    )

    nli_results = run_nli(
        num_epochs = 8,
        batch_size = 16,
        max_length = 128,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
        device=device,
    )

    ner_results = run_ner(
        num_epochs=8,
        batch_size=16,
        max_length=128,
        device=device,
        lr_grid=[3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
        dropout=0.1,
        weight_decay=0.1,
        early_stopping_patience=3,
    )

    print("\n=== SUMMARY (test metrics) ===")
    print("Sentiment:", sentiment_results)
    print("NLI:", nli_results)
    print("NER:", ner_results)

    # Persist sentiment and NLI training information (including per-batch loss where available)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Sentiment
    sent_path = results_dir / f"sentiment_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved Sentiment results to {sent_path}")

    # NLI
    nli_path = results_dir / f"nli_{timestamp}.json"
    with nli_path.open("w", encoding="utf-8") as f:
        json.dump(nli_results, f, indent=2)
    print(f"[Finetuning] Saved NLI results to {nli_path}")

    # NER 
    ner_path = results_dir / f"ner_{timestamp}.json"
    with sent_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)
    print(f"[Finetuning] Saved NER results to {sent_path}")


if __name__ == "__main__":
    main()