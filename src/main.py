import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.data.dataset import NativeSLMData
from src.training.trainer import DistillationTrainer
from src.training.logging import TrainingLogger
from torch.utils.data import DataLoader
from torch.optim import AdamW


@hydra.main(version_base=None, config_path="/home/krrish/Desktop/Programming/geneformer-scratch/configs", config_name="config")
def main(cfg: DictConfig):
    
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
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=cfg.paths.log_dir,
        experiment_name=cfg.names.experiment_name
    )
    
    logger.info("🔧 Loading models...")
    
    # Load teacher model & tokenizer
    teacher = TeacherModel(model_path=cfg.paths.teacher_model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.teacher_model_path, use_fast = True)
    
    logger.info(f"Teacher model loaded from {cfg.paths.teacher_model_path} with {teacher.get_num_parameters():,} parameters")
    
    # Create student model
    student = StudentModel(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    logger.info(f"Student model created with parameters:{student.get_num_parameters():,} total, {student.get_trainable_parameters():,} trainable")
    
    # Load the datasets
    logger.info("Loading datasets...")
    
    train_dataset = NativeSLMData(cfg.data.dataset_path, train_split=cfg.data.train_split, tokenizer=tokenizer, split="train")
    val_dataset = NativeSLMData(cfg.data.dataset_path, train_split=(1 - cfg.data.train_split), tokenizer=tokenizer, split="val")

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


if __name__ == "__main__":
    main()