"""Main entry point for knowledge distillation training.

This module orchestrates the entire distillation pipeline for compressing multilingual
models (XLM-RoBERTa) to monolingual models. It handles:
- Model initialization (teacher and student)
- Data loading and preprocessing
- Training loop with various distillation losses (KD, CE, score matching)
- Evaluation and checkpointing
- Hyperparameter optimization via Hydra

Example:
    Basic training:
        $ python -m src.main
    
    Hyperparameter sweep with Optuna:
        $ python -m src.main -m --config-name=sweep_config
    
    Override specific config parameters:
        $ python -m src.main training.learning_rate=1e-4 training.batch_size=32
    
    Background execution:
        $ nohup python -m src.main -m --config-name=sweep_config > sweep.log 2>&1 &
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.data.nativeSLM import NativeSLMData
from src.training.trainer import DistillationTrainer
from src.training.logging import TrainingLogger
from torch.utils.data import DataLoader
from torch.optim import AdamW


@hydra.main(version_base=None, config_path="../configs", config_name="config")
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

    logger.info("Loading models...")
    
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
    
    # # Optimizer
    # score_optimizer = AdamW([
    #     {'params': student.parameters()},
    #     {'params': projection.parameters()}
    # ], 
    #     lr=cfg.training.learning_rate,
    #     betas=(0.9, 0.999),
    #     eps=cfg.training.eps,
    #     weight_decay=cfg.training.weight_decay,
    # )
    
    # Optimizer
    ce_optimizer = AdamW([
        {'params': student.parameters()},
    ], 
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        eps=cfg.training.eps,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Adding the score projection layer based on teacher d_model
    projection = nn.Linear(768, 128, bias=False)
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        score_projection=projection,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=ce_optimizer,
        cfg=cfg,
        device=device,
        logger=logger
    )
    
    trainer.train()
    
    #For sweep
    best_val_loss = trainer.checkpoint_manager.best_val_loss
    
    # Close logger
    logger.close()
    
    print("Training complete!")
    
    return best_val_loss  # For sweep
    

if __name__ == "__main__":
    main()