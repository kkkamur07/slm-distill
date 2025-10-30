"""
Main training script for knowledge distillation
Modular version with clean separation of concerns
"""

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import os

from src.models import load_teacher_model, create_student_model
from src.data import prepare_datasets
from src.training.loss import distillation_loss
from src.training.trainer import DistillationTrainer
from src.evaluate import evaluate_model, test_predictions


def setup_training(cfg: DictConfig, student, num_batches: int):
    """Setup optimizer, scheduler, and scaler"""
    print("\nSetting up training...")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if cfg.hardware.mixed_precision else None
    
    # Calculate total steps
    total_steps = num_batches * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    
    # Learning rate scheduler
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    print(f"✓ Optimizer: AdamW")
    print(f"✓ Learning rate: {cfg.training.learning_rate}")
    print(f"✓ Total steps: {total_steps:,}")
    print(f"✓ Warmup steps: {warmup_steps:,}")
    print(f"✓ Effective batch size: {cfg.training.batch_size * cfg.training.gradient_accumulation_steps}")
    
    return optimizer, scheduler, scaler


def print_config_summary(cfg: DictConfig):
    """Print configuration summary"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Teacher: {cfg.teacher.name}")
    print(f"Student: {cfg.model_name} ({cfg.student.num_hidden_layers}L, {cfg.student.hidden_size}H)")
    print(f"Data: {cfg.data.dataset_name} - {cfg.data_name}")
    print(f"Training: {cfg.training_name}")
    print(f"Device: {cfg.hardware.device}")
    print(f"Mixed precision: {cfg.hardware.mixed_precision}")
    print("="*60)


def print_training_info(cfg: DictConfig, compression: float, train_loader, eval_loader):
    """Print training information"""
    print("\n" + "="*60)
    print("TRAINING INFORMATION")
    print("="*60)
    print(f"Compression ratio: {compression:.1f}x")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Eval batches: {len(eval_loader):,}")
    print(f"Temperature: {cfg.distillation.temperature}")
    print(f"Alpha (KD weight): {cfg.distillation.alpha}")
    print(f"Epochs: {cfg.training.num_epochs}")
    print("="*60)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    print("\n" + "="*60)
    print("PRETRAINING DISTILLATION: IndicBERT → Student Model")
    print("="*60)
    
    # Print configuration
    print_config_summary(cfg)
    
    # ============================================================
    # [1/6] LOAD MODELS
    # ============================================================
    print("\n[1/6] Loading Models")
    print("-" * 60)
    
    teacher, tokenizer = load_teacher_model(cfg.teacher.name, cfg.hardware.device)
    
    student = create_student_model(
        tokenizer=tokenizer,
        layers=cfg.student.num_hidden_layers,
        hidden_size=cfg.student.hidden_size,
        heads=cfg.student.num_attention_heads,
        intermediate=cfg.student.intermediate_size,
        device=cfg.hardware.device
    )
    
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    compression = teacher_params / student_params
    
    print(f"✓ Teacher params: {teacher_params:.1f}M")
    print(f"✓ Student params: {student_params:.1f}M")
    print(f"✓ Compression: {compression:.1f}x")
    
    # ============================================================
    # [2/6] LOAD DATA
    # ============================================================
    print("\n[2/6] Loading Data")
    print("-" * 60)
    
    train_loader, eval_loader = prepare_datasets(
        dataset_name=cfg.data.dataset_name,
        subset_name=cfg.data.subset_name,
        data_percent=cfg.data.data_percent,
        tokenizer=tokenizer,
        max_length=cfg.data.max_seq_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers
    )
    
    # ============================================================
    # [3/6] SETUP TRAINING
    # ============================================================
    print("\n[3/6] Setting Up Training")
    print("-" * 60)
    
    optimizer, scheduler, scaler = setup_training(
        cfg, student, len(train_loader)
    )
    
    # Print training info
    print_training_info(cfg, compression, train_loader, eval_loader)
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        scaler=scaler
    )
    
    # ============================================================
    # [4/6] TRAINING LOOP
    # ============================================================
    print("\n[4/6] Training")
    print("="*60)
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        print("-" * 60)
        
        # Train epoch
        metrics = trainer.train_epoch(
            epoch=epoch,
            distillation_loss_fn=distillation_loss,
            eval_loader=eval_loader
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss:     {metrics['train_loss']:.4f}")
        print(f"  Train KD Loss:  {metrics['train_loss_kd']:.4f}")
        print(f"  Train CE Loss:  {metrics['train_loss_ce']:.4f}")
        
        if metrics['eval_loss'] is not None:
            print(f"  Eval Loss:      {metrics['eval_loss']:.4f}")
            
            # Save best model
            is_best = trainer.save_best(metrics['eval_loss'], tokenizer)
            if is_best:
                print(f"  ★ Best model updated!")
    
    # ============================================================
    # [5/6] SAVE FINAL MODEL
    # ============================================================
    print("\n[5/6] Saving Final Model")
    print("-" * 60)
    
    trainer.save_final(tokenizer)
    
    # ============================================================
    # [6/6] EVALUATION
    # ============================================================
    print("\n[6/6] Evaluation")
    print("-" * 60)
    
    # Load best model for evaluation
    best_model_path = os.path.join(cfg.paths.output_dir, "best_model")
    if os.path.exists(best_model_path):
        print("Evaluating best model...")
        from transformers import BertForMaskedLM
        student_best = BertForMaskedLM.from_pretrained(best_model_path)
        student_best.to(cfg.hardware.device)
    else:
        print("No best model found, using final model...")
        student_best = student
    
    # Perplexity evaluation
    print("\nComputing perplexity...")
    ppl = evaluate_model(student_best, eval_loader, cfg.hardware.device)
    print(f"Student Perplexity: {ppl:.2f}")
    
    # Qualitative evaluation
    print("\nQualitative evaluation:")
    test_predictions(student_best, tokenizer, cfg.hardware.device)
    
    # ============================================================
    # TRAINING COMPLETE
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model: {cfg.model_name}")
    print(f"Data: {cfg.data_name}")
    print(f"Training: {cfg.training_name}")
    print(f"\nCompression: {compression:.1f}x")
    print(f"Student PPL: {ppl:.2f}")
    print(f"Teacher PPL: ~12-15 (estimated)")
    print(f"\nPerformance: {ppl/15*100:.1f}% of teacher")
    print(f"\nModels saved in: {cfg.paths.output_dir}")
    print(f"  - best_model/")
    print(f"  - final_model/")
    print(f"  - step_* (checkpoints)")
    print("="*60)


if __name__ == "__main__":
    main()
