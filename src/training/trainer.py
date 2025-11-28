import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .logging import TrainingLogger
from .checkpointing import CheckpointManager
from .accumulator import AmpGrad
from .scheduler import WarmCosineLR
from .loss import distillation_loss


class DistillationTrainer:
    
    def __init__(
        self,
        student,
        teacher,
        train_loader,
        val_loader,
        optimizer,
        cfg: DictConfig, 
        device: torch.device,
        logger: Optional[TrainingLogger] = None,
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        
        if hasattr(self.student, "device"):
            self.student.device = device
        if hasattr(self.teacher, "device"):
            self.teacher.device = device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.cfg = cfg  # Store full config
        self.device = device
        self.logger = logger
        
        self.teacher.eval()
        
        # Training components
        self.amp_grad = AmpGrad(
            optimizer=optimizer,
            accum=cfg.training.grad_accum_steps,
            amp=cfg.training.mixed_precision
        )
        
        self.scheduler = WarmCosineLR(
            optimizer=optimizer,
            warmup_steps=cfg.training.warmup_steps,
            total_steps=cfg.training.total_steps,
            base_lr=cfg.training.learning_rate
        )
        
        self.checkpoint_manager = CheckpointManager(
            out_dir=cfg.paths.checkpoint_dir,
            keep_last_k=cfg.training.keep_last_k,
            logger=logger
        )
        
        try : 
            start_step, last_val_loss = self.checkpoint_manager.load_last(
                model=self.student,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            
            self.global_step = start_step
            self.last_val_loss = last_val_loss
        
        except FileNotFoundError:
            self.global_step = 0
            self.last_val_loss = float('inf')
        
        # Hyperparameters
        self.temperature = cfg.training.temperature
        self.alpha = cfg.training.alpha
        self.max_steps = cfg.training.total_steps
        self.val_every = cfg.training.val_every
        self.log_every = cfg.training.log_every
        
        
        if self.logger:
            self.logger.info("DistillationTrainer initialized")
            self.logger.log_config(OmegaConf.to_container(cfg, resolve=True))
            self.logger.log_model_info(teacher, "Teacher Model")
            self.logger.log_model_info(student, "Student Model")
    
    def train_step(self, batch: Dict[str, torch.Tensor]):
        
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Convert to long - important
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask']
        labels = batch['labels'].long()
        
        with torch.autocast(device_type=self.device.type, dtype = torch.float16, enabled=self.cfg.training.mixed_precision):
            
            with torch.no_grad():
                teacher_logits = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=True
                )
                
             # Student forward
            student_logits = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_logits=True
            )
             # Compute loss
            loss, kl_loss, ce_loss = self.distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=self.temperature,
                alpha=self.alpha
            )   
        
        # Backward pass with gradient accumulation
        self.amp_grad.backward(loss)
        
        return {
            'loss': loss.item(),
            'kl_loss': kl_loss,
            'ce_loss': ce_loss
        }
    
    @torch.no_grad()
    def validate(self):

        self.student.eval()
        
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].long().to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].long().to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype = torch.float16, enabled=self.cfg.training.mixed_precision):
                
                # Forward pass
                teacher_logits = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=True
                )
            
                student_logits = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=True
                )
            
                # Compute loss
                loss, kl_loss, ce_loss = self.distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=self.temperature,
                    alpha=self.alpha
                )
                
            total_loss += loss.item()
            total_kl_loss += kl_loss
            total_ce_loss += ce_loss
            num_batches += 1
        
        self.student.train()
        
        avg_loss = total_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_kl_loss': avg_kl_loss,
            'val_ce_loss': avg_ce_loss
        }
    
    def train(self):
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("Starting training...")
            self.logger.info("=" * 60)
        
        self.student.train()
        
        running_loss = 0.0
        running_kl_loss = 0.0
        running_ce_loss = 0.0
        
        while self.global_step < self.max_steps:
            
            for batch in tqdm(self.train_loader, desc="Training Batches"):
                # Training step
                metrics = self.train_step(batch)

                accum_steps = self.cfg.training.grad_accum_steps
                
                running_loss += metrics['loss'] / accum_steps
                running_kl_loss += metrics['kl_loss'] / accum_steps
                running_ce_loss += metrics['ce_loss'] / accum_steps
                             
                # Optimizer step
                if self.amp_grad.should_step():
                    
                    # Unscale gradients if using AMP
                    self.amp_grad.unscale_()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    
                    # Update weights
                    self.amp_grad.step()
                    self.amp_grad.zero_grad()
                    
                    # Update learning rate
                    lr = self.scheduler.step()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_every == 0:
                        avg_loss = running_loss / self.log_every
                        avg_kl_loss = running_kl_loss / self.log_every
                        avg_ce_loss = running_ce_loss / self.log_every
                        
                        if self.logger:
                            self.logger.log_training_step(
                                step=self.global_step,
                                loss=avg_loss,
                                lr=f"{lr:.8f}",
                                kl_loss=avg_kl_loss,
                                ce_loss=avg_ce_loss
                            )
                        
                        running_loss = 0.0
                        running_kl_loss = 0.0
                        running_ce_loss = 0.0
                    
                    if self.global_step % self.val_every == 0:
                        self.logger.info(f"Entered Validation Step: {self.global_step}")
                        val_metrics = self.validate()
                        
                        if self.logger:
                            self.logger.log_validation(
                                step=self.global_step,
                                val_loss=val_metrics['val_loss'],
                                val_kl_loss=val_metrics['val_kl_loss'],
                                val_ce_loss=val_metrics['val_ce_loss']
                            )
                        
                        # Save checkpoint
                        self.checkpoint_manager.save(
                            model=self.student,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=self.global_step,
                            val_loss=val_metrics['val_loss'],
                            config=self.cfg
                        )
                    
                    # Stop if max steps reached
                    if self.global_step >= self.max_steps:
                        break
            
            if self.global_step >= self.max_steps:
                break
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("Training completed!")
            self.logger.info(f"Total steps: {self.global_step}")
            self.logger.info("=" * 60)
        
        # Load best model
        if self.logger:
            self.logger.info("Loading best model...")
        
        self.checkpoint_manager.load_best(self.student)
        
        if self.logger:
            self.logger.info("Training finished! Best model loaded.")