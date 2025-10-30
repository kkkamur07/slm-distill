"""Training loop and optimization"""

import torch
import os
import shutil
from tqdm import tqdm
from typing import Optional, Callable


class DistillationTrainer:
    """Handles the training loop for knowledge distillation"""
    
    def __init__(
        self,
        student,
        teacher,
        train_loader,
        optimizer,
        scheduler,
        cfg,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.scaler = scaler
        
        self.global_step = 0
        self.best_loss = float('inf')
        
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    def train_epoch(
        self, 
        epoch: int, 
        distillation_loss_fn: Callable,
        eval_loader: Optional[torch.utils.data.DataLoader] = None
    ):
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()
        
        epoch_loss = 0
        epoch_loss_kd = 0
        epoch_loss_ce = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        try:
            for step, batch in enumerate(pbar):
                # Move to device
                input_ids = batch["input_ids"].to(self.cfg.device)
                attention_mask = batch["attention_mask"].to(self.cfg.device)
                labels = batch["labels"].to(self.cfg.device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                    # Teacher forward (no gradients)
                    with torch.no_grad():
                        teacher_logits = self.teacher(input_ids, attention_mask).logits
                    
                    # Student forward
                    student_logits = self.student(input_ids, attention_mask).logits
                    
                    # Compute loss
                    loss, loss_kd, loss_ce = distillation_loss_fn(
                        student_logits, teacher_logits, labels,
                        self.cfg.temperature, self.cfg.alpha
                    )
                    
                    # Scale for gradient accumulation
                    loss = loss / self.cfg.grad_accum
                
                # Backward pass
                if self.cfg.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step (every grad_accum steps)
                if (step + 1) % self.cfg.grad_accum == 0:
                    if self.cfg.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    
                    # Update weights
                    if self.cfg.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Track metrics
                loss_value = loss.item() * self.cfg.grad_accum
                epoch_loss += loss_value
                epoch_loss_kd += loss_kd
                epoch_loss_ce += loss_ce
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'kd': f'{loss_kd:.4f}',
                    'ce': f'{loss_ce:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Save checkpoint periodically
                if self.global_step % self.cfg.save_every == 0 and \
                   (step + 1) % self.cfg.grad_accum == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OOM Error at step {step}")
                print(f"Try: batch_size={self.cfg.batch_size//2} or grad_accum={self.cfg.grad_accum*2}")
                torch.cuda.empty_cache()
            raise
        
        # Calculate averages
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_loss_kd = epoch_loss_kd / num_batches
        avg_loss_ce = epoch_loss_ce / num_batches
        
        # Evaluate if eval_loader provided
        eval_loss = None
        if eval_loader is not None:
            eval_loss = self.evaluate(eval_loader, distillation_loss_fn)
        
        return {
            'train_loss': avg_loss,
            'train_loss_kd': avg_loss_kd,
            'train_loss_ce': avg_loss_ce,
            'eval_loss': eval_loss
        }
    
    def evaluate(self, eval_loader, distillation_loss_fn):
        """Evaluate on validation set"""
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.cfg.device)
                attention_mask = batch["attention_mask"].to(self.cfg.device)
                labels = batch["labels"].to(self.cfg.device)
                
                with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
                    teacher_logits = self.teacher(input_ids, attention_mask).logits
                    student_logits = self.student(input_ids, attention_mask).logits
                    
                    loss, _, _ = distillation_loss_fn(
                        student_logits, teacher_logits, labels,
                        self.cfg.temperature, self.cfg.alpha
                    )
                    
                    total_loss += loss.item()
        
        return total_loss / len(eval_loader)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.cfg.output_dir, name)
        self.student.save_pretrained(checkpoint_path)
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Keep only last N checkpoints"""
        checkpoints = [
            d for d in os.listdir(self.cfg.output_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(self.cfg.output_dir, d))
        ]
        
        if len(checkpoints) > keep_last:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("_")[1]))
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints[:-keep_last]:
                old_path = os.path.join(self.cfg.output_dir, old_checkpoint)
                shutil.rmtree(old_path)
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    def save_best(self, loss: float, tokenizer):
        """Save best model if loss improved"""
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.cfg.output_dir, "best_model")
            self.student.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"✓ Best model saved (loss: {loss:.4f})")
            return True
        return False
    
    def save_final(self, tokenizer):
        """Save final model"""
        final_path = os.path.join(self.cfg.output_dir, "final_model")
        self.student.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✓ Final model saved: {final_path}")
