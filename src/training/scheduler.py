"""Learning rate schedulers for training.

This module provides custom learning rate scheduling strategies:
- WarmCosineLR: Warmup followed by cosine annealing
  - Linear warmup phase for stability
  - Cosine decay for smooth convergence
  - Optional cycle restarts for longer training

The scheduler helps achieve stable training and better convergence by
gradually increasing learning rate during warmup and smoothly decreasing
it during the main training phase.

Example:
    >>> from src.training.scheduler import WarmCosineLR
    >>> from torch.optim import AdamW
    >>> 
    >>> optimizer = AdamW(model.parameters(), lr=1e-4)
    >>> scheduler = WarmCosineLR(
    ...     optimizer=optimizer,
    ...     warmup_steps=1000,
    ...     total_steps=50000,
    ...     base_lr=1e-4
    ... )
    >>> 
    >>> # During training loop
    >>> for step in range(50000):
    ...     # ... training code ...
    ...     current_lr = scheduler.step()
    ...     print(f"Step {step}, LR: {current_lr:.6f}")
"""

import math

class WarmCosineLR:
    def __init__(
        self,
        optimizer, 
        warmup_steps: int, 
        total_steps: int,
        base_lr: float,
        cycle_restart_steps: int = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.step_num = 0
        self.cycle_restart_steps = cycle_restart_steps 
        
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            if self.cycle_restart_steps is not None:
                steps_since_warmup = self.step_num - self.warmup_steps
                cycle_progress = steps_since_warmup % self.cycle_restart_steps
                
                progress = cycle_progress / self.cycle_restart_steps 
            else:
                progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))  
        
        for g in self.optimizer.param_groups:
            g["lr"] = lr
            
        return lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)