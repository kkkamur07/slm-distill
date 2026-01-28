"""Gradient accumulation and automatic mixed precision utilities.

This module provides the AmpGrad class for efficient training:
- Gradient accumulation to simulate larger batch sizes
- Automatic mixed precision (AMP) for faster training on CUDA
- Gradient scaling to prevent underflow with fp16
- Memory-efficient training for large models

The class wraps the optimizer and handles:
- Scaled backward passes
- Accumulated gradient updates
- Automatic unscaling before optimizer step

Example:
    >>> from src.training.accumulator import AmpGrad
    >>> from torch.optim import AdamW
    >>> 
    >>> optimizer = AdamW(model.parameters(), lr=1e-4)
    >>> amp_grad = AmpGrad(
    ...     optimizer=optimizer,
    ...     accum=4,  # Accumulate gradients over 4 steps
    ...     amp=True   # Use automatic mixed precision
    ... )
    >>> 
    >>> for batch in train_loader:
    ...     with torch.autocast(device_type='cuda', dtype=torch.float16):
    ...         loss = model(batch)
    ...     amp_grad.backward(loss)
    ...     if amp_grad.should_step():
    ...         amp_grad.step()
    ...         amp_grad.zero_grad()
"""

import torch

class AmpGrad:
    def __init__(
        self, 
        optimizer, 
        accum: int = 1,
        amp: bool = True,
        ):
        
        assert torch.cuda.is_available(), "AMP Training only works with NVIDIA GPUs, use them or disable it."
        
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self._n = 0
        
    def backward(self, loss: torch.Tensor):
        
        loss = loss / self.accum
        
        if self.amp:
            self.scaler.scale(loss).backward()
            
        else:
            loss.backward()
            
        self._n += 1
        
    def should_step(self):
        return (self._n % self.accum) == 0
    
    def unscale_(self): 
        if self.amp:
            self.scaler.unscale_(self.optim)
    
    def step(self):
        if self.amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
            
    def zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        