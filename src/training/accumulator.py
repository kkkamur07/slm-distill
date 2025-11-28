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
        