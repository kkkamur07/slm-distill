import math

class WarmCosineLR:
    def __init__(
        self,
        optimizer, 
        warmup_steps: int, 
        total_steps: int,
        base_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
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