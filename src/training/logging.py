import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class TrainingLogger:
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.jsonl"
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Redirect print statements
        self._original_stdout = sys.stdout
        sys.stdout = self._PrintCapture(self.logger)
        
        self.info(f"Logging initialized")
        self.info(f"Log file: {self.log_file}")
        self.info(f"Metrics file: {self.metrics_file}")
    
    class _PrintCapture:
        """Captures print() calls and redirects to logger."""
        def __init__(self, logger):
            self.logger = logger
            self.buffer = ""
        
        def write(self, text):
            if text and text != '\n':
                self.logger.info(text.rstrip())
        
        def flush(self):
            pass
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        # Also log to console/file
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
        self.info(f"Step {step:6d} | {metrics_str}")
    
    def log_training_step(self, step: int, loss: float, lr: float, **kwargs):


        metrics = {
            "train_loss": loss,
            "learning_rate": lr,
            **kwargs
        }
        self.log_metrics(step, metrics)
    
    def log_validation(self, step: int, val_loss: float, **kwargs):

        metrics = {
            "val_loss": val_loss,
            **kwargs
        }
        self.log_metrics(step, metrics)
        self.info(f"{'='*60}")
        self.info(f"📊 Validation at step {step}")
        self.info(f"   Loss: {val_loss:.4f}")
        for k, v in kwargs.items():
            self.info(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        self.info(f"{'='*60}")
    
    def log_config(self, config: Dict[str, Any]):

        self.info(f"{'='*60}")
        self.info("⚙️  Configuration:")
        for k, v in config.items():
            self.info(f"   {k}: {v}")
        self.info(f"{'='*60}")
    
    def log_model_info(self, model, name: str = "Model"):
    
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"{'='*60}")
        self.info(f"{name} Info:")
        self.info(f"Total parameters: {num_params:,}")
        self.info(f"Trainable parameters: {num_trainable:,}")
        self.info(f"Frozen parameters: {num_params - num_trainable:,}")
        self.info(f"{'='*60}")
    
    def close(self):

        sys.stdout = self._original_stdout
        self.info("📝 Logging finished")
        logging.shutdown()


def load_metrics(metrics_file: str) -> list[Dict[str, Any]]:
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics
