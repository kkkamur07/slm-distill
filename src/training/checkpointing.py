from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn
import shutil

if TYPE_CHECKING:
    from .logging import TrainingLogger

DEF_NAME = "model_last.pt"
BEST_NAME = "model_best.pt"


def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    step: int, 
    out_dir: str,
    val_loss: Optional[float] = None,
    config: dict | None = None,
    is_best: bool = False,
    logger: Optional['TrainingLogger'] = None
):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = _extract_config_from_model(model)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "config": config,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "version": "geneformer-distillation-v1",
    }

    # Save last checkpoint
    torch.save(checkpoint, out / DEF_NAME)
    _log(logger, f"Saved checkpoint at step {step} to {out / DEF_NAME}")

    # Save best checkpoint if this is the best validation loss
    if is_best:
        torch.save(checkpoint, out / BEST_NAME)
        _log(logger, f"New best model! Val loss: {val_loss:.4f} - Saved to {out / BEST_NAME}")


def load_checkpoint(
    model, 
    path: str, 
    optimizer=None, 
    scheduler=None, 
    strict: bool = True,
    logger: Optional['TrainingLogger'] = None
):

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Verify config matches if available
    cfg = ckpt.get("config")
    if cfg:
        ok, msg = _verify_model_matches(model, cfg)
        if not ok:
            raise RuntimeError(msg + "\nRebuild the model with this config, or load with strict=False.")

    # Load model state
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}")

    # Load optimizer and scheduler
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float('inf'))
    
    _log(logger, f"Loaded checkpoint from step {step}, val_loss: {val_loss:.4f}")
    
    return step, val_loss


def _log(logger: Optional['TrainingLogger'], msg: str):

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _extract_config_from_model(model) -> dict:

    cfg = {}
    try:
        if hasattr(model, "config"):
            cfg_obj = model.config
            if hasattr(cfg_obj, "__dict__"):
                cfg = cfg_obj.__dict__.copy()
            elif isinstance(cfg_obj, dict):
                cfg = cfg_obj.copy()
        
        # For BertForMaskedLM
        if hasattr(model, "model") and hasattr(model.model, "config"):
            bert_cfg = model.model.config
            cfg = {
                "vocab_size": bert_cfg.vocab_size,
                "hidden_size": bert_cfg.hidden_size,
                "num_hidden_layers": bert_cfg.num_hidden_layers,
                "num_attention_heads": bert_cfg.num_attention_heads,
                "intermediate_size": bert_cfg.intermediate_size,
                "max_position_embeddings": bert_cfg.max_position_embeddings,
            }
            
    except Exception:
        pass
    
    return cfg


def _verify_model_matches(model, cfg: Dict[str, Any]) -> Tuple[bool, str]:

    try:
        if hasattr(model, "model") and hasattr(model.model, "config"):
            model_cfg = model.model.config
            
            checks = {
                "vocab_size": (cfg.get("vocab_size"), model_cfg.vocab_size),
                "hidden_size": (cfg.get("hidden_size"), model_cfg.hidden_size),
                "num_hidden_layers": (cfg.get("num_hidden_layers"), model_cfg.num_hidden_layers),
            }
            
            diffs = [f"{k}: ckpt={v[0]} vs model={v[1]}" 
                    for k, v in checks.items() if v[0] is not None and v[0] != v[1]]
            
            if diffs:
                return False, "Architecture mismatch:\n  " + "\n  ".join(diffs)
        
        return True, "ok"
    except Exception as e:
        return True, f"Could not verify (skipping): {e}"


class CheckpointManager:
    
    def __init__(self, out_dir: str, keep_last_k: int = 3, logger: Optional['TrainingLogger'] = None):

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.best_val_loss = float('inf')
        self.logger = logger
        
        _log(self.logger, f"💾 Checkpoint manager initialized: {self.out_dir}")
        _log(self.logger, f"   Keeping last {keep_last_k} checkpoints")
    
    def save(
        self, 
        model, 
        optimizer, 
        scheduler, 
        step: int, 
        val_loss: Optional[float] = None,
        config: dict = None
    ):
        is_best = False
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            is_best = True
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            out_dir=str(self.out_dir),
            val_loss=val_loss,
            config=config,
            is_best=is_best,
            logger=self.logger
        )
        
        # Save per-step checkpoint
        per_step_path = self.out_dir / f"model_step{step:07d}.pt"
        try:
            shutil.copy2(self.out_dir / DEF_NAME, per_step_path)
            _log(self.logger, f"   Saved per-step checkpoint: {per_step_path.name}")
        except Exception as e:
            _log(self.logger, f"Warning: Could not save per-step checkpoint: {e}")
        
        # Clean up old per-step checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):

        try:
            ckpts = sorted(self.out_dir.glob("model_step*.pt"))
            for old in ckpts[:-self.keep_last_k]:
                old.unlink(missing_ok=True)
                _log(self.logger, f"Removed old checkpoint: {old.name}")
                
        except Exception as e:
            _log(self.logger, f"Warning: Could not cleanup checkpoints: {e}")
    
    def load_best(self, model, optimizer=None, scheduler=None):

        best_path = self.out_dir / BEST_NAME
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        
        _log(self.logger, f"Loading best checkpoint from {best_path}")
        return load_checkpoint(model, str(best_path), optimizer, scheduler, logger=self.logger)
    
    def load_last(self, model, optimizer=None, scheduler=None):
        last_path = self.out_dir / DEF_NAME
        if not last_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {last_path}")
        
        _log(self.logger, f"Loading last checkpoint from {last_path}")
        return load_checkpoint(model, str(last_path), optimizer, scheduler, logger=self.logger)