from transformers import (
    XLMRobertaConfig, 
    XLMRobertaForMaskedLM, 
    AutoTokenizer,
    PreTrainedTokenizerBase
)
from typing import Tuple
import torch
import transformers


def load_teacher_model(
    teacher_name: str, 
    device: str,
    use_fp16: bool = True  # Add option for half precision
) -> Tuple[XLMRobertaForMaskedLM, PreTrainedTokenizerBase]:
    """Load XLM-RoBERTa teacher model"""
    print(f"Loading teacher model: {teacher_name}")
    
    # Validate device
    if device not in ['cuda', 'cpu']:
        if not device.startswith('cuda:'):
            raise ValueError(f"Invalid device: {device}. Must be 'cuda', 'cpu', or 'cuda:N'")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
        use_fp16 = False  # Can't use fp16 on CPU
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        
        # Load model with optional fp16
        with torch.no_grad():
            load_kwargs = {}
            if use_fp16 and device != 'cpu':
                load_kwargs['torch_dtype'] = torch.float16
            
            teacher = XLMRobertaForMaskedLM.from_pretrained(
                teacher_name,
                **load_kwargs
            )
        
        # Freeze and set to eval
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        
        # Move to device
        teacher.to(device)
        
        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        dtype_str = "fp16" if use_fp16 else "fp32"
        print(f"✓ Teacher: {teacher_params:.1f}M params on {device} ({dtype_str})")
        
        return teacher, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher model '{teacher_name}': {str(e)}")


def create_student_model(
    tokenizer: PreTrainedTokenizerBase,
    embedding_size : int,
    layers: int,
    hidden_size: int,
    heads: int,
    intermediate: int,
    device: str,
    use_gradient_checkpointing: bool = True,
) -> XLMRobertaForMaskedLM:
    """Create XLM-RoBERTa student model with same vocabulary as teacher"""
    print("Creating student model...")
    
    # Validate
    assert hidden_size % heads == 0, f"hidden_size ({hidden_size}) must be divisible by heads ({heads})"
    
    # Get vocab and special tokens from teacher's tokenizer
    vocab_size = tokenizer.vocab_size
    
    # XLM-RoBERTa Config
    config = XLMRobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=514,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        # Use tokenizer's special token IDs
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Create model
    student = XLMRobertaForMaskedLM(config)
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        student.gradient_checkpointing_enable()
    
    student.to(device)
    
    # Print info
    params = sum(p.numel() for p in student.parameters()) / 1e6
    gc_str = "with grad checkpoint" if use_gradient_checkpointing else ""
    print(f"✓ Student: {params:.1f}M params ({layers}L, {hidden_size}H, {heads}A) {gc_str}")
    print(f"  Vocab size: {vocab_size} (matches teacher)")
    
    return student