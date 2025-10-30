from transformers import AlbertConfig, AlbertForMaskedLM, BertConfig, BertForMaskedLM, PreTrainedTokenizerBase, AutoTokenizer
from typing import Tuple
import torch


def load_teacher_model(teacher_name: str, device: str) -> Tuple[BertForMaskedLM, PreTrainedTokenizerBase]:
    print(f"Loading teacher model: {teacher_name}")
    
    # Validate device
    if device not in ['cuda', 'cpu']:
        if not device.startswith('cuda:'):
            raise ValueError(f"Invalid device: {device}. Must be 'cuda', 'cpu', or 'cuda:N'")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    try:
        # Load tokenizer first (lightweight)
        tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        
        # Load model
        teacher = AlbertForMaskedLM.from_pretrained(teacher_name)
        
        # Freeze parameters BEFORE moving to device (more efficient)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        
        # Move to device
        teacher.to(device)
        
        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        print(f"✓ Teacher: {teacher_params:.1f}M params on {device}")
        
        return teacher, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher model '{teacher_name}': {str(e)}")

"""Create a student model"""

def create_student_model(
    vocab_size: int,
    layers: int,
    hidden_size: int,
    embedding_size: int,
    heads: int,
    intermediate: int,
    device: str
):
    print("Creating student model...")
    
    # Validate
    assert hidden_size % heads == 0, f"hidden_size must be divisible by heads"
    
    # Config
    config = AlbertConfig(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_hidden_layers=layers,
        hidden_size=hidden_size,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=512,        # Standard
        type_vocab_size=2,                  # Standard BERT
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    # Create and move
    student = AlbertForMaskedLM(config)
    student.to(device)

    
    return student
