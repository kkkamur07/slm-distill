"""Test distillation loss function"""

import torch
import torch.nn.functional as F
from src.training.loss import distillation_loss


def test_distillation_loss():
    """Test the distillation loss with synthetic data"""
    
    print("="*60)
    print("TESTING DISTILLATION LOSS")
    print("="*60)
    
    # Setup
    batch_size = 4
    seq_len = 128
    vocab_size = 250002  # XLM-R vocab size
    temperature = 2.0
    alpha = 0.7
    
    # Create synthetic data
    print("\n[1] Creating synthetic data...")
    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)  # Add requires_grad
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)  # Teacher doesn't need grad
    
    # Create labels with masking (15% masked, rest -100)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    num_masked = int(batch_size * seq_len * 0.15)  # 15% masking
    masked_indices = torch.randperm(batch_size * seq_len)[:num_masked]
    labels_flat = labels.view(-1)
    labels_flat[masked_indices] = torch.randint(0, vocab_size, (num_masked,))
    labels = labels_flat.view(batch_size, seq_len)
    
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Seq length: {seq_len}")
    print(f"✓ Vocab size: {vocab_size}")
    print(f"✓ Total tokens: {batch_size * seq_len}")
    print(f"✓ Masked tokens: {(labels != -100).sum().item()}")
    print(f"✓ Masking ratio: {(labels != -100).sum().item() / (batch_size * seq_len) * 100:.1f}%")
    
    # Test loss computation
    print("\n[2] Computing loss...")
    total_loss, loss_kd, loss_ce = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        temperature=temperature,
        alpha=alpha
    )
    
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ KD loss: {loss_kd:.4f}")
    print(f"✓ CE loss: {loss_ce:.4f}")
    print(f"✓ Expected total: {alpha * loss_kd + (1-alpha) * loss_ce:.4f}")
    
    # Verify shapes and values
    print("\n[3] Validating...")
    assert total_loss.dim() == 0, "Total loss should be scalar"
    assert total_loss.item() > 0, "Loss should be positive"
    assert loss_kd >= 0, "KD loss should be non-negative"
    assert loss_ce > 0, "CE loss should be positive"
    
    # Check gradient flow
    print("\n[4] Testing gradients...")
    total_loss.backward()
    assert student_logits.grad is not None, "Student should have gradients"
    assert teacher_logits.grad is None, "Teacher should not have gradients"
    print("✓ Gradients flow correctly")
    
    # Test edge cases (create new tensors with requires_grad for each test)
    print("\n[5] Testing edge cases...")
    
    # All tokens masked
    student_logits_2 = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    teacher_logits_2 = torch.randn(batch_size, seq_len, vocab_size)
    labels_all = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss, _, _ = distillation_loss(student_logits_2, teacher_logits_2, labels_all, temperature, alpha)
    print(f"✓ All masked - Loss: {loss.item():.4f}")
    
    # No tokens masked (shouldn't happen in practice)
    student_logits_3 = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    teacher_logits_3 = torch.randn(batch_size, seq_len, vocab_size)
    labels_none = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    loss, kd, ce = distillation_loss(student_logits_3, teacher_logits_3, labels_none, temperature, alpha)
    print(f"✓ None masked - Loss: {loss.item():.4f}, KD: {kd:.4f}, CE: {ce:.4f}")
    assert kd == 0.0, "KD loss should be 0 when no tokens are masked"
    
    # Different temperatures
    print("\n[6] Testing temperature scaling...")
    for temp in [1.0, 2.0, 4.0, 8.0]:
        student_temp = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_temp = torch.randn(batch_size, seq_len, vocab_size)
        loss, kd, _ = distillation_loss(student_temp, teacher_temp, labels, temp, alpha)
        print(f"  T={temp:.1f}: Total={loss.item():.4f}, KD={kd:.4f}")
    
    # Different alpha values
    print("\n[7] Testing alpha weighting...")
    for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
        student_alpha = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_alpha = torch.randn(batch_size, seq_len, vocab_size)
        loss, kd, ce = distillation_loss(student_alpha, teacher_alpha, labels, temperature, a)
        expected = a * kd + (1-a) * ce
        print(f"  α={a:.1f}: Total={loss.item():.4f}, Expected={expected:.4f}")
        assert abs(loss.item() - expected) < 1e-5, f"Loss mismatch at alpha={a}"
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    test_distillation_loss()