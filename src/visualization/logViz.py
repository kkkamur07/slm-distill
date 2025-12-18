import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_FILE = 'outputs/logs/run.log'
OUTPUT_DIR = 'outputs/plots'

# ============================================================================
# DATA PARSING
# ============================================================================
train_data = defaultdict(list)
val_data = defaultdict(list)

with open(LOG_FILE, 'r') as f:
    for line in f:
        # Training metrics
        train_match = re.search(
            r'Step\s+(\d+)\s+\|\s+train_loss:\s+([\d.]+)\s+\|\s+learning_rate:\s+([\d.]+)\s+\|\s+kl_loss:\s+([\d.]+)\s+\|\s+ce_loss:\s+([\d.]+)', 
            line
        )
        if train_match:
            step = int(train_match.group(1))
            train_data['step'].append(step)
            train_data['train_loss'].append(float(train_match.group(2)))
            train_data['learning_rate'].append(float(train_match.group(3)))
            train_data['kl_loss'].append(float(train_match.group(4)))
            train_data['ce_loss'].append(float(train_match.group(5)))
        
        # Validation metrics
        val_match = re.search(
            r'Step\s+(\d+)\s+\|\s+val_loss:\s+([\d.]+)\s+\|\s+teacher_perplexity:\s+([\d.]+)\s+\|\s+teacher_masked_accuracy:\s+([\d.]+)\s+\|\s+student_perplexity:\s+([\d.]+)\s+\|\s+student_masked_accuracy:\s+([\d.]+)\s+\|\s+val_kl_loss:\s+([\d.]+)\s+\|\s+val_ce_loss:\s+([\d.]+)', 
            line
        )
        if val_match:
            step = int(val_match.group(1))
            val_data['step'].append(step)
            val_data['val_loss'].append(float(val_match.group(2)))
            val_data['teacher_perplexity'].append(float(val_match.group(3)))
            val_data['teacher_masked_accuracy'].append(float(val_match.group(4)))
            val_data['student_perplexity'].append(float(val_match.group(5)))
            val_data['student_masked_accuracy'].append(float(val_match.group(6)))
            val_data['val_kl_loss'].append(float(val_match.group(7)))
            val_data['val_ce_loss'].append(float(val_match.group(8)))

# ============================================================================
# PLOT STYLING
# ============================================================================
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("husl")

# Custom color palette
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D62828',
    'teacher': '#4A5899',
    'student': '#E63946'
}

# ============================================================================
# COMPREHENSIVE METRICS DASHBOARD (9 subplots)
# ============================================================================
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')

# 1. Learning Rate
ax1 = plt.subplot(3, 3, 1)
sns.lineplot(x=train_data['step'], y=train_data['learning_rate'], 
             color=colors['primary'], linewidth=2.5, ax=ax1)
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Learning Rate', fontsize=10)
ax1.set_title('Learning Rate Schedule', fontsize=11, fontweight='600', pad=10)
sns.despine(ax=ax1)

# 2. Perplexity (Teacher vs Student)
ax2 = plt.subplot(3, 3, 2)
sns.lineplot(x=val_data['step'], y=val_data['teacher_perplexity'], 
             label='Teacher', color=colors['teacher'], linewidth=2.5, ax=ax2, marker='o', markersize=5, markevery=5)
sns.lineplot(x=val_data['step'], y=val_data['student_perplexity'], 
             label='Student', color=colors['student'], linewidth=2.5, ax=ax2, marker='s', markersize=5, markevery=5)
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Perplexity', fontsize=10)
ax2.set_title('Model Perplexity', fontsize=11, fontweight='600', pad=10)
ax2.set_yscale('log')
ax2.legend(frameon=True, fancybox=False, shadow=False, fontsize=9)
sns.despine(ax=ax2)

# 3. Masked Accuracy (Teacher vs Student)
ax3 = plt.subplot(3, 3, 3)
sns.lineplot(x=val_data['step'], y=val_data['teacher_masked_accuracy'], 
             label='Teacher', color=colors['teacher'], linewidth=2.5, ax=ax3, marker='o', markersize=5, markevery=5)
sns.lineplot(x=val_data['step'], y=val_data['student_masked_accuracy'], 
             label='Student', color=colors['student'], linewidth=2.5, ax=ax3, marker='s', markersize=5, markevery=5)
ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('Accuracy', fontsize=10)
ax3.set_title('Masked Token Accuracy', fontsize=11, fontweight='600', pad=10)
ax3.legend(frameon=True, fancybox=False, shadow=False, fontsize=9)
sns.despine(ax=ax3)

# 4. KL Loss (Train vs Validation)
ax4 = plt.subplot(3, 3, 4)
sns.lineplot(x=train_data['step'], y=train_data['kl_loss'], 
             label='Train', color=colors['primary'], linewidth=2, alpha=0.6, ax=ax4)
sns.lineplot(x=val_data['step'], y=val_data['val_kl_loss'], 
             label='Validation', color=colors['accent'], linewidth=2.5, ax=ax4, marker='o', markersize=5, markevery=5)
ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('KL Divergence', fontsize=10)
ax4.set_title('KL Divergence Loss', fontsize=11, fontweight='600', pad=10)
ax4.legend(frameon=True, fancybox=False, shadow=False, fontsize=9)
sns.despine(ax=ax4)

# 5. Validation CE Loss
ax5 = plt.subplot(3, 3, 5)
sns.lineplot(x=val_data['step'], y=val_data['val_ce_loss'], 
             color=colors['secondary'], linewidth=2.5, ax=ax5, marker='o', markersize=5, markevery=5)
ax5.set_xlabel('Training Step', fontsize=10)
ax5.set_ylabel('CE Loss', fontsize=10)
ax5.set_title('Validation Cross-Entropy Loss', fontsize=11, fontweight='600', pad=10)
sns.despine(ax=ax5)

# 6. Train Score Loss (with log scale)
ax6 = plt.subplot(3, 3, 6)
sns.lineplot(x=train_data['step'], y=train_data['ce_loss'], 
             color=colors['accent'], linewidth=2, alpha=0.8, ax=ax6)
ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('Score Loss', fontsize=10)
ax6.set_title('Score Loss', fontsize=11, fontweight='600', pad=10)
ax6.set_yscale('log')
sns.despine(ax=ax6)

# 7. Combined Loss View
ax7 = plt.subplot(3, 3, 7)
sns.lineplot(x=train_data['step'], y=train_data['train_loss'], 
             label='Train', color=colors['primary'], linewidth=2, alpha=0.6, ax=ax7)
sns.lineplot(x=val_data['step'], y=val_data['val_loss'], 
             label='Validation', color=colors['warning'], linewidth=2.5, ax=ax7, marker='o', markersize=5, markevery=5)
ax7.set_xlabel('Training Step', fontsize=10)
ax7.set_ylabel('Total Loss', fontsize=10)
ax7.set_title('Total Loss', fontsize=11, fontweight='600', pad=10)
ax7.legend(frameon=True, fancybox=False, shadow=False, fontsize=9)
sns.despine(ax=ax7)

# 8. Student Perplexity Progress
ax8 = plt.subplot(3, 3, 8)
sns.lineplot(x=val_data['step'], y=val_data['student_perplexity'], 
             color=colors['student'], linewidth=2.5, ax=ax8, marker='s', markersize=5, markevery=5)
ax8.set_xlabel('Training Step', fontsize=10)
ax8.set_ylabel('Perplexity', fontsize=10)
ax8.set_title('Student Model Perplexity', fontsize=11, fontweight='600', pad=10)
sns.despine(ax=ax8)

# 9. Accuracy Gap
ax9 = plt.subplot(3, 3, 9)
accuracy_gap = [t - s for t, s in zip(val_data['teacher_masked_accuracy'], 
                                        val_data['student_masked_accuracy'])]
sns.lineplot(x=val_data['step'], y=accuracy_gap, 
             color=colors['success'], linewidth=2.5, ax=ax9, marker='d', markersize=5, markevery=5)
ax9.axhline(y=0, color=colors['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
ax9.set_xlabel('Training Step', fontsize=10)
ax9.set_ylabel('Accuracy Gap', fontsize=10)
ax9.set_title('Teacher-Student Gap', fontsize=11, fontweight='600', pad=10)
sns.despine(ax=ax9)

plt.tight_layout(pad=2.0)
plt.savefig(f'{OUTPUT_DIR}/training_metrics_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# DETAILED CE LOSS COMPARISON
# ============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('white')

sns.lineplot(x=train_data['step'], y=train_data['ce_loss'], 
             color=colors['accent'], linewidth=2.5, alpha=0.9, ax=ax1)
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Score Loss', fontsize=11)
ax1.set_title('Score Loss', fontsize=12, fontweight='600', pad=12)
ax1.set_yscale('log')
sns.despine(ax=ax1)

sns.lineplot(x=val_data['step'], y=val_data['val_ce_loss'], 
             color=colors['secondary'], linewidth=2.5, ax=ax2, marker='o', markersize=6, markevery=5)
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('CE Loss', fontsize=11)
ax2.set_title('Validation Cross-Entropy Loss', fontsize=12, fontweight='600', pad=12)
sns.despine(ax=ax2)

plt.tight_layout(pad=2.0)
plt.savefig(f'{OUTPUT_DIR}/ce_loss_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# DETAILED KL LOSS
# ============================================================================
fig3, ax = plt.subplots(1, 1, figsize=(11, 5))
fig3.patch.set_facecolor('white')

sns.lineplot(x=train_data['step'], y=train_data['kl_loss'], 
             label='Train', color=colors['primary'], linewidth=2.5, alpha=0.7, ax=ax)
sns.lineplot(x=val_data['step'], y=val_data['val_kl_loss'], 
             label='Validation', color=colors['accent'], linewidth=2.5, ax=ax, marker='o', markersize=6, markevery=5)
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('KL Divergence', fontsize=11)
ax.set_title('KL Divergence Loss', fontsize=12, fontweight='600', pad=12)
ax.legend(frameon=True, fancybox=False, shadow=False, fontsize=10)
sns.despine(ax=ax)

plt.tight_layout(pad=2.0)
plt.savefig(f'{OUTPUT_DIR}/kl_loss_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# PERPLEXITY AND ACCURACY COMBINED
# ============================================================================
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig4.patch.set_facecolor('white')

sns.lineplot(x=val_data['step'], y=val_data['teacher_perplexity'], 
             label='Teacher', color=colors['teacher'], linewidth=2.5, ax=ax1, marker='o', markersize=6, markevery=5)
sns.lineplot(x=val_data['step'], y=val_data['student_perplexity'], 
             label='Student', color=colors['student'], linewidth=2.5, ax=ax1, marker='s', markersize=6, markevery=5)
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Perplexity', fontsize=11)
ax1.set_title('Model Perplexity Comparison', fontsize=12, fontweight='600', pad=12)
ax1.legend(frameon=True, fancybox=False, shadow=False, fontsize=10)
ax1.set_yscale('log')
sns.despine(ax=ax1)

sns.lineplot(x=val_data['step'], y=val_data['teacher_masked_accuracy'], 
             label='Teacher', color=colors['teacher'], linewidth=2.5, ax=ax2, marker='o', markersize=6, markevery=5)
sns.lineplot(x=val_data['step'], y=val_data['student_masked_accuracy'], 
             label='Student', color=colors['student'], linewidth=2.5, ax=ax2, marker='s', markersize=6, markevery=5)
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Masked Token Prediction Accuracy', fontsize=12, fontweight='600', pad=12)
ax2.legend(frameon=True, fancybox=False, shadow=False, fontsize=10)
sns.despine(ax=ax2)

plt.tight_layout(pad=2.0)
plt.savefig(f'{OUTPUT_DIR}/perplexity_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("TRAINING SUMMARY STATISTICS")
print("="*60)
print(f"Total training steps: {len(train_data['step'])}")
print(f"Total validation evaluations: {len(val_data['step'])}")
print(f"\nFinal Metrics (Step {val_data['step'][-1]}):")
print(f"  - Student Perplexity: {val_data['student_perplexity'][-1]:.4f}")
print(f"  - Student Masked Accuracy: {val_data['student_masked_accuracy'][-1]:.4f}")
print(f"  - Teacher Masked Accuracy: {val_data['teacher_masked_accuracy'][-1]:.4f}")
print(f"  - Validation CE Loss: {val_data['val_ce_loss'][-1]:.4f}")
print(f"  - Validation KL Loss: {val_data['val_kl_loss'][-1]:.4f}")
print(f"\nBest Metrics:")
print(f"  - Lowest Student Perplexity: {min(val_data['student_perplexity']):.4f} at step {val_data['step'][val_data['student_perplexity'].index(min(val_data['student_perplexity']))]}")
print(f"  - Highest Student Accuracy: {max(val_data['student_masked_accuracy']):.4f} at step {val_data['step'][val_data['student_masked_accuracy'].index(max(val_data['student_masked_accuracy']))]}")
print(f"  - Lowest Validation CE Loss: {min(val_data['val_ce_loss']):.4f} at step {val_data['step'][val_data['val_ce_loss'].index(min(val_data['val_ce_loss']))]}")
print("="*60)