"""
Training curve visualization script.
Parses training logs and creates publication-quality plots.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log to extract metrics."""
    print(f"Parsing log file: {log_file}")

    steps = []
    train_losses = []
    eval_losses = []
    perplexities = []
    learning_rates = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse training steps
            # Example: "Step 10: loss=3.4707, lr=5.60e-05, grad_norm=2.4528, mem=4.94GB"
            step_match = re.search(r'Step (\d+): loss=([\d.]+), lr=([\d.e-]+)', line)
            if step_match:
                step = int(step_match.group(1))
                loss = float(step_match.group(2))
                lr = float(step_match.group(3))

                steps.append(step)
                train_losses.append(loss)
                learning_rates.append(lr)

            # Parse evaluation results
            # Example: "Eval loss: 1.0876, Perplexity: 2.9672"
            eval_match = re.search(r'Eval loss: ([\d.]+), Perplexity: ([\d.]+)', line)
            if eval_match:
                eval_loss = float(eval_match.group(1))
                perplexity = float(eval_match.group(2))

                eval_losses.append(eval_loss)
                perplexities.append(perplexity)

    print(f"  Found {len(steps)} training steps")
    print(f"  Found {len(eval_losses)} evaluation points")

    return {
        'steps': steps,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'perplexities': perplexities,
        'learning_rates': learning_rates
    }

def plot_training_curves(data, output_dir):
    """Create training curve plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-quality style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

    # 1. Loss curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Training loss
    ax1.plot(data['steps'], data['train_losses'],
             linewidth=2, color='#2ecc71', label='Training Loss', alpha=0.8)

    # Add evaluation points
    eval_steps = [100, 200, 300, 400, 500]  # Known evaluation steps
    if len(data['eval_losses']) >= len(eval_steps):
        ax1.scatter(eval_steps[:len(data['eval_losses'])],
                   data['eval_losses'][:len(eval_steps)],
                   s=100, color='#e74c3c', label='Evaluation Loss',
                   zorder=5, edgecolors='white', linewidth=2)

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Evaluation Loss Over Time', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Learning rate schedule
    ax2.plot(data['steps'], data['learning_rates'],
             linewidth=2, color='#3498db', label='Learning Rate')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    loss_plot = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
    print(f"Saved: {loss_plot}")
    plt.close()

    # 2. Perplexity over time
    if len(data['perplexities']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate eval_steps dynamically
        eval_steps = list(range(100, 100 * (len(data['perplexities']) + 1), 100))
        ax.plot(eval_steps[:len(data['perplexities'])], data['perplexities'],
                marker='o', linewidth=2.5, markersize=8,
                color='#9b59b6', label='Perplexity')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Perplexity')
        ax.set_title('Model Perplexity Over Time', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Add value annotations
        for step, ppl in zip(eval_steps, data['perplexities']):
            ax.annotate(f'{ppl:.2f}',
                       (step, ppl),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       fontsize=9)

        plt.tight_layout()
        ppl_plot = os.path.join(output_dir, 'perplexity_curve.png')
        plt.savefig(ppl_plot, dpi=300, bbox_inches='tight')
        print(f"Saved: {ppl_plot}")
        plt.close()

    # 3. Summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    print(f"Total steps: {len(data['steps'])}")
    print(f"Initial loss: {data['train_losses'][0]:.4f}")
    print(f"Final loss: {data['train_losses'][-1]:.4f}")
    print(f"Loss reduction: {(1 - data['train_losses'][-1]/data['train_losses'][0])*100:.1f}%")

    if len(data['perplexities']) > 0:
        print(f"\nInitial perplexity: {data['perplexities'][0]:.4f}")
        print(f"Final perplexity: {data['perplexities'][-1]:.4f}")
        print(f"Perplexity reduction: {(1 - data['perplexities'][-1]/data['perplexities'][0])*100:.1f}%")

    print(f"\nLearning rate range: {min(data['learning_rates']):.2e} - {max(data['learning_rates']):.2e}")
    print("="*60)

def main():
    script_dir = Path(__file__).parent
    log_file = script_dir / "../../results/baseline_training.log"
    output_dir = script_dir / "../../results/analysis"

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return

    print("="*60)
    print("TRAINING CURVE ANALYSIS")
    print("="*60)

    # Parse log
    data = parse_training_log(log_file)

    # Create plots
    plot_training_curves(data, output_dir)

    print("\n✅ Analysis complete!")
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
