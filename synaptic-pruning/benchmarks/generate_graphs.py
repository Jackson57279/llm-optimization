"""Generate additional benchmark visualization graphs.

This script creates comprehensive visualizations for all benchmark results:
1. Task-shift recovery training curves
2. Ablation study comparisons
3. Compression vs Accuracy detailed breakdown
4. Sparsity progression over epochs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def plot_task_shift_recovery(results: dict, output_dir: Path) -> None:
    """Plot task-shift recovery training curves.
    
    Creates 3 subplots:
    1. Loss over epochs (Task A and Task B)
    2. Accuracy over epochs
    3. Sparsity progression
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract history
    phase1_history = results["phase1_task_a_training"]["history"]
    phase3_history = results["phase3_task_b_recovery"]["history"]
    
    phase1_epochs = [h["epoch"] for h in phase1_history]
    phase3_epochs = [h["epoch"] for h in phase3_history]
    
    # Add offset for phase 3 to show continuous training
    phase3_epochs_offset = [e + max(phase1_epochs) + 1 for e in phase3_epochs]
    
    # Phase separator line position
    phase_sep = max(phase1_epochs) + 0.5
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(phase1_epochs, [h["train_loss"] for h in phase1_history], 
            'b-', linewidth=2, label='Task A (Training)', marker='o', markersize=4)
    ax.plot(phase3_epochs_offset, [h["train_loss"] for h in phase3_history], 
            'r-', linewidth=2, label='Task B (Recovery)', marker='s', markersize=4)
    ax.axvline(x=phase_sep, color='gray', linestyle='--', alpha=0.7, label='Task Shift')
    ax.axvline(x=phase_sep + 1, color='green', linestyle=':', alpha=0.7, label='Recovery Start')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax = axes[0, 1]
    ax.plot(phase1_epochs, [h["val_accuracy"] for h in phase1_history], 
            'b-', linewidth=2, label='Task A', marker='o', markersize=4)
    ax.plot(phase3_epochs_offset, [h["val_accuracy"] for h in phase3_history], 
            'r-', linewidth=2, label='Task B', marker='s', markersize=4)
    ax.axvline(x=phase_sep, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=phase_sep + 1, color='green', linestyle=':', alpha=0.7)
    # Mark the before-recovery point
    acc_before = results["phase2_task_b_before_recovery"]["accuracy"]
    ax.scatter([phase_sep], [acc_before], color='orange', s=150, zorder=5, 
               marker='*', label='Before Recovery', edgecolors='black')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity Progression
    ax = axes[1, 0]
    ax.plot(phase1_epochs, [h["sparsity"] for h in phase1_history], 
            'b-', linewidth=2, label='Task A', marker='o', markersize=4)
    ax.plot(phase3_epochs_offset, [h["sparsity"] for h in phase3_history], 
            'r-', linewidth=2, label='Task B', marker='s', markersize=4)
    ax.axvline(x=phase_sep, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=phase_sep + 1, color='green', linestyle=':', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Sparsity', fontsize=11)
    ax.set_title('Sparsity Progression', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Plot 4: Threshold Evolution
    ax = axes[1, 1]
    ax.plot(phase1_epochs, [h["threshold"] for h in phase1_history], 
            'b-', linewidth=2, label='Task A', marker='o', markersize=4)
    ax.plot(phase3_epochs_offset, [h["threshold"] for h in phase3_history], 
            'r-', linewidth=2, label='Task B', marker='s', markersize=4)
    ax.axvline(x=phase_sep, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=phase_sep + 1, color='green', linestyle=':', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Warm Threshold', fontsize=11)
    ax.set_title('Pruning Threshold Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Task-Shift Recovery Experiment (VAL-CROSS-002)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "task_shift_recovery_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'task_shift_recovery_curves.png'}")


def plot_ablation_comparison(results_50: dict, results_70: dict, output_dir: Path) -> None:
    """Plot ablation study comparison graphs.
    
    Creates:
    1. Loss comparison bar chart
    2. Perplexity comparison bar chart
    3. Sparsity and compression comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    methods = ['Activity-Driven', 'Random']
    colors = ['#2E86AB', '#A23B72']
    
    # Extract data
    def extract_data(results: dict):
        return {
            'loss': [results['activity_driven']['final_loss'], results['random']['final_loss']],
            'perplexity': [results['activity_driven']['perplexity'], results['random']['perplexity']],
            'sparsity': [results['activity_driven']['compression_stats']['sparsity'], 
                        results['random']['compression_stats']['sparsity']],
            'compression': [results['activity_driven']['compression_stats']['effective_compression'],
                           results['random']['compression_stats']['effective_compression']],
            'hot_params': [results['activity_driven']['compression_stats']['hot_params'],
                          results['random']['compression_stats']['hot_params']],
            'warm_params': [results['activity_driven']['compression_stats']['warm_params'],
                           results['random']['compression_stats']['warm_params']],
            'cold_params': [results['activity_driven']['compression_stats']['cold_params'],
                           results['random']['compression_stats']['cold_params']],
        }
    
    data_50 = extract_data(results_50)
    data_70 = extract_data(results_70)
    
    # Plot 1: Final Loss Comparison
    ax = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data_50['loss'], width, label='50% Sparsity, 2 Epochs', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, data_70['loss'], width, label='70% Sparsity, 3 Epochs', color='#F18F01', alpha=0.8)
    
    ax.set_ylabel('Final Loss', fontsize=11)
    ax.set_title('Final Loss: Activity-Driven vs Random Pruning', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Perplexity Comparison (clipped for visibility)
    ax = axes[0, 1]
    
    # Handle infinity values for visualization
    ppl_50 = [p if p < 1000 else 1000 for p in data_50['perplexity']]
    ppl_70 = [p if p < 1000 else 1000 for p in data_70['perplexity']]
    
    bars1 = ax.bar(x - width/2, ppl_50, width, label='50% Sparsity', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, ppl_70, width, label='70% Sparsity', color='#F18F01', alpha=0.8)
    
    ax.set_ylabel('Perplexity (capped at 1000)', fontsize=11)
    ax.set_title('Perplexity Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Plot 3: Sparsity Comparison
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, data_50['sparsity'], width, label='50% Target', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, data_70['sparsity'], width, label='70% Target', color='#F18F01', alpha=0.8)
    
    ax.set_ylabel('Actual Sparsity', fontsize=11)
    ax.set_title('Achieved Sparsity Levels', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.999, 1.0005)
    
    # Plot 4: Parameter Distribution (stacked bar)
    ax = axes[1, 1]
    
    categories = ['Activity-Driven\n(50%)', 'Random\n(50%)', 'Activity-Driven\n(70%)', 'Random\n(70%)']
    hot = [data_50['hot_params'][0], data_50['hot_params'][1], data_70['hot_params'][0], data_70['hot_params'][1]]
    warm = [data_50['warm_params'][0], data_50['warm_params'][1], data_70['warm_params'][0], data_70['warm_params'][1]]
    cold = [data_50['cold_params'][0], data_50['cold_params'][1], data_70['cold_params'][0], data_70['cold_params'][1]]
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    ax.bar(x_pos, hot, width, label='Hot (FP16)', color='#C73E1D', alpha=0.9)
    ax.bar(x_pos, warm, width, bottom=hot, label='Warm (4-bit)', color='#F18F01', alpha=0.9)
    ax.bar(x_pos, cold, width, bottom=[h+w for h,w in zip(hot, warm)], label='Cold (1-bit)', color='#2E86AB', alpha=0.9)
    
    ax.set_ylabel('Number of Parameters', fontsize=11)
    ax.set_title('Parameter Distribution by Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.suptitle('Ablation Study: Activity-Driven vs Random Pruning', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'ablation_comparison.png'}")


def plot_method_comparison_detailed(results: dict, output_dir: Path) -> None:
    """Create detailed comparison charts for different compression methods.
    
    Creates:
    1. Compression ratio comparison
    2. Perplexity comparison
    3. Compression vs Perplexity scatter with labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    baseline = results["baseline_fp16"]
    synaptic = results["synaptic_pruning"]
    gptq = results["gptq"]
    awq = results["awq"]
    
    # Data for plotting
    methods = []
    compression_ratios = []
    perplexities = []
    colors = []
    markers = []
    
    # Baseline
    methods.append('FP16 Baseline')
    compression_ratios.append(baseline["compression_ratio"])
    perplexities.append(baseline["perplexity"])
    colors.append('gold')
    markers.append('*')
    
    # Synaptic Pruning
    for r in synaptic:
        methods.append(f'Synaptic {r["target_sparsity"]:.0%}')
        compression_ratios.append(r["compression_ratio"])
        # Cap infinity for visualization
        ppl = r["perplexity"] if r["perplexity"] < 10000 else 10000
        perplexities.append(ppl)
        colors.append('#2E86AB')
        markers.append('o')
    
    # GPTQ
    for r in gptq:
        methods.append(f'GPTQ {r["bits"]}bit')
        compression_ratios.append(r["compression_ratio"])
        ppl = r["perplexity"] if r["perplexity"] < 10000 else 10000
        perplexities.append(ppl)
        colors.append('#C73E1D')
        markers.append('s')
    
    # AWQ
    for r in awq:
        methods.append(f'AWQ {r["bits"]}bit')
        compression_ratios.append(r["compression_ratio"])
        ppl = r["perplexity"] if r["perplexity"] < 10000 else 10000
        perplexities.append(ppl)
        colors.append('#3B1F2B')
        markers.append('^')
    
    # Plot 1: Compression Ratio Bar Chart
    ax = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, compression_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Compression Ratio (×)', fontsize=11)
    ax.set_title('Compression Ratio by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(compression_ratios) * 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}×', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Perplexity Bar Chart (clipped)
    ax = axes[1]
    ppl_clipped = [min(p, 1200) for p in perplexities]  # Clip for visibility
    bars = ax.bar(x_pos, ppl_clipped, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Perplexity (capped at 1200)', fontsize=11)
    ax.set_title('Perplexity by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 200)
    
    # Plot 3: Efficiency Score (Compression / log(Perplexity))
    ax = axes[2]
    # Calculate efficiency: higher compression with lower perplexity is better
    efficiency_scores = []
    for comp, ppl in zip(compression_ratios, perplexities):
        if ppl > 0 and ppl < 10000:
            # Higher is better: compression divided by log perplexity
            score = comp / (np.log(ppl + 1) + 1)
        else:
            score = 0
        efficiency_scores.append(score)
    
    bars = ax.bar(x_pos, efficiency_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Efficiency Score (Compression / log(PPL))', fontsize=11)
    ax.set_title('Compression Efficiency Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Detailed Method Comparison (VAL-CROSS-003)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_detailed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'method_comparison_detailed.png'}")


def plot_parameter_breakdown(results: dict, output_dir: Path) -> None:
    """Plot detailed parameter breakdown for Synaptic Pruning.
    
    Shows how parameters are distributed across hot/warm/cold categories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    synaptic_results = results["synaptic_pruning"]
    
    # Extract data
    sparsity_levels = [r["target_sparsity"] for r in synaptic_results]
    hot_params = [r["hot_params"] for r in synaptic_results]
    warm_params = [r["warm_params"] for r in synaptic_results]
    cold_params = [r["cold_params"] for r in synaptic_results]
    
    # Plot 1: Absolute parameter counts (log scale)
    ax = axes[0]
    x = np.arange(len(sparsity_levels))
    width = 0.25
    
    ax.bar(x - width, hot_params, width, label='Hot (FP16)', color='#C73E1D', alpha=0.9)
    ax.bar(x, warm_params, width, label='Warm (4-bit)', color='#F18F01', alpha=0.9)
    ax.bar(x + width, cold_params, width, label='Cold (1-bit)', color='#2E86AB', alpha=0.9)
    
    ax.set_xlabel('Target Sparsity Level', fontsize=11)
    ax.set_ylabel('Number of Parameters (log scale)', fontsize=11)
    ax.set_title('Parameter Count by Category', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.0%}' for s in sparsity_levels])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Percentage breakdown (stacked bar)
    ax = axes[1]
    
    total_params = [h + w + c for h, w, c in zip(hot_params, warm_params, cold_params)]
    hot_pct = [h / t * 100 for h, t in zip(hot_params, total_params)]
    warm_pct = [w / t * 100 for w, t in zip(warm_params, total_params)]
    cold_pct = [c / t * 100 for c, t in zip(cold_params, total_params)]
    
    ax.bar(x, hot_pct, label='Hot (FP16)', color='#C73E1D', alpha=0.9)
    ax.bar(x, warm_pct, bottom=hot_pct, label='Warm (4-bit)', color='#F18F01', alpha=0.9)
    ax.bar(x, cold_pct, bottom=[h + w for h, w in zip(hot_pct, warm_pct)], 
           label='Cold (1-bit)', color='#2E86AB', alpha=0.9)
    
    ax.set_xlabel('Target Sparsity Level', fontsize=11)
    ax.set_ylabel('Percentage of Parameters', fontsize=11)
    ax.set_title('Parameter Distribution (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.0%}' for s in sparsity_levels])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Synaptic Pruning: Parameter Breakdown', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'parameter_breakdown.png'}")


def create_summary_report(results_dir: Path, output_dir: Path) -> None:
    """Create a summary text report of all benchmark results."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SYNAPTIC PRUNING BENCHMARK RESULTS SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Load baseline comparison
    try:
        baseline = load_json(results_dir / "baseline_comparison.json")
        report_lines.append("VAL-CROSS-003: Baseline Comparison")
        report_lines.append("-" * 40)
        report_lines.append(f"Model: {baseline['config']['model_name']}")
        report_lines.append(f"Baseline FP16: PPL={baseline['baseline_fp16']['perplexity']:.2f}")
        report_lines.append("")
        report_lines.append("Synaptic Pruning Results:")
        for r in baseline["synaptic_pruning"]:
            report_lines.append(f"  Sparsity {r['target_sparsity']:.0%}: "
                              f"Compression={r['compression_ratio']:.1f}×, "
                              f"PPL={r['perplexity']:.2f}")
        report_lines.append("")
        report_lines.append("GPTQ Results:")
        for r in baseline["gptq"]:
            report_lines.append(f"  {r['bits']}bit: Compression={r['compression_ratio']:.1f}×, "
                              f"PPL={r['perplexity']:.2f}")
        report_lines.append("")
        report_lines.append("AWQ Results:")
        for r in baseline["awq"]:
            report_lines.append(f"  {r['bits']}bit: Compression={r['compression_ratio']:.1f}×, "
                              f"PPL={r['perplexity']:.2f}")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"Could not load baseline comparison: {e}")
        report_lines.append("")
    
    # Load task shift recovery
    try:
        task_shift = load_json(results_dir / "task_shift_recovery_results.json")
        report_lines.append("VAL-CROSS-002: Task-Shift Recovery")
        report_lines.append("-" * 40)
        report_lines.append(f"Task A: {task_shift['config']['task_a_id']} → Task B: {task_shift['config']['task_b_id']}")
        report_lines.append(f"Task A Final Accuracy: {task_shift['phase1_task_a_training']['final_accuracy']:.2%}")
        report_lines.append(f"Task B Before Recovery: {task_shift['phase2_task_b_before_recovery']['accuracy']:.2%}")
        report_lines.append(f"Task B After Recovery: {task_shift['phase3_task_b_recovery']['final_accuracy']:.2%}")
        report_lines.append(f"Accuracy Improvement: {task_shift['recovery_summary']['accuracy_improvement']:+.2%}")
        report_lines.append(f"Final Compression: {task_shift['recovery_summary']['compression_maintained']:.1f}×")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"Could not load task shift recovery: {e}")
        report_lines.append("")
    
    # Load ablation studies
    try:
        ablation_50 = load_json(results_dir / "ablation_sparsity50%_epochs2.json")
        report_lines.append("VAL-BEN-004: Ablation Study (50% Sparsity, 2 Epochs)")
        report_lines.append("-" * 40)
        report_lines.append(f"Activity-Driven: Loss={ablation_50['activity_driven']['final_loss']:.4f}, "
                          f"PPL={ablation_50['activity_driven']['perplexity']:.2f}")
        report_lines.append(f"Random Pruning:  Loss={ablation_50['random']['final_loss']:.4f}, "
                          f"PPL={ablation_50['random']['perplexity']:.2f}")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"Could not load ablation 50%: {e}")
        report_lines.append("")
    
    try:
        ablation_70 = load_json(results_dir / "ablation_sparsity70%_epochs3.json")
        report_lines.append("VAL-BEN-004: Ablation Study (70% Sparsity, 3 Epochs)")
        report_lines.append("-" * 40)
        report_lines.append(f"Activity-Driven: Loss={ablation_70['activity_driven']['final_loss']:.4f}, "
                          f"PPL={ablation_70['activity_driven']['perplexity']:.2f}")
        report_lines.append(f"Random Pruning:  Loss={ablation_70['random']['final_loss']:.4f}, "
                          f"PPL={ablation_70['random']['perplexity']:.2f}")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"Could not load ablation 70%: {e}")
        report_lines.append("")
    
    report_lines.append("=" * 70)
    report_lines.append("Generated Graphs:")
    report_lines.append("  - task_shift_recovery_curves.png")
    report_lines.append("  - ablation_comparison.png")
    report_lines.append("  - method_comparison_detailed.png")
    report_lines.append("  - parameter_breakdown.png")
    report_lines.append("  - pareto_frontier.png (existing)")
    report_lines.append("=" * 70)
    
    # Write report
    report_path = output_dir / "benchmark_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")


def main():
    """Generate all benchmark visualizations."""
    # Paths
    results_dir = Path("./results")
    output_dir = results_dir
    
    print("=" * 60)
    print("GENERATING BENCHMARK VISUALIZATIONS")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    try:
        task_shift_results = load_json(results_dir / "task_shift_recovery_results.json")
        plot_task_shift_recovery(task_shift_results, output_dir)
    except Exception as e:
        print(f"Could not generate task shift plots: {e}")
    
    try:
        ablation_50 = load_json(results_dir / "ablation_sparsity50%_epochs2.json")
        ablation_70 = load_json(results_dir / "ablation_sparsity70%_epochs3.json")
        plot_ablation_comparison(ablation_50, ablation_70, output_dir)
    except Exception as e:
        print(f"Could not generate ablation plots: {e}")
    
    try:
        baseline_results = load_json(results_dir / "baseline_comparison.json")
        plot_method_comparison_detailed(baseline_results, output_dir)
        plot_parameter_breakdown(baseline_results, output_dir)
    except Exception as e:
        print(f"Could not generate method comparison plots: {e}")
    
    # Generate summary report
    try:
        create_summary_report(results_dir, output_dir)
    except Exception as e:
        print(f"Could not generate summary report: {e}")
    
    print("=" * 60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nAll graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
