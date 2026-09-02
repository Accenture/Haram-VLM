"""
Visualization Utilities for POPE Validation Results
===================================================

Creates plots and visualizations for analysis of multi-model validation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def set_plot_style():
    """Set consistent plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_hallucination_vs_resolution(results: Dict,
                                    save_path: Optional[str] = None):
    """
    Plot hallucination rate vs resolution for all models.

    Args:
        results: Validation results dictionary
        save_path: Path to save the plot
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    for model_key, model_results in results["models"].items():
        if "error" in model_results:
            continue

        resolutions = []
        hallucination_rates = []

        for res, data in model_results["resolutions"].items():
            resolutions.append(int(res))
            hallucination_rates.append(data["metrics"]["hallucination_rate"] * 100)

        # Sort by resolution
        sorted_data = sorted(zip(resolutions, hallucination_rates))
        resolutions, hallucination_rates = zip(*sorted_data)

        ax.plot(resolutions, hallucination_rates,
               marker='o', linewidth=2, markersize=8,
               label=model_key.replace('-', ' ').title())

    ax.set_xlabel("Resolution (pixels)", fontsize=12)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12)
    ax.set_title("Hallucination Rate vs Resolution Across Models", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add optimal zone shading
    ax.axhspan(0, 10, alpha=0.1, color='green', label='Target Zone (<10%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_performance_tradeoff(results: Dict,
                             save_path: Optional[str] = None):
    """
    Plot the tradeoff between accuracy and efficiency.

    Args:
        results: Validation results dictionary
        save_path: Path to save the plot
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model_key, model_results in results["models"].items():
        if "error" in model_results:
            continue

        tokens = []
        accuracies = []
        inference_times = []

        for res, data in model_results["resolutions"].items():
            tokens.append(data["performance"]["avg_tokens"])
            accuracies.append(data["metrics"]["accuracy"] * 100)
            inference_times.append(data["performance"]["avg_inference_time"])

        # Plot 1: Accuracy vs Tokens
        ax1.plot(tokens, accuracies,
                marker='o', linewidth=2, markersize=8,
                label=model_key.replace('-', ' ').title())

        # Plot 2: Accuracy vs Inference Time
        ax2.plot(inference_times, accuracies,
                marker='s', linewidth=2, markersize=8,
                label=model_key.replace('-', ' ').title())

    # Configure subplot 1
    ax1.set_xlabel("Visual Tokens", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Token Usage", fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Configure subplot 2
    ax2.set_xlabel("Inference Time (seconds)", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Inference Time", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_correlation_matrix(analysis: Dict,
                           save_path: Optional[str] = None):
    """
    Plot correlation matrix for resolution vs metrics.

    Args:
        analysis: Analysis results dictionary
        save_path: Path to save the plot
    """
    set_plot_style()

    # Extract correlations
    correlations = []
    models = []

    for model_key, corr_data in analysis["correlations"].items():
        models.append(model_key.replace('-', ' ').title())
        corr_val = corr_data["resolution_vs_hallucination"]["correlation"]
        correlations.append(corr_val)

    # Create dataframe
    df = pd.DataFrame({
        'Model': models,
        'Resolution vs Hallucination': correlations
    })

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df['Model'], df['Resolution vs Hallucination'])

    # Color bars based on correlation strength
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        if abs(corr) > 0.7:
            bar.set_color('darkgreen' if corr < 0 else 'darkred')
        elif abs(corr) > 0.5:
            bar.set_color('green' if corr < 0 else 'red')
        else:
            bar.set_color('gray')

    ax.set_ylabel("Pearson Correlation Coefficient", fontsize=12)
    ax.set_title("Resolution vs Hallucination Correlation by Model",
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-0.3, color='green', linestyle='--', alpha=0.5,
              label='Target Correlation (< -0.3)')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{corr:.3f}',
               ha='center', va='bottom' if height > 0 else 'top')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def create_comparison_plots(results: Dict, analysis: Dict,
                          output_dir: str = "./plots"):
    """
    Create all comparison plots.

    Args:
        results: Validation results
        analysis: Analysis results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Hallucination vs Resolution
    plot_hallucination_vs_resolution(
        results,
        save_path=output_dir / "hallucination_vs_resolution.png"
    )

    # Plot 2: Performance Tradeoff
    plot_performance_tradeoff(
        results,
        save_path=output_dir / "performance_tradeoff.png"
    )

    # Plot 3: Correlation Matrix
    if analysis.get("correlations"):
        plot_correlation_matrix(
            analysis,
            save_path=output_dir / "correlation_matrix.png"
        )

    print(f"\nAll plots saved to {output_dir}")


def create_summary_table(results: Dict, analysis: Dict) -> pd.DataFrame:
    """
    Create a summary table of results.

    Args:
        results: Validation results
        analysis: Analysis results

    Returns:
        DataFrame with summary
    """
    summary_data = []

    for model_key in results["models"]:
        if "error" in results["models"][model_key]:
            continue

        model_data = {
            "Model": model_key.replace('-', ' ').title()
        }

        # Add optimal resolution info
        if model_key in analysis["optimal_resolutions"]:
            opt = analysis["optimal_resolutions"][model_key]
            model_data["Optimal Resolution"] = f"{opt['resolution']}px"
            model_data["Best Hallucination Rate"] = f"{opt['hallucination_rate']:.2%}"
            model_data["Best Accuracy"] = f"{opt['accuracy']:.2%}"

        # Add correlation
        if model_key in analysis["correlations"]:
            corr = analysis["correlations"][model_key]["resolution_vs_hallucination"]["correlation"]
            model_data["Correlation"] = f"{corr:.3f}"

        summary_data.append(model_data)

    return pd.DataFrame(summary_data)