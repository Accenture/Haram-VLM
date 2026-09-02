"""
Utility Functions for Analysis and Visualization
================================================
"""

from .visualization import (
    plot_hallucination_vs_resolution,
    plot_performance_tradeoff,
    plot_correlation_matrix,
    create_comparison_plots,
    create_summary_table
)

from .analysis import (
    calculate_correlations,
    perform_statistical_tests,
    find_optimal_resolution,
    analyze_attention_diffusion,
    compare_models,
    calculate_hypothesis_support
)

__all__ = [
    # Visualization
    'plot_hallucination_vs_resolution',
    'plot_performance_tradeoff',
    'plot_correlation_matrix',
    'create_comparison_plots',
    'create_summary_table',
    # Analysis
    'calculate_correlations',
    'perform_statistical_tests',
    'find_optimal_resolution',
    'analyze_attention_diffusion',
    'compare_models',
    'calculate_hypothesis_support'
]