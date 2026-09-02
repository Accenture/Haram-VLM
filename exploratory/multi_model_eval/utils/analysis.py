"""
Analysis Utilities for POPE Validation Results
===============================================

Statistical analysis and correlation calculations for validation results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def calculate_correlations(resolutions: List[int],
                         hallucination_rates: List[float]) -> Dict:
    """
    Calculate various correlation metrics.

    Args:
        resolutions: List of resolutions
        hallucination_rates: Corresponding hallucination rates

    Returns:
        Dictionary with correlation metrics
    """
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(resolutions, hallucination_rates)

    # Spearman correlation (rank-based, more robust to outliers)
    spearman_r, spearman_p = stats.spearmanr(resolutions, hallucination_rates)

    # Kendall's Tau (another rank correlation)
    kendall_tau, kendall_p = stats.kendalltau(resolutions, hallucination_rates)

    return {
        "pearson": {
            "correlation": pearson_r,
            "p_value": pearson_p,
            "significant": pearson_p < 0.05
        },
        "spearman": {
            "correlation": spearman_r,
            "p_value": spearman_p,
            "significant": spearman_p < 0.05
        },
        "kendall": {
            "correlation": kendall_tau,
            "p_value": kendall_p,
            "significant": kendall_p < 0.05
        }
    }


def perform_statistical_tests(model_results: Dict) -> Dict:
    """
    Perform statistical tests on model results.

    Args:
        model_results: Results for all models

    Returns:
        Dictionary with test results
    """
    tests = {}

    # Extract data for all models
    model_data = {}
    for model_key, results in model_results.items():
        if "error" in results:
            continue

        model_data[model_key] = {
            "resolutions": [],
            "hallucination_rates": [],
            "accuracies": []
        }

        for res, data in results["resolutions"].items():
            model_data[model_key]["resolutions"].append(int(res))
            model_data[model_key]["hallucination_rates"].append(
                data["metrics"]["hallucination_rate"]
            )
            model_data[model_key]["accuracies"].append(
                data["metrics"]["accuracy"]
            )

    # ANOVA test across models at each resolution
    if len(model_data) > 2:
        resolution_set = set()
        for data in model_data.values():
            resolution_set.update(data["resolutions"])

        for resolution in sorted(resolution_set):
            hall_rates = []
            for model_key, data in model_data.items():
                if resolution in data["resolutions"]:
                    idx = data["resolutions"].index(resolution)
                    hall_rates.append(data["hallucination_rates"][idx])

            if len(hall_rates) >= 3:
                f_stat, p_value = stats.f_oneway(*hall_rates)
                tests[f"anova_{resolution}px"] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }

    # Paired t-tests between consecutive resolutions
    for model_key, data in model_data.items():
        resolutions = data["resolutions"]
        hall_rates = data["hallucination_rates"]

        # Sort by resolution
        sorted_pairs = sorted(zip(resolutions, hall_rates))
        sorted_res, sorted_hall = zip(*sorted_pairs)

        tests[f"{model_key}_resolution_tests"] = {}

        for i in range(len(sorted_res) - 1):
            res1, res2 = sorted_res[i], sorted_res[i+1]
            hall1, hall2 = sorted_hall[i], sorted_hall[i+1]

            # Since we only have single values, we can't do paired t-test
            # Instead, calculate the difference
            diff = hall2 - hall1
            tests[f"{model_key}_resolution_tests"][f"{res1}vs{res2}"] = {
                "difference": diff,
                "improvement": diff < 0,
                "percent_change": (diff / hall1 * 100) if hall1 > 0 else 0
            }

    return tests


def find_optimal_resolution(resolutions: List[int],
                           hallucination_rates: List[float],
                           token_counts: List[int]) -> Dict:
    """
    Find the optimal resolution balancing accuracy and efficiency.

    Args:
        resolutions: List of resolutions
        hallucination_rates: Corresponding hallucination rates
        token_counts: Corresponding token counts

    Returns:
        Dictionary with optimal resolution analysis
    """
    # Normalize metrics
    hall_norm = np.array(hallucination_rates) / max(hallucination_rates)
    token_norm = np.array(token_counts) / max(token_counts)

    # Calculate efficiency score (lower is better)
    # Weight hallucination more heavily than token count
    efficiency_scores = 0.7 * hall_norm + 0.3 * token_norm

    # Find optimal
    optimal_idx = np.argmin(efficiency_scores)

    return {
        "optimal_resolution": resolutions[optimal_idx],
        "hallucination_rate": hallucination_rates[optimal_idx],
        "token_count": token_counts[optimal_idx],
        "efficiency_score": efficiency_scores[optimal_idx],
        "all_scores": list(zip(resolutions, efficiency_scores.tolist()))
    }


def analyze_attention_diffusion(attention_entropies: Dict[int, float]) -> Dict:
    """
    Analyze attention diffusion across resolutions.

    Args:
        attention_entropies: Dictionary mapping resolution to entropy

    Returns:
        Analysis of attention patterns
    """
    resolutions = sorted(attention_entropies.keys())
    entropies = [attention_entropies[r] for r in resolutions]

    # Find where entropy starts increasing (diffusion point)
    diffusion_point = None
    for i in range(1, len(entropies)):
        if entropies[i] > entropies[i-1] * 1.1:  # 10% increase threshold
            diffusion_point = resolutions[i]
            break

    # Calculate rate of entropy increase
    if len(resolutions) > 1:
        entropy_slope, _ = np.polyfit(resolutions, entropies, 1)
    else:
        entropy_slope = 0

    return {
        "diffusion_point": diffusion_point,
        "entropy_slope": entropy_slope,
        "min_entropy_resolution": resolutions[np.argmin(entropies)],
        "max_entropy_resolution": resolutions[np.argmax(entropies)],
        "entropy_range": max(entropies) - min(entropies)
    }


def compare_models(model_results: Dict) -> pd.DataFrame:
    """
    Create a comparison table of all models.

    Args:
        model_results: Results for all models

    Returns:
        DataFrame with model comparison
    """
    comparison_data = []

    for model_key, results in model_results.items():
        if "error" in results:
            continue

        # Calculate aggregated metrics
        hall_rates = []
        accuracies = []
        inference_times = []
        token_counts = []

        for res, data in results["resolutions"].items():
            hall_rates.append(data["metrics"]["hallucination_rate"])
            accuracies.append(data["metrics"]["accuracy"])
            inference_times.append(data["performance"]["avg_inference_time"])
            token_counts.append(data["performance"]["avg_tokens"])

        comparison_data.append({
            "Model": model_key,
            "Avg Hallucination Rate": np.mean(hall_rates),
            "Min Hallucination Rate": np.min(hall_rates),
            "Avg Accuracy": np.mean(accuracies),
            "Max Accuracy": np.max(accuracies),
            "Avg Inference Time": np.mean(inference_times),
            "Avg Token Count": np.mean(token_counts)
        })

    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values("Avg Hallucination Rate")

    return df


def calculate_hypothesis_support(analysis: Dict) -> Dict:
    """
    Calculate how well the results support the HARAM-VLM hypothesis.

    Args:
        analysis: Analysis results

    Returns:
        Dictionary with hypothesis support metrics
    """
    support_metrics = {
        "correlation_support": 0,
        "optimal_resolution_consistency": 0,
        "diffusion_evidence": 0,
        "overall_support": 0
    }

    # Check correlation support
    correlation_count = 0
    strong_correlation_count = 0

    for model_key, corr_data in analysis.get("correlations", {}).items():
        corr = corr_data["resolution_vs_hallucination"]["correlation"]
        if corr < -0.3:  # Negative correlation as expected
            correlation_count += 1
        if corr < -0.5:  # Strong negative correlation
            strong_correlation_count += 1

    total_models = len(analysis.get("correlations", {}))
    if total_models > 0:
        support_metrics["correlation_support"] = correlation_count / total_models
        support_metrics["strong_correlation_ratio"] = strong_correlation_count / total_models

    # Check optimal resolution consistency
    optimal_resolutions = []
    for model_key, opt_data in analysis.get("optimal_resolutions", {}).items():
        optimal_resolutions.append(opt_data["resolution"])

    if optimal_resolutions:
        # Check if optimal resolutions fall in expected range (336-672)
        in_range = sum(1 for r in optimal_resolutions if 336 <= r <= 672)
        support_metrics["optimal_resolution_consistency"] = in_range / len(optimal_resolutions)

        # Calculate standard deviation of optimal resolutions
        support_metrics["optimal_resolution_std"] = np.std(optimal_resolutions)

    # Overall support score
    support_metrics["overall_support"] = np.mean([
        support_metrics["correlation_support"],
        support_metrics["optimal_resolution_consistency"]
    ])

    # Interpretation
    if support_metrics["overall_support"] > 0.7:
        support_metrics["interpretation"] = "Strong support for hypothesis"
    elif support_metrics["overall_support"] > 0.5:
        support_metrics["interpretation"] = "Moderate support for hypothesis"
    else:
        support_metrics["interpretation"] = "Weak support for hypothesis"

    return support_metrics