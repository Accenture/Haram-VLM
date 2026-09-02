#!/usr/bin/env python3
"""
Quick Framework Test without Large Models
==========================================

Tests the validation framework using mock models to verify everything works
without needing to download multi-GB models.
"""

import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from datasets.pope_dataset import POPEDataset, POPEMetrics
from PIL import Image


def mock_model_inference(image_path: str, question: str, resolution: int) -> dict:
    """
    Mock model that simulates hallucination patterns based on resolution.

    Simulates the hypothesis: higher resolution → lower hallucination
    """
    # Simulate processing time (faster on CPU for mock)
    time.sleep(0.1)

    # Base hallucination probability
    base_hall_rate = 0.25

    # Adjust based on resolution (simulating the hypothesis)
    resolution_factor = {
        224: 1.2,   # 20% worse
        336: 1.0,   # baseline
        448: 0.8,   # 20% better
        560: 0.6,   # 40% better
        672: 0.5,   # 50% better
        896: 0.7,   # Worse again (attention diffusion)
    }

    factor = resolution_factor.get(resolution, 1.0)
    hall_probability = base_hall_rate * factor

    # For "Is there X in the image?" questions
    # Randomly answer yes/no with hallucination probability
    if "Is there" in question:
        # Ground truth would be 50/50 in POPE balanced set
        # Add hallucination bias toward "yes"
        if random.random() < hall_probability:
            answer = "yes"  # Hallucinated yes
        else:
            answer = "yes" if random.random() < 0.5 else "no"
    else:
        answer = "yes" if random.random() < 0.5 else "no"

    return {
        "text": answer,
        "tokens": (resolution // 14) ** 2,  # Approximate vision tokens
        "inference_time": 0.1
    }


def test_framework():
    """Run a quick test of the validation framework"""
    print("="*60)
    print("HARAM-VLM Framework Test (Mock Models)")
    print("="*60)
    print("\nThis test verifies the framework without downloading large models.\n")

    # Initialize POPE dataset
    print("1. Loading POPE dataset...")
    pope = POPEDataset(
        data_dir="./data/pope",
        coco_image_dir="./data/coco/samples"
    )

    # Get some test questions
    questions = pope.create_balanced_subset(
        category="random",
        size=10,
        yes_ratio=0.5
    )

    print(f"✓ Loaded {len(questions)} POPE questions")

    # Test different resolutions
    resolutions = [224, 336, 448, 672]
    results = {}

    print("\n2. Running mock validation across resolutions...")
    print("-" * 40)

    for resolution in resolutions:
        predictions = []
        tokens_used = []
        times = []

        print(f"\nResolution: {resolution}px")

        for q in questions:
            # Mock inference
            result = mock_model_inference(
                q.image_path,
                q.question,
                resolution
            )

            predictions.append(result["text"])
            tokens_used.append(result["tokens"])
            times.append(result["inference_time"])

        # Calculate metrics
        metrics = pope.evaluate_predictions(predictions, questions)

        results[resolution] = {
            "accuracy": metrics.accuracy,
            "hallucination_rate": metrics.hallucination_rate,
            "avg_tokens": np.mean(tokens_used),
            "avg_time": np.mean(times)
        }

        print(f"  Accuracy: {metrics.accuracy:.2f}")
        print(f"  Hallucination Rate: {metrics.hallucination_rate:.2f}")
        print(f"  Tokens: {np.mean(tokens_used):.0f}")

    # Analyze correlation
    print("\n3. Analyzing correlation (resolution vs hallucination)...")
    print("-" * 40)

    res_list = list(results.keys())
    hall_rates = [results[r]["hallucination_rate"] for r in res_list]

    # Calculate correlation
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(res_list, hall_rates)

    print(f"  Pearson Correlation: r = {correlation:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Find optimal resolution
    optimal_res = min(results, key=lambda r: results[r]["hallucination_rate"])
    print(f"\n  Optimal Resolution: {optimal_res}px")
    print(f"  Best Hallucination Rate: {results[optimal_res]['hallucination_rate']:.2f}")

    # Save results
    print("\n4. Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "test_type": "mock_validation",
        "timestamp": timestamp,
        "resolutions": res_list,
        "results": results,
        "correlation": {
            "pearson_r": correlation,
            "p_value": p_value,
            "optimal_resolution": optimal_res
        }
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"mock_test_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {output_file}")

    # Summary
    print("\n" + "="*60)
    print("Framework Test Complete!")
    print("="*60)
    print("\n✅ All components working:")
    print("  - POPE dataset loader: OK")
    print("  - Metrics calculation: OK")
    print("  - Multi-resolution testing: OK")
    print("  - Correlation analysis: OK")
    print("  - Results export: OK")

    print("\n📊 Mock Results Summary:")
    print(f"  - Tested {len(resolutions)} resolutions")
    print(f"  - Correlation: r = {correlation:.3f}")
    print(f"  - Optimal: {optimal_res}px")

    if correlation < -0.3:
        print("\n✅ Hypothesis SUPPORTED (negative correlation found)")
    else:
        print("\n❌ Hypothesis NOT supported in mock (expected)")

    print("\n📝 Next Steps:")
    print("  1. Wait for Phi-3 Vision download to complete")
    print("  2. Or install a smaller model for real testing")
    print("  3. Or download the full COCO dataset for comprehensive evaluation")

    return True


if __name__ == "__main__":
    success = test_framework()
    sys.exit(0 if success else 1)