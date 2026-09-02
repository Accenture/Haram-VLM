#!/usr/bin/env python3
"""
POPE Evaluation Script for HARAM-VLM Hypothesis Testing
========================================================

Runs comprehensive POPE evaluation across multiple resolutions
to test the resolution-hallucination hypothesis.
"""

import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run POPE evaluation for HARAM-VLM hypothesis testing"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 samples"
    )

    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium test with 50 samples"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full evaluation with 200+ samples"
    )

    args = parser.parse_args()

    # Check if COCO 2014 is available
    coco_dir = Path("data/coco/val2014")
    if not coco_dir.exists():
        print("❌ COCO 2014 dataset not found!")
        print("Download is in progress. Please wait for it to complete.")
        print("\nYou can check download status with:")
        print("  python3 scripts/download_coco.py --info")
        return 1

    # Count images
    num_images = len(list(coco_dir.glob("*.jpg")))
    print(f"✅ Found {num_images:,} COCO images")

    # Determine test size
    if args.full:
        num_samples = 200
        resolutions = [224, 336, 448, 560, 672, 784, 896]
        print("\n🔬 Running FULL evaluation (200 samples, 7 resolutions)")
        print("⏱️  Estimated time on CPU: 2-3 hours")
    elif args.medium:
        num_samples = 50
        resolutions = [224, 448, 672, 896]
        print("\n🔬 Running MEDIUM evaluation (50 samples, 4 resolutions)")
        print("⏱️  Estimated time on CPU: 30-45 minutes")
    else:  # Quick test (default)
        num_samples = 10
        resolutions = [224, 448, 672]
        print("\n🔬 Running QUICK evaluation (10 samples, 3 resolutions)")
        print("⏱️  Estimated time on CPU: 5-10 minutes")

    # Build command
    import subprocess

    cmd = [
        "python3", "run_validation.py",
        "--models", "phi3-vision",
        "--resolutions", *[str(r) for r in resolutions],
        "--num-samples", str(num_samples),
        "--pope-category", "adversarial",  # Hardest split
        "--coco-image-dir", str(coco_dir)
    ]

    print(f"\n📋 Configuration:")
    print(f"  Model: Phi-3 Vision (4.2B)")
    print(f"  Samples: {num_samples}")
    print(f"  Resolutions: {resolutions}")
    print(f"  POPE Split: Adversarial (hardest)")
    print(f"  Device: CPU")

    print("\n" + "="*60)
    print("Starting POPE Evaluation")
    print("="*60)

    # Run evaluation
    start_time = time.time()
    result = subprocess.run(cmd)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Evaluation completed in {elapsed/60:.1f} minutes")

    if result.returncode == 0:
        print("\n✅ Success! Check the results/ directory for outputs:")
        print("  - validation_results_*.json - Raw data")
        print("  - analysis_*.json - Statistical analysis")
        print("  - summary_*.csv - Summary table")

        # Try to show summary
        try:
            import json
            import glob

            # Find latest analysis file
            analysis_files = glob.glob("results/analysis_*.json")
            if analysis_files:
                latest = max(analysis_files)
                with open(latest, 'r') as f:
                    analysis = json.load(f)

                print("\n📊 Quick Summary:")
                if "correlations" in analysis and "phi3-vision" in analysis["correlations"]:
                    corr = analysis["correlations"]["phi3-vision"]["resolution_vs_hallucination"]["correlation"]
                    p_val = analysis["correlations"]["phi3-vision"]["resolution_vs_hallucination"]["p_value"]
                    print(f"  Correlation: r = {corr:.3f}, p = {p_val:.4f}")

                    if corr < -0.3:
                        print("  ✅ Hypothesis SUPPORTED (negative correlation)")
                    else:
                        print("  ❌ Hypothesis NOT supported")

                if "optimal_resolutions" in analysis and "phi3-vision" in analysis["optimal_resolutions"]:
                    opt = analysis["optimal_resolutions"]["phi3-vision"]
                    print(f"  Optimal Resolution: {opt['resolution']}px")
                    print(f"  Best Hallucination Rate: {opt['hallucination_rate']:.2%}")
        except Exception as e:
            print(f"Could not load summary: {e}")

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())