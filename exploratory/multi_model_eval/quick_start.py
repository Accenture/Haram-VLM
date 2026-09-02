#!/usr/bin/env python3
"""
Quick Start Script for HARAM-VLM Validation
===========================================

Simplified interface to run common validation scenarios.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available, will use CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} installed")
        version_parts = transformers.__version__.split('.')
        if int(version_parts[0]) >= 4 and int(version_parts[1]) >= 57:
            print("✓ Transformers version supports Qwen3-VL")
        else:
            print("ℹ Transformers version may not support Qwen3-VL (need 4.57+)")
    except ImportError:
        print("✗ Transformers not installed")
        return False

    try:
        import PIL
        print("✓ PIL installed")
    except ImportError:
        print("✗ PIL not installed")
        return False

    return True


def run_quick_test():
    """Run a quick test with minimal samples"""
    print("\n" + "="*60)
    print("Running Quick Test (5 samples, 1 model, 3 resolutions)")
    print("="*60)

    cmd = [
        "python", "run_validation.py",
        "--models", "qwen3-2b",
        "--resolutions", "224", "448", "672",
        "--num-samples", "5",
        "--pope-category", "random"
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_hypothesis_test():
    """Run hypothesis validation test"""
    print("\n" + "="*60)
    print("Running Hypothesis Validation")
    print("="*60)

    cmd = [
        "python", "run_validation.py",
        "--models", "qwen3-2b", "phi3-vision",
        "--resolutions", "224", "336", "448", "560", "672", "784", "896",
        "--num-samples", "50",
        "--pope-category", "adversarial"
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_full_validation():
    """Run complete validation with all models"""
    print("\n" + "="*60)
    print("Running Full Validation (All Models)")
    print("="*60)

    cmd = [
        "python", "run_validation.py",
        "--num-samples", "100",
        "--pope-category", "adversarial"
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def analyze_latest_results():
    """Analyze the most recent results"""
    print("\n" + "="*60)
    print("Analyzing Latest Results")
    print("="*60)

    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found. Run validation first.")
        return False

    # Find latest results file
    result_files = list(results_dir.glob("validation_results_*.json"))
    if not result_files:
        print("No results found. Run validation first.")
        return False

    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)

    # Print summary
    print("\n" + "-"*40)
    print("Summary of Results")
    print("-"*40)

    for model_key, model_results in results["models"].items():
        if "error" in model_results:
            print(f"\n{model_key}: ERROR - {model_results['error']}")
            continue

        print(f"\n{model_key}:")

        best_acc = 0
        best_res = 0
        lowest_hall = 1.0
        optimal_res = 0

        for res, data in model_results["resolutions"].items():
            acc = data["metrics"]["accuracy"]
            hall = data["metrics"]["hallucination_rate"]

            if acc > best_acc:
                best_acc = acc
                best_res = int(res)

            if hall < lowest_hall:
                lowest_hall = hall
                optimal_res = int(res)

        print(f"  Best Accuracy: {best_acc:.3f} at {best_res}px")
        print(f"  Lowest Hallucination: {lowest_hall:.3f} at {optimal_res}px")

    return True


def download_coco_samples():
    """Download sample COCO images for testing"""
    print("\n" + "="*60)
    print("Downloading COCO Sample Images")
    print("="*60)

    # Create data directory
    data_dir = Path("data/coco_samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download a few sample images
    import requests
    sample_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "http://images.cocodataset.org/val2017/000000000285.jpg"
    ]

    for url in sample_urls:
        filename = url.split('/')[-1]
        filepath = data_dir / filename

        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        try:
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

    print(f"\nSample images saved to: {data_dir}")
    return True


def main():
    """Main entry point for quick start"""
    parser = argparse.ArgumentParser(
        description="Quick start for HARAM-VLM validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --check        # Check dependencies
  python quick_start.py --test         # Run quick test
  python quick_start.py --hypothesis   # Test hypothesis
  python quick_start.py --full         # Full validation
  python quick_start.py --analyze      # Analyze results
        """
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if dependencies are installed"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test (5 samples)"
    )

    parser.add_argument(
        "--hypothesis",
        action="store_true",
        help="Run hypothesis validation test"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation (all models)"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze latest results"
    )

    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Download sample COCO images"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps in sequence"
    )

    args = parser.parse_args()

    # Default to check if no args
    if not any(vars(args).values()):
        args.check = True

    print("="*60)
    print("HARAM-VLM Validation Framework - Quick Start")
    print("="*60)

    success = True

    if args.all or args.check:
        if not check_dependencies():
            print("\n⚠ Please install missing dependencies:")
            print("  pip install -r requirements.txt")
            return 1

    if args.all or args.download_samples:
        success = download_coco_samples() and success

    if args.all or args.test:
        if not check_dependencies():
            print("⚠ Dependencies not satisfied")
            return 1
        success = run_quick_test() and success

    if args.all or args.hypothesis:
        if not check_dependencies():
            print("⚠ Dependencies not satisfied")
            return 1
        success = run_hypothesis_test() and success

    if args.full:
        if not check_dependencies():
            print("⚠ Dependencies not satisfied")
            return 1
        success = run_full_validation() and success

    if args.all or args.analyze:
        success = analyze_latest_results() and success

    if success:
        print("\n✅ All requested operations completed successfully!")
    else:
        print("\n⚠ Some operations failed. Check output above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())