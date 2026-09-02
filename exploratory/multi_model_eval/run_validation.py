#!/usr/bin/env python3
"""
Unified Multi-Model POPE Validation Runner
==========================================

Main script to run hallucination validation across multiple VLM models
using the POPE benchmark at different resolutions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import traceback

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from datasets.pope_dataset import POPEDataset, POPEMetrics
from models.base_model import ModelConfig, ModelOutput
from models.qwen_model import QwenVLMWrapper
from models.llava_model import LLaVAWrapper
from models.phi3_model import Phi3VisionWrapper
from utils.visualization import create_comparison_plots, plot_correlation_matrix
from utils.analysis import calculate_correlations, perform_statistical_tests


class MultiModelValidator:
    """
    Orchestrates validation across multiple models and resolutions.
    """

    def __init__(self,
                 pope_data_dir: str = "./data/pope",
                 coco_image_dir: Optional[str] = None,
                 results_dir: str = "./results",
                 cache_dir: Optional[str] = None):
        """
        Initialize validator.

        Args:
            pope_data_dir: Directory for POPE dataset
            coco_image_dir: Path to COCO images
            results_dir: Directory to save results
            cache_dir: Model cache directory
        """
        self.pope_data_dir = pope_data_dir
        self.coco_image_dir = coco_image_dir
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize POPE dataset
        self.pope_dataset = POPEDataset(
            data_dir=pope_data_dir,
            coco_image_dir=coco_image_dir,
            download=True
        )

        # Available models registry
        self.available_models = {
            "qwen3-2b": {
                "class": QwenVLMWrapper,
                "model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "description": "Qwen3-VL 2B - Lightweight"
            },
            "qwen2.5-7b": {
                "class": QwenVLMWrapper,
                "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "description": "Qwen2.5-VL 7B - Baseline"
            },
            "llava-1.5-7b": {
                "class": LLaVAWrapper,
                "model_name": "llava-hf/llava-1.5-7b-hf",
                "description": "LLaVA 1.5 7B - Classic"
            },
            "llava-1.6-7b": {
                "class": LLaVAWrapper,
                "model_name": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "description": "LLaVA 1.6 7B - Latest"
            },
            "phi3-vision": {
                "class": Phi3VisionWrapper,
                "model_name": "microsoft/Phi-3-vision-128k-instruct",
                "description": "Phi-3 Vision 4.2B - Edge-optimized"
            }
        }

        self.results = {}

    def load_model(self, model_key: str, device: str = "auto") -> any:
        """
        Load a specific model.

        Args:
            model_key: Key from available_models
            device: Device to use (cuda, cpu, auto)

        Returns:
            Model wrapper instance
        """
        if model_key not in self.available_models:
            raise ValueError(f"Unknown model: {model_key}")

        model_info = self.available_models[model_key]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = ModelConfig(
            model_name=model_info["model_name"],
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            temperature=0.0,  # Greedy for consistency
            cache_dir=self.cache_dir
        )

        print(f"\nLoading {model_info['description']}...")
        model_class = model_info["class"]

        try:
            return model_class(config)
        except Exception as e:
            print(f"Error loading {model_key}: {e}")
            return None

    def validate_single_model(self,
                            model_key: str,
                            resolutions: List[int],
                            pope_category: str = "adversarial",
                            num_samples: int = 100,
                            device: str = "auto") -> Dict:
        """
        Validate a single model across multiple resolutions.

        Args:
            model_key: Model to validate
            resolutions: List of resolutions to test
            pope_category: POPE split to use
            num_samples: Number of samples per resolution
            device: Device to use

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Validating {model_key}")
        print(f"{'='*60}")

        # Load model
        model = self.load_model(model_key, device)
        if model is None:
            return {"error": "Failed to load model"}

        # Get POPE questions
        questions = self.pope_dataset.create_balanced_subset(
            category=pope_category,
            size=num_samples
        )

        print(f"Using {len(questions)} questions from POPE {pope_category} split")

        results = {
            "model": model_key,
            "pope_category": pope_category,
            "num_samples": num_samples,
            "resolutions": {}
        }

        # Test each resolution
        for resolution in resolutions:
            print(f"\nTesting resolution: {resolution}px")

            predictions = []
            inference_times = []
            tokens_used = []
            attention_entropies = []

            # Process questions
            pbar = tqdm(questions, desc=f"Resolution {resolution}px")
            for question in pbar:
                # Load image
                image = self.pope_dataset.load_image(question.image_path)
                if image is None:
                    predictions.append("no")  # Default if image not found
                    continue

                try:
                    # Run inference
                    start_time = time.time()
                    output = model.process(
                        image,
                        question.question,
                        resolution=resolution,
                        return_attention=False,
                        return_metrics=True
                    )
                    inference_time = time.time() - start_time

                    # Parse answer from response
                    response = output.text.lower()
                    if "yes" in response and "no" not in response:
                        answer = "yes"
                    elif "no" in response:
                        answer = "no"
                    else:
                        answer = "no"  # Default

                    predictions.append(answer)
                    inference_times.append(inference_time)

                    if output.tokens_used:
                        tokens_used.append(output.tokens_used)

                    if output.attention_maps is not None:
                        entropy = model.get_attention_entropy(output.attention_maps)
                        attention_entropies.append(entropy)

                except Exception as e:
                    print(f"Error processing image: {e}")
                    predictions.append("no")
                    inference_times.append(0)

            # Calculate metrics
            metrics = self.pope_dataset.evaluate_predictions(predictions, questions)

            # Store results
            results["resolutions"][resolution] = {
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "yes_ratio": metrics.yes_ratio,
                    "hallucination_rate": metrics.hallucination_rate,
                },
                "confusion_matrix": {
                    "tp": metrics.true_positive,
                    "tn": metrics.true_negative,
                    "fp": metrics.false_positive,
                    "fn": metrics.false_negative
                },
                "performance": {
                    "avg_inference_time": np.mean(inference_times),
                    "std_inference_time": np.std(inference_times),
                    "avg_tokens": np.mean(tokens_used) if tokens_used else 0,
                    "avg_attention_entropy": np.mean(attention_entropies) if attention_entropies else 0
                }
            }

            # Print summary
            print(f"  Accuracy: {metrics.accuracy:.3f}")
            print(f"  Hallucination Rate: {metrics.hallucination_rate:.3f}")
            print(f"  Avg Inference Time: {np.mean(inference_times):.3f}s")

        # Cleanup model
        model.cleanup()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def validate_all_models(self,
                           models: Optional[List[str]] = None,
                           resolutions: List[int] = [224, 336, 448, 672, 896],
                           pope_category: str = "adversarial",
                           num_samples: int = 100) -> Dict:
        """
        Validate all specified models.

        Args:
            models: List of model keys (None for all)
            resolutions: Resolutions to test
            pope_category: POPE split
            num_samples: Samples per resolution

        Returns:
            All results
        """
        if models is None:
            models = list(self.available_models.keys())

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "resolutions": resolutions,
                "pope_category": pope_category,
                "num_samples": num_samples
            },
            "models": {}
        }

        for model_key in models:
            try:
                results = self.validate_single_model(
                    model_key,
                    resolutions,
                    pope_category,
                    num_samples
                )
                all_results["models"][model_key] = results
            except Exception as e:
                print(f"Failed to validate {model_key}: {e}")
                traceback.print_exc()
                all_results["models"][model_key] = {"error": str(e)}

        return all_results

    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze validation results for patterns and correlations.

        Args:
            results: Validation results

        Returns:
            Analysis dictionary
        """
        analysis = {
            "correlations": {},
            "optimal_resolutions": {},
            "model_comparison": {}
        }

        for model_key, model_results in results["models"].items():
            if "error" in model_results:
                continue

            resolutions = []
            hallucination_rates = []
            accuracies = []
            inference_times = []
            tokens = []

            for res, data in model_results["resolutions"].items():
                resolutions.append(int(res))
                hallucination_rates.append(data["metrics"]["hallucination_rate"])
                accuracies.append(data["metrics"]["accuracy"])
                inference_times.append(data["performance"]["avg_inference_time"])
                tokens.append(data["performance"]["avg_tokens"])

            # Calculate correlations
            if len(resolutions) > 1:
                from scipy.stats import pearsonr

                # Resolution vs hallucination
                corr_hall, p_hall = pearsonr(resolutions, hallucination_rates)
                analysis["correlations"][model_key] = {
                    "resolution_vs_hallucination": {
                        "correlation": corr_hall,
                        "p_value": p_hall
                    }
                }

                # Find optimal resolution (lowest hallucination)
                optimal_idx = np.argmin(hallucination_rates)
                analysis["optimal_resolutions"][model_key] = {
                    "resolution": resolutions[optimal_idx],
                    "hallucination_rate": hallucination_rates[optimal_idx],
                    "accuracy": accuracies[optimal_idx]
                }

        return analysis

    def save_results(self, results: Dict, analysis: Dict):
        """Save results and analysis to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = self.results_dir / f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

        # Save analysis
        analysis_file = self.results_dir / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_file}")

        # Create summary DataFrame
        summary_data = []
        for model_key, model_results in results["models"].items():
            if "error" in model_results:
                continue

            for res, data in model_results["resolutions"].items():
                summary_data.append({
                    "model": model_key,
                    "resolution": int(res),
                    "accuracy": data["metrics"]["accuracy"],
                    "hallucination_rate": data["metrics"]["hallucination_rate"],
                    "inference_time": data["performance"]["avg_inference_time"],
                    "tokens": data["performance"]["avg_tokens"]
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.results_dir / f"summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to {csv_file}")

        return results_file, analysis_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run multi-model POPE validation for HARAM-VLM hypothesis testing"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen3-2b", "qwen2.5-7b", "llava-1.5-7b", "llava-1.6-7b", "phi3-vision"],
        help="Models to validate (default: all)"
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[224, 336, 448, 672, 896],
        help="Resolutions to test"
    )

    parser.add_argument(
        "--pope-category",
        choices=["random", "popular", "adversarial"],
        default="adversarial",
        help="POPE split to use (default: adversarial - hardest)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per resolution (default: 100)"
    )

    parser.add_argument(
        "--coco-image-dir",
        type=str,
        help="Path to COCO val2014 images"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./model_cache",
        help="Model cache directory"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = MultiModelValidator(
        coco_image_dir=args.coco_image_dir,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir
    )

    # Run validation
    print("\n" + "="*60)
    print("HARAM-VLM Multi-Model Validation")
    print("="*60)
    print(f"Models: {args.models or 'all'}")
    print(f"Resolutions: {args.resolutions}")
    print(f"POPE Category: {args.pope_category}")
    print(f"Samples per resolution: {args.num_samples}")
    print(f"Device: {args.device}")
    print("="*60)

    # Validate models
    results = validator.validate_all_models(
        models=args.models,
        resolutions=args.resolutions,
        pope_category=args.pope_category,
        num_samples=args.num_samples
    )

    # Analyze results
    print("\n" + "="*60)
    print("Analyzing Results")
    print("="*60)

    analysis = validator.analyze_results(results)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for model_key, model_analysis in analysis["optimal_resolutions"].items():
        print(f"\n{model_key}:")
        print(f"  Optimal Resolution: {model_analysis['resolution']}px")
        print(f"  Hallucination Rate: {model_analysis['hallucination_rate']:.3f}")
        print(f"  Accuracy: {model_analysis['accuracy']:.3f}")

    print("\nCorrelations (Resolution vs Hallucination):")
    for model_key, corr_data in analysis["correlations"].items():
        corr = corr_data["resolution_vs_hallucination"]["correlation"]
        p_val = corr_data["resolution_vs_hallucination"]["p_value"]
        print(f"  {model_key}: r={corr:.3f}, p={p_val:.4f}")

    # Save results
    validator.save_results(results, analysis)

    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()