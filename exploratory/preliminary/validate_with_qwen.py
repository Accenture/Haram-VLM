"""
HARAM-VLM: Resolution-Hallucination Validation with Qwen2.5-VL-7B
Tests the correlation between image resolution and hallucination rates
"""

import torch
import numpy as np
import json
import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Check if transformers has Qwen2.5-VL support
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    print("✅ Using Qwen2VL from transformers")
except ImportError:
    print("⚠️ Qwen2VL not found in transformers, using AutoModel")
    from transformers import AutoModelForCausalLM as Qwen2VLForConditionalGeneration
    from transformers import AutoProcessor as Qwen2VLProcessor

from datasets import load_dataset

@dataclass
class ExperimentConfig:
    """Configuration for validation experiment"""
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    resolutions: List[int] = None
    num_samples: int = 100
    batch_size: int = 1
    output_dir: str = "./validation_results"
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 42
    max_new_tokens: int = 256
    temperature: float = 0.7

    def __post_init__(self):
        if self.resolutions is None:
            # Test multiple resolutions
            self.resolutions = [224, 336, 448, 672, 896, 1024]

class HallucinationEvaluator:
    """Evaluates hallucinations in VLM outputs"""

    def __init__(self):
        # Load COCO dataset for ground truth
        print("📊 Loading evaluation datasets...")
        self.coco_objects = self._load_coco_categories()

    def _load_coco_categories(self):
        """Load COCO object categories"""
        # Standard COCO-80 categories
        categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return set(categories)

    def evaluate_hallucination(self,
                              generated_text: str,
                              ground_truth: Dict,
                              image_id: str) -> Dict:
        """
        Evaluate hallucinations in generated text

        Args:
            generated_text: Model output
            ground_truth: Ground truth annotations
            image_id: Image identifier

        Returns:
            Dictionary with hallucination metrics
        """
        # Extract mentioned objects from generated text
        mentioned_objects = self._extract_objects(generated_text.lower())

        # Get ground truth objects
        gt_objects = set(ground_truth.get('objects', []))

        # Calculate hallucinations
        hallucinated_objects = mentioned_objects - gt_objects
        correct_objects = mentioned_objects & gt_objects
        missed_objects = gt_objects - mentioned_objects

        # Calculate metrics
        hallucination_rate = len(hallucinated_objects) / max(len(mentioned_objects), 1)
        precision = len(correct_objects) / max(len(mentioned_objects), 1)
        recall = len(correct_objects) / max(len(gt_objects), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return {
            'image_id': image_id,
            'hallucination_rate': hallucination_rate,
            'num_hallucinated': len(hallucinated_objects),
            'hallucinated_objects': list(hallucinated_objects),
            'correct_objects': list(correct_objects),
            'missed_objects': list(missed_objects),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'generated_length': len(generated_text.split())
        }

    def _extract_objects(self, text: str) -> set:
        """Extract object mentions from text"""
        found_objects = set()

        # Simple extraction - look for COCO objects in text
        for obj in self.coco_objects:
            if obj in text:
                found_objects.add(obj)

        # Also check for plural forms
        for obj in self.coco_objects:
            if obj + 's' in text or obj + 'es' in text:
                found_objects.add(obj)

        return found_objects

class Qwen2VLExperiment:
    """Main experiment class for testing Qwen2.5-VL"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = HallucinationEvaluator()
        self.results = {}

        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        print(f"🔧 Device: {config.device}")

    def load_model(self):
        """Load Qwen2.5-VL model and processor"""
        print(f"📥 Loading {self.config.model_name}...")

        try:
            # Load processor
            self.processor = Qwen2VLProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            # Load model with appropriate precision
            if self.config.device == "cuda":
                # Use bfloat16 for GPU
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Use float32 for CPU/MPS
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    device_map="auto",
                    trust_remote_code=True
                )

            self.model.eval()
            print(f"✅ Model loaded on {self.config.device}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try running: pip install transformers>=4.40.0 accelerate qwen-vl-utils")
            raise

    def load_dataset(self):
        """Load test dataset"""
        print("📚 Loading COCO validation dataset...")

        try:
            # Load COCO validation split
            dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)

            # Convert to list and sample
            samples = []
            for i, item in enumerate(dataset):
                if i >= self.config.num_samples:
                    break

                # Extract image and annotations
                image = item['image']

                # Get object annotations
                objects = []
                if 'objects' in item and item['objects']:
                    for obj in item['objects']['category']:
                        if obj < len(self.evaluator.coco_objects):
                            objects.append(list(self.evaluator.coco_objects)[obj])

                samples.append({
                    'image': image,
                    'objects': objects,
                    'image_id': str(i)
                })

            print(f"✅ Loaded {len(samples)} samples")
            return samples

        except Exception as e:
            print(f"⚠️ Error loading COCO: {e}")
            print("Using synthetic dataset instead...")
            return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self):
        """Create synthetic dataset for testing"""
        # Create sample images with known objects
        samples = []

        for i in range(min(10, self.config.num_samples)):
            # Create a simple test image
            image = Image.new('RGB', (512, 512), color=(255, 255, 255))

            samples.append({
                'image': image,
                'objects': ['person', 'dog', 'car'],  # Known objects
                'image_id': f'synthetic_{i}'
            })

        return samples

    def process_image_at_resolution(self, image: Image.Image, resolution: int) -> Image.Image:
        """Resize image to specific resolution"""
        # Qwen2-VL supports dynamic resolution, but we'll test specific sizes
        aspect_ratio = image.width / image.height

        if aspect_ratio > 1:
            new_width = resolution
            new_height = int(resolution / aspect_ratio)
        else:
            new_height = resolution
            new_width = int(resolution * aspect_ratio)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def generate_description(self, image: Image.Image, resolution: int) -> Tuple[str, float]:
        """Generate description for image at specific resolution"""

        # Resize image
        processed_image = self.process_image_at_resolution(image, resolution)

        # Create prompt for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail. List all objects you can see."}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[processed_image],
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        inference_time = time.time() - start_time

        # Decode output
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract only the model's response
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()

        return generated_text, inference_time

    def run_experiment(self):
        """Run the complete validation experiment"""
        print("\n" + "="*60)
        print("🚀 Starting HARAM-VLM Hypothesis Validation")
        print("="*60)

        # Load model
        self.load_model()

        # Load dataset
        dataset = self.load_dataset()

        # Test each resolution
        for resolution in self.config.resolutions:
            print(f"\n📐 Testing resolution: {resolution}x{resolution}")

            resolution_results = []

            for sample in tqdm(dataset[:20], desc=f"Resolution {resolution}"):
                try:
                    # Generate description
                    description, inference_time = self.generate_description(
                        sample['image'],
                        resolution
                    )

                    # Evaluate hallucination
                    eval_metrics = self.evaluator.evaluate_hallucination(
                        description,
                        {'objects': sample['objects']},
                        sample['image_id']
                    )

                    # Add timing and resolution info
                    eval_metrics['resolution'] = resolution
                    eval_metrics['inference_time'] = inference_time
                    eval_metrics['tokens'] = (resolution // 14) ** 2  # Approximate

                    resolution_results.append(eval_metrics)

                except Exception as e:
                    print(f"⚠️ Error processing image: {e}")
                    continue

            # Store results
            self.results[resolution] = resolution_results

            # Print summary
            if resolution_results:
                avg_hallucination = np.mean([r['hallucination_rate'] for r in resolution_results])
                avg_time = np.mean([r['inference_time'] for r in resolution_results])
                print(f"   Average hallucination rate: {avg_hallucination:.1%}")
                print(f"   Average inference time: {avg_time:.2f}s")

        # Analyze and save results
        self._analyze_results()
        self._save_results()
        self._generate_plots()

    def _analyze_results(self):
        """Analyze correlation between resolution and hallucination"""
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)

        resolutions = []
        hallucination_rates = []
        inference_times = []

        print(f"\n{'Resolution':<12} {'Hallucination':<15} {'Tokens':<10} {'Time (s)':<10}")
        print("-"*50)

        for resolution in sorted(self.results.keys()):
            if self.results[resolution]:
                hall_rate = np.mean([r['hallucination_rate'] for r in self.results[resolution]])
                inf_time = np.mean([r['inference_time'] for r in self.results[resolution]])
                tokens = (resolution // 14) ** 2

                resolutions.append(resolution)
                hallucination_rates.append(hall_rate)
                inference_times.append(inf_time)

                print(f"{resolution:<12} {hall_rate:<15.1%} {tokens:<10} {inf_time:<10.2f}")

        # Calculate correlation
        if len(resolutions) > 1:
            correlation = np.corrcoef(resolutions, hallucination_rates)[0, 1]
            print(f"\n🔬 Pearson Correlation (resolution vs hallucination): {correlation:.3f}")

            if abs(correlation) > 0.3:
                print("✅ Strong correlation detected! Hypothesis supported.")
            else:
                print("⚠️ Weak correlation. Need more investigation.")

            # Find optimal resolution
            optimal_idx = np.argmin(hallucination_rates)
            print(f"🎯 Optimal resolution: {resolutions[optimal_idx]}px")

            # Check for attention diffusion
            if len(hallucination_rates) > 4:
                if hallucination_rates[-1] > hallucination_rates[-2]:
                    print("⚠️ Attention diffusion detected at high resolution!")

    def _save_results(self):
        """Save experiment results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_dir / f"qwen_validation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': self.results
            }, f, indent=2, default=str)

        print(f"\n📁 Results saved to: {results_file}")

    def _generate_plots(self):
        """Generate visualization plots"""
        if not self.results:
            return

        output_dir = Path(self.config.output_dir)

        # Prepare data
        resolutions = []
        hallucination_rates = []
        inference_times = []

        for resolution in sorted(self.results.keys()):
            if self.results[resolution]:
                resolutions.append(resolution)
                hallucination_rates.append(
                    np.mean([r['hallucination_rate'] for r in self.results[resolution]])
                )
                inference_times.append(
                    np.mean([r['inference_time'] for r in self.results[resolution]])
                )

        if not resolutions:
            return

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Hallucination vs Resolution
        axes[0].plot(resolutions, hallucination_rates, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Resolution (pixels)', fontsize=12)
        axes[0].set_ylabel('Hallucination Rate', fontsize=12)
        axes[0].set_title('Hallucination vs Resolution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Inference Time vs Resolution
        axes[1].plot(resolutions, inference_times, 'r-s', linewidth=2, markersize=8)
        axes[1].set_xlabel('Resolution (pixels)', fontsize=12)
        axes[1].set_ylabel('Inference Time (s)', fontsize=12)
        axes[1].set_title('Inference Time vs Resolution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Trade-off Analysis
        tokens = [(r // 14) ** 2 for r in resolutions]
        axes[2].scatter(tokens, hallucination_rates, s=100, c='green', alpha=0.6)
        for i, res in enumerate(resolutions):
            axes[2].annotate(f'{res}px', (tokens[i], hallucination_rates[i]),
                           fontsize=9, ha='center')
        axes[2].set_xlabel('Visual Tokens', fontsize=12)
        axes[2].set_ylabel('Hallucination Rate', fontsize=12)
        axes[2].set_title('Efficiency vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('HARAM-VLM Hypothesis Validation with Qwen2.5-VL', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"qwen_validation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
        plt.show()

def main():
    """Main entry point"""
    print("\n🎯 HARAM-VLM Validation with Qwen2.5-VL-7B")
    print("="*60)

    # Configuration
    config = ExperimentConfig(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        resolutions=[224, 336, 448, 672],  # Multiple resolutions to test
        num_samples=20,  # Start with small sample for testing
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Resolutions: {config.resolutions}")
    print(f"   Samples: {config.num_samples}")
    print(f"   Device: {config.device}")

    # Run experiment
    experiment = Qwen2VLExperiment(config)
    experiment.run_experiment()

    print("\n✅ Validation complete!")
    print("📝 Check ./validation_results/ for detailed results")

if __name__ == "__main__":
    main()