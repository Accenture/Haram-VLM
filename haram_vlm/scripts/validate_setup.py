#!/usr/bin/env python3
"""
Simple validation script to check if HARAM-VLM training setup is working
"""

import json
import os
import sys
from pathlib import Path

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def validate_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")

    dependencies = {
        'torch': False,
        'transformers': False,
        'peft': False,
        'PIL': False,
    }

    try:
        import torch
        dependencies['torch'] = True
        print(f"  ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not found")

    try:
        import transformers
        dependencies['transformers'] = True
        print(f"  ✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("  ❌ Transformers not found")

    try:
        from peft import LoraConfig
        dependencies['peft'] = True
        print("  ✅ PEFT available")
    except ImportError:
        print("  ❌ PEFT not found")

    try:
        from PIL import Image
        dependencies['PIL'] = True
        print("  ✅ PIL/Pillow available")
    except ImportError:
        print("  ❌ PIL/Pillow not found")

    return all(dependencies.values())


def validate_data(data_path: str, image_dir: str):
    """Check if training data and images are accessible."""
    print(f"\n📊 Validating training data...")

    if not os.path.exists(data_path):
        print(f"  ❌ Training data not found: {data_path}")
        return False

    # Load and check data format
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  ✅ Loaded {len(data)} training samples")

    # Check first sample
    if data:
        sample = data[0]
        required_keys = ['id', 'image', 'conversations']

        for key in required_keys:
            if key not in sample:
                print(f"  ❌ Missing required key in data: {key}")
                return False

        print(f"  ✅ Data format is correct")

        # Check if image exists
        image_path = sample['image']
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_dir, image_path)

        if os.path.exists(image_path):
            print(f"  ✅ Sample image accessible: {os.path.basename(image_path)}")
        else:
            print(f"  ⚠️  Sample image not found: {image_path}")
            # Not a critical error for validation

    return True


def validate_model_loading():
    """Test if we can load the model configuration."""
    print(f"\n🤖 Validating model loading...")

    try:
        from transformers import AutoConfig

        # Just try to load the config, not the full model
        config = AutoConfig.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct",
            trust_remote_code=True
        )
        print(f"  ✅ Model config loaded successfully")
        print(f"     Model type: {config.model_type}")
        print(f"     Hidden size: {config.hidden_size}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to load model config: {e}")
        return False


def validate_training_imports():
    """Test if training modules can be imported."""
    print(f"\n📦 Validating training modules...")

    try:
        from training.params import ModelArguments, DataArguments, TrainingArguments
        print(f"  ✅ Training parameters imported")
    except Exception as e:
        print(f"  ❌ Failed to import training parameters: {e}")
        return False

    try:
        from training.data import make_supervised_data_module
        print(f"  ✅ Data module imported")
    except Exception as e:
        print(f"  ⚠️  Failed to import data module: {e}")
        # Not critical for basic validation

    try:
        from training.trainer import Phi3VTrainer
        print(f"  ✅ Trainer imported")
    except Exception as e:
        print(f"  ⚠️  Failed to import trainer: {e}")
        # Not critical for basic validation

    return True


def main():
    print("=" * 50)
    print("🚀 HARAM-VLM Training Setup Validation")
    print("=" * 50)

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "test_train.json"
    image_dir = os.environ.get("HARAM_IMAGE_DIR", HARAM_ROOT + "/coco_build/images")

    # Run validations
    results = {
        'dependencies': validate_dependencies(),
        'data': validate_data(str(data_path), image_dir),
        'model': validate_model_loading(),
        'imports': validate_training_imports(),
    }

    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)

    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {check.capitalize():15} {status}")

    if all(results.values()):
        print("\n🎉 All validations passed! Training setup is ready.")
        print("\n📝 Next steps:")
        print("  1. Smoke test:      bash scripts/setup_and_smoke_test.sh")
        print("  2. Full training:    bash scripts/run_full_training_4gpu.sh")
        return 0
    else:
        print("\n⚠️  Some validations failed. Please fix the issues above.")
        print("\n📝 Critical components that must pass:")
        print("  - Dependencies (PyTorch, Transformers, PEFT)")
        print("  - Data format and accessibility")
        print("  - Model configuration loading")
        return 1


if __name__ == "__main__":
    sys.exit(main())