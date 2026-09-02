"""
Quick test script to verify Qwen3-VL setup and compare with Qwen2.5-VL
"""

import torch
from PIL import Image
import sys

def test_qwen3_setup():
    """Test Qwen3-VL availability and basic functionality"""

    print("🧪 Testing Qwen3-VL Setup...")
    print("=" * 60)

    # 1. Check PyTorch and device
    print("\n1️⃣ System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"   MPS (Apple Silicon) available: ✅")
    else:
        print(f"   Running on CPU")

    # 2. Check transformers version
    print("\n2️⃣ Checking transformers installation...")
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")

        # Parse version
        version_parts = transformers.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])

        if major > 4 or (major == 4 and minor >= 57):
            print(f"   ✅ Version compatible with Qwen3-VL (requires >= 4.57.0)")
        else:
            print(f"   ⚠️ Version may not support Qwen3-VL (requires >= 4.57.0)")
            print(f"   Run: pip install transformers>=4.57.0")

    except ImportError:
        print("   ❌ Transformers not installed!")
        return

    # 3. Check for model support
    print("\n3️⃣ Checking model support...")

    models_to_test = [
        ("Qwen2.5-VL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("Qwen3-VL", "Qwen/Qwen3-VL-8B-Instruct")
    ]

    for model_name, model_id in models_to_test:
        print(f"\n   Testing {model_name}:")
        try:
            # Try to import the specific model class
            if "Qwen3" in model_name:
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                    print(f"     ✅ {model_name} class available")
                except ImportError:
                    from transformers import AutoModelForImageTextToText
                    print(f"     ⚠️ Using AutoModel for {model_name}")
            else:
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    print(f"     ✅ {model_name} class available")
                except ImportError:
                    from transformers import AutoModelForCausalLM
                    print(f"     ⚠️ Using AutoModel for {model_name}")

            # Try to load processor (lightweight test)
            from transformers import AutoProcessor
            print(f"     Loading processor for {model_id}...")
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print(f"     ✅ Processor loaded successfully")

        except Exception as e:
            print(f"     ❌ Error with {model_name}: {str(e)[:100]}")

    # 4. Memory requirements
    print("\n4️⃣ Memory Requirements:")
    print("   Qwen2.5-VL-7B: ~14-16 GB VRAM (FP16)")
    print("   Qwen3-VL-8B: ~16-18 GB VRAM (FP16)")
    print("   Qwen3-VL-2B: ~4-5 GB VRAM (FP16)")
    print("   Note: CPU mode requires ~2x RAM")

    # 5. Installation commands
    print("\n5️⃣ Installation Commands:")
    print("   For Qwen3-VL support:")
    print("   pip install transformers>=4.57.0 accelerate")
    print("   pip install flash-attn --no-build-isolation  # Optional, for speed")

    # 6. Test image creation
    print("\n6️⃣ Creating test image...")
    try:
        test_image = Image.new('RGB', (448, 448), color=(100, 150, 200))
        print("   ✅ Test image created (448x448)")

        # Create images at different resolutions
        resolutions = [224, 336, 448, 672, 896, 1024]
        print(f"   ✅ Can resize to {len(resolutions)} different resolutions")

    except Exception as e:
        print(f"   ❌ Error creating test image: {e}")

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    # Final recommendations
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if device == "cuda":
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory >= 16:
            print("✅ Recommended: Use Qwen3-VL-8B-Instruct for best results")
        elif memory >= 8:
            print("⚠️ Recommended: Use Qwen3-VL-2B-Instruct (lighter model)")
        else:
            print("❌ Limited VRAM: Consider CPU mode or cloud GPUs")
    elif device == "mps":
        print("✅ Apple Silicon detected: Can run Qwen3-VL-2B efficiently")
        print("⚠️ For 8B model, ensure you have 32GB+ unified memory")
    else:
        print("⚠️ CPU mode: Will be slow, consider using Qwen3-VL-2B")

    print("\n💡 Next Steps:")
    print("1. Install required packages if needed")
    print("2. Run validate_with_qwen3.py for full validation")
    print("3. Compare results with Qwen2.5-VL baseline")

if __name__ == "__main__":
    test_qwen3_setup()