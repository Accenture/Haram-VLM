"""
Quick test script to verify Qwen2.5-VL setup
"""

import torch
from PIL import Image
import requests
from io import BytesIO

def test_qwen_setup():
    """Quick test to verify model loading and basic inference"""

    print("🧪 Testing Qwen2.5-VL Setup...")
    print("-" * 50)

    # 1. Check PyTorch
    print("1️⃣ Checking PyTorch installation...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"   MPS (Apple Silicon) available: ✅")
    else:
        print(f"   Running on CPU")

    # 2. Check transformers
    print("\n2️⃣ Checking transformers installation...")
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")

        # Check for Qwen2VL support
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            print("   ✅ Qwen2VL support available")
            model_class = Qwen2VLForConditionalGeneration
            processor_class = Qwen2VLProcessor
        except ImportError:
            print("   ⚠️ Qwen2VL not found, trying AutoModel...")
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_class = AutoModelForCausalLM
            processor_class = AutoProcessor

    except ImportError:
        print("   ❌ Transformers not installed!")
        print("   Run: pip install transformers>=4.40.0")
        return

    # 3. Test model loading (lightweight test)
    print("\n3️⃣ Testing model loading...")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    try:
        print(f"   Loading processor from {model_name}...")
        processor = processor_class.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("   ✅ Processor loaded successfully")

        # Note: Loading full model requires significant memory
        print("\n   ⚠️ Full model loading requires ~15GB RAM")
        print("   To load the model, uncomment the code below")

        # Uncomment to actually load the model
        # print(f"   Loading model from {model_name}...")
        # model = model_class.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        # print("   ✅ Model loaded successfully")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n   💡 Troubleshooting:")
        print("      1. Make sure you have enough disk space (~15GB)")
        print("      2. Install required packages:")
        print("         pip install transformers>=4.40.0 accelerate qwen-vl-utils")
        print("      3. If on Mac, you might need to use CPU mode")
        return

    # 4. Test with sample image
    print("\n4️⃣ Testing with sample image...")
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
        print("   ✅ Created test image (red square)")

        # Or download a sample image
        # url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        # response = requests.get(url)
        # test_image = Image.open(BytesIO(response.content))
        # print("   ✅ Downloaded sample image")

    except Exception as e:
        print(f"   ❌ Error creating test image: {e}")

    print("\n" + "="*50)
    print("✅ Basic setup test complete!")
    print("\nNext steps:")
    print("1. Uncomment model loading code if you have enough memory")
    print("2. Run validate_with_qwen.py for full validation")
    print("3. Or use the lighter test_resolution_hypothesis.py")

def check_dependencies():
    """Check all required dependencies"""
    print("\n📦 Checking dependencies...")

    packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'accelerate': 'Accelerate',
        'PIL': 'Pillow',
        'datasets': 'Hugging Face Datasets',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }

    missing = []
    for package, name in packages.items():
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing.append(package)

    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    else:
        print("\n✅ All dependencies installed!")

    return len(missing) == 0

if __name__ == "__main__":
    print("="*60)
    print("HARAM-VLM: Quick Setup Test")
    print("="*60)

    # Check dependencies first
    if check_dependencies():
        # Run setup test
        test_qwen_setup()
    else:
        print("\n❌ Please install missing dependencies first")

    print("\n" + "="*60)