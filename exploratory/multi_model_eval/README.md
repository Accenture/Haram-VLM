# Multi-model resolution/hallucination harness

> **Exploratory — not in the paper.** This standalone harness predates the paper's
> controller and clean protocol. Its runs are small-sample and single-seed, and **no
> number in the paper comes from it**. It is the lineage of the Fig. 1 motivation curve;
> see [`../../figures/README.md`](../../figures/README.md) for why that curve is
> illustrative only. Scripts that produced *simulated* rather than measured results, and
> the plots that mixed the two, have been removed from this release.
>
> To reproduce the paper, use [`../../protocol/`](../../protocol/) and
> [`../../controller/`](../../controller/) instead.

## 🎯 Overview

A comprehensive, modular framework for validating the HARAM-VLM hypothesis across multiple Vision-Language Models (VLMs) using real-world benchmarks. This framework tests whether adaptive resolution based on hallucination risk can improve both accuracy and efficiency in VLMs.

### Key Features

- **Multi-Model Support**: Test across diverse VLM architectures (Qwen, LLaVA, Phi-3, etc.)
- **POPE Benchmark Integration**: Automatic download and evaluation on POPE dataset
- **Resolution Analysis**: Test models at multiple resolutions (224px - 1152px)
- **Statistical Analysis**: Correlation analysis, hypothesis testing, and more
- **Visualization**: Automatic generation of plots and comparison charts
- **Modular Design**: Easy to extend with new models and benchmarks

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support, optional)
- 16GB+ RAM (32GB+ recommended for larger models)
- 50GB+ disk space for models and datasets

### Quick Setup

```bash
# Clone the repository
cd exploratory/multi_model_eval

# Option 1: Automated setup (recommended)
./scripts/setup.sh  # Handles everything automatically

# Option 2: Manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download COCO images (choose one):
# - Quick testing (20 sample images, ~5MB)
python scripts/download_coco.py --samples-only

# - Full POPE evaluation (40K images, ~6.6GB)
python scripts/download_coco.py --year 2014

# - Smaller alternative (5K images, ~778MB)
python scripts/download_coco.py --year 2017
```

## 🚀 Quick Start

### Basic Validation Run

```bash
# Test a single model with default settings
python run_validation.py --models qwen3-2b --num-samples 50

# Test multiple models
python run_validation.py --models qwen3-2b llava-1.6-7b phi3-vision

# Full validation with all models and resolutions
python run_validation.py --num-samples 100 --pope-category adversarial
```

### Custom Resolution Testing

```bash
# Test specific resolutions
python run_validation.py \
    --models qwen3-2b \
    --resolutions 224 448 672 896 \
    --num-samples 100
```

### With COCO Images

```bash
# Specify path to COCO val2014 images
python run_validation.py \
    --coco-image-dir /path/to/coco/val2014 \
    --models llava-1.6-7b \
    --num-samples 200
```

## 📊 Framework Structure

```
exploratory/multi_model_eval/
├── datasets/                # Dataset handlers
│   ├── __init__.py
│   └── pope_dataset.py     # POPE benchmark loader
│
├── models/                  # Model wrappers
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class
│   ├── qwen_model.py       # Qwen2/3-VL wrapper
│   ├── llava_model.py      # LLaVA 1.5/1.6 wrapper
│   └── phi3_model.py       # Phi-3 Vision wrapper
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── visualization.py    # Plotting functions
│   └── analysis.py         # Statistical analysis
│
├── results/                 # Output directory
│   ├── validation_results_*.json
│   ├── analysis_*.json
│   └── plots/
│
├── scripts/                 # Utility scripts
│   ├── setup.sh            # Automated setup script
│   ├── download_coco.py    # COCO dataset downloader
│   └── README.md           # Scripts documentation
│
├── run_validation.py        # Main validation script
├── quick_start.py          # Simplified interface
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Supported Models

| Model | Size | Key Features | Status |
|-------|------|--------------|--------|
| **Qwen3-VL-2B** | 2B | Latest, efficient | ✅ Implemented |
| **Qwen2.5-VL-7B** | 7B | Strong baseline | ✅ Implemented |
| **LLaVA-1.5-7B** | 7B | Classic, widely used | ✅ Implemented |
| **LLaVA-1.6-7B** | 7B | Multi-resolution support | ✅ Implemented |
| **Phi-3-Vision** | 4.2B | Edge-optimized | ✅ Implemented |
| **InternVL2** | 2-8B | SOTA Chinese models | 🔄 Coming Soon |
| **CogVLM2** | 19B | Strong on hallucination | 🔄 Coming Soon |

## 📈 Metrics & Analysis

### Primary Metrics

- **Hallucination Rate**: False positive rate on object detection
- **Accuracy**: Overall correctness on POPE questions
- **Yes-Bias**: Tendency to over-predict object presence
- **Inference Time**: Processing time per image
- **Token Usage**: Number of visual tokens used

### Statistical Tests

- **Pearson Correlation**: Linear relationship between resolution and hallucination
- **Spearman Correlation**: Rank-based correlation (robust to outliers)
- **ANOVA**: Compare models across resolutions
- **Optimal Resolution**: Balance between accuracy and efficiency

## 🎨 Visualization

The framework automatically generates:

1. **Hallucination vs Resolution Plot**: Shows how hallucination changes with resolution
2. **Performance Tradeoff Plot**: Accuracy vs tokens/time
3. **Correlation Matrix**: Statistical correlations by model
4. **Summary Tables**: Comparative metrics across all models

## 💻 Advanced Usage

### Adding a New Model

1. Create a new wrapper in `models/`:

```python
from models.base_model import BaseVLMWrapper, ModelConfig, ModelOutput

class MyModelWrapper(BaseVLMWrapper):
    def _load_model(self):
        # Load your model
        pass

    def _preprocess_image(self, image, resolution):
        # Preprocess image
        pass

    def _generate_response(self, image_input, text_prompt, return_attention=False):
        # Generate response
        pass
```

2. Register in `run_validation.py`:

```python
self.available_models["my-model"] = {
    "class": MyModelWrapper,
    "model_name": "organization/model-name",
    "description": "My Model Description"
}
```

### Custom Analysis

```python
from exploratory.multi_model_eval.utils.analysis import calculate_hypothesis_support
import json

# Load results
with open("results/validation_results_*.json", "r") as f:
    results = json.load(f)

# Analyze
with open("results/analysis_*.json", "r") as f:
    analysis = json.load(f)

# Calculate hypothesis support
support = calculate_hypothesis_support(analysis)
print(f"Hypothesis Support: {support['overall_support']:.2%}")
print(f"Interpretation: {support['interpretation']}")
```

## 📊 Expected Results

Based on the HARAM-VLM hypothesis, you should observe:

1. **Negative Correlation**: Resolution ↑ → Hallucination ↓ (up to a point)
2. **Optimal Range**: Best performance typically at 448-672px
3. **Attention Diffusion**: Performance degradation beyond 896-1024px
4. **Model Consistency**: Pattern should hold across different architectures

### Example Results

```
Model: Qwen3-VL-2B
  224px: Hallucination Rate = 22.4%, Tokens = 256
  448px: Hallucination Rate = 13.1%, Tokens = 1024
  672px: Hallucination Rate = 7.1%, Tokens = 2304  ← Optimal
  896px: Hallucination Rate = 8.2%, Tokens = 4096

Correlation: r = -0.85, p < 0.001 ✅
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
```bash
# Use smaller batch size or lower resolution
python run_validation.py --models qwen3-2b --resolutions 224 336
```

2. **COCO Images Not Found**
```bash
# Ensure correct path to val2014 folder
python run_validation.py --coco-image-dir /absolute/path/to/val2014
```

3. **Model Download Failed**
```bash
# Use cache directory with more space
python run_validation.py --cache-dir /path/with/space
```

## 📝 Output Files

After validation, you'll find:

- `results/validation_results_TIMESTAMP.json`: Raw validation data
- `results/analysis_TIMESTAMP.json`: Statistical analysis
- `results/summary_TIMESTAMP.csv`: Summary table
- `results/plots/`: Generated visualizations

## 🤝 Contributing

To contribute new models or benchmarks:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## 📚 References

- [POPE Benchmark](https://github.com/AoiDragon/POPE)
- [COCO Dataset](https://cocodataset.org/)
- [Repository README](../../README.md) — what the paper actually claims
- [Figure provenance](../../figures/README.md) — why the Fig. 1 curve is illustrative

## 📄 License

Apache License 2.0 — see [LICENSE](../../LICENSE) at the repository root.

## 🙋 Support

For issues or questions:
- Check the [main documentation](../README.md)
- Read [../README.md](../README.md) for what in this tree is and is not paper-backed
- Open an issue with detailed error messages and system info

---

**Note**: This framework is designed for research validation. For production deployment, additional optimizations and error handling may be required.