#!/bin/bash
#
# Setup script for HARAM-VLM Real-World Validation Framework
# This script installs dependencies and downloads required datasets
#

set -e  # Exit on error

echo "=================================================="
echo "HARAM-VLM Real-World Validation Framework Setup"
echo "=================================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected (>= $required_version required)"
else
    echo "✗ Python $python_version detected. Python $required_version or higher is required."
    exit 1
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"

# Install requirements
echo
echo "Installing requirements..."
echo "This may take several minutes..."

# Install PyTorch first (CPU version by default)
echo "  Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA detected, installing GPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "  No CUDA detected, installing CPU version..."
    pip install torch torchvision -q
fi

# Install other requirements
echo "  Installing other packages..."
pip install -r requirements.txt -q
echo "✓ All requirements installed"

# Check for COCO dataset
echo
echo "Checking for COCO dataset..."
coco_dir="./data/coco/val2014"
sample_dir="./data/coco/samples"

if [ -d "$coco_dir" ]; then
    num_images=$(find "$coco_dir" -name "*.jpg" | wc -l)
    echo "✓ COCO val2014 found with $num_images images"
elif [ -d "$sample_dir" ]; then
    num_samples=$(find "$sample_dir" -name "*.jpg" | wc -l)
    echo "✓ COCO samples found with $num_samples images"
else
    echo "✗ No COCO images found"
    echo
    read -p "Would you like to download sample images for testing? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading sample images..."
        python scripts/download_coco.py --samples-only --num-samples 20
    else
        echo "Skipping COCO download."
        echo "You can download later with:"
        echo "  python scripts/download_coco.py"
    fi
fi

# Run quick test
echo
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo
echo "To run a quick test:"
echo "  python quick_start.py --test"
echo
echo "To download full COCO dataset (6.6GB):"
echo "  python scripts/download_coco.py --year 2014"
echo
echo "To run full validation:"
echo "  python run_validation.py --help"
echo

# Optional: Run quick dependency check
read -p "Would you like to run a quick dependency check? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python quick_start.py --check
fi