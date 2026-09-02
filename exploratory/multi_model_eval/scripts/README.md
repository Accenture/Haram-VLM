# Validation Framework Scripts

This folder contains utility scripts for setting up and managing the HARAM-VLM validation framework.

## Available Scripts

### 1. `setup.sh` - Complete Setup Script
Automated setup for the entire validation framework.

```bash
./scripts/setup.sh
```

**Features:**
- Creates Python virtual environment
- Installs all dependencies (PyTorch, transformers, etc.)
- Checks for CUDA and installs appropriate PyTorch version
- Downloads sample COCO images (optional)
- Runs dependency verification

### 2. `download_coco.py` - COCO Dataset Downloader

Download COCO validation images for POPE benchmark evaluation.

#### Quick Examples:

**Download sample images for testing (recommended for quick start):**
```bash
python scripts/download_coco.py --samples-only
```

**Download full COCO 2014 validation set (6.6GB, required for POPE):**
```bash
python scripts/download_coco.py --year 2014
```

**Download COCO 2017 validation set (smaller, 778MB):**
```bash
python scripts/download_coco.py --year 2017
```

#### Full Options:
```bash
python scripts/download_coco.py [OPTIONS]

Options:
  --output-dir PATH       Directory to save COCO data (default: ./data/coco)
  --year {2014,2017}      COCO dataset year (default: 2014)
  --samples-only          Download only sample images for testing
  --num-samples N         Number of samples (default: 10)
  --with-annotations      Also download COCO annotations
  --force                 Force re-download even if files exist
  --info                  Show dataset information only
```

## Setup Workflow

### Option 1: Quick Setup (Recommended)
```bash
# Run complete setup
./scripts/setup.sh

# This will:
# 1. Check Python version
# 2. Create virtual environment
# 3. Install all dependencies
# 4. Optionally download sample images
# 5. Run dependency check
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download COCO samples
python scripts/download_coco.py --samples-only

# 4. Run quick test
python quick_start.py --test
```

### Option 3: Full Dataset Setup
```bash
# For complete POPE evaluation with full dataset
# Note: This downloads 6.6GB of data

# 1. Run basic setup
./scripts/setup.sh

# 2. Download full COCO 2014 validation set
python scripts/download_coco.py --year 2014

# 3. Run full validation
python run_validation.py --coco-image-dir data/coco/val2014
```

## Dataset Information

### COCO 2014 (Recommended for POPE)
- **Size**: ~6.6 GB
- **Images**: 40,504 validation images
- **Used by**: POPE benchmark
- **Download**: `python scripts/download_coco.py --year 2014`

### COCO 2017 (Alternative, smaller)
- **Size**: ~778 MB
- **Images**: 5,000 validation images
- **Download**: `python scripts/download_coco.py --year 2017`

### Sample Images (Quick testing)
- **Size**: ~2-5 MB
- **Images**: 10-20 sample images
- **Download**: `python scripts/download_coco.py --samples-only`

## Troubleshooting

### Issue: Download interrupted
```bash
# Resume download (automatic)
python scripts/download_coco.py --year 2014

# Or force re-download
python scripts/download_coco.py --year 2014 --force
```

### Issue: Not enough disk space
```bash
# Download only samples for testing
python scripts/download_coco.py --samples-only

# Or use different output directory
python scripts/download_coco.py --output-dir /path/with/space
```

### Issue: Slow download
```bash
# The COCO servers can be slow. Consider:
# 1. Download samples first for testing
# 2. Download full dataset overnight
# 3. Use a download manager with the URLs shown
```

### Issue: Can't find downloaded images
```bash
# Check where images were downloaded
python scripts/download_coco.py --info

# This will show the current state and location of COCO data
```

## Notes

- **POPE Compatibility**: POPE benchmark uses COCO 2014 validation set by default
- **Storage Requirements**: Ensure at least 10GB free space for full COCO 2014
- **Network**: Downloads are from official COCO servers which can be slow
- **Checksums**: MD5 verification is performed when available
- **Extraction**: ZIP files are automatically extracted after download

## See Also

- [Main README](../README.md) - Framework documentation
- [Quick Start](../quick_start.py) - Quick validation runs
- [Run Validation](../run_validation.py) - Full validation script