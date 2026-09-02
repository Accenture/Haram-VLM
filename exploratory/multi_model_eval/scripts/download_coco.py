#!/usr/bin/env python3
"""
COCO Dataset Downloader for POPE Validation
============================================

Downloads COCO validation images required for POPE benchmark evaluation.
Supports resuming interrupted downloads and verification.
"""

import os
import sys
import argparse
import zipfile
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
import time
from typing import Optional, Tuple


class COCODownloader:
    """
    COCO dataset downloader with progress tracking and resume support.
    """

    # COCO 2014 validation dataset (used by POPE)
    COCO_VAL_2014 = {
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "size_mb": 6645,  # Approximate size in MB
        "md5": "c72670d3dc0a87b3d4c8e8be7d8f1e0c",  # Optional checksum
        "extract_dir": "val2014"
    }

    # COCO 2017 validation dataset (alternative)
    COCO_VAL_2017 = {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "size_mb": 778,
        "md5": "442b8da7639aecaf257c1dceb8ba8c80",
        "extract_dir": "val2017"
    }

    # COCO annotations (optional, for reference)
    COCO_ANNOTATIONS = {
        "2014": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "size_mb": 252
        },
        "2017": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "size_mb": 252
        }
    }

    def __init__(self, output_dir: str = "./data/coco", dataset_year: str = "2014"):
        """
        Initialize COCO downloader.

        Args:
            output_dir: Directory to save COCO data
            dataset_year: Year of dataset (2014 or 2017)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_year = dataset_year

        if dataset_year == "2014":
            self.dataset_info = self.COCO_VAL_2014
        elif dataset_year == "2017":
            self.dataset_info = self.COCO_VAL_2017
        else:
            raise ValueError(f"Unsupported dataset year: {dataset_year}")

    def _print_progress(self, block_num: int, block_size: int, total_size: int):
        """Progress callback for download."""
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # Print progress
        sys.stdout.write(f'\rDownloading: |{bar}| {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()

        if percent >= 100:
            print()  # New line when complete

    def _verify_checksum(self, file_path: Path, expected_md5: Optional[str]) -> bool:
        """Verify file checksum if provided."""
        if not expected_md5:
            return True

        print("Verifying checksum...")
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        actual_md5 = md5_hash.hexdigest()
        if actual_md5 == expected_md5:
            print("✓ Checksum verified")
            return True
        else:
            print(f"✗ Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
            return False

    def download_dataset(self, force_download: bool = False) -> Tuple[bool, str]:
        """
        Download COCO validation dataset.

        Args:
            force_download: Force re-download even if file exists

        Returns:
            Tuple of (success, path_to_images)
        """
        # Check if already extracted
        extract_dir = self.output_dir / self.dataset_info["extract_dir"]
        if extract_dir.exists() and not force_download:
            num_images = len(list(extract_dir.glob("*.jpg")))
            if num_images > 0:
                print(f"✓ COCO {self.dataset_year} validation images already exist ({num_images} images)")
                return True, str(extract_dir)

        # Download zip file
        zip_filename = f"val{self.dataset_year}.zip"
        zip_path = self.output_dir / zip_filename

        # Check if zip already exists
        if zip_path.exists() and not force_download:
            print(f"Found existing zip file: {zip_path}")
            if self._verify_checksum(zip_path, self.dataset_info.get("md5")):
                print("Using existing zip file")
            else:
                print("Checksum failed, re-downloading...")
                zip_path.unlink()

        # Download if needed
        if not zip_path.exists():
            url = self.dataset_info["url"]
            print(f"Downloading COCO {self.dataset_year} validation images")
            print(f"URL: {url}")
            print(f"Size: ~{self.dataset_info['size_mb']} MB")
            print(f"Destination: {zip_path}")
            print("-" * 60)

            try:
                urlretrieve(url, zip_path, reporthook=self._print_progress)
                print(f"✓ Download complete: {zip_path}")

                # Verify checksum
                if not self._verify_checksum(zip_path, self.dataset_info.get("md5")):
                    print("Warning: Checksum verification failed")

            except (URLError, OSError) as e:
                print(f"\n✗ Download failed: {e}")
                if zip_path.exists():
                    zip_path.unlink()
                return False, ""

        # Extract zip file
        print(f"\nExtracting {zip_filename}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total number of files
                total_files = len(zip_ref.namelist())
                print(f"Extracting {total_files} files...")

                # Extract with progress
                for i, member in enumerate(zip_ref.namelist(), 1):
                    zip_ref.extract(member, self.output_dir)
                    if i % 100 == 0 or i == total_files:
                        percent = (i / total_files) * 100
                        sys.stdout.write(f'\rExtracting: {percent:.1f}% ({i}/{total_files} files)')
                        sys.stdout.flush()
                print()

            print(f"✓ Extraction complete: {extract_dir}")

            # Verify extraction
            num_images = len(list(extract_dir.glob("*.jpg")))
            print(f"✓ Found {num_images} images")

            # Optionally remove zip to save space
            # zip_path.unlink()
            # print("✓ Removed zip file to save space")

            return True, str(extract_dir)

        except zipfile.BadZipFile as e:
            print(f"✗ Extraction failed: {e}")
            print("The zip file may be corrupted. Try downloading again with --force")
            return False, ""

    def download_sample_images(self, num_samples: int = 10) -> Tuple[bool, str]:
        """
        Download just a few sample images for quick testing.

        Args:
            num_samples: Number of sample images to download

        Returns:
            Tuple of (success, path_to_samples)
        """
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {num_samples} sample COCO images for quick testing...")

        # Sample image URLs (2017 dataset, smaller files)
        sample_urls = [
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000000776.jpg",
            "http://images.cocodataset.org/val2017/000000000785.jpg",
            "http://images.cocodataset.org/val2017/000000000802.jpg",
            "http://images.cocodataset.org/val2017/000000000872.jpg",
            "http://images.cocodataset.org/val2017/000000000885.jpg",
            "http://images.cocodataset.org/val2017/000000001000.jpg"
        ]

        downloaded = 0
        for i, url in enumerate(sample_urls[:num_samples], 1):
            filename = url.split('/')[-1]
            filepath = samples_dir / filename

            if filepath.exists():
                print(f"  [{i}/{num_samples}] ✓ {filename} (exists)")
                downloaded += 1
                continue

            try:
                print(f"  [{i}/{num_samples}] Downloading {filename}...", end='')
                urlretrieve(url, filepath)
                print(" ✓")
                downloaded += 1
            except Exception as e:
                print(f" ✗ Failed: {e}")

        print(f"\n✓ Downloaded {downloaded}/{num_samples} sample images to {samples_dir}")
        return downloaded > 0, str(samples_dir)

    def download_annotations(self) -> bool:
        """Download COCO annotations (optional)."""
        anno_info = self.COCO_ANNOTATIONS[self.dataset_year]
        anno_filename = f"annotations_trainval{self.dataset_year}.zip"
        anno_path = self.output_dir / anno_filename

        if anno_path.exists():
            print(f"✓ Annotations already exist: {anno_path}")
            return True

        print(f"Downloading COCO {self.dataset_year} annotations...")
        print(f"Size: ~{anno_info['size_mb']} MB")

        try:
            urlretrieve(anno_info["url"], anno_path, reporthook=self._print_progress)
            print(f"✓ Annotations downloaded: {anno_path}")

            # Extract annotations
            print("Extracting annotations...")
            with zipfile.ZipFile(anno_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            print("✓ Annotations extracted")

            return True
        except Exception as e:
            print(f"✗ Failed to download annotations: {e}")
            return False

    def print_dataset_info(self):
        """Print information about the dataset."""
        print("\n" + "="*60)
        print("COCO Dataset Information")
        print("="*60)
        print(f"Dataset Year: {self.dataset_year}")
        print(f"Type: Validation Set")
        print(f"Download URL: {self.dataset_info['url']}")
        print(f"Approximate Size: {self.dataset_info['size_mb']} MB")
        print(f"Output Directory: {self.output_dir}")

        # Check existing data
        extract_dir = self.output_dir / self.dataset_info["extract_dir"]
        if extract_dir.exists():
            num_images = len(list(extract_dir.glob("*.jpg")))
            size_mb = sum(f.stat().st_size for f in extract_dir.glob("*.jpg")) / (1024 * 1024)
            print(f"\nExisting Data:")
            print(f"  Images: {num_images}")
            print(f"  Total Size: {size_mb:.1f} MB")
        else:
            print("\nNo existing data found")
        print("="*60)


def main():
    """Main entry point for COCO downloader."""
    parser = argparse.ArgumentParser(
        description="Download COCO dataset for POPE validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download COCO 2014 validation set (used by POPE)
  python download_coco.py --year 2014

  # Download to specific directory
  python download_coco.py --output-dir /path/to/coco

  # Download sample images for testing
  python download_coco.py --samples-only

  # Force re-download
  python download_coco.py --force

  # Download with annotations
  python download_coco.py --with-annotations
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/coco",
        help="Directory to save COCO data (default: ./data/coco)"
    )

    parser.add_argument(
        "--year",
        type=str,
        choices=["2014", "2017"],
        default="2014",
        help="COCO dataset year (default: 2014 for POPE compatibility)"
    )

    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Download only sample images for quick testing"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample images to download (default: 10)"
    )

    parser.add_argument(
        "--with-annotations",
        action="store_true",
        help="Also download COCO annotations"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information only"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = COCODownloader(
        output_dir=args.output_dir,
        dataset_year=args.year
    )

    # Show info if requested
    if args.info:
        downloader.print_dataset_info()
        return 0

    print("="*60)
    print("COCO Dataset Downloader for HARAM-VLM")
    print("="*60)

    success = True

    # Download samples or full dataset
    if args.samples_only:
        success, path = downloader.download_sample_images(args.num_samples)
        if success:
            print(f"\n✓ Sample images ready at: {path}")
            print("\nYou can now run validation with:")
            print(f"  python run_validation.py --coco-image-dir {path}")
    else:
        # Download full dataset
        success, path = downloader.download_dataset(force_download=args.force)

        if success:
            print(f"\n✓ COCO {args.year} validation images ready at: {path}")

            # Download annotations if requested
            if args.with_annotations:
                downloader.download_annotations()

            print("\n" + "="*60)
            print("Setup Complete!")
            print("="*60)
            print("\nYou can now run POPE validation with:")
            print(f"  python run_validation.py --coco-image-dir {path}")
            print("\nFor quick testing:")
            print("  python quick_start.py --test")
        else:
            print("\n✗ Download failed. Please check your internet connection and try again.")
            print("For debugging, you can try downloading sample images first:")
            print("  python download_coco.py --samples-only")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())