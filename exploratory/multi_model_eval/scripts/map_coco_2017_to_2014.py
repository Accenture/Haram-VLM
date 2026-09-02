#!/usr/bin/env python3
"""
Map COCO 2017 images to COCO 2014 naming convention for POPE compatibility.
Creates symlinks from val2017 images to val2014 naming pattern.
"""

import os
import sys
from pathlib import Path


def create_coco_2014_mapping(val2017_dir: str = "data/coco/val2017",
                             val2014_dir: str = "data/coco/val2014"):
    """Create symlinks from COCO 2017 to 2014 naming convention"""

    val2017_path = Path(val2017_dir)
    val2014_path = Path(val2014_dir)

    if not val2017_path.exists():
        print(f"❌ COCO 2017 directory not found: {val2017_path}")
        return 1

    # Create val2014 directory
    val2014_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {val2014_path}")

    # Get all images from val2017
    images = list(val2017_path.glob("*.jpg"))
    print(f"Found {len(images)} images in {val2017_path}")

    # Create symlinks with 2014 naming
    created = 0
    skipped = 0

    for img_path in images:
        # Extract image ID (e.g., 000000123456.jpg)
        img_name = img_path.name

        # Create 2014-style name (e.g., COCO_val2014_000000123456.jpg)
        new_name = f"COCO_val2014_{img_name}"
        new_path = val2014_path / new_name

        if new_path.exists():
            skipped += 1
            continue

        # Create symlink
        try:
            new_path.symlink_to(img_path.absolute())
            created += 1

            if created % 500 == 0:
                print(f"  Created {created} symlinks...")
        except Exception as e:
            print(f"Warning: Could not create symlink for {img_name}: {e}")

    print(f"\n✅ Mapping complete!")
    print(f"  Created: {created} symlinks")
    print(f"  Skipped: {skipped} (already existed)")
    print(f"  Total: {len(images)} images")

    return 0


if __name__ == "__main__":
    sys.exit(create_coco_2014_mapping())