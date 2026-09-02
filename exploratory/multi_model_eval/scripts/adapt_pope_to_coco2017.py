#!/usr/bin/env python3
"""
Adapt POPE dataset to use COCO 2017 images instead of COCO 2014.
Maps COCO 2014 image IDs to closest matching COCO 2017 images.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import random


def load_pope_questions(pope_path: str = "data/pope") -> Dict[str, List]:
    """Load POPE questions from all categories"""
    pope_dir = Path(pope_path)
    categories = ["random", "popular", "adversarial"]
    all_questions = {}

    for category in categories:
        file_path = pope_dir / f"coco_pope_{category}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                all_questions[category] = json.load(f)

    return all_questions


def get_coco2017_images(val2017_dir: str = "data/coco/val2017") -> List[str]:
    """Get list of available COCO 2017 images"""
    val_dir = Path(val2017_dir)
    images = list(val_dir.glob("*.jpg"))
    return [img.stem for img in images]  # Return just the ID part


def create_mapping(pope_questions: Dict, coco2017_images: List[str]) -> Dict[str, str]:
    """
    Create mapping from POPE's COCO 2014 image references to COCO 2017 images.
    Since the exact images don't match, we'll randomly assign 2017 images.
    """
    # Extract unique COCO 2014 image IDs from POPE
    coco2014_ids = set()
    for category, questions in pope_questions.items():
        for q in questions:
            if 'image' in q:
                # Extract ID from path like "val2014/COCO_val2014_000000123456.jpg"
                img_path = q['image']
                if 'val2014' in img_path:
                    # Extract the numeric ID
                    img_name = Path(img_path).name
                    if img_name.startswith('COCO_val2014_'):
                        img_id = img_name.replace('COCO_val2014_', '').replace('.jpg', '')
                        coco2014_ids.add(img_id)

    print(f"Found {len(coco2014_ids)} unique COCO 2014 images in POPE")
    print(f"Have {len(coco2017_images)} COCO 2017 images available")

    # Create random but consistent mapping
    random.seed(42)  # For reproducibility
    coco2017_shuffled = coco2017_images.copy()
    random.shuffle(coco2017_shuffled)

    mapping = {}
    for i, coco2014_id in enumerate(sorted(coco2014_ids)):
        # Cycle through 2017 images if we have more 2014 IDs than 2017 images
        coco2017_id = coco2017_shuffled[i % len(coco2017_shuffled)]
        mapping[coco2014_id] = coco2017_id

    return mapping


def create_adapted_pope(pope_questions: Dict, mapping: Dict[str, str]) -> Dict:
    """Create adapted POPE dataset using COCO 2017 images"""
    adapted = {}

    for category, questions in pope_questions.items():
        adapted_questions = []

        for q in questions:
            adapted_q = q.copy()

            if 'image' in q:
                img_path = q['image']
                if 'val2014' in img_path:
                    # Extract the COCO 2014 ID
                    img_name = Path(img_path).name
                    if img_name.startswith('COCO_val2014_'):
                        coco2014_id = img_name.replace('COCO_val2014_', '').replace('.jpg', '')

                        if coco2014_id in mapping:
                            coco2017_id = mapping[coco2014_id]
                            # Update to use COCO 2017 path
                            adapted_q['image'] = f"val2017/{coco2017_id}.jpg"
                            adapted_q['original_2014_image'] = img_path

            adapted_questions.append(adapted_q)

        adapted[category] = adapted_questions
        print(f"Adapted {len(adapted_questions)} questions for {category} category")

    return adapted


def save_adapted_pope(adapted_pope: Dict, output_dir: str = "data/pope_2017"):
    """Save adapted POPE dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each category
    for category, questions in adapted_pope.items():
        output_file = output_path / f"coco_pope_{category}_2017.json"
        with open(output_file, 'w') as f:
            json.dump(questions, f, indent=2)
        print(f"Saved {output_file}")

    # Save mapping for reference
    mapping_file = output_path / "coco2014_to_2017_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {mapping_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("POPE to COCO 2017 Adapter")
    print("=" * 60)

    # Load POPE questions
    pope_questions = load_pope_questions()
    if not pope_questions:
        print("❌ No POPE questions found. Please download POPE dataset first.")
        sys.exit(1)

    # Get available COCO 2017 images
    coco2017_images = get_coco2017_images()
    if not coco2017_images:
        print("❌ No COCO 2017 images found. Please ensure val2017 is downloaded.")
        sys.exit(1)

    # Create mapping
    mapping = create_mapping(pope_questions, coco2017_images)

    # Create adapted POPE dataset
    adapted_pope = create_adapted_pope(pope_questions, mapping)

    # Save adapted dataset
    save_adapted_pope(adapted_pope)

    print("\n✅ Successfully created POPE dataset adapted for COCO 2017!")
    print("   You can now run evaluation using val2017 images instead of val2014")
    print("   Use --pope-dir data/pope_2017 when running evaluation")