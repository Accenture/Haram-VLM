"""
POPE Dataset Handler
====================

Handles loading and evaluation of the POPE (Polling-based Object Probing Evaluation) benchmark
for hallucination detection in Vision-Language Models.

POPE has three splits:
- Random: Random negative sampling
- Popular: Popular (frequent) objects as negative samples
- Adversarial: Co-occurring objects as negative samples (hardest)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import requests
from tqdm import tqdm
import random
from PIL import Image
import pandas as pd


@dataclass
class POPEQuestion:
    """Single POPE question with metadata"""
    question_id: str
    image_path: str
    question: str
    answer: str  # "yes" or "no"
    category: str  # "random", "popular", or "adversarial"
    object_mentioned: Optional[str] = None
    question_type: Optional[str] = None  # "existence", "attribute", etc.


@dataclass
class POPEMetrics:
    """Metrics for POPE evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    yes_ratio: float  # Ratio of "yes" predictions (bias metric)
    hallucination_rate: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    total_questions: int


class POPEDataset:
    """
    POPE Dataset handler for hallucination evaluation.

    Supports downloading, caching, and evaluation of POPE benchmark.
    """

    POPE_URLS = {
        "random": "https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_random.json",
        "popular": "https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_popular.json",
        "adversarial": "https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_adversarial.json"
    }

    def __init__(self,
                 data_dir: str = "./data/pope",
                 coco_image_dir: Optional[str] = None,
                 download: bool = True,
                 use_cached: bool = True):
        """
        Initialize POPE dataset.

        Args:
            data_dir: Directory to store POPE annotations
            coco_image_dir: Path to COCO images (val2014)
            download: Whether to download POPE data if not found
            use_cached: Whether to use cached data if available
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.coco_image_dir = Path(coco_image_dir) if coco_image_dir else None
        self.download = download
        self.use_cached = use_cached

        self.questions = {
            "random": [],
            "popular": [],
            "adversarial": []
        }

        self._load_data()

    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def _load_data(self):
        """Load POPE questions from files or download if needed"""
        for category, url in self.POPE_URLS.items():
            file_path = self.data_dir / f"coco_pope_{category}.json"

            # Check if file exists
            if not file_path.exists() and self.download:
                print(f"Downloading POPE {category} split...")
                self._download_file(url, file_path)

            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f]

                for item in data:
                    question = POPEQuestion(
                        question_id=item.get("question_id", ""),
                        image_path=item.get("image", ""),
                        question=item.get("text", ""),
                        answer=item.get("label", "").lower(),
                        category=category,
                        object_mentioned=self._extract_object(item.get("text", ""))
                    )
                    self.questions[category].append(question)

                print(f"Loaded {len(self.questions[category])} questions for {category} split")

    def _extract_object(self, question: str) -> Optional[str]:
        """Extract the object being asked about from the question"""
        # Simple extraction - looks for pattern "Is there a/an [object] in the image?"
        import re
        pattern = r"Is there (?:a|an) ([a-zA-Z\s]+) in the image\?"
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
        return None

    def get_questions(self,
                     category: str = "adversarial",
                     limit: Optional[int] = None,
                     shuffle: bool = False) -> List[POPEQuestion]:
        """
        Get questions for a specific category.

        Args:
            category: One of "random", "popular", "adversarial"
            limit: Maximum number of questions to return
            shuffle: Whether to shuffle questions

        Returns:
            List of POPEQuestion objects
        """
        if category not in self.questions:
            raise ValueError(f"Invalid category: {category}. Choose from {list(self.questions.keys())}")

        questions = self.questions[category].copy()

        if shuffle:
            random.shuffle(questions)

        if limit:
            questions = questions[:limit]

        return questions

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from COCO dataset.

        Args:
            image_path: Relative path from POPE dataset (e.g., "val2014/COCO_val2014_000000000000.jpg")

        Returns:
            PIL Image or None if not found
        """
        if self.coco_image_dir is None:
            print(f"Warning: COCO image directory not set. Cannot load {image_path}")
            return None

        full_path = self.coco_image_dir / image_path

        if not full_path.exists():
            # Try without the val2014 prefix
            filename = Path(image_path).name
            full_path = self.coco_image_dir / filename

        if full_path.exists():
            return Image.open(full_path).convert("RGB")

        print(f"Warning: Image not found: {full_path}")
        return None

    def evaluate_predictions(self,
                           predictions: List[str],
                           questions: List[POPEQuestion]) -> POPEMetrics:
        """
        Evaluate model predictions against ground truth.

        Args:
            predictions: List of model predictions ("yes" or "no")
            questions: List of corresponding POPEQuestion objects

        Returns:
            POPEMetrics with evaluation results
        """
        if len(predictions) != len(questions):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(questions)} questions")

        # Normalize predictions
        normalized_preds = []
        for pred in predictions:
            pred_lower = pred.lower().strip()
            # Handle various response formats
            if "yes" in pred_lower and "no" not in pred_lower:
                normalized_preds.append("yes")
            elif "no" in pred_lower:
                normalized_preds.append("no")
            else:
                # Default to "no" for unclear responses
                normalized_preds.append("no")

        # Calculate metrics
        tp = tn = fp = fn = 0
        yes_count = 0

        for pred, question in zip(normalized_preds, questions):
            ground_truth = question.answer

            if pred == "yes":
                yes_count += 1

            if pred == "yes" and ground_truth == "yes":
                tp += 1
            elif pred == "no" and ground_truth == "no":
                tn += 1
            elif pred == "yes" and ground_truth == "no":
                fp += 1  # Hallucination
            elif pred == "no" and ground_truth == "yes":
                fn += 1

        total = len(questions)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        yes_ratio = yes_count / total if total > 0 else 0
        hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        return POPEMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            yes_ratio=yes_ratio,
            hallucination_rate=hallucination_rate,
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            total_questions=total
        )

    def create_balanced_subset(self,
                             category: str = "adversarial",
                             size: int = 100,
                             yes_ratio: float = 0.5) -> List[POPEQuestion]:
        """
        Create a balanced subset of questions.

        Args:
            category: POPE category to sample from
            size: Total number of questions
            yes_ratio: Ratio of "yes" answers (default 0.5 for balanced)

        Returns:
            Balanced list of questions
        """
        all_questions = self.get_questions(category)

        yes_questions = [q for q in all_questions if q.answer == "yes"]
        no_questions = [q for q in all_questions if q.answer == "no"]

        n_yes = int(size * yes_ratio)
        n_no = size - n_yes

        sampled_yes = random.sample(yes_questions, min(n_yes, len(yes_questions)))
        sampled_no = random.sample(no_questions, min(n_no, len(no_questions)))

        balanced = sampled_yes + sampled_no
        random.shuffle(balanced)

        return balanced

    def get_statistics(self) -> pd.DataFrame:
        """Get dataset statistics as a DataFrame"""
        stats = []
        for category in self.questions:
            questions = self.questions[category]
            if questions:
                yes_count = sum(1 for q in questions if q.answer == "yes")
                no_count = len(questions) - yes_count

                stats.append({
                    "Category": category,
                    "Total": len(questions),
                    "Yes": yes_count,
                    "No": no_count,
                    "Yes Ratio": yes_count / len(questions) if questions else 0
                })

        return pd.DataFrame(stats)

    def export_results(self,
                      results: Dict[str, POPEMetrics],
                      output_path: str):
        """
        Export evaluation results to file.

        Args:
            results: Dictionary mapping condition names to metrics
            output_path: Path to save results
        """
        output_data = {}
        for condition, metrics in results.items():
            output_data[condition] = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "yes_ratio": metrics.yes_ratio,
                "hallucination_rate": metrics.hallucination_rate,
                "confusion_matrix": {
                    "true_positive": metrics.true_positive,
                    "true_negative": metrics.true_negative,
                    "false_positive": metrics.false_positive,
                    "false_negative": metrics.false_negative
                },
                "total_questions": metrics.total_questions
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results exported to {output_path}")