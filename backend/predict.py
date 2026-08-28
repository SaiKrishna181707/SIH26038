"""Command-line prediction utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_service import model_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict diabetic-retinopathy severity from a retinal image."
    )
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or WebP image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = model_service.predict_file(args.image)
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.4f}")
    print(json.dumps(result.probabilities, indent=2))


if __name__ == "__main__":
    main()

