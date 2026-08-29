"""Command-line prediction utility."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from model_service import model_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict diabetic-retinopathy severity from a retinal image."
    )
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or WebP image")
    parser.add_argument(
        "--heatmap",
        type=Path,
        metavar="PATH",
        help="Write the Grad-CAM overlay to this file (extension sets the format)",
    )
    return parser.parse_args()


def save_heatmap(data_uri: str, destination: Path) -> None:
    payload = data_uri.partition(",")[2]
    destination.write_bytes(base64.b64decode(payload))


def main() -> int:
    args = parse_args()
    result = model_service.predict_file(args.image, explain=args.heatmap is not None)

    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.4f}")
    print(json.dumps(result.probabilities, indent=2))

    if args.heatmap:
        if result.heatmap is None:
            print(
                "Grad-CAM is unavailable for this model; no heat map written.",
                file=sys.stderr,
            )
            return 1
        save_heatmap(result.heatmap, args.heatmap)
        print(f"Grad-CAM overlay written to {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
