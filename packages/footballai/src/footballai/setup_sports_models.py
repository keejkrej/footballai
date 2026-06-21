#!/usr/bin/env python3
"""Download the YOLOv8 model weights used by the Roboflow sports soccer example.

The weights are placed under models/ so the overlay and live scripts can load
them without relying on a sibling data/ directory or the example setup.sh script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown

from footballai._paths import REPO_ROOT


MODELS = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models"),
        help="Directory to write model weights",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in MODELS.items():
        destination = output_dir / filename
        if destination.exists() and not args.force:
            print(f"Skipping {destination} (already exists)")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename} ...")
        gdown.download(url, str(destination), quiet=False)
        if not destination.exists():
            raise RuntimeError(f"Failed to download {filename}")

    print("Sports models ready in:", output_dir)


if __name__ == "__main__":
    main()
