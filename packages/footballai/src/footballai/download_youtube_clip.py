#!/usr/bin/env python3
"""Download a short YouTube segment for local football video analysis."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from footballai._paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument("--start", default="00:00:00", help="Start timestamp, HH:MM:SS")
    parser.add_argument("--end", default="00:02:00", help="End timestamp, HH:MM:SS")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/raw/youtube_clip.mp4"),
        help="Output mp4 path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--download-sections",
        f"*{args.start}-{args.end}",
        "-f",
        "bv*[vcodec^=avc1][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b",
        "--merge-output-format",
        "mp4",
        "--recode-video",
        "mp4",
        "--postprocessor-args",
        "-c:v libx264 -crf 23 -c:a aac",
        "-o",
        str(output),
        args.url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "yt-dlp not found on PATH. Install it with your package manager "
            "or add it to the virtual environment (e.g. `uv pip install yt-dlp`)."
        ) from exc


if __name__ == "__main__":
    main()
