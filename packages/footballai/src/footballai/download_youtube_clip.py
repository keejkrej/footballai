#!/usr/bin/env python3
"""Download a short YouTube segment for local football video analysis."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from footballai._paths import REPO_ROOT


def download_youtube_clip(
    url: str,
    output: Path | str,
    *,
    start: str = "00:00:00",
    end: str = "00:02:00",
    height: int = 720,
) -> None:
    """Download a YouTube clip and transcode to H.264 MP4."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise FileNotFoundError(
            "yt-dlp not found on PATH. Install it (e.g. `uv tool install yt-dlp` "
            "or `uv pip install yt-dlp`)."
        )

    cmd = [
        yt_dlp,
        "--no-playlist",
        "--download-sections",
        f"*{start}-{end}",
        "-f",
        f"bv*[vcodec^=avc1][height<={height}]+ba[ext=m4a]/b[ext=mp4][height<={height}]/b",
        "--merge-output-format",
        "mp4",
        "--recode-video",
        "mp4",
        "--postprocessor-args",
        "-c:v libx264 -crf 23 -c:a aac",
        "--newline",
        "--progress",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube URL to download")
    parser.add_argument("--start", default="00:00:00", help="Start timestamp, HH:MM:SS")
    parser.add_argument("--end", default="00:00:00", help="End timestamp, HH:MM:SS")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "raw" / "youtube_clip.mp4"),
        help="Output mp4 path",
    )
    parser.add_argument("--height", type=int, default=720, help="Max video height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_youtube_clip(args.url, args.output, start=args.start, end=args.end, height=args.height)
    print(f"Downloaded: {args.output}")


if __name__ == "__main__":
    main()
