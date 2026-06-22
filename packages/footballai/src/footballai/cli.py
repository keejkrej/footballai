#!/usr/bin/env python3
"""Single FootballAI CLI: ``inference full`` and ``inference download``."""

from __future__ import annotations

import argparse

from footballai._paths import REPO_ROOT
from footballai.download_youtube_clip import download_youtube_clip
from footballai.setup_sports_models import ensure_models
from footballai.sports_football_overlay import run_full


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="Directory containing YOLOv8 .pt weights",
    )
    parser.add_argument("--device", default="cuda", help="Torch device: cpu, cuda, mps")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--img-size", type=int, default=1280, help="Player/pitch inference size"
    )
    parser.add_argument(
        "--skip-team-fit",
        action="store_true",
        help="Skip team classifier training (faster, no team colors)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference", description="FootballAI sports inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a YouTube clip")
    download.add_argument("url", help="YouTube URL to download")
    download.add_argument(
        "--start", default="00:00:00", help="Start timestamp, HH:MM:SS"
    )
    download.add_argument("--end", default="00:02:00", help="End timestamp, HH:MM:SS")
    download.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "raw" / "youtube_clip.mp4"),
        help="Output mp4 path",
    )
    download.add_argument("--height", type=int, default=720, help="Max video height")

    full = subparsers.add_parser("full", help="Render a full overlay for a local MP4")
    full.add_argument("--input", required=True, help="Input MP4 path")
    full.add_argument("--output", required=True, help="Output overlay MP4 path")
    full.add_argument("--csv", required=True, help="Output detections CSV path")
    full.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum source frames to process (0 = unlimited)",
    )
    full.add_argument(
        "--stride", type=int, default=1, help="Process every Nth source frame"
    )
    full.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Frames per GPU forward pass for player/pitch models",
    )
    full.add_argument(
        "--team-sample-stride",
        type=int,
        default=60,
        help="Frame stride for collecting team training crops",
    )
    full.add_argument(
        "--siglip-batch-size",
        type=int,
        default=64,
        help="Batch size for SigLIP feature extraction",
    )
    full.add_argument(
        "--team-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache fitted team classifier to disk",
    )
    full.add_argument(
        "--team-cache-dir",
        default=str(REPO_ROOT / "data" / "cache" / "team_classifier"),
        help="Directory for team classifier cache",
    )
    full.add_argument(
        "--decoder-queue-size",
        type=int,
        default=32,
        help="Max decoded frames queued ahead of inference",
    )
    full.add_argument(
        "--writer-queue-size",
        type=int,
        default=32,
        help="Max annotated frames queued for video/CSV writers",
    )
    _add_common_args(full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_models(args.models_dir)

    if args.command == "download":
        download_youtube_clip(
            args.url,
            args.output,
            start=args.start,
            end=args.end,
            height=args.height,
        )
        print(f"Downloaded: {args.output}")
    elif args.command == "full":
        run_full(
            input_path=args.input,
            output_path=args.output,
            csv_path=args.csv,
            models_dir=args.models_dir,
            device=args.device,
            conf=args.conf,
            img_size=args.img_size,
            max_frames=args.max_frames,
            stride=args.stride,
            batch_size=args.batch_size,
            skip_team_fit=args.skip_team_fit,
            team_sample_stride=args.team_sample_stride,
            siglip_batch_size=getattr(args, "siglip_batch_size", 64),
            team_cache=args.team_cache,
            team_cache_dir=args.team_cache_dir,
            decoder_queue_size=args.decoder_queue_size,
            writer_queue_size=args.writer_queue_size,
        )


if __name__ == "__main__":
    main()
