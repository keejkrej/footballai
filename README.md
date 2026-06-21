# footballai

Hackathon prototype for extracting football game state from broadcast video and live streams.

## What is implemented

- Download short YouTube clips with `yt-dlp`
- Render player-position overlays with public Ultralytics YOLO26 + ByteTrack
- Render football-specific overlays with the Roboflow YOLOv5 model
- Run live inference from webcam, file, HLS, RTMP, HTTP, or capture-device inputs
- Expose generated overlays and live inference state in a SvelteKit dashboard

## Python setup

Use Python 3.11+ with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Or use `uv`:

```bash
uv pip install -r requirements.txt
```

You also need `yt-dlp` and `ffmpeg` available on PATH.

The football-specific YOLOv5 flow expects:

```text
models/football_yolov5_best.pt
external/yolov5
```

These are intentionally ignored by Git because they are large/local artifacts.

The improved Roboflow `sports` YOLOv8 flow expects:

```text
models/football-player-detection.pt
models/football-pitch-detection.pt
models/football-ball-detection.pt
```

Download them with:

```bash
python scripts/setup_sports_models.py
```

## Frontend setup

```bash
pnpm install
pnpm run dev
```

Open:

```text
http://localhost:5173
```

## Download a YouTube clip

```bash
python scripts/download_youtube_clip.py \
  "https://www.youtube.com/watch?v=6pdRXPx5ScQ&list=PL_YXogOtf6IVrnpc5ExgKpnNWRolDfj5P" \
  --start 00:00:00 \
  --end 00:00:30 \
  --output data/raw/youtube_clip.mp4
```

The script now requests H.264 video so OpenCV can read the result without
needing an AV1 decoder. If you already have an AV1 clip, transcode it with:

```bash
ffmpeg -i data/raw/youtube_clip.mp4 -c:v libx264 -crf 23 -c:a aac data/raw/youtube_clip_h264.mp4
```

## Render a YOLO26 positioning overlay

```bash
python scripts/player_position_overlay.py \
  --video data/raw/youtube_clip.mp4 \
  --output data/outputs/player_overlay.mp4 \
  --csv data/outputs/player_positions.csv
```

## Render the football-specific YOLOv5 overlay

```bash
python scripts/football_yolov5_overlay.py \
  --video data/raw/youtube_clip.mp4 \
  --weights models/football_yolov5_best.pt \
  --yolov5-repo external/yolov5 \
  --output data/outputs/football_yolov5_overlay.mp4 \
  --csv data/outputs/football_yolov5_positions.csv
```

The Roboflow tutorial model uses four football classes:

- `ball`
- `goalkeeper`
- `player`
- `referee`

## Render the improved Roboflow `sports` overlay

This uses YOLOv8 models plus ByteTrack, pitch keypoint homography, team
classification, and a dedicated ball tracker.

```bash
python scripts/sports_football_overlay.py \
  --video data/raw/youtube_clip.mp4 \
  --output data/outputs/sports_overlay.mp4 \
  --csv data/outputs/sports_positions.csv \
  --max-frames 900 \
  --stride 2
```

Options:

- `--skip-team-fit` – skip SigLIP/UMAP/KMeans team clustering (much faster).
- `--device cuda` / `mps` / `cpu`.
- `--team-sample-stride 60` – how often to sample frames for team crops.

Outputs:

- `data/outputs/sports_overlay.mp4` – annotated video with radar/minimap.
- `data/outputs/sports_positions.csv` – per-frame detections with broadcast
  coordinates and real-world pitch x/y in centimeters.

## Run live stream inference

The live script accepts any OpenCV/FFmpeg-readable source:

- webcam index: `0`
- local file: `data/raw/youtube_clip.mp4`
- HLS: `https://example.com/live/playlist.m3u8`
- RTMP: `rtmp://example.com/live/key`
- HTTP video URL
- capture device path

Local file smoke test with the improved `sports` backend (default):

```bash
python scripts/live_stream_inference.py \
  --source data/raw/youtube_clip.mp4 \
  --backend sports \
  --state data/live/latest.json \
  --max-frames 300 \
  --stride 10 \
  --overlay-output data/outputs/sports_live_overlay.mp4
```

Local file smoke test with the original YOLOv5 backend:

```bash
python scripts/live_stream_inference.py \
  --source data/raw/youtube_clip.mp4 \
  --backend yolov5 \
  --weights models/football_yolov5_best.pt \
  --yolov5-repo external/yolov5 \
  --state data/live/latest.json \
  --max-frames 300 \
  --stride 10
```

Webcam with the `sports` backend:

```bash
python scripts/live_stream_inference.py \
  --source 0 \
  --backend sports \
  --state data/live/latest.json
```

HLS/RTMP livestream with the `sports` backend:

```bash
python scripts/live_stream_inference.py \
  --source "https://example.com/live/playlist.m3u8" \
  --backend sports \
  --state data/live/latest.json
```

The frontend polls:

```text
/api/live
```

and displays:

- current detection counts
- model latency
- visual territory signal
- pressure side and pressure score
- a first-pass trading-style pressure edge

## Current limitations

The live trading signal is a heuristic over visible broadcast-frame coordinates. It is not yet calibrated odds, team-aware market making, or official match state. The next required steps are:

- team assignment
- pitch homography
- scoreboard OCR or official score feed
- event labeling for shots, corners, fouls, and cards
- calibrated prediction heads for one target market
