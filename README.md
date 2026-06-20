# footballai

Hackathon prototype for extracting football game state from broadcast video and live streams.

## What is implemented

- Download short YouTube clips with `yt-dlp`
- Render player-position overlays with public Ultralytics YOLO26 + ByteTrack
- Render football-specific overlays with the Roboflow YOLOv5 model
- Run live inference from webcam, file, HLS, RTMP, HTTP, or capture-device inputs
- Expose generated overlays and live inference state in a SvelteKit dashboard

## Python setup

Use Python 3.11+:

```bash
python -m pip install -r requirements.txt
```

You also need `yt-dlp` and `ffmpeg` available on PATH.

The football-specific YOLOv5 flow expects:

```text
models/football_yolov5_best.pt
external/yolov5
```

These are intentionally ignored by Git because they are large/local artifacts.

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

## Run live stream inference

The live script accepts any OpenCV/FFmpeg-readable source:

- webcam index: `0`
- local file: `data/raw/youtube_clip.mp4`
- HLS: `https://example.com/live/playlist.m3u8`
- RTMP: `rtmp://example.com/live/key`
- HTTP video URL
- capture device path

Local file smoke test:

```bash
python scripts/live_stream_inference.py \
  --source data/raw/youtube_clip.mp4 \
  --weights models/football_yolov5_best.pt \
  --yolov5-repo external/yolov5 \
  --state data/live/latest.json \
  --max-frames 300 \
  --stride 10
```

Webcam:

```bash
python scripts/live_stream_inference.py \
  --source 0 \
  --weights models/football_yolov5_best.pt \
  --yolov5-repo external/yolov5 \
  --state data/live/latest.json
```

HLS/RTMP livestream:

```bash
python scripts/live_stream_inference.py \
  --source "https://example.com/live/playlist.m3u8" \
  --weights models/football_yolov5_best.pt \
  --yolov5-repo external/yolov5 \
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
