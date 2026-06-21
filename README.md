# FootballAI

A Turbo + uv monorepo that combines a SvelteKit dashboard with Python-based football video analysis.

```
footballai/
├── apps/
│   └── web/                 # SvelteKit frontend + API routes
├── packages/
│   └── footballai/            # Python inference package
├── data/
│   ├── raw/                   # downloaded/source videos
│   ├── outputs/               # rendered overlays + CSVs
│   └── live/                  # rolling JSON live state
├── models/                    # downloaded .pt weights
└── external/                  # optional YOLOv5 clone
```

## Prerequisites

- Node.js 20+ and pnpm
- Python 3.12 (managed by uv)
- NVIDIA GPU recommended (scripts default to `--device cuda`)

## Install

```bash
# JavaScript workspace
pnpm install

# Python workspace
uv sync
```

## Download models

```bash
uv run footballai-setup-models
```

This writes `models/football-player-detection.pt`, `models/football-pitch-detection.pt`, and `models/football-ball-detection.pt`.

## Download a sample clip

```bash
uv run footballai-download-youtube "https://www.youtube.com/watch?v=..." \
  --start 00:00:00 --end 00:02:00 \
  --output data/raw/youtube_clip.mp4
```

## Render a sports overlay

```bash
uv run footballai-overlay-sports \
  --video data/raw/youtube_clip.mp4 \
  --output data/outputs/sports_overlay.mp4 \
  --csv data/outputs/sports_positions.csv \
  --max-frames 90 --stride 3
```

Run with `--skip-team-fit` to skip SigLIP/UMAP/KMeans team clustering and go faster.

## Run live inference

```bash
uv run footballai-live \
  --source data/raw/youtube_clip.mp4 \
  --backend sports \
  --state data/live/latest.json \
  --max-frames 30 --stride 10
```

Then start the dashboard:

```bash
pnpm dev
```

The dashboard reads `data/live/latest.json` via `GET /api/live` and lists rendered overlays from `data/outputs` via `GET /api/runs`.

## All CLI entry points

| Command | Purpose |
|---|---|
| `uv run footballai-download-youtube` | Download a YouTube clip (H.264 transcode) |
| `uv run footballai-setup-models` | Download the Roboflow sports YOLOv8 weights |
| `uv run footballai-overlay-sports` | Full sports overlay video + CSV + radar |
| `uv run footballai-live` | Rolling JSON live inference |
| `uv run footballai-overlay-yolov5` | Legacy Roboflow YOLOv5 overlay |
| `uv run footballai-overlay-players` | Simple player position overlay |

## Useful workspace commands

```bash
pnpm dev        # start the SvelteKit app
pnpm build      # build all workspace packages
pnpm check      # run TypeScript checks
```

## Data layout

- `data/raw/` — source/input videos
- `data/outputs/` — rendered MP4s and CSVs
- `data/live/` — rolling JSON snapshot consumed by the dashboard
- `models/` — YOLOv8 / YOLOv5 weights
- `external/yolov5/` — optional clone of the YOLOv5 repository

These directories are gitignored and treated as shared runtime artifacts at the repo root.
