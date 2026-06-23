# FootballAI

A Turbo + uv monorepo that combines a Svelte SPA with Python-based football
video analysis. The Python layer uses the **Roboflow sports** YOLOv8 soccer stack
(player, pitch, ball, and team classification). The webapp provides two modes:

- **Full**: paste a YouTube link, the Python WebSocket server downloads the clip
  and runs the full inference pipeline, then the SPA plays the annotated MP4.
- **Live**: describe a video source (local MP4, YouTube URL, or OBS device
  output) and send it to the Python WebSocket server. The backend decodes the
  source, runs inference, and pushes annotated JPEG frames + live metrics back
  to the SPA.

```
footballai/
├── apps/
│   └── web/                 # Svelte SPA (Vite + Svelte 5)
├── packages/
│   └── footballai/          # Python inference package
├── data/
│   ├── raw/                 # downloaded/source videos
│   ├── outputs/             # rendered overlays + CSVs
│   └── jobs/                # pipeline job state
└── models/                  # downloaded .pt weights
```

## Prerequisites

- Node.js 20+ and pnpm
- [vite-plus](https://vite.plus) (`vp`) for the web toolchain
- Python 3.12 (managed by uv)
- NVIDIA GPU recommended (`--device cuda` is the default)
- Nginx (or any static file server) for serving the built SPA

## Install

```bash
# Install the vite-plus CLI globally
curl -fsSL https://vite.plus | bash

# JavaScript workspace
pnpm install

# Python workspace
uv sync --all-packages
```

Models are downloaded automatically on the first inference run. You can also
trigger it explicitly:

```bash
uv run python -m footballai.setup_sports_models
```

## Python CLI

The Python package exposes two entry points:

```bash
# Render a full overlay for a local MP4
uv run inference full \
  --input data/raw/clip.mp4 \
  --output data/outputs/overlay.mp4 \
  --csv data/outputs/positions.csv

# Start the unified WebSocket server used by the web UI
uv run web --port 8000
```

### Common options

| Option         | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `--models-dir` | Directory containing the Roboflow `.pt` weights (default: `models/`) |
| `--device`     | Torch device: `cuda`, `cpu`, `mps`                                   |
| `--conf`       | Detection confidence threshold (default: 0.25)                       |
| `--img-size`   | Player/pitch inference size (default: 1280)                          |

### `inference full` options

| Option                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `--input`              | Input MP4 path                                   |
| `--output`             | Output overlay MP4 path                          |
| `--csv`                | Output detections CSV path                       |
| `--max-frames`         | Maximum source frames to process (0 = unlimited) |
| `--stride`             | Process every Nth source frame                   |
| `--batch-size`         | Frames per GPU forward pass                      |
| `--team-sample-stride` | Frame stride for collecting team training crops  |
| `--siglip-batch-size`  | Batch size for SigLIP feature extraction         |

### `inference live` / `web` options

| Option   | Description                                |
| -------- | ------------------------------------------ |
| `--host` | WebSocket server host (default: `0.0.0.0`) |
| `--port` | WebSocket server port (default: `8000`)    |

The `web` server accepts JSON commands and pushes annotated JPEG frames plus
JSON metadata back to the client for live jobs:

```json
{"action": "configure", "options": {"device": "cuda"}}
{"action": "full", "youtubeUrl": "...", "start": "00:00:00", "end": "00:02:00"}
{"action": "live_start", "source": {"type": "file", "path": "data/raw/clip.mp4", "start": "00:00:00", "end": "00:02:00"}, "options": {"device": "cuda"}}
{"action": "live_start", "source": {"type": "youtube", "url": "...", "start": "00:00:00", "end": "00:02:00"}}
{"action": "live_start", "source": {"type": "obs", "device": "/dev/video2"}}
{"action": "live_stop"}
{"action": "runs"}
{"action": "job", "id": "..."}
{"action": "stop"}
```

## Webapp

### Build

```bash
cd apps/web
vp build
```

The static assets land in `apps/web/dist`.

### Serve

Point Nginx (or your static server of choice) at `apps/web/dist` and proxy
`/ws` to the Python WebSocket server. Example Nginx config:

```nginx
server {
    listen 80;
    server_name localhost;

    root /home/jack/workspace/footballai_main/apps/web/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;
    }

    location /media {
        alias /home/jack/workspace/footballai_main/data/outputs;
    }
}
```

Start the Python WebSocket server:

```bash
uv run web --port 8000
```

Then open the static server URL (e.g. `http://localhost`).

### Dev mode

For development you can use vite-plus and proxy `/ws` to the Python server:

```bash
uv run web --port 8000     # in one shell
cd apps/web && vp dev      # in another shell
```

Configure the WebSocket URL by setting `window.__FOOTBALLAI_WS__ = "ws://localhost:8000"`
in `index.html` or in your Vite proxy config.

### Full mode

1. Paste a YouTube URL.
2. Optionally set start/end timestamps.
3. Click **Run pipeline**.
4. A progress modal shows download + inference status.
5. When finished, the generated overlay appears in the runs list and the video
   player plays the annotated MP4.

### Live mode

1. Make sure the Python WebSocket server is running.
2. Pick a source type and enter the source value:
   - **Local MP4 file**: an absolute or repo-relative path such as `data/raw/clip.mp4`.
     Optional `start` and `end` timestamps (`HH:MM:SS`) limit inference to a
     segment; otherwise the whole file is decoded.
   - **YouTube URL**: a YouTube video URL. The backend uses yt-dlp to resolve the
     stream and ffmpeg to decode it frame-by-frame while inference runs.
   - **OBS device**: a virtual-camera device path such as `/dev/video2`.
3. Set the max inference FPS and device options.
4. Click **Start live**. The backend decodes the source, runs inference, and
   streams annotated JPEG frames back to the browser.

### Stream compatibility note

Only sources that can be decoded directly by ffmpeg or OpenCV are supported.
YouTube URLs are resolved by yt-dlp and passed to ffmpeg. Geo-restricted or
DRM-encrypted broadcaster streams cannot be processed. This app is intended for:

- Public-domain or Creative Commons footage
- Streams you own or have explicit rights to process
- Local video files you own or have rights to process
- OBS-captured sources you control

## Data layout

- `data/raw/` — downloaded source videos
- `data/outputs/` — rendered MP4s and CSVs
- `data/jobs/` — pipeline job state for the Full mode progress UI
- `models/` — Roboflow YOLOv8 weights

These directories are gitignored and treated as shared runtime artifacts at the
repo root.

## Useful workspace commands

```bash
vp i                           # install dependencies
vp run                         # run configured tasks (or use `vp dev` / `vp build`)
vp dev                         # start the dev server from the root
vp build                       # build the SPA from the root
vp check                       # format, lint, and type checks from the root
vp preview                     # preview the built SPA from the root
```

```bash
uv run inference full --help
uv run web --help
```
