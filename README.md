# FootballAI

A Turbo + uv monorepo that combines a Svelte SPA with Python-based football
video analysis. The Python layer uses the **Roboflow sports** YOLOv8 soccer stack
(player, pitch, ball, and team classification). The webapp provides two modes:

- **Full**: paste a YouTube link, the Python WebSocket server downloads the clip
  and runs the full inference pipeline, then the SPA plays the annotated MP4.
- **Live**: paste any stream URL the browser can play (or use your webcam), the
  browser captures frames, sends them to the Python WebSocket server, and
  displays the annotated frames + live metrics.

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

# Start the live WebSocket server (frame-in / frame-out)
uv run inference live --host 0.0.0.0 --port 8000

# Start the unified WebSocket server used by the web UI
uv run web --port 8000
```

### Common options

| Option | Description |
|---|---|
| `--models-dir` | Directory containing the Roboflow `.pt` weights (default: `models/`) |
| `--device` | Torch device: `cuda`, `cpu`, `mps` |
| `--conf` | Detection confidence threshold (default: 0.25) |
| `--img-size` | Player/pitch inference size (default: 1280) |
| `--skip-team-fit` | Skip SigLIP/UMAP/KMeans team clustering (faster) |

### `inference full` options

| Option | Description |
|---|---|
| `--input` | Input MP4 path |
| `--output` | Output overlay MP4 path |
| `--csv` | Output detections CSV path |
| `--max-frames` | Maximum source frames to process (0 = unlimited) |
| `--stride` | Process every Nth source frame |
| `--batch-size` | Frames per GPU forward pass |
| `--team-sample-stride` | Frame stride for collecting team training crops |
| `--siglip-batch-size` | Batch size for SigLIP feature extraction |

### `inference live` / `web` options

| Option | Description |
|---|---|
| `--host` | WebSocket server host (default: `0.0.0.0`) |
| `--port` | WebSocket server port (default: `8000`) |

The `web` server accepts raw JPEG frames over the WebSocket and returns an
annotated JPEG frame plus JSON metadata. It also accepts JSON commands for full
jobs:

```json
{"action": "configure", "options": {"device": "cuda"}}
{"action": "full", "youtubeUrl": "...", "start": "00:00:00", "end": "00:02:00"}
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
2. Paste a stream URL (HLS `.m3u8`, MP4, WebM, etc.) or type `0`/`webcam`.
3. Click **Start live**.
4. The browser plays the stream, captures frames, sends them to Python, and
   renders the returned annotated frames.

### Stream compatibility note (e.g. ZDF World Cup streams)

Live mode can handle **any stream the browser is able to play unencrypted**. If
you paste an HLS URL, the app uses `hls.js`. If you paste a direct MP4/WebM URL,
the native `<video>` element handles it. Webcam input uses `getUserMedia`.

Official broadcaster streams such as **ZDF** are typically **geo-restricted**
and use DRM/encrypted DASH or HLS. Most browsers will refuse to surface
decrypted frames, and capturing them would likely violate the broadcaster's
terms of service and applicable copyright law. This app is intended for:

- Public-domain or Creative Commons footage
- Streams you own or have explicit rights to process
- Unencrypted practice/test HLS streams
- Your own webcam or local video files

## Headless capture

For streams that only work inside a browser, use the Playwright-based headless
capture CLI:

```bash
uv run inference-live-capture \
  --url "https://example.com/stream.m3u8" \
  --ws ws://localhost:8000 \
  --fps 5
```

This launches a headless Chromium instance, captures `<video>` frames, and feeds
them directly to the Python WebSocket server.

## Data layout

- `data/raw/` — downloaded source videos
- `data/outputs/` — rendered MP4s and CSVs
- `data/jobs/` — pipeline job state for the Full mode progress UI
- `models/` — Roboflow YOLOv8 weights

These directories are gitignored and treated as shared runtime artifacts at the
repo root.

## Useful workspace commands

```bash
cd apps/web && vp dev          # start the dev server
cd apps/web && vp build        # build for production
cd apps/web && vp check        # format, lint, and type checks
cd apps/web && vp preview      # preview the built SPA
```

```bash
uv run inference full --help
uv run inference live --help
uv run web --help
```
