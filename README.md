# FootballAI

A Turbo + uv monorepo that combines a SvelteKit dashboard with Python-based
football video analysis. The Python layer uses the **Roboflow sports** YOLOv8
soccer stack (player, pitch, ball, and team classification). The webapp provides
two modes:

- **Full**: paste a YouTube link, the TypeScript backend downloads the clip and
  runs the full Python inference pipeline, then plays the annotated MP4.
- **Live**: paste any stream URL the browser can play (or use your webcam), the
  browser captures frames, sends them to a local Python WebSocket server, and
  displays the annotated frames + live metrics.

```
footballai/
├── apps/
│   └── web/                 # SvelteKit frontend + API routes
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
- Python 3.12 (managed by uv)
- NVIDIA GPU recommended (`--device cuda` is the default)
- `yt-dlp` on PATH for the webapp to download YouTube clips

## Install

```bash
# JavaScript workspace
pnpm install

# Python workspace (installs the footballai package and all dependencies)
uv sync --all-packages
```

Models are downloaded automatically on the first `inference full` or
`inference live` run. You can also trigger it explicitly:

```bash
uv run python -m footballai.setup_sports_models
```

## Python CLI

The Python package exposes exactly one command with two subcommands:

```bash
# Render a full overlay for a local MP4
uv run inference full \
  --input data/raw/clip.mp4 \
  --output data/outputs/overlay.mp4 \
  --csv data/outputs/positions.csv

# Start the live WebSocket server
uv run inference live --host 0.0.0.0 --port 8000
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
| `--team-sample-stride` | Frame stride for collecting team training crops |
| `--siglip-batch-size` | Batch size for SigLIP feature extraction |

### `inference live` options

| Option | Description |
|---|---|
| `--host` | WebSocket server host (default: `0.0.0.0`) |
| `--port` | WebSocket server port (default: `8000`) |

The live server accepts raw JPEG frames over the WebSocket and returns an
annotated JPEG frame plus JSON metadata. It expects a `configure` message first
and a `stop` message to end the session:

```json
{"action": "configure", "options": {}}
{"action": "stop"}
```

## Webapp

Start the SvelteKit dev server:

```bash
pnpm dev
```

Then open http://localhost:5173.

### Full mode

1. Paste a YouTube URL.
2. Optionally set start/end timestamps.
3. Click **Run pipeline**.
4. A progress modal shows the download and inference status.
5. When finished, the generated overlay appears in the runs list and the video
   player plays the annotated MP4.

The pipeline is orchestrated by the webapp: it spawns `yt-dlp` to download the
clip, then spawns `uv run inference full ...` and streams progress via stdout.

### Live mode

1. Make sure the Python WebSocket server is running:
   ```bash
   uv run inference live --port 8000
   ```
2. Paste a stream URL (HLS `.m3u8`, MP4, WebM, etc.) or type `0`/`webcam`.
3. Click **Start live**.
4. The browser plays the stream, captures frames, sends them to the Python
   server, and renders the returned annotated frames.

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

## Data layout

- `data/raw/` — downloaded source videos
- `data/outputs/` — rendered MP4s and CSVs
- `data/jobs/` — pipeline job state for the Full mode progress UI
- `models/` — Roboflow YOLOv8 weights

These directories are gitignored and treated as shared runtime artifacts at the
repo root.

## Useful workspace commands

```bash
pnpm dev        # start the SvelteKit app
pnpm build      # build all workspace packages
pnpm check      # run TypeScript checks
```

```bash
uv run inference full --help
uv run inference live --help
```
