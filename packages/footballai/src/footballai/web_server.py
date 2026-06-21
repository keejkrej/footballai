#!/usr/bin/env python3
"""Pure WebSocket server for the FootballAI web UI.

The static Svelte SPA should be served by Nginx (or any static file server) and
proxy WebSocket connections to this server. Example Nginx snippet:

    server {
        listen 80;
        root /home/jack/workspace/footballai_main/apps/web/dist;
        location / {
            try_files $uri $uri/ /index.html;
        }
        location /ws {
            proxy_pass http://127.0.0.1:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        location /media {
            alias /home/jack/workspace/footballai_main/data/outputs;
        }
    }

WebSocket protocol:
- Text messages are JSON commands:
  { "action": "full", "youtubeUrl": "...", "start": "...", "end": "..." }
  { "action": "configure", "options": { "device": "cuda" } }
  { "action": "stop" }
- Binary messages are JPEG frames for live inference.
- Server replies with:
  - text: { "type": "metadata", ... } or { "type": "job_progress", ... }
  - binary: annotated JPEG frame (for live mode)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import websockets
import websockets.exceptions
from websockets.server import WebSocketServerProtocol

from footballai._paths import REPO_ROOT
from footballai.download_youtube_clip import download_youtube_clip
from footballai.live_server import (
    MIN_TEAM_CROPS,
    _decode_frame,
    _encode_frame,
    _pressure_from_records,
)
from footballai.setup_sports_models import ensure_models
from footballai.sports_football_overlay import SportsProcessor, run_full

# Paths
OUTPUTS_DIR = REPO_ROOT / "data" / "outputs"
RAW_DIR = REPO_ROOT / "data" / "raw"
JOBS_DIR = REPO_ROOT / "data" / "jobs"

for d in (OUTPUTS_DIR, RAW_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: dict) -> None:
    _job_path(job_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_name(url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in url.split("?")[0])[:60]


def _summarize_csv(csv_path: Path) -> tuple[int, dict[str, int]]:
    if not csv_path.exists():
        return 0, {}
    rows = 0
    classes: Counter[str] = Counter()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            classes[row.get("class_name", "unknown")] += 1
    return rows, dict(classes)


def _video_duration_label(path: Path) -> str:
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        seconds = frames / fps if fps else 0
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    except Exception:
        return "00:00"


def _load_runs() -> list[dict]:
    runs: list[dict] = []
    for mp4 in sorted(OUTPUTS_DIR.glob("*_overlay.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        csv_path = mp4.with_name(mp4.name.replace("_overlay.mp4", ".csv"))
        detections, classes = _summarize_csv(csv_path)
        runs.append(
            {
                "id": mp4.stem,
                "label": mp4.stem,
                "video": f"/media/{mp4.name}",
                "csv": f"/media/{csv_path.name}" if csv_path.exists() else None,
                "sizeBytes": mp4.stat().st_size,
                "durationLabel": _video_duration_label(mp4),
                "detections": detections,
                "classes": classes,
            }
        )
    return runs


# ---------------------------------------------------------------------------
# Full job runner
# ---------------------------------------------------------------------------


def _run_full_job(
    job_id: str,
    youtube_url: str,
    start: str,
    end: str,
    options: dict,
    send_json: Any,
) -> None:
    stem = f"{uuid.uuid4().hex[:8]}_{_safe_name(youtube_url)}"
    raw_path = RAW_DIR / f"{stem}.mp4"
    overlay_path = OUTPUTS_DIR / f"{stem}_overlay.mp4"
    csv_path = OUTPUTS_DIR / f"{stem}.csv"

    def progress(status: str, pct: int, msg: str, extra: dict | None = None) -> None:
        payload = {"type": "job_progress", "job_id": job_id, "status": status, "progress": pct, "message": msg}
        if extra:
            payload.update(extra)
        _write_job(job_id, {**payload, "id": job_id})
        send_json(payload)

    try:
        progress("downloading", 5, "Downloading YouTube clip")
        download_youtube_clip(
            youtube_url,
            raw_path,
            start=start,
            end=end,
            height=options.get("height", 720),
        )

        progress("inferencing", 10, "Running inference")

        def on_progress(progress: dict) -> None:
            stage = progress.get("stage", "inference")
            processed = progress.get("processed", 0)
            total = progress.get("total")
            if stage == "done":
                pct = 100
                msg = "Done"
                status = "done"
            else:
                if total and total > 0:
                    pct = min(99, int(10 + 90 * processed / total))
                else:
                    pct = min(99, 10 + processed // 100)
                msg = f"Inference {processed}/{total or '?'} frames @ {progress.get('inference_fps', 0):.1f} fps"
                status = "inferencing"
            progress_payload = {
                "type": "job_progress",
                "job_id": job_id,
                "status": status,
                "progress": pct,
                "message": msg,
                "classes": progress.get("classes", {}),
            }
            _write_job(job_id, {**progress_payload, "id": job_id})
            send_json(progress_payload)

        run_full(
            input_path=raw_path,
            output_path=overlay_path,
            csv_path=csv_path,
            models_dir=options.get("models_dir", REPO_ROOT / "models"),
            device=options.get("device", "cuda"),
            conf=options.get("conf", 0.25),
            img_size=options.get("img_size", 1280),
            max_frames=options.get("max_frames", 0),
            stride=options.get("stride", 1),
            batch_size=options.get("batch_size", 4),
            skip_team_fit=options.get("skip_team_fit", False),
            team_sample_stride=options.get("team_sample_stride", 60),
            siglip_batch_size=options.get("siglip_batch_size", 64),
            team_cache=options.get("team_cache", True),
            team_cache_dir=options.get("team_cache_dir", REPO_ROOT / "data" / "cache" / "team_classifier"),
            decoder_queue_size=options.get("decoder_queue_size", 32),
            writer_queue_size=options.get("writer_queue_size", 32),
            on_progress=on_progress,
        )

        detections, classes = _summarize_csv(csv_path)
        video_url = f"/media/{overlay_path.name}"
        csv_url = f"/media/{csv_path.name}"
        _write_job(
            job_id,
            {
                "id": job_id,
                "status": "done",
                "progress": 100,
                "message": "Done",
                "videoUrl": video_url,
                "csvUrl": csv_url,
                "detections": detections,
                "classes": classes,
            },
        )
        send_json(
            {
                "type": "job_progress",
                "job_id": job_id,
                "status": "done",
                "progress": 100,
                "message": "Done",
                "video_url": video_url,
                "csv_url": csv_url,
                "detections": detections,
                "classes": classes,
            }
        )
    except Exception as exc:
        progress("error", 0, str(exc))


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def _handle_connection(
    websocket: WebSocketServerProtocol,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
) -> None:
    loop = asyncio.get_running_loop()
    processor: SportsProcessor | None = None
    live_frame_idx = 0
    team_crops: list[np.ndarray] = []
    team_sample_stride = 60
    skip_team_fit = False
    current_device = device

    def send_json_sync(payload: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(websocket.send(json.dumps(payload)), loop)
        except Exception:
            pass

    async def send_json(payload: dict) -> None:
        try:
            await websocket.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def send_binary(data: bytes) -> None:
        try:
            await websocket.send(data)
        except websockets.exceptions.ConnectionClosed:
            pass

    while True:
        try:
            message = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            break

        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "Text message is not valid JSON"})
                continue

            action = payload.get("action")
            if action == "stop":
                await send_json({"type": "status", "status": "stopped"})
                continue
            if action == "cancel":
                await send_json({"type": "status", "status": "cancelled"})
                continue
            if action == "configure":
                opts = payload.get("options", {})
                current_device = opts.get("device", current_device)
                team_sample_stride = opts.get("team_sample_stride", team_sample_stride)
                skip_team_fit = opts.get("skip_team_fit", skip_team_fit)
                await send_json({"type": "status", "status": "configured", "options": opts})
                continue
            if action == "full":
                job_id = str(uuid.uuid4())
                options = {
                    "models_dir": models_dir,
                    "device": payload.get("device", current_device),
                    "conf": conf,
                    "img_size": img_size,
                    "max_frames": payload.get("max_frames", 0),
                    "stride": payload.get("stride", 1),
                    "batch_size": payload.get("batch_size", 4),
                    "skip_team_fit": payload.get("skip_team_fit", skip_team_fit),
                    "team_sample_stride": payload.get("team_sample_stride", team_sample_stride),
                    "siglip_batch_size": payload.get("siglip_batch_size", 64),
                    "team_cache": payload.get("team_cache", True),
                }
                _write_job(job_id, {"id": job_id, "status": "pending", "progress": 0, "message": "Queued"})
                await send_json({"type": "job_progress", "job_id": job_id, "status": "pending", "progress": 0, "message": "Queued"})

                def run_job() -> None:
                    _run_full_job(
                        job_id,
                        payload.get("youtubeUrl", ""),
                        payload.get("start", "00:00:00"),
                        payload.get("end", "00:02:00"),
                        options,
                        send_json_sync,
                    )

                threading.Thread(target=run_job, daemon=True).start()
                continue

            await send_json({"type": "error", "message": f"Unknown action: {action}"})
            continue

        # Binary messages are live frames.
        started = time.time()
        try:
            frame = _decode_frame(message)
        except Exception as exc:
            await send_json({"type": "error", "message": f"Failed to decode frame: {exc}"})
            continue

        if processor is None:
            processor = SportsProcessor(
                models_dir=models_dir,
                device=current_device,
                conf=conf,
                img_size=img_size,
                skip_team_fit=skip_team_fit,
                team_sample_stride=team_sample_stride,
            )
            processor.load_models()

        height, width = frame.shape[:2]
        try:
            records, annotated, meta = processor.process_frame(frame, live_frame_idx, 25.0)
        except Exception as exc:
            await send_json({"type": "error", "message": f"Inference failed: {exc}"})
            continue

        if not skip_team_fit and not processor._team_fit_done:
            if live_frame_idx % team_sample_stride == 0:
                result = processor.player_model(frame, imgsz=img_size, verbose=False, device=current_device)[0]
                detections = sv.Detections.from_ultralytics(result)
                players_det = detections[detections.class_id == 2]
                team_crops += [sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy]
                if len(team_crops) >= MIN_TEAM_CROPS:
                    processor.fit_team_classifier_from_crops(team_crops)
                    team_crops = []

        try:
            jpeg = _encode_frame(annotated)
        except Exception as exc:
            await send_json({"type": "error", "message": f"Failed to encode annotated frame: {exc}"})
            continue

        classes = Counter(r["class_name"] for r in records)
        pressure = _pressure_from_records(records)
        possession = meta.get("possession")

        await send_binary(jpeg)
        await send_json(
            {
                "type": "metadata",
                "frame": live_frame_idx,
                "latency_ms": round((time.time() - started) * 1000, 1),
                "width": width,
                "height": height,
                "classes": dict(classes),
                "detections": len(records),
                "possession": possession,
                "pressure": pressure,
                "team_ready": processor._team_fit_done or skip_team_fit,
            }
        )
        live_frame_idx += 1


async def _connection_handler(
    websocket: WebSocketServerProtocol,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
) -> None:
    try:
        await _handle_connection(
            websocket,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
        )
    except Exception as exc:
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
        except websockets.exceptions.ConnectionClosed:
            pass


async def _server_main(
    *,
    host: str,
    port: int,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
) -> None:
    async def handler(websocket: WebSocketServerProtocol) -> None:
        await _connection_handler(
            websocket,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
        )

    print(f"Starting FootballAI WebSocket server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def run_web_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    models_dir: Path | str = REPO_ROOT / "models",
    device: str = "cuda",
    conf: float = 0.25,
    img_size: int = 1280,
) -> None:
    models_dir = Path(models_dir)
    ensure_models(models_dir)
    asyncio.run(
        _server_main(
            host=host,
            port=port,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FootballAI web UI WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=1280)
    args = parser.parse_args()
    run_web_server(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        device=args.device,
        conf=args.conf,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
