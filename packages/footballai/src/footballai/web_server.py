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
  { "action": "live_start", "source": { "type": "file", "path": "...", "start": "00:00:00", "end": "00:02:00" } }
  { "action": "live_start", "source": { "type": "youtube", "url": "...", "start": "00:00:00", "end": "00:02:00" } }
  { "action": "live_start", "source": { "type": "obs", "device": "/dev/video2" } }
  { "action": "live_stop" }
  { "action": "configure", "options": { "device": "cuda" } }
  { "action": "runs" }
  { "action": "job", "id": "..." }
  { "action": "stop" }
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
from footballai.setup_sports_models import ensure_models
from footballai.sports_football_overlay import SportsProcessor, run_full
from footballai.state_model.predictor import FootballStatePredictor
from footballai.video_sources import VideoSource, create_video_source

# Paths
OUTPUTS_DIR = REPO_ROOT / "data" / "outputs"
RAW_DIR = REPO_ROOT / "data" / "raw"
JOBS_DIR = REPO_ROOT / "data" / "jobs"

for d in (OUTPUTS_DIR, RAW_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MIN_TEAM_CROPS = 64


# ---------------------------------------------------------------------------
# Live helpers (inlined from the removed live_server.py)
# ---------------------------------------------------------------------------


def _encode_frame(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return encoded.tobytes()


def _pressure_from_records(records: list[dict]) -> dict:
    players = [
        r
        for r in records
        if r["class_name"] in {"player", "goalkeeper"}
        and r.get("pitch_x_cm") is not None
    ]
    if not players:
        players = [r for r in records if r["class_name"] in {"player", "goalkeeper"}]
        left = sum(1 for r in players if r["x_norm"] < 0.33)
        right = sum(1 for r in players if r["x_norm"] > 0.66)
        advanced_left = sum(1 for r in players if r["x_norm"] < 0.25)
        advanced_right = sum(1 for r in players if r["x_norm"] > 0.75)
        delta = right - left
        side = (
            "right"
            if advanced_right > advanced_left
            else "left"
            if advanced_left > advanced_right
            else "balanced"
        )
        score = min(100, abs(delta) * 12 + max(advanced_left, advanced_right) * 10)
        return {
            "pressure_side": side,
            "pressure_score": score,
            "pitch_territory_delta": None,
        }

    length = 10500.0
    left = sum(1 for r in players if r["pitch_x_cm"] < length * 0.33)
    right = sum(1 for r in players if r["pitch_x_cm"] > length * 0.66)
    advanced_left = sum(1 for r in players if r["pitch_x_cm"] < length * 0.25)
    advanced_right = sum(1 for r in players if r["pitch_x_cm"] > length * 0.75)
    delta = right - left
    side = (
        "right"
        if advanced_right > advanced_left
        else "left"
        if advanced_left > advanced_right
        else "balanced"
    )
    score = min(100, abs(delta) * 8 + max(advanced_left, advanced_right) * 6)
    return {
        "pressure_side": side,
        "pressure_score": score,
        "pitch_territory_delta": delta,
    }


# ---------------------------------------------------------------------------
# Full job runner
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
    for mp4 in sorted(
        OUTPUTS_DIR.glob("*_overlay.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
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


def _run_full_job(
    job_id: str,
    youtube_url: str,
    start: str,
    end: str,
    options: dict,
    send_json: Any,
    state_predictor: FootballStatePredictor | None,
) -> None:
    stem = f"{uuid.uuid4().hex[:8]}_{_safe_name(youtube_url)}"
    raw_path = RAW_DIR / f"{stem}.mp4"
    overlay_path = OUTPUTS_DIR / f"{stem}_overlay.mp4"
    csv_path = OUTPUTS_DIR / f"{stem}.csv"

    def progress(status: str, pct: int, msg: str, extra: dict | None = None) -> None:
        payload = {
            "type": "job_progress",
            "job_id": job_id,
            "status": status,
            "progress": pct,
            "message": msg,
        }
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
            team_sample_stride=options.get("team_sample_stride", 60),
            siglip_batch_size=options.get("siglip_batch_size", 64),
            team_cache=options.get("team_cache", True),
            team_cache_dir=options.get(
                "team_cache_dir", REPO_ROOT / "data" / "cache" / "team_classifier"
            ),
            decoder_queue_size=options.get("decoder_queue_size", 32),
            writer_queue_size=options.get("writer_queue_size", 32),
            on_progress=on_progress,
            state_predictor=state_predictor,
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
# Live loop
# ---------------------------------------------------------------------------


async def _run_live_loop(
    websocket: WebSocketServerProtocol,
    source: VideoSource,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
    team_sample_stride: int,
    state_predictor: FootballStatePredictor | None,
) -> None:
    processor = SportsProcessor(
        models_dir=models_dir,
        device=device,
        conf=conf,
        img_size=img_size,
        team_sample_stride=team_sample_stride,
        state_predictor=state_predictor,
    )
    processor.load_models()
    if state_predictor is not None:
        state_predictor.reset()

    team_crops: list[np.ndarray] = []
    frame_idx = 0

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

    try:
        await source.open()
        async for _, frame, fps in source.frames():
            started = time.time()
            try:
                records, annotated, meta = processor.process_frame(
                    frame, frame_idx, fps
                )
            except Exception as exc:
                await send_json(
                    {"type": "error", "message": f"Inference failed: {exc}"}
                )
                frame_idx += 1
                continue

            # Lazy team fitting from live crops
            if not processor._team_fit_done:
                if frame_idx % team_sample_stride == 0:
                    result = processor.player_model(
                        frame, imgsz=img_size, verbose=False, device=device
                    )[0]
                    detections = sv.Detections.from_ultralytics(result)
                    players_det = detections[detections.class_id == 2]
                    team_crops += [
                        sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy
                    ]
                    if len(team_crops) >= MIN_TEAM_CROPS:
                        processor.fit_team_classifier_from_crops(team_crops)
                        team_crops = []

            try:
                jpeg = _encode_frame(annotated)
            except Exception as exc:
                await send_json(
                    {
                        "type": "error",
                        "message": f"Failed to encode annotated frame: {exc}",
                    }
                )
                frame_idx += 1
                continue

            classes = Counter(r["class_name"] for r in records)
            pressure = _pressure_from_records(records)
            possession = meta.get("possession")
            state_probs = meta.get("state")
            # Send only the human-readable probability readouts, not the full
            # 128-d state vector, to keep WebSocket messages small.
            state_summary = (
                {k: v for k, v in state_probs.items() if k != "state_vector"}
                if state_probs else None
            )

            await send_binary(jpeg)
            await send_json(
                {
                    "type": "metadata",
                    "frame": frame_idx,
                    "latency_ms": round((time.time() - started) * 1000, 1),
                    "width": source.width,
                    "height": source.height,
                    "classes": dict(classes),
                    "detections": len(records),
                    "possession": possession,
                    "pressure": pressure,
                    "state": state_summary,
                    "team_ready": processor._team_fit_done,
                }
            )
            frame_idx += 1
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        try:
            await websocket.send(
                json.dumps({"type": "error", "message": f"Live source failed: {exc}"})
            )
        except websockets.exceptions.ConnectionClosed:
            pass
    finally:
        await source.close()
        try:
            await websocket.send(json.dumps({"type": "status", "status": "live_ended"}))
        except websockets.exceptions.ConnectionClosed:
            pass


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
    state_predictor: FootballStatePredictor | None,
) -> None:
    loop = asyncio.get_running_loop()

    # Defaults configurable via "configure"
    current_device = device
    team_sample_stride = 60

    live_task: asyncio.Task | None = None
    live_source: VideoSource | None = None

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

    async def stop_live() -> None:
        nonlocal live_task, live_source
        if live_source is not None:
            source = live_source
            live_source = None
            await source.close()
        if live_task is not None and not live_task.done():
            live_task.cancel()
            try:
                await live_task
            except asyncio.CancelledError:
                pass
            live_task = None

    try:
        while True:
            try:
                message = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                break

            if isinstance(message, str):
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await send_json(
                        {"type": "error", "message": "Text message is not valid JSON"}
                    )
                    continue

                action = payload.get("action")
                if action == "stop" or action == "live_stop":
                    await stop_live()
                    await send_json({"type": "status", "status": "stopped"})
                    continue
                if action == "cancel":
                    await send_json({"type": "status", "status": "cancelled"})
                    continue
                if action == "configure":
                    opts = payload.get("options", {})
                    current_device = opts.get("device", current_device)
                    team_sample_stride = opts.get(
                        "team_sample_stride", team_sample_stride
                    )
                    await send_json(
                        {"type": "status", "status": "configured", "options": opts}
                    )
                    continue
                if action == "live_start":
                    await stop_live()
                    source_config = payload.get("source")
                    if not source_config:
                        await send_json(
                            {
                                "type": "error",
                                "message": "live_start requires a 'source' object",
                            }
                        )
                        continue
                    try:
                        source = create_video_source(source_config)
                    except Exception as exc:
                        await send_json(
                            {"type": "error", "message": f"Invalid source: {exc}"}
                        )
                        continue

                    live_options = payload.get("options", {})
                    live_source = source
                    live_task = asyncio.create_task(
                        _run_live_loop(
                            websocket,
                            source,
                            models_dir=models_dir,
                            device=live_options.get("device", current_device),
                            conf=live_options.get("conf", conf),
                            img_size=live_options.get("img_size", img_size),
                            team_sample_stride=live_options.get(
                                "team_sample_stride", team_sample_stride
                            ),
                            state_predictor=state_predictor,
                        )
                    )
                    await send_json({"type": "status", "status": "live_started"})
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
                        "team_sample_stride": payload.get(
                            "team_sample_stride", team_sample_stride
                        ),
                        "siglip_batch_size": payload.get("siglip_batch_size", 64),
                        "team_cache": payload.get("team_cache", True),
                    }
                    _write_job(
                        job_id,
                        {
                            "id": job_id,
                            "status": "pending",
                            "progress": 0,
                            "message": "Queued",
                        },
                    )
                    await send_json(
                        {
                            "type": "job_progress",
                            "job_id": job_id,
                            "status": "pending",
                            "progress": 0,
                            "message": "Queued",
                        }
                    )

                    def run_job() -> None:
                        _run_full_job(
                            job_id,
                            payload.get("youtubeUrl", ""),
                            payload.get("start", "00:00:00"),
                            payload.get("end", "00:02:00"),
                            options,
                            send_json_sync,
                            state_predictor,
                        )

                    threading.Thread(target=run_job, daemon=True).start()
                    continue
                if action == "runs":
                    await send_json({"type": "runs", "runs": _load_runs()})
                    continue
                if action == "job":
                    job_id = payload.get("id", "")
                    path = _job_path(job_id)
                    if path.exists():
                        await send_json(
                            {
                                "type": "job",
                                "job": json.loads(path.read_text(encoding="utf-8")),
                            }
                        )
                    else:
                        await send_json({"type": "error", "message": "job not found"})
                    continue

                await send_json(
                    {"type": "error", "message": f"Unknown action: {action}"}
                )
                continue

            # Binary messages are no longer accepted as live input.
            await send_json(
                {
                    "type": "error",
                    "message": "Binary frames are not accepted; use live_start with a source configuration",
                }
            )
    finally:
        await stop_live()


async def _connection_handler(
    websocket: WebSocketServerProtocol,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
    state_predictor: FootballStatePredictor | None,
) -> None:
    try:
        await _handle_connection(
            websocket,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            state_predictor=state_predictor,
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
    state_predictor: FootballStatePredictor | None,
) -> None:
    async def handler(websocket: WebSocketServerProtocol) -> None:
        await _connection_handler(
            websocket,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            state_predictor=state_predictor,
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
    state_checkpoint: Path | str | None = REPO_ROOT / "models" / "footballai_state_model.ckpt",
) -> None:
    models_dir = Path(models_dir)
    ensure_models(models_dir)

    state_predictor: FootballStatePredictor | None = None
    if state_checkpoint:
        checkpoint_path = Path(state_checkpoint)
        if checkpoint_path.exists():
            try:
                state_predictor = FootballStatePredictor(
                    checkpoint_path,
                    device=device,
                    fps=25.0,
                )
                print(f"Loaded state model from {checkpoint_path}")
            except Exception as exc:
                print(f"Failed to load state model: {exc}; continuing without state readouts")
        else:
            print(f"State checkpoint not found at {checkpoint_path}; continuing without state readouts")

    asyncio.run(
        _server_main(
            host=host,
            port=port,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            state_predictor=state_predictor,
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
    parser.add_argument(
        "--state-checkpoint",
        default=str(REPO_ROOT / "models" / "footballai_state_model.ckpt"),
        help="Path to the football state model Lightning checkpoint (.ckpt)",
    )
    args = parser.parse_args()
    run_web_server(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        device=args.device,
        conf=args.conf,
        img_size=args.img_size,
        state_checkpoint=args.state_checkpoint,
    )


if __name__ == "__main__":
    main()
