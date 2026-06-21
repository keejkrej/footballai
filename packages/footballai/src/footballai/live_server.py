#!/usr/bin/env python3
"""WebSocket server for live frame-in / frame-out sports inference.

Clients connect and push raw JPEG frames. The server annotates each frame using
the Roboflow sports YOLOv8 stack and returns:
- a binary WebSocket message containing the annotated JPEG frame
- a text WebSocket message containing JSON metadata
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
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
from footballai.sports_football_overlay import SportsProcessor


MIN_TEAM_CROPS = 64


def _encode_frame(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return encoded.tobytes()


def _decode_frame(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode incoming JPEG frame")
    return frame


def _pressure_from_records(records: list[dict]) -> dict:
    players = [
        r for r in records
        if r["class_name"] in {"player", "goalkeeper"} and r.get("pitch_x_cm") is not None
    ]
    if not players:
        players = [r for r in records if r["class_name"] in {"player", "goalkeeper"}]
        left = sum(1 for r in players if r["x_norm"] < 0.33)
        right = sum(1 for r in players if r["x_norm"] > 0.66)
        advanced_left = sum(1 for r in players if r["x_norm"] < 0.25)
        advanced_right = sum(1 for r in players if r["x_norm"] > 0.75)
        delta = right - left
        side = "right" if advanced_right > advanced_left else "left" if advanced_left > advanced_right else "balanced"
        score = min(100, abs(delta) * 12 + max(advanced_left, advanced_right) * 10)
        return {"pressure_side": side, "pressure_score": score, "pitch_territory_delta": None}

    # Sports pitch config length is around 10500 cm; use normalized pitch x.
    length = 10500.0
    left = sum(1 for r in players if r["pitch_x_cm"] < length * 0.33)
    right = sum(1 for r in players if r["pitch_x_cm"] > length * 0.66)
    advanced_left = sum(1 for r in players if r["pitch_x_cm"] < length * 0.25)
    advanced_right = sum(1 for r in players if r["pitch_x_cm"] > length * 0.75)
    delta = right - left
    side = "right" if advanced_right > advanced_left else "left" if advanced_left > advanced_right else "balanced"
    score = min(100, abs(delta) * 8 + max(advanced_left, advanced_right) * 6)
    return {"pressure_side": side, "pressure_score": score, "pitch_territory_delta": delta}


async def _handle_connection(
    websocket: WebSocketServerProtocol,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
    skip_team_fit: bool,
    team_sample_stride: int,
) -> None:
    processor = SportsProcessor(
        models_dir=models_dir,
        device=device,
        conf=conf,
        img_size=img_size,
        skip_team_fit=skip_team_fit,
        team_sample_stride=team_sample_stride,
    )
    processor.load_models()

    running = True
    frame_idx = 0
    team_crops: list[np.ndarray] = []

    async def send_metadata(meta: dict) -> None:
        try:
            await websocket.send(json.dumps(meta))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def send_frame(jpeg: bytes) -> None:
        try:
            await websocket.send(jpeg)
        except websockets.exceptions.ConnectionClosed:
            pass

    while running:
        try:
            message = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            break

        # Text messages are treated as control/configuration messages.
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await send_metadata({"type": "error", "message": "Text message is not valid JSON"})
                continue

            action = payload.get("action")
            if action == "stop":
                running = False
                await send_metadata({"type": "status", "status": "stopped"})
                break
            if action == "configure":
                # Options sent here can override defaults for future frames if we
                # decide to expose them; for now we just acknowledge.
                await send_metadata({"type": "status", "status": "configured", "options": payload.get("options", {})})
                continue
            await send_metadata({"type": "error", "message": f"Unknown action: {action}"})
            continue

        # Binary messages are expected to be JPEG frames.
        started = time.time()
        try:
            frame = _decode_frame(message)
        except Exception as exc:
            await send_metadata({"type": "error", "message": f"Failed to decode frame: {exc}"})
            continue

        height, width = frame.shape[:2]
        fps = 25.0  # nominal; timestamps are synthetic for live mode

        try:
            records, annotated, meta = processor.process_frame(frame, frame_idx, fps)
        except Exception as exc:
            await send_metadata({"type": "error", "message": f"Inference failed: {exc}"})
            continue

        # Lazy team fitting from live crops
        if not skip_team_fit and not processor._team_fit_done:
            players = records
            # Actually records already include all classes; we need crops from the original frame.
            # Re-detect players cheaply from existing detections: we don't have sv.Detections here.
            # Simpler: fit from crops collected by re-running player model once in a while.
            # To keep latency sane, collect by reusing the player model on this frame.
            # This is slightly duplicative but acceptable for live warm-up.
            if frame_idx % team_sample_stride == 0:
                result = processor.player_model(frame, imgsz=img_size, verbose=False, device=device)[0]
                detections = sv.Detections.from_ultralytics(result)
                players_det = detections[detections.class_id == 2]
                team_crops += [sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy]
                if len(team_crops) >= MIN_TEAM_CROPS:
                    processor.fit_team_classifier_from_crops(team_crops)
                    team_crops = []

        try:
            jpeg = _encode_frame(annotated)
        except Exception as exc:
            await send_metadata({"type": "error", "message": f"Failed to encode annotated frame: {exc}"})
            continue

        classes = Counter(r["class_name"] for r in records)
        pressure = _pressure_from_records(records)
        possession = meta.get("possession")

        await send_frame(jpeg)
        await send_metadata(
            {
                "type": "metadata",
                "frame": frame_idx,
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

        frame_idx += 1


async def _connection_handler(
    websocket: WebSocketServerProtocol,
    path: str,
    *,
    models_dir: Path,
    device: str,
    conf: float,
    img_size: int,
    skip_team_fit: bool,
    team_sample_stride: int,
) -> None:
    try:
        await _handle_connection(
            websocket,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            skip_team_fit=skip_team_fit,
            team_sample_stride=team_sample_stride,
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
    skip_team_fit: bool,
    team_sample_stride: int,
) -> None:
    async def handler(websocket: WebSocketServerProtocol) -> None:
        await _connection_handler(
            websocket,
            "",
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            skip_team_fit=skip_team_fit,
            team_sample_stride=team_sample_stride,
        )

    print(f"Starting live inference WebSocket server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


def run_live_server(
    *,
    host: str,
    port: int,
    models_dir: Path | str = REPO_ROOT / "models",
    device: str = "cuda",
    conf: float = 0.25,
    img_size: int = 1280,
    skip_team_fit: bool = False,
    team_sample_stride: int = 60,
) -> None:
    models_dir = Path(models_dir)
    asyncio.run(
        _server_main(
            host=host,
            port=port,
            models_dir=models_dir,
            device=device,
            conf=conf,
            img_size=img_size,
            skip_team_fit=skip_team_fit,
            team_sample_stride=team_sample_stride,
        )
    )
