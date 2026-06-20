#!/usr/bin/env python3
"""Run live football inference from a file, webcam, HLS, RTMP, or HTTP stream.

This writes a rolling JSON snapshot for the SvelteKit frontend. It is designed
as a low-friction live loop first; calibrated pitch coordinates, team assignment,
and betting-market heads can be layered on top of the same snapshot schema.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from football_yolov5_overlay import (
    CLASS_COLORS,
    ball_holder,
    box_top_center,
    draw_ellipse,
    draw_label,
    draw_marker,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index, file path, HLS .m3u8, RTMP, HTTP URL, or capture device",
    )
    parser.add_argument("--weights", default="models/football_yolov5_best.pt", help="YOLOv5 custom weights")
    parser.add_argument("--yolov5-repo", default="external/yolov5", help="Local YOLOv5 repository path")
    parser.add_argument("--state", default="data/live/latest.json", help="Rolling JSON state output")
    parser.add_argument("--overlay-output", default="", help="Optional MP4 overlay recording path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--img-size", type=int, default=960, help="YOLOv5 inference image size")
    parser.add_argument("--stride", type=int, default=5, help="Run inference every Nth source frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N source frames; 0 means unlimited")
    parser.add_argument("--window", type=int, default=60, help="Number of inference snapshots for rolling metrics")
    return parser.parse_args()


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def side_pressure(detections: list[dict]) -> dict:
    players = [det for det in detections if det["class_name"] in {"player", "goalkeeper"}]
    left_third = sum(1 for det in players if det["x_norm"] < 0.33)
    middle_third = sum(1 for det in players if 0.33 <= det["x_norm"] <= 0.66)
    right_third = sum(1 for det in players if det["x_norm"] > 0.66)
    advanced_left = sum(1 for det in players if det["x_norm"] < 0.25)
    advanced_right = sum(1 for det in players if det["x_norm"] > 0.75)

    territory_delta = right_third - left_third
    pressure_side = "right" if advanced_right > advanced_left else "left" if advanced_left > advanced_right else "balanced"
    pressure_score = min(100, abs(territory_delta) * 12 + max(advanced_left, advanced_right) * 10)

    return {
        "left_third_players": left_third,
        "middle_third_players": middle_third,
        "right_third_players": right_third,
        "advanced_left_players": advanced_left,
        "advanced_right_players": advanced_right,
        "territory_delta": territory_delta,
        "pressure_side": pressure_side,
        "pressure_score": pressure_score,
    }


def trading_snapshot(history: deque[dict]) -> dict:
    if not history:
        return {"leader": "unknown", "confidence": 0, "commentary": "Waiting for live detections."}

    sides = [item["pressure"]["pressure_side"] for item in history if item["pressure"]["pressure_side"] != "balanced"]
    side_counts = Counter(sides)
    leader = side_counts.most_common(1)[0][0] if side_counts else "balanced"
    avg_pressure = sum(item["pressure"]["pressure_score"] for item in history) / len(history)
    ball_seen_rate = sum(1 for item in history if item["classes"].get("ball", 0) > 0) / len(history)
    possession_seen_rate = sum(1 for item in history if item["possession"] is not None) / len(history)
    confidence = int(min(100, avg_pressure * 0.65 + ball_seen_rate * 20 + possession_seen_rate * 15))

    commentary = (
        "The visual territory signal is balanced; no clear pressure edge yet."
        if leader == "balanced"
        else f"The {leader} side has the stronger recent territory and pressure signal."
    )

    return {
        "leader": leader,
        "confidence": confidence,
        "avg_pressure": round(avg_pressure, 2),
        "ball_seen_rate": round(ball_seen_rate, 3),
        "possession_seen_rate": round(possession_seen_rate, 3),
        "commentary": commentary,
    }


def detections_from_prediction(pred: np.ndarray, names: dict[int, str], width: int, height: int, frame_idx: int, fps: float) -> list[dict]:
    detections = []
    for x1, y1, x2, y2, confidence, class_id in pred:
        class_name = names[int(class_id)]
        foot_x = float((x1 + x2) / 2)
        foot_y = float(y2)
        detections.append(
            {
                "frame": frame_idx,
                "time_sec": frame_idx / fps if fps else 0,
                "class_name": class_name,
                "confidence": float(confidence),
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "x_norm": min(1.0, max(0.0, foot_x / width)),
                "y_norm": min(1.0, max(0.0, foot_y / height)),
            }
        )
    return detections


def json_safe_detection(det: dict | None) -> str | None:
    return det["class_name"] if det else None


def render_overlay(frame: np.ndarray, detections: list[dict], possession: dict | None, snapshot: dict) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        color = CLASS_COLORS.get(det["class_name"], (255, 255, 255))
        draw_ellipse(annotated, det["box"], color)
        draw_label(annotated, det["box"], f"{det['class_name']} {det['confidence']:.2f}", color)

    for ball in [det for det in detections if det["class_name"] == "ball"]:
        draw_marker(annotated, box_top_center(ball["box"]), (0, 255, 0))
    if possession:
        draw_marker(annotated, box_top_center(possession["box"]), (0, 0, 255))

    headline = (
        f"pressure: {snapshot['pressure']['pressure_side']} {snapshot['pressure']['pressure_score']} | "
        f"edge: {snapshot['trading']['leader']} {snapshot['trading']['confidence']}%"
    )
    cv2.putText(annotated, headline, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return annotated


def main() -> None:
    args = parse_args()
    state_path = Path(args.state)
    model = load_model(args.yolov5_repo, args.weights, args.conf)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    writer = None
    if args.overlay_output:
        output_path = Path(args.overlay_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, fps / max(1, args.stride)),
            (width, height),
        )

    history: deque[dict] = deque(maxlen=args.window)
    frame_idx = 0
    last_snapshot = None

    while args.max_frames <= 0 or frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.25)
            frame_idx += 1
            continue

        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        started = time.time()
        results = model(frame, size=args.img_size)
        pred = results.pred[0].detach().cpu().numpy()
        detections = detections_from_prediction(pred, model.names, width, height, frame_idx, fps)
        classes = dict(Counter(det["class_name"] for det in detections))
        players = [det for det in detections if det["class_name"] in {"player", "goalkeeper"}]
        balls = [det for det in detections if det["class_name"] == "ball"]
        possession = ball_holder(players, balls)
        pressure = side_pressure(detections)

        current = {
            "status": "running",
            "source": args.source,
            "updated_at": time.time(),
            "frame": frame_idx,
            "fps": fps,
            "latency_ms": round((time.time() - started) * 1000, 1),
            "classes": classes,
            "detections": len(detections),
            "possession": json_safe_detection(possession),
            "pressure": pressure,
        }
        history.append(current)
        snapshot = {**current, "trading": trading_snapshot(history)}
        last_snapshot = snapshot
        atomic_write_json(state_path, snapshot)

        if writer:
            writer.write(render_overlay(frame, detections, possession, snapshot))

        print(json.dumps({k: snapshot[k] for k in ["frame", "classes", "pressure", "trading"]}), flush=True)
        frame_idx += 1

    if last_snapshot:
        atomic_write_json(state_path, {**last_snapshot, "status": "stopped"})
    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
