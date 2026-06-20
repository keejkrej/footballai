#!/usr/bin/env python3
"""Detect visible players and render a first-pass position overlay.

This is intentionally a broadcast-frame overlay, not a calibrated pitch map yet.
The next step is to replace normalized image coordinates with homography-projected
pitch coordinates once pitch-line calibration is stable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/youtube_clip.mp4", help="Input video path")
    parser.add_argument("--output", default="data/outputs/player_overlay.mp4", help="Output overlay video")
    parser.add_argument("--csv", default="data/outputs/player_positions.csv", help="Output detections CSV")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics model name or path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=900, help="Maximum frames to process")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    return parser.parse_args()


def draw_minimap(frame: np.ndarray, detections: list[dict[str, float]]) -> None:
    height, width = frame.shape[:2]
    map_w, map_h = 220, 140
    pad = 16
    x0, y0 = width - map_w - pad, pad
    x1, y1 = x0 + map_w, y0 + map_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 110, 55), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (235, 235, 235), 2)
    cv2.line(frame, (x0 + map_w // 2, y0), (x0 + map_w // 2, y1), (220, 220, 220), 1)
    cv2.circle(frame, (x0 + map_w // 2, y0 + map_h // 2), 18, (220, 220, 220), 1)

    for det in detections:
        px = int(x0 + det["x_norm"] * map_w)
        py = int(y0 + det["y_norm"] * map_h)
        cv2.circle(frame, (px, py), 4, (0, 220, 255), -1)

    cv2.putText(frame, "visible player map", (x0, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1, fps / max(1, args.stride))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    with csv_path.open("w", newline="") as csv_file:
        fieldnames = ["frame", "time_sec", "track_id", "x_norm", "y_norm", "x1", "y1", "x2", "y2", "confidence"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        frame_idx = 0
        processed = 0
        while processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            result = model.track(
                frame,
                persist=True,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                tracker="bytetrack.yaml",
            )[0]

            frame_detections: list[dict[str, float]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(boxes))

                for box, confidence, track_id in zip(boxes, confs, ids):
                    x1, y1, x2, y2 = box
                    foot_x = float((x1 + x2) / 2)
                    foot_y = float(y2)
                    x_norm = min(1.0, max(0.0, foot_x / width))
                    y_norm = min(1.0, max(0.0, foot_y / height))
                    det = {
                        "frame": frame_idx,
                        "time_sec": frame_idx / fps,
                        "track_id": int(track_id),
                        "x_norm": x_norm,
                        "y_norm": y_norm,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidence),
                    }
                    frame_detections.append(det)
                    csv_writer.writerow(det)

                    color = (0, 220, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (int(foot_x), int(foot_y)), 4, color, -1)
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (int(x1), max(18, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            draw_minimap(frame, frame_detections)
            cv2.putText(
                frame,
                f"visible players: {len(frame_detections)}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
            processed += 1
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote overlay video: {output_path}")
    print(f"Wrote detections CSV: {csv_path}")


if __name__ == "__main__":
    main()
