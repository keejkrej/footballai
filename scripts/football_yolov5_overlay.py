#!/usr/bin/env python3
"""Render football-specific overlays using the Roboflow YOLOv5 model.

The model expected by default is the deprecated-but-useful tutorial checkpoint
with classes: ball, goalkeeper, player, referee.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


CLASS_COLORS = {
    "ball": (255, 255, 255),
    "goalkeeper": (1, 1, 133),
    "player": (187, 212, 0),
    "referee": (0, 255, 255),
}
TRACK_CLASSES = {"player", "goalkeeper", "referee"}
POSSESSION_PROXIMITY_PX = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/youtube_clip.mp4", help="Input video path")
    parser.add_argument("--output", default="data/outputs/football_yolov5_overlay.mp4", help="Output overlay video")
    parser.add_argument("--csv", default="data/outputs/football_yolov5_positions.csv", help="Output detections CSV")
    parser.add_argument("--weights", default="models/football_yolov5_best.pt", help="YOLOv5 custom weights")
    parser.add_argument("--yolov5-repo", default="external/yolov5", help="Local YOLOv5 repository path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--img-size", type=int, default=1280, help="YOLOv5 inference image size")
    parser.add_argument("--max-frames", type=int, default=900, help="Maximum source frames to process")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth source frame")
    return parser.parse_args()


def draw_ellipse(frame: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> None:
    x1, y1, x2, _ = box.astype(int)
    width = max(4, x2 - x1)
    center = (int((x1 + x2) / 2), int(box[3]))
    axes = (int(width / 2), max(3, int(0.18 * width)))
    cv2.ellipse(frame, center, axes, 0, -45, 235, color, thickness, lineType=cv2.LINE_4)


def draw_label(frame: np.ndarray, box: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, _, _ = box.astype(int)
    cv2.putText(frame, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_marker(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    marker = np.array([[x - 10, y - 30], [x, y - 10], [x + 10, y - 30]], dtype=np.int32)
    cv2.drawContours(frame, [marker], 0, color, -1)
    cv2.drawContours(frame, [marker], 0, (0, 0, 0), 2)


def box_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)


def box_top_center(box: np.ndarray) -> tuple[int, int]:
    return int((box[0] + box[2]) / 2), int(box[1])


def ball_holder(players: list[dict], balls: list[dict]) -> dict | None:
    if len(balls) != 1:
        return None
    bx, by = box_center(balls[0]["box"])
    for player in players:
        x1, y1, x2, y2 = player["box"]
        if x1 - POSSESSION_PROXIMITY_PX <= bx <= x2 + POSSESSION_PROXIMITY_PX and y1 - POSSESSION_PROXIMITY_PX <= by <= y2 + POSSESSION_PROXIMITY_PX:
            return player
    return None


def load_model(yolov5_repo: str, weights: str, conf: float):
    repo_path = Path(yolov5_repo).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    model = torch.hub.load(str(repo_path), "custom", weights, source="local", device="cpu")
    model.conf = conf
    return model


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.yolov5_repo, args.weights, args.conf)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1, fps / max(1, args.stride))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))

    with csv_path.open("w", newline="") as csv_file:
        fieldnames = ["frame", "time_sec", "class_name", "x_norm", "y_norm", "x1", "y1", "x2", "y2", "confidence"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        frame_idx = 0
        written = 0
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            results = model(frame, size=args.img_size)
            pred = results.pred[0].detach().cpu().numpy()
            detections: list[dict] = []
            for x1, y1, x2, y2, confidence, class_id in pred:
                class_name = model.names[int(class_id)]
                box = np.array([x1, y1, x2, y2], dtype=float)
                foot_x = float((x1 + x2) / 2)
                foot_y = float(y2)
                det = {
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,
                    "class_name": class_name,
                    "x_norm": min(1.0, max(0.0, foot_x / width)),
                    "y_norm": min(1.0, max(0.0, foot_y / height)),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(confidence),
                    "box": box,
                }
                detections.append(det)
                csv_writer.writerow({key: det[key] for key in fieldnames})

            players = [d for d in detections if d["class_name"] in {"player", "goalkeeper"}]
            balls = [d for d in detections if d["class_name"] == "ball"]
            holder = ball_holder(players, balls)

            for det in detections:
                color = CLASS_COLORS.get(det["class_name"], (255, 255, 255))
                draw_ellipse(frame, det["box"], color)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                draw_label(frame, det["box"], label, color)

            for ball in balls:
                draw_marker(frame, box_top_center(ball["box"]), (0, 255, 0))
            if holder:
                draw_marker(frame, box_top_center(holder["box"]), (0, 0, 255))

            cv2.putText(
                frame,
                f"players: {sum(d['class_name'] == 'player' for d in detections)}  ball: {len(balls)}  refs: {sum(d['class_name'] == 'referee' for d in detections)}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
            written += 1
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote overlay video: {output_path}")
    print(f"Wrote detections CSV: {csv_path}")


if __name__ == "__main__":
    main()
