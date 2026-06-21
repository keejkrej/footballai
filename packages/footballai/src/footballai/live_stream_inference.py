#!/usr/bin/env python3
"""Run live football inference from a file, webcam, HLS, RTMP, or HTTP stream.

This writes a rolling JSON snapshot for the SvelteKit frontend. It supports two
backends:

1. The original Roboflow YOLOv5 tutorial model (requires external/yolov5 + .pt).
2. The Roboflow `sports` YOLOv8 stack: player/pitch/ball detection + ByteTrack
   + team classification + pitch homography.

The JSON snapshot is kept backward-compatible with the existing dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from footballai._paths import REPO_ROOT


try:
    from footballai.football_yolov5_overlay import (
        CLASS_COLORS,
        ball_holder,
        box_top_center,
        draw_ellipse,
        draw_label,
        draw_marker,
        load_model,
    )
except Exception as exc:  # pragma: no cover - YOLOv5 repo may be absent
    CLASS_COLORS = {}

    def ball_holder(players, balls):
        return None

    def box_top_center(box):
        return (0, 0)

    def draw_ellipse(frame, box, color, thickness=3):
        return frame

    def draw_label(frame, box, text, color):
        return frame

    def draw_marker(frame, point, color):
        return frame

    def load_model(yolov5_repo, weights, conf):
        raise ImportError(f"YOLOv5 repo not available: {exc}")


BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
CONFIG = SoccerPitchConfiguration()

TEAM_HEX_COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
TEAM_SV_COLORS = [sv.Color.from_hex(c) for c in TEAM_HEX_COLORS]
SPORTS_ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_HEX_COLORS), thickness=2
)
SPORTS_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_HEX_COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)
SPORTS_BALL_ANNOTATOR = BallAnnotator(radius=6, buffer_size=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index, file path, HLS .m3u8, RTMP, HTTP URL, or capture device",
    )
    parser.add_argument("--backend", choices=["yolov5", "sports"], default="sports", help="Inference backend")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models/football_yolov5_best.pt"),
        help="YOLOv5 custom weights",
    )
    parser.add_argument(
        "--yolov5-repo",
        default=str(REPO_ROOT / "external/yolov5"),
        help="Local YOLOv5 repository path",
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="Directory containing YOLOv8 .pt weights for sports backend",
    )
    parser.add_argument(
        "--state",
        default=str(REPO_ROOT / "data/live/latest.json"),
        help="Rolling JSON state output",
    )
    parser.add_argument(
        "--overlay-output",
        default="",
        help="Optional MP4 overlay recording path",
    )
    parser.add_argument(
        "--detections-csv",
        default="",
        help="Optional rolling CSV output for sports backend",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--img-size", type=int, default=1280, help="Inference image size")
    parser.add_argument("--stride", type=int, default=5, help="Run inference every Nth source frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N source frames; 0 means unlimited")
    parser.add_argument("--window", type=int, default=60, help="Number of inference snapshots for rolling metrics")
    parser.add_argument("--device", default="cuda", help="Torch device for sports backend")
    parser.add_argument("--skip-team-fit", action="store_true", help="Skip sports team classification")
    parser.add_argument("--team-sample-stride", type=int, default=60, help="Stride for collecting sports team crops")
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


def pitch_side_pressure(detections: list[dict]) -> dict:
    """Terrory pressure computed from calibrated pitch x (cm)."""
    players = [det for det in detections if det["class_name"] in {"player", "goalkeeper"} and det.get("pitch_x_cm") is not None]
    if not players:
        return {"pressure_side": "balanced", "pressure_score": 0, "pitch_territory_delta": 0}

    left = sum(1 for det in players if det["pitch_x_cm"] < CONFIG.length * 0.33)
    right = sum(1 for det in players if det["pitch_x_cm"] > CONFIG.length * 0.66)
    advanced_left = sum(1 for det in players if det["pitch_x_cm"] < CONFIG.length * 0.25)
    advanced_right = sum(1 for det in players if det["pitch_x_cm"] > CONFIG.length * 0.75)
    delta = right - left
    side = "right" if advanced_right > advanced_left else "left" if advanced_left > advanced_right else "balanced"
    score = min(100, abs(delta) * 8 + max(advanced_left, advanced_right) * 6)
    return {"pressure_side": side, "pressure_score": score, "pitch_territory_delta": delta}


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


def render_yolov5_overlay(frame: np.ndarray, detections: list[dict], possession: dict | None, snapshot: dict) -> np.ndarray:
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


class SportsBackend:
    """Encapsulates the Roboflow sports YOLOv8 inference pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.models_dir = Path(args.models_dir)
        self.device = args.device
        self.conf = args.conf
        self.img_size = args.img_size

        self.player_model = YOLO(str(self.models_dir / "football-player-detection.pt")).to(device=self.device)
        self.pitch_model = YOLO(str(self.models_dir / "football-pitch-detection.pt")).to(device=self.device)
        self.ball_model = YOLO(str(self.models_dir / "football-ball-detection.pt")).to(device=self.device)
        self.player_model.conf = self.conf
        self.ball_model.conf = self.conf

        self.byte_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        self.ball_tracker = BallTracker(buffer_size=20)
        self.ball_slicer = self._build_ball_slicer()
        self.team_classifier: TeamClassifier | None = None

        self.team_fit_done = False

    def _build_ball_slicer(self) -> sv.InferenceSlicer:
        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = self.ball_model(image_slice, imgsz=640, verbose=False, device=self.device)[0]
            return sv.Detections.from_ultralytics(result)

        return sv.InferenceSlicer(
            callback=callback,
            overlap_filter=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
        )

    def fit_team_classifier(self, video_path: str, max_frames: int = 0) -> None:
        if self.args.skip_team_fit or self.team_classifier is not None:
            return

        cap = cv2.VideoCapture(video_path)
        crops: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break
            if frame_idx % self.args.team_sample_stride == 0:
                result = self.player_model(frame, imgsz=self.img_size, verbose=False, device=self.device)[0]
                detections = sv.Detections.from_ultralytics(result)
                players = detections[detections.class_id == PLAYER_CLASS_ID]
                crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            frame_idx += 1
        cap.release()

        if len(crops) >= 4:
            self.team_classifier = TeamClassifier(device=self.device)
            self.team_classifier.fit(crops)
        self.team_fit_done = True

    def _resolve_goalkeepers_team_id(self, players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections) -> np.ndarray:
        if len(players) == 0 or len(goalkeepers) == 0:
            return np.array([], dtype=int)
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
        ids = []
        for gk_xy in goalkeepers_xy:
            ids.append(0 if np.linalg.norm(gk_xy - team_0_centroid) < np.linalg.norm(gk_xy - team_1_centroid) else 1)
        return np.array(ids, dtype=int)

    def process_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> tuple[list[dict], np.ndarray, dict]:
        height, width = frame.shape[:2]

        pitch_result = self.pitch_model(frame, verbose=False, device=self.device)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

        player_result = self.player_model(frame, imgsz=self.img_size, verbose=False, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = self.byte_tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        color_lookup_parts: list[list[int]] = []
        if self.team_classifier is not None and len(players):
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            players_team_id = self.team_classifier.predict(player_crops)
            goalkeepers_team_id = self._resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            color_lookup_parts = [players_team_id.tolist(), goalkeepers_team_id.tolist(), [REFEREE_CLASS_ID] * len(referees)]
        else:
            color_lookup_parts = [[PLAYER_CLASS_ID] * len(players), [GOALKEEPER_CLASS_ID] * len(goalkeepers), [REFEREE_CLASS_ID] * len(referees)]

        merged = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array([idx for part in color_lookup_parts for idx in part], dtype=int)

        ball_dets = self.ball_slicer(frame).with_nms(threshold=0.1)
        ball_dets = self.ball_tracker.update(ball_dets)

        annotated = frame.copy()
        if len(merged):
            annotated = SPORTS_ELLIPSE_ANNOTATOR.annotate(annotated, merged, custom_color_lookup=color_lookup)
            labels = [str(tid) for tid in merged.tracker_id]
            annotated = SPORTS_LABEL_ANNOTATOR.annotate(annotated, merged, labels=labels, custom_color_lookup=color_lookup)
        if len(ball_dets):
            annotated = SPORTS_BALL_ANNOTATOR.annotate(annotated, ball_dets)

        # Pitch transformer for calibrated records
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer: ViewTransformer | None = None
        if mask.sum() >= 4:
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32),
            )

        merged_xy = merged.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) if len(merged) else np.zeros((0, 2))
        pitch_xy = transformer.transform_points(merged_xy.astype(np.float32)) if transformer is not None and len(merged_xy) else np.full_like(merged_xy, np.nan)

        class_names = {BALL_CLASS_ID: "ball", GOALKEEPER_CLASS_ID: "goalkeeper", PLAYER_CLASS_ID: "player", REFEREE_CLASS_ID: "referee"}
        records: list[dict] = []
        for i in range(len(merged)):
            x1, y1, x2, y2 = merged.xyxy[i]
            class_id = int(merged.class_id[i])
            team_id = int(color_lookup[i]) if class_id in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID} else -1
            records.append(
                {
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps if fps else 0,
                    "track_id": int(merged.tracker_id[i]) if merged.tracker_id is not None else -1,
                    "class_name": class_names.get(class_id, "unknown"),
                    "team_id": team_id,
                    "x_norm": min(1.0, max(0.0, float(x1 + (x2 - x1) / 2) / width)),
                    "y_norm": min(1.0, max(0.0, float(y2) / height)),
                    "pitch_x_cm": float(pitch_xy[i, 0]) if not np.isnan(pitch_xy[i, 0]) else None,
                    "pitch_y_cm": float(pitch_xy[i, 1]) if not np.isnan(pitch_xy[i, 1]) else None,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(merged.confidence[i]),
                }
            )

        ball_records = []
        if len(ball_dets):
            ball_xy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            ball_pitch = transformer.transform_points(ball_xy.astype(np.float32)) if transformer is not None else np.full_like(ball_xy, np.nan)
            for i in range(len(ball_dets)):
                x1, y1, x2, y2 = ball_dets.xyxy[i]
                ball_records.append(
                    {
                        "frame": frame_idx,
                        "time_sec": frame_idx / fps if fps else 0,
                        "track_id": -1,
                        "class_name": "ball",
                        "team_id": -1,
                        "x_norm": min(1.0, max(0.0, float((x1 + x2) / 2) / width)),
                        "y_norm": min(1.0, max(0.0, float(y2) / height)),
                        "pitch_x_cm": float(ball_pitch[i, 0]) if not np.isnan(ball_pitch[i, 0]) else None,
                        "pitch_y_cm": float(ball_pitch[i, 1]) if not np.isnan(ball_pitch[i, 1]) else None,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(ball_dets.confidence[i]) if ball_dets.confidence is not None else None,
                    }
                )

        all_records = records + ball_records

        # Possession
        possession: dict | None = None
        if len(ball_dets) and len(players):
            ball_xy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[0]
            players_xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            distances = np.linalg.norm(players_xy - ball_xy, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 80:
                nearest_player = records[nearest] if nearest < len(records) else None
                if nearest_player:
                    possession = {"class_name": nearest_player["class_name"], "team_id": nearest_player["team_id"]}
                    holder_xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[nearest]
                    cv2.circle(annotated, (int(holder_xy[0]), int(holder_xy[1])), 8, (0, 0, 255), 2)

        # Pressure in pitch coordinates when available, fall back to frame coords
        pressure = pitch_side_pressure(all_records) if any(r.get("pitch_x_cm") is not None for r in all_records) else side_pressure(all_records)

        # Headline
        counts = Counter(r["class_name"] for r in all_records)
        headline = (
            f"players: {counts.get('player', 0)}  gk: {counts.get('goalkeeper', 0)}  "
            f"ref: {counts.get('referee', 0)}  ball: {counts.get('ball', 0)} | "
            f"{pressure['pressure_side']} {pressure['pressure_score']}"
        )
        cv2.putText(annotated, headline, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return all_records, annotated, {"pressure": pressure, "possession": possession, "counts": dict(counts)}


def main() -> None:
    args = parse_args()
    state_path = Path(args.state)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    backend: SportsBackend | None = None
    yolov5_model = None
    writer = None
    csv_writer = None

    if args.backend == "sports":
        backend = SportsBackend(args)
        if args.detections_csv:
            csv_path = Path(args.detections_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="")
            fieldnames = [
                "frame",
                "time_sec",
                "track_id",
                "class_name",
                "team_id",
                "x_norm",
                "y_norm",
                "pitch_x_cm",
                "pitch_y_cm",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
        # Fit team classifier on a finite preview of the source if it's a file/stream.
        backend.fit_team_classifier(str(args.source), max_frames=300)
    else:
        yolov5_model = load_model(args.yolov5_repo, args.weights, args.conf)

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

        if backend is not None:
            records, annotated, meta = backend.process_frame(frame, frame_idx, fps)
            classes = Counter(r["class_name"] for r in records)
            pressure = meta["pressure"]
            possession = meta["possession"]
        else:
            results = yolov5_model(frame, size=args.img_size)
            pred = results.pred[0].detach().cpu().numpy()
            records = detections_from_prediction(pred, yolov5_model.names, width, height, frame_idx, fps)
            classes = Counter(det["class_name"] for det in records)
            players = [det for det in records if det["class_name"] in {"player", "goalkeeper"}]
            balls = [det for det in records if det["class_name"] == "ball"]
            possession_det = ball_holder(players, balls)
            possession = {"class_name": possession_det["class_name"], "team_id": -1} if possession_det else None
            pressure = side_pressure(records)
            annotated = render_yolov5_overlay(frame, records, possession_det, {})

        current = {
            "status": "running",
            "source": args.source,
            "backend": args.backend,
            "updated_at": time.time(),
            "frame": frame_idx,
            "fps": fps,
            "latency_ms": round((time.time() - started) * 1000, 1),
            "classes": dict(classes),
            "detections": len(records),
            "possession": possession["class_name"] if possession else None,
            "possession_team": possession.get("team_id") if possession else None,
            "pressure": pressure,
        }
        history.append(current)
        snapshot = {**current, "trading": trading_snapshot(history)}
        last_snapshot = snapshot
        atomic_write_json(state_path, snapshot)

        if csv_writer:
            for record in records:
                csv_writer.writerow(record)

        if writer:
            writer.write(annotated)

        print(json.dumps({k: snapshot[k] for k in ["frame", "classes", "pressure", "trading", "possession_team"]}), flush=True)
        frame_idx += 1

    if last_snapshot:
        atomic_write_json(state_path, {**last_snapshot, "status": "stopped"})
    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
