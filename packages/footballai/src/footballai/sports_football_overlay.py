#!/usr/bin/env python3
"""Render a football overlay using the Roboflow `sports` soccer utilities.

Combines:
- YOLOv8 player detection + ByteTrack (player, goalkeeper, referee, ball)
- YOLOv8 pitch keypoint detection and ViewTransformer homography
- SigLIP/UMAP/KMeans team classification (batched for speed)
- YOLOv8 ball detection with InferenceSlicer + BallTracker
- Radar/minimap projection onto a real-world soccer pitch

Outputs:
- Annotated video
- CSV of detections with broadcast-frame and pitch coordinates, team IDs, track IDs
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import create_batches
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from footballai._paths import REPO_ROOT


BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
TEAM_COLORS = [sv.Color.from_hex(c) for c in COLORS]

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)
BALL_ANNOTATOR = BallAnnotator(radius=6, buffer_size=10)

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=str(REPO_ROOT / "data/raw/youtube_clip.mp4"),
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/outputs/sports_overlay.mp4"),
        help="Output overlay video",
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data/outputs/sports_positions.csv"),
        help="Output detections CSV",
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="Directory containing YOLOv8 .pt weights",
    )
    parser.add_argument("--device", default="cuda", help="Torch device: cpu, cuda, mps")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--img-size", type=int, default=1280, help="Player/pitch inference size")
    parser.add_argument("--max-frames", type=int, default=900, help="Maximum source frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth source frame")
    parser.add_argument("--skip-team-fit", action="store_true", help="Skip team classifier training (faster, no team colors)")
    parser.add_argument("--team-sample-stride", type=int, default=STRIDE, help="Frame stride for collecting team training crops")
    parser.add_argument("--siglip-batch-size", type=int, default=64, help="Batch size for SigLIP feature extraction")
    return parser.parse_args()


def model_path(models_dir: Path, name: str) -> Path:
    return models_dir / name


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections
) -> np.ndarray:
    """Assign each goalkeeper to the nearest player-team centroid."""
    if len(players) == 0 or len(goalkeepers) == 0:
        return np.array([], dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id, dtype=int)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_xy: np.ndarray | None = None,
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if mask.sum() < 4:
        raise ValueError("Not enough valid pitch keypoints to compute homography")

    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    for color_idx, color in enumerate(TEAM_COLORS):
        points = transformed_xy[color_lookup == color_idx]
        if len(points):
            radar = draw_points_on_pitch(
                config=CONFIG, xy=points, face_color=color, radius=20, pitch=radar
            )
    # Referee color index 3
    referee_points = transformed_xy[color_lookup == REFEREE_CLASS_ID]
    if len(referee_points):
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=referee_points,
            face_color=sv.Color.from_hex(COLORS[3]),
            radius=20,
            pitch=radar,
        )
    if ball_xy is not None and len(ball_xy):
        ball_pitch = transformer.transform_points(ball_xy.astype(np.float32))
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=ball_pitch,
            face_color=sv.Color.from_hex("#FFFFFF"),
            edge_color=sv.Color.from_hex("#000000"),
            radius=14,
            pitch=radar,
        )
    return radar


def detect_ball(
    frame: np.ndarray,
    slicer: sv.InferenceSlicer,
    tracker: BallTracker,
) -> sv.Detections:
    detections = slicer(frame).with_nms(threshold=0.1)
    detections = tracker.update(detections)
    return detections


def extract_siglip_features(
    crops: list[np.ndarray],
    processor: AutoProcessor,
    model: SiglipVisionModel,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Extract mean-pooled SigLIP embeddings for a list of player crops."""
    if not crops:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    pillow_crops = [sv.cv2_to_pillow(crop) for crop in crops]
    data: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(create_batches(pillow_crops, batch_size), desc="Embedding extraction", total=max(1, len(pillow_crops) // batch_size)):
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    return np.concatenate(data, axis=0)


class TeamClassifierState:
    """Holds the fitted team-classification artefacts and predicts on new crops."""

    def __init__(
        self,
        processor: AutoProcessor,
        siglip_model: SiglipVisionModel,
        reducer: Any,
        cluster_model: Any,
        device: str,
        siglip_batch_size: int,
    ):
        self.processor = processor
        self.siglip_model = siglip_model
        self.reducer = reducer
        self.cluster_model = cluster_model
        self.device = device
        self.siglip_batch_size = siglip_batch_size

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([], dtype=int)
        # UMAP.transform is very slow for single samples; process at least 4
        # crops together by padding with a blank crop if needed. This keeps
        # per-frame latency reasonable without changing results.
        min_batch = 4
        pad_count = max(0, min_batch - len(crops))
        padded_crops = crops + [np.zeros((40, 40, 3), dtype=np.uint8)] * pad_count
        features = extract_siglip_features(
            padded_crops, self.processor, self.siglip_model, self.device, self.siglip_batch_size
        )
        projections = self.reducer.transform(features)
        labels = self.cluster_model.predict(projections)
        return labels[: len(crops)]


def build_team_classifier(
    video_path: str,
    player_model: YOLO,
    device: str,
    sample_stride: int,
    img_size: int,
    max_frames: int,
    siglip_batch_size: int,
) -> TeamClassifierState:
    """Fit a team classifier on all player crops and return a reusable predictor.

    This batches both model inference and UMAP/KMeans so it is much faster than
    predicting per-frame.
    """
    from sklearn.cluster import KMeans
    import umap

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_crops: list[np.ndarray] = []

    frame_idx = 0
    pbar = tqdm(desc="collecting team crops")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break
        if frame_idx % sample_stride == 0:
            result = player_model(frame, imgsz=img_size, verbose=False, device=device)[0]
            detections = sv.Detections.from_ultralytics(result)
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            all_crops += get_crops(frame, players)
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    if len(all_crops) < 4:
        raise ValueError(f"Not enough player crops ({len(all_crops)}) to fit team classifier")

    print(f"Fitting team classifier on {len(all_crops)} crops...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)

    features = extract_siglip_features(all_crops, processor, siglip_model, device, siglip_batch_size)
    reducer = umap.UMAP(n_components=3)
    projections = reducer.fit_transform(features)
    cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)
    cluster_model.fit(projections)

    return TeamClassifierState(processor, siglip_model, reducer, cluster_model, device, siglip_batch_size)


def detections_to_records(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_detections: sv.Detections | None,
    frame_idx: int,
    fps: float,
    width: int,
    height: int,
) -> list[dict]:
    """Convert frame detections to CSV rows."""
    records: list[dict] = []

    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = None
    if mask.sum() >= 4:
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        )

    class_names = {BALL_CLASS_ID: "ball", GOALKEEPER_CLASS_ID: "goalkeeper", PLAYER_CLASS_ID: "player", REFEREE_CLASS_ID: "referee"}
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    pitch_xy = transformer.transform_points(xy.astype(np.float32)) if transformer is not None else np.full_like(xy, np.nan)

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        class_id = int(detections.class_id[i])
        team_id = int(color_lookup[i]) if class_id == PLAYER_CLASS_ID or class_id == GOALKEEPER_CLASS_ID else -1
        record = {
            "frame": frame_idx,
            "time_sec": frame_idx / fps if fps else 0.0,
            "track_id": int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
            "class_name": class_names.get(class_id, "unknown"),
            "team_id": team_id,
            "x_norm": float(x1 + (x2 - x1) / 2) / width,
            "y_norm": float(y2) / height,
            "pitch_x_cm": float(pitch_xy[i, 0]) if not np.isnan(pitch_xy[i, 0]) else None,
            "pitch_y_cm": float(pitch_xy[i, 1]) if not np.isnan(pitch_xy[i, 1]) else None,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(detections.confidence[i]),
        }
        records.append(record)

    if ball_detections is not None and len(ball_detections):
        ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        for i in range(len(ball_detections)):
            x1, y1, x2, y2 = ball_detections.xyxy[i]
            conf = float(ball_detections.confidence[i]) if ball_detections.confidence is not None else None
            pitch_ball = transformer.transform_points(ball_xy[i : i + 1].astype(np.float32)) if transformer is not None else np.full((1, 2), np.nan)
            records.append(
                {
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps if fps else 0.0,
                    "track_id": -1,
                    "class_name": "ball",
                    "team_id": -1,
                    "x_norm": float((x1 + x2) / 2) / width,
                    "y_norm": float(y2) / height,
                    "pitch_x_cm": float(pitch_ball[0, 0]) if not np.isnan(pitch_ball[0, 0]) else None,
                    "pitch_y_cm": float(pitch_ball[0, 1]) if not np.isnan(pitch_ball[0, 1]) else None,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": conf,
                }
            )

    return records


def run_overlay(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    models_dir = Path(args.models_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    player_model = YOLO(str(model_path(models_dir, "football-player-detection.pt"))).to(device=args.device)
    pitch_model = YOLO(str(model_path(models_dir, "football-pitch-detection.pt"))).to(device=args.device)
    ball_model = YOLO(str(model_path(models_dir, "football-ball-detection.pt"))).to(device=args.device)

    player_model.conf = args.conf
    ball_model.conf = args.conf

    team_classifier: TeamClassifierState | None = None
    if not args.skip_team_fit:
        try:
            team_classifier = build_team_classifier(
                str(video_path),
                player_model,
                args.device,
                args.team_sample_stride,
                args.img_size,
                args.max_frames,
                args.siglip_batch_size,
            )
        except Exception as exc:
            print(f"Team classification failed: {exc}; continuing without team colors")
            team_classifier = None

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_model(image_slice, imgsz=640, verbose=False, device=args.device)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
        overlap_wh=(0, 0),
    )
    ball_tracker = BallTracker(buffer_size=20)
    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or video_info.fps or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or video_info.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or video_info.height
    out_fps = max(1.0, fps / max(1, args.stride))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

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

    possession_buffer: deque[dict] = deque(maxlen=5)

    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        frame_idx = 0
        pbar = tqdm(total=args.max_frames if args.max_frames > 0 else None, desc="overlay")

        while args.max_frames <= 0 or frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            annotated = frame.copy()

            # Pitch keypoints
            pitch_result = pitch_model(frame, verbose=False, device=args.device)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            # Player detections + tracking
            player_result = player_model(frame, imgsz=args.img_size, verbose=False, device=args.device)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = byte_tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            merged = sv.Detections.merge([players, goalkeepers, referees])

            color_lookup_parts: list[list[int]] = []
            if team_classifier is not None and len(players):
                player_crops = get_crops(frame, players)
                players_team_id = team_classifier.predict(player_crops)
                goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
                color_lookup_parts = [players_team_id.tolist(), goalkeepers_team_id.tolist(), [REFEREE_CLASS_ID] * len(referees)]
            else:
                color_lookup_parts = [
                    [PLAYER_CLASS_ID] * len(players),
                    [GOALKEEPER_CLASS_ID] * len(goalkeepers),
                    [REFEREE_CLASS_ID] * len(referees),
                ]

            color_lookup = np.array([idx for part in color_lookup_parts for idx in part], dtype=int)

            # Ball detection
            ball_detections = detect_ball(frame, ball_slicer, ball_tracker)
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) if len(ball_detections) else None

            # Annotate
            if len(merged):
                annotated = ELLIPSE_ANNOTATOR.annotate(annotated, merged, custom_color_lookup=color_lookup)
                labels = [str(tid) for tid in merged.tracker_id]
                annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
                    annotated, merged, labels=labels, custom_color_lookup=color_lookup
                )
            if len(ball_detections):
                annotated = BALL_ANNOTATOR.annotate(annotated, ball_detections)

            # Possession marker
            if ball_xy is not None and len(players):
                holder = _nearest_player_to_ball(players, ball_xy[0])
                if holder is not None:
                    hx, hy = holder.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                    cv2.circle(annotated, (int(hx), int(hy)), 8, (0, 0, 255), 2)

            # Radar overlay
            try:
                radar = render_radar(merged, keypoints, color_lookup, ball_xy)
                radar = sv.resize_image(radar, (width // 3, height // 3))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(x=16, y=height - radar_h - 16, width=radar_w, height=radar_h)
                annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
            except ValueError as exc:
                pbar.set_postfix({"radar": str(exc)})

            # Headline
            counts = {
                "player": int((merged.class_id == PLAYER_CLASS_ID).sum()),
                "goalkeeper": int((merged.class_id == GOALKEEPER_CLASS_ID).sum()),
                "referee": int((merged.class_id == REFEREE_CLASS_ID).sum()),
                "ball": len(ball_detections),
            }
            headline = (
                f"players: {counts['player']}  gk: {counts['goalkeeper']}  "
                f"ref: {counts['referee']}  ball: {counts['ball']}"
            )
            cv2.putText(annotated, headline, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(annotated)

            records = detections_to_records(
                merged, keypoints, color_lookup, ball_detections, frame_idx, fps, width, height
            )
            for record in records:
                csv_writer.writerow(record)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()
    print(f"Wrote overlay video: {output_path}")
    print(f"Wrote detections CSV: {csv_path}")


def _nearest_player_to_ball(players: sv.Detections, ball_xy: np.ndarray) -> sv.Detections | None:
    if len(players) == 0:
        return None
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    distances = np.linalg.norm(players_xy - ball_xy, axis=1)
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] > 80:
        return None
    return players[nearest_idx : nearest_idx + 1]


def main() -> None:
    run_overlay(parse_args())


if __name__ == "__main__":
    main()
