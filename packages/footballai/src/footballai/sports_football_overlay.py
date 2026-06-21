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

The public entry point is :func:`run_full`; see also :class:`SportsProcessor` for
single-frame processing used by the live WebSocket server.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import TeamClassifier, create_batches
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


def _model_path(models_dir: Path, name: str) -> Path:
    return models_dir / name


def _get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def _resolve_goalkeepers_team_id(
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


def _render_radar(
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


def _detect_ball(
    frame: np.ndarray,
    slicer: sv.InferenceSlicer,
    tracker: BallTracker,
) -> sv.Detections:
    detections = slicer(frame).with_nms(threshold=0.1)
    detections = tracker.update(detections)
    return detections


def _extract_siglip_features(
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
        features = _extract_siglip_features(
            padded_crops, self.processor, self.siglip_model, self.device, self.siglip_batch_size
        )
        projections = self.reducer.transform(features)
        labels = self.cluster_model.predict(projections)
        return labels[: len(crops)]


def _build_team_classifier(
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
            all_crops += _get_crops(frame, players)
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()

    if len(all_crops) < 4:
        raise ValueError(f"Not enough player crops ({len(all_crops)}) to fit team classifier")

    print(f"Fitting team classifier on {len(all_crops)} crops...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)

    features = _extract_siglip_features(all_crops, processor, siglip_model, device, siglip_batch_size)
    reducer = umap.UMAP(n_components=3)
    projections = reducer.fit_transform(features)
    cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)
    cluster_model.fit(projections)

    return TeamClassifierState(processor, siglip_model, reducer, cluster_model, device, siglip_batch_size)


def _detections_to_records(
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


def _nearest_player_to_ball(players: sv.Detections, ball_xy: np.ndarray) -> sv.Detections | None:
    if len(players) == 0:
        return None
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    distances = np.linalg.norm(players_xy - ball_xy, axis=1)
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] > 80:
        return None
    return players[nearest_idx : nearest_idx + 1]


class SportsProcessor:
    """Encapsulates the Roboflow sports YOLOv8 inference pipeline for a single source.

    Used by both the full-file overlay renderer and the live WebSocket server.
    """

    def __init__(
        self,
        models_dir: Path | str = REPO_ROOT / "models",
        device: str = "cuda",
        conf: float = 0.25,
        img_size: int = 1280,
        skip_team_fit: bool = False,
        team_sample_stride: int = STRIDE,
        siglip_batch_size: int = 64,
    ):
        self.models_dir = Path(models_dir)
        self.device = device
        self.conf = conf
        self.img_size = img_size
        self.skip_team_fit = skip_team_fit
        self.team_sample_stride = team_sample_stride
        self.siglip_batch_size = siglip_batch_size

        self.player_model: YOLO | None = None
        self.pitch_model: YOLO | None = None
        self.ball_model: YOLO | None = None
        self.byte_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        self.ball_tracker = BallTracker(buffer_size=20)
        self.ball_slicer: sv.InferenceSlicer | None = None
        self.team_classifier: TeamClassifier | None = None
        self._team_classifier_state: TeamClassifierState | None = None
        self._team_fit_done = False

    def load_models(self) -> None:
        if self.player_model is not None:
            return
        player_weights = _model_path(self.models_dir, "football-player-detection.pt")
        pitch_weights = _model_path(self.models_dir, "football-pitch-detection.pt")
        ball_weights = _model_path(self.models_dir, "football-ball-detection.pt")
        self.player_model = YOLO(str(player_weights)).to(device=self.device)
        self.pitch_model = YOLO(str(pitch_weights)).to(device=self.device)
        self.ball_model = YOLO(str(ball_weights)).to(device=self.device)
        self.player_model.conf = self.conf
        self.ball_model.conf = self.conf

        def ball_callback(image_slice: np.ndarray) -> sv.Detections:
            result = self.ball_model(image_slice, imgsz=640, verbose=False, device=self.device)[0]  # type: ignore[union-attr]
            return sv.Detections.from_ultralytics(result)

        self.ball_slicer = sv.InferenceSlicer(
            callback=ball_callback,
            overlap_filter=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
            overlap_wh=(0, 0),
        )

    def fit_team_classifier_from_video(self, video_path: str, max_frames: int = 0) -> None:
        if self.skip_team_fit or self._team_fit_done or self.player_model is None:
            return
        try:
            self._team_classifier_state = _build_team_classifier(
                video_path,
                self.player_model,
                self.device,
                self.team_sample_stride,
                self.img_size,
                max_frames,
                self.siglip_batch_size,
            )
        except Exception as exc:
            print(f"Team classification failed: {exc}; continuing without team colors")
        self._team_fit_done = True

    def fit_team_classifier_from_crops(self, crops: list[np.ndarray]) -> None:
        """Fit the classifier lazily from a batch of player crops."""
        if self.skip_team_fit or self._team_fit_done or len(crops) < 4:
            return
        try:
            self.team_classifier = TeamClassifier(device=self.device)
            self.team_classifier.fit(crops)
            self._team_fit_done = True
        except Exception as exc:
            print(f"Lazy team classification failed: {exc}; continuing without team colors")

    def _resolve_goalkeepers_team_id(
        self, players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections
    ) -> np.ndarray:
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
        if self.player_model is None or self.pitch_model is None or self.ball_model is None or self.ball_slicer is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

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
            player_crops = _get_crops(frame, players)
            players_team_id = self.team_classifier.predict(player_crops)
            goalkeepers_team_id = self._resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            color_lookup_parts = [players_team_id.tolist(), goalkeepers_team_id.tolist(), [REFEREE_CLASS_ID] * len(referees)]
        elif self._team_classifier_state is not None and len(players):
            player_crops = _get_crops(frame, players)
            players_team_id = self._team_classifier_state.predict(player_crops)
            goalkeepers_team_id = self._resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            color_lookup_parts = [players_team_id.tolist(), goalkeepers_team_id.tolist(), [REFEREE_CLASS_ID] * len(referees)]
        else:
            color_lookup_parts = [[PLAYER_CLASS_ID] * len(players), [GOALKEEPER_CLASS_ID] * len(goalkeepers), [REFEREE_CLASS_ID] * len(referees)]

        merged = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array([idx for part in color_lookup_parts for idx in part], dtype=int)

        ball_dets = _detect_ball(frame, self.ball_slicer, self.ball_tracker)

        annotated = frame.copy()
        if len(merged):
            annotated = ELLIPSE_ANNOTATOR.annotate(annotated, merged, custom_color_lookup=color_lookup)
            labels = [str(tid) for tid in merged.tracker_id]
            annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated, merged, labels=labels, custom_color_lookup=color_lookup
            )
        if len(ball_dets):
            annotated = BALL_ANNOTATOR.annotate(annotated, ball_dets)

        # Possession marker
        ball_xy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) if len(ball_dets) else None
        possession: dict | None = None
        if ball_xy is not None and len(players):
            holder = _nearest_player_to_ball(players, ball_xy[0])
            if holder is not None:
                hx, hy = holder.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                cv2.circle(annotated, (int(hx), int(hy)), 8, (0, 0, 255), 2)
                holder_record = next(
                    (r for r in _detections_to_records(merged, keypoints, color_lookup, None, frame_idx, fps, width, height)
                     if int(r["track_id"]) == int(holder.tracker_id[0])), None
                ) if merged.tracker_id is not None else None
                if holder_record:
                    possession = {"class_name": holder_record["class_name"], "team_id": holder_record["team_id"]}

        # Radar overlay
        try:
            radar = _render_radar(merged, keypoints, color_lookup, ball_xy)
            radar = sv.resize_image(radar, (width // 3, height // 3))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(x=16, y=height - radar_h - 16, width=radar_w, height=radar_h)
            annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
        except ValueError:
            pass

        # Headline
        counts = {
            "player": int((merged.class_id == PLAYER_CLASS_ID).sum()),
            "goalkeeper": int((merged.class_id == GOALKEEPER_CLASS_ID).sum()),
            "referee": int((merged.class_id == REFEREE_CLASS_ID).sum()),
            "ball": len(ball_dets),
        }
        headline = (
            f"players: {counts['player']}  gk: {counts['goalkeeper']}  "
            f"ref: {counts['referee']}  ball: {counts['ball']}"
        )
        cv2.putText(annotated, headline, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        records = _detections_to_records(
            merged, keypoints, color_lookup, ball_dets, frame_idx, fps, width, height
        )

        return records, annotated, {"counts": counts, "possession": possession}


def _default_progress(progress: dict) -> None:
    print(json.dumps({"type": "progress", **progress}), flush=True)


def run_full(
    input_path: Path | str,
    output_path: Path | str,
    csv_path: Path | str,
    *,
    models_dir: Path | str = REPO_ROOT / "models",
    device: str = "cuda",
    conf: float = 0.25,
    img_size: int = 1280,
    max_frames: int = 0,
    stride: int = 1,
    skip_team_fit: bool = False,
    team_sample_stride: int = STRIDE,
    siglip_batch_size: int = 64,
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Render the full sports overlay for a local MP4 file and write a CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    emit = on_progress or _default_progress

    processor = SportsProcessor(
        models_dir=models_dir,
        device=device,
        conf=conf,
        img_size=img_size,
        skip_team_fit=skip_team_fit,
        team_sample_stride=team_sample_stride,
        siglip_batch_size=siglip_batch_size,
    )
    processor.load_models()
    if not skip_team_fit:
        processor.fit_team_classifier_from_video(str(input_path), max_frames=300)

    video_info = sv.VideoInfo.from_video_path(str(input_path))
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or video_info.fps or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or video_info.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or video_info.height
    out_fps = max(1.0, fps / max(1, stride))

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
        processed = 0
        total = max_frames if max_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        pbar = tqdm(total=total if total > 0 else None, desc="overlay")

        while max_frames <= 0 or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            records, annotated, meta = processor.process_frame(frame, frame_idx, fps)
            for record in records:
                csv_writer.writerow(record)

            # Simple possession buffer for progress metadata
            if meta["possession"]:
                possession_buffer.append(meta["possession"])

            writer.write(annotated)
            processed += 1

            if processed % 10 == 0:
                emit(
                    {
                        "stage": "inference",
                        "frame": frame_idx,
                        "processed": processed,
                        "total": total if total > 0 else None,
                        "classes": meta["counts"],
                    }
                )

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    emit({"stage": "done", "output": str(output_path), "csv": str(csv_path), "processed": processed})
    print(f"Wrote overlay video: {output_path}")
    print(f"Wrote detections CSV: {csv_path}")
