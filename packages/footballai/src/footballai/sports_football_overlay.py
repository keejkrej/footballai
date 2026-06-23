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

Performance optimizations
-----------------------
- Frame decoding runs in a dedicated thread so the GPU does not wait on OpenCV.
- Player/pitch models process a batch of frames per forward pass (``--batch-size``).
- Team classifier is cached to disk after fitting so repeated runs skip the CPU
  warm-up.
- Video writing and CSV serialization run in background writer threads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import queue
import threading
import time
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
        min_batch = 4
        pad_count = max(0, min_batch - len(crops))
        padded_crops = crops + [np.zeros((40, 40, 3), dtype=np.uint8)] * pad_count
        features = _extract_siglip_features(
            padded_crops, self.processor, self.siglip_model, self.device, self.siglip_batch_size
        )
        projections = self.reducer.transform(features)
        labels = self.cluster_model.predict(projections)
        return labels[: len(crops)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "processor": self.processor,
                    "siglip_model": self.siglip_model,
                    "reducer": self.reducer,
                    "cluster_model": self.cluster_model,
                    "device": self.device,
                    "siglip_batch_size": self.siglip_batch_size,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TeamClassifierState":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            processor=data["processor"],
            siglip_model=data["siglip_model"],
            reducer=data["reducer"],
            cluster_model=data["cluster_model"],
            device=data["device"],
            siglip_batch_size=data["siglip_batch_size"],
        )


def _build_team_classifier(
    video_path: str,
    player_model: YOLO,
    device: str,
    sample_stride: int,
    img_size: int,
    max_frames: int,
    siglip_batch_size: int,
) -> TeamClassifierState:
    """Fit a team classifier on player crops and return a reusable predictor."""
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
    """Encapsulates the Roboflow sports YOLOv8 inference pipeline for a single source."""

    def __init__(
        self,
        models_dir: Path | str = REPO_ROOT / "models",
        device: str = "cuda",
        conf: float = 0.25,
        img_size: int = 1280,
        team_sample_stride: int = STRIDE,
        siglip_batch_size: int = 64,
    ):
        self.models_dir = Path(models_dir)
        self.device = device
        self.conf = conf
        self.img_size = img_size
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
        self.team_by_track: dict[int, int] = {}

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
            result = self.ball_model(image_slice, imgsz=640, verbose=False, device=self.device)[0]
            return sv.Detections.from_ultralytics(result)

        self.ball_slicer = sv.InferenceSlicer(
            callback=ball_callback,
            overlap_filter=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
            overlap_wh=(0, 0),
        )

    def fit_team_classifier_from_video(self, video_path: str, max_frames: int = 0) -> None:
        if self._team_fit_done or self.player_model is None:
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
        if self._team_fit_done or len(crops) < 4:
            return
        try:
            self.team_classifier = TeamClassifier(device=self.device)
            self.team_classifier.fit(crops)
            self._team_fit_done = True
        except Exception as exc:
            print(f"Lazy team classification failed: {exc}; continuing without team colors")

    def _classify_crop(self, crop: np.ndarray) -> int:
        """Classify a single player crop using the fitted team classifier."""
        if self.team_classifier is not None:
            return int(self.team_classifier.predict([crop])[0])
        if self._team_classifier_state is not None:
            return int(self._team_classifier_state.predict([crop])[0])
        return PLAYER_CLASS_ID

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

    def _lookup_team_id(self, frame: np.ndarray, players: sv.Detections, goalkeepers: sv.Detections, referees: sv.Detections) -> list[list[int]]:
        if (self.team_classifier is None and self._team_classifier_state is None) or len(players) == 0:
            return [[PLAYER_CLASS_ID] * len(players), [GOALKEEPER_CLASS_ID] * len(goalkeepers), [REFEREE_CLASS_ID] * len(referees)]

        player_crops = _get_crops(frame, players)
        players_team_id: list[int] = []
        for crop, track_id in zip(player_crops, players.tracker_id if players.tracker_id is not None else range(len(player_crops))):
            if track_id in self.team_by_track:
                players_team_id.append(self.team_by_track[track_id])
            else:
                team_id = self._classify_crop(crop)
                self.team_by_track[int(track_id)] = team_id
                players_team_id.append(team_id)
        players_team_id_arr = np.array(players_team_id, dtype=int)
        goalkeepers_team_id = self._resolve_goalkeepers_team_id(players, players_team_id_arr, goalkeepers)
        return [players_team_id, goalkeepers_team_id.tolist(), [REFEREE_CLASS_ID] * len(referees)]

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

        color_lookup_parts = self._lookup_team_id(frame, players, goalkeepers, referees)
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

        ball_xy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) if len(ball_dets) else None
        possession: dict | None = None
        if ball_xy is not None and len(players):
            holder = _nearest_player_to_ball(players, ball_xy[0])
            if holder is not None:
                hx, hy = holder.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                cv2.circle(annotated, (int(hx), int(hy)), 8, (0, 0, 255), 2)
                if merged.tracker_id is not None:
                    holder_record = next(
                        (r for r in _detections_to_records(merged, keypoints, color_lookup, None, frame_idx, fps, width, height)
                         if int(r["track_id"]) == int(holder.tracker_id[0])), None
                    )
                    if holder_record:
                        possession = {"class_name": holder_record["class_name"], "team_id": holder_record["team_id"]}

        try:
            radar = _render_radar(merged, keypoints, color_lookup, ball_xy)
            radar = sv.resize_image(radar, (width // 3, height // 3))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(x=16, y=height - radar_h - 16, width=radar_w, height=radar_h)
            annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
        except ValueError:
            pass

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

    def process_batch(
        self, frames: list[tuple[int, np.ndarray]], fps: float
    ) -> list[tuple[int, list[dict], np.ndarray, dict]]:
        """Run batched player/pitch inference on a list of (frame_idx, frame) pairs.

        Returns one result tuple per input frame in the same order.
        """
        if self.player_model is None or self.pitch_model is None or self.ball_model is None or self.ball_slicer is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if not frames:
            return []

        indices = [idx for idx, _ in frames]
        frame_array = [frame for _, frame in frames]

        # Batched forward passes on GPU
        player_results = self.player_model(frame_array, imgsz=self.img_size, verbose=False, device=self.device)
        pitch_results = self.pitch_model(frame_array, verbose=False, device=self.device)

        results: list[tuple[int, list[dict], np.ndarray, dict]] = []
        for idx, frame, player_result, pitch_result in zip(indices, frame_array, player_results, pitch_results, strict=False):
            height, width = frame.shape[:2]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            detections = sv.Detections.from_ultralytics(player_result)
            detections = self.byte_tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            color_lookup_parts = self._lookup_team_id(frame, players, goalkeepers, referees)
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

            ball_xy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) if len(ball_dets) else None
            possession: dict | None = None
            if ball_xy is not None and len(players):
                holder = _nearest_player_to_ball(players, ball_xy[0])
                if holder is not None:
                    hx, hy = holder.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                    cv2.circle(annotated, (int(hx), int(hy)), 8, (0, 0, 255), 2)
                    if merged.tracker_id is not None:
                        holder_record = next(
                            (r for r in _detections_to_records(merged, keypoints, color_lookup, None, idx, fps, width, height)
                             if int(r["track_id"]) == int(holder.tracker_id[0])), None
                        )
                        if holder_record:
                            possession = {"class_name": holder_record["class_name"], "team_id": holder_record["team_id"]}

            try:
                radar = _render_radar(merged, keypoints, color_lookup, ball_xy)
                radar = sv.resize_image(radar, (width // 3, height // 3))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(x=16, y=height - radar_h - 16, width=radar_w, height=radar_h)
                annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
            except ValueError:
                pass

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
                merged, keypoints, color_lookup, ball_dets, idx, fps, width, height
            )
            results.append((idx, records, annotated, {"counts": counts, "possession": possession}))

        return results


def _default_progress(progress: dict) -> None:
    print(json.dumps({"type": "progress", **progress}), flush=True)


def _team_cache_path(
    video_path: Path, cache_dir: Path, img_size: int, conf: float, team_sample_stride: int
) -> Path:
    key = f"{video_path.resolve()}:{img_size}:{conf}:{team_sample_stride}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"team_classifier_{digest}.pkl"


class _FrameReader(threading.Thread):
    """Decode frames in a background thread and push them into a queue."""

    def __init__(
        self,
        video_path: Path,
        stride: int,
        max_frames: int,
        queue_size: int,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.stride = stride
        self.max_frames = max_frames
        self.frame_queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=queue_size)
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.total = 0
        self.error: Exception | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.error = RuntimeError(f"Could not open video: {self.video_path}")
            self.frame_queue.put(None)
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        frame_idx = 0
        emitted = 0
        while self.max_frames <= 0 or frame_idx < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % self.stride != 0:
                frame_idx += 1
                continue
            self.frame_queue.put((frame_idx, frame))
            emitted += 1
            frame_idx += 1
        cap.release()
        self.frame_queue.put(None)


class _VideoWriterThread(threading.Thread):
    """Consume ordered frames from a queue and write them to a video file."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        queue_size: int,
    ):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=queue_size)
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                self.size,
            )
            buffer: dict[int, np.ndarray] = {}
            next_idx = 0
            while True:
                item = self.queue.get()
                if item is None:
                    break
                idx, frame = item
                buffer[idx] = frame
                while next_idx in buffer:
                    writer.write(buffer.pop(next_idx))
                    next_idx += 1
            writer.release()
        except Exception as exc:
            self.error = exc


class _CsvWriterThread(threading.Thread):
    """Consume ordered records from a queue and write them to a CSV file."""

    def __init__(
        self,
        csv_path: Path,
        fieldnames: list[str],
        queue_size: int,
    ):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.queue: queue.Queue[tuple[int, list[dict]] | None] = queue.Queue(maxsize=queue_size)
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            with self.csv_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()
                buffer: dict[int, list[dict]] = {}
                next_idx = 0
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    idx, records = item
                    buffer[idx] = records
                    while next_idx in buffer:
                        for record in buffer.pop(next_idx):
                            writer.writerow(record)
                        next_idx += 1
        except Exception as exc:
            self.error = exc


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
    batch_size: int = 4,
    team_sample_stride: int = STRIDE,
    siglip_batch_size: int = 64,
    team_cache: bool = True,
    team_cache_dir: Path | str = REPO_ROOT / "data" / "cache" / "team_classifier",
    decoder_queue_size: int = 32,
    writer_queue_size: int = 32,
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Render the full sports overlay for a local MP4 file and write a CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)
    team_cache_dir = Path(team_cache_dir)
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
        team_sample_stride=team_sample_stride,
        siglip_batch_size=siglip_batch_size,
    )
    processor.load_models()

    # Team classifier with disk cache.
    cache_path = _team_cache_path(input_path, team_cache_dir, img_size, conf, team_sample_stride)
    if team_cache and cache_path.exists():
        print(f"Loading cached team classifier from {cache_path}")
        try:
            processor._team_classifier_state = TeamClassifierState.load(cache_path)
            processor._team_fit_done = True
        except Exception as exc:
            print(f"Failed to load team classifier cache: {exc}; refitting...")
            processor._team_fit_done = False
    if not processor._team_fit_done:
        processor.fit_team_classifier_from_video(str(input_path), max_frames=300)
        if team_cache and processor._team_classifier_state is not None:
            processor._team_classifier_state.save(cache_path)

    # Start async frame decoder
    reader = _FrameReader(input_path, stride, max_frames, decoder_queue_size)
    reader.start()
    reader.join(timeout=0.1)  # let it open and set metadata
    while reader.fps == 25.0 and reader.is_alive():
        reader.join(timeout=0.05)
    fps = reader.fps
    width = reader.width or 1280
    height = reader.height or 720
    total = reader.total if max_frames <= 0 else min(max_frames, reader.total or 0)
    out_fps = max(1.0, fps / max(1, stride))

    # Start async writers
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
    video_writer = _VideoWriterThread(output_path, out_fps, (width, height), writer_queue_size)
    csv_writer = _CsvWriterThread(csv_path, fieldnames, writer_queue_size)
    video_writer.start()
    csv_writer.start()

    possession_buffer: deque[dict] = deque(maxlen=5)
    processed = 0
    finished = False
    pbar = tqdm(total=total if total > 0 else None, desc="overlay")
    decode_fps = 0.0
    inference_fps = 0.0

    decode_times: deque[float] = deque(maxlen=20)
    infer_times: deque[float] = deque(maxlen=20)

    while not finished:
        batch: list[tuple[int, np.ndarray]] = []
        decode_start = time.time()
        while len(batch) < batch_size:
            item = reader.frame_queue.get()
            if item is None:
                finished = True
                break
            batch.append(item)
        decode_elapsed = time.time() - decode_start
        if batch:
            decode_times.append(decode_elapsed / len(batch))
            decode_fps = 1.0 / (sum(decode_times) / len(decode_times)) if decode_times else 0.0

        if not batch:
            continue

        infer_start = time.time()
        results = processor.process_batch(batch, fps)
        infer_elapsed = time.time() - infer_start
        infer_times.append(infer_elapsed / len(batch))
        inference_fps = len(batch) / infer_elapsed

        for frame_idx, records, annotated, meta in results:
            if meta.get("possession"):
                possession_buffer.append(meta["possession"])
            video_writer.queue.put((frame_idx, annotated))
            csv_writer.queue.put((frame_idx, records))
            processed += 1

        if processed % 10 == 0:
            emit(
                {
                    "stage": "inference",
                    "frame": frame_idx,
                    "processed": processed,
                    "total": total if total > 0 else None,
                    "classes": meta["counts"],
                    "decode_fps": round(decode_fps, 1),
                    "inference_fps": round(inference_fps, 1),
                }
            )

        pbar.update(len(batch))

    pbar.close()

    # Flush writers
    video_writer.queue.put(None)
    csv_writer.queue.put(None)
    video_writer.join()
    csv_writer.join()

    if reader.error:
        raise reader.error
    if video_writer.error:
        raise video_writer.error
    if csv_writer.error:
        raise csv_writer.error

    emit({"stage": "done", "output": str(output_path), "csv": str(csv_path), "processed": processed})
    print(f"Wrote overlay video: {output_path}")
    print(f"Wrote detections CSV: {csv_path}")
