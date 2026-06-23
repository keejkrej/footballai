#!/usr/bin/env python3
"""Pluggable video sources for backend-owned live inference.

The :class:`VideoSource` abstraction hides where frames come from so the live
inference server can consume local files, browser-rendered URLs, YouTube clips,
webcams, or OBS outputs through the same interface.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from playwright.async_api import async_playwright


class VideoSource(ABC):
    """Abstract source of decoded video frames.

    Implementations are opened once, then produce frames via ``frames()`` until
    the source ends (file) or is closed (live stream).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self._closed = False

    @abstractmethod
    async def open(self) -> None:
        """Prepare the source and populate width/height/fps."""

    @abstractmethod
    async def close(self) -> None:
        """Release all resources."""

    @abstractmethod
    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        """Yield ``(frame_index, frame_bgr, fps)`` tuples."""


class FileVideoSource(VideoSource):
    """Read frames from a local video file with optional stride/throttle."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0
        self._emitted_idx = 0

    async def open(self) -> None:
        path = Path(self.config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        def _open() -> cv2.VideoCapture:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            return cap

        self._cap = await asyncio.to_thread(_open)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self._frame_idx = 0
        self._emitted_idx = 0

    async def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            cap = self._cap
            self._cap = None
            await asyncio.to_thread(cap.release)

    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        if self._cap is None:
            raise RuntimeError("Source not opened")

        stride = max(1, int(self.config.get("stride", 1)))
        max_fps = self.config.get("max_fps")
        interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
        last_emit = 0.0

        while not self._closed:
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok:
                break

            idx = self._frame_idx
            self._frame_idx += 1
            if idx % stride != 0:
                continue

            if interval:
                now = time.monotonic()
                if now - last_emit < interval:
                    continue
                last_emit = now

            yield self._emitted_idx, frame, self.fps
            self._emitted_idx += 1


class WebcamVideoSource(VideoSource):
    """Capture frames from a camera device index (e.g. 0) or path."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0

    async def open(self) -> None:
        device = self.config.get("device", 0)

        def _open() -> cv2.VideoCapture:
            cap = cv2.VideoCapture(
                int(device) if isinstance(device, str) and device.isdigit() else device
            )
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam device: {device}")
            return cap

        self._cap = await asyncio.to_thread(_open)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self._frame_idx = 0

    async def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            cap = self._cap
            self._cap = None
            await asyncio.to_thread(cap.release)

    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        if self._cap is None:
            raise RuntimeError("Source not opened")

        max_fps = self.config.get("max_fps")
        interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
        last_emit = 0.0

        while not self._closed:
            ok, frame = await asyncio.to_thread(self._cap.read)
            if not ok:
                await asyncio.sleep(0.05)
                continue

            if interval:
                now = time.monotonic()
                if now - last_emit < interval:
                    continue
                last_emit = now

            yield self._frame_idx, frame, self.fps
            self._frame_idx += 1


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG frame from browser")
    return frame


class BrowserVideoSource(VideoSource):
    """Capture frames from a URL using a headless Chromium browser.

    This is useful for streams or pages that only expose video through a
    browser, including DRM-free HLS or DASH players.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._page: Any | None = None

    async def open(self) -> None:
        url = self.config["url"]
        viewport_width = int(self.config.get("viewport_width", 1280))
        viewport_height = int(self.config.get("viewport_height", 720))
        executable_path = self.config.get("executable_path")
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--allow-file-access-from-files",
                "--autoplay-policy=no-user-gesture-required",
            ],
        }
        if executable_path:
            launch_kwargs["executable_path"] = executable_path

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._page = await self._browser.new_page(
            viewport={"width": viewport_width, "height": viewport_height}
        )

        # Local files / streams never reach "networkidle" because media keeps connections alive.
        wait_event = (
            "load"
            if url.startswith(("file://", "http://", "https://"))
            else "networkidle"
        )
        await self._page.goto(url, wait_until=wait_event)

        selector = self.config.get("video_selector", "video")
        wait_for_video = float(self.config.get("wait_for_video", 5))
        await self._page.wait_for_selector(selector, timeout=wait_for_video * 1000)

        # Try to start playback and allow canvas capture.
        await self._page.evaluate(
            f"""() => {{
                const video = document.querySelector('{selector}');
                if (video) {{
                    video.crossOrigin = 'anonymous';
                    video.muted = true;
                    if (video.paused) {{
                        video.play().catch(() => {{}});
                    }}
                }}
            }}"""
        )

        click_selector = self.config.get("click_selector")
        if click_selector:
            click_delay = float(self.config.get("click_delay", 2))
            await self._page.click(click_selector)
            await asyncio.sleep(click_delay)

        dims = await self._page.evaluate(
            f"""() => {{
                const video = document.querySelector('{selector}');
                return video ? {{ w: video.videoWidth, h: video.videoHeight }} : null;
            }}"""
        )
        self.width = int(dims["w"]) if dims and dims.get("w") else viewport_width
        self.height = int(dims["h"]) if dims and dims.get("h") else viewport_height
        self.fps = float(self.config.get("fps", 5))

    async def close(self) -> None:
        self._closed = True
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self._page = None

    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        if self._page is None:
            raise RuntimeError("Source not opened")

        selector = self.config.get("video_selector", "video")
        quality = float(self.config.get("quality", 0.85))
        fps = self.fps
        interval = 1.0 / max(1.0, fps)
        frame_idx = 0

        while not self._closed:
            data_url = await self._page.evaluate(
                f"""() => {{
                    const video = document.querySelector('{selector}');
                    if (!video || video.readyState < 2) return null;
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth || {self.width};
                    canvas.height = video.videoHeight || {self.height};
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    return canvas.toDataURL('image/jpeg', {quality});
                }}"""
            )

            if data_url is None:
                await asyncio.sleep(interval)
                continue

            try:
                header, b64 = data_url.split(",", 1)
                jpeg_bytes = base64.b64decode(b64)
                frame = await asyncio.to_thread(_decode_jpeg, jpeg_bytes)
            except Exception as exc:
                raise RuntimeError(f"Failed to decode browser frame: {exc}") from exc

            yield frame_idx, frame, fps
            frame_idx += 1
            await asyncio.sleep(interval)


class YoutubeVideoSource(VideoSource):
    """Stream frames from a YouTube URL through yt-dlp + ffmpeg.

    yt-dlp resolves the best playable video URL and ffmpeg decodes it
    frame-by-frame in real time. The source supports optional start/end
    timestamps so only a segment is streamed to inference.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._proc: subprocess.Popen | None = None
        self._frame_idx = 0

    async def open(self) -> None:
        url = self.config.get("url")
        if not url:
            raise ValueError("YouTube source requires a 'url' field")

        height = int(self.config.get("height", 720))
        start = self.config.get("start", "")
        end = self.config.get("end", "")

        yt_dlp = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if yt_dlp is None:
            raise FileNotFoundError(
                "yt-dlp not found on PATH. Install it (e.g. "
                "`uv tool install yt-dlp` or `uv pip install yt-dlp`)."
            )
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg not found on PATH. Install it to use the YouTube source."
            )

        # Resolve stream metadata and direct URL. Prefer a combined MP4 format
        # when available so a single URL can be handed to ffmpeg.
        info_cmd = [
            yt_dlp,
            "--no-playlist",
            "--no-warnings",
            "-f",
            f"best[ext=mp4][height<={height}]/best[height<={height}]/best",
            "--print",
            "%(width)s,%(height)s,%(fps)s,%(url)s",
            url,
        ]
        try:
            info = await asyncio.to_thread(
                subprocess.check_output, info_cmd, stderr=subprocess.PIPE, text=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"yt-dlp failed to resolve YouTube URL: {exc.stderr or exc.output}"
            ) from exc

        parts = info.strip().rsplit(",", 3)
        if len(parts) < 4:
            raise RuntimeError(f"Unexpected yt-dlp output: {info!r}")
        raw_width, raw_height, raw_fps, stream_url = parts

        def _parse_int(value: str, default: int) -> int:
            try:
                return int(value) if value and value.lower() != "na" else default
            except ValueError:
                return default

        def _parse_float(value: str, default: float) -> float:
            try:
                return float(value) if value and value.lower() != "na" else default
            except ValueError:
                return default

        self.width = _parse_int(raw_width, 1280)
        self.height = _parse_int(raw_height, 720)
        self.fps = _parse_float(raw_fps, 25.0)

        # Build the ffmpeg command. Start/end timestamps are applied before the
        # input so ffmpeg seeks quickly on the network stream.
        ffmpeg_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-thread_queue_size",
            "512",
        ]
        if start:
            ffmpeg_cmd.extend(["-ss", str(start)])
        if end:
            ffmpeg_cmd.extend(["-to", str(end)])
        ffmpeg_cmd.extend(["-i", stream_url])
        max_fps = self.config.get("max_fps")
        if max_fps and max_fps > 0:
            # Ask ffmpeg to deliver frames at the target inference rate so we
            # don't have to discard already-read network frames.
            ffmpeg_cmd.extend(["-r", str(max_fps)])
        ffmpeg_cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )

        def _start_ffmpeg() -> subprocess.Popen:
            return subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self._proc = await asyncio.to_thread(_start_ffmpeg)
        if self._proc.stdout is None:
            raise RuntimeError("ffmpeg stdout is not available")

    async def close(self) -> None:
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                await asyncio.to_thread(proc.wait, 5)
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Source not opened")

        frame_size = self.width * self.height * 3

        while not self._closed:
            chunk = await asyncio.to_thread(self._proc.stdout.read, frame_size)
            if len(chunk) < frame_size:
                # EOF or partial frame at the end of the stream.
                break

            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )

            yield self._frame_idx, frame, self.fps
            self._frame_idx += 1


class ObsVideoSource(VideoSource):
    """Wrap an OBS output as a video source.

    MVP supports two modes:
    - ``url``: OBS is streaming to an RTMP/RTSP/SRT/HTTP URL that the backend
      can consume. This delegates to :class:`BrowserVideoSource` or a future
      FFmpeg source depending on the URL.
    - ``device``: OBS is feeding a virtual camera (e.g. v4l2loopback on Linux).
      This delegates to :class:`WebcamVideoSource`.

    A richer OBS-websocket integration can be added later behind the same
    ``{"type": "obs"}`` interface.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._delegate: VideoSource | None = None

    async def open(self) -> None:
        mode = self.config.get("mode", "url")
        if mode == "url":
            url = self.config.get("url")
            if not url:
                raise ValueError("OBS url mode requires a 'url' field")
            delegate_config = {**self.config, "type": "url"}
            self._delegate = BrowserVideoSource(delegate_config)
        elif mode == "device":
            device = self.config.get("device")
            if device is None:
                raise ValueError("OBS device mode requires a 'device' field")
            delegate_config = {**self.config, "type": "webcam", "device": device}
            self._delegate = WebcamVideoSource(delegate_config)
        else:
            raise ValueError(f"Unknown OBS source mode: {mode}")

        await self._delegate.open()
        self.width = self._delegate.width
        self.height = self._delegate.height
        self.fps = self._delegate.fps

    async def close(self) -> None:
        self._closed = True
        if self._delegate is not None:
            await self._delegate.close()

    async def frames(self) -> AsyncIterator[tuple[int, np.ndarray, float]]:
        if self._delegate is None:
            raise RuntimeError("Source not opened")
        async for item in self._delegate.frames():
            yield item


_SOURCE_CLASSES: dict[str, type[VideoSource]] = {
    "file": FileVideoSource,
    "url": BrowserVideoSource,
    "webcam": WebcamVideoSource,
    "obs": ObsVideoSource,
    "youtube": YoutubeVideoSource,
}


def create_video_source(config: dict[str, Any]) -> VideoSource:
    """Create a :class:`VideoSource` from a configuration dict."""
    source_type = config.get("type")
    if source_type not in _SOURCE_CLASSES:
        raise ValueError(f"Unknown video source type: {source_type}")
    return _SOURCE_CLASSES[source_type](config)
