#!/usr/bin/env python3
"""Pluggable video sources for backend-owned live inference.

The :class:`VideoSource` abstraction hides where frames come from so the live
inference server can consume local files, browser-rendered URLs, webcams, or OBS
outputs through the same interface.
"""

from __future__ import annotations

import asyncio
import base64
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
}


def create_video_source(config: dict[str, Any]) -> VideoSource:
    """Create a :class:`VideoSource` from a configuration dict."""
    source_type = config.get("type")
    if source_type not in _SOURCE_CLASSES:
        raise ValueError(f"Unknown video source type: {source_type}")
    return _SOURCE_CLASSES[source_type](config)
