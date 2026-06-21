#!/usr/bin/env python3
"""Headless browser frame capture for live football inference.

Uses Playwright to open a web page with a <video> element, capture frames at a
given FPS, and send them as JPEG blobs to the live inference WebSocket server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
import websockets.exceptions
from playwright.async_api import async_playwright

from footballai._paths import REPO_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inference-live-capture",
        description="Headless browser capture to live inference WebSocket server",
    )
    parser.add_argument("--url", required=True, help="URL to open in headless browser")
    parser.add_argument("--ws", default="ws://localhost:8000", help="WebSocket server URL")
    parser.add_argument("--fps", type=float, default=5, help="Frames to capture per second")
    parser.add_argument("--width", type=int, default=1280, help="Browser viewport width")
    parser.add_argument("--height", type=int, default=720, help="Browser viewport height")
    parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = unlimited)")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "live" / "frames"),
        help="Directory to save annotated frames received from the server",
    )
    parser.add_argument("--send-only", action="store_true", help="Do not wait for/writes annotated frames back")
    parser.add_argument("--video-selector", default="video", help="CSS selector for the video element")
    parser.add_argument("--wait-for-video", type=float, default=5, help="Seconds to wait for the video to start playing")
    parser.add_argument("--wait-for-selector", default="", help="Optional CSS selector to wait for before capturing")
    return parser.parse_args()


async def _send_jpeg(ws, jpeg_bytes: bytes) -> None:
    try:
        await ws.send(jpeg_bytes)
    except websockets.exceptions.ConnectionClosed:
        raise RuntimeError("WebSocket closed while sending frame")


async def _run_capture(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / max(1, args.fps)

    async with websockets.connect(args.ws) as ws:
        await ws.send(json.dumps({"action": "configure", "options": {}}))

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                executable_path="/snap/bin/chromium",
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--allow-file-access-from-files",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            page = await browser.new_page(viewport={"width": args.width, "height": args.height})
            # Local files / streams never reach "networkidle" because the media keeps the connection alive.
            wait_event = "load" if args.url.startswith(("file://", "http://", "https://")) else "networkidle"
            await page.goto(args.url, wait_until=wait_event)

            if args.wait_for_selector:
                try:
                    await page.wait_for_selector(args.wait_for_selector, timeout=args.wait_for_video * 1000)
                except Exception as exc:
                    await browser.close()
                    raise RuntimeError(f"Wait selector not found: {exc}")

            # Wait until the video element exists and is playing.
            try:
                await page.wait_for_selector(args.video_selector, timeout=args.wait_for_video * 1000)
            except Exception as exc:
                await browser.close()
                raise RuntimeError(f"Video selector not found: {exc}")

            # Try to start the video if it is not already playing.
            await page.evaluate(
                f"""() => {{
                    const video = document.querySelector('{args.video_selector}');
                    if (video) {{
                        video.crossOrigin = 'anonymous';
                        video.muted = true;
                        if (video.paused) {{
                            video.play().catch(() => {{}});
                        }}
                    }}
                }}"""
            )

            start_time = time.time()
            frame_count = 0

            print(f"Capturing from {args.url} at {args.fps} fps...")

            while True:
                if args.duration > 0 and time.time() - start_time >= args.duration:
                    break

                # Capture current video frame to an offscreen canvas.
                data_url = await page.evaluate(
                    f"""() => {{
                        const video = document.querySelector('{args.video_selector}');
                        if (!video || video.readyState < 2) return null;
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth || {args.width};
                        canvas.height = video.height || {args.height};
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        return canvas.toDataURL('image/jpeg', 0.85);
                    }}"""
                )

                if data_url is None:
                    await asyncio.sleep(interval)
                    continue

                header, b64 = data_url.split(",", 1)
                jpeg_bytes = base64.b64decode(b64)

                await _send_jpeg(ws, jpeg_bytes)
                frame_count += 1

                if not args.send_only:
                    # The server sends back annotated frame as binary and metadata as text.
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        if isinstance(msg, bytes):
                            array = np.frombuffer(msg, dtype=np.uint8)
                            annotated = cv2.imdecode(array, cv2.IMREAD_COLOR)
                            if annotated is not None:
                                path = output_dir / f"frame_{frame_count:06d}.jpg"
                                cv2.imwrite(str(path), annotated)
                        else:
                            try:
                                payload = json.loads(msg)
                                if payload.get("type") == "metadata":
                                    print(json.dumps(payload))
                            except json.JSONDecodeError:
                                pass
                    except asyncio.TimeoutError:
                        pass

                await asyncio.sleep(interval)

            await browser.close()
            print(f"Sent {frame_count} frames to {args.ws}")


async def main_async() -> None:
    args = _parse_args()
    await _run_capture(args)


def main() -> None:
    import asyncio

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
