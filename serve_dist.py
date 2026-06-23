#!/usr/bin/env python3
"""Serve apps/web/dist with correct MIME types for modern module scripts."""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "apps" / "web" / "dist"


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        ".html": "text/html",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".ico": "image/x-icon",
        "": "application/octet-stream",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the built FootballAI SPA")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Avoid poisoning the global MIME registry on systems with broken defaults.
    mimetypes.init()

    with socketserver.TCPServer((args.host, args.port), _Handler) as httpd:
        print(f"Serving {ROOT} at http://{args.host}:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
