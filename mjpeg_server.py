"""
mjpeg_server.py
───────────────
Tiny MJPEG streaming server that the AI service populates with annotated
frames.  The backend /camera/stream proxy connects here instead of the raw
camera, so the browser sees the fully-overlaid feed (slots, path, green dot).

Usage
-----
    from mjpeg_server import MJPEGServer

    server = MJPEGServer(port=8081)
    server.start()          # non-blocking background thread

    # Inside the main frame loop:
    server.push_frame(annotated_bgr_img)   # thread-safe
"""

import io
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_latest_jpeg: Optional[bytes] = None
_frame_lock = threading.Lock()
_frame_event = threading.Event()   # signals that a new frame is ready


def _encode(img: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ── HTTP request handler ──────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.rstrip("/") in ("/stream", ""):
            self._stream()
        elif self.path == "/snapshot":
            self._snapshot()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                # Wait up to 2 s for a new frame
                _frame_event.wait(timeout=2.0)
                _frame_event.clear()
                with _frame_lock:
                    data = _latest_jpeg
                if data is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n\r\n"
                                     + data + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break   # client disconnected
        except Exception:
            pass

    def _snapshot(self):
        with _frame_lock:
            data = _latest_jpeg
        if data is None:
            self.send_error(503, "No frame available yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass   # silence per-request logs


# ── Public API ────────────────────────────────────────────────────────────────
class MJPEGServer:
    """Wraps the HTTP server so the rest of the code stays clean."""

    def __init__(self, port: int = 8081):
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the MJPEG server in a background daemon thread."""
        try:
            self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        except OSError as exc:
            logger.warning(f"MJPEG server could not bind to port {self.port}: {exc}")
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="mjpeg-server"
        )
        self._thread.start()
        logger.info(f"MJPEG server started → http://localhost:{self.port}/stream")

    def push_frame(self, img: np.ndarray, quality: int = 70) -> None:
        """Encode and publish a new annotated frame (called from main loop)."""
        global _latest_jpeg
        jpeg = _encode(img, quality)
        with _frame_lock:
            _latest_jpeg = jpeg
        _frame_event.set()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        logger.info("MJPEG server stopped.")
