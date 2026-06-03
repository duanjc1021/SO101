#!/home/jincheng/Desktop/vla/lerobot/.miniforge3/envs/lerobot/bin/python
"""Serve connected camera feeds in a local browser page.

Defaults for this setup:
  RealSense D435i: /dev/video4
  USB2.0 CAM1:    /dev/video6

Run:
  python view_cameras.py

Then open:
  http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import html
import signal
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np


DEFAULT_CAMERAS = ("/dev/video4", "/dev/video6")
DEFAULT_CAMERA_LABELS = {
    "/dev/video4": "RealSense D435i",
    "/dev/video6": "USB2.0 CAM1",
}
DEFAULT_CAMERA_CROPS = {
    # Keep the tabletop workspace shown in the screenshots and crop out the less useful edges.
    # Values are normalized left, top, right, bottom coordinates.
    "/dev/video4": (0.00, 0.04, 0.94, 0.93),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display two camera feeds side by side.")
    parser.add_argument(
        "cameras",
        nargs="*",
        default=list(DEFAULT_CAMERAS),
        help="Camera indices or device paths. Defaults to the external RealSense and USB cameras.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    source: int | str
    source = int(device) if device.isdigit() else device
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {device}")
    return cap


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def blank_frame(width: int, height: int, label: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, label, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
    return frame


def crop_frame(frame: np.ndarray, crop: tuple[float, float, float, float] | None) -> np.ndarray:
    if crop is None:
        return frame

    height, width = frame.shape[:2]
    left, top, right, bottom = crop
    x0 = max(0, min(width - 1, round(left * width)))
    y0 = max(0, min(height - 1, round(top * height)))
    x1 = max(x0 + 1, min(width, round(right * width)))
    y1 = max(y0 + 1, min(height, round(bottom * height)))
    return frame[y0:y1, x0:x1]


class CameraStream:
    def __init__(
        self,
        device: str,
        label: str,
        crop: tuple[float, float, float, float] | None,
        width: int,
        height: int,
        fps: int,
        stop_event: Event,
    ) -> None:
        self.device = device
        self.label = label
        self.crop = crop
        self.width = width
        self.height = height
        self.fps = fps
        self.stop_event = stop_event
        self.lock = Lock()
        self.frame = blank_frame(width, height, f"Starting {label}")
        self.cap = open_camera(device, width, height, fps)
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        delay_s = 1.0 / self.fps if self.fps > 0 else 0.001
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                frame = blank_frame(self.width, self.height, f"No frame from {self.label}")
            else:
                frame = crop_frame(frame, self.crop)
                frame = cv2.resize(frame, (self.width, self.height))
                frame = label_frame(frame, self.label)
            with self.lock:
                self.frame = frame
            time.sleep(delay_s)

    def jpeg(self) -> bytes:
        with self.lock:
            frame = self.frame.copy()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError(f"Could not encode frame from {self.device}")
        return encoded.tobytes()

    def close(self) -> None:
        self.cap.release()


def make_handler(streams: list[CameraStream]) -> type[BaseHTTPRequestHandler]:
    class CameraHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/":
                self._send_index()
                return
            if self.path.startswith("/stream/") and self.path.endswith(".mjpg"):
                try:
                    index = int(self.path.removeprefix("/stream/").removesuffix(".mjpg"))
                    stream = streams[index]
                except (ValueError, IndexError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_stream(stream)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send_index(self) -> None:
            cards = "\n".join(
                f'<section><h2>{html.escape(stream.label)} <small>{html.escape(stream.device)}</small></h2>'
                f'<img src="/stream/{index}.mjpg" alt="{html.escape(stream.label)}"></section>'
                for index, stream in enumerate(streams)
            )
            body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SO101 Camera Scenes</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; padding: 12px; }}
    section {{ background: #1b1b1b; border-radius: 10px; padding: 10px; }}
    h2 {{ font-size: 16px; font-weight: 500; margin: 0 0 8px; }}
    img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  </style>
</head>
<body>
  <main>
    {cards}
  </main>
</body>
</html>""".encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_stream(self, stream: CameraStream) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not stream.stop_event.is_set():
                try:
                    jpeg = stream.jpeg()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / max(stream.fps, 1))
                except (BrokenPipeError, ConnectionResetError):
                    break

    return CameraHandler


def main() -> int:
    args = parse_args()
    devices = [str(Path(camera)) if camera.startswith("/dev/") else camera for camera in args.cameras]
    stop_event = Event()
    streams: list[CameraStream] = []

    try:
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        for device in devices:
            streams.append(
                CameraStream(
                    device,
                    DEFAULT_CAMERA_LABELS.get(device, device),
                    DEFAULT_CAMERA_CROPS.get(device),
                    args.width,
                    args.height,
                    args.fps,
                    stop_event,
                )
            )

        server = ThreadingHTTPServer((args.host, args.port), make_handler(streams))
        server.timeout = 0.5
        print(f"Camera scenes: http://{args.host}:{args.port}", flush=True)
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        for stream in streams:
            stream.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
