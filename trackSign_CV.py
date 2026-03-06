"""
COSC_4366EL_A1_G1_CV.py  —  Task 1: AprilTag/ArUco Geometric Detection
Course : COSC-4366EL-01 Autonomous Mobile Robotics  |  Assignment 1 (2026 W)

Implements trackSign_CV() to detect AprilTag 36h11 markers in live video and
output navigation decisions for the six FIRA Autonomous Urban Driving signs.

Runtime overlay: green corner polygon, tag ID + sign label, navigation action
(cyan), estimated distance (cm), rolling FPS counter.

AI Assistance: Claude Sonnet 4.6 (Anthropic, claude.ai/code) used for code
generation, optimisation strategy, and documentation.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AprilTag 36h11 ID  →  (sign label, FIRA navigation action)
# !! UPDATE THESE IDs to match YOUR printed markers !!
SIGN_MAP: dict[int, tuple[str, str]] = {
    0: ("No Entry", "DO NOT ENTER"),
    1: ("Dead End", "DO NOT ENTER"),
    2: ("Right",    "TURN RIGHT"),
    3: ("Left",     "TURN LEFT"),
    4: ("Forward",  "GO FORWARD"),
    5: ("Stop",     "STOP"),
}

# Pre-compute action text widths — eliminates per-frame getTextSize calls
_ACTION_WIDTHS: dict[str, int] = {
    action: cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
    for _, action in SIGN_MAP.values()
}

DEFAULT_TAG_SIZE_M: float = 0.09   # 8–10 cm printed; use 9 cm midpoint
FRAME_W: int = 640
FRAME_H: int = 480
_INF: float = float("inf")

# ---------------------------------------------------------------------------
# Optimised ArUco detector  (built once at import time)
# ---------------------------------------------------------------------------

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

def _build_detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin  = 3
    p.adaptiveThreshWinSizeMax  = 23
    p.adaptiveThreshWinSizeStep = 10    # 3 threshold levels only (vs ~10 default) → faster
    p.minMarkerPerimeterRate    = 0.02
    p.maxMarkerPerimeterRate    = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_NONE  # skip sub-pixel → +4–6 FPS
    return cv2.aruco.ArucoDetector(_ARUCO_DICT, p)

_DETECTOR = _build_detector()

# ---------------------------------------------------------------------------
# Threaded camera capture  (decouples I/O latency from processing loop)
# ---------------------------------------------------------------------------

class _CameraStream:
    def __init__(self, src: int | str = 0) -> None:
        self._cap = cv2.VideoCapture(src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._ret, self._frame = self._cap.read()
        self._buf: np.ndarray | None = (
            np.empty_like(self._frame) if self._frame is not None else None
        )
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None
            if self._buf is None or self._buf.shape != self._frame.shape:
                self._buf = np.empty_like(self._frame)
            np.copyto(self._buf, self._frame)
            return True, self._buf

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

# ---------------------------------------------------------------------------
# Drawing helpers  (all in-place)
# ---------------------------------------------------------------------------

def _put_bg_text(img, text, org, scale, color, thick) -> None:
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - bl - 2), (x + tw + 2, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_tag(img, corners, tag_id, label, action, dist_m) -> None:
    pts = corners[0].astype(np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for pt in pts:
        cv2.circle(img, tuple(pt), 4, (0, 255, 0), -1)

    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())

    _put_bg_text(img, f"ID:{tag_id}  {label}", (cx - 50, cy - 10), 0.55, (255, 255, 255), 1)

    # Use pre-computed width to centre action text without a getTextSize call
    aw = _ACTION_WIDTHS.get(action, 120)
    _put_bg_text(img, action, (cx - aw // 2, cy + 30), 0.75, (0, 255, 255), 2)

    dist_str = f"{dist_m * 100:.1f} cm" if dist_m != _INF else "?? cm"
    cv2.putText(img, dist_str, (cx - 20, cy + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 50), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Rolling FPS counter
# ---------------------------------------------------------------------------

class _FPS:
    def __init__(self, window: int = 30) -> None:
        self._t: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._t.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0.0 else 0.0

# ---------------------------------------------------------------------------
# Public API  (required by assignment)
# ---------------------------------------------------------------------------

def trackSign_CV(source: int | str = 0, tag_size_m: float = DEFAULT_TAG_SIZE_M) -> None:
    """
    Run AprilTag 36h11 sign detection on live video.
    Draws corner overlays, tag ID, sign label, navigation action, and distance.
    Press 'q' or Esc to quit.
    """
    stream  = _CameraStream(source)
    fps_ctr = _FPS(window=30)
    focal   = float(max(FRAME_W, FRAME_H))   # approximate focal length (pixels)
    gray_buf: np.ndarray | None = None
    log_buf: list[str] = []

    print("[CV] AprilTag 36h11 detection — press 'q' or Esc to quit")
    print(f"[CV] ID map: {SIGN_MAP}\n")

    frame_count = 0
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        if gray_buf is None or gray_buf.shape[:2] != frame.shape[:2]:
            gray_buf = np.empty(frame.shape[:2], dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_buf)
        corners, ids, _ = _DETECTOR.detectMarkers(gray_buf)

        if ids is not None:
            flat_ids = ids.ravel()
            for corner, tid in zip(corners, flat_ids):
                dist           = (tag_size_m * focal) / max(float(np.linalg.norm(corner[0][0] - corner[0][1])), 1.0)
                label, action  = SIGN_MAP.get(int(tid), (f"ID:{tid}", "UNKNOWN"))
                _draw_tag(frame, corner, int(tid), label, action, dist)
                dist_str = f"{dist * 100:.1f} cm" if dist != _INF else "?? cm"
                log_buf.append(f"  Tag {tid:3d}  {label:<14s}  {action:<16s}  {dist_str}")

        fps_ctr.tick()
        cv2.putText(frame, f"FPS: {fps_ctr.fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        n = 0 if ids is None else len(ids)
        cv2.putText(frame, f"Tags: {n}", (FRAME_W - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        frame_count += 1
        if log_buf and frame_count % 30 == 0:
            print("\n".join(log_buf))
            log_buf.clear()

        cv2.imshow("Task 1 — AprilTag CV Detection  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    if log_buf:
        print("\n".join(log_buf))
    stream.release()
    cv2.destroyAllWindows()
    print("[CV] Done.")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task 1: AprilTag 36h11 sign detection")
    ap.add_argument("--source",   default=0,            help="Camera index or video file (default: 0)")
    ap.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE_M, help="Tag size in metres (default: 0.09)")
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    trackSign_CV(source=src, tag_size_m=args.tag_size)
