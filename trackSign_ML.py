"""
COSC_4366EL_A1_G1_ML.py  —  Task 2: Visual Sign Detection with YOLO
Course : COSC-4366EL-01 Autonomous Mobile Robotics  |  Assignment 1 (2026 W)

Implements trackSign_ML() to run a fine-tuned YOLOv8n model on live video
and output navigation decisions for the six FIRA Urban Driving signs.

Runtime overlay: bounding box, label + confidence, navigation action (cyan),
estimated distance (cm), rolling FPS counter.

Training mode (--train): fine-tunes yolov8n.pt on a YOLOv8-format YAML dataset
and copies best weights to best_yolo.pt.

AI Assistance: Claude Sonnet 4.6 (Anthropic, claude.ai/code) used for code
generation, optimisation strategy, and documentation.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Class index (must match data.yaml nc order)  →  (label, FIRA action)
CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("No Entry",        "DO NOT ENTER"),
    1: ("Dead End",        "DO NOT ENTER"),
    2: ("Proceed Right",   "TURN RIGHT"),
    3: ("Proceed Left",    "TURN LEFT"),
    4: ("Proceed Forward", "GO FORWARD"),
    5: ("Stop",            "STOP"),
}

# Per-class bounding-box colours (BGR)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0,   0,   220),   # No Entry        – red
    1: (0,   140, 255),   # Dead End        – orange
    2: (0,   200,   0),   # Proceed Right   – green
    3: (255, 200,   0),   # Proceed Left    – teal
    4: (200, 255,   0),   # Proceed Forward – lime
    5: (0,   0,   180),   # Stop            – dark red
}

# Pre-compute action text pixel widths (eliminates per-frame getTextSize calls)
_ACTION_WIDTHS: dict[str, int] = {
    action: cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    for _, action in CLASS_MAP.values()
}

SIGN_WIDTH_M: float    = 0.12        # estimated physical sign width for distance
CONF_THRESH:  float    = 0.45
DEFAULT_MODEL: str     = "best_yolo.pt"
FRAME_W: int           = 640
FRAME_H: int           = 480
_INF: float            = float("inf")

# ---------------------------------------------------------------------------
# Threaded camera capture
# ---------------------------------------------------------------------------

class _CameraStream:
    """
    Grab frames in a daemon thread to decouple I/O latency from inference.
    CAP_PROP_BUFFERSIZE=1 keeps the grabbed frame as fresh as possible.
    """

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
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_bg_text(img, text, org, scale, color, thick) -> None:
    """Render text with a solid black background rectangle."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - bl - 2), (x + tw + 2, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_detection(img, x1, y1, x2, y2, cls_id, label, action, conf, dist_m) -> None:
    """Draw box, label, confidence, action, and distance in-place."""
    color = CLASS_COLORS.get(cls_id, (128, 128, 128))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    _put_bg_text(img, f"{label}  {conf:.0%}", (x1, max(y1 - 6, 14)), 0.55, (255, 255, 255), 1)
    # Use pre-computed width to centre action text without a getTextSize call
    aw = _ACTION_WIDTHS.get(action, 100)
    _put_bg_text(img, action, ((x1 + x2) // 2 - aw // 2, y1 + 26), 0.7, (0, 255, 255), 2)
    dist_str = f"{dist_m * 100:.1f} cm" if dist_m != _INF else "?? cm"
    cv2.putText(img, dist_str, (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1, cv2.LINE_AA)


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
# Training
# ---------------------------------------------------------------------------

def train_model(
    data_yaml: str,
    base_model: str  = "yolov8n.pt",
    epochs: int      = 150,
    imgsz: int       = 640,
    batch: int       = 16,
    output_name: str = "signs_yolov8n",
) -> Path:
    """
    Fine-tune YOLOv8n on a six-class sign dataset using the native yolo CLI.

    1. Trains the model  →  saves best.pt
    2. Evaluates on test split  →  prints mAP@0.5, per-class P/R, saves confusion matrix

    Augmentations: HSV jitter (brightness/contrast), scale, mosaic.
    Flips and rotation disabled to preserve arrow direction (Left/Right/Forward).
    """
    save_dir = Path("runs") / "detect" / "runs" / output_name

    # --- Train ---------------------------------------------------------------
    subprocess.run([
        "yolo", "detect", "train",
        f"data={data_yaml}", f"model={base_model}",
        f"epochs={epochs}", f"imgsz={imgsz}", f"batch={batch}",
        f"name={output_name}", "project=runs", "exist_ok=True",
        # Augmentations
        "hsv_h=0.02", "hsv_s=0.5", "hsv_v=0.4",
        "scale=0.7", "mosaic=1.0", "perspective=0.0005",
        # Disabled (preserve arrow direction)
        "degrees=0.0", "translate=0.1",
        "fliplr=0.0", "flipud=0.0",
        "mixup=0.15", "copy_paste=0.0",
        "rect=True", "patience=25",
        # Tuning
        "lr0=0.001", "freeze=5", "dropout=0.1", "optimizer=AdamW", "weight_decay=0.0005",
        "erasing=0.0", "label_smoothing=0.1", "cos_lr=True",
        # Device + speed (no accuracy cost)
        "device=mps",
        "cache=True", "workers=0", "deterministic=False",
        "close_mosaic=20", "plots=False",
    ], check=True)

    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, "best_yolo.pt")
        print("[ML] Training complete  →  ./best_yolo.pt")

    # --- Evaluate on test split ----------------------------------------------
    # Prints mAP@0.5, per-class precision/recall; saves confusion_matrix.png
    model_to_eval = str(best_pt) if best_pt.exists() else base_model
    subprocess.run([
        "yolo", "detect", "val",
        f"model={model_to_eval}", f"data={data_yaml}",
        f"imgsz={imgsz}", "split=test",
        f"name={output_name}_eval", "project=runs", "exist_ok=True",
        "plots=True",   # saves confusion matrix + PR curves
    ], check=True)

    eval_dir = save_dir.parent / f"{output_name}_eval"
    print(f"\n[ML] Evaluation results + confusion matrix saved → {eval_dir}")

    return best_pt


# ---------------------------------------------------------------------------
# Public API  (required by assignment)
# ---------------------------------------------------------------------------

def trackSign_ML(
    source:       int | str = 0,
    model_path:   str       = DEFAULT_MODEL,
    sign_width_m: float     = SIGN_WIDTH_M,
    conf:         float     = CONF_THRESH,
    imgsz:        int       = 640,
) -> None:
    """
    Run YOLOv8n sign detection on live video.
    Draws boxes, labels, confidence, action, and estimated distance.
    Press 'q' or Esc to quit.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        raise SystemExit("ultralytics not installed.  Run: pip install ultralytics")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.  Run with --train first."
        )

    model = YOLO(model_path)
    focal    = float(max(FRAME_W, FRAME_H))   # approximate focal length (pixels)

    stream  = _CameraStream(source)
    fps_ctr = _FPS(window=30)

    print(f"[ML] model={model_path}  device={model.device}  conf={conf:.0%}")
    print("[ML] Press 'q' or Esc to quit\n")

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        results = model.predict(
            source=frame, imgsz=imgsz, conf=conf,
            device="mps", verbose=False, stream=False,
        )

        n_det = 0
        if results and results[0].boxes:
            boxes = results[0].boxes
            xyxy_np  = boxes.xyxy.cpu().numpy().astype(np.int32)
            cls_np   = boxes.cls.cpu().numpy().astype(np.int32).ravel()
            conf_np  = boxes.conf.cpu().numpy().ravel()
            widths   = np.maximum(xyxy_np[:, 2] - xyxy_np[:, 0], 1)
            dists    = (sign_width_m * focal) / widths
            n_det    = len(cls_np)
            for i in range(n_det):
                x1, y1, x2, y2 = xyxy_np[i]
                cls_id = int(cls_np[i])
                label, action = CLASS_MAP.get(cls_id, (f"cls{cls_id}", "UNKNOWN"))
                _draw_detection(frame, int(x1), int(y1), int(x2), int(y2),
                                cls_id, label, action, float(conf_np[i]), float(dists[i]))

        fps_ctr.tick()
        cv2.putText(frame, f"FPS: {fps_ctr.fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Det: {n_det}", (FRAME_W - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Task 2 — YOLO Sign Detection  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    stream.release()
    cv2.destroyAllWindows()
    print("[ML] Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Task 2: YOLO visual sign detection")
    ap.add_argument("--train",      action="store_true",  help="Run training instead of inference")
    ap.add_argument("--data",       default="data.yaml",  help="YOLOv8 dataset YAML (training)")
    ap.add_argument("--base-model", default="yolov8n.pt", help="Base weights for training")
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--imgsz",      type=int,   default=640)
    ap.add_argument("--batch",      type=int,   default=16)
    ap.add_argument("--model",      default=DEFAULT_MODEL, help="Trained .pt for inference")
    ap.add_argument("--source",     default=0,  help="Camera index or video file")
    ap.add_argument("--conf",       type=float, default=CONF_THRESH)
    args = ap.parse_args()

    if args.train:
        train_model(args.data, args.base_model, args.epochs, args.imgsz, args.batch)
    else:
        src = int(args.source) if str(args.source).isdigit() else args.source
        trackSign_ML(src, args.model, conf=args.conf, imgsz=args.imgsz)
