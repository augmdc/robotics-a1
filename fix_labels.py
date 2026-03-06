"""
fix_labels.py — Convert polygon (segmentation) labels to bounding box (detection) format.

Scans all .txt label files in train/valid/test splits.
Lines with >5 values are polygon annotations; they are converted to
axis-aligned bounding boxes (class cx cy w h) in-place.
"""

from pathlib import Path

DATASET = Path("fira-signs.v4i.yolov8")
SPLITS  = ["train/labels", "valid/labels", "test/labels"]


def poly_to_bbox(parts: list[str]) -> str:
    cls_id = parts[0]
    coords = [float(v) for v in parts[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


converted = 0
skipped   = 0

for split in SPLITS:
    label_dir = DATASET / split
    if not label_dir.exists():
        continue
    for txt in sorted(label_dir.glob("*.txt")):
        lines = txt.read_text().strip().splitlines()
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:
                new_lines.append(poly_to_bbox(parts))
                converted += 1
                changed = True
            else:
                new_lines.append(line.strip())
                skipped += 1
        if changed:
            txt.write_text("\n".join(new_lines) + "\n")

print(f"Converted {converted} polygon labels to bounding boxes")
print(f"Skipped {skipped} labels (already bbox format)")
