"""
SAR Ship Detection Pipeline — Diagnostic Edge-Case Tests
=========================================================
Creates synthetic SAR-like images with numpy and runs the YOLO model
to verify behaviour at tile boundaries, overlapping tiles, and
centre-vs-edge confidence differences.

The synthetic ships are designed to mimic real SAR backscatter:
  - High-intensity core with elongated shape (aspect ratio > 1.2)
  - Gaussian brightness falloff around the hull
  - Sidelobe/wake streaks trailing behind
  - Speckle-noise ocean background

Run from project root:
  python scripts/test_pipeline_edges.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# ── Resolve project root & model ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _find_model() -> str:
    candidates = [
        PROJECT_ROOT / "models" / "ssdd_yolov8s_110e10" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "runs" / "yolov8s_sar3" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "yolov8s.pt",
        PROJECT_ROOT / "yolov8n.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n.pt"


# ── Constants ─────────────────────────────────────────────────────────────────
TILE = 640
CONF = 0.10  # very low threshold — synthetic targets are hard for a real model


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dark_sea(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Realistic SAR ocean background: Rayleigh-distributed speckle on dark water."""
    rng = np.random.RandomState(seed)
    # Rayleigh noise simulates SAR amplitude over water
    sigma = 12.0
    amplitude = rng.rayleigh(scale=sigma, size=(h, w)).astype(np.float32)
    amplitude = np.clip(amplitude, 0, 60).astype(np.uint8)
    return np.stack([amplitude] * 3, axis=-1)


def paint_ship(img: np.ndarray, cx: int, cy: int,
               length: int = 80, beam: int = 16, angle_deg: float = 0.0):
    """
    Paint a SAR-realistic ship target:
      1) Bright elongated ellipse (hull)
      2) Very bright core pixels
      3) Wake / sidelobe streak behind the ship
      4) Gaussian halo for point-spread function simulation
    """
    h, w = img.shape[:2]
    canvas = np.zeros((h, w), dtype=np.float32)

    # ── Hull: a filled, rotated ellipse ──
    axes = (length // 2, beam // 2)
    cv2.ellipse(canvas, (cx, cy), axes, angle_deg, 0, 360, 255, -1)

    # ── Super-bright core (strong scatterer) ──
    core_axes = (length // 4, beam // 4)
    cv2.ellipse(canvas, (cx, cy), core_axes, angle_deg, 0, 360, 255, -1)

    # ── Wake streak (line trailing behind ship) ──
    rad = np.deg2rad(angle_deg)
    wake_len = length * 2
    wx = int(cx - np.cos(rad) * wake_len)
    wy = int(cy - np.sin(rad) * wake_len)
    wx = np.clip(wx, 0, w - 1)
    wy = np.clip(wy, 0, h - 1)
    cv2.line(canvas, (cx, cy), (wx, wy), 100, max(1, beam // 6))

    # ── Gaussian blur to simulate SAR point-spread function ──
    ksize = max(3, (beam // 2) | 1)  # must be odd
    canvas = cv2.GaussianBlur(canvas, (ksize, ksize), sigmaX=beam / 3)

    # ── Composite onto image ──
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) + canvas, 0, 255
        ).astype(np.uint8)


def infer_tile(model, tile_rgb: np.ndarray) -> list:
    """Run YOLO on a single tile, return list of (x1, y1, x2, y2, conf)."""
    results = model.predict(source=tile_rgb, conf=CONF, verbose=False, imgsz=640)
    dets = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
            c = float(r.boxes.conf[i].cpu().numpy())
            dets.append((float(x1), float(y1), float(x2), float(y2), c))
    return dets


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ship_on_tile_boundary(model) -> bool:
    """
    Edge Case 1: Ship straddling the exact boundary between two tiles.
    Create 1280x640 canvas, place ship at x=640, infer both halves.
    PASS if ship detected in at least one tile.
    """
    img = make_dark_sea(TILE, TILE * 2, seed=10)
    paint_ship(img, cx=TILE, cy=TILE // 2, length=100, beam=22, angle_deg=0)

    left_tile = img[:, :TILE].copy()
    right_tile = img[:, TILE:].copy()

    dets_l = infer_tile(model, left_tile)
    dets_r = infer_tile(model, right_tile)

    total = len(dets_l) + len(dets_r)
    print(f"    Left tile detections : {len(dets_l)}")
    print(f"    Right tile detections: {len(dets_r)}")
    return total >= 1


def test_overlapping_tiles_duplicate(model) -> tuple:
    """
    Edge Case 2: Two tiles overlap by 100px, ship sits in the overlap zone.
    PASS if at least one detection. Report duplicate count.
    """
    overlap = 100
    canvas_w = TILE * 2 - overlap
    img = make_dark_sea(TILE, canvas_w, seed=20)
    ship_cx = TILE - overlap // 2  # centre of overlap zone
    paint_ship(img, cx=ship_cx, cy=TILE // 2, length=90, beam=20, angle_deg=15)

    tile_a = img[:, :TILE].copy()
    tile_b = img[:, canvas_w - TILE:].copy()

    dets_a = infer_tile(model, tile_a)
    dets_b = infer_tile(model, tile_b)

    total_boxes = len(dets_a) + len(dets_b)
    return total_boxes >= 1, total_boxes


def test_center_vs_edge_confidence(model) -> tuple:
    """
    Edge Case 3: Same ship at tile centre vs near tile edge.
    PASS if at least one position yields a detection.
    """
    # Centre placement
    img_c = make_dark_sea(TILE, TILE, seed=30)
    paint_ship(img_c, cx=TILE // 2, cy=TILE // 2, length=90, beam=20, angle_deg=30)
    dets_c = infer_tile(model, img_c)

    # Edge placement (ship near right edge, partially clipped)
    img_e = make_dark_sea(TILE, TILE, seed=31)
    paint_ship(img_e, cx=TILE - 50, cy=TILE // 2, length=90, beam=20, angle_deg=30)
    dets_e = infer_tile(model, img_e)

    conf_c = max((d[4] for d in dets_c), default=0.0)
    conf_e = max((d[4] for d in dets_e), default=0.0)
    detected = conf_c > 0 or conf_e > 0
    return detected, conf_c, conf_e


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  SAR Pipeline Edge-Case Diagnostic")
    print("=" * 64)

    model_path = _find_model()
    print(f"\n  Model : {model_path}")
    print(f"  Tile  : {TILE}x{TILE}   Conf threshold: {CONF}\n")

    from ultralytics import YOLO
    model = YOLO(model_path)

    results = {}

    # -- Check 1 ---------------------------------------------------------------
    print("-" * 64)
    print("  CHECK 1 -- Ship on tile boundary")
    passed = test_ship_on_tile_boundary(model)
    tag = "PASS" if passed else "FAIL"
    results["boundary"] = passed
    print(f"  Result: {tag}")

    # -- Check 2 ---------------------------------------------------------------
    print("-" * 64)
    print("  CHECK 2 -- Overlapping tiles (duplicate detection)")
    passed, total_boxes = test_overlapping_tiles_duplicate(model)
    tag = "PASS" if passed else "FAIL"
    results["overlap"] = passed
    print(f"  Total raw bounding boxes from 2 tiles: {total_boxes}")
    if total_boxes > 1:
        print(f"  NOTE: Duplicate detections present -- NMS required")
    elif total_boxes == 1:
        print(f"  Single detection -- no duplicate")
    else:
        print(f"  No detections at all")
    print(f"  Result: {tag}")

    # -- Check 3 ---------------------------------------------------------------
    print("-" * 64)
    print("  CHECK 3 -- Centre vs edge confidence")
    passed, conf_c, conf_e = test_center_vs_edge_confidence(model)
    tag = "PASS" if passed else "FAIL"
    results["confidence"] = passed
    print(f"  Confidence at tile centre : {conf_c:.4f}")
    print(f"  Confidence at tile edge   : {conf_e:.4f}")
    if conf_c > 0 and conf_e > 0:
        delta = abs(conf_c - conf_e) * 100
        print(f"  Delta confidence          : {delta:.1f} pp")
        if conf_c > conf_e:
            print(f"  NOTE: Edge placement reduces confidence (expected)")
        else:
            print(f"  Edge confidence >= centre (unusual but possible)")
    print(f"  Result: {tag}")

    # -- Summary ---------------------------------------------------------------
    print("=" * 64)
    total_pass = sum(results.values())
    total_tests = len(results)
    print(f"\n  SUMMARY: {total_pass}/{total_tests} checks passed\n")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"    {name:20s}  {status}")
    print()

    sys.exit(0 if total_pass == total_tests else 1)


if __name__ == "__main__":
    main()
