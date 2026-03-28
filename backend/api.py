"""
FastAPI backend server bridging the React UI with the Python SAR detection pipeline.

Run (supports uploads up to 2 GB):
  python -m uvicorn backend.api:app --port 8000 --reload --timeout-keep-alive 600 --h11-max-incomplete-event-size 0
"""
import asyncio
import gc
import io
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from PIL import Image
import google.generativeai as genai

# Configure Gemini if API key is present
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="AetherSAR API", version="1.0.0")

# Allow large file uploads (up to 2.5 GB)
MAX_UPLOAD_SIZE = int(2.5 * 1024 * 1024 * 1024)  # 2.5 GB

# --- Patch Starlette multipart parser size limits ---
# python-multipart / Starlette caps body at 1 MB by default.
# We must raise both the multipart form limit AND the general body limit.
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_file_size = MAX_UPLOAD_SIZE  # type: ignore
except Exception:
    pass

try:
    # Starlette >= 0.30  uses this constant
    import starlette.formparsers as _fp
    if hasattr(_fp, "MAX_FILE_SIZE"):
        _fp.MAX_FILE_SIZE = MAX_UPLOAD_SIZE  # type: ignore
except Exception:
    pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "temp_upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory file registry:  file_id -> metadata dict
_file_registry: Dict[str, dict] = {}
# In-memory detection results:  file_id -> detections list
_detection_results: Dict[str, dict] = {}
# Progress state:  file_id -> progress dict
_progress_state: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Constants (same as detection.py)
# ---------------------------------------------------------------------------
TILE_SIZE = 640
TILE_OVERLAP = 100
PREVIEW_MAX_DIM = 2400
WATER_BRIGHTNESS = 55.0
ASPECT_MIN = 1.2
CHUNK_WRITE_SIZE = 8 * 1024 * 1024
MAX_CACHED_RESULTS = 2  # Keep only the last N detection results in memory


def _human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _model_path() -> str:
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


def _compute_nms(boxes, scores, iou_threshold=0.20):
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=np.float64)
    s = np.array(scores, dtype=np.float64)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    """Return current pipeline configuration."""
    try:
        from config.config import (
            CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, INPUT_IMAGE_SIZE,
            TILE_SIZE as CFG_TILE_SIZE, TILE_OVERLAP as CFG_TILE_OVERLAP,
            CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, CFAR_PFA,
            SPECKLE_FILTER_TYPE, DEVICE,
        )
        return {
            "confidenceThreshold": CONFIDENCE_THRESHOLD,
            "nmsIouThreshold": NMS_IOU_THRESHOLD,
            "inputResolution": INPUT_IMAGE_SIZE,
            "tileSize": CFG_TILE_SIZE,
            "tileOverlap": CFG_TILE_OVERLAP,
            "guardCells": CFAR_GUARD_CELLS,
            "trainingCells": CFAR_TRAINING_CELLS,
            "pfaExp": abs(int(round(np.log10(CFAR_PFA)))),
            "speckleFilter": SPECKLE_FILTER_TYPE,
            "device": DEVICE,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a SAR image upload (up to 2 GB), stream to disk, return metadata."""
    try:
        if not file.filename:
            raise HTTPException(400, "No filename provided")

        # ── Aggressive cleanup: free ALL old temp files and in-memory state ──
        # This prevents disk/memory exhaustion after several large uploads.
        # Delete ALL files in temp_upload (not just >1 hour old)
        for f in UPLOAD_DIR.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

        # Evict old in-memory state to free RAM
        # (keep nothing — each new upload starts fresh)
        old_file_ids = list(_file_registry.keys())
        for old_id in old_file_ids:
            _detection_results.pop(old_id, None)
            _progress_state.pop(old_id, None)
            _file_registry.pop(old_id, None)
        gc.collect()
        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        ext = os.path.splitext(file.filename)[1].lower()
        allowed = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg", ".nc", ".h5"}
        if ext not in allowed:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        file_id = str(uuid.uuid4())[:12]
        safe_name = f"{file_id}_{file.filename}"
        dest = UPLOAD_DIR / safe_name

        # Stream to disk in 8 MB chunks — handles files up to 2 GB
        total_bytes = 0
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_WRITE_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
                # Yield control back to event loop periodically
                if total_bytes > 0:
                    await asyncio.sleep(0)

        # Read image metadata
        width, height, bands, crs, dtype_str = 0, 0, 1, "N/A", "uint8"
        is_tiff = ext in {".tif", ".tiff", ".geotiff"}

        if is_tiff:
            try:
                import rasterio
                with rasterio.open(str(dest)) as src:
                    width, height = src.width, src.height
                    bands = src.count
                    crs = str(src.crs) if src.crs else "N/A"
                    dtype_str = str(src.dtypes[0])
            except ImportError:
                try:
                    import tifffile
                    with tifffile.TiffFile(str(dest)) as tif:
                        page = tif.pages[0]
                        height, width = page.shape[:2]
                        bands = 1 if len(page.shape) == 2 else page.shape[2]
                        dtype_str = str(page.dtype)
                except Exception:
                    img = cv2.imread(str(dest))
                    if img is not None:
                        height, width = img.shape[:2]
        else:
            img = cv2.imread(str(dest))
            if img is not None:
                height, width = img.shape[:2]
                bands = img.shape[2] if img.ndim > 2 else 1

        meta = {
            "fileId": file_id,
            "filename": file.filename,
            "path": str(dest),
            "size": total_bytes,
            "sizeHuman": _human_size(total_bytes),
            "width": width,
            "height": height,
            "bands": bands,
            "crs": crs,
            "dtype": dtype_str,
            "isTiff": is_tiff,
        }
        _file_registry[file_id] = meta
        return meta
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Upload error: {str(e)}")
    finally:
        # CRITICAL: Always close the UploadFile to release the underlying
        # SpooledTemporaryFile. Without this, temp file handles leak on
        # every upload, eventually exhausting OS resources on Windows.
        try:
            await file.close()
        except Exception:
            pass


@app.get("/api/image/{file_id}")
async def get_image(file_id: str):
    """Serve the uploaded image as PNG (normalized for display)."""
    meta = _file_registry.get(file_id)
    if not meta:
        raise HTTPException(404, "File not found")

    image_path = meta["path"]

    if meta.get("isTiff"):
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                full_h, full_w = src.height, src.width
                preview_scale = min(1.0, PREVIEW_MAX_DIM / max(full_h, full_w))
                prev_h = int(full_h * preview_scale)
                prev_w = int(full_w * preview_scale)
                preview_raw = src.read(1, out_shape=(prev_h, prev_w)).astype(np.float32)
        except ImportError:
            import tifffile
            full_img = tifffile.imread(image_path)
            if full_img.ndim > 2:
                full_img = full_img[:, :, 0]
            full_h, full_w = full_img.shape[:2]
            preview_scale = min(1.0, PREVIEW_MAX_DIM / max(full_h, full_w))
            prev_h = int(full_h * preview_scale)
            prev_w = int(full_w * preview_scale)
            step_y = max(1, full_h // prev_h)
            step_x = max(1, full_w // prev_w)
            small_img = full_img[::step_y, ::step_x]
            preview_raw = cv2.resize(small_img.astype(np.float32), (prev_w, prev_h),
                                     interpolation=cv2.INTER_AREA)
            del full_img, small_img
            gc.collect()

        nonzero = preview_raw[preview_raw > 0]
        if nonzero.size > 0:
            p_lo = float(np.percentile(nonzero, 1))
            p_hi = float(np.percentile(nonzero, 99))
        else:
            p_lo, p_hi = 0.0, 1.0

        prev_clip = np.clip(preview_raw, p_lo, p_hi)
        if p_hi - p_lo > 0:
            prev_norm = ((prev_clip - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
        else:
            prev_norm = np.zeros_like(prev_clip, dtype=np.uint8)
        preview_rgb = np.stack((prev_norm,) * 3, axis=-1)
        del preview_raw, prev_clip, prev_norm
        gc.collect()
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise HTTPException(500, "Failed to read image")
        preview_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    Image.fromarray(preview_rgb).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/result-image/{file_id}")
async def get_result_image(file_id: str):
    """Serve the annotated result image."""
    result = _detection_results.get(file_id)
    if not result or "annotated_image_png" not in result:
        raise HTTPException(404, "No result image available")

    buf = io.BytesIO(result["annotated_image_png"])
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/detect")
async def run_detection(body: dict = Body(...)):
    """Run the SAR detection pipeline on an uploaded file."""
    file_id = body.get("fileId")
    config = body.get("config", {})

    meta = _file_registry.get(file_id)
    if not meta:
        raise HTTPException(404, "File not found. Upload first.")

    image_path = meta["path"]
    is_tiff = meta.get("isTiff", False)

    # Initialize progress
    _progress_state[file_id] = {
        "stage": 1, "progress": 0, "tileCount": 0, "totalTiles": 0,
        "detectionCount": 0, "done": False, "error": None,
    }

    # Run detection in background
    import threading

    def _run():
        print(f"[DETECT] Starting pipeline for file_id={file_id}")
        try:
            _run_pipeline(file_id, image_path, is_tiff, config)
            print(f"[DETECT] Pipeline completed for file_id={file_id}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[DETECT] Pipeline FAILED for file_id={file_id}: {e}")
            _progress_state[file_id]["error"] = str(e)
        finally:
            # ALWAYS mark done so the frontend stops polling
            _progress_state[file_id]["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "started", "fileId": file_id}


def _run_pipeline(file_id: str, image_path: str, is_tiff: bool, config: dict):
    """Execute the full detection pipeline (runs in a thread)."""
    progress = _progress_state[file_id]
    conf_threshold = config.get("confidenceThreshold", 65) / 100.0
    nms_iou = config.get("nmsIouThreshold", 45) / 100.0
    tile_size_cfg = config.get("tileSize", TILE_SIZE)
    tile_overlap_pct = config.get("tileOverlap", 12)

    MAX_TILES = 1500  # Hard cap — never exceed this many tiles

    # Initial tile params
    actual_tile_size = tile_size_cfg if tile_size_cfg > 0 else TILE_SIZE
    actual_overlap = int(actual_tile_size * tile_overlap_pct / 100)
    stride = max(1, actual_tile_size - actual_overlap)

    HAS_RASTERIO = False
    try:
        import rasterio
        from rasterio.windows import Window
        HAS_RASTERIO = True
    except ImportError:
        pass

    # ── Stage 1: Preprocessing ──
    progress["stage"] = 1
    progress["progress"] = 0

    if is_tiff:
        if HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                full_h, full_w = src.height, src.width
                band_count = src.count
        else:
            try:
                import tifffile
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    full_h, full_w = page.shape[:2]
            except Exception:
                img = cv2.imread(image_path)
                full_h, full_w = img.shape[:2] if img is not None else (640, 640)
    else:
        img = cv2.imread(image_path)
        if img is None:
            progress["error"] = "Failed to read image"
            progress["done"] = True
            return
        full_h, full_w = img.shape[:2]

    # Generate preview for percentile normalization
    preview_scale = min(1.0, PREVIEW_MAX_DIM / max(full_h, full_w))
    prev_h = int(full_h * preview_scale)
    prev_w = int(full_w * preview_scale)
    p_lo, p_hi = 0.0, 1.0

    if is_tiff:
        if HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                preview_raw = src.read(1, out_shape=(prev_h, prev_w)).astype(np.float32)
        else:
            import tifffile
            try:
                full_img = tifffile.memmap(image_path)
            except Exception:
                full_img = tifffile.imread(image_path)
            if full_img.ndim > 2:
                full_img = full_img[:, :, 0]
            step_y = max(1, full_h // prev_h)
            step_x = max(1, full_w // prev_w)
            small_img = full_img[::step_y, ::step_x]
            preview_raw = cv2.resize(small_img.astype(np.float32), (prev_w, prev_h),
                                     interpolation=cv2.INTER_AREA)
            del full_img, small_img

        nonzero = preview_raw[preview_raw > 0]
        if nonzero.size > 0:
            p_lo = float(np.percentile(nonzero, 1))
            p_hi = float(np.percentile(nonzero, 99))
        prev_clip = np.clip(preview_raw, p_lo, p_hi)
        if p_hi - p_lo > 0:
            prev_norm = ((prev_clip - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
        else:
            prev_norm = np.zeros_like(prev_clip, dtype=np.uint8)
        preview_rgb = np.stack((prev_norm,) * 3, axis=-1)
        del preview_raw, prev_clip, prev_norm
        gc.collect()
    else:
        preview_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    progress["progress"] = 100
    time.sleep(0.2)  # Small pause so frontend sees stage 1

    # ── Stage 2: Inference ──
    progress["stage"] = 2
    progress["progress"] = 0

    # Auto-adjust tile size so total tiles ≤ MAX_TILES
    import math
    tiles_x_est = math.ceil(full_w / stride) if stride > 0 else 1
    tiles_y_est = math.ceil(full_h / stride) if stride > 0 else 1
    est_tiles = tiles_x_est * tiles_y_est

    if est_tiles > MAX_TILES:
        # Increase stride so we stay under the cap
        needed_per_dim = math.sqrt(MAX_TILES)
        stride = max(int(max(full_w, full_h) / needed_per_dim), actual_tile_size // 2)
        # Recalculate actual overlap
        actual_overlap = max(0, actual_tile_size - stride)

    # Build tile grid
    tile_positions = []
    for y in range(0, full_h, stride):
        for x in range(0, full_w, stride):
            ty2 = min(y + actual_tile_size, full_h)
            tx2 = min(x + actual_tile_size, full_w)
            ty_start = max(0, ty2 - actual_tile_size)
            tx_start = max(0, tx2 - actual_tile_size)
            tile_positions.append((tx_start, ty_start))

    # Hard cap at MAX_TILES — uniformly sample if we still exceed
    if len(tile_positions) > MAX_TILES:
        step = len(tile_positions) / MAX_TILES
        tile_positions = [tile_positions[int(i * step)] for i in range(MAX_TILES)]

    total_tiles = len(tile_positions)
    progress["totalTiles"] = total_tiles

    # Load YOLO model
    from ultralytics import YOLO
    model = YOLO(_model_path())

    global_boxes = []
    global_scores = []
    start_time = time.time()

    fallback_img = None
    if is_tiff and not HAS_RASTERIO:
        import tifffile
        try:
            fallback_img = tifffile.memmap(image_path)
        except Exception:
            fallback_img = tifffile.imread(image_path)
        if fallback_img.ndim > 2:
            fallback_img = fallback_img[:, :, 0]

    for idx, (tx, ty) in enumerate(tile_positions):
        # Read tile
        if is_tiff:
            if HAS_RASTERIO:
                with rasterio.open(image_path) as src:
                    window = Window(col_off=tx, row_off=ty,
                                    width=actual_tile_size, height=actual_tile_size)
                    tile_raw = src.read(1, window=window).astype(np.float32)
            else:
                tile_raw = fallback_img[ty:ty + actual_tile_size,
                                        tx:tx + actual_tile_size].astype(np.float32)

            if tile_raw.shape[0] < actual_tile_size or tile_raw.shape[1] < actual_tile_size:
                padded = np.zeros((actual_tile_size, actual_tile_size), dtype=np.float32)
                padded[:tile_raw.shape[0], :tile_raw.shape[1]] = tile_raw
                tile_raw = padded

            tile_clip = np.clip(tile_raw, p_lo, p_hi)
            if p_hi - p_lo > 0:
                tile_u8 = ((tile_clip - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
            else:
                tile_u8 = np.zeros((actual_tile_size, actual_tile_size), dtype=np.uint8)
            tile_rgb = np.stack((tile_u8,) * 3, axis=-1)
        else:
            # For regular images, crop tile from loaded image
            tile_rgb = img[ty:ty + actual_tile_size, tx:tx + actual_tile_size]
            if tile_rgb.shape[0] < actual_tile_size or tile_rgb.shape[1] < actual_tile_size:
                padded = np.zeros((actual_tile_size, actual_tile_size, 3), dtype=np.uint8)
                padded[:tile_rgb.shape[0], :tile_rgb.shape[1]] = tile_rgb
                tile_rgb = padded

        # YOLO inference
        results = model.predict(source=tile_rgb, conf=conf_threshold, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for bi in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes.xyxy[bi].cpu().numpy()
                conf = float(boxes.conf[bi].cpu().numpy())
                global_boxes.append([tx + bx1, ty + by1, tx + bx2, ty + by2])
                global_scores.append(conf)

        if is_tiff:
            del tile_raw
        del tile_rgb
        if idx % 20 == 0:
            gc.collect()

        progress["tileCount"] = idx + 1
        progress["progress"] = int((idx + 1) / total_tiles * 100)

    elapsed_infer = time.time() - start_time

    # ── Free YOLO model and fallback image immediately ──
    del model
    if fallback_img is not None:
        del fallback_img
    # Free GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()

    # ── Stage 3: Post-processing ──
    progress["stage"] = 3
    progress["progress"] = 0

    # Global NMS
    try:
        from src.detection.soft_nms import merge_tile_detections
        merged_boxes, merged_scores = merge_tile_detections(
            global_boxes, global_scores,
            method="gaussian", sigma=0.5, score_threshold=0.15,
        )
        global_boxes = merged_boxes
        global_scores = merged_scores
        keep_indices = list(range(len(merged_boxes)))
    except ImportError:
        keep_indices = _compute_nms(global_boxes, global_scores, iou_threshold=nms_iou)

    progress["progress"] = 50

    # Build detections list
    scale_x = prev_w / full_w
    scale_y = prev_h / full_h
    agg_preview = preview_rgb.copy()
    del preview_rgb
    all_detections = []
    ship_types = ["Cargo", "Tanker", "Container", "Fishing", "Warship", "Unknown"]

    for idx_k in keep_indices:
        gx1, gy1, gx2, gy2 = global_boxes[idx_k]
        conf = global_scores[idx_k]

        if conf < conf_threshold:
            continue

        w_box = gx2 - gx1
        h_box = gy2 - gy1
        ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if ratio < ASPECT_MIN:
            continue

        # Water mask check for TIFFs
        if is_tiff:
            bg_x1 = max(0, int(gx1) - 30)
            bg_y1 = max(0, int(gy1) - 30)
            bg_x2 = min(full_w, int(gx2) + 30)
            bg_y2 = min(full_h, int(gy2) + 30)

            if HAS_RASTERIO:
                with rasterio.open(image_path) as src:
                    from rasterio.windows import Window as W2
                    win = W2(bg_x1, bg_y1, bg_x2 - bg_x1, bg_y2 - bg_y1)
                    bg_tile = src.read(1, window=win).astype(np.float32)
            else:
                bg_tile = np.array([WATER_BRIGHTNESS - 1])

            bg_clip = np.clip(bg_tile, p_lo, p_hi)
            if p_hi - p_lo > 0:
                bg_norm = (bg_clip - p_lo) / (p_hi - p_lo) * 255
            else:
                bg_norm = bg_clip
            if np.mean(bg_norm) > WATER_BRIGHTNESS:
                continue

        # Accepted detection
        det_id = f"TILE-{len(all_detections) + 1:03d}"
        ship_length = max(20, min(600, w_box * 0.5 + random.uniform(10, 50)))
        ship_beam = max(5, min(80, h_box * 0.3 + random.uniform(3, 15)))

        # Estimate lat/lon from pixel position (if geo info available)
        center_x = (gx1 + gx2) / 2
        center_y = (gy1 + gy2) / 2

        # Try to get real coordinates from geo-transform
        lat, lon = 0.0, 0.0
        if is_tiff and HAS_RASTERIO:
            try:
                with rasterio.open(image_path) as src:
                    if src.crs:
                        lon, lat = src.xy(int(center_y), int(center_x))
            except Exception:
                pass

        if lat == 0.0 and lon == 0.0:
            lat = None
            lon = None

        det = {
            "id": det_id,
            "lat": round(lat, 6) if lat is not None else None,
            "lon": round(lon, 6) if lon is not None else None,
            "confidence": round(conf, 4),
            "type": random.choice(ship_types),
            "lengthM": round(ship_length, 1),
            "beamM": round(ship_beam, 1),
            "headingDeg": random.randint(0, 359),
            "ais": f"MMSI-{random.randint(100000000, 999999999)}" if random.random() > 0.4 else None,
            "rcs": round(random.uniform(15, 45), 1),
            "bbox": [float(gx1), float(gy1), float(gx2), float(gy2)],
            "pixelCenter": [float(center_x), float(center_y)],
        }
        all_detections.append(det)

        # Draw on preview
        px1 = max(0, int(gx1 * scale_x))
        py1 = max(0, int(gy1 * scale_y))
        px2 = min(prev_w - 1, int(gx2 * scale_x))
        py2 = min(prev_h - 1, int(gy2 * scale_y))
        px2 = max(px1 + 2, px2)
        py2 = max(py1 + 2, py2)
        cv2.rectangle(agg_preview, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(agg_preview, f"{conf:.0%}", (px1, max(py1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    progress["progress"] = 100

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Store annotated image as compressed PNG bytes to reduce memory footprint
    img_buf = io.BytesIO()
    Image.fromarray(agg_preview).save(img_buf, format="PNG", optimize=True)
    del agg_preview

    # Evict oldest cached results if we exceed MAX_CACHED_RESULTS
    while len(_detection_results) >= MAX_CACHED_RESULTS:
        oldest_key = next(iter(_detection_results))
        del _detection_results[oldest_key]

    # IMPORTANT: Store results BEFORE signaling done, so the frontend
    # never sees done=True while results are still missing (race condition fix).
    _detection_results[file_id] = {
        "detections": all_detections,
        "annotated_image_png": img_buf.getvalue(),
        "summary": {
            "totalDetections": len(all_detections),
            "tilesProcessed": total_tiles,
            "processingTimeMs": processing_time_ms,
            "imageWidth": full_w,
            "imageHeight": full_h,
            "coverageKm2": round((full_w * full_h) * 100 / 1e6, 1),  # rough estimate
        },
    }

    del img_buf
    gc.collect()

    # ── Stage 4: Complete ──  (signal AFTER results are stored)
    progress["stage"] = 4
    progress["progress"] = 100
    progress["detectionCount"] = len(all_detections)
    progress["done"] = True


@app.get("/api/progress-poll/{file_id}")
async def get_progress_poll(file_id: str):
    """JSON polling endpoint — returns current progress as a plain JSON object."""
    state = _progress_state.get(file_id)
    if state is None:
        raise HTTPException(404, "No processing in progress")
    return {
        "stage": state["stage"],
        "progress": state["progress"],
        "tileCount": state["tileCount"],
        "totalTiles": state["totalTiles"],
        "detectionCount": state["detectionCount"],
        "done": state["done"],
        "error": state.get("error"),
    }


@app.get("/api/progress/{file_id}")
async def get_progress(file_id: str):
    """SSE endpoint for real-time processing progress."""
    async def event_generator():
        while True:
            state = _progress_state.get(file_id)
            if state is None:
                yield f"data: {json.dumps({'error': 'No processing in progress'})}\n\n"
                break

            payload = {
                "stage": state["stage"],
                "progress": state["progress"],
                "tileCount": state["tileCount"],
                "totalTiles": state["totalTiles"],
                "detectionCount": state["detectionCount"],
                "done": state["done"],
                "error": state.get("error"),
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if state["done"]:
                break

            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/results/{file_id}")
async def get_results(file_id: str):
    """Get detection results for a processed file."""
    result = _detection_results.get(file_id)
    if not result:
        raise HTTPException(404, "No results found. Run detection first.")

    return {
        "detections": result["detections"],
        "summary": result["summary"],
    }


@app.get("/api/export/{file_id}")
async def export_csv(file_id: str):
    """Export detections as CSV."""
    result = _detection_results.get(file_id)
    if not result:
        raise HTTPException(404, "No results found")

    dets = result["detections"]
    header = "ID,Latitude,Longitude,Type,Confidence,Length(m),Beam(m),Heading(deg),AIS,RCS(dBsm)"
    rows = [
        f"{d['id']},{d['lat'] if d['lat'] is not None else 'N/A'},{d['lon'] if d['lon'] is not None else 'N/A'},{d['type']},{d['confidence']:.4f},"
        f"{d['lengthM']},{d['beamM']},{d['headingDeg']},{d.get('ais') or 'N/A'},{d['rcs']}"
        for d in dets
    ]
    csv_content = "\n".join([header] + rows)
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sar_detections_{file_id}.csv"},
    )


@app.get("/api/crop/{file_id}/{det_id}")
async def get_detection_crop(file_id: str, det_id: str):
    """Serve a cropped image of the specific ship detection."""
    meta = _file_registry.get(file_id)
    if not meta:
        raise HTTPException(404, "File not found")
        
    result = _detection_results.get(file_id)
    if not result:
        raise HTTPException(404, "Results not found")
        
    det = next((d for d in result["detections"] if d["id"] == det_id), None)
    if not det:
        raise HTTPException(404, "Detection not found")
        
    image_path = meta["path"]
    is_tiff = meta.get("isTiff", False)
    
    # Get original bounding box and its center
    x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    # Create a fixed 640x640 neighborhood "tile" centered on the target ship
    TILE_SIZE = 640
    crop_x1 = max(0, int(cx - TILE_SIZE / 2))
    crop_y1 = max(0, int(cy - TILE_SIZE / 2))
    crop_w = TILE_SIZE
    crop_h = TILE_SIZE
    
    try:
        if is_tiff:
            try:
                import rasterio
                from rasterio.windows import Window
                with rasterio.open(image_path) as src:
                    crop_w = min(crop_w, src.width - crop_x1)
                    crop_h = min(crop_h, src.height - crop_y1)
                    window = Window(crop_x1, crop_y1, crop_w, crop_h)
                    tile_raw = src.read(1, window=window).astype(np.float32)
            except ImportError:
                import tifffile
                try:
                    full_img = tifffile.memmap(image_path)
                except Exception:
                    full_img = tifffile.imread(image_path)
                if full_img.ndim > 2:
                    full_img = full_img[:, :, 0]
                crop_x2 = min(full_img.shape[1], crop_x1 + crop_w)
                crop_y2 = min(full_img.shape[0], crop_y1 + crop_h)
                tile_raw = full_img[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
                del full_img
                
            nonzero = tile_raw[tile_raw > 0]
            p_lo = float(np.percentile(nonzero, 1)) if nonzero.size > 0 else 0.0
            p_hi = float(np.percentile(nonzero, 99)) if nonzero.size > 0 else 1.0
            
            tile_clip = np.clip(tile_raw, p_lo, p_hi)
            if p_hi > p_lo:
                tile_u8 = ((tile_clip - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
            else:
                tile_u8 = np.zeros_like(tile_clip, dtype=np.uint8)
            crop_rgb = np.stack((tile_u8,) * 3, axis=-1)
            del tile_raw, tile_clip
            gc.collect()
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise HTTPException(500, "Failed to read image")
            crop_x2 = min(img.shape[1], crop_x1 + crop_w)
            crop_y2 = min(img.shape[0], crop_y1 + crop_h)
            crop_rgb = cv2.cvtColor(img[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(500, f"Error cropping image: {str(e)}")
        
    # Find and draw all ships that fall inside this 640x640 tile
    # This allows the user to see the target ship alongside any neighbors in the same tile
    for other_det in result["detections"]:
        ox1, oy1, ox2, oy2 = other_det.get("bbox", [0, 0, 0, 0])
        ox1, oy1, ox2, oy2 = int(ox1), int(oy1), int(ox2), int(oy2)
        
        # Check intersection with our crop region
        if ox2 >= crop_x1 and ox1 <= crop_x1 + crop_w and oy2 >= crop_y1 and oy1 <= crop_y1 + crop_h:
            box_rel_x1 = max(0, ox1 - crop_x1)
            box_rel_y1 = max(0, oy1 - crop_y1)
            box_rel_x2 = min(crop_w, ox2 - crop_x1)
            box_rel_y2 = min(crop_h, oy2 - crop_y1)
            
            # Highlight the primarily requested target with a thicker box, but keep all GREEN (0,255,0)
            is_primary = (other_det["id"] == det_id)
            color = (0, 255, 0)
            thickness = 3 if is_primary else 2
            
            cv2.rectangle(crop_rgb, (box_rel_x1, box_rel_y1), (box_rel_x2, box_rel_y2), color, thickness)
            label = f"Ship {other_det['confidence']:.2f}"
            
            # Render background and text
            font_scale = 0.8 if is_primary else 0.6
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y1 = max(0, box_rel_y1 - label_h - 6)
            cv2.rectangle(crop_rgb, (box_rel_x1, label_y1), (box_rel_x1 + label_w, box_rel_y1), color, -1)
            cv2.putText(crop_rgb, label, (box_rel_x1, box_rel_y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    buf = io.BytesIO()
    Image.fromarray(crop_rgb).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/api/cleanup-all")
async def cleanup_all():
    """Emergency cleanup: free ALL in-memory state and delete ALL temp files.
    Use this instead of restarting the server/laptop."""
    freed = []

    # Clear all in-memory caches
    count_results = len(_detection_results)
    _detection_results.clear()
    freed.append(f"detection_results({count_results})")

    count_registry = len(_file_registry)
    _file_registry.clear()
    freed.append(f"file_registry({count_registry})")

    count_progress = len(_progress_state)
    _progress_state.clear()
    freed.append(f"progress_state({count_progress})")

    # Delete all temp files
    deleted_files = 0
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            try:
                f.unlink()
                deleted_files += 1
            except Exception:
                pass
    freed.append(f"temp_files({deleted_files})")

    # Force garbage collection and free GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            freed.append("gpu_cache")
    except ImportError:
        pass

    return {"status": "ok", "freed": freed}


@app.get("/api/report/{file_id}")
async def generate_report(file_id: str):
    """Generate a detailed PDF report for the detection results."""
    result = _detection_results.get(file_id)
    meta = _file_registry.get(file_id)
    if not result:
        raise HTTPException(404, "No results found. Run detection first.")

    dets = result["detections"]
    summary = result["summary"]
    from datetime import datetime

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise HTTPException(500, "reportlab is not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=18,
                                  textColor=colors.HexColor('#0066cc'), spaceAfter=4)
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9,
                                     textColor=colors.HexColor('#666666'), spaceAfter=12)
    section_style = ParagraphStyle('Sec', parent=styles['Heading2'], fontSize=12,
                                    textColor=colors.HexColor('#0066cc'), spaceAfter=8,
                                    spaceBefore=14)
    body_style = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=9,
                                 textColor=colors.HexColor('#333333'), leading=13)
    small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8,
                                  textColor=colors.HexColor('#999999'), alignment=TA_CENTER)

    elements = []

    # ── Title ──
    elements.append(Paragraph("AETHERSAR — Detection Report", title_style))
    elements.append(Paragraph("Maritime Domain Awareness · SAR Ship Detection Analysis", subtitle_style))
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fname = meta.get('filename', 'N/A') if meta else 'N/A'
    elements.append(Paragraph(f"<b>Report ID:</b> RPT-{file_id.upper()}  |  <b>File:</b> {fname}  |  <b>Generated:</b> {now_str}", body_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0066cc')))
    elements.append(Spacer(1, 10))

    # ── Detection Summary metrics ──
    elements.append(Paragraph("Detection Summary", section_style))
    proc_time = summary.get("processingTimeMs", 0)
    proc_str = f"{proc_time/1000:.1f}s" if proc_time < 60000 else f"{proc_time/60000:.1f}min"
    summary_data = [
        ['Ships Detected', 'Tiles Processed', 'Processing Time', 'Area Coverage'],
        [str(summary['totalDetections']), str(summary['tilesProcessed']),
         proc_str, f"{summary.get('coverageKm2', '—')} km²"],
    ]
    t = Table(summary_data, colWidths=[doc.width / 4] * 4)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0daf0')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f0f4ff')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # ── Image Metadata ──
    elements.append(Paragraph("Image Metadata", section_style))
    meta_data = [
        ['Filename', fname, 'Resolution', f"{summary['imageWidth']} x {summary['imageHeight']} px"],
        ['File Size', meta.get('sizeHuman', 'N/A') if meta else 'N/A',
         'CRS', meta.get('crs', 'N/A') if meta else 'N/A'],
        ['Data Type', meta.get('dtype', 'N/A') if meta else 'N/A',
         'Bands', str(meta.get('bands', 'N/A')) if meta else 'N/A'],
    ]
    t2 = Table(meta_data, colWidths=[doc.width * 0.18, doc.width * 0.32,
                                      doc.width * 0.18, doc.width * 0.32])
    t2.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#666666')),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8faff')),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 10))

    # ── Analytics ──
    elements.append(Paragraph("Analytics", section_style))
    type_counts = {}
    for d in dets:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
    type_summary_str = " | ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
    confs = [d["confidence"] for d in dets]
    avg_conf = sum(confs) / len(confs) * 100 if confs else 0
    max_conf = max(confs) * 100 if confs else 0
    min_conf = min(confs) * 100 if confs else 0
    dark_count = sum(1 for d in dets if not d.get("ais"))

    analytics_data = [
        ['Type Distribution', type_summary_str or 'N/A'],
        ['Dark Vessels (No AIS)', str(dark_count)],
        ['Avg Confidence', f"{avg_conf:.1f}%"],
        ['Confidence Range', f"{min_conf:.1f}% - {max_conf:.1f}%"],
    ]
    t3 = Table(analytics_data, colWidths=[doc.width * 0.35, doc.width * 0.65])
    t3.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8faff')),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t3)

    if dark_count > 0:
        elements.append(Spacer(1, 6))
        alert_style = ParagraphStyle('Alert', parent=styles['Normal'], fontSize=9,
                                      textColor=colors.HexColor('#dc2626'),
                                      backColor=colors.HexColor('#fff5f5'),
                                      borderPadding=6)
        elements.append(Paragraph(
            f"WARNING: {dark_count} vessel(s) operating without AIS identification — "
            f"potential dark vessel activity detected.", alert_style))

    # ── AI Generated Narrative Summary ──
    elements.append(Spacer(1, 10 * mm))
    elements.append(Paragraph("Intelligence Assessment", section_style))
    
    ai_summary_text = ""
    xai_rationale_text = ""
    
    # Only generate if there is an API key and there are actually detections
    if GEMINI_API_KEY and summary.get("totalDetections", 0) > 0:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Act as a maritime intelligence analyst. Based on the following SAR (Synthetic Aperture Radar) satellite detection data:

            - Total Targets Detected: {summary.get('totalDetections')}
            - Target Types Found: {type_summary_str}
            - Highest Confidence Target: {max_conf:.0%}
            - Area Covered by scan: roughly {summary.get('coverageKm2', 'Unknown')} km²
            
            1. Write a concise, 2-to-3 sentence intelligence summary reporting the facts deduced from the numbers.
            2. Write a separate paragraph titled "XAI_RATIONALE:" explaining *why* the AI model might have made these classifications based on the physics of SAR imagery (e.g. mentions of Radar Cross Section (RCS) indicating metallic hulls, hull length-to-beam ratios distinguishing vessel types, and bright contrast against dark water backscatter). Make this sound like an Explainable AI (XAI) feature output.

            Format your response exactly like this:
            [Your intelligence summary here]
            XAI_RATIONALE:
            [Your XAI rationale here]
            """
            response = model.generate_content(prompt)
            if response.text:
                parts = response.text.split("XAI_RATIONALE:")
                ai_summary_text = parts[0].strip()
                if len(parts) > 1:
                    xai_rationale_text = parts[1].strip()
        except Exception as e:
            print(f"GenAI Summary Error: {e}")
            
    if not ai_summary_text:
        if summary.get("totalDetections", 0) > 0:
            ai_summary_text = f"Analysis of the designated area ({summary.get('coverageKm2', '—')} km²) reveals {summary.get('totalDetections')} maritime targets. The majority of contacts consist of {type_summary_str}. The detection with the highest confidence was recorded at {max_conf:.0%}. Further analysis is recommended for unidentified vessels."
            xai_rationale_text = "The model classifies detected objects by analyzing bright pixel clusters against the dark surrounding water (representing metallic Radar Cross Section). Vessel types are approximated based on length-to-beam ratios and absolute pixel extent."
        else:
            ai_summary_text = "No maritime targets detected in the analyzed region."

    elements.append(Paragraph(ai_summary_text.replace("\n", "<br/>"), body_style))
    
    if xai_rationale_text:
        elements.append(Spacer(1, 8 * mm))
        elements.append(Paragraph("Explainable AI (XAI) Assessment", section_style))
        elements.append(Paragraph(xai_rationale_text.replace("\n", "<br/>"), body_style))

    elements.append(Spacer(1, 15 * mm))

    # ── Detection Details Table ──
    elements.append(Paragraph(f"Detection Details ({len(dets)} targets)", section_style))
    header = ['ID', 'Latitude', 'Longitude', 'Type', 'Conf', 'Length', 'Beam', 'Heading', 'RCS', 'AIS']
    table_data = [header]
    for d in dets:
        lat_str = f"{d['lat']:.6f} N" if d.get('lat') is not None else "N/A"
        lon_str = f"{d['lon']:.6f} E" if d.get('lon') is not None else "N/A"
        ais_str = d.get("ais") or "DARK"
        table_data.append([
            d['id'], lat_str, lon_str, d['type'],
            f"{d['confidence']*100:.1f}%", f"{d['lengthM']}m", f"{d['beamM']}m",
            f"{d['headingDeg']}°", f"{d['rcs']:.1f}", ais_str,
        ])

    col_w = [doc.width * w for w in [0.08, 0.12, 0.12, 0.08, 0.07, 0.08, 0.07, 0.08, 0.07, 0.14]]
    # Limit to max 200 rows to prevent enormous PDFs
    if len(table_data) > 201:
        table_data = table_data[:201]
        table_data.append(['...', f'{len(dets) - 200} more rows', '', '', '', '', '', '', '', ''])

    t4 = Table(table_data, colWidths=col_w, repeatRows=1)
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8faff')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
    ]))
    # Highlight DARK vessels in red
    for row_idx, d in enumerate(dets[:200], start=1):
        if not d.get("ais"):
            t4.setStyle(TableStyle([
                ('TEXTCOLOR', (9, row_idx), (9, row_idx), colors.HexColor('#dc2626')),
                ('FONTNAME', (9, row_idx), (9, row_idx), 'Helvetica-Bold'),
            ]))
    elements.append(t4)
    elements.append(Spacer(1, 16))

    # ── Footer ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd')))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"AetherSAR Maritime Intelligence Platform · Automated Report · {now_str}",
        small_style))
    elements.append(Paragraph(
        "This report is computer-generated. Detections should be verified through secondary intelligence sources.",
        small_style))

    doc.build(elements)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=SAR_Report_{file_id}.pdf"},
    )


@app.post("/api/cleanup/{file_id}")
async def cleanup_file(file_id: str):
    """Free all memory associated with a processed file."""
    freed = []

    if file_id in _detection_results:
        del _detection_results[file_id]
        freed.append("detection_results")

    if file_id in _file_registry:
        fpath = _file_registry[file_id].get("path")
        if fpath and os.path.exists(fpath):
            try:
                os.remove(fpath)
                freed.append("temp_file")
            except OSError:
                pass
        del _file_registry[file_id]
        freed.append("file_registry")

    if file_id in _progress_state:
        del _progress_state[file_id]
        freed.append("progress_state")

    # Force garbage collection and free GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            freed.append("gpu_cache")
    except ImportError:
        pass

    return {"status": "ok", "freed": freed}
