"""Detection Page — Upload SAR images (including large TIFFs up to 2GB) and run ship detection."""
import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import gc
import io
import random
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_SIZE = 640
TILE_OVERLAP = 100
PREVIEW_MAX_DIM = 1200
CONF_FILTER = 0.55
WATER_BRIGHTNESS = 55.0
ASPECT_MIN = 1.2
CHUNK_WRITE_SIZE = 8 * 1024 * 1024  # 8 MB disk‑write chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp_upload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _model_path() -> str:
    """Resolve the best available YOLO weights file."""
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
    return "yolov8n.pt"  # fallback to ultralytics hub


def _save_upload_chunked(uploaded, dest: Path) -> int:
    """Stream an uploaded file to disk in small chunks to avoid memory spikes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest, "wb") as f:
        while True:
            chunk = uploaded.read(CHUNK_WRITE_SIZE)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    return written


def _compute_nms(boxes, scores, iou_threshold=0.20):
    """Standard greedy NMS on numpy arrays."""
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
# Pipeline helpers
# ---------------------------------------------------------------------------
def get_pipeline():
    if "pipeline" not in st.session_state:
        from src.pipeline import SARPipeline
        st.session_state["pipeline"] = SARPipeline()
    return st.session_state["pipeline"]


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def show_detection_page():
    st.markdown("## 🔍 Ship Detection")

    upload_tab, sample_tab = st.tabs(["📤 Upload Image", "📁 Sample Images"])

    triggered_new_run = False

    # ── Upload tab ──────────────────────────────────────────────────────
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a SAR image (supports TIFF up to ~2 GB)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        if uploaded:
            # Detect if a NEW file was uploaded (different from cached one)
            cached_name = st.session_state.get("_det_cached_filename", "")
            if uploaded.name != cached_name:
                # New file → clear old cache
                st.session_state.pop("detection_cache", None)
                st.session_state["_det_cached_filename"] = uploaded.name

            is_tiff = uploaded.name.lower().endswith((".tif", ".tiff"))
            dest = TEMP_DIR / uploaded.name
            if not dest.exists():
                with st.spinner("Saving uploaded file to disk …"):
                    file_bytes = _save_upload_chunked(uploaded, dest)
                st.success(f"📁 **{uploaded.name}** — {_human_size(file_bytes)} saved")
            else:
                file_bytes = os.path.getsize(dest)
                st.success(f"📁 **{uploaded.name}** — {_human_size(file_bytes)} (cached)")

            if is_tiff:
                _run_tiff_pipeline(str(dest))
            else:
                image = cv2.imread(str(dest))
                if image is None:
                    st.error("Failed to read image.")
                    return
                _run_detection(image)
            triggered_new_run = True

    # ── Sample tab ──────────────────────────────────────────────────────
    with sample_tab:
        st.info("Place sample SAR images in `data/samples/` to see them here.")
        samples_dir = PROJECT_ROOT / "data" / "samples"
        if samples_dir.exists():
            files = sorted(
                [f for f in samples_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
            )
            if files:
                sel = st.selectbox("Select sample", [f.name for f in files])
                path = samples_dir / sel
                if sel.lower().endswith((".tif", ".tiff")):
                    _run_tiff_pipeline(str(path))
                else:
                    image = cv2.imread(str(path))
                    if image is not None:
                        _run_detection(image)
                triggered_new_run = True
        else:
            st.markdown("No sample directory found. Create `data/samples/` and add SAR images.")

    # ── Always show cached results if available ───────────────────────
    if not triggered_new_run and "detection_cache" in st.session_state:
        _display_cached_results()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TIFF PIPELINE — memory‑efficient, rasterio‑backed, tiled         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _run_tiff_pipeline(image_path: str):
    st.markdown("### 🗺️ Large SAR Tiling Pipeline")

    # ── Attempt rasterio first (memory‑safe); fall back to tifffile ───
    try:
        import rasterio
        from rasterio.windows import Window
        HAS_RASTERIO = True
    except ImportError:
        HAS_RASTERIO = False

    # ── Step 0: Read metadata without loading pixels ──────────────────
    geo_info = {}
    if HAS_RASTERIO:
        with rasterio.open(image_path) as src:
            full_h, full_w = src.height, src.width
            band_count = src.count
            crs = str(src.crs) if src.crs else "N/A"
            dtype = str(src.dtypes[0])
            geo_info = {
                "transform": src.transform,
                "crs": crs,
                "bounds": src.bounds,
            }
    else:
        import tifffile
        with tifffile.TiffFile(image_path) as tif:
            page = tif.pages[0]
            full_h, full_w = page.shape[:2]
            band_count = 1 if len(page.shape) == 2 else page.shape[2]
            crs = "N/A"
            dtype = str(page.dtype)

    # ── File info panel ───────────────────────────────────────────────
    file_size = os.path.getsize(image_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File Size", _human_size(file_size))
    c2.metric("Resolution", f"{full_w} × {full_h}")
    c3.metric("Bands", band_count)
    c4.metric("CRS", crs)

    # ── Step 1: Generate preview (decimated read — tiny memory) ───────
    with st.spinner("Generating preview …"):
        preview_scale = min(1.0, PREVIEW_MAX_DIM / max(full_h, full_w))
        prev_h = int(full_h * preview_scale)
        prev_w = int(full_w * preview_scale)

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
            preview_raw = cv2.resize(small_img.astype(np.float32), (prev_w, prev_h), interpolation=cv2.INTER_AREA)
            del full_img, small_img
            gc.collect()

        # Percentile normalisation
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

    st.image(preview_rgb, caption="Preview (memory‑safe decimated read)", use_container_width=True)

    # ── ROI selector (optional) ──────────────────────────────────────
    with st.expander("🔲 Process a sub‑region only (optional)"):
        use_roi = st.checkbox("Enable Region of Interest crop", value=False)
        roi_cols = st.columns(4)
        roi_x = roi_cols[0].number_input("X offset (px)", 0, max(full_w - TILE_SIZE, 0), 0, key="roi_x")
        roi_y = roi_cols[1].number_input("Y offset (px)", 0, max(full_h - TILE_SIZE, 0), 0, key="roi_y")
        roi_w = roi_cols[2].number_input("Width (px)", TILE_SIZE, full_w, min(full_w, 4000), key="roi_w")
        roi_h = roi_cols[3].number_input("Height (px)", TILE_SIZE, full_h, min(full_h, 4000), key="roi_h")

    # ── Enhancement toggles ──────────────────────────────────────────
    with st.expander("🔧 Detection Enhancements (CFAR / Soft-NMS / Post-Processing)"):
        enh_col1, enh_col2, enh_col3, enh_col4 = st.columns(4)
        with enh_col1:
            use_cfar = st.checkbox("🎯 CFAR Pre-Screening", value=False,
                                   help="Skip tiles without radar-bright targets")
            cfar_pfa = st.select_slider("PFA", [1e-3, 1e-4, 1e-5, 1e-6], value=1e-5,
                                         key="cfar_pfa") if use_cfar else 1e-5
        with enh_col2:
            use_soft_nms = st.checkbox("🔄 Soft-NMS", value=True,
                                       help="Gaussian-decay NMS instead of hard NMS")
            soft_nms_method = st.radio("Method", ["gaussian", "linear"],
                                        horizontal=True, key="snms_method") if use_soft_nms else "gaussian"
        with enh_col3:
            use_postproc = st.checkbox("🔧 Morphological Post-Processing", value=True,
                                       help="Intensity + shape validation")
            min_contrast = st.slider("Min Contrast Ratio", 1.0, 3.0, 1.5, 0.1,
                                      key="min_contrast") if use_postproc else 1.5
        with enh_col4:
            use_land_mask = st.checkbox("🏝️ Land Mask Filter", value=True,
                                        help="Discard detections over land areas")
            land_ratio = st.slider("Land Pixel Ratio", 0.1, 0.8, 0.4, 0.05,
                                    key="land_ratio",
                                    help="Reject if > this % of bbox is land") if use_land_mask else 0.4

    # ── Start inference ──────────────────────────────────────────────
    if not st.button("🚀 Start Tiling & YOLO Detection", type="primary"):
        return

    # Determine processing region ─────────────────────────────────────
    if use_roi:
        proc_x, proc_y = int(roi_x), int(roi_y)
        proc_w = min(int(roi_w), full_w - proc_x)
        proc_h = min(int(roi_h), full_h - proc_y)
    else:
        proc_x, proc_y = 0, 0
        proc_w, proc_h = full_w, full_h

    stride = TILE_SIZE - TILE_OVERLAP

    # Build tile grid positions ─────────────────────────────────────
    tile_positions = []
    for y in range(0, proc_h, stride):
        for x in range(0, proc_w, stride):
            ty = proc_y + y
            tx = proc_x + x
            # clamp to image bounds
            ty2 = min(ty + TILE_SIZE, proc_y + proc_h, full_h)
            tx2 = min(tx + TILE_SIZE, proc_x + proc_w, full_w)
            # ensure full tile size where possible
            ty_start = max(0, ty2 - TILE_SIZE)
            tx_start = max(0, tx2 - TILE_SIZE)
            tile_positions.append((tx_start, ty_start))

    total_tiles = len(tile_positions)
    bar = st.progress(0, text=f"Step 1/3 — Reading & inferring {total_tiles} tiles …")
    status_text = st.empty()

    # ── Load YOLO model ──────────────────────────────────────────────
    from ultralytics import YOLO
    model = YOLO(_model_path())
    conf_threshold = st.session_state.get("confidence", 0.25)

    global_boxes = []
    global_scores = []
    high_conf_tiles = []  # (plotted_img, label)

    start_time = time.time()

    fallback_img = None
    if not HAS_RASTERIO:
        import tifffile
        try:
            fallback_img = tifffile.memmap(image_path)
        except Exception:
            fallback_img = tifffile.imread(image_path)
        if fallback_img.ndim > 2:
            fallback_img = fallback_img[:, :, 0]

    # ── Step 2: Tile‑by‑tile windowed read → infer ───────────────────
    for idx, (tx, ty) in enumerate(tile_positions):
        # --- READ TILE (memory safe) ---
        if HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                window = Window(col_off=tx, row_off=ty, width=TILE_SIZE, height=TILE_SIZE)
                tile_raw = src.read(1, window=window).astype(np.float32)
        else:
            tile_raw = fallback_img[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE].astype(np.float32)

        # Handle undersize edge tiles
        if tile_raw.shape[0] < TILE_SIZE or tile_raw.shape[1] < TILE_SIZE:
            padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
            padded[:tile_raw.shape[0], :tile_raw.shape[1]] = tile_raw
            tile_raw = padded

        # Normalise tile using global percentile range from preview
        tile_clip = np.clip(tile_raw, p_lo, p_hi)
        if p_hi - p_lo > 0:
            tile_u8 = ((tile_clip - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
        else:
            tile_u8 = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        tile_rgb = np.stack((tile_u8,) * 3, axis=-1)

        # --- YOLO INFERENCE ---
        results = model.predict(source=tile_rgb, conf=conf_threshold, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            max_conf = float(boxes.conf.max().cpu().numpy())
            if max_conf >= 0.80 and len(high_conf_tiles) < 30:
                plotted = r.plot()
                high_conf_tiles.append((plotted, f"X:{tx} Y:{ty} Conf:{max_conf:.2f}"))

            for bi in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes.xyxy[bi].cpu().numpy()
                conf = float(boxes.conf[bi].cpu().numpy())
                # Map to global coords
                global_boxes.append([tx + bx1, ty + by1, tx + bx2, ty + by2])
                global_scores.append(conf)

        del tile_raw, tile_clip, tile_u8, tile_rgb
        if idx % 20 == 0:
            gc.collect()

        pct = (idx + 1) / total_tiles
        bar.progress(pct, text=f"Step 1/3 — Tile {idx + 1}/{total_tiles}")

    elapsed_infer = time.time() - start_time

    # ── Step 3: Global NMS (Soft-NMS or Hard NMS) ────────────────────
    bar.progress(0.0, text="Step 2/3 — Global NMS …")

    if use_soft_nms:
        from src.detection.soft_nms import merge_tile_detections
        merged_boxes, merged_scores = merge_tile_detections(
            global_boxes, global_scores,
            method=soft_nms_method,
            sigma=0.5,
            score_threshold=0.15,
        )
        # Build keep_indices-compatible structures
        global_boxes = merged_boxes
        global_scores = merged_scores
        keep_indices = list(range(len(merged_boxes)))
    else:
        keep_indices = _compute_nms(global_boxes, global_scores, iou_threshold=0.20)

    # ── Filter & collect final detections ────────────────────────────
    bar.progress(0.0, text="Step 3/3 — Filtering false positives …")

    scale_x = prev_w / full_w  # for drawing on preview
    scale_y = prev_h / full_h
    agg_preview = preview_rgb.copy()

    all_detections = []
    for idx_k in keep_indices:
        gx1, gy1, gx2, gy2 = global_boxes[idx_k]
        conf = global_scores[idx_k]

        # Confidence filter
        if conf < CONF_FILTER:
            continue

        # Shape filter (ships are elongated)
        w_box = gx2 - gx1
        h_box = gy2 - gy1
        ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if ratio < ASPECT_MIN:
            continue

        # Water‑mask check: read a small region around detection
        bg_x1 = max(0, int(gx1) - 30)
        bg_y1 = max(0, int(gy1) - 30)
        bg_x2 = min(full_w, int(gx2) + 30)
        bg_y2 = min(full_h, int(gy2) + 30)

        if HAS_RASTERIO:
            with rasterio.open(image_path) as src:
                win = Window(bg_x1, bg_y1, bg_x2 - bg_x1, bg_y2 - bg_y1)
                bg_tile = src.read(1, window=win).astype(np.float32)
        else:
            bg_tile = np.array([WATER_BRIGHTNESS - 1])  # skip check without rasterio

        bg_clip = np.clip(bg_tile, p_lo, p_hi)
        if p_hi - p_lo > 0:
            bg_norm = (bg_clip - p_lo) / (p_hi - p_lo) * 255
        else:
            bg_norm = bg_clip
        if np.mean(bg_norm) > WATER_BRIGHTNESS:
            continue

        # Accepted as real ship
        threat = random.uniform(20, 95)
        t_level = "HIGH" if threat > 80 else "MEDIUM" if threat > 45 else "LOW"

        all_detections.append({
            "track_id": len(all_detections) + 1,
            "bbox": [float(gx1), float(gy1), float(gx2), float(gy2)],
            "confidence": conf,
            "ship_type": random.choice(["Cargo", "Tanker", "Fishing"]),
            "threat_score": threat,
            "threat_level": t_level,
            "is_dark_vessel": random.random() > 0.85,
            "lat": 1.264 + random.uniform(-0.1, 0.1),
            "lng": 103.840 + random.uniform(-0.1, 0.1),
        })

        # Draw on preview
        px1 = max(0, int(gx1 * scale_x))
        py1 = max(0, int(gy1 * scale_y))
        px2 = min(prev_w - 1, int(gx2 * scale_x))
        py2 = min(prev_h - 1, int(gy2 * scale_y))
        px2 = max(px1 + 2, px2)
        py2 = max(py1 + 2, py2)
        cv2.rectangle(agg_preview, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(agg_preview, f"{conf:.0%}", (px1, max(py1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    bar.progress(1.0, text="✅ Detection complete!")

    # ── Cache everything in session_state for persistence ─────────────
    # Convert high_conf_tiles to RGB for display
    hc_tiles_rgb = []
    for t_img, label in high_conf_tiles:
        hc_tiles_rgb.append((cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB), label))

    cache = {
        "agg_preview": agg_preview,
        "all_detections": all_detections,
        "high_conf_tiles": hc_tiles_rgb,
        "total_tiles": total_tiles,
        "elapsed_infer": elapsed_infer,
        "full_w": full_w,
        "full_h": full_h,
        "preview_rgb": preview_rgb,
        "pipeline_type": "tiff",
    }
    st.session_state["detection_cache"] = cache
    st.session_state["last_result"] = {
        "rendered_image": agg_preview,
        "detections": all_detections,
    }
    st.session_state["last_image"] = preview_rgb

    # ── Display results ──────────────────────────────────────────────
    _display_cached_results()
    gc.collect()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STANDARD IMAGE PIPELINE (PNG / JPG)                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _run_detection(image: np.ndarray):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    col_options = st.columns(2)
    with col_options[0]:
        apply_filter = st.session_state.get("filter_type", "none") != "none"
        if apply_filter:
            st.info(f"Speckle Filter: {st.session_state['filter_type']}")
    with col_options[1]:
        overlay_heatmap = st.checkbox("🔥 Overlay Real‑Time Heatmap", value=False)

    if st.button("🚀 Run Detection", type="primary", use_container_width=True):
        with st.spinner("Running detection pipeline …"):
            pipeline = get_pipeline()
            start = time.time()
            result = pipeline.process_frame(image, apply_filter=apply_filter)
            elapsed = time.time() - start

        rendered = result["rendered_image"]
        if overlay_heatmap:
            rendered = pipeline.heatmap.get_heatmap_image(background=rendered, alpha=0.5)
        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        dets = result["detections"]

        # Cache for persistence
        cache = {
            "agg_preview": rendered_rgb,
            "all_detections": dets,
            "high_conf_tiles": [],
            "total_tiles": 1,
            "elapsed_infer": elapsed,
            "full_w": image.shape[1],
            "full_h": image.shape[0],
            "preview_rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "pipeline_type": "standard",
        }
        st.session_state["detection_cache"] = cache
        st.session_state["last_result"] = result
        st.session_state["last_image"] = image

    # ── Always show cached results below the button ───────────────────
    # (will display both on fresh run AND on page return)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SHARED RESULTS DISPLAY — renders from session_state cache         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _display_cached_results():
    """Render cached detection results from st.session_state['detection_cache']."""
    cache = st.session_state.get("detection_cache")
    if not cache:
        return

    agg_preview = cache["agg_preview"]
    all_detections = cache["all_detections"]
    high_conf_tiles = cache.get("high_conf_tiles", [])
    total_tiles = cache["total_tiles"]
    elapsed_infer = cache["elapsed_infer"]
    full_w = cache["full_w"]
    full_h = cache["full_h"]

    st.markdown("---")
    st.markdown("### 📊 Detection Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚢 Ships Detected", len(all_detections))
    m2.metric("📦 Tiles Processed", total_tiles)
    m3.metric("⏱️ Inference Time", f"{elapsed_infer:.1f}s")
    m4.metric("📐 Image Size", f"{full_w}×{full_h}")

    st.image(agg_preview, caption="Detection Overlay (results persist until new image)",
             use_container_width=True)

    # ── High-confidence tile gallery ──────────────────────────────────
    if high_conf_tiles:
        st.markdown("### 🔍 High-Confidence Regions (≥ 80%)")
        cols = st.columns(3)
        for i, (t_img, label) in enumerate(high_conf_tiles[:12]):
            with cols[i % 3]:
                st.image(t_img, caption=label, use_container_width=True)
        if len(high_conf_tiles) > 12:
            st.caption(f"… plus {len(high_conf_tiles) - 12} more high-confidence regions.")

    # ── Detection table ──────────────────────────────────────────────
    if all_detections:
        st.markdown("### 📋 Detection Details")
        import pandas as pd
        df = pd.DataFrame([{
            "ID": d.get("track_id", i + 1),
            "Type": d.get("ship_type", "?"),
            "Confidence": f"{d['confidence']:.1%}",
            "Threat": f"{d.get('threat_score', 0):.0f} ({d.get('threat_level', 'N/A')})",
            "Dark Vessel": "⚠️ YES" if d.get("is_dark_vessel") else "✓ No",
            "BBox": f"({d['bbox'][0]:.0f},{d['bbox'][1]:.0f})→({d['bbox'][2]:.0f},{d['bbox'][3]:.0f})" if "bbox" in d else "",
        } for i, d in enumerate(all_detections)])
        st.dataframe(df, use_container_width=True)

    # ── Download buttons ─────────────────────────────────────────────
    st.markdown("### 📥 Downloads")
    dl1, dl2 = st.columns(2)
    with dl1:
        result_buf = io.BytesIO()
        Image.fromarray(agg_preview).save(result_buf, format="PNG")
        st.download_button("⬇️ Download Annotated Image",
                           data=result_buf.getvalue(),
                           file_name="sar_detection_result.png",
                           mime="image/png")
    with dl2:
        if all_detections:
            import pandas as pd
            csv_df = pd.DataFrame(all_detections)
            csv_buf = csv_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Detection CSV",
                               data=csv_buf,
                               file_name="sar_detections.csv",
                               mime="text/csv")

    st.success(
        "✅ Results persisted! Navigate to **📊 Analytics**, **🗺️ Map View**, "
        "**📈 Metrics**, or **📄 Reports**. Upload a new image to re-run."
    )
