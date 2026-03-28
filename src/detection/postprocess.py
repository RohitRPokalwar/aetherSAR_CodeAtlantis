"""
Morphological Post-Processing for SAR Ship Detection.

Applies morphological operations and heuristic filters to reduce
false positives from YOLO detections on SAR imagery. SAR-specific
false positive sources include:
  - Sea clutter (Bragg scattering from ocean waves)
  - Azimuth ambiguities (ghost targets from SAR processing)
  - Bright land structures (harbors, bridges, oil platforms)
  - Radar side-lobe artifacts

This module provides pixel-level validation and shape-based filtering
to eliminate these common false alarms.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def intensity_validation(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    min_contrast_ratio: float = 1.5,
    background_margin: int = 20
) -> bool:
    """
    Validate that a detected target is significantly brighter than
    its surrounding water background (target-to-clutter ratio).

    Ships in SAR appear as bright scatterers against dark ocean.
    If the detection region is not significantly brighter than
    surrounding pixels, it is likely a false positive.

    Args:
        image: Grayscale image (uint8 or float).
        bbox: (x1, y1, x2, y2) bounding box.
        min_contrast_ratio: Minimum ratio of target mean to background mean.
        background_margin: Pixels around bbox to sample for background.

    Returns:
        True if detection passes intensity validation.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False

    # Target intensity
    target_region = image[y1:y2, x1:x2].astype(np.float64)
    target_mean = np.mean(target_region)

    # Background region (expanded bbox minus target)
    bg_x1 = max(0, x1 - background_margin)
    bg_y1 = max(0, y1 - background_margin)
    bg_x2 = min(w, x2 + background_margin)
    bg_y2 = min(h, y2 + background_margin)

    bg_region = image[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float64)

    # Create mask to exclude target from background
    mask = np.ones_like(bg_region, dtype=bool)
    inner_y1 = y1 - bg_y1
    inner_x1 = x1 - bg_x1
    inner_y2 = inner_y1 + (y2 - y1)
    inner_x2 = inner_x1 + (x2 - x1)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    bg_pixels = bg_region[mask]
    if len(bg_pixels) == 0:
        return True  # can't compute background, keep detection

    bg_mean = np.mean(bg_pixels)

    if bg_mean < 1e-6:
        return target_mean > 5  # dark background, any brightness counts

    ratio = target_mean / (bg_mean + 1e-6)
    return ratio >= min_contrast_ratio


def shape_filter(
    bbox: Tuple[int, int, int, int],
    min_aspect_ratio: float = 1.2,
    max_aspect_ratio: float = 15.0,
    min_area: int = 30,
    max_area: int = 500000
) -> bool:
    """
    Filter detections by bounding box shape.

    Ships in SAR are typically elongated (aspect ratio > 1.2).
    Very square or extremely thin detections are likely false positives.

    Args:
        bbox: (x1, y1, x2, y2) bounding box.
        min_aspect_ratio: Minimum ratio of long side to short side.
        max_aspect_ratio: Maximum ratio (removes line artifacts).
        min_area: Minimum bounding box area in pixels².
        max_area: Maximum area (removes land blobs).

    Returns:
        True if detection passes shape filter.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    area = w * h
    if area < min_area or area > max_area:
        return False

    aspect = max(w, h) / (min(w, h) + 1e-6)
    return min_aspect_ratio <= aspect <= max_aspect_ratio


def morphological_cleanup(
    binary_mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
    open_iterations: int = 1,
    close_iterations: int = 2
) -> np.ndarray:
    """
    Apply morphological opening then closing to clean detection masks.

    Opening (erosion → dilation) removes small noise speckles.
    Closing (dilation → erosion) fills small gaps in ship masks.

    Args:
        binary_mask: Binary mask (0 or 255).
        open_kernel_size: Kernel size for opening.
        close_kernel_size: Kernel size for closing.
        open_iterations: Number of opening passes.
        close_iterations: Number of closing passes.

    Returns:
        Cleaned binary mask.
    """
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (open_kernel_size, open_kernel_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (close_kernel_size, close_kernel_size))

    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel,
                               iterations=open_iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel,
                               iterations=close_iterations)

    return cleaned


def connected_component_filter(
    binary_mask: np.ndarray,
    min_area: int = 20,
    max_area: int = 100000
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Extract and filter connected components by area.

    Returns cleaned mask and bounding boxes for valid components.

    Args:
        binary_mask: Binary mask (uint8, 0 or 255).
        min_area: Minimum component area to keep.
        max_area: Maximum component area to keep.

    Returns:
        Tuple of (filtered_mask, bounding_boxes_xyxy).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    filtered = np.zeros_like(binary_mask)
    bboxes = []

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bboxes.append((x, y, x + w, y + h))

    return filtered, bboxes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAND MASK FILTERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_land_mask(
    image: np.ndarray,
    brightness_threshold: float = 55.0,
    texture_threshold: float = 25.0,
    kernel_size: int = 15,
    morph_kernel: int = 21,
    morph_iterations: int = 3,
) -> np.ndarray:
    """
    Generate a binary land/water mask from a SAR image.

    Land areas in SAR imagery are characterised by:
      - High mean backscatter (bright, textured surfaces)
      - High local variance (heterogeneous scattering)

    Ocean areas are characterised by:
      - Low mean backscatter (dark, smooth surface)
      - Low local variance (homogeneous Bragg scattering)

    The mask combines both brightness and texture (local standard
    deviation) to robustly distinguish land from water, then applies
    morphological closing to fill gaps and remove noise.

    Args:
        image: Grayscale SAR image (uint8).
        brightness_threshold: Mean-brightness cutoff (pixels above = land).
        texture_threshold: Local-std-dev cutoff (pixels above = land).
        kernel_size: Window size for local statistics.
        morph_kernel: Morphological closing kernel size.
        morph_iterations: Closing iterations (higher = smoother boundary).

    Returns:
        Binary mask — 255 = land, 0 = water.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float64)

    # ── Brightness criterion ──────────────────────────────────────────
    local_mean = cv2.blur(gray_f, (kernel_size, kernel_size))
    bright_mask = (local_mean > brightness_threshold).astype(np.uint8) * 255

    # ── Texture criterion (local standard deviation) ──────────────────
    local_sq_mean = cv2.blur(gray_f ** 2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.clip(local_var, 0, None)
    local_std = np.sqrt(local_var)
    texture_mask = (local_std > texture_threshold).astype(np.uint8) * 255

    # ── Combine: pixel is land if BOTH bright AND textured ────────────
    land_mask = cv2.bitwise_and(bright_mask, texture_mask)

    # ── Morphological closing to fill small holes in land ─────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
    )
    land_mask = cv2.morphologyEx(
        land_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )

    # ── Optional: remove tiny water "holes" inside land ───────────────
    land_mask = cv2.morphologyEx(
        land_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    logger.info(
        f"Land mask: {np.count_nonzero(land_mask) / land_mask.size * 100:.1f}% "
        f"classified as land (bright>{brightness_threshold}, "
        f"texture>{texture_threshold})"
    )
    return land_mask


def land_mask_filter(
    land_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    land_pixel_ratio: float = 0.4,
    margin: int = 10,
) -> bool:
    """
    Check whether a detection overlaps with land.

    A detection is considered "on land" if more than `land_pixel_ratio`
    of the pixels inside an expanded bounding box are classified as
    land in the precomputed land mask.

    Args:
        land_mask: Binary mask (255 = land, 0 = water).
        bbox: (x1, y1, x2, y2) bounding box of the detection.
        land_pixel_ratio: Max fraction of land pixels to tolerate
                          (0.4 = reject if >40 % of bbox is land).
        margin: Extra pixels around the bbox to check (catches
                detections right at the coastline).

    Returns:
        True if detection is over WATER (keep it).
        False if detection is over LAND  (discard it).
    """
    h, w = land_mask.shape[:2]
    x1, y1, x2, y2 = bbox

    # Expand by margin (catches coastline false positives)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = land_mask[y1:y2, x1:x2]
    land_fraction = np.count_nonzero(roi) / (roi.size + 1e-6)

    return land_fraction < land_pixel_ratio


def postprocess_detections(
    image: np.ndarray,
    detections: List[Dict],
    enable_intensity: bool = True,
    enable_shape: bool = True,
    enable_land_mask: bool = True,
    land_mask: Optional[np.ndarray] = None,
    min_contrast_ratio: float = 1.5,
    min_aspect_ratio: float = 1.2,
    max_aspect_ratio: float = 15.0,
    min_area: int = 30,
    max_area: int = 500000,
    land_pixel_ratio: float = 0.4,
) -> List[Dict]:
    """
    Apply full post-processing pipeline to a list of detections.

    Pipeline order:
        1. Shape filter  (aspect ratio + area bounds)
        2. Land mask      (discard detections over land)
        3. Intensity check (target-to-clutter ratio)

    Args:
        image: Original SAR image (for intensity validation).
        detections: List of detection dicts with 'bbox' key [x1,y1,x2,y2].
        enable_intensity: Whether to apply intensity validation.
        enable_shape: Whether to apply shape filtering.
        enable_land_mask: Whether to apply land mask filtering.
        land_mask: Precomputed binary land mask (255=land). If None and
                   enable_land_mask is True, one will be generated.
        min_contrast_ratio: For intensity validation.
        min_aspect_ratio: For shape filter.
        max_aspect_ratio: For shape filter.
        min_area: Minimum bbox area.
        max_area: Maximum bbox area.
        land_pixel_ratio: Fraction above which a detection is "on land".

    Returns:
        Filtered list of detections.
    """
    if not detections:
        return []

    # Generate land mask on demand
    if enable_land_mask and land_mask is None:
        land_mask = generate_land_mask(image)

    initial_count = len(detections)
    filtered = []
    land_rejected = 0

    for det in detections:
        bbox = det.get("bbox", det.get("box", [0, 0, 0, 0]))
        bbox_tuple = tuple(int(v) for v in bbox[:4])

        # 1. Shape filter
        if enable_shape and not shape_filter(
            bbox_tuple, min_aspect_ratio, max_aspect_ratio, min_area, max_area
        ):
            continue

        # 2. Land mask filter
        if enable_land_mask and land_mask is not None:
            if not land_mask_filter(land_mask, bbox_tuple, land_pixel_ratio):
                land_rejected += 1
                continue

        # 3. Intensity validation
        if enable_intensity and not intensity_validation(
            image, bbox_tuple, min_contrast_ratio
        ):
            continue

        filtered.append(det)

    logger.info(
        f"Post-processing: {initial_count} → {len(filtered)} detections "
        f"(intensity={enable_intensity}, shape={enable_shape}, "
        f"land_mask={enable_land_mask}, land_rejected={land_rejected})"
    )
    return filtered
