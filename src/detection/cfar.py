"""
CA-CFAR (Cell-Averaging Constant False Alarm Rate) Pre-Screening for SAR Imagery.

CFAR is the radar-native target detection algorithm used operationally
by maritime surveillance systems. It adaptively thresholds each pixel
against the local background level estimated from surrounding "training"
cells, maintaining a constant probability of false alarm (PFA).

Pipeline position: runs BEFORE YOLO to produce candidate regions, reducing
the number of tiles that need deep-learning inference.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def ca_cfar_1d(signal: np.ndarray, guard_cells: int = 2,
               training_cells: int = 8, pfa: float = 1e-4) -> np.ndarray:
    """
    1-D Cell-Averaging CFAR detector.

    For each cell under test (CUT), the threshold is:
        T = α · (1/N) · Σ training_cells
    where α = N · (PFA^(-1/N) - 1)  (from Neyman-Pearson criterion).

    Args:
        signal: 1-D array of power values.
        guard_cells: Number of guard cells on each side of CUT.
        training_cells: Number of training cells on each side.
        pfa: Desired probability of false alarm.

    Returns:
        Boolean mask — True where a target is detected.
    """
    n = len(signal)
    N = 2 * training_cells
    alpha = N * (pfa ** (-1.0 / N) - 1)

    detections = np.zeros(n, dtype=bool)
    half_win = guard_cells + training_cells

    for i in range(half_win, n - half_win):
        # Leading training cells
        leading = signal[i - half_win: i - guard_cells]
        # Lagging training cells
        lagging = signal[i + guard_cells + 1: i + half_win + 1]

        noise_level = np.mean(np.concatenate([leading, lagging]))
        threshold = alpha * noise_level

        if signal[i] > threshold:
            detections[i] = True

    return detections


def ca_cfar_2d(image: np.ndarray, guard_cells: int = 3,
               training_cells: int = 12, pfa: float = 1e-5,
               min_area: int = 10) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    2-D Cell-Averaging CFAR detector for SAR images.

    Uses a ring-shaped averaging kernel (training annulus) around each
    pixel, excluding the guard band. Vectorised implementation via
    box-filter subtraction.

    Args:
        image: Grayscale SAR image (uint8 or float).
        guard_cells: Radius of the guard band (pixels).
        training_cells: Width of the training annulus (pixels).
        pfa: Probability of false alarm (lower = fewer false alarms).
        min_area: Minimum connected-component area to keep.

    Returns:
        Tuple of (binary_mask, list_of_bounding_boxes_xyxy).
    """
    img = image.astype(np.float64)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Outer and inner box sizes
    outer_r = guard_cells + training_cells
    inner_r = guard_cells

    outer_size = 2 * outer_r + 1
    inner_size = 2 * inner_r + 1

    outer_area = outer_size ** 2
    inner_area = inner_size ** 2
    N = outer_area - inner_area  # number of training cells

    if N <= 0:
        logger.error("Training ring has non-positive area; check guard/training cell sizes.")
        return np.zeros_like(img, dtype=np.uint8), []

    # CFAR scaling factor from Neyman-Pearson
    alpha = N * (pfa ** (-1.0 / max(N, 1)) - 1)

    # Box-filter sums
    outer_sum = cv2.boxFilter(img, -1, (outer_size, outer_size),
                              normalize=False, borderType=cv2.BORDER_REFLECT)
    inner_sum = cv2.boxFilter(img, -1, (inner_size, inner_size),
                              normalize=False, borderType=cv2.BORDER_REFLECT)

    # Training cell average
    training_mean = (outer_sum - inner_sum) / N

    # Adaptive threshold
    threshold = alpha * training_mean

    # Detect
    binary = (img > threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components → bounding boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, x + w, y + h))

    logger.info(f"CFAR detected {len(bboxes)} candidate regions "
                f"(guard={guard_cells}, train={training_cells}, PFA={pfa:.0e})")
    return binary, bboxes


def cfar_prescreen_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    guard_cells: int = 3,
    training_cells: int = 12,
    pfa: float = 1e-5
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Filter tiles — keep only those containing CFAR detections.

    This dramatically reduces the number of tiles requiring expensive
    YOLO inference by discarding featureless ocean tiles.

    Args:
        tiles: List of (tile_image, (row_offset, col_offset)).
        guard_cells, training_cells, pfa: CFAR parameters.

    Returns:
        Filtered list of tiles that contain at least one CFAR hit.
    """
    kept = []
    for tile_img, offset in tiles:
        gray = tile_img
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)

        _, bboxes = ca_cfar_2d(gray, guard_cells, training_cells, pfa)
        if bboxes:
            kept.append((tile_img, offset))

    logger.info(f"CFAR pre-screen: {len(kept)}/{len(tiles)} tiles retained")
    return kept
