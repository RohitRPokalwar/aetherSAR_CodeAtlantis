"""
Soft-NMS (Soft Non-Maximum Suppression) for SAR Ship Detection.

Unlike standard greedy NMS which hard-suppresses overlapping boxes,
Soft-NMS gradually decays confidence scores of overlapping detections.
This is critical for tiled SAR inference where the same ship appears
in adjacent tiles at slightly different positions and confidence levels.

References:
    Bodla et al., "Soft-NMS — Improving Object Detection With One Line of Code", ICCV 2017
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.3,
    method: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Soft-NMS implementation with Gaussian and linear decay methods.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
        scores: (N,) array of confidence scores.
        sigma: Gaussian decay parameter (lower = more aggressive suppression).
        score_threshold: Minimum score to keep a detection.
        iou_threshold: IoU threshold for linear method.
        method: "gaussian" or "linear".

    Returns:
        Tuple of (kept_boxes, kept_scores, kept_indices).
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), []

    boxes = np.array(boxes, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    N = len(boxes)

    # Working copies
    indices = np.arange(N)

    kept_boxes = []
    kept_scores = []
    kept_indices = []

    while len(scores) > 0:
        # Find highest scoring box
        max_idx = np.argmax(scores)
        max_box = boxes[max_idx].copy()
        max_score = scores[max_idx]
        max_orig_idx = indices[max_idx]

        # Keep this detection
        kept_boxes.append(max_box)
        kept_scores.append(max_score)
        kept_indices.append(int(max_orig_idx))

        # Remove it from consideration
        boxes = np.delete(boxes, max_idx, axis=0)
        scores = np.delete(scores, max_idx)
        indices = np.delete(indices, max_idx)

        if len(boxes) == 0:
            break

        # Compute IoU with remaining boxes
        iou = _compute_iou_vectorized(max_box, boxes)

        # Decay scores based on method
        if method == "gaussian":
            decay = np.exp(-(iou ** 2) / sigma)
            scores *= decay
        elif method == "linear":
            decay = np.where(iou > iou_threshold, 1.0 - iou, 1.0)
            scores *= decay
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gaussian' or 'linear'.")

        # Remove boxes below threshold
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]

    return (
        np.array(kept_boxes) if kept_boxes else np.array([]),
        np.array(kept_scores) if kept_scores else np.array([]),
        kept_indices
    )


def _compute_iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection
    return intersection / (union + 1e-6)


def merge_tile_detections(
    all_boxes: List[List[float]],
    all_scores: List[float],
    method: str = "gaussian",
    sigma: float = 0.5,
    score_threshold: float = 0.15,
    iou_threshold: float = 0.25
) -> Tuple[List[List[float]], List[float]]:
    """
    Merge detections from overlapping tiles using Soft-NMS.

    This is the main entry point for tile-based SAR inference pipelines.
    Replaces hard NMS for better handling of tile-boundary ships.

    Args:
        all_boxes: List of [x1, y1, x2, y2] in global image coordinates.
        all_scores: Corresponding confidence scores.
        method: "gaussian" or "linear".
        sigma: Gaussian decay parameter.
        score_threshold: Minimum score for final output.
        iou_threshold: IoU threshold for linear method.

    Returns:
        Tuple of (filtered_boxes, filtered_scores).
    """
    if not all_boxes:
        return [], []

    boxes_arr = np.array(all_boxes, dtype=np.float64)
    scores_arr = np.array(all_scores, dtype=np.float64)

    kept_boxes, kept_scores, _ = soft_nms(
        boxes_arr, scores_arr,
        sigma=sigma,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        method=method
    )

    result_boxes = kept_boxes.tolist() if len(kept_boxes) > 0 else []
    result_scores = kept_scores.tolist() if len(kept_scores) > 0 else []

    logger.info(f"Soft-NMS ({method}): {len(all_boxes)} → {len(result_boxes)} detections")
    return result_boxes, result_scores
