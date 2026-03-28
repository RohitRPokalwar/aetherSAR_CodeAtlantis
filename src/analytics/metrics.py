"""
Detection Metrics Computation Module.

Computes precision, recall, F1 score, and mAP for ship detection
evaluation. Works with both YOLO-format ground truth labels and
custom detection dictionaries.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / (union + 1e-6)


def match_detections(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Match predicted boxes to ground truth using IoU-based greedy matching.

    Args:
        pred_boxes: Predicted bounding boxes [x1,y1,x2,y2].
        pred_scores: Confidence scores for predictions.
        gt_boxes: Ground truth bounding boxes [x1,y1,x2,y2].
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).
    """
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    # Sort predictions by confidence (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    matched_gt = set()
    tp = 0
    fp = 0

    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_precision_recall(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, and F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_precision_recall_curve(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    num_points: int = 50
) -> Dict[str, List[float]]:
    """
    Compute precision-recall curve at multiple confidence thresholds.

    Args:
        pred_boxes: Predicted bounding boxes.
        pred_scores: Confidence scores.
        gt_boxes: Ground truth boxes.
        iou_threshold: IoU threshold for matching.
        num_points: Number of threshold points.

    Returns:
        Dict with 'thresholds', 'precisions', 'recalls', 'f1_scores'.
    """
    thresholds = np.linspace(0.05, 0.95, num_points)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        # Filter predictions by threshold
        filtered_boxes = [b for b, s in zip(pred_boxes, pred_scores) if s >= thresh]
        filtered_scores = [s for s in pred_scores if s >= thresh]

        tp, fp, fn = match_detections(filtered_boxes, filtered_scores, gt_boxes, iou_threshold)
        metrics = compute_precision_recall(tp, fp, fn)

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1"])

    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
    }


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Compute Average Precision (AP) using the 11-point interpolation method.

    Args:
        precisions: Precision values at different thresholds.
        recalls: Corresponding recall values.

    Returns:
        AP score (0-1).
    """
    if not precisions or not recalls:
        return 0.0

    # 11-point interpolation
    recall_levels = np.linspace(0, 1, 11)
    interpolated_precisions = []

    for r_level in recall_levels:
        # Find max precision at recall >= r_level
        precs_at_recall = [p for p, r in zip(precisions, recalls) if r >= r_level]
        if precs_at_recall:
            interpolated_precisions.append(max(precs_at_recall))
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


def compute_confusion_matrix(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute confusion matrix values for binary detection (ship / no-ship).

    Returns dict with TP, FP, FN, TN (TN is always 0 for detection tasks).
    """
    filtered_boxes = [b for b, s in zip(pred_boxes, pred_scores) if s >= confidence_threshold]
    filtered_scores = [s for s in pred_scores if s >= confidence_threshold]

    tp, fp, fn = match_detections(filtered_boxes, filtered_scores, gt_boxes, iou_threshold)

    return {"TP": tp, "FP": fp, "FN": fn, "TN": 0}


def generate_detection_statistics(detections: List[Dict]) -> Dict:
    """
    Generate statistical summary of detections.

    Args:
        detections: List of detection dicts with 'confidence', 'ship_type', etc.

    Returns:
        Summary dict with counts, distributions, and statistics.
    """
    if not detections:
        return {
            "total": 0,
            "mean_confidence": 0,
            "confidence_std": 0,
            "confidence_histogram": [],
            "type_distribution": {},
            "threat_distribution": {},
        }

    confidences = [d.get("confidence", 0) for d in detections]

    # Confidence histogram (10 bins)
    hist, bin_edges = np.histogram(confidences, bins=10, range=(0, 1))

    # Ship type distribution
    types = {}
    for d in detections:
        t = d.get("ship_type", "Unknown")
        types[t] = types.get(t, 0) + 1

    # Threat distribution
    threats = {}
    for d in detections:
        level = d.get("threat_level", "N/A")
        threats[level] = threats.get(level, 0) + 1

    # Detection area statistics
    areas = []
    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        areas.append(w * h)

    return {
        "total": len(detections),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "confidence_std": round(float(np.std(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "max_confidence": round(float(np.max(confidences)), 4),
        "confidence_histogram": hist.tolist(),
        "confidence_bin_edges": bin_edges.tolist(),
        "type_distribution": types,
        "threat_distribution": threats,
        "mean_area": round(float(np.mean(areas)), 1) if areas else 0,
        "dark_vessels": sum(1 for d in detections if d.get("is_dark_vessel")),
    }
