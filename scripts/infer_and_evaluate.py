import os
import argparse
from ultralytics import YOLO
import glob

def run_inference(model_path, source_dir, conf_thresh=0.25):
    """
    Load a trained YOLOv8 model and run inference on a directory of images.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on images in {source_dir}...")
    # Results are saved automatically to runs/detect/predict (or similar)
    results = model.predict(
        source=source_dir,
        conf=conf_thresh,
        save=True,          # Save images with boxes
        save_txt=True,      # Save detection coordinates
        save_conf=True,     # Save confidences
        project="runs/detect",
        name="sentinel_inference",
        exist_ok=True       # Overwrite previous runs if necessary
    )
    
    total_images = len(results)
    images_with_ships = 0
    
    import re
    import numpy as np
    
    global_boxes = []
    global_scores = []
    
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            images_with_ships += 1
            
            # Extract anchor coordinates from filename (e.g. tile_yY_xX.jpg) to reverse map global position
            filename = os.path.basename(r.path)
            match = re.search(r'y(\d+)_x(\d+)', filename)
            if match:
                ty = int(match.group(1))
                tx = int(match.group(2))
                
                for b_idx, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    conf = float(boxes.conf[b_idx].cpu().numpy())
                    
                    # 1. Absolute Confidence Filter
                    if conf < 0.60:
                        continue
                        
                    # 2. Shape Filter (Ships are elongated. Reject irregular box blobs where ratio < 1.2)
                    w_box = x2 - x1
                    h_box = y2 - y1
                    ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
                    if ratio < 1.2:
                        continue
                        
                    # 3. Dynamic Water Masking (Detect if background surrounding the detection is bright land)
                    orig = r.orig_img
                    bx1, by1 = max(0, int(x1)-30), max(0, int(y1)-30)
                    bx2, by2 = min(orig.shape[1], int(x2)+30), min(orig.shape[0], int(y2)+30)
                    if np.mean(orig[by1:by2, bx1:bx2]) > 55.0:
                        continue
                        
                    gx1, gy1 = tx + x1, ty + y1
                    gx2, gy2 = tx + x2, ty + y2
                    
                    global_boxes.append([gx1, gy1, gx2, gy2])
                    global_scores.append(conf)
            else:
                # Fallback if filename format doesn't match
                for b_idx, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    conf = float(boxes.conf[b_idx].cpu().numpy())
                    if conf < 0.60: continue
                    global_boxes.append(box.cpu().numpy().tolist())
                    global_scores.append(conf)

    def compute_nms(boxes, scores, iou_threshold=0.20):
        if len(boxes) == 0: return []
        b = np.array(boxes)
        s = np.array(scores)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
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

    real_ship_count = len(compute_nms(global_boxes, global_scores))
            
    print("\n" + "="*40)
    print("INFERENCE SUMMARY:")
    print("="*40)
    print(f"Total tiles processed: {total_images}")
    print(f"Tiles containing ships: {images_with_ships}")
    print(f"Gross overlapping boxes detected: {len(global_boxes)}")
    print(f"REAL unique ships detected (After NMS): {real_ship_count}")
    print("\nOutputs saved to: runs/detect/sentinel_inference/")
    
    if total_images == 0:
        print("Warning: No images processed. Check your source directory.")
        return

    # Qualitative explanation placeholder for the user
    print("\n--- Evaluation Metrics Note ---")
    print("Since this is a blind inference on an unannotated Sentinel-1 scene, we do not have")
    print("ground truth bounding boxes to automatically compute Precision and Recall.")
    print("To compute exact MVP metrics (Precision/Recall):")
    print("1. Manually review a few output tiles in runs/detect/sentinel_inference/.")
    print("2. Count True Positives (correct ships), False Positives (sea clutter/artifacts),")
    print("   and False Negatives (missed ships).")
    print("3. Use the formulas:")
    print("     Precision = TP / (TP + FP)")
    print("     Recall = TP / (TP + FN)")
    print("===============================\n")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on tiled Sentinel-1 SAR images.")
    parser.add_argument("--model", "-m", type=str, default="runs/detect/train/weights/best.pt", help="Path to trained YOLOv8 weights")
    parser.add_argument("--source", "-s", type=str, default="data/inference_tiles", help="Directory containing inference tiles")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model weights not found at {args.model}")
        print("Please ensure the YOLOv8 model has finished training, or provide ")
        print("the correct path to best.pt using the --model flag.")
        return
        
    # We check if directory exists and has some files in it
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} does not exist.")
        print("Please run preprocess_sentinel.py first to generate the tiles.")
        return
        
    run_inference(args.model, args.source, conf_thresh=args.conf)

if __name__ == "__main__":
    main()
