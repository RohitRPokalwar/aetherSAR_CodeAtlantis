import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

def compute_nms(boxes, scores, iou_threshold=0.35):
    """Apply Non-Maximum Suppression to overlapping bounding boxes."""
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

def load_image(image_path):
    """Load TIFF or PNG/JPG and normalize to 8-bit RGB."""
    print(f"Loading image from {image_path}...")
    ext = os.path.splitext(image_path)[1].lower()
    img_rgb = None
    
    if ext in {'.tif', '.tiff', '.geotiff'}:
        try:
            import tifffile
            img = tifffile.imread(image_path)
            # Handle multi-page or weird shapes
            if img.ndim > 2 and img.shape[2] > 3:
                img = img[:, :, 0] # Take first band
            elif img.ndim > 2 and img.shape[0] < 5:
                # Channels first (e.g., [C, H, W])
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize to 8-bit if it's float or 16-bit
            if img.dtype != np.uint8:
                nonzero = img[img > 0]
                if nonzero.size > 0:
                    p_lo = np.percentile(nonzero, 1)
                    p_hi = np.percentile(nonzero, 99)
                else:
                    p_lo, p_hi = 0.0, 1.0
                    
                img = np.clip(img, p_lo, p_hi)
                if p_hi > p_lo:
                    img = ((img - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
                    
            if img.ndim == 2:
                img_rgb = np.stack((img,)*3, axis=-1)
            else:
                img_rgb = img
                if img_rgb.shape[2] == 3:
                    # Depending on how it's stored, might need to drop alpha
                    pass
        except ImportError:
            print("tifffile not installed. Falling back to cv2.imread...")
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    if img_rgb is None:
        raise ValueError(f"Failed to load image from {image_path}. Check path and format.")
        
    return img_rgb

def main():
    parser = argparse.ArgumentParser(description="Run YOLO on SAR images via tiling. Generates bounding boxes and scores.")
    parser.add_argument("--input", required=True, help="Path to input TIFF or image.")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model weights.")
    parser.add_argument("--output", default="annotated_output.jpg", help="Path to save annotated output image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0 to 1).")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for inference sliding window.")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between sliding window tiles (pixels).")
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)

    # 2. Load Image
    img = load_image(args.input)
    h, w = img.shape[:2]
    print(f"Image successfully loaded. Dimensions: {w}x{h} px")

    # 3. Sliding Window Setup
    stride = max(1, args.tile_size - args.overlap)
    
    # Generate tile coordinates
    tile_coords = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = min(x, max(0, w - args.tile_size))
            y1 = min(y, max(0, h - args.tile_size))
            x2 = min(x1 + args.tile_size, w)
            y2 = min(y1 + args.tile_size, h)
            tile_coords.append((x1, y1, x2, y2))
            
    # Deduplicate coords if any
    tile_coords = list(set(tile_coords))
    total_tiles = len(tile_coords)

    all_boxes = []
    all_scores = []
    
    print(f"Starting inference over {total_tiles} tiles...")
    
    for idx, (tx1, ty1, tx2, ty2) in enumerate(tile_coords, 1):
        tile = img[ty1:ty2, tx1:tx2]
        
        # Add padding if the tile is smaller than expected at the edges
        if tile.shape[0] < args.tile_size or tile.shape[1] < args.tile_size:
            padded = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
            
        results = model.predict(source=tile, conf=args.conf, verbose=False)
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # Bounding Box
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    
                    # Convert local tile coordinates to global image coordinates
                    gx1 = tx1 + bx1
                    gy1 = ty1 + by1
                    gx2 = tx1 + bx2
                    gy2 = ty1 + by2
                    
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_scores.append(score)
                    
        if idx % 10 == 0 or idx == total_tiles:
            print(f"  Processed {idx}/{total_tiles} tiles...")

    print(f"Completed inference. Found {len(all_boxes)} raw detections.")
    
    # 4. Global Non-Maximum Suppression (Remove overlaps)
    keep_indices = compute_nms(all_boxes, all_scores, iou_threshold=0.35)
    print(f"After NMS, {len(keep_indices)} valid ships remain.")

    # 5. Draw Boxes and Scores
    print(f"Drawing bounding boxes and saving to {args.output}...")
    output_img = img.copy()
    
    for idx in keep_indices:
        x1, y1, x2, y2 = map(int, all_boxes[idx])
        conf = all_scores[idx]
        
        # Draw bounding box (Green)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label = f"{conf*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(output_img, (x1, y1 - lh - 4), (x1 + lw, y1), (0, 255, 0), -1)
        
        # Draw text (Black)
        cv2.putText(output_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 6. Save File
    # Convert RGB back to BGR for cv2 saving
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(args.output, output_img_bgr)
    
    if success:
        print(f"Success! Annotated image saved to: {os.path.abspath(args.output)}")
    else:
        print("Error: Could not save the image. Please check your output path.")

if __name__ == '__main__':
    main()
