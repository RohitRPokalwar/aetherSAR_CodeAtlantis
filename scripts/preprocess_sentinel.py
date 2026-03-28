import os
import argparse
import numpy as np
import cv2

def tile_large_sar(image_path, output_dir, tile_size=640, overlap=100, clip_percentile=99.0, prefix="tile"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading SAR image metadata from {image_path}...")
    
    # 1. Open the image avoiding full RAM load
    HAS_RASTERIO = False
    img = None
    
    try:
        import rasterio
        src = rasterio.open(image_path)
        h, w = src.height, src.width
        HAS_RASTERIO = True
    except ImportError:
        try:
            import tifffile
            try:
                img = tifffile.memmap(image_path)
            except Exception:
                img = tifffile.imread(image_path)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            h, w = img.shape[:2]
        except ImportError:
            # Fallback to cv2
            print("Warning: rasterio/tifffile not installed. Falling back to OpenCV full load.")
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image at {image_path}")
            if len(img.shape) > 2:
                img = img[:, :, 0]
            h, w = img.shape[:2]

    # 2. Compute percentiles on a decimated version for memory safety
    print(f"Image size is {w}x{h}. Computing normalization bounds...")
    step_y = max(1, h // 1000)
    step_x = max(1, w // 1000)
    
    if HAS_RASTERIO:
        from rasterio.windows import Window
        # read decimated preview for percentiles
        preview = src.read(1, out_shape=(h // step_y, w // step_x)).astype(np.float32)
    else:
        preview = img[::step_y, ::step_x]
    
    nonzero = preview[preview > 0]
    if nonzero.size > 0:
        p_max = float(np.percentile(nonzero, clip_percentile))
        p_min = float(np.percentile(nonzero, 1.0))
    else:
        p_max, p_min = 1.0, 0.0

    print(f"Computed normalization bounds: min={p_min:.2f}, max={p_max:.2f}")

    # 3. Tile extraction
    print("Beginning memory-safe tiling...")
    stride = tile_size - overlap
    tile_count = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, y + tile_size
            x1, x2 = x, x + tile_size

            # Adjust edge cases
            if y2 > h:
                y1, y2 = max(0, h - tile_size), h
            if x2 > w:
                x1, x2 = max(0, w - tile_size), w

            # Read raw tile
            if HAS_RASTERIO:
                window = Window(col_off=x1, row_off=y1, width=(x2 - x1), height=(y2 - y1))
                raw_tile = src.read(1, window=window).astype(np.float32)
            else:
                raw_tile = img[y1:y2, x1:x2].astype(np.float32)

            # Normalize on the fly
            tile_clipped = np.clip(raw_tile, p_min, p_max)
            if p_max - p_min > 0:
                tile_norm = ((tile_clipped - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                tile_norm = np.zeros_like(tile_clipped, dtype=np.uint8)

            tile_rgb = np.stack((tile_norm,) * 3, axis=-1)

            # Pad edge tiles if smaller than tile_size
            if tile_rgb.shape[0] < tile_size or tile_rgb.shape[1] < tile_size:
                pad_h = tile_size - tile_rgb.shape[0]
                pad_w = tile_size - tile_rgb.shape[1]
                tile_rgb = cv2.copyMakeBorder(tile_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            tile_name = f"{prefix}_y{y1}_x{x1}.jpg"
            tile_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(tile_path, tile_rgb)
            tile_count += 1
            
            if tile_count % 500 == 0:
                print(f"Generated {tile_count} tiles...")

    if HAS_RASTERIO:
        src.close()

    print(f"✅ Generated {tile_count} tiles of size {tile_size}x{tile_size} in '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tile Sentinel-1 SAR imagery for YOLO inference.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw Sentinel-1 .tiff / .img / .png file")
    parser.add_argument("--output", "-o", type=str, default="data/inference_tiles", help="Output directory for tiles")
    parser.add_argument("--tile-size", type=int, default=640, help="Size of the square tiles (default 640 for YOLOv8)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between tiles in pixels")
    parser.add_argument("--clip-percentile", type=float, default=99.0, help="Percentile to clip to prevent over-darkening")
    args = parser.parse_args()

    try:
        tile_large_sar(args.input, args.output, tile_size=args.tile_size, overlap=args.overlap, clip_percentile=args.clip_percentile)
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
