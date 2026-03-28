# SAR TIFF Preprocessing and Detection Pipeline: Technical Explanation

This document details the exact mathematical and programmatic methodology used to process massive ($900+$ MB) Sentinel-1 SAR `.tiff` images and perform YOLO-based maritime object detection.

---

## Part 1: How the TIFF File is Preprocessed

Processing raw Synthetic Aperture Radar (SAR) imagery acquired from space presents three primary challenges: astronomical resolution, extreme radiometric ranges (backscatter), and single-channel format. The preprocessing pipeline overcomes these in four steps:

### 1. Dynamic Matrix Loading & Band Extraction
A massive 900MB TIFF often translates to an image matrix of over $20,000 \times 20,000$ pixels. Attempting to directly decode this using standard web libraries crashes the system due to decompression limits (e.g., PIL DecompressionBomb DOS limits). 
**Solution:** The pipeline utilizes the `tifffile` library to read the raw radar matrix efficiently into a numpy array via a memory buffer (`io.BytesIO`). If the `.tiff` contains multiple polarization bands (like VV and VH), the pipeline isolates the primary band (Band 0) for consistent intensity calculation.

### 2. Radiometric Normalization (Intensity Clipping)
Unlike standard optical cameras, radar backscatter intensity is not capped at `255`. Highly reflective materials (like metal ship hulls) cause extreme specular reflection, producing pixel values in the thousands. If we scaled the image linearly from `0` to the absolute maximum value, the water would turn completely pitch black, and the ships would be single tiny white dots.
**Solution:** We apply statistical percentile clipping.
- We find the **1st percentile** ($P_{min}$) and the **99th percentile** ($P_{max}$) of all non-zero pixels.
- Any pixel value above $P_{max}$ is hard-clipped down to $P_{max}$ (removing extreme outlier spikes).
- Any value below $P_{min}$ is clipped to $P_{min}$.
- We then linearly map the remaining values to standard 8-bit display format ($0$ to $255$).

### 3. Dimensionality Conversion (RGB Stacking)
YOLOv8 is pre-trained on the COCO dataset, which consists of standard 3-channel (RGB) images. SAR images are naturally single-channel (grayscale). To prevent matrix-shape errors entirely and leverage YOLO's pre-trained convolutional filters, the pipeline mathematically stacks the normalized 1-channel array three times (`np.stack`) to synthesize an artificial 3-channel RGB image.

### 4. Sliding-Window Tiling
If you feed a $20,000 \times 20,000$ image into YOLO, the model will automatically squash the entire image down to its default input size of $640 \times 640$. This compression destroys the spatial resolution, and smaller ships (which might only be 20 pixels wide originally) completely vanish into a sub-pixel fraction, resulting in $0$ detections.
**Solution:** We employ a Tile-and-Stitch methodology:
- The giant image is sliced into smaller patches, exactly $640 \times 640$ pixels in size.
- We use an **overlapping stride constraint** (e.g., $100$ pixels of overlap). This guarantees that if a ship happens to be sitting exactly on the border between two tiles, it isn't cleanly sliced in half (which prevents detection). It will appear whole in at least one tile.

---

## Part 2: How the System Detects Ships

Once the data is fractured into hundreds of normalized $640 \times 640$ tiles, the detection phase begins:

### 1. Neural Network Inference
Each tile is passed consecutively to the YOLOv8 (You Only Look Once) object detection model loaded with your custom SAR-specific weights (`best.pt`). YOLO rapidly passes the matrix through its darknet CNN backbone and Feature Pyramid Network (FPN), analyzing the unique backscatter silhouette, edge gradients, and shadowing present in SAR ships.

### 2. Confidence Thresholding
The model predicts bounding box coordinates (`x_center, y_center, width, height`) alongside a **Confidence Score** (e.g., $0.85$ means the model is $85\%$ sure it is a ship). Our pipeline applies a strict threshold (default `0.25`), actively discarding any low-confidence predictions that are likely just sea clutter, wave artifacts, or radar speckle noise.

### 3. Coordinate Remapping & Global Stitching
Because YOLO analyzed a detached $640 \times 640$ tile, the bounding box coordinates it outputs are strictly local to that specific tile (e.g., `x_1=30, y_1=40`). 
To show the user the final result on the Streamlit dashboard, the pipeline reverses the slicing math:
- It takes the anchor coordinate of where the tile was extracted from the master image (e.g., `tile_x=4000, tile_y=8000`).
- It adds the global anchor to the local detection (`global_x = 4000 + 30`).
- It dynamically draws the globally-mapped bounding box onto the downscaled master preview image on the dashboard, resulting in a seamless, holistic map of all detections across the entire $900$MB zone!
