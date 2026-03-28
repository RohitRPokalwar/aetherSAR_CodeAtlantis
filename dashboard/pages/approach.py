"""Approach Explanation Page — System architecture, methodology, and limitations."""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def show_approach_page():
    st.markdown("## 🔬 Technical Approach & Methodology")

    # ── System Architecture ───────────────────────────────────────────
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```mermaid
    graph LR
        A[Sentinel-1 GeoTIFF] --> B[Preprocessing]
        B --> C[CFAR Pre-Screen]
        C --> D[YOLO Detection]
        D --> E[Soft-NMS]
        E --> F[Post-Processing]
        F --> G[Analytics]
        G --> H[Dashboard]
    ```
    """)

    st.markdown("""
    The system follows a **7-stage pipeline** from raw SAR data to interactive visualization:

    | Stage | Module | Purpose |
    |-------|--------|---------|
    | 1. Preprocessing | `sentinel_prep.py` | Read GeoTIFF, calibrate σ₀, despeckle, tile |
    | 2. CFAR Pre-Screen | `cfar.py` | Radar-native target detection to filter empty tiles |
    | 3. YOLO Detection | `detector.py` | Deep learning ship detection on candidate tiles |
    | 4. Soft-NMS | `soft_nms.py` | Merge overlapping detections from adjacent tiles |
    | 5. Post-Processing | `postprocess.py` | Intensity validation, shape filtering |
    | 6. Analytics | `pipeline.py` | Tracking, classification, threat scoring |
    | 7. Dashboard | `app.py` | Interactive visualization and reporting |
    """)

    st.markdown("---")

    # ── Preprocessing ─────────────────────────────────────────────────
    st.markdown("### 📡 Stage 1: SAR Preprocessing")

    with st.expander("🔍 Radiometric Calibration", expanded=True):
        st.markdown(r"""
        **Problem:** SAR images store raw Digital Numbers (DN) that don't directly represent
        physical radar backscatter.

        **Solution:** Convert DN to σ₀ (sigma-nought) backscatter coefficient in dB:

        $$\sigma^0_{dB} = 10 \cdot \log_{10}(DN^2 + \epsilon)$$

        This normalizes the data so that different SAR scenes are comparable.
        """)

    with st.expander("🌊 Speckle Filtering"):
        st.markdown(r"""
        **Problem:** SAR images suffer from multiplicative speckle noise caused by coherent
        radar wave interference. This creates a granular "salt-and-pepper" texture that can
        confuse object detectors.

        **Solution:** Apply adaptive filters that smooth homogeneous areas while preserving edges:

        **Lee Filter** uses local statistics:
        $$I_{out} = \bar{I} + w \cdot (I_{in} - \bar{I})$$
        where $w = \sigma^2_{local} / (\sigma^2_{local} + \sigma^2_{noise})$

        **Frost Filter** applies exponential weighting based on the coefficient of variation:
        $$K(d) = e^{-\alpha \cdot C_v \cdot d}$$
        where $C_v = \sigma / \mu$ and $d$ is distance from center pixel.
        """)

    with st.expander("📐 Tiling Strategy"):
        st.markdown("""
        **Problem:** Sentinel-1 scenes are massive (20,000+ × 20,000+ pixels). YOLO expects
        640×640 input. Downsampling would destroy small ship signatures.

        **Solution:** Sliding window with overlap:
        - **Tile size:** 640×640 pixels (matching YOLO input)
        - **Overlap:** 100 pixels between adjacent tiles
        - **Edge handling:** Zero-padding for boundary tiles

        The overlap ensures ships at tile boundaries appear fully in at least one tile.
        """)

    st.markdown("---")

    # ── CFAR ──────────────────────────────────────────────────────────
    st.markdown("### 🎯 Stage 2: CFAR Pre-Screening")

    with st.expander("📖 How CA-CFAR Works", expanded=True):
        st.markdown(r"""
        **Cell-Averaging CFAR** is the standard radar target detection algorithm. For each
        pixel (Cell Under Test), it estimates the local noise level from surrounding
        "training cells" and sets an adaptive threshold:

        $$T = \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} x_i$$

        where:
        - $\alpha = N \cdot (P_{FA}^{-1/N} - 1)$ is the CFAR scaling factor
        - $N$ = number of training cells
        - $P_{FA}$ = probability of false alarm (typically 10⁻⁵)

        **Guard cells** prevent target energy from leaking into the noise estimate.

        **Why use CFAR before YOLO?**
        - Eliminates 70-90% of empty ocean tiles → dramatic speedup
        - Provides radar-domain validation of YOLO detections
        - Maintains constant false alarm rate regardless of sea state
        """)

    st.markdown("---")

    # ── YOLO Detection ────────────────────────────────────────────────
    st.markdown("### 🧠 Stage 3: YOLOv8 Deep Learning Detection")

    with st.expander("📖 Model Architecture", expanded=True):
        st.markdown("""
        **YOLOv8** (You Only Look Once v8) is a single-shot object detector:

        | Component | Description |
        |-----------|-------------|
        | **Backbone** | CSPDarknet53 — extracts multi-scale features |
        | **Neck** | PANet/FPN — fuses features at different scales |
        | **Head** | Decoupled head — separate classification and regression |
        | **Loss** | CIoU loss + DFL (Distribution Focal Loss) |

        **Training:** Fine-tuned on SSDD (SAR Ship Detection Dataset, ~1,160 images)
        with SAR-specific augmentations:
        - Random flipping (horizontal + vertical)
        - 90°/180°/270° rotation
        - Random scaling (0.8× to 1.2×)
        - Random cropping with bbox adjustment
        - Gaussian noise injection (simulating varying sea states)
        - Brightness perturbation
        """)

    st.markdown("---")

    # ── NMS ───────────────────────────────────────────────────────────
    st.markdown("### 🔄 Stage 4: Soft Non-Maximum Suppression")

    with st.expander("📖 Soft-NMS vs Hard NMS", expanded=True):
        st.markdown(r"""
        **Hard NMS** completely removes overlapping boxes — problematic for tiled inference
        where the same ship may appear in adjacent tiles with different confidence levels.

        **Soft-NMS** (Bodla et al., ICCV 2017) instead decays confidence scores:

        **Gaussian decay:**
        $$s_i \leftarrow s_i \cdot e^{-\frac{IoU^2}{\sigma}}$$

        **Linear decay:**
        $$s_i \leftarrow \begin{cases} s_i \cdot (1 - IoU) & \text{if } IoU > \theta \\ s_i & \text{otherwise} \end{cases}$$

        This preserves nearby but distinct ships while still suppressing true duplicates.
        """)

    st.markdown("---")

    # ── Post-Processing ───────────────────────────────────────────────
    st.markdown("### 🔧 Stage 5: Morphological Post-Processing")

    with st.expander("📖 False Positive Reduction", expanded=True):
        st.markdown(r"""
        SAR-specific false positive sources and their mitigation:

        | False Positive Source | Filter | Logic |
        |---------------------|--------|-------|
        | Sea clutter | Intensity validation | Target-to-clutter ratio < 1.5× → reject |
        | Land structures | Water masking | Mean background brightness > 55 → reject |
        | Azimuth ambiguities | Shape filter | Aspect ratio < 1.2 → reject |
        | Noise speckles | Area filter | Bbox area < 30 px² → reject |
        | SAR artifacts | Morphological opening | Remove isolated small components |

        **Intensity Validation** checks that the detection region is significantly
        brighter than its surrounding water:

        $$\text{TCR} = \frac{\mu_{target}}{\mu_{background}} \geq 1.5$$
        """)

    st.markdown("---")

    # ── Analytics ─────────────────────────────────────────────────────
    st.markdown("### 📊 Stage 6: Analytics Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Core Analytics:**
        - **ByteTrack** — Multi-object tracking across frames
        - **Ship Classifier** — EfficientNet-based (Cargo/Tanker/Fishing/Military)
        - **Threat Scoring** — Multi-factor 0-100 score:
          - Confidence weight (30%)
          - Zone proximity (30%)
          - Speed anomaly (20%)
          - Dwell time (20%)
        """)

    with col2:
        st.markdown("""
        **Advanced Analytics:**
        - **Dark Vessel Detection** — AIS correlation (no match = dark)
        - **Fleet Detection** — DBSCAN spatial clustering
        - **Trajectory Prediction** — Kalman filter forecasting
        - **Zone Alerts** — Named zones with cooldown timers
        """)

    st.markdown("---")

    # ── Limitations ───────────────────────────────────────────────────
    st.markdown("### ⚠️ Known Limitations")

    st.warning("""
    **Current system limitations:**

    1. **Limited training data** — SSDD has ~1,160 images; production systems use 50,000+ annotated samples
    2. **Simplified calibration** — Full radiometric calibration requires parsing Sentinel-1 annotation XML for calibration LUTs
    3. **Simulated AIS data** — Real dark vessel detection requires live AIS feed integration (MarineTraffic, AIS Hub)
    4. **Single polarization** — Only VV band is used; dual-pol (VV+VH) analysis could improve classification accuracy
    5. **No SAR-specific backbone** — Using COCO-pretrained YOLOv8; a SAR-pretrained backbone would improve feature extraction
    6. **No land masking** — Coastline masking using OpenStreetMap or GSHHS databases would eliminate land false positives
    7. **Fixed tile size** — Adaptive tiling based on ship density could improve efficiency
    """)

    st.markdown("---")

    # ── Future Work ───────────────────────────────────────────────────
    st.markdown("### 🚀 Future Improvements")

    st.success("""
    **Planned enhancements:**

    1. **Multi-scale detection** — Process tiles at 320, 640, and 1280 for different ship sizes
    2. **Attention mechanisms** — Add CBAM or SE blocks to the YOLO backbone for SAR features
    3. **Ship length estimation** — Use SAR resolution metadata to convert pixel sizes to meters
    4. **Wake detection** — Detect ship wakes to estimate heading and speed
    5. **Multi-temporal fusion** — Compare consecutive Sentinel-1 passes for change detection
    6. **Edge deployment** — ONNX/TensorRT export for real-time maritime surveillance
    """)
