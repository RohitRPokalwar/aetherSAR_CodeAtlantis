"""
Global configuration for Ship Detection in SAR Imagery project.
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SSDD_DIR = DATA_DIR / "ssdd"
SENTINEL_DIR = DATA_DIR / "sentinel1"
YOLO_DATA_DIR = DATA_DIR / "yolo_format"
MOCK_AIS_PATH = DATA_DIR / "mock_ais.csv"
ZONES_PATH = PROJECT_ROOT / "config" / "zones.json"

# ── Model Settings ─────────────────────────────────────────────────────────
YOLO_WEIGHTS = MODELS_DIR / "yolov8s_sar.pt"
CLASSIFIER_WEIGHTS = MODELS_DIR / "ship_classifier.pt"
YOLO_PRETRAINED = "yolov8s.pt"  # Base pretrained weights for fine-tuning

# ── Detection Settings ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
INPUT_IMAGE_SIZE = 640
DEVICE = "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"

# ── Speckle Filter Settings ───────────────────────────────────────────────
SPECKLE_FILTER_TYPE = "lee"   # "lee" or "frost"
SPECKLE_WINDOW_SIZE = 7

# ── Tracker Settings ───────────────────────────────────────────────────────
TRACKER_TYPE = "bytetrack"
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
TRACK_BUFFER = 30  # frames to keep lost tracks

# ── Threat Score Weights ───────────────────────────────────────────────────
THREAT_WEIGHTS = {
    "confidence": 0.3,
    "zone_proximity": 0.3,
    "speed": 0.2,
    "dwell_time": 0.2,
}
THREAT_LEVELS = {
    "LOW": (0, 33),
    "MEDIUM": (34, 66),
    "HIGH": (67, 100),
}

# ── Zone Alert Settings ───────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 30
MAX_ALERT_LOG_SIZE = 500

# ── Dark Vessel Detection ─────────────────────────────────────────────────
AIS_MATCH_RADIUS_METERS = 500

# ── Fleet Detection (DBSCAN) ──────────────────────────────────────────────
FLEET_DBSCAN_EPS = 100          # pixels
FLEET_DBSCAN_MIN_SAMPLES = 2

# ── Trajectory Prediction (Kalman) ────────────────────────────────────────
KALMAN_PREDICT_STEPS = 10
KALMAN_DT = 1.0  # time step in seconds

# ── Ship Classifier ───────────────────────────────────────────────────────
SHIP_CLASSES = ["cargo", "tanker", "fishing", "military"]
CLASSIFIER_INPUT_SIZE = 224

# ── Heatmap Settings ──────────────────────────────────────────────────────
HEATMAP_RESOLUTION = (640, 640)
HEATMAP_COLORMAP = "hot"
HEATMAP_GIF_FPS = 5

# ── Dashboard Settings ────────────────────────────────────────────────────
DASHBOARD_TITLE = "SAR Maritime Intelligence System"
DASHBOARD_ICON = "🛳️"
DASHBOARD_LAYOUT = "wide"
DASHBOARD_THEME = "dark"

# ── Training Settings ─────────────────────────────────────────────────────
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 4
TRAIN_IMAGE_SIZE = 640
TRAIN_PATIENCE = 10

# ── Large TIFF Processing ─────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 2000         # Streamlit upload limit (must match config.toml)
TILE_SIZE = 640                    # YOLO input tile size
TILE_OVERLAP = 100                 # Overlap between tiles for NMS
PREVIEW_MAX_DIM = 1200             # Max dimension for decimated preview
CONFIDENCE_FILTER_THRESHOLD = 0.55 # Post-NMS confidence cut-off
WATER_MASK_BRIGHTNESS = 55.0       # Mean brightness threshold for land rejection
SHIP_ASPECT_RATIO_MIN = 1.2       # Min aspect ratio to keep (ships are elongated)

# ── CFAR Pre-Screening ─────────────────────────────────────────────
CFAR_GUARD_CELLS = 3              # Guard band radius (pixels)
CFAR_TRAINING_CELLS = 12          # Training annulus width (pixels)
CFAR_PFA = 1e-5                   # Probability of false alarm
CFAR_MIN_AREA = 10                # Min connected component area (pixels²)
CFAR_ENABLED = True               # Enable CFAR tile pre-screening

# ── Soft-NMS ───────────────────────────────────────────────────────
SOFT_NMS_METHOD = "gaussian"       # "gaussian" or "linear"
SOFT_NMS_SIGMA = 0.5              # Gaussian decay parameter
SOFT_NMS_SCORE_THRESHOLD = 0.15   # Min score after decay
SOFT_NMS_IOU_THRESHOLD = 0.25     # IoU threshold (linear method)

# ── Morphological Post-Processing ─────────────────────────────────
POSTPROC_MIN_CONTRAST_RATIO = 1.5 # Target-to-clutter ratio
POSTPROC_MIN_ASPECT_RATIO = 1.2   # Ships are elongated
POSTPROC_MAX_ASPECT_RATIO = 15.0  # Reject line artifacts
POSTPROC_MIN_AREA = 30            # Min bbox area (pixels²)
POSTPROC_MAX_AREA = 500000        # Max bbox area (reject land blobs)
POSTPROC_INTENSITY_ENABLED = True  # Enable intensity validation
POSTPROC_SHAPE_ENABLED = True      # Enable shape filtering

# ── Land Mask Filtering ───────────────────────────────────────────
LAND_MASK_ENABLED = True           # Enable land-based false positive rejection
LAND_MASK_BRIGHTNESS = 55.0       # Local-mean brightness threshold for land
LAND_MASK_TEXTURE = 25.0          # Local-std-dev threshold for land texture
LAND_MASK_PIXEL_RATIO = 0.4      # Reject detection if > 40% bbox is land
LAND_MASK_MARGIN = 10             # Extra pixels around bbox for coastline check

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
