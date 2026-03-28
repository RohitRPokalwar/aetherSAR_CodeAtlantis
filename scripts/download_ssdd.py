"""
SSDD (SAR Ship Detection Dataset) Download & Setup Script.

Downloads the Official SSDD dataset from its public repository,
extracts it, and organizes the files for YOLO training.

Usage:
    python scripts/download_ssdd.py
    python scripts/convert_annotations.py   # then convert to YOLO format
    python scripts/train.py --data data/yolo_format/dataset.yaml
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SSDD_DIR = DATA_DIR / "Official-SSDD-OPEN"

# ── SSDD Repository Info ──────────────────────────────────────────────
# Official SSDD repository: https://github.com/TianwenZhang0825/Official-SSDD
SSDD_GITHUB_URL = "https://github.com/TianwenZhang0825/Official-SSDD/archive/refs/heads/master.zip"
SSDD_ZIP_NAME = "ssdd_download.zip"


def download_progress(block_num, block_size, total_size):
    """Report download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_down:.1f}/{mb_total:.1f} MB ({percent:.0f}%)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB")
    sys.stdout.flush()


def download_ssdd():
    """Download SSDD dataset from GitHub."""
    print("=" * 60)
    print("SSDD Dataset Download & Setup")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / SSDD_ZIP_NAME

    # Check if already downloaded
    if SSDD_DIR.exists():
        print(f"\n✅ SSDD directory already exists: {SSDD_DIR}")
        print("   Delete it first if you want to re-download.")
        return

    # Download
    print(f"\n📥 Downloading SSDD from GitHub...")
    print(f"   URL: {SSDD_GITHUB_URL}")
    try:
        urlretrieve(SSDD_GITHUB_URL, str(zip_path), download_progress)
        print(f"\n   ✅ Download complete: {zip_path}")
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        print("\n   Manual download instructions:")
        print(f"   1. Visit: https://github.com/TianwenZhang0825/Official-SSDD")
        print(f"   2. Click 'Code' → 'Download ZIP'")
        print(f"   3. Extract to: {SSDD_DIR}")
        return

    # Extract
    print(f"\n📦 Extracting...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(DATA_DIR))
        print(f"   ✅ Extracted successfully")
    except zipfile.BadZipFile:
        print(f"   ❌ Corrupted ZIP file. Try downloading manually.")
        return

    # Rename extracted directory
    extracted_dir = DATA_DIR / "Official-SSDD-master"
    if extracted_dir.exists():
        shutil.move(str(extracted_dir), str(SSDD_DIR))
        print(f"   ✅ Moved to: {SSDD_DIR}")

    # Cleanup zip
    if zip_path.exists():
        zip_path.unlink()
        print(f"   🗑️ Cleaned up: {SSDD_ZIP_NAME}")

    # Verify structure
    print(f"\n🔍 Verifying dataset structure...")
    bbox_dir = SSDD_DIR / "BBox_SSDD" / "voc_style"
    if bbox_dir.exists():
        train_imgs = list((bbox_dir / "JPEGImages_train").glob("*")) if (bbox_dir / "JPEGImages_train").exists() else []
        test_imgs = list((bbox_dir / "JPEGImages_test").glob("*")) if (bbox_dir / "JPEGImages_test").exists() else []
        print(f"   Train images: {len(train_imgs)}")
        print(f"   Test images:  {len(test_imgs)}")
    else:
        print(f"   ⚠️ Expected VOC directory not found at: {bbox_dir}")
        print(f"   Contents of {SSDD_DIR}:")
        for p in sorted(SSDD_DIR.iterdir()):
            print(f"     {'📁' if p.is_dir() else '📄'} {p.name}")

    print(f"\n{'='*60}")
    print(f"✅ SSDD setup complete!")
    print(f"\nNext steps:")
    print(f"  1. Convert to YOLO format:")
    print(f"     python scripts/convert_annotations.py")
    print(f"  2. Train the model:")
    print(f"     python scripts/train.py --data data/yolo_format/dataset.yaml")
    print(f"{'='*60}")


if __name__ == "__main__":
    download_ssdd()
