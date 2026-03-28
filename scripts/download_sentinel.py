"""
Sentinel-1 Sample Scene Download Guide & Helper Script.

Provides direct download links and programmatic access guides for
obtaining Sentinel-1 SAR scenes from open-access platforms.

Usage:
    python scripts/download_sentinel.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTINEL_DIR = PROJECT_ROOT / "data" / "sentinel1"


def show_download_guide():
    """Display Sentinel-1 download guide with direct links."""
    print("=" * 70)
    print("Sentinel-1 SAR Scene Download Guide")
    print("=" * 70)

    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"""
📡 Sentinel-1 SAR scenes can be downloaded from these open-access platforms:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🌍 Copernicus Data Space Ecosystem (RECOMMENDED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   URL: https://dataspace.copernicus.eu/
   
   Steps:
   a) Create a free account at https://identity.dataspace.copernicus.eu/
   b) Open the Browser: https://browser.dataspace.copernicus.eu/
   c) Draw an AOI over a maritime region (e.g., strait of Malacca, 
      English Channel, Singapore Strait)
   d) Filter: Collection = Sentinel-1, Product Type = GRD
   e) Download the .SAFE folder or GeoTIFF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. 🛰️ ASF DAAC (Alaska Satellite Facility)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   URL: https://search.asf.alaska.edu/
   
   Steps:
   a) Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov/
   b) Open ASF Vertex search: https://search.asf.alaska.edu/
   c) Search: Platform = Sentinel-1, File Type = GRD_HD
   d) Draw polygon over a shipping lane
   e) Download GeoTIFF product

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 🌐 Google Earth Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   URL: https://earthengine.google.com/
   
   GEE Code Example:
   ```javascript
   var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
     .filterBounds(ee.Geometry.Point(103.84, 1.26))  // Singapore
     .filterDate('2024-01-01', '2024-02-01')
     .filter(ee.Filter.eq('instrumentMode', 'IW'))
     .select('VV')
     .first();
   
   Export.image.toDrive({{
     image: s1,
     description: 'sentinel1_singapore',
     scale: 10,
     maxPixels: 1e10
   }});
   ```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Recommended Maritime Regions for Testing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   | Region                | Lat/Lon          | Ship Density |
   |-----------------------|------------------|-------------|
   | Singapore Strait      | 1.26°N, 103.84°E | Very High   |
   | English Channel       | 50.8°N, 1.2°W    | Very High   |
   | Strait of Malacca     | 2.5°N, 101.5°E   | High        |
   | Suez Canal approach   | 30.0°N, 32.5°E   | High        |
   | Gulf of Aden          | 12.5°N, 45.0°E   | Medium      |
   | South China Sea       | 15.0°N, 115.0°E  | Medium      |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 After Download
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   1. Place the GeoTIFF (.tif) in:
      {SENTINEL_DIR}

   2. Preprocess & tile:
      python scripts/preprocess_sentinel.py -i data/sentinel1/<scene>.tif

   3. Run inference:
      python scripts/infer_and_evaluate.py -m models/yolov8s_sar.pt -s data/inference_tiles

   OR upload directly in the Streamlit dashboard:
      streamlit run dashboard/app.py
      (Navigate to 🔍 Detection → Upload Image)
""")
    print("=" * 70)


if __name__ == "__main__":
    show_download_guide()
