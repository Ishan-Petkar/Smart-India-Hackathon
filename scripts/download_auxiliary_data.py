import os
import json
import requests
import subprocess
import pickle
from datetime import datetime, timedelta
import asf_search as asf
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    import planetary_computer
except ImportError:
    planetary_computer = None

METADATA_FILE = "/Volumes/ishan hdd/Dataset/Paired/liss4_metadata.json"
BASE_OUT_DIR = "/Volumes/ishan hdd/Dataset/Auxiliary"
ARIA2C = "/opt/homebrew/bin/aria2c"

# NASA Credentials (provided by user)
ASF_USER = "ishanpetkar"
ASF_PASS = "Petkar@99222"

# ── aria2c-powered download engine ──────────────────────────────────────────

def download_with_aria2c(url, out_path, cookie_header=None):
    """Download a single file using aria2c with 16 parallel segments.
       If speed drops below threshold, aria2c's --lowest-speed-limit
       auto-aborts and we retry with fresh connections."""
    if os.path.exists(out_path):
        return True

    # If it's a DEM planetary computer URL, make sure it is signed freshly
    if "blob.core.windows.net" in url and planetary_computer:
        try:
            url = planetary_computer.sign(url)
        except Exception:
            pass

    out_dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)

    cmd = [
        ARIA2C,
        "-c",                             # Continue partially downloaded file
        "--split=16",                     # Split file into 16 segments
        "--max-connection-per-server=16",  # 16 connections to same server
        "--min-split-size=1M",            # Split even small files
        "--max-tries=15",                 # Retry up to 15 times
        "--retry-wait=3",                 # Wait 3s between retries
        "--timeout=30",                   # 30s timeout per connection
        "--connect-timeout=15",           # 15s connection timeout
        "--lowest-speed-limit=50K",       # Auto-abort if speed < 50KB/s
        "--file-allocation=none",         # Fast allocation on HDD
        "--auto-file-renaming=false",     # Don't rename if exists
        "--allow-overwrite=false",        # Don't overwrite
        "--console-log-level=error",      # Quiet output
        "--summary-interval=0",           # No summary
        f"--dir={out_dir}",
        f"--out={out_name}",
    ]

    if cookie_header:
        cmd.append(f"--header=Cookie: {cookie_header}")

    cmd.append(url)

    for attempt in range(5):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 or os.path.exists(out_path):
                # Clean up .aria2 file if present and main file exists
                aria2_file = out_path + ".aria2"
                if os.path.exists(aria2_file):
                    try:
                        os.remove(aria2_file)
                    except OSError:
                        pass
                return True
            elif result.returncode == 13:
                return True  # File already exists
            else:
                time.sleep(2 * (attempt + 1))
                continue
        except subprocess.TimeoutExpired:
            time.sleep(2)
            continue
        except Exception as e:
            print(f"      [!] aria2c exception: {e}")
            return False

    print(f"      [!] Failed after 5 attempts: {out_name}")
    return False


def batch_download_aria2c(tasks, cookie_header=None, max_workers=4, label=""):
    """Download multiple files in parallel, each with aria2c multi-segment.
       Filters out already-existing files before starting."""
    # Filter out already-downloaded files
    pending = [(url, path) for url, path in tasks if not os.path.exists(path)]
    skipped = len(tasks) - len(pending)

    if skipped > 0:
        print(f"    [{label}] Skipped {skipped} existing files, downloading {len(pending)} remaining...")
    if not pending:
        print(f"    [{label}] All files already downloaded!")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_with_aria2c, url, path, cookie_header): os.path.basename(path)
            for url, path in pending
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            future.result()
            if done_count % 10 == 0 or done_count == len(pending):
                print(f"    [{label}] {done_count}/{len(pending)} files processed")


# ── Data source query functions ─────────────────────────────────────────────

def get_s1_results(min_lon, min_lat, max_lon, max_lat, target_date):
    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    start = (date_obj - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    end = (date_obj + timedelta(days=7)).strftime("%Y-%m-%dT23:59:59Z")
    wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    try:
        return asf.geo_search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel="GRD_HD",
            intersectsWith=wkt,
            start=start,
            end=end
        )
    except Exception:
        return []

def get_s2_assets(min_lon, min_lat, max_lon, max_lat, target_date):
    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    start = (date_obj - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    end = (date_obj + timedelta(days=7)).strftime("%Y-%m-%dT23:59:59Z")
    stac_url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "datetime": f"{start}/{end}",
        "limit": 5,
        "query": {"eo:cloud_cover": {"lt": 70}}
    }
    try:
        resp = requests.post(stac_url, json=payload).json()
        return resp.get('features', [])
    except Exception:
        return []

def get_dem_assets(min_lon, min_lat, max_lon, max_lat):
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["cop-dem-glo-30"],
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "limit": 5
    }
    try:
        resp = requests.post(stac_url, json=payload).json()
        return resp.get('features', [])
    except Exception:
        return []


# ── Main pipeline (TWO-PHASE: Collect all URLs, then download non-stop) ────

def main():
    if not os.path.exists(METADATA_FILE):
        print("Metadata file not found!")
        return

    with open(METADATA_FILE, 'r') as f:
        scenes = json.load(f)

    # Load NASA cookies and build a cookie header string for aria2c
    print("Loading NASA Authentication Cookies...")
    cookie_header = None
    try:
        with open("nasa_cookies.pkl", "rb") as f:
            cookies = pickle.load(f)
            cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        print(f"✅ Loaded {len(cookies)} cookies for NASA auth!")
    except FileNotFoundError:
        print("❌ Error: nasa_cookies.pkl not found! Please run auth_nasa.py first.")
        return

    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Collect ALL download URLs across all 14 scenes (no downloads)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 1: Collecting all download URLs across all scenes...")
    print("="*70)

    all_s1_tasks = []  # (url, out_path) — needs NASA cookies
    all_s2_tasks = []  # (url, out_path) — no auth
    all_dem_tasks = [] # (url, out_path) — no auth

    for i, scene in enumerate(scenes):
        name = scene['scene_name']
        bounds = scene['bounds_wgs84']
        date = scene['acquisition_date']

        print(f"\n  [{i+1}/{len(scenes)}] Querying data for: {name}")
        if date == "UNKNOWN_DATE":
            print("    Skipping (unknown date)")
            continue

        scene_dir = os.path.join(BASE_OUT_DIR, name)

        # Sentinel-1
        s1_dir = os.path.join(scene_dir, "Sentinel1")
        os.makedirs(s1_dir, exist_ok=True)
        s1_results = get_s1_results(bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat'], date)
        for r in s1_results:
            url = r.properties['url']
            filename = url.split('/')[-1]
            all_s1_tasks.append((url, os.path.join(s1_dir, filename)))
        print(f"    S1: {len(s1_results)} scenes found")

        # Sentinel-2
        s2_dir = os.path.join(scene_dir, "Sentinel2")
        os.makedirs(s2_dir, exist_ok=True)
        s2_features = get_s2_assets(bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat'], date)
        s2_count = 0
        for feature in s2_features:
            s2_id = feature['id']
            assets = feature.get('assets', {})
            for t in ['blue', 'green', 'red', 'nir', 'scl']:
                if t in assets:
                    href = assets[t]['href']
                    ext = href.split('?')[0].split('.')[-1]
                    out_name = f"{s2_id}_{t}.{ext}"
                    all_s2_tasks.append((href, os.path.join(s2_dir, out_name)))
                    s2_count += 1
        print(f"    S2: {s2_count} band files found")

        # DEM
        dem_dir = os.path.join(scene_dir, "DEM")
        os.makedirs(dem_dir, exist_ok=True)
        dem_features = get_dem_assets(bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat'])
        dem_count = 0
        for feature in dem_features:
            dem_id = feature['id']
            if 'data' in feature.get('assets', {}):
                href = feature['assets']['data']['href']
                if planetary_computer:
                    href = planetary_computer.sign(href)
                out_name = f"{dem_id}.tif"
                all_dem_tasks.append((href, os.path.join(dem_dir, out_name)))
                dem_count += 1
        print(f"    DEM: {dem_count} tiles found")

    total_files = len(all_s1_tasks) + len(all_s2_tasks) + len(all_dem_tasks)
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE: {total_files} total files queued")
    print(f"  Sentinel-1: {len(all_s1_tasks)} files (NASA auth)")
    print(f"  Sentinel-2: {len(all_s2_tasks)} files (AWS)")
    print(f"  DEM:        {len(all_dem_tasks)} files (Azure)")
    print(f"{'='*70}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: NON-STOP continuous download blast (no gaps between scenes)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 2: DOWNLOADING ALL FILES (non-stop continuous blast)...")
    print("="*70)

    # Download S1 first (largest files, ~80% of total data)
    if all_s1_tasks:
        print(f"\n🛰️  Downloading ALL {len(all_s1_tasks)} Sentinel-1 files (16-segment aria2c)...")
        batch_download_aria2c(all_s1_tasks, cookie_header=cookie_header, max_workers=4, label="S1")

    # Download S2 next (medium files)
    if all_s2_tasks:
        print(f"\n🌍 Downloading ALL {len(all_s2_tasks)} Sentinel-2 band files (16-segment aria2c)...")
        batch_download_aria2c(all_s2_tasks, cookie_header=None, max_workers=4, label="S2")

    # Download DEM last (smallest files)
    if all_dem_tasks:
        print(f"\n⛰️  Downloading ALL {len(all_dem_tasks)} Copernicus DEM tiles (16-segment aria2c)...")
        batch_download_aria2c(all_dem_tasks, cookie_header=None, max_workers=4, label="DEM")

    print("\n✅ All downloads complete!")

if __name__ == "__main__":
    main()
