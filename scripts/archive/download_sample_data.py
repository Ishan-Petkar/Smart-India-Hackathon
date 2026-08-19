import os
import json
import requests

DATA_DIR = os.path.abspath("data/raw")
S2_DIR = os.path.join(DATA_DIR, "sentinel2")
S1_DIR = os.path.join(DATA_DIR, "sentinel1")
DEM_DIR = os.path.join(DATA_DIR, "dem")

os.makedirs(S2_DIR, exist_ok=True)
os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(DEM_DIR, exist_ok=True)

# Bounding Box for NER (Meghalaya / Assam)
# Lon: 91.5 to 92.0, Lat: 25.5 to 26.0
BBOX = [91.5, 25.5, 92.0, 26.0]

def search_stac(collection, datetime_range, query_filters=None, limit=2):
    url = "https://earth-search.aws.element84.com/v1/search"
    payload = {
        "collections": [collection],
        "bbox": BBOX,
        "datetime": datetime_range,
        "limit": limit
    }
    if query_filters:
        payload["query"] = query_filters
    
    print(f"Querying STAC API for {collection} ({datetime_range})...")
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    print(f"Found {len(features)} matching scenes.")
    return features

def download_file(url, out_path):
    print(f"Downloading: {os.path.basename(out_path)} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total_bytes = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    print(f"\rProgress: {pct:.1f}% ({downloaded/(1024*1024):.1f} MB)", end="", flush=True)
    print("\nDownload complete.")

def main():
    print("=== 1. Searching for Sentinel-2 Optical (Cloudy vs Clear) ===")
    # 1. Cloudy Sentinel-2 (Monsoon 2024)
    cloudy_features = search_stac(
        "sentinel-2-l2a",
        "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        query_filters={"eo:cloud_cover": {"gte": 50}},
        limit=1
    )
    
    # 2. Clear Sentinel-2 (Dry Season 2024)
    clear_features = search_stac(
        "sentinel-2-l2a",
        "2024-01-01T00:00:00Z/2024-03-31T23:59:59Z",
        query_filters={"eo:cloud_cover": {"lte": 10}},
        limit=1
    )

    if cloudy_features:
        f = cloudy_features[0]
        print(f"\nSelected Cloudy Scene: {f['id']} (Cloud cover: {f['properties'].get('eo:cloud_cover')}%)")
        # Download key VNIR bands (Green: B03, Red: B04, NIR: B08) + Visual TCI
        for band in ["red", "green", "nir", "visual"]:
            if band in f.get("assets", {}):
                url = f["assets"][band]["href"]
                ext = url.split("?")[0].split(".")[-1]
                out_file = os.path.join(S2_DIR, f"cloudy_{band}.{ext}")
                download_file(url, out_file)

    if clear_features:
        f = clear_features[0]
        print(f"\nSelected Clear Scene: {f['id']} (Cloud cover: {f['properties'].get('eo:cloud_cover')}%)")
        for band in ["red", "green", "nir", "visual"]:
            if band in f.get("assets", {}):
                url = f["assets"][band]["href"]
                ext = url.split("?")[0].split(".")[-1]
                out_file = os.path.join(S2_DIR, f"clear_{band}.{ext}")
                download_file(url, out_file)

    print("\n=== 2. Searching for Sentinel-1 SAR ===")
    s1_features = search_stac(
        "sentinel-1-grd",
        "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z",
        limit=1
    )
    if s1_features:
        f = s1_features[0]
        print(f"Selected Sentinel-1 Scene: {f['id']}")
        for pol in ["vv", "vh"]:
            if pol in f.get("assets", {}):
                url = f["assets"][pol]["href"]
                ext = url.split("?")[0].split(".")[-1]
                out_file = os.path.join(S1_DIR, f"sar_{pol}.{ext}")
                download_file(url, out_file)

    print("\n=== 3. Searching for Copernicus DEM (GLO-30) ===")
    dem_features = search_stac(
        "cop-dem-glo-30",
        "2020-01-01T00:00:00Z/2022-12-31T23:59:59Z",
        limit=1
    )
    if dem_features:
        f = dem_features[0]
        print(f"Selected DEM Tile: {f['id']}")
        if "data" in f.get("assets", {}):
            url = f["assets"]["data"]["href"]
            out_file = os.path.join(DEM_DIR, "copernicus_dem_30m.tif")
            download_file(url, out_file)

    print("\nAll datasets downloaded successfully!")

if __name__ == "__main__":
    main()
