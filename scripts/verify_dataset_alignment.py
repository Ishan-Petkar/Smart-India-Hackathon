import os
import json
import zipfile
import glob
import xml.etree.ElementTree as ET
from datetime import datetime
import rasterio
from shapely.geometry import box, Polygon

LISS4_DIR = "/Volumes/ishan hdd/Dataset/Paired"
AUX_DIR = "/Volumes/ishan hdd/Dataset/Auxiliary"
METADATA_FILE = "/Volumes/ishan hdd/Dataset/Paired/liss4_metadata.json"
OUTPUT_LEDGER = "/Volumes/ishan hdd/Dataset/dataset_verification_ledger.json"

def get_polygon_from_bounds(min_lon, min_lat, max_lon, max_lat):
    return box(min_lon, min_lat, max_lon, max_lat)

def verify_liss4_metadata(scene_dir, scene_meta):
    """Verifies LISS-IV header metadata without unpacking or loading full rasters."""
    name = os.path.basename(scene_dir)
    res = {
        "scene_name": name,
        "valid": True,
        "errors": [],
        "bands": {},
        "crs": None,
        "pixel_size": None,
        "width": None,
        "height": None,
        "bounds_wgs84": scene_meta.get("bounds_wgs84", {}),
        "acquisition_date": scene_meta.get("acquisition_date", "UNKNOWN")
    }

    for b in ["BAND2", "BAND3", "BAND4"]:
        b_path = os.path.join(scene_dir, f"{b}.tif")
        if not os.path.exists(b_path):
            res["valid"] = False
            res["errors"].append(f"Missing {b}.tif")
            continue
        try:
            with rasterio.open(b_path) as src:
                if res["crs"] is None:
                    res["crs"] = str(src.crs)
                    res["pixel_size"] = (round(src.res[0], 3), round(src.res[1], 3))
                    res["width"] = src.width
                    res["height"] = src.height

                res["bands"][b] = {
                    "dtype": str(src.dtypes[0]),
                    "shape": [src.height, src.width],
                    "size_mb": round(os.path.getsize(b_path) / (1024*1024), 2),
                    "nodata": src.nodata
                }
        except Exception as e:
            res["valid"] = False
            res["errors"].append(f"Header read error {b}.tif: {e}")

    return res

def verify_s1_metadata_only(zip_path, liss_poly, liss_date_str):
    """Inspects Sentinel-1 zip metadata without unpacking."""
    name = os.path.basename(zip_path)
    res = {
        "filename": name,
        "valid": True,
        "errors": [],
        "size_mb": round(os.path.getsize(zip_path) / (1024*1024), 2),
        "polarizations": [],
        "orbit_direction": None,
        "acq_time": None,
        "delta_days": None,
        "overlaps_liss": False
    }

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            manifest_names = [n for n in zf.namelist() if n.endswith("manifest.safe")]
            if not manifest_names:
                res["valid"] = False
                res["errors"].append("manifest.safe not found in zip")
                return res

            manifest_data = zf.read(manifest_names[0])
            root = ET.fromstring(manifest_data)
            
            for pol_elem in root.findall(".//{http://www.esa.int/safe/sentinel-1.0}transmitterReceiverPolarisation"):
                if pol_elem.text not in res["polarizations"]:
                    res["polarizations"].append(pol_elem.text)

            pass_elem = root.find(".//{http://www.esa.int/safe/sentinel-1.0}pass")
            if pass_elem is not None:
                res["orbit_direction"] = pass_elem.text

            start_elem = root.find(".//{http://www.esa.int/safe/sentinel-1.0}startTime")
            if start_elem is not None and start_elem.text:
                res["acq_time"] = start_elem.text[:19]
                if liss_date_str != "UNKNOWN_DATE":
                    s1_dt = datetime.strptime(res["acq_time"][:10], "%Y-%m-%d")
                    liss_dt = datetime.strptime(liss_date_str, "%Y-%m-%d")
                    res["delta_days"] = (s1_dt - liss_dt).days

            coords_elem = root.find(".//{http://www.opengis.net/gml}coordinates")
            if coords_elem is not None and coords_elem.text:
                raw_coords = coords_elem.text.strip().split()
                poly_pts = []
                for pt in raw_coords:
                    lat, lon = map(float, pt.split(','))
                    poly_pts.append((lon, lat))
                if len(poly_pts) >= 3:
                    s1_poly = Polygon(poly_pts)
                    res["overlaps_liss"] = s1_poly.intersects(liss_poly)
    except Exception as e:
        res["valid"] = False
        res["errors"].append(f"Header exception: {e}")

    return res

def verify_s2_metadata_only(s2_dir, liss_poly, liss_date_str):
    """Inspects Sentinel-2 GeoTIFF headers only without reading pixels."""
    tifs = glob.glob(os.path.join(s2_dir, "*.tif"))
    granules = {}

    for tif in tifs:
        name = os.path.basename(tif)
        parts = name.replace(".tif", "").split("_")
        if len(parts) >= 6:
            granule_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            band = parts[-1]
            date_raw = parts[2]
            try:
                date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            except Exception:
                date_str = "UNKNOWN"
        else:
            granule_id = name
            band = "unknown"
            date_str = "UNKNOWN"

        if granule_id not in granules:
            granules[granule_id] = {
                "granule_id": granule_id,
                "date": date_str,
                "delta_days": None,
                "bands": {},
                "valid": True,
                "errors": [],
                "crs": None,
                "overlaps_liss": False
            }
            if date_str != "UNKNOWN" and liss_date_str != "UNKNOWN_DATE":
                s2_dt = datetime.strptime(date_str, "%Y-%m-%d")
                liss_dt = datetime.strptime(liss_date_str, "%Y-%m-%d")
                granules[granule_id]["delta_days"] = (s2_dt - liss_dt).days

        try:
            with rasterio.open(tif) as src:
                granules[granule_id]["crs"] = str(src.crs)
                s2_poly = box(*src.bounds)
                granules[granule_id]["overlaps_liss"] = s2_poly.intersects(liss_poly)
                
                granules[granule_id]["bands"][band] = {
                    "shape": [src.height, src.width],
                    "res": (round(src.res[0], 2), round(src.res[1], 2)),
                    "dtype": str(src.dtypes[0]),
                    "size_mb": round(os.path.getsize(tif) / (1024*1024), 2)
                }
        except Exception as e:
            granules[granule_id]["valid"] = False
            granules[granule_id]["errors"].append(f"Header error {name}: {e}")

    return granules

def verify_dem_metadata_only(dem_dir, liss_poly):
    """Inspects DEM GeoTIFF headers only without loading full grids."""
    tifs = glob.glob(os.path.join(dem_dir, "*.tif"))
    dem_results = []

    for tif in tifs:
        name = os.path.basename(tif)
        res = {
            "filename": name,
            "valid": True,
            "errors": [],
            "shape": None,
            "res": None,
            "crs": None,
            "overlaps_liss": False
        }
        try:
            with rasterio.open(tif) as src:
                res["crs"] = str(src.crs)
                res["shape"] = [src.height, src.width]
                res["res"] = (round(src.res[0], 5), round(src.res[1], 5))
                dem_poly = box(*src.bounds)
                res["overlaps_liss"] = dem_poly.intersects(liss_poly)
        except Exception as e:
            res["valid"] = False
            res["errors"].append(f"Header error {name}: {e}")

        dem_results.append(res)

    return dem_results

def main():
    print("=" * 80)
    print("🚀 STARTING FAST METADATA-ONLY VERIFICATION (LISS-IV -> S1 -> S2 -> DEM)")
    print("=" * 80)

    if not os.path.exists(METADATA_FILE):
        print(f"❌ Error: Metadata file {METADATA_FILE} not found!")
        return

    with open(METADATA_FILE, 'r') as f:
        scenes_meta = json.load(f)

    ledger = {
        "verification_time": datetime.now().isoformat(),
        "total_scenes": len(scenes_meta),
        "scenes": []
    }

    summary_table = []

    for idx, s_meta in enumerate(scenes_meta):
        scene_name = s_meta["scene_name"]
        date_str = s_meta["acquisition_date"]
        bounds = s_meta["bounds_wgs84"]
        liss_poly = get_polygon_from_bounds(bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"])

        print(f"\n[{idx+1}/{len(scenes_meta)}] 🛰️ Scene: {scene_name}")
        print(f"     Target Date: {date_str} | Bounds: [{bounds['min_lon']:.2f}, {bounds['min_lat']:.2f}, {bounds['max_lon']:.2f}, {bounds['max_lat']:.2f}]")

        # 1. LISS-IV
        liss_path = os.path.join(LISS4_DIR, scene_name)
        liss_res = verify_liss4_metadata(liss_path, s_meta)
        print(f"     [LISS-IV] Valid: {'✅' if liss_res['valid'] else '❌'} | Bands: {list(liss_res['bands'].keys())} | Res: {liss_res['pixel_size']} m | CRS: {liss_res['crs']}")

        # 2. Sentinel-1 (Metadata only from manifest inside zip)
        s1_dir = os.path.join(AUX_DIR, scene_name, "Sentinel1")
        s1_zips = glob.glob(os.path.join(s1_dir, "*.zip"))
        s1_results = [verify_s1_metadata_only(zp, liss_poly, date_str) for zp in s1_zips]
        valid_s1 = [s for s in s1_results if s["valid"]]
        best_s1 = min(valid_s1, key=lambda x: abs(x["delta_days"])) if valid_s1 and date_str != "UNKNOWN_DATE" else (valid_s1[0] if valid_s1 else None)
        print(f"     [Sentinel-1] Files: {len(s1_results)} | Valid: {len(valid_s1)} | Best Match: Δt={best_s1['delta_days'] if best_s1 else 'N/A'}d ({best_s1['orbit_direction'] if best_s1 else 'N/A'}, Pol: {best_s1['polarizations'] if best_s1 else 'N/A'})")

        # 3. Sentinel-2 (GeoTIFF headers only)
        s2_dir = os.path.join(AUX_DIR, scene_name, "Sentinel2")
        s2_granules = verify_s2_metadata_only(s2_dir, liss_poly, date_str)
        valid_s2 = [g for g in s2_granules.values() if g["valid"] and len(g["bands"]) >= 4]
        best_s2 = min(valid_s2, key=lambda x: abs(x["delta_days"])) if valid_s2 and date_str != "UNKNOWN_DATE" else (valid_s2[0] if valid_s2 else None)
        print(f"     [Sentinel-2] Dates/Granules: {len(s2_granules)} | Complete (≥4 bands): {len(valid_s2)} | Best Match: Δt={best_s2['delta_days'] if best_s2 else 'N/A'}d (Bands: {list(best_s2['bands'].keys()) if best_s2 else 'N/A'})")

        # 4. DEM (GeoTIFF headers only)
        dem_dir = os.path.join(AUX_DIR, scene_name, "DEM")
        dem_results = verify_dem_metadata_only(dem_dir, liss_poly)
        valid_dem = [d for d in dem_results if d["valid"]]
        print(f"     [Copernicus DEM] Tiles: {len(dem_results)} | Valid: {len(valid_dem)} | Overlaps LISS: {all(d['overlaps_liss'] for d in valid_dem)}")

        scene_entry = {
            "scene_name": scene_name,
            "liss4": liss_res,
            "sentinel1": {
                "total_files": len(s1_results),
                "valid_files": len(valid_s1),
                "best_match": best_s1,
                "all_files": s1_results
            },
            "sentinel2": {
                "total_granules": len(s2_granules),
                "valid_granules": len(valid_s2),
                "best_match": best_s2,
                "all_granules": s2_granules
            },
            "dem": {
                "total_tiles": len(dem_results),
                "valid_tiles": len(valid_dem),
                "tiles": dem_results
            },
            "is_fully_matched": (liss_res["valid"] and len(valid_s1) > 0 and len(valid_s2) > 0 and len(valid_dem) > 0)
        }
        ledger["scenes"].append(scene_entry)

        summary_table.append({
            "name": scene_name[:22] + "...",
            "date": date_str,
            "liss4_ok": "✅" if liss_res["valid"] else "❌",
            "s1_count": f"{len(valid_s1)}/{len(s1_results)}",
            "s1_best_dt": f"{best_s1['delta_days']}d" if best_s1 and best_s1['delta_days'] is not None else "N/A",
            "s2_granules": f"{len(valid_s2)}/{len(s2_granules)}",
            "s2_best_dt": f"{best_s2['delta_days']}d" if best_s2 and best_s2['delta_days'] is not None else "N/A",
            "dem_tiles": f"{len(valid_dem)}/{len(dem_results)}",
            "fully_matched": "✅ READY" if scene_entry["is_fully_matched"] else "❌ INCOMPLETE"
        })

    with open(OUTPUT_LEDGER, "w") as f:
        json.dump(ledger, f, indent=2)

    print("\n" + "=" * 90)
    print("📊 MULTI-MODAL METADATA ALIGNMENT & VERIFICATION MATRIX")
    print("=" * 90)
    header = f"{'Scene Name':<26} | {'Acq Date':<10} | {'LISS4':<6} | {'S1 (Val/Tot)':<12} | {'S1 Δt':<7} | {'S2 (Val/Tot)':<12} | {'S2 Δt':<7} | {'DEM':<5} | {'Status':<10}"
    print(header)
    print("-" * len(header))
    for row in summary_table:
        print(f"{row['name']:<26} | {row['date']:<10} | {row['liss4_ok']:<6} | {row['s1_count']:<12} | {row['s1_best_dt']:<7} | {row['s2_granules']:<12} | {row['s2_best_dt']:<7} | {row['dem_tiles']:<5} | {row['fully_matched']:<10}")

    print("=" * 90)
    print(f"✅ Metadata verification ledger saved to: {OUTPUT_LEDGER}")

if __name__ == "__main__":
    main()
