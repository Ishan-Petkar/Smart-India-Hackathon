import os
import glob
import json
import rasterio
from rasterio.warp import transform_bounds
from datetime import datetime
import xml.etree.ElementTree as ET

def extract_metadata(data_dir="/Volumes/ishan hdd/Dataset/Paired"):
    """
    Scans the LISS4_Raw directory for GeoTIFFs and XMLs to extract
    exact spatial bounding boxes (in WGS84) and acquisition dates.
    """
    print(f"Scanning directory: {data_dir} for LISS-IV data...")
    
    # Recursively find all GeoTIFFs (Band 2, 3, or 4)
    # LISS-IV usually names them with 'B2', 'B3', 'B4'
    tif_files = glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
    
    if not tif_files:
        print("No .tif files found. Please ensure you have extracted the downloaded .zip files.")
        return []

    processed_scenes = {}
    
    for tif_path in tif_files:
        # Group by folder (which usually represents one scene)
        scene_dir = os.path.dirname(tif_path)
        scene_name = os.path.basename(scene_dir)
        
        if scene_name in processed_scenes:
            continue
            
        print(f"\nProcessing scene: {scene_name}")
        scene_metadata = {
            "scene_name": scene_name,
            "directory": scene_dir
        }
        
        # 1. Extract Spatial Bounds using rasterio
        try:
            with rasterio.open(tif_path) as src:
                # Get bounds in native CRS
                bounds = src.bounds
                crs = src.crs
                
                # Transform to EPSG:4326 (WGS84 Lat/Lon) for API queries (Sentinel/DEM)
                min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, 'EPSG:4326', *bounds)
                
                scene_metadata['bounds_wgs84'] = {
                    'min_lon': min_lon,
                    'min_lat': min_lat,
                    'max_lon': max_lon,
                    'max_lat': max_lat
                }
                scene_metadata['native_crs'] = crs.to_string()
                print(f"  [+] Bounds (WGS84): [{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]")
        except Exception as e:
            print(f"  [-] Failed to read GeoTIFF with rasterio: {e}")
            continue
            
        # 2. Extract Date from Filename (e.g. R2F03JUN2026...)
        scene_date = None
        if scene_name.startswith("R2F") or scene_name.startswith("R2A"):
            date_str = scene_name[3:12] # e.g., '03JUN2026'
            try:
                # Validate it's a date
                parsed_date = datetime.strptime(date_str, "%d%b%Y")
                scene_date = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
                
        if not scene_date:
            print("  [!] Could not parse date from filename. Will require manual validation.")
            scene_date = "UNKNOWN_DATE"
            
        scene_metadata['acquisition_date'] = scene_date
        print(f"  [+] Acquisition Date: {scene_date}")
        
        processed_scenes[scene_name] = scene_metadata
        
    # Save the consolidated metadata to a JSON file for the next script to use
    output_file = os.path.join(data_dir, "liss4_metadata.json")
    with open(output_file, 'w') as f:
        json.dump(list(processed_scenes.values()), f, indent=4)
        
    print(f"\nSuccessfully extracted metadata for {len(processed_scenes)} scenes.")
    print(f"Metadata saved to: {output_file}")
    return list(processed_scenes.values())

if __name__ == "__main__":
    # Ensure the target directory exists
    target_dir = "/Volumes/ishan hdd/Dataset/Paired"
    os.makedirs(target_dir, exist_ok=True)
    extract_metadata(target_dir)
