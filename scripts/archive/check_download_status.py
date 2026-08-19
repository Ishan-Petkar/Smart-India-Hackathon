#!/usr/bin/env python3
"""
scripts/check_download_status.py
Minimalist plain status monitor for RTC downloads.
"""

import os
import time
from datetime import datetime, timezone, timedelta

TOTAL_PACKAGES = 50
TOTAL_PAYLOAD_GB = 438.44
OUTPUT_DIR = "/Volumes/ishan hdd/Dataset/sentinel1_rtc"

def get_ist_time():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_tz).strftime("%I:%M:%S %p IST")

def main():
    ist_time = get_ist_time()
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Directory {OUTPUT_DIR} not found.")
        return

    all_files = [f for f in os.listdir(OUTPUT_DIR) if not f.startswith(".")]
    completed_zips = [f for f in all_files if f.endswith(".zip") and not f.endswith(".part")]
    in_progress_parts = [f for f in all_files if f.endswith(".part")]
    
    total_bytes_on_disk = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in all_files)
    
    # 1s speed check
    t1 = time.time()
    b1 = total_bytes_on_disk
    time.sleep(1.0)
    b2 = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if not f.startswith("."))
    t2 = time.time()
    
    speed_bps = max((b2 - b1) / max(t2 - t1, 0.1), 0)
    speed_mb_sec = speed_bps / (1024**2)
    speed_mbps = (speed_bps * 8) / (1024**2)
    
    total_gb = b2 / (1024**3)
    pct = (total_gb / TOTAL_PAYLOAD_GB) * 100
    
    remaining_bytes = max((TOTAL_PAYLOAD_GB * (1024**3)) - b2, 0)
    if speed_bps > 100 * 1024:
        eta_sec = remaining_bytes / speed_bps
        eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
    else:
        eta_str = "N/A"

    print(f"Time: {ist_time}")
    print(f"Completed Packages: {len(completed_zips)}/{TOTAL_PACKAGES} (Active: {len(in_progress_parts)}, Queued: {TOTAL_PACKAGES - len(completed_zips) - len(in_progress_parts)})")
    print(f"Data Progress: {total_gb:.2f} GB / {TOTAL_PAYLOAD_GB:.2f} GB ({pct:.2f}%)")
    print(f"Speed: {speed_mb_sec:.2f} MB/s ({speed_mbps:.1f} Mbps) | ETA: {eta_str}")
    print("Active Streams:")
    for f in sorted(in_progress_parts):
        sz_gb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**3)
        stream_pct = min((sz_gb / 8.8) * 100, 99.0)
        name = f.replace(".zip.part", "")
        print(f"  - {name[:35]}...: {sz_gb:.2f} GB / ~8.8 GB ({stream_pct:.1f}%)")
    if completed_zips:
        print("Completed:")
        for f in sorted(completed_zips):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
