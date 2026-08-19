#!/usr/bin/env python3
"""
scripts/fast_download_rtc.py

High-performance multi-connection parallel downloader for ASF HyP3 RTC products.
Saturates high-bandwidth internet connections (e.g. 80-100+ Mbps) by:
- Concurrently downloading multiple products (ThreadPoolExecutor)
- Using HTTP Range headers to resume partially downloaded files
- Using large streaming memory buffers (2 MB chunks)
- Direct CloudFront CDN edge streaming
"""

import os
import sys
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from hyp3_sdk import HyP3

ASF_USER = "ishanpetkar"
ASF_PASS = "Petkar@99222"
JOB_TAG = "liss4_sikkim_rtc"
OUTPUT_DIR = "/Volumes/ishan hdd/Dataset/sentinel1_rtc"
MAX_CONCURRENT_DOWNLOADS = 4  # 4 parallel file downloads
CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB buffer

def download_file_with_resume(url: str, filename: str, expected_size: int, output_dir: str):
    target_path = os.path.join(output_dir, filename)
    part_path = target_path + ".part"
    
    # Check if already fully downloaded
    if os.path.exists(target_path):
        if os.path.getsize(target_path) == expected_size:
            print(f"⏩ [SKIP] {filename} already fully downloaded.")
            return filename, expected_size, 0
        else:
            # Move incomplete final file to part file
            os.rename(target_path, part_path)
            
    existing_bytes = 0
    if os.path.exists(part_path):
        existing_bytes = os.path.getsize(part_path)
        if existing_bytes >= expected_size:
            os.rename(part_path, target_path)
            print(f"✅ [DONE] {filename} completed from cache.")
            return filename, expected_size, 0
            
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"🔄 [RESUME] {filename} from {existing_bytes / (1024**2):.1f} MB / {expected_size / (1024**2):.1f} MB")
    else:
        print(f"📥 [START] {filename} ({expected_size / (1024**2):.1f} MB)")
        
    # Speed-dip watchdog configuration
    MIN_STREAM_SPEED_BPS = 500 * 1024  # 500 KB/s (~4 Mbps per stream)
    SPEED_WINDOW_SEC = 5.0             # Sample over 5 seconds
    MAX_RECONNECTS = 100
    
    start_time = time.time()
    downloaded_this_session = 0
    reconnect_count = 0
    
    while reconnect_count < MAX_RECONNECTS:
        # Check current progress on disk
        current_bytes = 0
        if os.path.exists(part_path):
            current_bytes = os.path.getsize(part_path)
            
        if expected_size and current_bytes >= expected_size:
            os.rename(part_path, target_path)
            elapsed = time.time() - start_time
            speed_mb = (downloaded_this_session / (1024**2)) / max(elapsed, 0.1)
            print(f"🎉 [COMPLETE] {filename} in {elapsed:.1f}s ({speed_mb:.2f} MB/s)")
            return filename, expected_size, downloaded_this_session
            
        headers = {}
        if current_bytes > 0:
            headers["Range"] = f"bytes={current_bytes}-"
            
        reconnect_needed = False
        try:
            with requests.get(url, headers=headers, stream=True, timeout=20) as r:
                if r.status_code not in (200, 206):
                    raise Exception(f"HTTP Error {r.status_code} on {filename}")
                    
                mode = "ab" if current_bytes > 0 and r.status_code == 206 else "wb"
                with open(part_path, mode) as f:
                    window_start = time.time()
                    window_bytes = 0
                    
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                            
                        f.write(chunk)
                        chunk_len = len(chunk)
                        downloaded_this_session += chunk_len
                        window_bytes += chunk_len
                        
                        now = time.time()
                        window_elapsed = now - window_start
                        
                        # Check rolling speed every 5 seconds
                        if window_elapsed >= SPEED_WINDOW_SEC:
                            current_rate = window_bytes / window_elapsed
                            if current_rate < MIN_STREAM_SPEED_BPS:
                                print(f"⚡ [AUTO-RECOVERY] {filename} dipped to {current_rate/1024:.1f} KB/s (< 500 KB/s). Reconnecting with fresh stream...")
                                reconnect_needed = True
                                break  # Break out of chunk loop to reconnect
                            window_start = now
                            window_bytes = 0
                            
            if reconnect_needed:
                reconnect_count += 1
                time.sleep(0.5)
                continue
                
            # If stream finished cleanly
            final_size = os.path.getsize(part_path)
            if expected_size and final_size == expected_size:
                os.rename(part_path, target_path)
                elapsed = time.time() - start_time
                speed_mb = (downloaded_this_session / (1024**2)) / max(elapsed, 0.1)
                print(f"🎉 [COMPLETE] {filename} in {elapsed:.1f}s ({speed_mb:.2f} MB/s)")
                return filename, final_size, downloaded_this_session
                
        except (requests.exceptions.RequestException, Exception) as e:
            reconnect_count += 1
            print(f"🔄 [RECONNECT #{reconnect_count}] {filename} connection reset ({e}). Resuming in 1s...")
            time.sleep(1.0)
            
    print(f"❌ [FAILED after {MAX_RECONNECTS} retries] {filename}")
    return filename, 0, downloaded_this_session

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if there is an in-progress file from the previous downloader (without .part)
    # and rename it to .part so we can resume it!
    for f in os.listdir(OUTPUT_DIR):
        if not f.startswith(".") and f.endswith(".zip") and not f.endswith(".part"):
            p = os.path.join(OUTPUT_DIR, f)
            if os.path.exists(p) and os.path.getsize(p) < 7 * 1024 * 1024 * 1024:
                print(f"Found partial file {f}, renaming to .part for resumption...")
                os.rename(p, p + ".part")

    print("🌍 Authenticating with ASF HyP3...")
    hyp3 = HyP3(username=ASF_USER, password=ASF_PASS)
    print(f"✅ Authenticated as: {ASF_USER}")
    
    print(f"🔎 Fetching completed jobs for tag '{JOB_TAG}'...")
    jobs = hyp3.find_jobs(name=JOB_TAG)
    succ = jobs.filter_jobs(succeeded=True, running=False, pending=False)
    print(f"📦 Total completed products: {len(succ)} / {len(jobs)}")
    
    download_tasks = []
    total_payload_bytes = 0
    for j in succ:
        if j.files:
            for f in j.files:
                fn = f['filename']
                url = f['url']
                sz = f.get('size', 0)
                download_tasks.append((url, fn, sz))
                total_payload_bytes += sz
                
    print(f"🚀 Launching Parallel Downloader with {MAX_CONCURRENT_DOWNLOADS} concurrent streams...")
    print(f"📊 Total Payload to sync: {total_payload_bytes / (1024**3):.2f} GB across {len(download_tasks)} files")
    
    overall_start = time.time()
    total_session_bytes = 0
    completed_files = 0
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        futures = {
            executor.submit(download_file_with_resume, url, fn, sz, OUTPUT_DIR): fn
            for url, fn, sz in download_tasks
        }
        
        for future in as_completed(futures):
            fn = futures[future]
            try:
                fname, fsize, session_bytes = future.result()
                total_session_bytes += session_bytes
                completed_files += 1
                elapsed = time.time() - overall_start
                avg_speed = (total_session_bytes / (1024**2)) / max(elapsed, 0.1)
                print(f"📈 Progress: {completed_files}/{len(download_tasks)} files processed | Aggregate Speed: {avg_speed:.2f} MB/s ({avg_speed*8:.1f} Mbps)")
            except Exception as exc:
                print(f"❌ File {fn} generated an exception: {exc}")
                
    print("\n🏁 All downloads completed!")

if __name__ == "__main__":
    main()
