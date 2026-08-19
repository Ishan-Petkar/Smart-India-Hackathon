#!/usr/bin/env python3
"""
scripts/submit_hyp3_rtc.py

Submits Sentinel-1 GRD granules associated with the 14 LISS-IV scenes
to ASF HyP3 for Radiometric Terrain Correction (RTC).
Also provides monitoring and auto-download functionality for completed products.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from hyp3_sdk import HyP3

# Configuration
ASF_USER = "ishanpetkar"
ASF_PASS = "Petkar@99222"
JOB_TAG = "liss4_sikkim_rtc"
LEDGER_PATH = "/Volumes/ishan hdd/Dataset/dataset_verification_ledger.json"
JOBS_RECORD_PATH = "/Volumes/ishan hdd/Dataset/hyp3_rtc_jobs.json"
OUTPUT_DIR = "/Volumes/ishan hdd/Dataset/sentinel1_rtc"

def get_unique_granules(ledger_path: str, best_only: bool = False):
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(f"Ledger not found at: {ledger_path}")
    
    with open(ledger_path) as f:
        ledger = json.load(f)
    
    granules = set()
    scenes = ledger.get("scenes", [])
    
    for sc in scenes:
        s1_info = sc.get("sentinel1", {})
        if best_only:
            best = s1_info.get("best_match")
            if best and best.get("filename"):
                granules.add(best["filename"].replace(".zip", ""))
        else:
            for f_info in s1_info.get("all_files", []):
                fn = f_info.get("filename")
                if fn:
                    granules.add(fn.replace(".zip", ""))
                    
    return sorted(list(granules))

def submit_jobs(hyp3: HyP3, granules: list, dry_run: bool = False):
    print(f"\n🔍 Found {len(granules)} unique Sentinel-1 granules to process.")
    
    # Check current user credits
    credits_remaining = hyp3.check_credits()
    cost_per_job = 60  # 10m RTC_GAMMA
    total_cost = len(granules) * cost_per_job
    print(f"💰 Account credits: {credits_remaining}")
    print(f"💳 Estimated cost for {len(granules)} jobs @ 10m resolution: {total_cost} credits")
    
    if total_cost > credits_remaining:
        print(f"⚠️ Warning: Total cost ({total_cost}) exceeds available credits ({credits_remaining}).")
        if not dry_run:
            print("❌ Aborting submission to avoid exceeding credit limit.")
            return None

    # Check for existing jobs with the same tag
    print(f"🔎 Checking existing HyP3 jobs with tag '{JOB_TAG}'...")
    existing_jobs = hyp3.find_jobs(name=JOB_TAG)
    existing_granules = set()
    for j in existing_jobs:
        for g in j.job_parameters.get("granules", []):
            existing_granules.add(g)
            
    granules_to_submit = [g for g in granules if g not in existing_granules]
    print(f"ℹ️ Already submitted: {len(existing_granules)}")
    print(f"🚀 To be submitted now: {len(granules_to_submit)}")
    
    if dry_run:
        print("\n[DRY RUN] Would submit the following granules:")
        for g in granules_to_submit:
            print(f"  • {g}")
        return None
        
    if not granules_to_submit:
        print("✅ All granules have already been submitted!")
        return existing_jobs
        
    prepared_jobs = []
    for g in granules_to_submit:
        job = HyP3.prepare_rtc_job(
            granule=g,
            name=JOB_TAG,
            dem_matching=False,
            include_dem=True,
            include_inc_map=True,
            include_scattering_area=True,
            radiometry="gamma0",
            resolution=10,
            scale="power",
            speckle_filter=False,
            dem_name="copernicus"
        )
        prepared_jobs.append(job)
        
    print(f"📡 Submitting {len(prepared_jobs)} jobs to HyP3...")
    submitted_batch = hyp3.submit_prepared_jobs(prepared_jobs)
    print(f"✅ Successfully submitted batch with {len(submitted_batch)} jobs!")
    
    # Save job IDs to record
    all_jobs = hyp3.find_jobs(name=JOB_TAG)
    jobs_summary = [j.to_dict() for j in all_jobs]
        
    os.makedirs(os.path.dirname(JOBS_RECORD_PATH), exist_ok=True)
    with open(JOBS_RECORD_PATH, "w") as f:
        json.dump(jobs_summary, f, indent=2, default=str)
    print(f"📝 Saved jobs registry ({len(jobs_summary)} jobs) to: {JOBS_RECORD_PATH}")
    
    return all_jobs

def monitor_and_download(hyp3: HyP3, download: bool = True):
    print(f"\n📊 Querying status of jobs with tag '{JOB_TAG}'...")
    jobs = hyp3.find_jobs(name=JOB_TAG)
    if not jobs:
        print("❌ No jobs found with tag:", JOB_TAG)
        return
        
    status_counts = {}
    for j in jobs:
        st = j.status_code
        status_counts[st] = status_counts.get(st, 0) + 1
        
    print("Job Status Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  • {status}: {count}/{len(jobs)}")
        
    if download:
        succeeded_jobs = jobs.filter_jobs(succeeded=True, failed=False, running=False, pending=False)
        if len(succeeded_jobs) > 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"\n📥 Downloading {len(succeeded_jobs)} completed products to {OUTPUT_DIR}...")
            succeeded_jobs.download_files(OUTPUT_DIR)
            print("✅ Downloads completed!")
        else:
            print("\n⏳ No jobs have completed yet. They are currently queued/running in ASF cloud.")

def main():
    parser = argparse.ArgumentParser(description="Submit and manage ASF HyP3 RTC jobs for LISS-IV dataset.")
    parser.add_argument("--submit", action="store_true", help="Submit RTC jobs to HyP3")
    parser.add_argument("--status", action="store_true", help="Check status of existing jobs")
    parser.add_argument("--download", action="store_true", help="Download completed RTC products")
    parser.add_argument("--watch", action="store_true", help="Watch jobs until completion and download")
    parser.add_argument("--best-only", action="store_true", help="Only submit best temporal match per scene (13 granules)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate job preparation and print details")
    
    args = parser.parse_args()
    
    print("🌍 Authenticating with ASF HyP3...")
    hyp3 = HyP3(username=ASF_USER, password=ASF_PASS)
    print(f"✅ Authenticated as: {ASF_USER}")
    
    if args.submit:
        granules = get_unique_granules(LEDGER_PATH, best_only=args.best_only)
        submit_jobs(hyp3, granules, dry_run=args.dry_run)
    elif args.status:
        monitor_and_download(hyp3, download=False)
    elif args.download:
        monitor_and_download(hyp3, download=True)
    elif args.watch:
        print(f"👀 Watching jobs with tag '{JOB_TAG}'...")
        jobs = hyp3.find_jobs(name=JOB_TAG)
        if jobs:
            hyp3.watch(jobs)
            monitor_and_download(hyp3, download=True)
    else:
        # Default behavior: show status or help
        print("\nNo specific action specified. Checking job status:")
        monitor_and_download(hyp3, download=False)
        print("\nUsage options:")
        print("  python3 scripts/submit_hyp3_rtc.py --submit [--best-only] [--dry-run]")
        print("  python3 scripts/submit_hyp3_rtc.py --status")
        print("  python3 scripts/submit_hyp3_rtc.py --download")
        print("  python3 scripts/submit_hyp3_rtc.py --watch")

if __name__ == "__main__":
    main()
