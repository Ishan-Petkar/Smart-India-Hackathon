import os
import getpass
import asf_search as asf

TARGET_DIR = "/Volumes/ishan hdd/Dataset"
os.makedirs(TARGET_DIR, exist_ok=True)

URLS = [
    "https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20250811T115706_20250811T115731_060488_078554_2FD9.zip",
    "https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20250808T233852_20250808T233917_060451_0783E8_2A56.zip",
    "https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20250806T235522_20250806T235547_060422_0782BF_835C.zip",
    "https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20250808T233917_20250808T233942_060451_0783E8_8685.zip",
    "https://datapool.asf.alaska.edu/GRD_HD/SA/S1A_IW_GRDH_1SDV_20250811T115731_20250811T115756_060488_078554_D79B.zip"
]

def main():
    print("=" * 60)
    print("  NASA ASF Sentinel-1 Downloader (Modern SDK)")
    print(f"  Target Folder: {TARGET_DIR}")
    print(f"  Total Files to Download: {len(URLS)} (~4.5 GB)")
    print("=" * 60)
    
    username = input("Enter NASA Earthdata Username: ").strip()
    password = getpass.getpass("Enter NASA Earthdata Password: ")
    
    try:
        print("\nAuthenticating with NASA Earthdata...")
        session = asf.ASFSession().auth_with_creds(username, password)
        print("Authentication successful!")
        
        print("\nStarting downloads directly to your 2TB HDD...")
        asf.download_urls(urls=URLS, path=TARGET_DIR, session=session)
        print("\n All Sentinel-1 files downloaded successfully into /Volumes/ishan hdd/Dataset!")
    except Exception as e:
        print(f"\n❌ Error during download: {e}")

if __name__ == "__main__":
    main()
