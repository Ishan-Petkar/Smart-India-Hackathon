# Complete Dataset Architecture, Storage Hierarchy & Processing Manifest

> **Project:** Topography-Aware Multi-Modal Latent Diffusion (TA-MMLD) Framework for High-Resolution Cloud Removal in Complex Himalayan Terrain  
> **Workspace Path:** `/Users/ishanpetkar/Smart India Horizon 2026`  
> **External Primary Storage Array:** `/Volumes/ishan hdd/Dataset/` (~450+ GB)  
> **Last Updated:** August 2026

---

## 1. Executive Context & Storage Philosophy

Because raw high-resolution satellite imagery (ISRO LISS-IV, Sentinel-1 SAR RTC, Sentinel-2 Optical, and Copernicus DEM) exceeds standard Git storage limits (and GitHub's 100 MB per-file hard cap), all heavy raster datasets, HDF5 staging caches, and raw archive ZIPs are systematically managed outside the Git index via `.gitignore`.

This document serves as the **Single Source of Truth** for:
1. The **exact directory layout** and catalog of datasets on the primary storage array (`/Volumes/ishan hdd/Dataset`).
2. The **local cached samples** in `data/raw/`.
3. The **processed & aligned data pipelines** (Phase 2 Harmonization, Phase 3 Cloud Augmentation, Phase 4 HDF5 Tensor Staging).
4. The **11-channel multi-modal tensor specifications** for the diffusion model.
5. The complete **14-scene master catalog & audit ledger**.

---

## 2. Physical Storage & Directory Architecture

```
/Volumes/ishan hdd/Dataset/
├── Paired/                                    # 14 LISS-IV Master Scenes (5.0m, EPSG:32645)
│   ├── R2F03JUN2026078473010700051SSANSTUC00GTDB/
│   │   ├── BAND2.tif                          # Green (5.0m, ~18000 x 17000, UInt16, ~1.2 GB)
│   │   ├── BAND3.tif                          # Red (5.0m, ~18000 x 17000, UInt16, ~1.2 GB)
│   │   ├── BAND4.tif                          # NIR (5.0m, ~18000 x 17000, UInt16, ~1.2 GB)
│   │   ├── BAND5.tif                          # SWIR (if present / auxiliary)
│   │   └── metadata.xml                       # Scene Ephemeris, Sun Azimuth/Elevation
│   └── ... (13 other scene directories)
│
├── sentinel1_rtc/                             # 50 ASF HyP3 RTC Processed ZIP Archives (438.44 GB)
│   ├── S1A_IW_20260531T120604_DVP_RTC10_G_gpuned_2504.zip
│   │   └── S1A_IW_.../
│   │       ├── *_VV.tif                       # γ⁰ VV backscatter (10m, power scale)
│   │       ├── *_VH.tif                       # γ⁰ VH backscatter (10m, power scale)
│   │       ├── *_ls_map.tif                   # Layover/Shadow Mask (10m, categorical {0, 1})
│   │       ├── *_dem.tif                      # Co-registered DEM (10m, float32)
│   │       └── *_inc_map.tif                  # Local Incidence Angle (10m, float32)
│   └── ... (49 other full RTC ZIP archives)
│
├── Auxiliary/                                 # Multi-Modal Auxiliary Imagery per Scene
│   ├── R2F03JUN2026078473010700051SSANSTUC00GTDB/
│   │   ├── DEM/
│   │   │   └── Copernicus_DSM_COG_10_N28_00_E089_00_DEM.tif (30m elevation COG)
│   │   └── Sentinel2/
│   │       ├── S2A_45RXM_20260609_0_L2A_blue.tif   (10m Blue, Band 2)
│   │       ├── S2A_45RXM_20260609_0_L2A_green.tif  (10m Green, Band 3)
│   │       ├── S2A_45RXM_20260609_0_L2A_red.tif    (10m Red, Band 4)
│   │       ├── S2A_45RXM_20260609_0_L2A_nir.tif    (10m NIR, Band 8)
│   │       └── S2A_45RXM_20260609_0_L2A_scl.tif    (20m Scene Classification Layer)
│   └── ... (13 other scene directories)
│
├── dataset_verification_ledger.json           # Master Ledger linking scenes, bounds, and files
├── granule_to_rtc_mapping.json                # Mapping S1 input granules to HyP3 output zips
├── aligned/                                   # [Phase 2 Target] 11-Channel Stacked GeoTIFFs (5.0m, EPSG:32645)
└── h5_staging/                                # [Phase 4 Target] ML-Ready Train/Val/Test HDF5 Chunks
```

---

## 3. Local Workspace Cache (`data/raw/`)

For rapid local testing and verification without mounting the external HDD, a curated subset of high-resolution rasters is preserved locally:

| File Path | Modality & Description | Spatial Resolution | File Size |
| :--- | :--- | :---: | :---: |
| `data/raw/sentinel2/clear_visual.tif` | Sentinel-2 L2A RGB Composite (Clear Sky) | 10.0m | 305 MB |
| `data/raw/sentinel2/clear_nir.tif` | Sentinel-2 L2A Near-Infrared Band 8 (Clear Sky) | 10.0m | 234 MB |
| `data/raw/sentinel2/clear_red.tif` | Sentinel-2 L2A Red Band 4 (Clear Sky) | 10.0m | 217 MB |
| `data/raw/sentinel2/clear_green.tif` | Sentinel-2 L2A Green Band 3 (Clear Sky) | 10.0m | 212 MB |
| `data/raw/sentinel2/cloudy_visual.tif` | Sentinel-2 L2A RGB Composite (Cloud Contaminated) | 10.0m | 204 MB |
| `data/raw/sentinel2/cloudy_nir.tif` | Sentinel-2 L2A Near-Infrared Band 8 (Cloud Contaminated) | 10.0m | 223 MB |
| `data/raw/sentinel2/cloudy_red.tif` | Sentinel-2 L2A Red Band 4 (Cloud Contaminated) | 10.0m | 215 MB |
| `data/raw/sentinel2/cloudy_green.tif` | Sentinel-2 L2A Green Band 3 (Cloud Contaminated) | 10.0m | 216 MB |
| `data/raw/dem/copernicus_dem_30m_N25_E091.tif` | Copernicus 30m Global Elevation Model (NER Region) | 30.0m | 44.4 MB |

---

## 4. Multi-Modal Modality Specifications

### 4.1 ISRO Resourcesat-2/2A LISS-IV (Anchor Modality)
- **Spatial Resolution:** **5.0m** (delivered ground sampling distance).
- **Spectral Bands:**
  - **Band 2 (Green):** $0.52 - 0.59\ \mu\text{m}$ (Vegetation vigor & water bodies)
  - **Band 3 (Red):** $0.62 - 0.68\ \mu\text{m}$ (Chlorophyll absorption & soil)
  - **Band 4 (NIR):** $0.77 - 0.86\ \mu\text{m}$ (Biomass reflectance & structural boundaries)
- **Dimensions per Scene:** $\approx 18,000 \times 17,000\text{ pixels}$ ($\approx 90\text{ km} \times 85\text{ km}$ ground footprint).
- **Coordinate Reference System (CRS):** `EPSG:32645` (WGS 84 / UTM Zone 45N).

### 4.2 Sentinel-1 C-Band SAR Radiometric Terrain Corrected (ASF HyP3 RTC)
- **Sensor:** Sentinel-1 Synthetic Aperture Radar (5.405 GHz C-Band).
- **Correction:** Radiometric Terrain Flattening ($\gamma^0$) using Copernicus 30m DEM in NASA ASF HyP3.
- **Polarizations:** Dual-pol $\text{VV}$ and $\text{VH}$ in linear power scale.
- **dB Compression & Normalization:**
  $$\gamma^0_{\text{dB}} = 10 \cdot \log_{10}\left(\max\left(\gamma^0_{\text{power}}, \, 10^{-3}\right)\right)$$
  $$\gamma^0_{\text{norm}} = \text{Clip}\left(\frac{\gamma^0_{\text{dB}} + 30.0}{30.0}, \, 0.0, \, 1.0\right)$$
- **Auxiliary Masks Provided:**
  - **Layover/Shadow Mask (LSM):** Categorical flag marking true radar shadow pockets where backscatter is thermal noise.
  - **Incidence Angle Map:** Pixel-level local radar beam geometry.

### 4.3 Sentinel-2 L2A Multispectral Optical & Quality Gating
- **Bands Used:** B2 (Blue, 10m), B3 (Green, 10m), B4 (Red, 10m), B8 (NIR, 10m).
- **Scene Classification Layer (SCL, 20m):** Used to quality-gate optical baseline data and prevent cloud leakage:
  $$\mathcal{M}_{\text{bad}} = \{0\text{ (NoData)}, 1\text{ (Saturated)}, 3\text{ (Shadow)}, 7\text{ (Unclassified)}, 8\text{ (Cloud Med)}, 9\text{ (Cloud High)}, 10\text{ (Cirrus)}, 11\text{ (Snow)}\}$$
  Pixels in $\mathcal{M}_{\text{bad}}$ are masked to $\text{NaN}$ prior to bicubic interpolation.

### 4.4 Copernicus 30m Global DEM & Topographic Derivatives
- **Elevation ($z$):** 30m GLO-30 Copernicus Digital Surface Model.
- **Topographic Slope ($\theta_{\text{slope}}$):** Derived via Sobel/finite-difference gradient operators:
  $$\theta_{\text{slope}} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \cdot \frac{180^\circ}{\pi}$$

---

## 5. Master 14-Scene Verification Ledger

All 14 LISS-IV scenes stored on `/Volumes/ishan hdd/Dataset/Paired/` have been audited for coordinate boundaries, DEM tile coverage, Sentinel-1 RTC temporal proximity, and Sentinel-2 optical cloud-free baselines:

| # | Scene Directory Identifier | Date | DEM Tiles | S1 Matches | S2 Granules ($\Delta t$) |
| :-: | :--- | :---: | :---: | :---: | :---: |
| **01** | `R2F03JUN2026078473010700051SSANSTUC00GTDB` | 2026-06-03 | 4 tiles | 8 RTC zips | 5 granules ($\Delta t=4\text{d}$) |
| **02** | `R2F03JUN2026078473010700051SSANSTUC00GTDD` | 2026-06-03 | 4 tiles | 9 RTC zips | 5 granules ($\Delta t=1\text{d}$) |
| **03** | `R2F03JUN2026078473010700052SSANSTUC00GTDB` | 2026-06-03 | 4 tiles | 6 RTC zips | 5 granules ($\Delta t=1\text{d}$) |
| **04** | `R2F05MAY2026078061010600051SSANSTUC00GTDC` | 2026-05-05 | 4 tiles | 7 RTC zips | 5 granules ($\Delta t=3\text{d}$) |
| **05** | `R2F09AUG2026079425010600051SSANSTUC00GTDA` | 2026-08-09 | 4 tiles | 8 RTC zips | 3 granules ($\Delta t=2\text{d}$) |
| **06** | `R2F09AUG2026079425010600051SSANSTUC00GTDC` | 2026-08-09 | 4 tiles | 6 RTC zips | 2 granules ($\Delta t=2\text{d}$) |
| **07** | `R2F14AUG2026079503010700052SSANSTUC00GTDA` | 2026-08-14 | 2 tiles | 3 RTC zips | 2 granules ($\Delta t=0\text{d}$) |
| **08** | `R2F10MAY2026078132010700052SSANSTUC00GTDA` | 2026-05-10 | 2 tiles | 4 RTC zips | 4 granules ($\Delta t=3\text{d}$) |
| **09** | `R2F16JUL2026079084010600050SSANSTUC00GTDD` | 2026-07-16 | 2 tiles | 3 RTC zips | 1 granule ($\Delta t=4\text{d}$) |
| **10** | `R2F22JUN2026078743010600050SSANSTUC00GTDC` | 2026-06-22 | 2 tiles | 7 RTC zips | 1 granule ($\Delta t=5\text{d}$) |
| **11** | `R2F22JUN2026078743010600051SSANSTUC00GTDC` | 2026-06-22 | 4 tiles | 8 RTC zips | 2 granules ($\Delta t=0\text{d}$) |
| **12** | `R2F29MAY2026078402010600051SSANSTUC00GTDB` | 2026-05-29 | 4 tiles | 9 RTC zips | 1 granule ($\Delta t=4\text{d}$) |
| **13** | `R2F29MAY2026078402010600051SSANSTUC00GTDD` | 2026-05-29 | 4 tiles | 9 RTC zips | 2 granules ($\Delta t=4\text{d}$) |
| **14** | `R2F29MAY2026078402010600052SSANSTUC00GTDB` | 2026-05-29 | 4 tiles | 6 RTC zips | 3 granules ($\Delta t=4\text{d}$) |

---

## 6. Processed Data Pipeline & ML Tensor Layout

### 6.1 Phase 2 Harmonization Target (`/Volumes/ishan hdd/Dataset/aligned/`)
All modalities are resampled into 11-channel stacked GeoTIFF cubes locked to the 5.0m $\times$ 5.0m LISS-IV grid in `EPSG:32645`:

```
INPUT TENSOR (X) — Shape: (11, 512, 512) Float32
├── Ch [0]: LISS-IV Green (Clear baseline; corrupted dynamically with M_cloud at runtime)
├── Ch [1]: LISS-IV Red   (Clear baseline; corrupted dynamically with M_cloud at runtime)
├── Ch [2]: LISS-IV NIR   (Clear baseline; corrupted dynamically with M_cloud at runtime)
├── Ch [3]: S1 γ⁰ VV (Normalized [0, 1] from dB scale)
├── Ch [4]: S1 γ⁰ VH (Normalized [0, 1] from dB scale)
├── Ch [5]: S2 L2A Blue (SCL quality-gated, normalized [0, 1])
├── Ch [6]: S2 L2A Green (SCL quality-gated, normalized [0, 1])
├── Ch [7]: S2 L2A Red (SCL quality-gated, normalized [0, 1])
├── Ch [8]: S2 L2A NIR (SCL quality-gated, normalized [0, 1])
├── Ch [9]: Layover/Shadow Mask (LSM: 0 = Valid SAR, 1 = Mountain Shadow/Layover)
└── Ch [10]: Topographic Surface (Normalized Elevation + Gradient Slope θ_slope)

TARGET TENSOR (Y) — Shape: (3, 512, 512) Float32
├── Ch [0]: Ground Truth LISS-IV Green (Clean 5.0m)
├── Ch [1]: Ground Truth LISS-IV Red   (Clean 5.0m)
└── Ch [2]: Ground Truth LISS-IV NIR   (Clean 5.0m)
```

### 6.2 Phase 4 HDF5 Staging (`/Volumes/ishan hdd/Dataset/h5_staging/`)
- **Tiling Footprint:** $512 \times 512$ pixels ($2.56\text{ km} \times 2.56\text{ km}$ at 5.0m ground resolution).
- **Stride:** 384 pixels ($25\%$ overlap).
- **Chunking:** Chunk-aligned compression (`gzip-4` / `lzf`) optimized for high-throughput GPU training on NVMe arrays.
- **Partitioning:** Strict scene-level train (70%, 10 scenes), validation (15%, 2 scenes), and test (15%, 2 scenes) isolation to prevent any spatial or cloud-morphology leakage.

---

## 7. How to Mount and Run Pipelines

1. **Attach External HDD:**
   Ensure the external drive named `ishan hdd` is connected. It will automatically mount at:
   `/Volumes/ishan hdd/Dataset`

2. **Verify Dataset Integrity:**
   ```bash
   python3 scripts/verify_dataset_alignment.py
   ```

3. **Run Phase 2 Grid Harmonization:**
   ```bash
   python3 scripts/grid_harmonize.py --data-dir "/Volumes/ishan hdd/Dataset" --out-dir "/Volumes/ishan hdd/Dataset/aligned"
   ```
