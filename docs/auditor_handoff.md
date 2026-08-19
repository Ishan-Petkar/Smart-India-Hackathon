# MASTER AUDITOR HANDOFF DOCUMENT
## Topography-Aware Multi-Modal Latent Diffusion (TA-MMLD) Framework for High-Resolution Cloud Removal in Complex Himalayan Terrain

* **Document Version:** 2.0 (Master Comprehensive)
* **Author/System:** Antigravity AI Engineering Team (Google DeepMind)
* **Target Audience:** Master Technical Auditor & Lead Scientific Architect
* **Project Context:** Smart India Horizon / SIH Remote Sensing AI Challenge
* **Workspace Path:** `/Users/ishanpetkar/Smart India Horizon 2026`
* **Primary Storage Array:** `/Volumes/ishan hdd/Dataset`
* **Status Date:** August 19, 2026

---

## Table of Contents
1. [Executive Summary & Core Scientific Mission](#1-executive-summary--core-scientific-mission)
2. [Chronological Engineering History & Deep Audit Discoveries](#2-chronological-engineering-history--deep-audit-discoveries)
3. [Physical Storage & Data Directory Architecture](#3-physical-storage--data-directory-architecture)
4. [The Master Implementation Plan v2 (Phases 1–7)](#4-the-master-implementation-plan-v2-phases-17)
   - [Phase 1: Sentinel-1 ASF HyP3 RTC Product Acquisition](#phase-1-sentinel-1-asf-hyp3-rtc-product-acquisition)
   - [Phase 2: Grid Harmonization & Co-Registration Pipeline](#phase-2-grid-harmonization--co-registration-pipeline)
   - [Phase 3: SCL-Guided Stochastic Hybrid Cloud Augmentation Engine](#phase-3-scl-guided-stochastic-hybrid-cloud-augmentation-engine)
   - [Phase 4: Multi-Modal HDF5 Tensor Staging](#phase-4-multi-modal-hdf5-tensor-staging)
   - [Phase 5: Inference-Time Lightweight Cloud Segmentation Model](#phase-5-inference-time-lightweight-cloud-segmentation-model)
   - [Phase 6: Core TA-MMLD Diffusion Architecture & Training Curriculum](#phase-6-core-ta-mmld-diffusion-architecture--training-curriculum)
   - [Phase 7: Uncertainty Quantification & Operational Product Export](#phase-7-uncertainty-quantification--operational-product-export)
5. [Current Execution State & Verification Proofs](#5-current-execution-state--verification-proofs)
6. [Mathematical & Scientific Formulations](#6-mathematical--scientific-formulations)
7. [Master Auditor Challenge Checklist (Critical Invariants & Gotchas)](#7-master-auditor-challenge-checklist-critical-invariants--gotchas)

---

## 1. Executive Summary & Core Scientific Mission

### 1.1 The Challenge
The Indian Space Research Organisation (ISRO) operates the **Resourcesat-2/2A LISS-IV** sensor, delivering multi-spectral imagery (Green, Red, NIR) at an ultra-fine spatial resolution of **5.0 meters**. In the Himalayan and Northeast Indian corridors, persistent cloud cover, dense monsoon atmospheric fronts, and extreme topography (steep slopes, deep ravines, perennial snow ridges) render large fractions of optical acquisitions unusable.

### 1.2 The Proposed Solution: TA-MMLD
We formulate cloud removal as a **multi-modal conditional image-to-image generative inpainting problem in latent space**:
$$\hat{Y}_{\text{clear}} = \mathcal{G}_{\theta}\left(X_{\text{cloudy}}, \, S1_{\text{SAR}}, \, S2_{\text{opt}}, \, \text{DEM}_{\text{topo}}, \, \text{LSM}_{\text{mask}}, \, M_{\text{cloud}}\right)$$

Where:
- **Optical Reference:** LISS-IV 5.0m VNIR bands (Green: 0.52–0.59 µm, Red: 0.62–0.68 µm, NIR: 0.77–0.86 µm).
- **All-Weather Penetration Guide:** Sentinel-1 C-Band SAR ($5.405\text{ GHz}$) Radiometrically Terrain Corrected (RTC) backscatter ($\gamma^0_{\text{VV}}, \gamma^0_{\text{VH}}$) in dB scale.
- **Spectral Baseline:** Sentinel-2 L2A 10m multispectral imagery (Blue, Green, Red, NIR) with Scene Classification Layer (SCL) quality filtering.
- **Topographic Context:** Copernicus 30m Digital Elevation Model (DEM) and derived gradient slope fields.
- **Topographic Reliability Gate:** Sentinel-1 Layover/Shadow Mask (LSM) that dynamically suppresses SAR attention in radar shadow pockets.

---

## 2. Chronological Engineering History & Deep Audit Discoveries

During initial system setup and data audits, five critical architectural flaws and data anomalies were uncovered and systematically fixed:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                CHRONOLOGY OF AUDIT FIXES                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. PAIRING AUDIT:                                                                      │
│    Discovered all 14 LISS-IV scenes were 100% clear-sky (no cloudy/clear pairs).       │
│    -> Formulated Phase 3: SCL-Guided Stochastic Hybrid Synthetic Cloud Augmentation.   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SAR TERRAIN CORRECTION AUDIT:                                                       │
│    Custom SAR RTC in mountainous terrain is mathematically intractable locally.        │
│    -> Migrated to NASA Alaska Satellite Facility (ASF) HyP3 cloud Gamma0 RTC pipeline. │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. CORRUPT TILE HEALING AUDIT:                                                         │
│    Byte-level testing revealed 4 corrupt DEM tiles and 142 corrupt Sentinel-2 TIFFs.   │
│    -> Repaired 20 via local duplicates; redownloaded 142 via 16-segment aria2c stream. │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. GRID RESOLUTION LOCK:                                                               │
│    Clarified nominal (5.8m) vs delivered (5.0m) pixel geometry.                        │
│    -> Locked master target grid to strictly 5.0m x 5.0m in EPSG:32645.                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. SAR HALLUCINATION SAFEGUARD:                                                        │
│    SAR signals in radar shadow zones are pure thermal noise.                           │
│    -> Engineered Topographic Cross-Attention Gating using the Layover/Shadow Mask.    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Physical Storage & Data Directory Architecture

All data resides on an external storage array mounted at `/Volumes/ishan hdd/Dataset/`:

```
/Volumes/ishan hdd/Dataset/
├── Paired/                                    # 14 LISS-IV Master Scenes (5.0m, EPSG:32645)
│   ├── R2F03JUN2026078473010700051SSANSTUC00GTDB/
│   │   ├── BAND2.tif                          # Green (5.0m, ~18000 x 17000, UInt16)
│   │   ├── BAND3.tif                          # Red (5.0m, ~18000 x 17000, UInt16)
│   │   ├── BAND4.tif                          # NIR (5.0m, ~18000 x 17000, UInt16)
│   │   ├── BAND5.tif                          # SWIR (if present / auxiliary)
│   │   └── metadata.xml                       # Ephemeris, Sun Azimuth/Elevation
│   └── ... (13 other scene directories)
│
├── sentinel1_rtc/                             # 50 ASF HyP3 RTC Processed ZIP Archives (438.44 GB)
│   ├── S1A_IW_20260531T120604_DVP_RTC10_G_gpuned_2504.zip
│   │   └── S1A_IW_.../
│   │       ├── *_VV.tif                       # γ⁰ VV backscatter (10m, power scale)
│   │       ├── *_VH.tif                       # γ⁰ VH backscatter (10m, power scale)
│   │       ├── *_ls_map.tif                   # Layover/Shadow Mask (10m, categorical)
│   │       ├── *_dem.tif                      # Co-registered DEM (10m, float32)
│   │       └── *_inc_map.tif                  # Local Incidence Angle (10m, float32)
│   └── ... (49 other full RTC ZIP archives)
│
├── Auxiliary/                                 # Multi-Modal Auxiliary Imagery per Scene
│   ├── R2F03JUN2026078473010700051SSANSTUC00GTDB/
│   │   ├── DEM/
│   │   │   └── Copernicus_DSM_COG_10_N28_00_E089_00_DEM.tif (30m elevation)
│   │   └── Sentinel2/
│   │       ├── S2A_45RXM_20260609_0_L2A_blue.tif
│   │       ├── S2A_45RXM_20260609_0_L2A_green.tif
│   │       ├── S2A_45RXM_20260609_0_L2A_red.tif
│   │       ├── S2A_45RXM_20260609_0_L2A_nir.tif
│   │       └── S2A_45RXM_20260609_0_L2A_scl.tif
│   └── ... (13 other scene directories)
│
├── dataset_verification_ledger.json           # Master Ledger linking scenes, bounds, and files
├── granule_to_rtc_mapping.json                # Mapping S1 input granules to HyP3 output zips
├── aligned/                                   # [Phase 2 Output] 11-Channel Stacked GeoTIFFs
└── h5_staging/                                # [Phase 4 Output] ML-Ready Train/Val/Test HDF5
```

---

## 4. The Master Implementation Plan v2 (Phases 1–7)

### Phase 1: Sentinel-1 ASF HyP3 RTC Product Acquisition
* **Status:** **100% COMPLETE & VERIFIED**
* **Objective:** Produce Radiometrically Terrain Corrected $\gamma^0$ backscatter and pixel-level Layover/Shadow Masks (LSM) for the complex terrain of the Himalayas.
* **Execution Summary:** 
  - Submitted 50 batch jobs via `hyp3_sdk` querying Sentinel-1 GRD granules.
  - Specified Gamma-0 ($\gamma^0$) radiometric terrain flattening, Copernicus 30m DEM, 10m spatial resolution, power-scale output, and layover/shadow masking (`include_ls_map=True`).
  - Successfully downloaded 50/50 ZIP packages (**438.44 GB total payload**).
  - Resolved exact mapping ledger between input S1 granules and output HyP3 ZIP files saved to `granule_to_rtc_mapping.json`.

---

### Phase 2: Grid Harmonization & Co-Registration Pipeline
* **Status:** **READY FOR EXECUTION**
* **Objective:** Resample and co-register all heterogeneous satellite data onto the LISS-IV 5.0m master grid (`EPSG:32645`).

#### Master Grid Definition
$$\text{Target CRS: } \text{EPSG:32645 (WGS 84 / UTM Zone 45N)}, \quad \Delta x = 5.0\text{m}, \quad \Delta y = 5.0\text{m}$$

#### Resampling & Quality Gating Strategy
| Source Layer | Native Res | Target Res | Resampling Method | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **LISS-IV (B2, B3, B4)** | **5.0m** | 5.0m | *None (Reference)* | Anchor grid for all other modalities |
| **S1 RTC ($\gamma^0_{\text{VV}}, \gamma^0_{\text{VH}}$)** | 10.0m | 5.0m | **Bilinear** | Continuous backscatter fields |
| **S2 L2A (B2, B3, B4, B8)** | 10.0m | 5.0m | **Bicubic** | Optical reflectance spatial gradients |
| **S2 SCL (Scene Mask)** | 20.0m | 5.0m | **Nearest Neighbor** | Categorical labels $\{0, \dots, 11\}$ |
| **Copernicus DEM** | 30.0m | 5.0m | **Bilinear** | Continuous elevation surface |
| **HyP3 LSM** | 10.0m | 5.0m | **Nearest Neighbor** | Binary shadow/layover categorical flags |
| **HyP3 Incidence Angle** | 10.0m | 5.0m | **Bilinear** | Continuous angular geometry |

#### SCL Quality Masking Formulation
Before resampling Sentinel-2 optical bands, cloud and shadow contamination is masked to $\text{NaN}$:
$$\mathcal{M}_{\text{bad}} = \{0\text{ (NoData)}, 1\text{ (Saturated)}, 3\text{ (Cloud Shadow)}, 7\text{ (Unclassified)}, 8\text{ (Cloud Med)}, 9\text{ (Cloud High)}, 10\text{ (Cirrus)}, 11\text{ (Snow)}\}$$
$$S2_{\text{filtered}}(x, y) = \begin{cases} \text{NaN}, & \text{if } SCL(x, y) \in \mathcal{M}_{\text{bad}} \\ S2_{\text{raw}}(x, y), & \text{otherwise} \end{cases}$$

#### Derived Topographic Gradients
Terrain slope $\theta_{\text{slope}}$ is derived from the resampled DEM:
$$\theta_{\text{slope}} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \cdot \frac{180^\circ}{\pi}$$

---

### Phase 3: SCL-Guided Stochastic Hybrid Cloud Augmentation Engine
* **Status:** **PLANNED / READY**
* **Objective:** Generate realistic, continuous cloud opacity and shadow fields to train the diffusion model without overfitting to synthetic procedural artifacts.

#### SCL-Guided Hybrid Synthesis Algorithm
1. **Template Extraction (Offline):** Extract real cloud shapes from Sentinel-2 SCL (classes 8, 9, 10), removing components $< 50\text{ px}$.
2. **Split Isolation (Zero Leakage):** Partition cloud templates strictly by scene ID ($70\%$ train, $15\%$ val, $15\%$ test). Never apply a test-scene cloud shape to a training patch.
3. **Continuous Opacity Modulation:** Modulate template masks using multi-octave Perlin noise:
   $$M_{\text{cloud}}(x, y) = \text{Clamp}\left(M_{\text{SCL}}(x, y) \odot \mathcal{P}_{\text{perlin}}(x, y), \, 0, \, 1\right)$$
4. **Radiometric Regime Compositing:**
   $$X_{\text{cloudy}} = (1 - k \cdot M_{\text{cloud}}) \odot Y_{\text{clear}} + (k \cdot M_{\text{cloud}}) \odot \mathcal{C}_{\text{spectral}} + \epsilon_{\text{texture}}$$
   - **Thin Cirrus ($M \in [0.05, 0.35]$):** Partial optical penetration ($k < 1.0$), surface features preserved.
   - **Moderate Cumulus ($M \in [0.35, 0.65]$):** Non-linear attenuation with spectral jitter.
   - **Thick Dense Cloud ($M \in [0.65, 1.0]$):** Complete optical occlusion, surface signal $\to 0$.

---

### Phase 4: Multi-Modal HDF5 Tensor Staging
* **Status:** **PLANNED**
* **Objective:** Serialize aligned rasters into chunked, high-throughput HDF5 files for distributed GPU training.

#### Tiling Specifications
- **Patch Dimension:** $512 \times 512$ pixels ($2.56\text{ km} \times 2.56\text{ km}$ footprint at 5.0m).
- **Stride:** $384\text{ pixels}$ ($25\%$ overlap).
- **NoData Rejection:** Discard patches with $> 10\%$ invalid/NoData pixels.
- **Estimated Yield:** $\approx 1,500\text{ patches}$ across 14 scenes.

#### Tensor Structure: `X` (Input: $N \times 11 \times 512 \times 512$) vs. `Y` (Target: $N \times 3 \times 512 \times 512$)
```
INPUT TENSOR (X) — Shape: (11, 512, 512) Float32
├── Ch [0]: LISS-IV Green (Clear at rest; corrupted at runtime with M_cloud)
├── Ch [1]: LISS-IV Red   (Clear at rest; corrupted at runtime with M_cloud)
├── Ch [2]: LISS-IV NIR   (Clear at rest; corrupted at runtime with M_cloud)
├── Ch [3]: S1 γ⁰ VV (dB scale: clip [-30, 0] dB → normalized [0, 1])
├── Ch [4]: S1 γ⁰ VH (dB scale: clip [-30, 0] dB → normalized [0, 1])
├── Ch [5]: S2 L2A Blue (Quality-gated, normalized [0, 1])
├── Ch [6]: S2 L2A Green (Quality-gated, normalized [0, 1])
├── Ch [7]: S2 L2A Red (Quality-gated, normalized [0, 1])
├── Ch [8]: S2 L2A NIR (Quality-gated, normalized [0, 1])
├── Ch [9]: Layover/Shadow Mask (LSM: Binary/Categorical {0, 1})
└── Ch [10]: Topographic Surface (Normalized Elevation + Slope)

TARGET TENSOR (Y) — Shape: (3, 512, 512) Float32
├── Ch [0]: LISS-IV Clean Green (Ground Truth)
├── Ch [1]: LISS-IV Clean Red   (Ground Truth)
└── Ch [2]: LISS-IV Clean NIR   (Ground Truth)
```

---

### Phase 5: Inference-Time Lightweight Cloud Segmentation Model
* **Status:** **PLANNED**
* **Objective:** Automatically segment cloud and shadow masks from cloudy LISS-IV VNIR imagery during operational inference.
* **Architecture:** SegFormer-B0 / Swin-UNet-Tiny.
* **Rationale:** LISS-IV lacks a Short-Wave Infrared (SWIR) channel, making simple spectral thresholding (e.g. NDSI) fail on mountain snow. A contextual transformer model distinguishes clouds from snow using spatial structure and edge texture.
* **Loss Function:** $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$. Target IoU $> 0.85$.

---

### Phase 6: Core TA-MMLD Diffusion Architecture & Training Curriculum
* **Status:** **PLANNED**
* **Objective:** Conditional Latent Diffusion Model (LDM) with Topographic Cross-Attention Gating.

#### 1. Latent Space Representation
- Multi-spectral Variational Autoencoder (VAE): Encodes $512 \times 512 \times 3$ LISS-IV into latent space $z \in \mathbb{R}^{4 \times 64 \times 64}$ with compression factor $f=8$.

#### 2. Topographic Attention Gating (SAR Hallucination Prevention)
Standard cross-attention in diffusion models computes:
$$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
In extreme terrain, SAR backscatter inside radar shadows ($LSM = 1$) is purely receiver thermal noise. We inject **Topographic Attention Gating**:
$$\mathbf{A}_{\text{gated}} = \text{Softmax}\left(\frac{Q K_{\text{SAR}}^T}{\sqrt{d_k}}\right) \odot \left(1 - LSM_{\text{spatial}}\right)$$
When $LSM=1$, the SAR attention weight is forced to zero, requiring the UNet to rely on Sentinel-2 optical features and learned generative priors.

#### 3. Multi-Objective Loss Function
$$\mathcal{L}_{\text{total}} = \lambda_{\text{diff}} \mathcal{L}_{\text{denoise}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{SAM}} \mathcal{L}_{\text{SAM}}$$
- **Latent Denoising Loss:** $\mathcal{L}_{\text{denoise}} = \mathbb{E}_{t, z_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$
- **Perceptual Structural Loss:** $\mathcal{L}_{\text{perceptual}} = (1 - \text{SSIM}(\hat{Y}, Y)) + \alpha \mathcal{L}_{\text{VGG}}(\hat{Y}, Y)$
- **Spectral Angle Mapper (SAM):**
  $$\mathcal{L}_{\text{SAM}} = \arccos\left(\frac{\langle\hat{Y}, Y\rangle}{\|\hat{Y}\|_2 \|Y\|_2}\right)$$
  *Crucial for preserving band-to-band radiometric ratios and NDVI integrity.*

#### 4. Training Curriculum
| Stage | Epochs | Synthetic Cloud Range | Target Learning Goal |
| :--- | :---: | :---: | :--- |
| **Warmup** | 1–50 | 10%–30% (Thin Cirrus) | Basic spectral inpainting & VAE alignment |
| **Core** | 51–300 | 20%–60% (Mixed Cumulus) | Multi-modal fusion & SAR structural alignment |
| **Hardening** | 301–500 | 40%–80% (Dense Cloud) | Full reconstruction under complete optical occlusion |

---

### Phase 7: Uncertainty Quantification & Operational Product Export
* **Status:** **PLANNED**
* **Objective:** Produce confidence metrics and Analysis-Ready Data (ARD) GeoTIFFs.

#### Monte Carlo Stochastic Inference
For an input cloudy scene, run $T=8$ stochastic sampling trajectories with distinct diffusion noise seeds:
$$\hat{Y}_{\text{mean}}(x, y) = \frac{1}{T}\sum_{t=1}^T \hat{Y}_t(x, y), \qquad \sigma^2_{\text{uncertainty}}(x, y) = \frac{1}{T}\sum_{t=1}^T \left(\hat{Y}_t(x, y) - \hat{Y}_{\text{mean}}(x, y)\right)^2$$

#### Final 5-Band GeoTIFF Output Product
- **Band 1:** Reconstructed Green ($5.0\text{m}$)
- **Band 2:** Reconstructed Red ($5.0\text{m}$)
- **Band 3:** Reconstructed NIR ($5.0\text{m}$)
- **Band 4:** Detected Cloud/Shadow Mask ($5.0\text{m}$)
- **Band 5:** Per-Pixel Predictive Uncertainty $\sigma^2$ ($5.0\text{m}$)

---

## 5. Current Execution State & Verification Proofs

A 100% comprehensive audit was executed across all 14 scene directories on disk:

```text
================================================================================
COMPREHENSIVE MULTI-MODAL DATASET AUDIT & INTEGRITY VERIFICATION (PASSED: 14/14)
================================================================================
[01/14] R2F03JUN2026078473010700051SSANSTUC00GTDB: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 8 matches] | S2 [OK, 5 granules, dt=4d]
[02/14] R2F03JUN2026078473010700051SSANSTUC00GTDD: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 9 matches] | S2 [OK, 5 granules, dt=1d]
[03/14] R2F03JUN2026078473010700052SSANSTUC00GTDB: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 6 matches] | S2 [OK, 5 granules, dt=1d]
[04/14] R2F05MAY2026078061010600051SSANSTUC00GTDC: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 7 matches] | S2 [OK, 5 granules, dt=3d]
[05/14] R2F09AUG2026079425010600051SSANSTUC00GTDA: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 8 matches] | S2 [OK, 3 granules, dt=2d]
[06/14] R2F09AUG2026079425010600051SSANSTUC00GTDC: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 6 matches] | S2 [OK, 2 granules, dt=2d]
[07/14] R2F14AUG2026079503010700052SSANSTUC00GTDA: LISS-IV [OK] | DEM [OK, 2 tiles] | S1 RTC [OK, 3 matches] | S2 [OK, 2 granules, dt=0d]
[08/14] R2F10MAY2026078132010700052SSANSTUC00GTDA: LISS-IV [OK] | DEM [OK, 2 tiles] | S1 RTC [OK, 4 matches] | S2 [OK, 4 granules, dt=3d]
[09/14] R2F16JUL2026079084010600050SSANSTUC00GTDD: LISS-IV [OK] | DEM [OK, 2 tiles] | S1 RTC [OK, 3 matches] | S2 [OK, 1 granule,  dt=4d]
[10/14] R2F22JUN2026078743010600050SSANSTUC00GTDC: LISS-IV [OK] | DEM [OK, 2 tiles] | S1 RTC [OK, 7 matches] | S2 [OK, 1 granule,  dt=5d]
[11/14] R2F22JUN2026078743010600051SSANSTUC00GTDC: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 8 matches] | S2 [OK, 2 granules, dt=0d]
[12/14] R2F29MAY2026078402010600051SSANSTUC00GTDB: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 9 matches] | S2 [OK, 1 granule,  dt=4d]
[13/14] R2F29MAY2026078402010600051SSANSTUC00GTDD: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 9 matches] | S2 [OK, 2 granules, dt=4d]
[14/14] R2F29MAY2026078402010600052SSANSTUC00GTDB: LISS-IV [OK] | DEM [OK, 4 tiles] | S1 RTC [OK, 6 matches] | S2 [OK, 3 granules, dt=4d]
================================================================================
VERIFICATION RESULT: 14/14 SCENES 100% AUDITED, COMPLETE, AND CO-REGISTERABLE.
```

---

## 6. Mathematical & Scientific Formulations

### 6.1 SAR Radiometric Power-to-dB Conversion
HyP3 RTC provides linear power-scale radar cross-section $\gamma^0_{\text{power}}$. Terrestrial backscatter spans several orders of magnitude. We compress the dynamic range to decibels and normalize to $[0, 1]$:
$$\gamma^0_{\text{dB}} = 10 \cdot \log_{10}\left(\max\left(\gamma^0_{\text{power}}, \, 10^{-3}\right)\right)$$
$$\gamma^0_{\text{norm}} = \text{Clip}\left(\frac{\gamma^0_{\text{dB}} - (-30.0)}{0.0 - (-30.0)}, \, 0.0, \, 1.0\right) = \text{Clip}\left(\frac{\gamma^0_{\text{dB}} + 30.0}{30.0}, \, 0.0, \, 1.0\right)$$

### 6.2 Radiative Transfer Cloud Mixing
The composite at pixel $(x, y)$ in channel $c \in \{\text{Green, Red, NIR}\}$:
$$X_c(x, y) = Y_c(x, y) \cdot T_c(x, y) + L_{\text{path}, c}(x, y) \cdot \left(1 - T_c(x, y)\right)$$
Where transmittance $T_c(x, y) = \exp(-\tau(x, y) / \cos\theta_v)$ is parameterized by optical thickness $\tau(x, y)$ derived from $M_{\text{cloud}}(x, y)$.

---

## 7. Master Auditor Challenge Checklist (Critical Invariants & Gotchas)

When reviewing this pipeline for fundamental failure modes, verify these non-obvious engineering invariants:

1. **RAM Overflow in Windowed Warping:**
   - *Risk:* LISS-IV full rasters are $\approx 18,000 \times 17,000$ pixels ($\approx 1.2\text{ GB}$ per band uncompressed, float32 stack $\approx 14\text{ GB}$).
   - *Requirement:* `scripts/grid_harmonize.py` must use windowed/block-based processing (`rasterio.windows.Window`) or GDAL Virtual Format (`VRT`) rather than reading full-scene arrays into RAM simultaneously.
2. **Temporal SAR Look Angle Disparities:**
   - *Risk:* S1 images taken from Ascending vs. Descending orbits have opposite layover/shadow geometry.
   - *Requirement:* Match each scene's candidate S1 RTC by orbit direction and incidence angle proximity, or average dual-orbit RTC passes.
3. **Snow vs. Cloud False-Positive Rate:**
   - *Risk:* In the Himalayas, perennial glaciers share high VNIR reflectance with clouds.
   - *Requirement:* SCL Class 11 (Snow) must never be extracted as a cloud template, and the SegFormer cloud segmenter must be regularized against false positives over high-altitude glaciated terrain.
4. **Exact Deterministic Reproducibility:**
   - *Requirement:* Every synthetic augmentation call in `scripts/synthetic_clouds.py` must accept a seed generated from `(patch_index, epoch)` to guarantee exact repeatability during ablation studies.

---

## 8. Immediate Next Step
The system is parked at the entry of **Phase 2 (Grid Harmonization & Co-Registration)** awaiting the auditor's review and approval. Once approved, execute `scripts/grid_harmonize.py` to create the 11-channel co-registered GeoTIFF cubes.
