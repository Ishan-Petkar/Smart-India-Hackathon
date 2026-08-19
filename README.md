# Master Implementation Plan v2: TA-MMLD Cloud Removal Pipeline
## Audit-Corrected, End-to-End — From Raw Data to Operational Product

*Supersedes: `FINAL_PLAN.md`, `dataset_implementation_plan.md`*
*Incorporates all fixes from: [dataset_audit_report.md](file:///Users/ishanpetkar/.gemini/antigravity-ide/brain/44421873-f09d-452b-9366-415c3c1be165/dataset_audit_report.md)*

---

## Executive Summary

This plan defines the complete engineering roadmap for the **Topography-Aware Multi-Modal Latent Diffusion (TA-MMLD)** framework — from raw satellite data to a deployed, uncertainty-quantified cloud removal product for LISS-IV imagery over the Himalayas.

The plan is organized into **two parts**:
- **Part A (Phases 1–4):** Dataset Construction — turning raw downloads into training-ready HDF5 tensors.
- **Part B (Phases 5–7):** Model Development & Deployment — training, evaluation, and operational inference.

### Key Corrections from Deep Audit

| Issue | Impact | Resolution |
|:------|:-------|:-----------|
| No cloudy/clear LISS-IV pairs | ❌ Model cannot learn cloud removal | Synthetic cloud augmentation engine |
| Custom RTC is prohibitively complex | ❌ Weeks of signal processing R&D | Use ASF HyP3 free cloud RTC service |
| S2 SCL band unused | ⚠️ Missing quality gate + cloud templates | SCL-based S2 masking + cloud shape extraction |
| 5.0m vs 5.8m pixel size confusion | ⚠️ Inconsistent grid definition | Lock to **5.0m** (actual delivered resolution) |

---

# Part A: Dataset Construction Pipeline

## Phase 1: ASF HyP3 RTC Product Acquisition ✦ NEW
**Goal:** Obtain production-quality, terrain-corrected Sentinel-1 SAR backscatter (γ⁰_RTC) and Layover/Shadow Masks for all 14 LISS-IV footprints.

### 1.1 Prerequisites
- NASA Earthdata Login account (already configured via `scripts/auth_nasa.py`)
- `pip install hyp3_sdk asf_search`

### 1.2 Workflow
1. **Extract unique S1 granule IDs** from the 186 downloaded GRD ZIP filenames.
2. **Deduplicate to ~93 unique granules** (each has VV + VH inside).
3. **Submit RTC batch jobs** via `hyp3_sdk`:
   ```python
   hyp3.submit_rtc_job(
       granule=granule_name,
       dem_name='copernicus',       # Uses Copernicus 30m DEM (same as ours)
       radiometry='gamma0',         # γ⁰ (terrain-flattened)
       resolution=10,               # 10m output resolution
       scale='power',               # Linear power scale (we convert to dB ourselves)
       include_dem=True,            # Co-registered DEM raster
       include_inc_map=True,        # Local incidence angle map
       include_ls_map=True,         # Layover/Shadow mask (LSM) — CRITICAL
       include_scattering_area=True # Area normalization factor
   )
   ```
4. **Poll and download** completed products (~30–60 min processing time per batch).

### 1.3 Output Products (Per Granule)
| File | Description | Use |
|:-----|:------------|:----|
| `*_VV.tif` | γ⁰ VV backscatter (10m, power scale) | Input Ch [3] |
| `*_VH.tif` | γ⁰ VH backscatter (10m, power scale) | Input Ch [4] |
| `*_ls_map.tif` | Layover/Shadow binary mask | Input Ch [9] |
| `*_dem.tif` | Co-registered DEM (10m) | Input Ch [10] |
| `*_inc_map.tif` | Local incidence angle | QA / advanced normalization |

### 1.4 Credit Budget
- ~50 credits per RTC job × 93 granules = **~4,650 credits**
- Free tier: **8,000 credits/month** → single batch fits comfortably

### 1.5 Script
- **`scripts/submit_hyp3_rtc.py`** [NEW]
  - Reads S1 granule names from `/Volumes/ishan hdd/Dataset/sentinel1/`
  - Authenticates with Earthdata via `hyp3_sdk`
  - Submits all RTC jobs in a single batch
  - Polls for completion, downloads results to `/Volumes/ishan hdd/Dataset/sentinel1_rtc/`
  - Logs each job status and download path

### 1.6 Verification
- Confirm VV/VH GeoTIFFs open cleanly with `rasterio`
- Validate CRS is `EPSG:32645` or transformable to it
- Spot-check LSM map against DEM for physical plausibility (shadows on north-facing slopes)

---

## Phase 2: Grid Harmonization & Co-Registration
**Goal:** Align every data layer — S1 RTC, S2, DEM, LSM — pixel-for-pixel onto the LISS-IV 5.0m master grid.

### 2.1 Master Grid Specification
| Parameter | Value |
|:----------|:------|
| **CRS** | `EPSG:32645` (WGS 84 / UTM Zone 45N) |
| **Pixel Size** | **5.0m × 5.0m** (locked to LISS-IV product resolution) |
| **Bounds** | Per-scene, derived from LISS-IV GeoTIFF extent |
| **NoData** | `0` (LISS-IV), `NaN` (all floating-point layers) |

### 2.2 Resampling Strategy

| Source Layer | Native Resolution | Target | Interpolation | Rationale |
|:-------------|:-------------------|:-------|:--------------|:----------|
| LISS-IV (B2, B3, B4) | 5.0m | Master (no change) | — | Reference anchor |
| S1 RTC γ⁰ (VV, VH) | 10m | 5.0m (2× upsample) | **Bilinear** | SAR backscatter is smooth/continuous |
| S2 L2A (B2, B3, B4, B8) | 10m | 5.0m (2× upsample) | **Bicubic** | Optical reflectance benefits from sharper interpolation |
| S2 SCL (classification) | 20m | 5.0m (4× upsample) | **Nearest Neighbor** | Categorical mask — must not interpolate class values |
| Copernicus DEM | 30m | 5.0m (6× upsample) | **Bilinear** | Elevation is smooth/continuous |
| HyP3 LSM | 10m | 5.0m (2× upsample) | **Nearest Neighbor** | Binary mask — must preserve sharp boundaries |
| HyP3 Incidence Angle | 10m | 5.0m (2× upsample) | **Bilinear** | Angle field is smooth/continuous |

### 2.3 S2 Quality Gating (SCL Filtering)
Before resampling Sentinel-2 optical bands, apply SCL-based quality masking:
```python
# Mask out unreliable S2 pixels
bad_classes = {0, 1, 3, 7, 8, 9, 10, 11}  # nodata, saturated, cloud_shadow,
                                             # unclassified, cloud_med, cloud_hi,
                                             # thin_cirrus, snow
quality_mask = ~np.isin(scl_band, list(bad_classes))
# Set bad pixels to NaN before resampling
for band in [B2, B3, B4, B8]:
    band[~quality_mask] = np.nan
```

### 2.4 Derived Terrain Channels
From the co-registered DEM, compute:
1. **Terrain Slope** (degrees): `np.degrees(np.arctan(np.sqrt(dzdx² + dzdy²)))`
2. **Combined Elevation+Slope** channel: Normalize elevation to [0, 1] globally and slope to [0, 1] locally, then stack or average into a single channel (Ch [10]).

### 2.5 Script
- **`scripts/grid_harmonize.py`** [NEW]
  - Iterates over all 14 LISS-IV scenes from the verification ledger
  - For each scene: reads LISS-IV bounds → warps S1 RTC, S2, DEM, LSM to the same grid
  - Applies SCL quality gating on S2
  - Computes slope from DEM
  - Outputs per-scene aligned multi-band VRT or GeoTIFF cubes

### 2.6 Verification
- Open LISS-IV + aligned S1/S2/DEM in QGIS and confirm spatial overlay
- Check that pixel counts match exactly across all layers
- Validate no NaN leakage into LISS-IV ground truth

---

## Phase 3: Synthetic Cloud Augmentation Engine ✦ REWORKED
**Goal:** Implement an "SCL-Guided Stochastic Hybrid" engine that generates continuous cloud opacity masks and applies regime-based radiometric rendering to create highly realistic cloudy/clear training pairs.

### 3.1 Why Synthetic Clouds & The "Hybrid" Approach
Because the LISS-IV dataset lacks natural paired cloudy/clear images, we synthesize clouds. To avoid the model learning unrealistic procedural noise patterns (e.g., pure Perlin noise shortcuts) or memorizing static templates, we use a hybrid data-driven approach:
- **Geometry:** Base morphology comes from real Sentinel-2 SCL cloud masks.
- **Perturbation & Opacity:** Perlin noise modulates the edges and interior to create a continuous opacity field (optical thickness), rather than hard binary edges.
- **Strict Isolation:** To prevent data leakage, SCL templates extracted from test-split scenes are strictly forbidden from appearing during training.

### 3.2 Offline Template Library Creation
Before training begins, extract SCL cloud templates:
1. Extract classes 8 (medium prob) and 9 (high prob) into a "cloud" pool, and 10 (thin cirrus) into a "cirrus" pool.
2. Filter out small connected components (<50px) to remove salt-and-pepper noise. Apply snow-safeguards to avoid extracting snow ridges as cloud templates.
3. **CRITICAL:** Partition the source scenes into train/val/test splits *before* extraction. Build independent template libraries. A validation scene's cloud shape must never be pasted onto a training patch.

### 3.3 Runtime Generation Pipeline
For each training step, given a clear LISS-IV patch and its split assignment:
1. **Sample Context:** Phase 6 training curriculum dictates the target coverage. Sample the regime (thin, moderate, thick, dense).
2. **Template Selection & Transform:** Draw candidate templates *only* from the matching split's library. Apply stochastic geometric transforms (rotate, scale, warp, dilate) to break spatial memory.
3. **Opacity Mask ($M_{cloud}$):** Composite templates and modulate with Perlin noise to create a continuous opacity mask $M_{cloud} \in [0, 1]$. (Do not use binary masks).
4. **Shadow Mask:** Compute displacement *direction* from real sun azimuth (from metadata). Sample magnitude stochastically. Output a soft-edged continuous shadow mask. *Note: this is an explicit stochastic geometric approximation, not physically exact 3D ray-tracing.*

### 3.4 Radiometric Compositing (Regime-Based)
Do not use simple linear blending with a flat white color. Use opacity $a = M_{cloud}$:
- **Thin (Cirrus, $a \sim 0.05 - 0.35$):** $X = Y_{clear} \cdot (1 - k \cdot a) + C \cdot (k \cdot a)$. The surface signal partially survives (attenuation, not replacement).
- **Moderate ($a \sim 0.35 - 0.65$):** Standard linear mix, $C$ gets small spectral jitter.
- **Thick ($a \sim 0.65 - 1.0$):** $X = C$. As $a \to 1$, residual surface signal $\to 0$. Add slight spatial micro-texture so the cloud interior isn't a trivially flat color for the segmentation model to exploit.

### 3.5 Scripts
- **`scripts/synthetic_clouds.py`** [UPDATED]
  - `CloudTemplateLibrary`: Builds and partitions offline SCL templates.
  - `CloudGenerator`: Configurable runtime engine.
  - `generate(clear_patch, coverage, regime, sun_angles, seed)` $\to$ Returns `cloudy_patch`, `cloud_mask` ($M_{cloud}$), `shadow_mask`, and `metadata`.
  - All calls accept a seed to guarantee exact reproducibility of augmentations.

### 3.6 Verification & Testing
- **Split-Isolation Test:** CI-style assertion that no train/val/test library shares source templates.
- **Anti-Shortcut Test:** Ensure boundary gradients and interior textures aren't trivially separable from natural imagery, protecting Phase 5 from learning simple artifacts.
- **Regime Consistency:** Confirm sampled opacities match intended regimes, and thin clouds genuinely preserve underlying SSIM.

---

## Phase 4: HDF5 Tensor Staging
**Goal:** Tile all harmonized data into 512×512 patches and serialize into high-throughput HDF5 archives.

### 4.1 Tiling Protocol
| Parameter | Value |
|:----------|:------|
| Patch size | 512 × 512 pixels |
| Stride | 384 pixels (25% overlap) |
| NoData rejection | Discard patches with >10% NoData pixels |
| Estimated patches | ~1,200–1,800 total (14 scenes × ~90–130 patches/scene) |

### 4.2 Tensor Layout

#### Input Tensor: `X` — Shape `(N, 11, 512, 512)` Float32
| Channel | Source | Normalization |
|:--------|:-------|:--------------|
| `[0:3]` | LISS-IV Ground Truth (Green, Red, NIR) | Min-max to [0, 1] per-band global stats |
| `[3:5]` | S1 RTC γ⁰ (VV, VH) in **dB** | Clip [-30, 0] dB → scale to [0, 1] |
| `[5:9]` | S2 L2A (B2, B3, B4, B8) quality-gated | Min-max to [0, 1] per-band |
| `[9]` | Layover/Shadow Mask (LSM) | Binary {0, 1} or continuous [0, 1] |
| `[10]` | Terrain (Normalized Elevation + Slope) | Min-max to [0, 1] |

> [!IMPORTANT]
> **Channels [0:3] store the CLOUD-FREE LISS-IV.** Synthetic clouds are applied as a **runtime augmentation** during training, not pre-baked. This allows varying cloud patterns every epoch. The cloud mask `M` is generated on-the-fly and used both to corrupt Ch [0:3] and as a conditioning signal.

#### Target Tensor: `Y` — Shape `(N, 3, 512, 512)` Float32
| Channel | Source | Normalization |
|:--------|:-------|:--------------|
| `[0:3]` | LISS-IV Ground Truth (Green, Red, NIR) | Same normalization as input [0:3] |

> Note: `X[0:3]` and `Y[0:3]` are identical at rest. During training, `X[0:3]` is corrupted with synthetic clouds; `Y[0:3]` remains the clean target.

### 4.3 Data Splits
| Split | Allocation | Strategy |
|:------|:-----------|:---------|
| Train | 70% | Random, stratified by terrain slope (flat/moderate/steep) |
| Validation | 15% | Same stratification, disjoint scenes where possible |
| Test | 15% | Fully held-out scenes (spatial independence) |

### 4.4 SAR dB Conversion
Before staging, convert HyP3 power-scale backscatter to dB:
```python
gamma0_db = 10 * np.log10(np.clip(gamma0_power, 1e-10, None))
# Clip to [-30, 0] dB range (physical range for terrestrial surfaces)
gamma0_db = np.clip(gamma0_db, -30, 0)
# Normalize to [0, 1]
gamma0_norm = (gamma0_db + 30) / 30
```

### 4.5 Script
- **`scripts/stage_tensors_hdf5.py`** [NEW]
  - Reads aligned multi-band cubes from Phase 2
  - Tiles into 512×512 patches, filters NoData
  - Applies per-channel normalization
  - Stratified train/val/test split
  - Writes `dataset_train.h5`, `dataset_val.h5`, `dataset_test.h5`
  - HDF5 config: chunked (1, 11, 512, 512), LZF compression

### 4.6 Verification
- Load random patches and visualize RGB composites
- Verify normalization ranges are within expected bounds
- Confirm split sizes and stratification balance

---

# Part B: Model Development & Deployment

## Phase 5: Cloud Masking Model (Inference-Time)
**Goal:** Train a lightweight segmentation model that detects clouds in LISS-IV imagery at inference time, to generate the conditioning mask for the diffusion model.

> [!NOTE]
> This model is needed for **inference** (real-world deployment), not for training (where we use synthetic clouds with known masks). At inference time, we receive a cloudy LISS-IV scene and need to automatically detect which pixels are clouded before feeding them to the diffusion inpainter.

### 5.1 Architecture
- **Model:** SegFormer-B0 or Swin-UNet-Tiny
  - Lightweight enough for edge deployment
  - Input: 3-channel LISS-IV (Green, Red, NIR)
  - Output: Binary cloud/shadow probability mask
- **Why not a simple threshold?** LISS-IV lacks SWIR bands, making spectral cloud detection unreliable (clouds and snow are both bright in VNIR). A learned model can use spatial context and texture cues.

### 5.2 Training Data for Cloud Masking
- Use our synthetic cloud engine (Phase 3) to generate training pairs:
  - Input: Synthetically clouded LISS-IV patch
  - Target: Known synthetic cloud mask M
- Augmentations: Random rotations, flips, brightness/contrast jitter (Albumentations)

### 5.3 Training Protocol
| Parameter | Value |
|:----------|:------|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-2 |
| Loss | Dice Loss + Binary Cross-Entropy (combined) |
| Batch Size | 16 (on single GPU) |
| Epochs | 50 (with early stopping, patience=10) |
| Metrics | IoU, F1-Score, Precision, Recall |

### 5.4 Scripts
- **`scripts/train_cloud_mask.py`** [NEW]
- **`models/cloud_segformer.py`** [NEW]

### 5.5 Target Metrics
- Cloud IoU > 0.85
- Shadow IoU > 0.70
- False positive rate < 5% (critical: don't mask snow as cloud)

---

## Phase 6: TA-MMLD Core Diffusion Model
**Goal:** Train the conditional latent diffusion model that reconstructs cloud-free LISS-IV imagery using multi-modal guidance.

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LATENT SPACE (z)                       │
│                                                          │
│   Encoder E(x):  Cloudy LISS-IV → z_corrupted           │
│   Encoder E(y):  Clean LISS-IV → z_clean  (target)      │
│                                                          │
│   ┌──────────────────────────────────────┐               │
│   │     Conditional UNet Denoiser        │               │
│   │     ─────────────────────────        │               │
│   │  Input: z_noisy (latent + noise)     │               │
│   │  Conditioning:                        │               │
│   │    • Cloud Mask M (spatial mask)      │               │
│   │    • S1 γ⁰ VV/VH (structural guide)  │               │
│   │    • LSM (SAR reliability gate)       │               │
│   │    • S2 Optical (spectral baseline)   │               │
│   │    • DEM/Slope (terrain context)      │               │
│   │                                       │               │
│   │  Cross-Attention:                     │               │
│   │    attn_weight *= (1 - LSM)  for SAR  │               │
│   │    (downweight SAR in shadow zones)   │               │
│   └──────────────────────────────────────┘               │
│                                                          │
│   Decoder D(z): z_denoised → Reconstructed LISS-IV      │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Key Design: Topographic Attention Gating
The **LSM channel** is not just another input — it modulates the cross-attention mechanism:
```python
# In the cross-attention layer:
# S1 features attend to the UNet features, BUT
# attention weights are suppressed where LSM=1 (shadow/layover zones)
attn_weights = softmax(Q @ K.T / sqrt(d_k))
attn_weights_gated = attn_weights * (1 - lsm_spatial)  # Gate SAR influence
# In shadow zones: model relies on S2 optical + learned priors instead
```

This is the core innovation — it prevents the model from hallucinating structures based on unreliable SAR data in radar shadow pockets.

### 6.3 Variational Autoencoder (VAE)
- Use a pre-trained VAE (from Stable Diffusion or train a lightweight custom one)
- Latent space dimensionality: z ∈ R^(4×64×64) (for 512×512 input)
- If using a pre-trained VAE: fine-tune the decoder to output 3-channel VNIR instead of RGB

### 6.4 Training Protocol
| Parameter | Value |
|:----------|:------|
| Framework | PyTorch + Hugging Face Diffusers |
| Backbone | UNet2DConditionModel (customized for 11-ch conditioning) |
| Noise Scheduler | DDPM (1000 steps train) / DDIM (50 steps inference) |
| Optimizer | AdamW, lr=1e-4, cosine annealing to 1e-6 |
| Batch Size | 4–8 per GPU (gradient accumulation as needed) |
| Mixed Precision | FP16 (CUDA) or BF16 (ROCm) |
| Epochs | 200–500 (early stopping on val SSIM) |

### 6.5 Loss Functions (Multi-Objective)
```python
L_total = l1 * L_denoise + l2 * L_perceptual + l3 * L_SAM

# 1. Latent Denoising Loss (primary)
L_denoise = MSE(epsilon_predicted, epsilon_true)  # Standard diffusion objective

# 2. Perceptual + Structural Loss (decoded pixel space)
L_perceptual = lambda_ssim * (1 - SSIM(Y_pred, Y_true))
             + lambda_vgg  * VGG_FeatureLoss(Y_pred, Y_true)

# 3. Spectral Angle Mapper Loss (radiometric fidelity)
L_SAM = arccos(dot(Y_pred, Y_true) / (norm(Y_pred) * norm(Y_true)))
# Ensures band-to-band ratios are preserved (critical for NDVI computation)
```

**Loss weights (initial):** l1=1.0, l2=0.1, l3=0.05

### 6.6 Training Curriculum
| Stage | Epochs | Cloud Coverage | Description |
|:------|:-------|:---------------|:------------|
| Warmup | 1–50 | 10%–30% (thin) | Easy cases — model learns basic reconstruction |
| Core | 50–300 | 20%–60% (mixed) | Progressive difficulty — cirrus + cumulus |
| Hardening | 300–500 | 40%–80% (heavy) | Stress test — thick clouds, full occlusion |

### 6.7 Scripts & Modules
- **`models/ta_mmld_unet.py`** [NEW] — Custom UNet with topographic attention gating
- **`models/vae.py`** [NEW] — VAE encoder/decoder (or wrapper for pre-trained)
- **`scripts/train_ta_mmld.py`** [NEW] — Main training loop with curriculum scheduling
- **`scripts/dataset_loader.py`** [NEW] — HDF5 PyTorch Dataset with runtime cloud augmentation
- **`configs/train_config.yaml`** [NEW] — Hyperparameter configuration

### 6.8 Compute Requirements
| Resource | Minimum | Recommended |
|:---------|:--------|:------------|
| GPU VRAM | 24 GB (single GPU) | 48+ GB (Instinct MI250) |
| Training Time | ~3–5 days (single A100) | ~1–2 days (4x Instinct) |
| Disk (HDF5) | ~50 GB | ~50 GB |

---

## Phase 7: Uncertainty Estimation & Operational Deployment
**Goal:** Quantify per-pixel reconstruction confidence and package the pipeline for production inference.

### 7.1 Monte Carlo Dropout Inference
At inference time, run T=5–10 stochastic forward passes with different random noise seeds:
```python
predictions = []
for t in range(T):
    z_noisy = add_noise(z_encoded, noise_seed=t)
    y_pred = denoise_and_decode(z_noisy, conditioning)
    predictions.append(y_pred)

# Per-pixel mean and variance
Y_mean = torch.stack(predictions).mean(dim=0)     # Best estimate
Y_var  = torch.stack(predictions).var(dim=0)       # Uncertainty map
```

### 7.2 Uncertainty Calibration
- High uncertainty should correlate with:
  - Thick cloud centers (most information destroyed)
  - SAR shadow zones (LSM=1, no reliable structural guide)
  - Cloud edges (boundary artifacts)
- Low uncertainty should correlate with:
  - Clear pixels (trivial reconstruction)
  - Thin cirrus (easy to see through with SAR)

### 7.3 Analysis-Ready Product Export
Package final outputs as multi-band GeoTIFFs with full geospatial metadata:

| Band | Content | Resolution |
|:-----|:--------|:-----------|
| 1 | Reconstructed Green | 5.0m |
| 2 | Reconstructed Red | 5.0m |
| 3 | Reconstructed NIR | 5.0m |
| 4 | Cloud Mask (detected) | 5.0m |
| 5 | Uncertainty / Confidence | 5.0m |

```python
# Write with rasterio, preserving CRS and transform from input LISS-IV
with rasterio.open(output_path, 'w', driver='GTiff',
                   count=5, dtype='float32',
                   crs='EPSG:32645', transform=liss4_transform,
                   width=width, height=height) as dst:
    dst.write(green_band, 1)
    dst.write(red_band, 2)
    dst.write(nir_band, 3)
    dst.write(cloud_mask, 4)
    dst.write(uncertainty, 5)
```

### 7.4 Operational Inference Pipeline
```
Input: Cloudy LISS-IV Scene (GeoTIFF)
  |
  |-- [Cloud Masking Model] --> Cloud Mask M
  |
  |-- [ASF Search] --> Nearest S1 granule --> [HyP3 RTC] --> g0 VV/VH + LSM
  |
  |-- [STAC Search] --> Nearest S2 scene --> Quality-gated bands
  |
  |-- [DEM Fetch] --> Copernicus 30m
  |
  +-- [Grid Harmonize] --> 11-channel aligned stack
         |
         v
  [TA-MMLD Inference] x T stochastic passes
         |
         v
  Output: 5-band GeoTIFF (RGB + Mask + Uncertainty)
```

### 7.5 Scripts
- **`scripts/inference.py`** [NEW] — End-to-end inference pipeline
- **`scripts/export_geotiff.py`** [NEW] — GeoTIFF packaging with metadata

---

# Quantitative Evaluation Protocol

### Target Metrics (Synthetic Cloud Validation Set)
| Metric | Target | Interpretation |
|:-------|:-------|:---------------|
| **PSNR** | > 32 dB | Signal fidelity relative to noise floor |
| **SSIM** | > 0.90 | Structural preservation of features |
| **SAM** | < 4.0 degrees | Spectral angle deviation across VNIR bands |
| **LPIPS** | < 0.15 | Perceptual similarity (learned metric) |
| **Cloud Mask IoU** | > 0.85 | Cloud detection accuracy |

### Terrain-Stratified Evaluation
| Terrain Class | Slope Range | Evaluation Focus |
|:------------- |:------------|:-----------------|
| Flat Valleys | 0–10 degrees | Baseline performance, agricultural fields |
| Moderate Hills | 10–30 degrees | Mixed terrain, settlement continuity |
| Steep Mountains | 30+ degrees | SAR shadow zones, LSM effectiveness |

### Qualitative Verification
- **QGIS Inspection:** Overlay reconstructed scenes and verify road/river/bridge continuity
- **NDVI Consistency:** Compute NDVI from reconstructed bands and compare to clear-sky reference
- **Confidence Alignment:** Verify high-uncertainty pixels correlate with thick cloud centers and SAR shadow zones

---

# Timeline Estimate

| Phase | Duration | Dependencies | Key Deliverable |
|:------|:---------|:-------------|:----------------|
| **Phase 1:** HyP3 RTC | 1–2 days | NASA Earthdata auth | `sentinel1_rtc/` directory |
| **Phase 2:** Grid Harmonization | 2–3 days | Phase 1 complete | Aligned multi-band cubes |
| **Phase 3:** Synthetic Clouds | 1–2 days | None (parallel) | `CloudGenerator` class |
| **Phase 4:** HDF5 Staging | 1 day | Phases 2+3 | `dataset_train.h5` |
| **Phase 5:** Cloud Masking | 3–5 days | Phase 4 | Trained SegFormer checkpoint |
| **Phase 6:** TA-MMLD Training | 1–3 weeks | Phases 4+5 + GPU cluster | Trained diffusion model |
| **Phase 7:** Deployment | 3–5 days | Phase 6 | Inference pipeline + GeoTIFF export |

**Total Estimated Time:** 4–6 weeks (assuming GPU access for Phase 6)

---

# Complete File Manifest

### Existing (Phase 1 Complete)
| File | Purpose |
|:-----|:--------|
| [`auth_nasa.py`](file:///Users/ishanpetkar/Smart%20India%20Horizon%202026/scripts/auth_nasa.py) | NASA Earthdata authentication |
| [`download_auxiliary_data.py`](file:///Users/ishanpetkar/Smart%20India%20Horizon%202026/scripts/download_auxiliary_data.py) | S1/S2/DEM download orchestrator |
| [`verify_dataset_alignment.py`](file:///Users/ishanpetkar/Smart%20India%20Horizon%202026/scripts/verify_dataset_alignment.py) | Metadata-only verification |
| [`extract_liss4_metadata.py`](file:///Users/ishanpetkar/Smart%20India%20Horizon%202026/scripts/extract_liss4_metadata.py) | LISS-IV header extraction |

### To Build (Phases 2–7)
| File | Phase | Purpose |
|:-----|:------|:--------|
| `scripts/submit_hyp3_rtc.py` | 1 | Submit and download HyP3 RTC products |
| `scripts/grid_harmonize.py` | 2 | Co-register all layers to LISS-IV grid |
| `scripts/synthetic_clouds.py` | 3 | Cloud mask/texture generator |
| `scripts/stage_tensors_hdf5.py` | 4 | Tile and write HDF5 archives |
| `models/cloud_segformer.py` | 5 | Cloud masking model architecture |
| `scripts/train_cloud_mask.py` | 5 | Cloud mask training loop |
| `models/ta_mmld_unet.py` | 6 | Custom UNet with topographic gating |
| `models/vae.py` | 6 | VAE encoder/decoder |
| `scripts/dataset_loader.py` | 6 | HDF5 PyTorch Dataset with cloud augmentation |
| `scripts/train_ta_mmld.py` | 6 | Main diffusion training loop |
| `configs/train_config.yaml` | 6 | Hyperparameter config |
| `scripts/inference.py` | 7 | End-to-end inference pipeline |
| `scripts/export_geotiff.py` | 7 | GeoTIFF output packaging |
