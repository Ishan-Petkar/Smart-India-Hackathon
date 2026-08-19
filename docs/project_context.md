# Project Context & Evolution: Generative AI-Based Cloud Removal for LISS-IV
**Original Date:** August 14, 2026
**Last Updated:** August 19, 2026

## Project Overview
The objective is to develop a Generative AI model to remove cloud cover from high-resolution (5.0m) LISS-IV satellite imagery, specifically focusing on the mountainous North Eastern Region (NER) of India. 

## Architectural Paradigm
We adopted the **Topography-Aware Multi-Modal Latent Diffusion (TA-MMLD)** framework. This framework treats cloud removal as a multi-modal conditional image-to-image generative inpainting problem in latent space. It fuses LISS-IV, Sentinel-2 (Optical), Sentinel-1 (SAR), and DEM data to reconstruct clouded areas without hallucination.

### Hardware & Optimization Strategy
- **Environment:** AMD Server infrastructure equipped with Instinct GPUs.
- **Optimizations:** Leveraging `ROCm` (Radeon Open Compute) for high-performance PyTorch acceleration, distributed training strategies across multiple GPUs to handle the massive TA-MMLD parameter space, and high-throughput NVMe storage staging (via HDF5/Zarr) to maximize data ingestion rates for full-scale diffusion model training.

---

## Chronological Engineering History & Deep Audit Discoveries

Over the course of the project, a deep audit of the data and physical constraints led to massive architectural pivots. The following outlines the major challenges discovered and how they were resolved:

### 1. The "No Paired Data" Discovery
- **Problem:** A thorough dataset audit revealed that all 14 LISS-IV master scenes acquired were 100% clear-sky. We had no cloudy/clear pairs to train the diffusion model.
- **Solution (Phase 3 Rework):** Engineered an **SCL-Guided Stochastic Hybrid Cloud Augmentation Engine**. Instead of relying on pure procedural noise (which the model would exploit as a shortcut), we extract real cloud morphologies from the Sentinel-2 Scene Classification Layer (SCL) and perturb them continuously. 

### 2. SAR Terrain Correction Intractability
- **Problem:** Attempting to perform Radiometric Terrain Correction (RTC) on SAR data locally using our own Range-Doppler equations was too mathematically intractable and computationally heavy for our immediate timeline.
- **Solution:** Migrated the pipeline to use the **NASA Alaska Satellite Facility (ASF) HyP3** cloud computing service. This provides production-quality, terrain-flattened $\gamma^0$ backscatter and accurate Layover/Shadow Masks (LSM) effortlessly.

### 3. Corrupt Tiles and Download Instability
- **Problem:** Standard download scripts yielded corrupt DEM and Sentinel-2 TIFF files.
- **Solution:** Engineered `download_auxiliary_data.py`, leveraging a resilient **16-segment aria2c parallel download engine** with automatic retries and connection timeouts, ensuring a perfect sync of the dataset.

### 4. Grid Resolution Lock
- **Problem:** Confusion between nominal (5.8m) and delivered pixel geometries.
- **Solution:** Locked the master target grid to strictly **5.0m x 5.0m** in EPSG:32645 for all co-registration and HDF5 tensor staging.

### 5. Preventing SAR Hallucinations (Topographic Attention Gating)
- **Problem:** In extreme mountain topography, SAR signals in radar shadow zones are purely receiver thermal noise. A standard diffusion model would hallucinate false structural features when fed this noise.
- **Solution:** Engineered **Topographic Cross-Attention Gating**. Using the Sentinel-1 Layover/Shadow Mask (LSM), we explicitly suppress the attention weight for the SAR channel inside radar shadows. This physical guardrail forces the UNet to rely entirely on Sentinel-2 optical features and learned generative priors in those regions.

### 6. Phase 3 Cloud Synthesis Rework & Physics
- **Problem:** The original synthetic cloud engine used simple linear alpha-blending and lacked geographic isolation, risking massive data leakage (training on validation cloud shapes) and creating physically unrealistic clouds.
- **Solution:** Implemented strict train/val/test scene isolation before extracting cloud templates. Replaced linear blending with an optical thickness ($\tau$) representation and regime-dependent attenuation to ensure that Phase 5 (Segmentation) and Phase 6 (Diffusion) aren't given trivial shortcuts to solve the problem.

---

## Next Steps
- Execute Phase 2: **Grid Harmonization & Co-Registration Pipeline** to align the ASF HyP3 RTC products, Copernicus DEM, and Sentinel-2 imagery perfectly onto the 5.0m LISS-IV grid.
- Implement the refined **Phase 3 Synthetic Cloud Augmentation Engine** in code.
- Stage the co-registered tensors into multi-modal HDF5 chunks for high-throughput GPU training (Phase 4).
