# Architectural & Strategic Reasoning

## 1. Why TA-MMLD (Topography-Aware Multi-Modal Latent Diffusion)?
- **The Mountain Problem:** Standard SAR-Optical fusion models fail in mountainous regions (like the NER). SAR signals suffer from severe geometric distortions such as *layover* (radar hits the top of a mountain before the base), *foreshortening*, and *radar shadow* (areas hidden from the radar pulse).
- **The Solution:** By explicitly passing a **Topographic Layover/Shadow Mask** into the diffusion model's attention mechanism, we teach the model *not* to trust SAR data in shadow regions, relying instead on temporal optical data or learned priors. 

## 2. Why Radiometric Terrain Correction (RTC)?
- Raw SAR (Level-1 GRD) is not geographically accurate over terrain. RTC uses a high-resolution DEM to correct the SAR backscatter and project it onto the true Earth surface. Without RTC, the SAR features will not align with the 5.8m LISS-IV optical grid, rendering the multi-modal fusion useless.

## 3. Why AMD Instinct Servers & ROCm?
- **Scale and Compute:** Training a Multi-Modal Latent Diffusion Model from scratch or performing heavy fine-tuning on high-resolution (5.8m) geospatial imagery requires massive VRAM and compute capabilities, far exceeding typical local workstations. AMD Instinct GPUs provide the necessary memory bandwidth and capacity.
- **ROCm Backend:** By utilizing AMD's ROCm stack with PyTorch, we can achieve near-native CUDA-level performance, allowing for large batch sizes, faster convergence, and the ability to train the full UNet rather than relying strictly on LoRA bottlenecks.
- **High-Throughput Data Staging (HDF5/Zarr):** While servers have fast networking and NVMe storage, feeding thousands of independent GeoTIFFs to multi-GPU setups still causes IO starvation. Compacting the paired datasets into HDF5/Zarr chunks ensures that the Instinct GPUs are fully saturated with data during training.

## 4. Why API-based Data Ingestion (ASF / STAC)?
- Manual downloading of hundreds of GBs of satellite imagery is error-prone. The legacy scripts failed due to modern NASA URS cookie/EULA policies.
- Using the `asf_search` Python SDK and STAC APIs ensures programmatic, authenticated, and resumable downloads, which is critical when we scale from the 10-file prototype to the full dataset.
