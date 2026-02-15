# Spectral Diffusion Prior for Hyperspectral Image Super-Resolution

> This repository contains an implementation of the **Spectral Diffusion Prior (SDP) model** for fusion-based hyperspectral image (HSI) super-resolution, as proposed by Liu et al. ([paper attached](./SDP_paper.pdf)).  
The method fuses a low-resolution HSI with a high-resolution multispectral image using a diffusion-based spectral prior, achieving high-spatial-resolution HSI reconstruction.  
A complete pipeline is provided for Hyperspectral Image Super-Resolution on **EnMAP** and **Sentinel-2** data.

## 1. Extract EnMAP STAC metadata

Extracts and enriches EnMAP HSI image metadata using the DLR STAC API.
**Usage:**
```bash
python enmap_metadata_extractor.py --image /path/to/enmap.tif
```


## 2. Clean EnMAP bands 

Removes empty, water absorption, extreme and noisy bands based on SNR percentiles.
**Usage:**
```bash
python enmap_preprocessing.py --image /path/to/enmap.tif --metadata /path/to/metadata.json --snr-threshold 15 
```

## 3. Download and pre-process Sentinel-2 data 

Downloads Sentinel-2 .nc cube, reproject in WGS84/4326 and then split per date tiffs. After downloading, align with EnMAP image.
**Usage:**

- **Create Config Mode:**
  ```bash
  python s2_enmap_preprocessing.py --create-config
  ```

- **Download Mode:**
  ```bash
  python s2_enmap_preprocessing.py --mode download --config  /path/to/config.json
  ```
  
- **Align Mode:**
  ```bash
  python s2_enmap_preprocessing.py --mode align --s2-image /path/to/s2.tif --enmap-image /path/to/enmap.tif
  ```
                                                             
## 4. Generate proper SRF

**Usage:**
```bash
python srf_generator.py --image /path/to/enmap.tif
```

## 5. Create dataset

Creates dataset (.mat file) by first downsampling EnMAP (60m) and then cropping both EnMAP-S2 patches with ratio=6.
**Usage:**
```bash
python create_dataset.py --enmap_tif /path/to/enmap.tif --s2_tif /path/to/s2_tif --srf_mat /path/to/enmap_s2_srf.mat 
```

## 6. Run SDP

Executes the complete SDP pipeline (SDM, Blind, SDP) with specified ratio.
**Usage:**
```bash
python SDP_mlflow.py --nratio 6
```

## 7. Perform evaluation

Calculates spectral and spatial distortion of SR-HSI and exports geotiffs with user-defined buffer size. 
**Usage:**
```bash
python evaluation.py --data-mat /path/to/enmap_s2_data_r6.mat --sr-mat /path/to/X.mat --output-dir /path/to/outputs --original-s2-tif /path/to/s2_image.tif --export-geotiffs --border-crop 1
```