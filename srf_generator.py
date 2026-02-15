#!/usr/bin/env python3
"""
EnMAP to Sentinel-2 Spectral Response Function (SRF) Generator

This script generates Spectral Response Functions (SRF) for fusing EnMAP hyperspectral
imagery to Sentinel-2 multispectral bands. It creates a matrix that maps EnMAP bands
to S2 bands using Gaussian approximations based on center wavelengths and FWHM.

Usage:
    python srf_generator.py --image enmap.tif
    python srf_generator.py --batch /path/to/enmap_directory/
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SENTINEL-2 BAND SPECIFICATIONS
# ============================================================================
S2_BAND_SPECS = {
    'B02': {'name': 'Blue',   'center_wavelength': 492.4, 'fwhm': 66,  'resolution': 10},
    'B03': {'name': 'Green',  'center_wavelength': 559.8, 'fwhm': 36,  'resolution': 10},
    'B04': {'name': 'Red',    'center_wavelength': 664.6, 'fwhm': 31,  'resolution': 10},
    'B08': {'name': 'NIR',    'center_wavelength': 832.8, 'fwhm': 106, 'resolution': 10},
}

DEFAULT_S2_BANDS = ['B02', 'B03', 'B04', 'B08']


# ============================================================================
# SRF GENERATOR CLASS
# ============================================================================
class SRFGenerator:
    """
    Generate Spectral Response Functions for EnMAP to Sentinel-2 fusion.
    """
    
    def __init__(self, enmap_image: str, output_dir: Optional[str] = None):
        self.enmap_image = enmap_image
        self.output_dir = output_dir or str(Path(enmap_image).parent)
        self.s2_bands = DEFAULT_S2_BANDS
        
        if not os.path.exists(enmap_image):
            raise FileNotFoundError(f"EnMAP image not found: {enmap_image}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.enmap_wavelengths = None
        self.enmap_band_count = None
        self.srf_matrix = None
    
    @staticmethod
    def extract_wavelength_from_description(description: str) -> Optional[float]:
        if not description:
            return None
        match = re.search(r'(\d+\.?\d*)\s*nm', description)
        if match:
            return float(match.group(1))
        return None
    
    def load_enmap_wavelengths(self) -> Tuple[np.ndarray, int]:
        logger.info(f"Loading EnMAP wavelengths from: {os.path.basename(self.enmap_image)}")
        with rasterio.open(self.enmap_image) as src:
            n_bands = src.count
            wavelengths = []
            for i, desc in enumerate(src.descriptions, 1):
                wl = self.extract_wavelength_from_description(desc or "")
                if wl is None:
                    band_tags = src.tags(i)
                    if 'CENTER_WAVELENGTH_NM' in band_tags:
                        wl = float(band_tags['CENTER_WAVELENGTH_NM'])
                if wl is None:
                    raise ValueError(f"Could not extract wavelength for band {i}.")
                wavelengths.append(wl)
        wavelengths = np.array(wavelengths, dtype=np.float32)
        logger.info(f"Loaded {n_bands} EnMAP bands. Wavelength range: {wavelengths.min():.2f}-{wavelengths.max():.2f} nm")
        return wavelengths, n_bands
    
    @staticmethod
    def gaussian_spectral_response(wavelengths: np.ndarray, center: float, fwhm: float) -> np.ndarray:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        response = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
        response = response / (response.sum() + 1e-10)
        return response.astype(np.float32)
    
    def compute_srf_matrix(self) -> np.ndarray:
        self.enmap_wavelengths, self.enmap_band_count = self.load_enmap_wavelengths()
        n_s2_bands = len(self.s2_bands)
        n_enmap_bands = self.enmap_band_count
        R = np.zeros((n_s2_bands, n_enmap_bands), dtype=np.float32)
        
        for i, band_id in enumerate(self.s2_bands):
            specs = S2_BAND_SPECS[band_id]
            response = self.gaussian_spectral_response(self.enmap_wavelengths, specs['center_wavelength'], specs['fwhm'])
            R[i, :] = response
        self.srf_matrix = R
        return R
    
    def visualize_srf(self, save: bool = True) -> Optional[str]:
        if self.srf_matrix is None:
            raise ValueError("SRF matrix not computed.")
        
        n_bands = len(self.s2_bands)
        nrows, ncols = (2, 2) if n_bands <= 4 else (3, 3)
        fig = plt.figure(figsize=(6*ncols, 5*nrows))
        gs = GridSpec(nrows, ncols, figure=fig)
        
        for i, band_id in enumerate(self.s2_bands):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])
            specs = S2_BAND_SPECS[band_id]
            ax.plot(self.enmap_wavelengths, self.srf_matrix[i, :], 'b-', linewidth=2)
            ax.axvline(specs['center_wavelength'], color='r', linestyle='--', linewidth=2)
            fwhm_low = specs['center_wavelength'] - specs['fwhm']/2
            fwhm_high = specs['center_wavelength'] + specs['fwhm']/2
            ax.axvspan(fwhm_low, fwhm_high, alpha=0.15, color='red')
            peak_idx = np.argmax(self.srf_matrix[i, :])
            ax.plot(self.enmap_wavelengths[peak_idx], self.srf_matrix[i, peak_idx], 'go', markersize=8)
            ax.set_title(f"S2 {band_id} ({specs['name']}) → EnMAP", fontsize=12)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Spectral Response')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_ylim([0, self.srf_matrix[i, :].max() * 1.1])
        
        for i in range(len(self.s2_bands), nrows*ncols):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "enmap_s2_srf.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig_path
    
    def save_srf_matrix(self) -> str:
        if self.srf_matrix is None:
            raise ValueError("SRF matrix not computed.")
        mat_path = os.path.join(self.output_dir, "enmap_s2_srf.mat")
        mat_data = {
            'R': self.srf_matrix,
            'enmap_wavelengths': self.enmap_wavelengths,
            'enmap_band_count': self.enmap_band_count,
            's2_band_ids': self.s2_bands,
            's2_centers': np.array([S2_BAND_SPECS[band]['center_wavelength'] for band in self.s2_bands]),
            's2_fwhm': np.array([S2_BAND_SPECS[band]['fwhm'] for band in self.s2_bands]),
            's2_names': [S2_BAND_SPECS[band]['name'] for band in self.s2_bands],
            'description': 'Spectral Response Functions for EnMAP→S2 fusion.',
            'enmap_image': self.enmap_image
        }
        sio.savemat(mat_path, mat_data)
        return mat_path
    
    def generate(self) -> Tuple[str, str]:
        self.compute_srf_matrix()
        viz_path = self.visualize_srf(save=True)
        mat_path = self.save_srf_matrix()
        return mat_path, viz_path


# ============================================================================
# BATCH PROCESSING
# ============================================================================
def batch_generate_srf(directory: str) -> List[Tuple[str, str]]:
    import glob
    image_files = glob.glob(os.path.join(directory, "*.tif"))
    if not image_files:
        logger.warning(f"No TIFF files found in {directory}")
        return []
    results = []
    for enmap_image in image_files:
        try:
            generator = SRFGenerator(enmap_image)
            mat_path, viz_path = generator.generate()
            results.append((mat_path, viz_path))
        except Exception as e:
            logger.error(f"Failed to process {enmap_image}: {e}")
            continue
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='EnMAP to Sentinel-2 SRF Generator')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to EnMAP TIFF image')
    input_group.add_argument('--batch', type=str, help='Directory for batch processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Suppress all logging except errors')
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        if args.image:
            generator = SRFGenerator(args.image)
            mat_path, viz_path = generator.generate()
            logger.info(f"SRF matrix saved: {mat_path}")
            logger.info(f"Visualization saved: {viz_path}")
            sys.exit(0)
        elif args.batch:
            results = batch_generate_srf(args.batch)
            for mat, viz in results:
                logger.info(f"SRF matrix: {mat}")
                logger.info(f"Visualization: {viz}")
            sys.exit(0 if results else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
