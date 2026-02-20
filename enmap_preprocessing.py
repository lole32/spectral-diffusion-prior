"""
EnMAP Hyperspectral Image Preprocessing Tool

This script removes bands based on multiple criteria:
   - Empty/invalid bands
   - Wavelength extremes (<450 nm, >2400 nm)
   - Water absorption regions (1330-1480 nm, 1780-1980 nm)
   - Low SNR bands (percentile-based threshold)

Usage:
    python enmap_preprocessing.py --image /path/to/enmap.tif --snr-threshold 15
    python enmap_preprocessing.py --batch /path/to/enmap_dir/

"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnmapBandCleaner:

    def __init__(
        self,
        enmap_tif: str,
        output_dir: Optional[str] = None,
        snr_percentile: float = 15.0,
        min_wavelength: float = 450.0,
        max_wavelength: float = 2400.0,
        nodata_value: int = -32768
    ):
        self.enmap_tif     = enmap_tif
        self.output_dir    = output_dir or os.path.dirname(enmap_tif)
        self.snr_percentile = snr_percentile
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.nodata_value  = nodata_value

        if not os.path.exists(enmap_tif):
            raise FileNotFoundError(f"EnMAP TIFF not found: {enmap_tif}")

        os.makedirs(self.output_dir, exist_ok=True)

        name_no_ext = os.path.splitext(os.path.basename(enmap_tif))[0]
        self.output_file = os.path.join(
            self.output_dir,
            f"{name_no_ext}_clean{int(snr_percentile)}.tif"
        )

    def extract_wavelengths(self, src) -> np.ndarray:
        wavelengths = []
        for b in range(1, src.count + 1):
            band_tags = src.tags(b)
            if "CENTER_WAVELENGTH_NM" in band_tags:
                wl = float(band_tags["CENTER_WAVELENGTH_NM"])
            elif src.descriptions[b-1]:
                try:
                    wl = float(src.descriptions[b-1].split()[-1])
                except Exception:
                    wl = np.nan
            else:
                wl = np.nan
            wavelengths.append(wl)

        wavelengths = np.array(wavelengths)
        if np.any(np.isnan(wavelengths)):
            bad = np.where(np.isnan(wavelengths))[0] + 1
            raise ValueError(f"Wavelengths missing for bands: {bad.tolist()}")

        logger.info(f"  Wavelengths: {len(wavelengths)} bands, "
                    f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
        return wavelengths

    def identify_bands_to_remove(
        self,
        data: np.ndarray,
        wavelengths: np.ndarray,
        mask_combined: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:

        n_bands = data.shape[2]
        removal_reasons = {}

        # 1. Empty bands
        empty_bands = [
            i for i in range(n_bands)
            if np.count_nonzero(~mask_combined[:, :, i]) == 0
        ]
        removal_reasons['empty'] = empty_bands
        logger.info(f"  Empty bands:            {len(empty_bands)}")

        # 2. Wavelength out of range
        out_of_range_bands = np.where(
            (wavelengths < self.min_wavelength) | (wavelengths > self.max_wavelength)
        )[0].tolist()
        removal_reasons['out_of_range'] = out_of_range_bands
        logger.info(f"  Out-of-range bands:     {len(out_of_range_bands)}")

        # 3. Water absorption
        water_absorption_bands = np.where(
            ((wavelengths >= 1330) & (wavelengths <= 1480)) |
            ((wavelengths >= 1780) & (wavelengths <= 1980))
        )[0].tolist()
        removal_reasons['water_absorption'] = water_absorption_bands
        logger.info(f"  Water absorption bands: {len(water_absorption_bands)}")

        # 4. SNR — percentile computed ONLY on bands not already flagged
        logger.info(f"  Computing SNR per band...")
        snr = np.zeros(n_bands)
        for i in range(n_bands):
            band = np.where(mask_combined[:, :, i], np.nan, data[:, :, i])
            mean = np.nanmean(band)
            std  = np.nanstd(band)
            snr[i] = mean / std if (std > 0 and mean > 0) else 0

        already_flagged   = set(empty_bands + out_of_range_bands + water_absorption_bands)
        candidate_indices = np.array([
            i for i in range(n_bands)
            if i not in already_flagged and snr[i] > 0
        ])

        if len(candidate_indices) > 0:
            snr_threshold = np.percentile(snr[candidate_indices], self.snr_percentile)
            low_snr_bands = candidate_indices[snr[candidate_indices] < snr_threshold].tolist()
            logger.info(
                f"  Low SNR bands:          {len(low_snr_bands)} "
                f"(threshold={snr_threshold:.3f}, computed on {len(candidate_indices)} candidate bands)"
            )
        else:
            low_snr_bands = []
            snr_threshold = None
            logger.warning("  No valid SNR values — skipping SNR filter")

        removal_reasons['low_snr']       = low_snr_bands
        removal_reasons['snr_threshold'] = snr_threshold
        removal_reasons['snr_values']    = snr

        bands_to_remove = np.unique(
            empty_bands + out_of_range_bands + water_absorption_bands + low_snr_bands
        )

        logger.info(f"\n  Total bands:     {n_bands}")
        logger.info(f"  Bands removed:   {len(bands_to_remove)}")
        logger.info(f"  Bands remaining: {n_bands - len(bands_to_remove)}")

        return bands_to_remove, removal_reasons

    def visualize_cleaning(self, wavelengths, bands_to_remove):
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(wavelengths, label="All bands", linewidth=2)
        ax.scatter(
            bands_to_remove, wavelengths[bands_to_remove],
            color="red", label=f"Removed ({len(bands_to_remove)})", s=30, alpha=0.7
        )
        ax.axhspan(1330, 1480, alpha=0.15, color='blue', label='Water absorption')
        ax.axhspan(1780, 1980, alpha=0.15, color='blue')
        ax.set_xlabel("Band index"); ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Band Removal Report", fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        viz_path = self.output_file.replace('.tif', '_cleaning_report.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Visualization: {viz_path}")

    def clean_bands(self, visualize: bool = True) -> str:
        logger.info(f"\n{'='*60}")
        logger.info(f"EnMAP BAND CLEANING — SNR threshold: {self.snr_percentile}%")
        logger.info(f"{'='*60}")

        with rasterio.open(self.enmap_tif) as src:
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
            wavelengths = self.extract_wavelengths(src)

        data = np.transpose(data, (1, 2, 0))
        rows, cols, n_bands = data.shape
        logger.info(f"  Image: {rows} x {cols} x {n_bands}")

        mask_combined = (
            (data == self.nodata_value) | np.isnan(data) | (data >= 32000)
        )

        bands_to_remove, removal_reasons = self.identify_bands_to_remove(
            data, wavelengths, mask_combined
        )

        clean_bands       = np.delete(np.arange(n_bands), bands_to_remove)
        clean_wavelengths = wavelengths[clean_bands]

        if visualize:
            self.visualize_cleaning(wavelengths, bands_to_remove)

        profile.update(count=len(clean_bands), dtype=rasterio.int16, nodata=self.nodata_value)

        with rasterio.open(self.output_file, "w", **profile) as dst:
            for new_idx, orig_idx in enumerate(clean_bands, start=1):
                band_data = data[:, :, orig_idx].copy()
                band_data[mask_combined[:, :, orig_idx]] = self.nodata_value
                dst.write(band_data.astype(np.int16), new_idx)
                dst.set_band_description(new_idx, f"Band {new_idx:03d}: {wavelengths[orig_idx]:.3f}")
                dst.update_tags(new_idx, CENTER_WAVELENGTH_NM=f"{wavelengths[orig_idx]:.3f}")

            dst.update_tags(
                AREA_OR_POINT="Area",
                SNR_PERCENTILE_THRESHOLD=str(self.snr_percentile),
                ORIGINAL_BAND_COUNT=str(n_bands),
                CLEAN_BAND_COUNT=str(len(clean_bands)),
                REMOVED_BAND_COUNT=str(len(bands_to_remove))
            )

        logger.info(f"\n  Output:  {self.output_file}")
        logger.info(f"  Bands:   {len(clean_bands)} / {n_bands} retained")
        logger.info(f"  Range:   {clean_wavelengths.min():.2f} - {clean_wavelengths.max():.2f} nm\n")

        return self.output_file


class EnmapPreprocessor:
    def __init__(
        self,
        enmap_tif: str,
        output_dir: Optional[str] = None,
        snr_percentile: float = 15.0,
        min_wavelength: float = 450.0,
        max_wavelength: float = 2400.0
    ):
        self.enmap_tif      = enmap_tif
        self.output_dir     = output_dir or os.path.dirname(enmap_tif)
        self.snr_percentile = snr_percentile
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def process(self, visualize: bool = True) -> str:
        logger.info("\n" + "="*70)
        logger.info("EnMAP PREPROCESSING PIPELINE")
        logger.info("="*70)

        cleaner = EnmapBandCleaner(
            self.enmap_tif,
            output_dir=self.output_dir,
            snr_percentile=self.snr_percentile,
            min_wavelength=self.min_wavelength,
            max_wavelength=self.max_wavelength
        )
        return cleaner.clean_bands(visualize=visualize)


def batch_process(directory: str, pattern: str = "*.TIF",
                  snr_percentile: float = 15.0, visualize: bool = True) -> List[str]:
    import glob
    image_files = glob.glob(os.path.join(directory, pattern))
    if not image_files:
        logger.warning(f"No files matching '{pattern}' in {directory}")
        return []
    results = []
    for i, f in enumerate(image_files, 1):
        logger.info(f"\n[{i}/{len(image_files)}] {os.path.basename(f)}")
        try:
            results.append(EnmapPreprocessor(f, snr_percentile=snr_percentile).process(visualize))
        except Exception as e:
            logger.error(f"Failed: {e}")
    logger.info(f"\nBATCH COMPLETE: {len(results)}/{len(image_files)}")
    return results


def main():
    parser = argparse.ArgumentParser(description='EnMAP Preprocessing Tool')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', type=str, help='Path to EnMAP TIFF file')
    g.add_argument('--batch', type=str, help='Directory for batch processing')
    parser.add_argument('--snr-threshold', type=float, default=15.0)
    parser.add_argument('--min-wavelength', type=float, default=450.0)
    parser.add_argument('--max-wavelength', type=float, default=2400.0)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--pattern', type=str, default='*.TIF')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if args.quiet:     logger.setLevel(logging.ERROR)
    elif args.verbose: logger.setLevel(logging.DEBUG)

    try:
        if args.image:
            EnmapPreprocessor(
                args.image, args.output_dir,
                args.snr_threshold, args.min_wavelength, args.max_wavelength
            ).process()
            sys.exit(0)
        elif args.batch:
            results = batch_process(args.batch, args.pattern, args.snr_threshold)
            sys.exit(0 if results else 1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal: {e}")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()