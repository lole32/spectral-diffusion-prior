"""
EnMAP Hyperspectral Image Preprocessing Tool

This script provides comprehensive preprocessing for EnMAP L2A hyperspectral imagery:
1. Writes band metadata (wavelengths, FWHM, gain, offset) to TIFF tags
2. Removes problematic bands based on multiple criteria:
   - Wavelength extremes (<450 nm, >2400 nm)
   - Water absorption regions (1330-1480 nm, 1780-1980 nm)
   - Empty/invalid bands
   - Low SNR bands (percentile-based threshold)
3. Produces clean output with proper band descriptions

Usage:
    python enmap_preprocessing.py --image enmap.tif --metadata metadata.json
    python enmap_preprocessing.py --image enmap.tif --metadata metadata.json --snr-threshold 15
    python enmap_preprocessing.py --batch /path/to/enmap_dir/
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import warnings

warnings.filterwarnings('ignore')


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
# ENMAP METADATA WRITER
# ============================================================================
class EnmapMetadataWriter:
    """
    Write band metadata from STAC JSON to EnMAP TIFF tags.
    """

    def __init__(self, enmap_tif: str, metadata_json: str):
        self.enmap_tif = enmap_tif
        self.metadata_json = metadata_json

        if not os.path.exists(enmap_tif):
            raise FileNotFoundError(f"EnMAP TIFF not found: {enmap_tif}")
        if not os.path.exists(metadata_json):
            raise FileNotFoundError(f"Metadata JSON not found: {metadata_json}")

    def load_metadata(self) -> List[Dict]:
        logger.info(f"Loading metadata from: {os.path.basename(self.metadata_json)}")

        with open(self.metadata_json, "r") as f:
            stac = json.load(f)

        properties = stac.get("properties", {})
        eo_bands = properties.get("eo:bands", [])

        if not eo_bands:
            raise ValueError("No eo:bands found in metadata JSON")

        logger.info(f"Found {len(eo_bands)} spectral bands in metadata")
        return eo_bands

    def write_band_metadata(self) -> int:
        logger.info(f"\n{'='*60}")
        logger.info(f"Writing band metadata to TIFF")
        logger.info(f"{'='*60}\n")

        eo_bands = self.load_metadata()

        with rasterio.open(self.enmap_tif, "r+") as dst:
            if dst.count != len(eo_bands):
                logger.warning(
                    f"TIFF has {dst.count} bands, "
                    f"metadata has {len(eo_bands)} bands"
                )

            for band in eo_bands:
                band_index = int(band["name"])

                center_wl = band.get("center_wavelength") or band.get("eo:center_wavelength")
                fwhm = band.get("full_width_half_max") or band.get("eo:full_width_half_max")
                gain = band.get("enmap:gain_of_band")
                offset = band.get("enmap:offset_of_band")

                tags = {
                    "CENTER_WAVELENGTH_NM": str(center_wl) if center_wl is not None else None,
                    "FWHM_NM": str(fwhm) if fwhm is not None else None,
                    "GAIN": str(gain) if gain is not None else None,
                    "OFFSET": str(offset) if offset is not None else None,
                }

                tags = {k: v for k, v in tags.items() if v not in (None, "None")}

                dst.update_tags(band_index, **tags)

                description = f"{center_wl} nm" if center_wl is not None else ""
                dst.set_band_description(band_index, description)

                if band_index <= 3 or band_index % 50 == 0:
                    logger.debug(f"Band {band_index:03d} description set to: {description}")

        logger.info(f"All band tags written to TIFF\n")
        return len(eo_bands)


# ============================================================================
# ENMAP BAND CLEANER
# ============================================================================
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
        self.enmap_tif = enmap_tif
        self.output_dir = output_dir or os.path.dirname(enmap_tif)
        self.snr_percentile = snr_percentile
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.nodata_value = nodata_value

        if not os.path.exists(enmap_tif):
            raise FileNotFoundError(f"EnMAP TIFF not found: {enmap_tif}")

        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.basename(enmap_tif)
        name_no_ext = os.path.splitext(base_name)[0]
        self.output_file = os.path.join(
            self.output_dir,
            f"{name_no_ext}_clean{int(snr_percentile)}.tif"
        )

    def extract_wavelengths(self, src: rasterio.DatasetReader) -> np.ndarray:
        logger.info("Extracting wavelengths from band metadata...")

        wavelengths = []

        for b in range(1, src.count + 1):
            band_tags = src.tags(b)
            if "CENTER_WAVELENGTH_NM" in band_tags:
                wl = float(band_tags["CENTER_WAVELENGTH_NM"])
            elif src.descriptions[b-1]:
                try:
                    wl = float(src.descriptions[b-1].split()[0])
                except:
                    wl = np.nan
            else:
                global_tags = src.tags()
                if "wavelength" in global_tags:
                    wl_list = global_tags["wavelength"].split(",")
                    wl = float(wl_list[b-1])
                else:
                    wl = np.nan

            wavelengths.append(wl)

        wavelengths = np.array(wavelengths)

        if np.any(np.isnan(wavelengths)):
            bad = np.where(np.isnan(wavelengths))[0] + 1
            raise ValueError(f"Wavelengths missing for bands: {bad.tolist()}")

        logger.info(f"  Extracted wavelengths for {len(wavelengths)} bands")
        logger.info(f"  Range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")

        return wavelengths

    def identify_bands_to_remove(
        self,
        data: np.ndarray,
        wavelengths: np.ndarray,
        mask_combined: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:

        n_bands = data.shape[2]
        removal_reasons = {}

        logger.info("Identifying bands to remove...")

        empty_bands = [
            i for i in range(n_bands)
            if np.count_nonzero(~mask_combined[:, :, i]) == 0
        ]
        removal_reasons['empty'] = empty_bands
        logger.info(f"  Empty bands: {len(empty_bands)}")

        out_of_range_bands = np.where(
            (wavelengths < self.min_wavelength) | (wavelengths > self.max_wavelength)
        )[0].tolist()
        removal_reasons['out_of_range'] = out_of_range_bands
        logger.info(
            f"  Out of range (<{self.min_wavelength} or >{self.max_wavelength} nm): "
            f"{len(out_of_range_bands)}"
        )

        water_absorption_bands = np.where(
            ((wavelengths >= 1330) & (wavelengths <= 1480)) |
            ((wavelengths >= 1780) & (wavelengths <= 1980))
        )[0].tolist()
        removal_reasons['water_absorption'] = water_absorption_bands
        logger.info(f"  Water absorption regions: {len(water_absorption_bands)}")

        logger.info(f"  Computing SNR for all bands...")
        snr = np.zeros(n_bands)
        for i in range(n_bands):
            band = np.where(mask_combined[:, :, i], np.nan, data[:, :, i])
            mean = np.nanmean(band)
            std = np.nanstd(band)
            snr[i] = mean / std if (std > 0 and mean > 0) else 0

        valid_snr = snr > 0
        if np.any(valid_snr):
            snr_threshold = np.percentile(snr[valid_snr], self.snr_percentile)
            low_snr_bands = np.where((snr < snr_threshold) & valid_snr)[0].tolist()
            logger.info(
                f"  Low SNR (< {self.snr_percentile}th percentile = {snr_threshold:.2f}): "
                f"{len(low_snr_bands)}"
            )
        else:
            low_snr_bands = []
            snr_threshold = None
            logger.warning("  No valid SNR values computed")

        removal_reasons['low_snr'] = low_snr_bands
        removal_reasons['snr_threshold'] = snr_threshold
        removal_reasons['snr_values'] = snr

        bands_to_remove = np.unique(
            empty_bands + out_of_range_bands + water_absorption_bands + low_snr_bands
        )

        logger.info(f"  Summary:")
        logger.info(f"  Total bands: {n_bands}")
        logger.info(f"  Bands to remove: {len(bands_to_remove)}")
        logger.info(f"  Remaining bands: {n_bands - len(bands_to_remove)}")

        return bands_to_remove, removal_reasons

    def visualize_cleaning(
        self,
        data: np.ndarray,
        wavelengths: np.ndarray,
        bands_to_remove: np.ndarray,
        removal_reasons: Dict,
        mask_combined: np.ndarray
    ):
        logger.info("Creating visualization...")

        n_bands = data.shape[2]
        rows, cols = data.shape[:2]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(wavelengths, label="All bands", linewidth=2)
        ax1.scatter(
            bands_to_remove,
            wavelengths[bands_to_remove],
            color="red",
            label=f"Removed ({len(bands_to_remove)})",
            s=30,
            alpha=0.6
        )
        ax1.set_ylabel("Wavelength (nm)", fontsize=12)
        ax1.set_xlabel("Band index", fontsize=12)
        ax1.set_title("Band Wavelength Distribution", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        ax1.axhspan(1330, 1480, alpha=0.2, color='blue', label='Water absorption')
        ax1.axhspan(1780, 1980, alpha=0.2, color='blue')

        viz_path = self.output_file.replace('.tif', '_cleaning_report.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        logger.info(f" Visualization saved: {viz_path}")

        plt.close()

    def clean_bands(self, visualize: bool = True) -> str:

        logger.info(f"\n{'='*60}")
        logger.info(f"EnMAP BAND CLEANING")
        logger.info(f"{'='*60}\n")
        logger.info(f"Input: {os.path.basename(self.enmap_tif)}")
        logger.info(f"SNR percentile threshold: {self.snr_percentile}")
        logger.info(f"Wavelength range: {self.min_wavelength} - {self.max_wavelength} nm\n")

        with rasterio.open(self.enmap_tif) as src:
            logger.info("Reading image data...")
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
            wavelengths = self.extract_wavelengths(src)

        data = np.transpose(data, (1, 2, 0))
        rows, cols, n_bands = data.shape

        logger.info(f"Image shape: {rows} x {cols} x {n_bands}")

        logger.info("\nCreating validity masks...")
        mask_nodata = (data == self.nodata_value)
        mask_nan = np.isnan(data)
        mask_saturated = data >= 32000
        mask_combined = mask_nodata | mask_nan | mask_saturated

        bands_to_remove, removal_reasons = self.identify_bands_to_remove(
            data, wavelengths, mask_combined
        )

        clean_bands = np.delete(np.arange(n_bands), bands_to_remove)
        clean_wavelengths = wavelengths[clean_bands]

        if visualize:
            self.visualize_cleaning(
                data, wavelengths, bands_to_remove, removal_reasons, mask_combined
            )

        logger.info(f" Writing clean image...")
        logger.info(f"Output: {os.path.basename(self.output_file)}")

        profile.update(
            count=len(clean_bands),
            dtype=rasterio.int16,
            nodata=self.nodata_value
        )

        with rasterio.open(self.output_file, "w", **profile) as dst:
            for new_idx, orig_idx in enumerate(clean_bands, start=1):
                band_data = data[:, :, orig_idx].copy()
                band_data[mask_combined[:, :, orig_idx]] = self.nodata_value

                dst.write(band_data.astype(np.int16), new_idx)

                desc = f"Band {new_idx:03d}: {wavelengths[orig_idx]:.2f} nm"
                dst.set_band_description(new_idx, desc)

                dst.update_tags(
                    new_idx,
                    CENTER_WAVELENGTH_NM=f"{wavelengths[orig_idx]:.2f}"
                )

            dst.update_tags(AREA_OR_POINT="Area")
            dst.update_tags(
                SNR_PERCENTILE_THRESHOLD=str(self.snr_percentile),
                ORIGINAL_BAND_COUNT=str(n_bands),
                CLEAN_BAND_COUNT=str(len(clean_bands)),
                REMOVED_BAND_COUNT=str(len(bands_to_remove))
            )

        logger.info(f"\n{'='*60}")
        logger.info(f" CLEANING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Retained bands: {len(clean_bands)} / {n_bands}")
        logger.info(f"Wavelength range: {clean_wavelengths.min():.2f} - {clean_wavelengths.max():.2f} nm")
        logger.info(f"{'='*60}\n")

        return self.output_file


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================
class EnmapPreprocessor:

    def __init__(
        self,
        enmap_tif: str,
        metadata_json: Optional[str] = None,
        output_dir: Optional[str] = None,
        snr_percentile: float = 15.0,
        min_wavelength: float = 450.0,
        max_wavelength: float = 2400.0
    ):
        self.enmap_tif = enmap_tif
        self.output_dir = output_dir or os.path.dirname(enmap_tif)
        self.snr_percentile = snr_percentile
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

        if metadata_json is None:
            metadata_json = enmap_tif.replace(".TIF", "_metadata.json")
            if not os.path.exists(metadata_json):
                metadata_json = enmap_tif.replace(".tif", "_metadata.json")

        self.metadata_json = metadata_json

    def process(self, visualize: bool = True) -> str:

        logger.info("\n" + "="*70)
        logger.info("EnMAP PREPROCESSING PIPELINE")
        logger.info("="*70 + "\n")

        if os.path.exists(self.metadata_json):
            logger.info("STEP 1: Writing band metadata to TIFF")
            writer = EnmapMetadataWriter(self.enmap_tif, self.metadata_json)
            writer.write_band_metadata()
        else:
            logger.warning(f"Metadata JSON not found: {self.metadata_json}")
            logger.warning("Skipping metadata writing step")

        logger.info("STEP 2: Cleaning bands")
        cleaner = EnmapBandCleaner(
            self.enmap_tif,
            output_dir=self.output_dir,
            snr_percentile=self.snr_percentile,
            min_wavelength=self.min_wavelength,
            max_wavelength=self.max_wavelength
        )
        output_file = cleaner.clean_bands(visualize=visualize)

        logger.info("\n" + "="*70)
        logger.info(" PREPROCESSING PIPELINE COMPLETE")
        logger.info("="*70 + "\n")

        return output_file


# ============================================================================
# BATCH PROCESSING
# ============================================================================
def batch_process(
    directory: str,
    pattern: str = "*.TIF",
    snr_percentile: float = 15.0,
    visualize: bool = True
) -> List[str]:

    import glob

    logger.info("\n" + "="*70)
    logger.info(f"BATCH PREPROCESSING: {directory}")
    logger.info("="*70 + "\n")

    search_pattern = os.path.join(directory, pattern)
    image_files = glob.glob(search_pattern)

    if not image_files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return []

    logger.info(f"Found {len(image_files)} images to process\n")

    results = []
    for i, enmap_tif in enumerate(image_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{len(image_files)}] Processing: {os.path.basename(enmap_tif)}")
        logger.info(f"{'='*70}\n")

        try:
            preprocessor = EnmapPreprocessor(
                enmap_tif,
                snr_percentile=snr_percentile
            )
            output_file = preprocessor.process(visualize=True)
            results.append(output_file)
        except Exception as e:
            logger.error(f"Failed to process {enmap_tif}: {e}")
            continue

    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH PREPROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed: {len(results)}/{len(image_files)}")

    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='EnMAP Hyperspectral Image Preprocessing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enmap_preprocessing.py --image enmap.tif
  python enmap_preprocessing.py --image enmap.tif --metadata metadata.json
  python enmap_preprocessing.py --image enmap.tif --snr-threshold 20
  python enmap_preprocessing.py --image enmap.tif --min-wavelength 400 --max-wavelength 2500
  python enmap_preprocessing.py --batch /path/to/enmap_directory/
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to EnMAP TIFF file')
    input_group.add_argument('--batch', type=str, help='Directory for batch processing')

    parser.add_argument('--metadata', type=str, help='Path to metadata JSON')
    parser.add_argument('--snr-threshold', type=float, default=15.0)
    parser.add_argument('--min-wavelength', type=float, default=450.0)
    parser.add_argument('--max-wavelength', type=float, default=2400.0)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--pattern', type=str, default='*.TIF')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        if args.image:
            preprocessor = EnmapPreprocessor(
                enmap_tif=args.image,
                metadata_json=args.metadata,
                output_dir=args.output_dir,
                snr_percentile=args.snr_threshold,
                min_wavelength=args.min_wavelength,
                max_wavelength=args.max_wavelength
            )
            preprocessor.process(visualize=True)
            sys.exit(0)

        elif args.batch:
            results = batch_process(
                directory=args.batch,
                pattern=args.pattern,
                snr_percentile=args.snr_threshold,
                visualize=True
            )
            sys.exit(0 if results else 1)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
