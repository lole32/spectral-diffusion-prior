"""
EnMAP Metadata Extraction Tool

This script extracts and enriches EnMAP HSI imagery metadata using the DLR STAC API.
It connects to the EOC STAC catalog, finds matching STAC items based on image
filename and bbox, and extracts detailed band metadata (wavelengths, FWHM).

Usage:
    python enmap_metadata_extractor.py --image path/to/enmap.tif
    python enmap_metadata_extractor.py --image path/to/enmap.tif --output metadata.json
    python enmap_metadata_extractor.py --batch path/to/enmap_directory/
"""

import os
import sys
import re
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.warp import transform_bounds
from pystac_client import Client


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
# ENMAP METADATA EXTRACTOR CLASS
# ============================================================================
class EnmapMetadataExtractor:
    """
    Extract and enrich EnMAP imagery metadata using DLR STAC API.
    """

    STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
    COLLECTION_ID = "ENMAP_HSI_L2A"

    def __init__(self):
        """Initialize metadata extractor."""
        self.catalog = None

    def connect_to_stac(self):
        """Connect to STAC catalog."""
        if self.catalog is None:
            logger.info(f"Connecting to STAC catalog: {self.STAC_URL}")
            try:
                self.catalog = Client.open(self.STAC_URL)
                logger.info("Connected to STAC catalog")
            except Exception as e:
                logger.error(f"Failed to connect to STAC: {e}")
                raise

    def extract_metadata_from_filename(self, filename: str) -> Tuple[Optional[datetime], Optional[str]]:
        """
        Extract acquisition date and tile number from EnMAP filename.
        """
        pattern = r'_(\d{8})T\d{6}Z_(\d{3})_'
        match = re.search(pattern, filename)

        if match:
            date_str = match.group(1)
            tile_number = match.group(2)

            try:
                acquisition_date = datetime.strptime(date_str, '%Y%m%d')
                logger.debug(f"Extracted date: {acquisition_date}, tile: {tile_number}")
                return acquisition_date, tile_number
            except ValueError as e:
                logger.error(f"Failed to parse date: {e}")
                return None, None

        logger.warning(f"Could not parse filename: {filename}")
        return None, None

    def get_image_bbox(self, image_path: str) -> List[float]:
        """
        Extract bounding box from raster in WGS84 coordinates.
        """
        logger.debug(f"Extracting bbox from: {os.path.basename(image_path)}")

        try:
            with rasterio.open(image_path) as src:
                bounds = src.bounds

                if src.crs != 'EPSG:4326':
                    logger.debug(f"Transforming bbox from {src.crs} to EPSG:4326")
                    bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)

                bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
                logger.debug(f"Bbox: {bbox}")
                return bbox

        except Exception as e:
            logger.error(f"Failed to extract bbox: {e}")
            raise

    def find_matching_stac_item(
        self,
        bbox: List[float],
        acquisition_date: datetime,
        tile_number: str
    ):
        """
        Find exact matching STAC item by date, bbox, and tile number.
        """
        if self.catalog is None:
            self.connect_to_stac()

        date_str = acquisition_date.strftime('%Y-%m-%d')
        logger.info(f"Searching for tile {tile_number} on {date_str}")

        try:
            search = self.catalog.search(
                collections=[self.COLLECTION_ID],
                bbox=bbox,
                datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z"
            )

            items = list(search.items())
            logger.info(f"Found {len(items)} candidate items")

            tile_pattern = r'_(\d{3})_V\d+'
            for item in items:
                tile_match = re.search(tile_pattern, item.id)
                if tile_match and tile_match.group(1) == tile_number:
                    logger.info(f"Found matching item: {item.id}")
                    return item

            logger.warning(f"No item found with tile number {tile_number}")
            return None

        except Exception as e:
            logger.error(f"STAC search failed: {e}")
            raise

    @staticmethod
    def normalize_band_metadata(bands: List[Dict]) -> List[Dict]:
        """
        Normalize band metadata to remove 'eo:' prefix inconsistencies.
        """
        normalized_bands = []

        for band in bands:
            normalized_band = {}
            for key, value in band.items():
                clean_key = key.replace('eo:', '') if key.startswith('eo:') else key
                normalized_band[clean_key] = value
            normalized_bands.append(normalized_band)

        return normalized_bands

    def extract_band_metadata(self, item) -> List[Dict]:
        """
        Extract eo:bands metadata from STAC item.
        """
        logger.debug("Extracting band metadata from STAC item")

        item_dict = item.to_dict()

        if 'properties' in item_dict and 'eo:bands' in item_dict['properties']:
            logger.debug("Found eo:bands in properties")
            bands = item_dict['properties']['eo:bands']
            return self.normalize_band_metadata(bands)

        if 'assets' in item_dict:
            for asset_key, asset_data in item_dict['assets'].items():
                if 'eo:bands' in asset_data:
                    logger.debug(f"Found eo:bands in asset: {asset_key}")
                    bands = asset_data['eo:bands']
                    return self.normalize_band_metadata(bands)

        logger.warning("No eo:bands metadata found in STAC item")
        return []

    def create_stac_item(
        self,
        image_path: str,
        bbox: List[float],
        acquisition_date: datetime,
        band_metadata: List[Dict]
    ) -> Dict:
        """
        Create enriched STAC item with band metadata.
        """
        filename = os.path.basename(image_path)
        item_id = os.path.splitext(filename)[0]

        stac_item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": item_id,
            "bbox": bbox,
            "properties": {
                "datetime": acquisition_date.isoformat(),
                "platform": "enmap",
                "eo:bands": band_metadata
            },
            "assets": {
                "data": {
                    "href": os.path.abspath(image_path),
                    "type": "image/tiff; application=geotiff",
                    "eo:bands": band_metadata
                }
            }
        }

        return stac_item

    def process_image(
        self,
        image_path: str,
        output_json_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Main processing function.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {os.path.basename(image_path)}")
        logger.info(f"{'='*60}\n")

        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return None

        filename = os.path.basename(image_path)
        acquisition_date, tile_number = self.extract_metadata_from_filename(filename)

        if not acquisition_date or not tile_number:
            logger.error("Could not extract date and tile from filename")
            return None

        logger.info(f"Date: {acquisition_date.strftime('%Y-%m-%d')}")
        logger.info(f"Tile: {tile_number}\n")

        try:
            bbox = self.get_image_bbox(image_path)
            logger.info(f"Bbox: {bbox}\n")
        except Exception as e:
            logger.error(f"Failed to extract bbox: {e}")
            return None

        self.connect_to_stac()

        item = self.find_matching_stac_item(bbox, acquisition_date, tile_number)

        if not item:
            logger.error("No matching STAC item found")
            return None

        logger.info("\nExtracting band metadata...")
        band_metadata = self.extract_band_metadata(item)

        if not band_metadata:
            logger.warning("No band metadata found")
            return None

        logger.info(f"Found {len(band_metadata)} bands\n")

        logger.info("Sample bands:")
        for i, band in enumerate(band_metadata[:3]):
            logger.info(f"  Band {i+1}: {band.get('name', 'N/A')}")
            logger.info(f"    Center wavelength: {band.get('center_wavelength', 'N/A')} nm")
            logger.info(f"    FWHM: {band.get('full_width_half_max', 'N/A')} nm")

        if len(band_metadata) > 3:
            logger.info(f"  ... and {len(band_metadata) - 3} more bands")

        stac_item = self.create_stac_item(
            image_path,
            bbox,
            acquisition_date,
            band_metadata
        )

        if output_json_path is None:
            output_json_path = os.path.splitext(image_path)[0] + '_metadata.json'

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(stac_item, f, indent=2, ensure_ascii=False)

            logger.info(f"\nSaved metadata to: {output_json_path}")
            logger.info("Done!\n")

        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return None

        return stac_item

    def process_directory(
        self,
        directory: str
    ) -> List[Dict]:
        """
        Process all EnMAP images in a directory.
        """
        import glob

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING: {directory}")
        logger.info(f"{'='*60}\n")

        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return []

        search_pattern = os.path.join(directory, "*.TIF")
        image_files = glob.glob(search_pattern)

        if not image_files:
            logger.warning("No .TIF files found in directory")
            return []

        logger.info(f"Found {len(image_files)} images to process\n")

        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")

            try:
                result = self.process_image(image_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue

        logger.info(f"\n{'='*60}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {len(results)}/{len(image_files)}")

        if len(results) < len(image_files):
            logger.warning(f"Failed: {len(image_files) - len(results)}")

        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='EnMAP Metadata Extraction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python enmap_metadata_extractor.py --image /path/to/enmap.tif

  # Process with custom output path
  python enmap_metadata_extractor.py --image /path/to/enmap.tif --output metadata.json

  # Batch process directory
  python enmap_metadata_extractor.py --batch /path/to/enmap_directory/
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to single EnMAP image file'
    )
    input_group.add_argument(
        '--batch',
        type=str,
        help='Directory containing EnMAP images (batch processing)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON path (single image mode only)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all logging except errors'
    )

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    extractor = EnmapMetadataExtractor()

    try:
        if args.image:
            result = extractor.process_image(args.image, args.output)
            sys.exit(0 if result else 1)

        elif args.batch:
            results = extractor.process_directory(args.batch)
            sys.exit(0 if results else 1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
