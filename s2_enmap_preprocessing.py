"""
Sentinel-2 Data Download and EnMAP Alignment Pipeline

This script provides a complete pipeline for:
1. Downloading Sentinel-2 data from OpenEO
2. Processing and splitting NetCDF files into per-date GeoTIFFs
3. Aligning Sentinel-2 imagery with downsampled EnMAP grids

Usage:
    python s2_enmap_preprocessing.py --mode download --config config.json
    python s2_enmap_preprocessing.py --mode align --s2-image path/to/s2.tif --enmap-image path/to/enmap.tif
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import openeo
import xarray as xr
import numpy as np
import geopandas as gpd
import rasterio
from openeo.rest.connection import OpenEoApiError
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine, rowcol


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
# SENTINEL-2 DATA DOWNLOADER CLASS
# ============================================================================
class S2DataDownloader:
    """
    Download and process Sentinel-2 data from OpenEO platform.
    
    Attributes:
        aoi_geojson_path (str): Path to GeoJSON file with Area of Interest
        start_date (str): Start date (YYYY-MM-DD format)
        end_date (str): End date (YYYY-MM-DD format)
        max_cloud_cover (int): Maximum cloud coverage percentage
        output_dir (str): Output directory for downloaded files
        download_format (str): Output format (NetCDF, GTiff, CogGeoTIFF)
        native_10m_only (bool): Download only native 10m bands
    """
    
    def __init__(
        self,
        aoi_geojson_path: str,
        start_date: str,
        end_date: str,
        max_cloud_cover: int = 80,
        output_dir: str = "./output",
        download_format: str = "GTiff",
        native_10m_only: bool = True
    ):
        """Initialize S2 data downloader with configuration parameters."""
        self.aoi_geojson_path = aoi_geojson_path
        self.start_date = start_date
        self.end_date = end_date
        self.max_cloud_cover = max_cloud_cover
        self.output_dir = output_dir
        self.download_format = download_format
        self.native_10m_only = native_10m_only
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load AOI
        self.aoi_geometry = self._load_aoi(aoi_geojson_path)
        
        # Connect to OpenEO
        self.session = self._connect_to_openeo()
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if not os.path.exists(self.aoi_geojson_path):
            raise FileNotFoundError(f"AOI GeoJSON file not found: {self.aoi_geojson_path}")
        
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
        
        if self.download_format not in ["NetCDF", "GTiff", "CogGeoTIFF"]:
            raise ValueError(f"Unsupported format: {self.download_format}")
    
    def _load_aoi(self, geojson_path: str) -> Dict:
        """
        Load Area of Interest from GeoJSON file.
        
        Args:
            geojson_path: Path to GeoJSON file
            
        Returns:
            Dictionary with spatial_extent and geometry
        """
        logger.info(f"Loading AOI from: {geojson_path}")
        
        # Read GeoJSON with geopandas
        gdf = gpd.read_file(geojson_path)
        
        # Check and set CRS
        if gdf.crs is None:
            logger.warning("No CRS defined in GeoJSON, assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")
        
        # Reproject to WGS84 if needed
        if gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting AOI from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        else:
            logger.info("AOI already in EPSG:4326")
        
        # Get geometry (union if multiple features)
        if len(gdf) > 1:
            logger.info(f"Multiple features found ({len(gdf)}), using union")
            geom = gdf.unary_union
        else:
            geom = gdf.geometry.iloc[0]
        
        # Extract bounds
        bounds = geom.bounds
        spatial_extent = {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3]
        }
        
        # Convert to GeoJSON dict
        geometry = json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry']
        
        logger.info(f"AOI spatial extent: {spatial_extent}")
        
        return {
            "spatial_extent": spatial_extent,
            "geometry": geometry
        }
    
    def _connect_to_openeo(self):
        """Connect to OpenEO backend using OIDC authentication."""
        logger.info("Connecting to OpenEO (CDSE)...")
        
        try:
            session = openeo.connect("https://openeo.dataspace.copernicus.eu")
            session.authenticate_oidc()
            logger.info("Successfully connected to OpenEO")
            return session
        except Exception as e:
            logger.error(f"Failed to connect to OpenEO: {e}")
            raise
    
    def _get_time_coord(self, ds: xr.Dataset) -> Tuple[str, xr.DataArray]:
        """
        Find and return time coordinate from dataset.
        
        Args:
            ds: xarray Dataset
            
        Returns:
            Tuple of (time_dimension_name, time_coordinate)
        """
        for name in ["t", "time", "date"]:
            if name in ds.coords:
                return name, ds.coords[name]
        raise ValueError("No time coordinate found in dataset")
    
    def download_data(self) -> Optional[str]:
        """
        Download Sentinel-2 data from OpenEO.
        
        Returns:
            Path to downloaded file, or None if no data available
        """
        collection = "SENTINEL2_L2A"
        
        # Select bands based on configuration
        if self.native_10m_only:
            bands = ["B02", "B03", "B04", "B08"]
            logger.info("NATIVE 10m MODE: Downloading only B02, B03, B04, B08")
            logger.info("   This prevents resampling and preserves data quality")
        else:
            bands = [
                "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                "B8A", "B09", "B11", "B12", "SCL", "AOT", "WVP", "SNW"
            ]
            logger.warning("Downloading mixed-resolution bands")
            logger.warning("   OpenEO will resample, potentially degrading quality")
        
        temporal_extent = [self.start_date, self.end_date]
        
        # Set file extension
        format_extensions = {
            "NetCDF": ".nc",
            "GTiff": ".tif",
            "CogGeoTIFF": ".tif"
        }
        file_ext = format_extensions.get(self.download_format, ".tif")
        
        suffix = "_NATIVE10m" if self.native_10m_only else ""
        output_file = os.path.join(
            self.output_dir,
            f"S2_data{suffix}_{self.start_date}_{self.end_date}{file_ext}"
        )
        
        logger.info(f"\nDownloading Sentinel-2 data:")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")
        logger.info(f"  Max cloud cover: {self.max_cloud_cover}%")
        logger.info(f"  Bands: {bands}")
        logger.info(f"  Format: {self.download_format}")
        logger.info(f"  Spatial extent: {self.aoi_geometry['spatial_extent']}")
        
        try:
            # Load collection
            sat_data = self.session.load_collection(
                collection,
                temporal_extent=temporal_extent,
                bands=bands,
                spatial_extent=self.aoi_geometry['spatial_extent'],
                max_cloud_cover=self.max_cloud_cover
            )
            
            # Mask to polygon
            sat_data_clipped = sat_data.mask_polygon(self.aoi_geometry['geometry'])
            
            # Save result
            sat_data_output = sat_data_clipped.save_result(format=self.download_format)
            
            # Download
            logger.info(f" Downloading to: {output_file}")
            sat_data_output.download(output_file)
            
            logger.info(f" Successfully downloaded to {output_file}")
            
            # Verify quality
            if self.native_10m_only:
                self._verify_data_quality(output_file)
            
            return output_file
            
        except OpenEoApiError as e:
            if "NoDataAvailable" in str(e):
                logger.warning(f"No data available for given parameters")
                return None
            else:
                raise e
    
    def _verify_data_quality(self, file_path: str):
        """
        Verify data quality by checking unique value ratio.
        
        Args:
            file_path: Path to data file to verify
        """
        logger.info("\n Verifying data quality...")
        
        try:
            if self.download_format == "NetCDF":
                ds = xr.open_dataset(file_path)
                first_band = list(ds.data_vars.keys())[0]
                data = ds[first_band].values
            else:
                import rioxarray
                ds = rioxarray.open_rasterio(file_path)
                data = ds[0].values
            
            # Calculate unique ratio
            valid_data = data[data != -32768]
            if len(valid_data) > 0:
                unique_ratio = len(np.unique(valid_data)) / len(valid_data)
                logger.info(f"  Unique value ratio: {unique_ratio:.4f}")
                
                if unique_ratio > 0.5:
                    logger.info(f" EXCELLENT quality (>50% unique values)")
                elif unique_ratio > 0.3:
                    logger.info(f" Good quality (>30% unique values)")
                elif unique_ratio > 0.1:
                    logger.warning(f" Moderate quality ({unique_ratio*100:.1f}% unique)")
                else:
                    logger.error(f" POOR quality ({unique_ratio*100:.1f}% unique)")
                    logger.error(f"  Data appears heavily resampled/degraded!")
            else:
                logger.warning(f" No valid data found")
                
        except Exception as e:
            logger.warning(f" Could not verify quality: {e}")
    
    def split_netcdf_per_date(
        self,
        nc_path: str,
        output_dir: Optional[str] = None,
        driver: str = "COG"
    ) -> List[str]:
        """
        Split multi-temporal NetCDF into one GeoTIFF per date.
        
        Args:
            nc_path: Path to NetCDF file
            output_dir: Output directory (defaults to self.output_dir/per_date_tiffs)
            driver: GDAL driver (COG or GTiff)
            
        Returns:
            List of output file paths
        """
        import rioxarray
        
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "per_date_tiffs")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\n Splitting NetCDF into per-date GeoTIFFs:")
        logger.info(f"  Input:  {nc_path}")
        logger.info(f"  Output: {output_dir}")
        
        # Open dataset
        ds = xr.open_dataset(nc_path)
        
        # Ensure CRS is set
        if not hasattr(ds, "rio") or ds.rio.crs is None:
            crs_wkt = ds["crs"].attrs.get("crs_wkt")
            if crs_wkt is None:
                raise ValueError("No CRS info found in dataset")
            ds = ds.rio.write_crs(crs_wkt)
        
        # Get time coordinate
        time_name, time_coord = self._get_time_coord(ds)
        logger.info(f"  Time coordinate: {time_name} ({len(time_coord)} steps)")
        
        out_files = []
        for t in time_coord.values:
            # Convert to date string
            date_str = np.datetime_as_string(t, unit="D")
            logger.info(f" Exporting date {date_str}...")
            
            # Select single time slice
            ds_day = ds.sel({time_name: t})
            
            # Drop singleton dimensions
            ds_day = ds_day.squeeze(drop=True)
            
            # Create output path
            out_path = os.path.join(output_dir, f"S2_NATIVE10m_{date_str}.tif")
            
            # Write to GeoTIFF
            ds_day.rio.to_raster(out_path, driver=driver)
            out_files.append(out_path)
            logger.info(f" {out_path}")
        
        logger.info(f"\n Created {len(out_files)} GeoTIFFs")
        return out_files
    
    def format_dataset(self, input_file: str) -> Tuple[xr.Dataset, str]:
        """
        Format dataset by reprojecting to WGS84 if needed.
        
        Args:
            input_file: Path to input file
            
        Returns:
            Tuple of (formatted_dataset, output_file_path)
        """
        logger.info(f"\n Formatting dataset...")
        logger.info(f" Note: Reprojection may degrade quality")
        
        if self.download_format == "NetCDF":
            ds = xr.open_dataset(input_file)
            crs_wkt = ds["crs"].attrs.get("crs_wkt")
            
            logger.info(f"Available bands: {list(ds.data_vars.keys())}")
            
            # Extract EPSG code
            start = crs_wkt.rfind('EPSG","') + len('EPSG","')
            end = crs_wkt.find('"]', start)
            epsg_code = crs_wkt[start:end]
            
            logger.info(f"Original CRS: EPSG:{epsg_code}")
            
            # Reproject only if needed
            if epsg_code == "4326":
                logger.info("Already in EPSG:4326, skipping reprojection")
                formatted_ds = ds.rio.write_crs(epsg_code, inplace=True)
            else:
                logger.info(f"Reprojecting from EPSG:{epsg_code} to EPSG:4326")
                formatted_ds = ds.rio.write_crs(
                    epsg_code, inplace=True
                ).rio.reproject("epsg:4326")
            
            # Save
            output_file = os.path.join(
                self.output_dir,
                f"S2_formatted_{self.start_date}_{self.end_date}.nc"
            )
            formatted_ds.to_netcdf(output_file)
            
        elif self.download_format in ["GTiff", "CogGeoTIFF"]:
            import rioxarray
            ds = rioxarray.open_rasterio(input_file)
            
            logger.info(f"Bands: {ds.sizes.get('band', 'N/A')}")
            logger.info(f"Original CRS: {ds.rio.crs}")
            logger.info(f"Shape: {ds.shape}")
            
            # Reproject if needed
            if ds.rio.crs.to_epsg() == 4326:
                logger.info("Already in EPSG:4326, skipping reprojection")
                formatted_ds = ds
            else:
                logger.info(f"Reprojecting to EPSG:4326")
                formatted_ds = ds.rio.reproject("EPSG:4326")
            
            # Save as GeoTIFF
            output_file = os.path.join(
                self.output_dir,
                f"S2_formatted_{self.start_date}_{self.end_date}.tif"
            )
            formatted_ds.rio.to_raster(output_file, driver="COG")
            
        else:
            raise ValueError(f"Unsupported format: {self.download_format}")
        
        logger.info(f" Formatted dataset saved to: {output_file}")
        return formatted_ds, output_file


# ============================================================================
# SENTINEL-2 / ENMAP ALIGNMENT
# ============================================================================
class S2EnmapAligner:
    """
    Align Sentinel-2 imagery with downsampled EnMAP grids.
    
    This class handles reprojection of S2 data to match EnMAP's coordinate
    system, accounting for the downsampling from 30m to 60m.
    """
    
    def __init__(self, enmap_tif: str, s2_tif: str, output_suffix: str = "_UL"):
        """
        Initialize aligner.
        
        Args:
            enmap_tif: Path to EnMAP GeoTIFF
            s2_tif: Path to Sentinel-2 GeoTIFF
            output_suffix: Suffix for output filename
        """
        self.enmap_tif = enmap_tif
        self.s2_tif = s2_tif
        self.output_suffix = output_suffix
        
        # Validate inputs
        if not os.path.exists(enmap_tif):
            raise FileNotFoundError(f"EnMAP file not found: {enmap_tif}")
        if not os.path.exists(s2_tif):
            raise FileNotFoundError(f"S2 file not found: {s2_tif}")
        
        # Create output path
        self.s2_tif_aligned = os.path.splitext(s2_tif)[0] + f"{output_suffix}.tif"
    
    def align(self) -> str:
        """
        Perform alignment of S2 to EnMAP grid.
        
        Returns:
            Path to aligned S2 GeoTIFF
        """
        logger.info("\n" + "="*60)
        logger.info("SENTINEL-2 / ENMAP ALIGNMENT")
        logger.info("="*60)
        
        # Step 1: Read EnMAP and calculate 60m transform
        logger.info("\n Step 1: Reading EnMAP and creating 60m grid")
        with rasterio.open(self.enmap_tif) as src_enmap:
            enmap_crs = src_enmap.crs
            enmap_transform_30m = src_enmap.transform
            enmap_height_30m = src_enmap.height
            enmap_width_30m = src_enmap.width
            
            logger.info(f"  Original EnMAP CRS: {enmap_crs}")
            logger.info(f"  Original EnMAP transform (30m): {enmap_transform_30m}")
            logger.info(f"  Original EnMAP size: {enmap_width_30m} x {enmap_height_30m}")
            
            # Downsample to 60m
            enmap_width_60m = enmap_width_30m // 2
            enmap_height_60m = enmap_height_30m // 2
            
            # New 60m transform
            enmap_transform_60m = Affine(
                enmap_transform_30m.a * 2,  # 30 * 2 = 60
                enmap_transform_30m.b,
                enmap_transform_30m.c,
                enmap_transform_30m.d,
                enmap_transform_30m.e * 2,  # -30 * 2 = -60
                enmap_transform_30m.f,
            )
            
            logger.info(f"  Downsampled EnMAP transform (60m): {enmap_transform_60m}")
            logger.info(f"  Downsampled EnMAP size: {enmap_width_60m} x {enmap_height_60m}")
        
        # Step 2: Build 10m transform aligned with 60m EnMAP
        logger.info("\n Step 2: Creating aligned 10m S2 grid")
        scale_factor = 6  # 60m / 10m = 6
        
        new_transform = Affine(
            enmap_transform_60m.a / scale_factor,  # 60 / 6 = 10
            enmap_transform_60m.b,
            enmap_transform_60m.c,
            enmap_transform_60m.d,
            enmap_transform_60m.e / scale_factor,  # -60 / 6 = -10
            enmap_transform_60m.f,
        )
        
        new_width = enmap_width_60m * scale_factor
        new_height = enmap_height_60m * scale_factor
        
        logger.info(f"  New S2 transform (10m, aligned): {new_transform}")
        logger.info(f"  New S2 size: {new_width} x {new_height}")
        logger.info(f"  Note: Every 6x6 S2 pixels = one 60m EnMAP pixel")
        
        # Step 3: Reproject S2 to aligned grid
        logger.info("\n Step 3: Reprojecting S2 to aligned grid")
        with rasterio.open(self.s2_tif) as src_s2:
            s2_profile = src_s2.profile.copy()
            
            dst_profile = s2_profile.copy()
            dst_profile.update(
                crs=enmap_crs,
                transform=new_transform,
                width=new_width,
                height=new_height,
                dtype="float32",
            )
            
            # Allocate output array
            s2_reproj = np.zeros((src_s2.count, new_height, new_width), dtype=np.float32)
            
            # Reproject each band
            for b in range(src_s2.count):
                logger.info(f"  Reprojecting band {b+1}/{src_s2.count}...")
                reproject(
                    source=rasterio.band(src_s2, b + 1),
                    destination=s2_reproj[b],
                    src_transform=src_s2.transform,
                    src_crs=src_s2.crs,
                    dst_transform=new_transform,
                    dst_crs=enmap_crs,
                    resampling=Resampling.bilinear,
                )
            
            # Write to disk
            with rasterio.open(self.s2_tif_aligned, "w", **dst_profile) as dst:
                dst.write(s2_reproj)
        
        logger.info(f"\n Alignment complete!")
        logger.info(f"   Output: {self.s2_tif_aligned}")
        
        return self.s2_tif_aligned


# ============================================================================
# CONFIGURATION HANDLING
# ============================================================================
def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Dictionary with configuration parameters
    """
    logger.info(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def create_sample_config(output_path: str = "s2_config.json"):
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path for output config file
    """
    sample_config = {
        "aoi_geojson": "path/to/your/aoi.geojson",
        "start_date": "2025-12-04",
        "end_date": "2025-12-14",
        "max_cloud_cover": 10,
        "output_dir": "./output",
        "download_format": "NetCDF",
        "native_10m_only": True,
        "split_per_date": True,
        "enmap_tif": "path/to/enmap.tif",
        "s2_images": [
            "path/to/s2_image1.tif",
            "path/to/s2_image2.tif"
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    logger.info(f" Sample config created: {output_path}")
    logger.info("  Edit this file with your parameters before running")


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================
def run_download_pipeline(config: Dict):
    """
    Run the Sentinel-2 download pipeline.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("SENTINEL-2 DOWNLOAD PIPELINE")
    logger.info("="*60 + "\n")
    
    # Initialize downloader
    downloader = S2DataDownloader(
        aoi_geojson_path=config['aoi_geojson'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        max_cloud_cover=config.get('max_cloud_cover', 80),
        output_dir=config.get('output_dir', './output'),
        download_format=config.get('download_format', 'NetCDF'),
        native_10m_only=config.get('native_10m_only', True)
    )
    
    # Download data
    raw_file = downloader.download_data()
    
    if raw_file is None:
        logger.warning("No data available for specified parameters")
        return
    
    # Split into per-date files if requested
    if config.get('split_per_date', False) and config.get('download_format') == 'NetCDF':
        per_date_files = downloader.split_netcdf_per_date(raw_file)
        logger.info(f"\n Pipeline complete: {len(per_date_files)} per-date files created")
    else:
        logger.info(f"\n Pipeline complete: {raw_file}")


def run_alignment_pipeline(config: Dict):
    """
    Run the S2/EnMAP alignment pipeline.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("S2/ENMAP ALIGNMENT PIPELINE")
    logger.info("="*60 + "\n")
    
    enmap_tif = config.get('enmap_tif')
    s2_images = config.get('s2_images', [])
    
    if not enmap_tif:
        raise ValueError("enmap_tif must be specified in config")
    
    if not s2_images:
        raise ValueError("s2_images list must be specified in config")
    
    # Align each S2 image
    aligned_files = []
    for s2_tif in s2_images:
        logger.info(f"\nProcessing: {s2_tif}")
        aligner = S2EnmapAligner(enmap_tif, s2_tif)
        aligned_file = aligner.align()
        aligned_files.append(aligned_file)
    
    logger.info(f"\n Alignment pipeline complete!")
    logger.info(f"   Aligned {len(aligned_files)} images")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Sentinel-2 Download and EnMAP Alignment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample config file
  python s2_enmap_preprocessing.py --create-config
  
  # Download S2 data using config
  python s2_enmap_preprocessing.py --mode download --config config.json
  
  # Align S2 images with EnMAP
  python s2_enmap_preprocessing.py --mode align --config config.json
  
  # Direct alignment without config
  python s2_enmap_preprocessing.py --mode align --s2-image s2.tif --enmap-image enmap.tif
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['download', 'align'],
        help='Pipeline mode: download or align'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a sample configuration file'
    )
    
    # Direct alignment arguments (alternative to config)
    parser.add_argument(
        '--s2-image',
        type=str,
        help='Path to S2 image (for direct alignment mode)'
    )
    
    parser.add_argument(
        '--enmap-image',
        type=str,
        help='Path to EnMAP image (for direct alignment mode)'
    )
    
    args = parser.parse_args()
    
    # Create sample config
    if args.create_config:
        create_sample_config()
        return
    
    # Validate mode is specified
    if not args.mode:
        parser.error("--mode is required (unless using --create-config)")
    
    # Run appropriate pipeline
    try:
        if args.mode == 'download':
            if not args.config:
                parser.error("--config is required for download mode")
            config = load_config(args.config)
            run_download_pipeline(config)
            
        elif args.mode == 'align':
            # Check if direct arguments or config
            if args.s2_image and args.enmap_image:
                # Direct mode
                aligner = S2EnmapAligner(args.enmap_image, args.s2_image)
                aligner.align()
            elif args.config:
                # Config mode
                config = load_config(args.config)
                run_alignment_pipeline(config)
            else:
                parser.error("Alignment mode requires either --config or (--s2-image and --enmap-image)")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
