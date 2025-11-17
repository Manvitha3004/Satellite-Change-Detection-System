"""
Sentinel-2 data loader for .SAFE format products.
Supports both L1C and L2A processing levels with automatic band discovery.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import xml.etree.ElementTree as ET


class Sentinel2Product:
    """
    Handler for Sentinel-2 .SAFE product structure.
    Supports L1C (Top-of-Atmosphere) and L2A (Bottom-of-Atmosphere) products.
    """
    
    # Band mapping for different resolutions
    BANDS_10M = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # Red Edge, NIR narrow, SWIR
    BANDS_60M = ['B01', 'B09', 'B10']  # Coastal aerosol, Water vapour, Cirrus
    
    # Standard band names
    BAND_NAMES = {
        'B01': 'Coastal aerosol',
        'B02': 'Blue',
        'B03': 'Green',
        'B04': 'Red',
        'B05': 'Red Edge 1',
        'B06': 'Red Edge 2',
        'B07': 'Red Edge 3',
        'B08': 'NIR',
        'B8A': 'NIR narrow',
        'B09': 'Water vapour',
        'B10': 'SWIR - Cirrus',
        'B11': 'SWIR 1',
        'B12': 'SWIR 2'
    }
    
    def __init__(self, safe_path: Path):
        """
        Initialize Sentinel-2 product handler.
        
        Args:
            safe_path: Path to .SAFE folder
        """
        self.safe_path = Path(safe_path)
        if not self.safe_path.exists():
            raise FileNotFoundError(f"SAFE folder not found: {safe_path}")
        
        # Detect product level (L1C or L2A)
        self.product_level = self._detect_product_level()
        
        # Find granule folder
        self.granule_path = self._find_granule_path()
        
        # Find band files
        self.band_files = self._find_band_files()
        
    def _detect_product_level(self) -> str:
        """Detect if product is L1C or L2A from folder name."""
        folder_name = self.safe_path.name
        if 'MSIL1C' in folder_name:
            return 'L1C'
        elif 'MSIL2A' in folder_name:
            return 'L2A'
        else:
            raise ValueError(f"Cannot determine product level from: {folder_name}")
    
    def _find_granule_path(self) -> Path:
        """Find the granule folder containing image data."""
        # Handle nested .SAFE structure (common in some distributions)
        # Structure: *.SAFE/*.SAFE/GRANULE or *.SAFE/GRANULE
        
        # First try direct GRANULE
        granule_dir = self.safe_path / 'GRANULE'
        
        # If not found, check for nested .SAFE folder
        if not granule_dir.exists():
            nested_safe = list(self.safe_path.glob('*.SAFE'))
            if nested_safe and len(nested_safe) == 1:
                # Use the nested .SAFE folder as the actual product root
                self.safe_path = nested_safe[0]
                granule_dir = self.safe_path / 'GRANULE'
        
        if not granule_dir.exists():
            raise FileNotFoundError(f"GRANULE folder not found in {self.safe_path}")
        
        # Get first granule (typically only one)
        granules = list(granule_dir.iterdir())
        if not granules:
            raise FileNotFoundError(f"No granules found in {granule_dir}")
        
        return granules[0]
    
    def _find_band_files(self) -> Dict[str, Path]:
        """
        Find all band files in the product.
        
        Returns:
            Dictionary mapping band names (e.g., 'B02') to file paths
        """
        band_files = {}
        
        if self.product_level == 'L1C':
            # L1C structure: GRANULE/*/IMG_DATA/*_Bxx.jp2
            img_data_dir = self.granule_path / 'IMG_DATA'
        else:
            # L2A structure: GRANULE/*/IMG_DATA/R10m, R20m, R60m
            img_data_dir = self.granule_path / 'IMG_DATA'
        
        if not img_data_dir.exists():
            raise FileNotFoundError(f"IMG_DATA folder not found in {self.granule_path}")
        
        # Search for band files (handles both L1C and L2A structures)
        for pattern in ['**/*_B*.jp2', '**/*_B*.tif']:
            for file_path in img_data_dir.glob(pattern):
                # Extract band name (e.g., B02, B8A)
                match = re.search(r'_(B\d{2}A?)(?:_|\.)', file_path.name)
                if match:
                    band_name = match.group(1)
                    band_files[band_name] = file_path
        
        if not band_files:
            raise FileNotFoundError(f"No band files found in {img_data_dir}")
        
        return band_files
    
    def get_band_path(self, band_name: str) -> Optional[Path]:
        """
        Get path to a specific band file.
        
        Args:
            band_name: Band name (e.g., 'B02', 'B03', 'B04')
            
        Returns:
            Path to band file or None if not found
        """
        return self.band_files.get(band_name)
    
    def get_rgb_nir_bands(self) -> Tuple[List[Path], List[str]]:
        """
        Get paths for RGB + NIR bands suitable for change detection.
        Returns Green, Red, NIR to match ResourceSat-2 structure.
        
        Returns:
            Tuple of (band_paths, band_names)
        """
        # For change detection, use: Green (B03), Red (B04), NIR (B08)
        target_bands = ['B03', 'B04', 'B08']
        band_paths = []
        band_names = []
        
        for band in target_bands:
            path = self.get_band_path(band)
            if path is None:
                raise FileNotFoundError(f"Required band {band} not found in product")
            band_paths.append(path)
            band_names.append(self.BAND_NAMES.get(band, band))
        
        return band_paths, band_names
    
    def get_all_10m_bands(self) -> Tuple[List[Path], List[str]]:
        """
        Get all 10m resolution bands.
        
        Returns:
            Tuple of (band_paths, band_names)
        """
        band_paths = []
        band_names = []
        
        for band in self.BANDS_10M:
            path = self.get_band_path(band)
            if path:
                band_paths.append(path)
                band_names.append(self.BAND_NAMES.get(band, band))
        
        return band_paths, band_names
    
    def read_bands(
        self,
        band_names: List[str],
        window: Optional[rasterio.windows.Window] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Read and stack multiple bands.
        
        Args:
            band_names: List of band names to read (e.g., ['B03', 'B04', 'B08'])
            window: Optional window for reading subset
            
        Returns:
            Tuple of (stacked_array, metadata)
        """
        bands = []
        metadata = None
        
        for band_name in band_names:
            band_path = self.get_band_path(band_name)
            if band_path is None:
                raise FileNotFoundError(f"Band {band_name} not found")
            
            with rasterio.open(band_path) as src:
                if window is not None:
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                else:
                    data = src.read(1)
                    transform = src.transform
                
                bands.append(data)
                
                # Store metadata from first band
                if metadata is None:
                    metadata = {
                        'crs': src.crs,
                        'transform': transform,
                        'shape': data.shape,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata,
                        'bounds': src.bounds if window is None else rasterio.windows.bounds(window, src.transform),
                        'resolution': src.res
                    }
        
        # Stack bands (C, H, W)
        stacked = np.stack(bands, axis=0)
        
        return stacked, metadata
    
    def get_metadata(self) -> Dict:
        """
        Extract product metadata from XML files.
        
        Returns:
            Dictionary with product metadata
        """
        # Read main metadata file
        if self.product_level == 'L1C':
            metadata_file = self.safe_path / 'MTD_MSIL1C.xml'
        else:
            metadata_file = self.safe_path / 'MTD_MSIL2A.xml'
        
        if not metadata_file.exists():
            return {'product_level': self.product_level}
        
        try:
            tree = ET.parse(metadata_file)
            root = tree.getroot()
            
            # Extract basic info
            metadata = {
                'product_level': self.product_level,
                'spacecraft': self._extract_xml_text(root, './/SPACECRAFT_NAME'),
                'datatake_sensing_start': self._extract_xml_text(root, './/DATATAKE_SENSING_START'),
                'cloud_coverage': self._extract_xml_text(root, './/Cloud_Coverage_Assessment'),
            }
            
            return metadata
        except Exception:
            return {'product_level': self.product_level}
    
    def _extract_xml_text(self, root, xpath: str) -> Optional[str]:
        """Helper to extract text from XML."""
        element = root.find(xpath)
        return element.text if element is not None else None
    
    def __repr__(self):
        return (f"Sentinel2Product(level={self.product_level}, "
                f"bands={len(self.band_files)}, path={self.safe_path.name})")


def detect_sentinel2_product(directory: Path) -> Optional[Sentinel2Product]:
    """
    Detect if a directory contains a Sentinel-2 .SAFE product.
    
    Args:
        directory: Directory to check
        
    Returns:
        Sentinel2Product instance if valid, None otherwise
    """
    directory = Path(directory)
    
    # Check if directory itself is a .SAFE folder
    if directory.name.endswith('.SAFE') and directory.is_dir():
        try:
            return Sentinel2Product(directory)
        except Exception:
            return None
    
    # Check for .SAFE folders inside directory
    safe_folders = list(directory.glob('*.SAFE'))
    if safe_folders:
        try:
            return Sentinel2Product(safe_folders[0])
        except Exception:
            return None
    
    return None


def read_sentinel2_bands(
    safe_path: Path,
    band_names: Optional[List[str]] = None,
    window: Optional[rasterio.windows.Window] = None,
    use_rgb_nir: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to read Sentinel-2 bands.
    
    Args:
        safe_path: Path to .SAFE folder
        band_names: Specific bands to read (None = auto-select)
        window: Optional window for reading subset
        use_rgb_nir: If True and band_names is None, read Green/Red/NIR (matches LISS-4)
        
    Returns:
        Tuple of (stacked_array, metadata)
    """
    product = Sentinel2Product(safe_path)
    
    if band_names is None:
        if use_rgb_nir:
            # Get Green, Red, NIR to match ResourceSat-2 structure
            band_paths, _ = product.get_rgb_nir_bands()
            band_names = ['B03', 'B04', 'B08']
        else:
            # Get all 10m bands
            band_paths, _ = product.get_all_10m_bands()
            band_names = product.BANDS_10M
    
    return product.read_bands(band_names, window)
