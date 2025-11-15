"""
Vectorization utilities to convert raster masks to shapefiles.
Preserves CRS and geotransform information.
"""

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine
from pathlib import Path
from typing import Optional, Tuple, List
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely import simplify
import warnings
import fiona


def vectorize_mask(
    mask: np.ndarray,
    transform: Affine,
    crs: Optional[str] = None,
    simplify_tolerance: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Convert binary raster mask to vector polygons.
    
    Args:
        mask: Binary mask array (H, W)
        transform: Rasterio affine transform
        crs: Coordinate reference system
        simplify_tolerance: Tolerance for simplifying geometries (0 = no simplification)
        
    Returns:
        GeoDataFrame with polygons
    """
    # Ensure binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Extract shapes (polygons) from raster
    shapes_generator = features.shapes(
        mask_binary,
        mask=mask_binary,
        transform=transform,
        connectivity=8
    )
    
    # Convert to shapely geometries
    geometries = []
    values = []
    
    for geom_dict, value in shapes_generator:
        if value == 1:  # Only keep change areas
            geom = shape(geom_dict)
            
            # Simplify if requested
            if simplify_tolerance > 0:
                geom = simplify(geom, tolerance=simplify_tolerance)
            
            geometries.append(geom)
            values.append(value)
    
    # Create GeoDataFrame
    if len(geometries) == 0:
        # Empty result
        gdf = gpd.GeoDataFrame(
            {'value': [], 'area': [], 'perimeter': []},
            geometry=[],
            crs=crs
        )
    else:
        gdf = gpd.GeoDataFrame(
            {'value': values},
            geometry=geometries,
            crs=crs
        )
        
        # Calculate area and perimeter
        gdf['area'] = gdf.geometry.area
        gdf['perimeter'] = gdf.geometry.length
    
    return gdf


def vectorize_from_file(
    raster_path: Path,
    simplify_tolerance: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Vectorize mask from raster file.
    
    Args:
        raster_path: Path to raster file
        simplify_tolerance: Tolerance for simplifying geometries
        
    Returns:
        GeoDataFrame with polygons
    """
    with rasterio.open(raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    
    return vectorize_mask(mask, transform, crs, simplify_tolerance)


def filter_polygons_by_area(
    gdf: gpd.GeoDataFrame,
    min_area: float = 0.0,
    max_area: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Filter polygons by area.
    
    Args:
        gdf: Input GeoDataFrame
        min_area: Minimum area (in CRS units, typically square meters)
        max_area: Maximum area (None = no maximum)
        
    Returns:
        Filtered GeoDataFrame
    """
    # Filter by minimum area
    filtered = gdf[gdf['area'] >= min_area].copy()
    
    # Filter by maximum area if specified
    if max_area is not None:
        filtered = filtered[filtered['area'] <= max_area].copy()
    
    # Reset index
    filtered = filtered.reset_index(drop=True)
    
    return filtered


def buffer_geometries(
    gdf: gpd.GeoDataFrame,
    buffer_distance: float
) -> gpd.GeoDataFrame:
    """
    Apply buffer to all geometries.
    
    Args:
        gdf: Input GeoDataFrame
        buffer_distance: Buffer distance (in CRS units)
        
    Returns:
        GeoDataFrame with buffered geometries
    """
    gdf_buffered = gdf.copy()
    gdf_buffered.geometry = gdf_buffered.geometry.buffer(buffer_distance)
    
    # Recalculate area and perimeter
    gdf_buffered['area'] = gdf_buffered.geometry.area
    gdf_buffered['perimeter'] = gdf_buffered.geometry.length
    
    return gdf_buffered


def export_to_shapefile(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    driver: str = 'ESRI Shapefile'
):
    """
    Export GeoDataFrame to shapefile.
    
    Args:
        gdf: GeoDataFrame to export
        output_path: Path to output shapefile
        driver: Output driver
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save only if GeoDataFrame is not empty
    if gdf.empty:
        print(f"[WARN] GeoDataFrame is empty. No shapefile will be written to {output_path}")
        return
    
    # Work around NumPy 2.x GeoPandas export issue by writing via Fiona
    # Ensure geometry type consistency for Shapefile: write MultiPolygon
    geom_type = 'MultiPolygon'
    # Keep a small, safe set of attributes (Shapefile field name <= 10 chars)
    props = {}
    if 'area' in gdf.columns:
        props['area'] = 'float:24.6'
    if 'perimeter' in gdf.columns:
        props['perimeter'] = 'float:24.6'
    schema = {
        'geometry': geom_type,
        'properties': props
    }
    crs = gdf.crs
    output_path = str(output_path)
    written = 0
    with fiona.open(output_path, mode='w', driver=driver, schema=schema, crs=crs) as col:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                geom = MultiPolygon([geom])
            record = {
                'geometry': mapping(geom),
                'properties': {k: float(row[k]) for k in props.keys() if k in row}
            }
            col.write(record)
            written += 1
    print(f"✓ Saved {written} polygons to {output_path}")


def export_to_geojson(
    gdf: gpd.GeoDataFrame,
    output_path: Path
):
    """
    Export GeoDataFrame to GeoJSON.
    
    Args:
        gdf: GeoDataFrame to export
        output_path: Path to output GeoJSON
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write via Fiona to avoid GeoPandas/NumPy 2.x issues
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Standardize to MultiPolygon in the writer loop
    geom_type = 'MultiPolygon'
    props = {}
    if 'area' in gdf.columns:
        props['area'] = 'float:24.6'
    if 'perimeter' in gdf.columns:
        props['perimeter'] = 'float:24.6'
    schema = {
        'geometry': geom_type,
        'properties': props
    }
    written = 0
    with fiona.open(str(output_path), mode='w', driver='GeoJSON', schema=schema, crs=gdf.crs) as col:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                geom = MultiPolygon([geom])
            record = {
                'geometry': mapping(geom),
                'properties': {k: float(row[k]) for k in props.keys() if k in row}
            }
            col.write(record)
            written += 1
    print(f"✓ Saved {written} polygons to {output_path}")


def raster_to_vector_pipeline(
    raster_path: Path,
    output_shapefile: Path,
    simplify_tolerance: float = 0.5,
    min_area: float = 25.0,
    max_area: Optional[float] = None,
    buffer_distance: float = 0.0,
    output_geojson: Optional[Path] = None
) -> gpd.GeoDataFrame:
    """
    Complete pipeline to convert raster mask to vector format.
    
    Args:
        raster_path: Path to input raster
        output_shapefile: Path to output shapefile
        simplify_tolerance: Tolerance for simplifying geometries (meters)
        min_area: Minimum polygon area (square meters)
        max_area: Maximum polygon area
        buffer_distance: Buffer distance (meters)
        output_geojson: Optional path to save GeoJSON
        
    Returns:
        GeoDataFrame with polygons
    """
    print(f"Vectorizing {raster_path.name}...")
    
    # Step 1: Vectorize raster
    gdf = vectorize_from_file(raster_path, simplify_tolerance=simplify_tolerance)
    
    print(f"  Extracted {len(gdf)} polygons")
    
    if len(gdf) == 0:
        print("  ⚠ No change detected")
        # Create empty shapefile
        export_to_shapefile(gdf, output_shapefile)
        return gdf
    
    # Step 2: Filter by area
    if min_area > 0 or max_area is not None:
        gdf_filtered = filter_polygons_by_area(gdf, min_area, max_area)
        print(f"  After area filtering: {len(gdf_filtered)} polygons")
        gdf = gdf_filtered
    
    # Step 3: Apply buffer if requested
    if buffer_distance != 0:
        gdf = buffer_geometries(gdf, buffer_distance)
        print(f"  Applied buffer: {buffer_distance}m")
    
    # Step 4: Export to shapefile
    export_to_shapefile(gdf, output_shapefile)
    
    # Step 5: Export to GeoJSON if requested
    if output_geojson:
        export_to_geojson(gdf, output_geojson)
    
    # Print statistics
    if len(gdf) > 0:
        total_area = gdf['area'].sum()
        mean_area = gdf['area'].mean()
        print(f"  Total change area: {total_area:.2f} m²")
        print(f"  Mean polygon area: {mean_area:.2f} m²")
    
    return gdf


def create_change_polygons_with_attributes(
    raster_path: Path,
    transform: Affine,
    crs: str,
    image1_date: Optional[str] = None,
    image2_date: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Create change polygons with additional attributes.
    
    Args:
        raster_path: Path to raster mask
        transform: Affine transform
        crs: Coordinate reference system
        image1_date: Date of first image
        image2_date: Date of second image
        
    Returns:
        GeoDataFrame with enriched attributes
    """
    # Vectorize
    gdf = vectorize_from_file(raster_path)
    
    if len(gdf) == 0:
        return gdf
    
    # Add attributes
    gdf['change_id'] = range(1, len(gdf) + 1)
    
    if image1_date:
        gdf['date_t1'] = image1_date
    if image2_date:
        gdf['date_t2'] = image2_date
    
    # Calculate centroid coordinates
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y
    
    # Calculate compactness (perimeter² / area)
    gdf['compactness'] = (gdf['perimeter'] ** 2) / gdf['area']
    
    return gdf


def merge_nearby_polygons(
    gdf: gpd.GeoDataFrame,
    distance: float = 10.0
) -> gpd.GeoDataFrame:
    """
    Merge polygons that are within a certain distance of each other.
    
    Args:
        gdf: Input GeoDataFrame
        distance: Maximum distance for merging (meters)
        
    Returns:
        GeoDataFrame with merged polygons
    """
    if len(gdf) == 0:
        return gdf
    
    # Buffer polygons
    buffered = gdf.geometry.buffer(distance)
    
    # Dissolve overlapping buffers
    dissolved = buffered.unary_union
    
    # Create new GeoDataFrame
    if isinstance(dissolved, Polygon):
        geometries = [dissolved]
    elif isinstance(dissolved, MultiPolygon):
        geometries = list(dissolved.geoms)
    else:
        geometries = []
    
    # Create new GeoDataFrame
    gdf_merged = gpd.GeoDataFrame(
        geometry=geometries,
        crs=gdf.crs
    )
    
    # Remove buffer (negative buffer)
    gdf_merged.geometry = gdf_merged.geometry.buffer(-distance)
    
    # Remove empty geometries
    gdf_merged = gdf_merged[~gdf_merged.geometry.is_empty].copy()
    
    # Recalculate attributes
    gdf_merged['area'] = gdf_merged.geometry.area
    gdf_merged['perimeter'] = gdf_merged.geometry.length
    gdf_merged['value'] = 1
    
    gdf_merged = gdf_merged.reset_index(drop=True)
    
    return gdf_merged


def extract_bounds_from_filename(filename: str) -> Optional[Tuple[float, float]]:
    """
    Extract latitude and longitude from filename.
    
    Expected format: Change_Mask_Lat_Long.tif
    Example: Change_Mask_26_70.tif -> (26, 70)
    
    Args:
        filename: Input filename
        
    Returns:
        Tuple of (latitude, longitude) or None
    """
    try:
        # Remove extension
        name = Path(filename).stem
        
        # Split by underscore
        parts = name.split('_')
        
        # Try to find two consecutive numbers
        for i in range(len(parts) - 1):
            try:
                lat = float(parts[i])
                lon = float(parts[i + 1])
                return (lat, lon)
            except ValueError:
                continue
        
        return None
    
    except Exception:
        return None


if __name__ == '__main__':
    # Test vectorization
    print("Vectorization module loaded successfully")
