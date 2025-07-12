from collections.abc import Callable
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows
from rasterio.features import geometry_window
import torch
from rasterio.enums import Resampling
from shapely.geometry import box
from rasterio.env import Env
import logging
import time

logger = logging.getLogger(__name__)


class GeoDataFrameDatasetOptimized(torch.utils.data.Dataset):
    """Optimized Torch Dataset with memory mapping and caching."""

    def __init__(
        self,
        path: str,  # path to geotiff
        geometries_gdf: gpd.GeoDataFrame,  # GeoDataFrame with geometries
        transforms: Callable | None = None,
        target_size: tuple[int, int] = (224, 224),  # Target size for all patches
        use_memmap: bool = True,  # Use memory mapping
        cache_windows: bool = True,  # Cache window calculations
    ) -> None:
        """Initialize the dataset.

        Args:
            path: The path to the geotiff file.
            geometries_gdf: GeoDataFrame containing polygon geometries to extract.
            transforms: The transforms to apply to the image.
            target_size: Target size (height, width) for all extracted patches.
            use_memmap: Whether to use memory mapping for reading.
            cache_windows: Whether to cache window calculations.
        """
        self.path = path
        self.transforms = transforms
        self.target_size = target_size
        self.use_memmap = use_memmap
        
        # Configure rasterio environment for optimal performance
        self.env_kwargs = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'GDAL_NUM_THREADS': 'ALL_CPUS',
            'GDAL_CACHEMAX': 512  # MB as integer
        }
        
        with Env(**self.env_kwargs):
            with rasterio.open(path) as src:
                self.width, self.height = src.width, src.height
                self.crs = src.crs
                self.transform = src.transform
                self.count = src.count
                
                # Ensure geometries are in the same CRS as the raster
                if geometries_gdf.crs != self.crs:
                    self.geometries_gdf = geometries_gdf.to_crs(self.crs)
                else:
                    self.geometries_gdf = geometries_gdf.copy()
        
        # Pre-calculate windows if caching is enabled
        self._window_cache = {}
        if cache_windows:
            self._precalculate_windows()
        
        assert len(self) > 0, "No geometries provided"

    def _precalculate_windows(self):
        """Pre-calculate all windows for faster access."""
        logger.info(f"Pre-calculating windows for {len(self.geometries_gdf)} geometries...")
        start_time = time.time()
        
        with Env(**self.env_kwargs):
            with rasterio.open(self.path) as src:
                for idx in range(len(self.geometries_gdf)):
                    if idx % 1000 == 0:
                        logger.info(f"Pre-calculating window {idx}/{len(self.geometries_gdf)}")
                    
                    geometry = self.geometries_gdf.iloc[idx].geometry
                    window = geometry_window(src, [geometry])
                    self._window_cache[idx] = window
        
        elapsed = time.time() - start_time
        logger.info(f"Window pre-calculation completed in {elapsed:.2f}s")

    def read_window_from_geometry(
        self,
        geometry,
        index: int,
        image_size: tuple[int, int] | None = None,
        interpolation: Resampling = Resampling.bilinear,
    ) -> tuple[np.ndarray, dict]:
        """Read the window corresponding to a geometry with optimizations."""
        
        # Use cached window if available
        if index in self._window_cache:
            window = self._window_cache[index]
        else:
            logger.debug(f"Computing window for geometry {index} (not cached)")
            with Env(**self.env_kwargs):
                with rasterio.open(self.path) as src:
                    window = geometry_window(src, [geometry])
        
        # Configure read options
        if image_size is None:
            out_shape = (self.count, self.target_size[0], self.target_size[1])
        else:
            out_shape = (self.count, image_size[0], image_size[1])
        
        logger.debug(f"Reading window for geometry {index}, shape: {out_shape}")
        
        # Read with optimized settings
        read_start = time.time()
        with Env(**self.env_kwargs):
            with rasterio.open(self.path, 'r') as src:
                # Use memory mapping if enabled
                if self.use_memmap:
                    data = src.read(
                        boundless=False,
                        window=window,
                        fill_value=0,
                        out_shape=out_shape,
                        resampling=interpolation,
                        masked=False  # Avoid mask overhead
                    )
                else:
                    data = src.read(
                        boundless=False,
                        window=window,
                        fill_value=0,
                        out_shape=out_shape,
                        resampling=interpolation
                    )
        
        read_elapsed = time.time() - read_start
        if read_elapsed > 0.1:  # Log slow reads
            logger.warning(f"Slow read for geometry {index}: {read_elapsed:.3f}s")
        
        # Get bounds of the window in pixel coordinates
        col_off = window.col_off
        row_off = window.row_off
        width = window.width
        height = window.height
        
        # Store geometry info
        geom_info = {
            'geometry': geometry,
            'index': index,
            'window': window,
            'bounds_pixel': (col_off, row_off, col_off + width, row_off + height)
        }
        
        logger.debug(f"Successfully read window for geometry {index}, data shape: {data.shape}")
        return data, geom_info

    def __len__(self) -> int:
        return len(self.geometries_gdf)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        start_time = time.time()
        logger.debug(f"Starting __getitem__ for index {index}")
        
        # Get geometry from GeoDataFrame
        geom_start = time.time()
        row = self.geometries_gdf.iloc[index]
        geometry = row.geometry
        geom_elapsed = time.time() - geom_start
        logger.debug(f"Geometry lookup for index {index}: {geom_elapsed:.4f}s")
        
        # Read window
        read_start = time.time()
        x, geom_info = self.read_window_from_geometry(geometry, index)
        read_elapsed = time.time() - read_start
        logger.debug(f"Window read for index {index}: {read_elapsed:.4f}s")
        
        # Convert to float32 tensor directly (avoid double conversion)
        tensor_start = time.time()
        x = torch.from_numpy(x.astype(np.float32, copy=False))
        tensor_elapsed = time.time() - tensor_start
        logger.debug(f"Tensor conversion for index {index}: {tensor_elapsed:.4f}s")
        
        # Clip negative values in-place
        clip_start = time.time()
        x.clamp_(min=0)
        clip_elapsed = time.time() - clip_start
        logger.debug(f"Tensor clipping for index {index}: {clip_elapsed:.4f}s")

        # Apply transforms
        transform_start = time.time()
        if self.transforms is not None:
            logger.debug(f"Applying transforms for index {index}")
            x = self.transforms(x)
        transform_elapsed = time.time() - transform_start
        logger.debug(f"Transform application for index {index}: {transform_elapsed:.4f}s")

        # Create result
        result_start = time.time()
        result = {
            "image": x, 
            "geometry_index": index,
            "original_size": torch.tensor([x.shape[1], x.shape[2]]),
            "tile_id": str(row.get('tile_id', index))
        }
        result_elapsed = time.time() - result_start
        logger.debug(f"Result creation for index {index}: {result_elapsed:.4f}s")
        
        total_elapsed = time.time() - start_time
        if total_elapsed > 0.5:  # Log slow items
            logger.warning(f"Slow __getitem__ for index {index}: {total_elapsed:.3f}s total")
        else:
            logger.debug(f"Completed __getitem__ for index {index}: {total_elapsed:.4f}s total")
        
        return result