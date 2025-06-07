from dotenv import load_dotenv
import ee

def initialize_ee_with_credentials():
    """Initialize Earth Engine with user credentials or service account if configured."""
    load_dotenv()
    try:
        ee.Initialize(project='cr458-ee')  # Use user's default project
        print("✅ Earth Engine initialized with user credentials")
        return True
    except Exception as e:
        print(f"❌ Earth Engine authentication failed: {e}")
        print("\n🔧 To enable NDVI/NDWI basemaps, please run the following command:")
        print("    earthengine authenticate")
        print("⚠️  Continuing without Earth Engine (NDVI/NDWI basemaps will be unavailable)")
        return False  # Return False instead of raising an error


def get_s2_cloud_masked_collection(aoi: ee.Geometry,
                                 start_date: str = '2024-01-01', 
                                 end_date: str = '2025-12-31',
                                 clear_threshold: float = 0.80) -> ee.ImageCollection:
    """Get cloud-masked Sentinel-2 collection for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        
    Returns:
        Cloud-masked Sentinel-2 image collection
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    QA_BAND = 'cs_cdf'
    
    return s2.filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .linkCollection(csPlus, [QA_BAND]) \
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(clear_threshold)))


def get_s2_rgb_median(aoi: ee.Geometry,
                      start_date: str = '2024-01-01',
                      end_date: str = '2025-12-31', 
                      clear_threshold: float = 0.80,
                      scale_factor: float = 1) -> ee.Image:
    """Get median RGB composite from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        scale_factor: Factor to divide image values by
        
    Returns:
        Median RGB composite as Earth Engine Image with bands B4 (R), B3 (G), B2 (B)
    """
    collection = get_s2_cloud_masked_collection(aoi, start_date, end_date, clear_threshold)
    return collection.select(['B4', 'B3', 'B2']).median().divide(scale_factor)


def get_s2_ndvi_median(aoi: ee.Geometry,
                       start_date: str = '2024-01-01',
                       end_date: str = '2025-12-31',
                       clear_threshold: float = 0.80) -> ee.Image:
    """Get median NDVI from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        
    Returns:
        Median NDVI as Earth Engine Image
    """
    collection = get_s2_cloud_masked_collection(aoi, start_date, end_date, clear_threshold)
    ndvi = collection.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('ndvi'))
    return ndvi.median()


def get_s2_ndwi_median(aoi: ee.Geometry,
                       start_date: str = '2024-01-01',
                       end_date: str = '2025-12-31',
                       clear_threshold: float = 0.80) -> ee.Image:
    """Get median NDWI from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        
    Returns:
        Median NDWI as Earth Engine Image
    """
    collection = get_s2_cloud_masked_collection(aoi, start_date, end_date, clear_threshold)
    ndwi = collection.map(lambda img: img.normalizedDifference(['B3', 'B8']).rename('ndwi'))
    return ndwi.median()


def get_s2_hsv_median(aoi: ee.Geometry,
                      start_date: str = '2023-01-01',
                      end_date: str = '2024-12-31',
                      clear_threshold: float = 0.80) -> ee.Image:
    """Get median HSV composite from Sentinel-2 imagery for a given area and time period.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        clear_threshold: Minimum cloud score threshold (0-1) for including pixels
        
    Returns:
        Median HSV composite as Earth Engine Image with bands hue, saturation, value
    """
    rgb_median = get_s2_rgb_median(aoi, start_date, end_date, clear_threshold).divide(10000)
    hsv_median = ee.Image.rgbToHsv(rgb_median)
    return hsv_median.select(['hue', 'saturation', 'value'])
    

def get_ee_image_url(image: ee.Image, vis_params: dict) -> str:
    """Get tile URL for displaying Earth Engine image in web map.
    
    Args:
        image: Earth Engine Image to display
        vis_params: Dictionary of visualization parameters
        
    Returns:
        URL template string for map tiles
    """
    map_id = image.getMapId(vis_params)
    return map_id['tile_fetcher'].url_format