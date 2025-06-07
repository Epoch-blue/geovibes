"""
Constants for the GeoVibes application.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class UIConstants:
    """UI-related constants."""
    
    # Colors
    POS_COLOR = '#0072B2'       # Blue
    NEG_COLOR = '#D55E00'       # Orange  
    NEUTRAL_COLOR = '#999999'   # Grey
    REGION_COLOR = '#FAFAFA'    # Light gray
    SEARCH_COLOR = '#ffe014'    # Yellow
    DRAW_COLOR = '#6be5c3'      # Teal
    
    # Map settings
    DEFAULT_ZOOM = 7
    DEFAULT_HEIGHT = '700px'
    PANEL_WIDTH = '200px'
    
    # Search settings
    DEFAULT_NEIGHBORS = 1000
    MIN_NEIGHBORS = 100
    MAX_NEIGHBORS = 25000
    NEIGHBORS_STEP = 100
    
    # Point styles
    POINT_RADIUS = 4
    SEARCH_POINT_RADIUS = 3
    POINT_OPACITY = 1
    POINT_FILL_OPACITY = 0.7
    POINT_WEIGHT = 2
    SEARCH_POINT_WEIGHT = 1
    
    # UI dimensions
    BUTTON_HEIGHT = '40px'
    RESET_BUTTON_HEIGHT = '35px'
    COLLAPSE_BUTTON_SIZE = '25px'
    
    # Click threshold for point selection
    CLICK_THRESHOLD = 0.001
    
    # Label values
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    ERASE_LABEL = -100


class BasemapConfig:
    """Basemap configuration and tile URLs."""
    
    MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')
    
    # Attribution strings
    MAPTILER_ATTRIBUTION = (
        '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> '
        '<a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>'
    )
    
    # Base tile URLs
    BASEMAP_TILES = {
        'MAPTILER': f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
        'HUTCH_TILE': 'https://tiles.earthindex.ai/v1/tiles/sentinel2-yearly-mosaics/2024-01-01/2025-01-01/rgb/{z}/{x}/{y}.webp',
        'GOOGLE_HYBRID': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    }
    
    # Earth Engine basemap visualization parameters
    NDVI_VIS_PARAMS = {
        'min': -0.2,
        'max': 0.8,
        'palette': ['red', 'yellow', 'green']
    }
    
    NDWI_VIS_PARAMS = {
        'min': -0.5,
        'max': 0.5,
        'palette': ['brown', 'white', 'blue']
    }


class DatabaseConstants:
    """Database-related constants."""
    
    # Spatial query for DuckDB setup
    SPATIAL_SETUP_QUERY = """
    INSTALL spatial;
    LOAD spatial;
    """
    
    # Embedding dimension (could be made configurable later)
    EMBEDDING_DIM = 384
    
    # Chunk size for embedding fetching to avoid memory issues
    EMBEDDING_CHUNK_SIZE = 10000
    
    # Original search query with embeddings (kept for backward compatibility)
    SIMILARITY_SEARCH_QUERY = """
    WITH query(vec) AS (SELECT CAST(? AS FLOAT[384]))
    SELECT  g.id,
            g.embedding,
            ST_AsGeoJSON(g.geometry) AS geometry_json,
            ST_AsText(g.geometry) AS geometry_wkt,
            array_distance(g.embedding, q.vec) AS distance
    FROM    geo_embeddings AS g, query AS q
    ORDER BY distance
    LIMIT ?;
    """
    
    # Lightweight search query without embeddings (memory-efficient)
    SIMILARITY_SEARCH_LIGHT_QUERY = """
    WITH query(vec) AS (SELECT CAST(? AS FLOAT[384]))
    SELECT  g.id,
            ST_AsGeoJSON(g.geometry) AS geometry_json,
            ST_AsText(g.geometry) AS geometry_wkt,
            array_distance(g.embedding, q.vec) AS distance
    FROM    geo_embeddings AS g, query AS q
    ORDER BY distance
    LIMIT ?;
    """
    
    # Nearest point query without embedding (memory-efficient)
    NEAREST_POINT_LIGHT_QUERY = """
    SELECT  g.id,
            ST_AsText(g.geometry) AS wkt,
            ST_Distance(geometry, ST_Point(?, ?)) AS dist_m
    FROM    geo_embeddings g
    ORDER BY dist_m
    LIMIT   1
    """
    
    # Original nearest point query with embedding (kept for backward compatibility)
    NEAREST_POINT_QUERY = """
    SELECT  g.id,
            ST_AsText(g.geometry) AS wkt,
            ST_Distance(geometry, ST_Point(?, ?)) AS dist_m,
            g.embedding
    FROM    geo_embeddings g
    ORDER BY dist_m
    LIMIT   1
    """


class LayerStyles:
    """Map layer styling constants."""
    
    @classmethod
    def get_region_style(cls):
        """Get region boundary style."""
        return {
            'color': UIConstants.REGION_COLOR,
            'opacity': 1,
            'fillOpacity': 0,
            'weight': 1
        }
    
    @classmethod
    def get_point_style(cls, color):
        """Get point layer style for given color."""
        return {
            'color': color,
            'radius': UIConstants.POINT_RADIUS,
            'fillColor': color,
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.POINT_WEIGHT
        }
    
    @classmethod
    def get_erase_style(cls):
        """Get erase layer style."""
        return {
            'color': 'white',
            'radius': UIConstants.POINT_RADIUS,
            'fillColor': '#000000',
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.POINT_WEIGHT
        }
    
    @classmethod
    def get_search_style(cls):
        """Get search results layer style."""
        return {
            'color': 'black',
            'radius': UIConstants.SEARCH_POINT_RADIUS,
            'fillColor': UIConstants.SEARCH_COLOR,
            'opacity': UIConstants.POINT_OPACITY,
            'fillOpacity': UIConstants.POINT_FILL_OPACITY,
            'weight': UIConstants.SEARCH_POINT_WEIGHT
        }
    
    @classmethod
    def get_search_hover_style(cls):
        """Get search results hover style."""
        return {
            'fillColor': UIConstants.SEARCH_COLOR,
            'fillOpacity': 0.5
        }
    
    @classmethod
    def get_draw_options(cls):
        """Get draw control options."""
        return {
            "shapeOptions": {
                "color": UIConstants.DRAW_COLOR, 
                "fillOpacity": 0.5
            }
        } 