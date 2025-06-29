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
    
    EXTENSION_SETUP_QUERY = """
    INSTALL spatial;
    LOAD spatial;
    """
    
    # Chunk size for embedding fetching to avoid memory issues
    EMBEDDING_CHUNK_SIZE = 10000
    
    @staticmethod
    def detect_embedding_dimension(duckdb_connection) -> int:
        """Detect embedding dimension from first row of database.
        
        Args:
            duckdb_connection: Active DuckDB connection
            
        Returns:
            int: Embedding dimension
            
        Raises:
            ValueError: If no embeddings found or dimension cannot be detected
        """
        try:
            result = duckdb_connection.execute(
                "SELECT embedding FROM geo_embeddings LIMIT 1"
            ).fetchone()
            
            if result and result[0]:
                return len(result[0])
            else:
                raise ValueError("No embeddings found in database")
        except Exception as e:
            raise ValueError(f"Could not detect embedding dimension: {e}")
    
    @staticmethod
    def detect_embedding_dimension_from_parquet(parquet_path: str, embedding_column: str = 'embedding') -> int:
        """Detect embedding dimension from parquet file.
        
        Args:
            parquet_path: Path to parquet file
            embedding_column: Name of embedding column (default: 'embedding')
            
        Returns:
            int: Embedding dimension
            
        Raises:
            ValueError: If no embeddings found or dimension cannot be detected
        """
        try:
            import pandas as pd
            
            # Read just the first row
            df = pd.read_parquet(parquet_path, nrows=1)
            
            if embedding_column not in df.columns:
                raise ValueError(f"Embedding column '{embedding_column}' not found in parquet file")
            
            embedding = df[embedding_column].iloc[0]
            if hasattr(embedding, '__len__'):
                return len(embedding)
            else:
                raise ValueError(f"Embedding in column '{embedding_column}' is not array-like")
                
        except Exception as e:
            raise ValueError(f"Could not detect embedding dimension from parquet: {e}")
    
    @staticmethod
    def get_similarity_search_query(embedding_dim: int) -> str:
        """Generate similarity search query with embeddings for given dimension.
        
        Args:
            embedding_dim: Dimension of the embeddings
            
        Returns:
            str: SQL query string
        """
        return f"""
        WITH query(vec) AS (SELECT CAST(? AS FLOAT[{embedding_dim}]))
        SELECT  g.id,
                g.embedding,
                ST_AsGeoJSON(g.geometry) AS geometry_json,
                ST_AsText(g.geometry) AS geometry_wkt,
                array_distance(g.embedding, q.vec) AS distance
        FROM    geo_embeddings AS g, query AS q
        ORDER BY distance
        LIMIT ?;
        """
    
    @staticmethod
    def get_similarity_search_light_query(embedding_dim: int) -> str:
        """Generate lightweight similarity search query for given dimension.
        
        Args:
            embedding_dim: Dimension of the embeddings
            
        Returns:
            str: SQL query string
        """
        return f"""
        SELECT  g.id,
                ST_AsGeoJSON(g.geometry) AS geometry_json,
                ST_AsText(g.geometry)  AS geometry_wkt,
                distance
        FROM (
            SELECT  g.id,
                    g.geometry,
                    g.embedding <-> CAST(? AS FLOAT[{embedding_dim}]) AS distance
            FROM    geo_embeddings g
            WHERE   g.embedding IS NOT NULL
            ORDER BY distance
            LIMIT ?
        ) g;
        """
    
    # Legacy constants for backward compatibility (deprecated)
    @property
    def EMBEDDING_DIM(self):
        """Deprecated: Use detect_embedding_dimension() instead."""
        import warnings
        warnings.warn(
            "EMBEDDING_DIM is deprecated. Use detect_embedding_dimension() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return 1000  # Default fallback
    
    @property 
    def SIMILARITY_SEARCH_QUERY(self):
        """Deprecated: Use get_similarity_search_query() instead."""
        import warnings
        warnings.warn(
            "SIMILARITY_SEARCH_QUERY is deprecated. Use get_similarity_search_query() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_similarity_search_query(1000)
    
    @property
    def SIMILARITY_SEARCH_LIGHT_QUERY(self):
        """Deprecated: Use get_similarity_search_light_query() instead."""
        import warnings
        warnings.warn(
            "SIMILARITY_SEARCH_LIGHT_QUERY is deprecated. Use get_similarity_search_light_query() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_similarity_search_light_query(1000)

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