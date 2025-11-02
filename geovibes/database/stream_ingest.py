"""
Stream-based ingestion of coordinate points with embeddings into DuckDB.

This module provides efficient streaming ingestion using batching and
handles coordinate-based point data that will be stored alongside FAISS indices.
"""

from typing import Iterator, Optional, List
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging


def setup_stream_logging(log_file: str = "stream_ingest.log") -> None:
    """Configure logging for stream ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def create_embeddings_table(db_path: str, drop_existing: bool = False) -> None:
    """
    Create geo_embeddings table in DuckDB if it doesn't exist.
    
    Args:
        db_path: Path to DuckDB database
        drop_existing: Whether to drop existing table before creation
    """
    con = duckdb.connect(db_path)
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    if drop_existing:
        con.execute("DROP TABLE IF EXISTS geo_embeddings;")
        logging.info("Dropped existing geo_embeddings table")
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS geo_embeddings (
            id VARCHAR PRIMARY KEY,
            lon DOUBLE,
            lat DOUBLE,
            embedding FLOAT[],
            geometry GEOMETRY,
            tile_id VARCHAR
        );
    """)
    
    logging.info("Created geo_embeddings table in DuckDB")
    con.close()


def stream_ingest_dataframe(
    db_path: str,
    df: pd.DataFrame,
    batch_size: int = 10000,
    embedding_col: str = 'embedding',
    start_id: int = 0
) -> int:
    """
    Stream ingest a DataFrame into DuckDB in batches.
    
    Args:
        db_path: Path to DuckDB database
        df: DataFrame with columns: lon, lat, embedding, tile_id
        batch_size: Number of rows per batch
        embedding_col: Name of the embedding column
        
    Returns:
        Total number of rows ingested
    """
    con = duckdb.connect(db_path)
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    total_rows = 0
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end].copy()
        
        batch_df['id'] = [str(start_id + batch_start + i) for i in range(len(batch_df))]
        batch_df['geometry'] = batch_df.apply(
            lambda row: f"POINT({row['lon']} {row['lat']})",
            axis=1
        )
        
        con.execute("""
            INSERT INTO geo_embeddings (id, lon, lat, embedding, geometry, tile_id)
            SELECT 
                id,
                lon,
                lat,
                embedding,
                ST_GeomFromText(geometry),
                tile_id
            FROM batch_df
        """)
        
        batch_rows = batch_end - batch_start
        total_rows += batch_rows
        logging.info(f"Ingested batch {batch_start // batch_size + 1}: {batch_rows} rows (total: {total_rows})")
    
    con.close()
    logging.info(f"✅ Completed ingestion: {total_rows} total rows")
    
    return total_rows


def stream_ingest_generator(
    db_path: str,
    dataframe_generator: Iterator[pd.DataFrame],
    batch_size: int = 10000,
    embedding_col: str = 'embedding'
) -> int:
    """
    Stream ingest from a generator of DataFrames.
    
    Useful for processing large datasets that don't fit in memory.
    
    Args:
        db_path: Path to DuckDB database
        dataframe_generator: Iterator yielding DataFrames with embedding data
        batch_size: Number of rows per batch within each DataFrame
        embedding_col: Name of the embedding column
        
    Returns:
        Total number of rows ingested
    """
    con = duckdb.connect(db_path)
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    total_rows = 0
    global_batch_id = 0
    
    for df_chunk in dataframe_generator:
        for batch_start in range(0, len(df_chunk), batch_size):
            batch_end = min(batch_start + batch_size, len(df_chunk))
            batch_df = df_chunk.iloc[batch_start:batch_end].copy()
            
            start_id = total_rows + batch_start
            batch_df['id'] = [f"point_{i}" for i in range(start_id, start_id + len(batch_df))]
            batch_df['geometry'] = batch_df.apply(
                lambda row: f"POINT({row['lon']} {row['lat']})",
                axis=1
            )
            
            con.execute(f"""
                INSERT INTO geo_embeddings (id, lon, lat, embedding, geometry, tile_id)
                SELECT 
                    id,
                    lon,
                    lat,
                    embedding,
                    ST_GeomFromText(geometry),
                    tile_id
                FROM batch_df
            """)
            
            batch_rows = batch_end - batch_start
            total_rows += batch_rows
            global_batch_id += 1
            logging.info(f"Ingested batch {global_batch_id}: {batch_rows} rows (total: {total_rows})")
    
    con.close()
    logging.info(f"✅ Completed generator ingestion: {total_rows} total rows")
    
    return total_rows


def ingest_parquet_files_streaming(
    db_path: str,
    parquet_files: List[str],
    batch_size: int = 10000,
    embedding_col: str = 'embedding'
) -> int:
    """
    Stream ingest multiple Parquet files.
    
    Args:
        db_path: Path to DuckDB database
        parquet_files: List of paths to Parquet files
        batch_size: Number of rows per batch
        embedding_col: Name of the embedding column
        
    Returns:
        Total number of rows ingested
    """
    def parquet_generator():
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            logging.info(f"Loaded {len(df)} rows from {parquet_file}")
            yield df
    
    return stream_ingest_generator(db_path, parquet_generator(), batch_size, embedding_col)


def create_rtree_index(db_path: str) -> None:
    """
    Create R-Tree spatial index on geometry column for efficient spatial queries.
    
    Args:
        db_path: Path to DuckDB database
    """
    con = duckdb.connect(db_path)
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    try:
        con.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")
        logging.info("✅ Created R-Tree spatial index on geometry column")
    except Exception as e:
        logging.warning(f"⚠️  Could not create spatial index: {e}")
    finally:
        con.close()


def verify_ingestion(db_path: str) -> dict:
    """
    Verify ingestion by checking table statistics.
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with ingestion statistics
    """
    con = duckdb.connect(db_path)
    
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    
    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
    
    embedding_dim_query = con.execute("""
        SELECT LENGTH(embedding) as embedding_dim
        FROM geo_embeddings
        WHERE embedding IS NOT NULL
        LIMIT 1
    """).fetchone()
    
    embedding_dim = embedding_dim_query[0] if embedding_dim_query else None
    
    stats = {
        'total_rows': row_count,
        'embedding_dimension': embedding_dim,
        'bounds': None
    }
    
    if row_count > 0:
        bounds = con.execute("""
            SELECT 
                MIN(lon) as min_lon,
                MAX(lon) as max_lon,
                MIN(lat) as min_lat,
                MAX(lat) as max_lat
            FROM geo_embeddings
        """).fetchone()
        
        stats['bounds'] = {
            'min_lon': bounds[0],
            'max_lon': bounds[1],
            'min_lat': bounds[2],
            'max_lat': bounds[3]
        }
    
    con.close()
    
    logging.info(f"Ingestion statistics: {stats}")
    
    return stats

