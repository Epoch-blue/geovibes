import duckdb
import time
import argparse
import logging
from pathlib import Path
from typing import List

def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_database(
    parquet_files: List[Path], 
    output_file: str, 
    embedding_column: str,
    embedding_dim: int,
    metric: str
) -> None:
    """Create DuckDB database with spatial and HNSW indexes."""
    
    logging.info(f"Using embedding dimension: {embedding_dim}")
    
    con = duckdb.connect(database=output_file)
    
    logging.info("Loading extensions...")
    con.execute("SET enable_progress_bar=true")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    
    path_strings = [f"'{str(p.resolve()).replace('\\', '/')}'" for p in parquet_files]
    sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"
    
    create_table_sql = f"""
    CREATE OR REPLACE TABLE geo_embeddings AS
    SELECT
        id,
        CAST({embedding_column} AS FLOAT[{embedding_dim}]) as embedding,
        geometry
    FROM read_parquet({sql_parquet_files_list_str});
    """
    
    logging.info("Creating table and ingesting data...")
    start_time = time.time()
    con.execute(create_table_sql)
    ingest_time = time.time() - start_time
    
    row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
    logging.info(f"Ingested {row_count} rows in {ingest_time:.2f} seconds")
    
    if row_count > 0:
        logging.info(f"Creating HNSW index with {metric} metric...")
        start_index_time = time.time()
        con.execute(
            f"CREATE INDEX IF NOT EXISTS emb_hnsw_idx ON geo_embeddings USING HNSW (embedding) WITH (metric = '{metric}');")
        index_time = time.time() - start_index_time
        logging.info(f"HNSW index created in {index_time:.2f} seconds")
        
        logging.info("Creating RTree spatial index...")
        start_spatial_index_time = time.time()
        con.execute("CREATE INDEX IF NOT EXISTS geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")
        spatial_index_time = time.time() - start_spatial_index_time
        logging.info(f"RTree spatial index created in {spatial_index_time:.2f} seconds")
        
        db_size_info = con.execute("PRAGMA database_size;").fetchone()
        logging.info(f"Database size: {db_size_info}")
    
    con.close()
    logging.info(f"Database saved to {output_file}")

def main() -> None:
    """Main function to process GeoParquet files and create DuckDB database."""
    parser = argparse.ArgumentParser(
        description="Create DuckDB database with spatial and HNSW vector indexes from GeoParquet files"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing input Parquet files")
    parser.add_argument("output_file", type=str, help="Output database file path")
    parser.add_argument("--embedding_column", type=str, default="embedding", 
                       help="Name of embedding column (default: embedding)")
    parser.add_argument("--embedding_dim", type=int, default=384,
                       help="Embedding dimension (default: 384)")
    parser.add_argument("--metric", type=str, choices=["l2sq", "cosine", "ip"], 
                       default="cosine", help="Distance metric for HNSW index (default: cosine)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    input_dir = Path(args.input_dir)
    parquet_files = sorted(list(input_dir.glob("*.parquet")))
    
    if not parquet_files:
        logging.error(f"No .parquet files found in directory '{args.input_dir}'")
        return
    
    logging.info(f"Found {len(parquet_files)} Parquet files")
    
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    
    final_output_file = output_dir / f"{output_stem}_{args.metric}{output_suffix}"
    create_database(parquet_files, str(final_output_file), args.embedding_column, args.embedding_dim, args.metric)

if __name__ == "__main__":
    main()