import argparse
import logging
import pathlib
import time
import duckdb
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import gc

def setup_logging():
    """Configure basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('faiss_build.log')
        ]
    )

def ingest_parquet_to_duckdb(parquet_files: list[str], db_path: str, embedding_dim: int):
    """
    Ingests data from Parquet files into a DuckDB database.
    Args:
        parquet_files: List of paths to the input Parquet files.
        db_path: Path to the DuckDB database file to be created.
        embedding_dim: The dimension of the vector embeddings.
    """
    if not parquet_files:
        logging.error("No parquet files provided for ingestion.")
        return
        
    logging.info(f"Starting ingestion of {len(parquet_files)} parquet files into {db_path}...")

    # Inspect the schema of the first file to determine available columns
    try:
        with duckdb.connect() as con_check:
            df_schema = con_check.execute(f"DESCRIBE SELECT * FROM read_parquet(?);", (parquet_files[0],)).fetchdf()
            available_columns = set(df_schema['column_name'])
    except Exception as e:
        logging.error(f"Fatal: Could not read schema from {parquet_files[0]}: {e}")
        return

    has_geometry = 'geometry' in available_columns
    
    # Use 'tile_id' if available, otherwise fall back to 'id'
    id_column_in_parquet = 'tile_id' if 'tile_id' in available_columns else 'id'
    if id_column_in_parquet not in available_columns:
        logging.error(f"Fatal: Neither 'tile_id' nor 'id' column found in {parquet_files[0]}. Available: {available_columns}")
        return

    logging.info(f"Source ID column: '{id_column_in_parquet}'. Geometry column found: {has_geometry}.")

    with duckdb.connect(database=db_path) as con:
        # Load spatial extension, which is required for creating a spatial index
        if has_geometry:
            logging.info("Installing and loading spatial extension for DuckDB.")
            con.execute("INSTALL spatial; LOAD spatial;")

        logging.info("Creating table 'geo_embeddings' in DuckDB.")
        
        # Check if table already exists
        table_exists = con.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'geo_embeddings'
        """).fetchone()[0] > 0
        
        if table_exists:
            existing_count = con.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
            logging.warning(f"Table 'geo_embeddings' already exists with {existing_count} rows. Dropping and recreating...")
            con.execute("DROP TABLE geo_embeddings;")
            con.execute("DROP SEQUENCE IF EXISTS seq_geo_embeddings_id;")
        
        # Create a sequence for auto-incrementing IDs
        con.execute("CREATE SEQUENCE IF NOT EXISTS seq_geo_embeddings_id START 1;")
        
        # Create table with dynamic schema
        create_sql = f"""
        CREATE TABLE geo_embeddings (
            id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
            tile_id VARCHAR,
            embedding FLOAT[{embedding_dim}]
            {', geometry GEOMETRY' if has_geometry else ''}
        );
        """
        con.execute(create_sql)
        
        # Ingest all files in a single, efficient query
        try:
            sql_parquet_files_list_str = "['" + "', '".join(parquet_files) + "']"
            
            insert_columns = f"tile_id, embedding{', geometry' if has_geometry else ''}"
            
            # The geometry column from GeoParquet is already understood by DuckDB's spatial extension.
            # No explicit cast is needed if the target column is type GEOMETRY.
            select_clause = f"{id_column_in_parquet}, embedding"
            if has_geometry:
                select_clause += ", geometry"

            con.execute(f"""
                INSERT INTO geo_embeddings ({insert_columns})
                SELECT
                    {select_clause}
                FROM read_parquet({sql_parquet_files_list_str}, union_by_name=true);
            """)
        except Exception as e:
            logging.error(f"Failed to ingest parquet files: {e}")
            # Provide more context on the error
            logging.error("Please ensure all parquet files have a consistent schema.")
            return

        row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Successfully ingested {row_count} rows into DuckDB.")

        if has_geometry:
            logging.info("Creating R-Tree spatial index on geometry column...")
            con.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry);")
            logging.info("Spatial index created successfully.")


def create_faiss_index(db_path: str, index_path: str, embedding_dim: int, nlist: int, m: int, nbits: int):
    """
    Creates a FAISS index from embeddings stored in a DuckDB database.

    Args:
        db_path: Path to the DuckDB database.
        index_path: Path to save the final FAISS index.
        embedding_dim: The dimension of the embeddings.
        nlist: The number of cells for the IVF index.
        m: The number of sub-quantizers for Product Quantization.
        nbits: The number of bits per sub-quantizer code.
    """
    logging.info("Starting FAISS index creation.")
    
    with duckdb.connect(database=db_path, read_only=True) as con:
        total_vectors = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        logging.info(f"Total vectors to index: {total_vectors}")

        if total_vectors == 0:
            logging.error("The 'geo_embeddings' table is empty after ingestion. Nothing to index. Aborting.")
            logging.error("This might happen if the input parquet files are empty or have an incompatible schema.")
            return

        # 1. Define the FAISS index
        # We use IndexIVFPQ for memory efficiency and speed on large datasets.
        # The quantizer is a simple L2 index used to find the cluster centroids.
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
        
        # 2. Train the index on a sample of the data
        logging.info("Training FAISS index...")
        # Determine a good sample size for training, e.g., 10% or up to 2M vectors
        train_sample_size = min(total_vectors, max(total_vectors // 10, 2000000))
        logging.info(f"Using {train_sample_size} vectors for training...")

        # Use TABLESAMPLE for efficient random sampling from DuckDB
        training_vectors_df = con.execute(f"SELECT embedding FROM geo_embeddings TABLESAMPLE RESERVOIR({train_sample_size} ROWS);").fetchdf()
        
        if training_vectors_df.empty:
            logging.error("Failed to sample training vectors from the database, even though it appears to contain data. Aborting.")
            return
            
        training_vectors = np.vstack(training_vectors_df['embedding'].values).astype('float32')

        start_time = time.time()
        index.train(training_vectors)
        logging.info(f"Index training completed in {time.time() - start_time:.2f} seconds.")

        del training_vectors # Free up memory
        gc.collect()

        # 3. Populate the index in batches
        logging.info("Populating FAISS index in batches...")
        batch_size = 100_000
        num_batches = (total_vectors + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Populating FAISS Index"):
            offset = i * batch_size
            # Fetch a batch of IDs and embeddings from DuckDB
            batch_df = con.execute(f"SELECT id, embedding FROM geo_embeddings ORDER BY id LIMIT {batch_size} OFFSET {offset};").fetchdf()
            
            ids = batch_df['id'].values.astype('int64')
            vectors = np.vstack(batch_df['embedding'].values).astype('float32')

            # Add the batch to the index with their original DuckDB IDs
            index.add_with_ids(vectors, ids)
            
        logging.info("Index population complete.")
        
        # 4. Save the final index to disk
        logging.info(f"Writing index to {index_path}")
        faiss.write_index(index, index_path)
        logging.info("FAISS index successfully built and saved.")


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from geospatial embeddings stored in Parquet files.")
    parser.add_argument("parquet_files", nargs='+', help="Paths to input Parquet files or a glob pattern.")
    parser.add_argument("--output_dir", default=".", help="Directory to save the DuckDB and FAISS index files.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the embedding vectors.")
    parser.add_argument("--nlist", type=int, default=4096, help="Number of clusters (IVF cells) for the FAISS index.")
    parser.add_argument("--m", type=int, default=64, help="Number of sub-quantizers for FAISS Product Quantization.")
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits per sub-quantizer code.")
    parser.add_argument("--dry-run", action="store_true", help="Run the script on a small subset of files to test the pipeline.")
    parser.add_argument("--dry-run-size", type=int, default=5, help="Number of files to use in a dry run.")
    
    args = parser.parse_args()
    
    setup_logging()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = str(output_dir / "geovibes_metadata.db")
    index_path = str(output_dir / "geovibes_ann.index")

    # --- File Discovery ---
    all_files = []
    for path_pattern in args.parquet_files:
        # Check if it's a directory
        if pathlib.Path(path_pattern).is_dir():
            all_files.extend(glob.glob(f"{path_pattern}/**/*.parquet", recursive=True))
        else:
            # Assume it's a glob pattern or a single file
            all_files.extend(glob.glob(path_pattern))
            
    if not all_files:
        logging.error(f"No parquet files found for the given paths: {args.parquet_files}")
        return

    files_to_process = all_files
    if args.dry_run:
        logging.info(f"--- Starting DRY RUN using the first {args.dry_run_size} files. ---")
        files_to_process = files_to_process[:args.dry_run_size]

    if not files_to_process:
        logging.error("No files left to process after applying dry-run filter.")
        return
    
    start_total_time = time.time()

    # --- Phase 1: Ingest data into DuckDB ---
    ingest_parquet_to_duckdb(files_to_process, db_path, args.embedding_dim)

    # --- Phase 2: Build FAISS index ---
    create_faiss_index(db_path, index_path, args.embedding_dim, args.nlist, args.m, args.nbits)
    
    logging.info(f"Total process completed in {time.time() - start_total_time:.2f} seconds.")
    logging.info(f"Artifacts created:\n- DuckDB: {db_path}\n- FAISS Index: {index_path}")

if __name__ == "__main__":
    main() 