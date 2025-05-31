import duckdb
import pandas as pd
import numpy as np
from shapely.geometry import Point
# from shapely import wkb # wkb is part of shapely, imported as shapely.wkb
import shapely.wkb
import os
import tempfile
import shutil
import time
import argparse # Added for command-line arguments
from pathlib import Path # Added for easier path manipulation

# --- 0. Configuration ---\n
# NUM_FILES = 2 # No longer needed
# ROWS_PER_FILE = 500  # No longer needed
EMBEDDING_DIM = 384 # Assuming this is fixed for user's Parquet files
DB_PATH = ':memory:' # Use a file path for persistence e.g., 'geo_embeddings.db'

# --- 1. Helper Function to Create Dummy GeoParquet Files --- (REMOVED)

def main(args):
    input_parquet_dir = Path(args.input_dir)
    if not input_parquet_dir.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found or is not a directory.")
        return

    parquet_file_paths = sorted(list(input_parquet_dir.glob("*.parquet")))

    if not parquet_file_paths:
        print(f"No .parquet files found in directory '{args.input_dir}'.")
        return
    
    print(f"Found {len(parquet_file_paths)} Parquet files in '{args.input_dir}':")
    for p_file in parquet_file_paths:
        print(f"  - {p_file.name}")

    # Create a temporary directory for profiling output
    temp_dir_profiling = tempfile.mkdtemp()
    print(f"Temporary directory for profiling output: {temp_dir_profiling}")

    try:
        # --- Step 1: DuckDB Setup ---\n
        print("\n--- 1. DuckDB Setup ---")
        con = duckdb.connect(database=DB_PATH, read_only=False)
        
        # Install and load extensions
        print("Loading extensions...")
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL vss; LOAD vss;") # Vector Similarity Search
        print("Spatial and VSS extensions loaded.")

        # --- Step 2: Table Ingestion ---\n
        print("\n--- 2. Table Ingestion ---")
        
        # Enclose file paths in quotes for the SQL string
        # Convert Path objects to strings for SQL
        path_strings = []
        for p_path in parquet_file_paths:
            # Resolve the path and convert to string
            resolved_path_str = str(p_path.resolve())
            # Replace backslashes with forward slashes for consistency in SQL string
            corrected_path_str = resolved_path_str.replace('\\', '/')
            path_strings.append(f"'{corrected_path_str}'")
        
        sql_parquet_files_list_str = "[" + ", ".join(path_strings) + "]"

        # Assuming 'id', 'embedding', and 'geometry_wkb' columns exist in user's Parquet files
        # User might need to adjust column names if they differ.
        create_table_sql = f"""
        CREATE OR REPLACE TABLE geo_embeddings AS
        SELECT
            id,  -- Ensure this column exists in your Parquet files
            CAST(embedding AS FLOAT[{EMBEDDING_DIM}]) as embedding, -- Ensure this column exists
            ST_GeomFromWKB(geometry_wkb) as geometry -- Ensure this column exists and is WKB
        FROM read_parquet({sql_parquet_files_list_str});
        """
        print("\nExecuting CREATE TABLE AS SELECT...")
        start_ingest_time = time.time()
        con.execute(create_table_sql)
        ingest_time = time.time() - start_ingest_time
        print(f"Table 'geo_embeddings' created and data ingested from {len(parquet_file_paths)} files in {ingest_time:.2f} seconds.")
        
        row_count = con.execute("SELECT COUNT(*) FROM geo_embeddings;").fetchone()[0]
        print(f"Total rows in 'geo_embeddings': {row_count}")
        
        # Verify schema (optional)
        print("\nTable schema:")
        print(con.execute("DESCRIBE geo_embeddings;").df())
        if row_count > 0:
            print("\nSample embedding type and length from first row:")
            sample_emb_info = con.execute("SELECT typeof(embedding), array_length(embedding) FROM geo_embeddings LIMIT 1;").fetchall()
            print(sample_emb_info)
        else:
            print("Table is empty, cannot fetch sample embedding info.")


        # --- Step 3: Index Creation ---\n
        if row_count > 0: # Only create indexes if there's data
            print("\n--- 3. Index Creation ---")
            
            print("\nCreating HNSW index on 'embedding' column...")
            start_hnsw_time = time.time()
            con.execute(f"CREATE INDEX IF NOT EXISTS emb_hnsw_idx ON geo_embeddings USING HNSW (embedding);")
            hnsw_time = time.time() - start_hnsw_time
            print(f"HNSW index 'emb_hnsw_idx' created in {hnsw_time:.2f} seconds.")

            print("\nCreating R-tree spatial index on 'geometry' column...")
            start_rtree_time = time.time()
            con.execute("CREATE INDEX IF NOT EXISTS geom_rtree_idx ON geo_embeddings USING RTREE (geometry);")
            rtree_time = time.time() - start_rtree_time
            print(f"R-tree index 'geom_rtree_idx' created in {rtree_time:.2f} seconds.")

            # --- Test Queries ---\n
            print("\n--- Test Queries ---")
            # Test Spatial Query (find 1 nearest geometry to a point)
            test_point_lon, test_point_lat = 10.0, 20.0
            print(f"\nTest Spatial Query: Finding nearest geometry to ({test_point_lon}, {test_point_lat})...")
            spatial_query_result_df = con.execute(f"""
            SELECT id, ST_AsText(geometry) as geom_wkt, ST_Distance(geometry, ST_Point({test_point_lon}, {test_point_lat})) as distance
            FROM geo_embeddings
            ORDER BY distance
            LIMIT 1;
            """).df()
            print("Nearest geometry:")
            print(spatial_query_result_df)

            # Test Similarity Search Query (find 3 nearest neighbors to the first embedding in the table)
            print("\nTest Similarity Search Query: Finding 3 nearest neighbors to the first embedding...")
            first_embedding_array_result = con.execute("SELECT embedding FROM geo_embeddings LIMIT 1;").fetchone()
            if first_embedding_array_result:
                first_embedding_array = first_embedding_array_result[0]
                similarity_query_result_df = con.execute(
                    "SELECT id, array_distance(embedding, ?) AS distance FROM geo_embeddings ORDER BY distance LIMIT 3;",
                    [first_embedding_array] # Pass embedding as a parameter
                ).df()
                print("Top 3 similar embeddings:")
                print(similarity_query_result_df)
            else:
                print("Could not retrieve first embedding for similarity search (table might be empty or have NULLs).")

            # --- 4. Profiling ---\n
            print("\n--- 4. Profiling ---")
            
            # Enable JSON profiling to a file
            con.execute("PRAGMA enable_profiling='json'")
            profile_file_path = os.path.join(temp_dir_profiling, "profile_output.json").replace('\\\\', '/')
            con.execute(f"PRAGMA profile_output='{profile_file_path}'")
            
            if first_embedding_array_result: # Only run if we have an embedding
                print(f"\nRunning profiled similarity query (profile output will be in {profile_file_path}):")
                _ = con.execute(
                    "SELECT id, array_distance(embedding, ?) AS distance FROM geo_embeddings ORDER BY distance LIMIT 10;",
                    [first_embedding_array]
                ).fetchall()
            
            con.execute("PRAGMA disable_profiling;") # Crucial to write the file
            
            if os.path.exists(profile_file_path) and os.path.getsize(profile_file_path) > 0:
                print(f"JSON Profile output generated at: {profile_file_path}.")
            else:
                print(f"JSON Profile output file was not generated or is empty at {profile_file_path}. Ensure PRAGMA disable_profiling was called or query was run.")
        else:
            print("\nSkipping Index Creation, Test Queries, and Profiling as the table is empty.")

        # --- Memory/Storage Usage (More relevant for file-based DB) ---\n
        print("\n--- Memory/Storage Usage ---")
        db_size_info = con.execute("PRAGMA database_size;").fetchone()
        print(f"PRAGMA database_size: {db_size_info}")

        if row_count > 0:
            print("\nStorage info for 'geo_embeddings' table (duckdb_storage()):")
            storage_info_df = con.execute("SELECT * FROM duckdb_storage('geo_embeddings');").df()
            print(storage_info_df[['segment_type', 'column_name', 'row_group_id', 'count', 'has_updates', 'persistent', 'estimated_size']])
            
            total_estimated_size = storage_info_df['estimated_size'].sum()
            print(f"\nTotal estimated in-memory size from duckdb_storage: {total_estimated_size / (1024*1024):.2f} MB")
        else:
            print("Skipping duckdb_storage() info as table is empty.")

        con.close()

    except duckdb.Error as e:
        print(f"\nA DuckDB error occurred: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up temporary directory for profiling
        print(f"\nCleaning up temporary profiling directory: {temp_dir_profiling}")
        shutil.rmtree(temp_dir_profiling)
        # If DB_PATH was a file and you want to clean it up:
        # if DB_PATH != ':memory:' and os.path.exists(DB_PATH):
        #     try:
        #         os.remove(DB_PATH)
        #         if os.path.exists(DB_PATH + ".wal"): # Also remove WAL file
        #             os.remove(DB_PATH + ".wal")
        #         print(f"Cleaned up database file: {DB_PATH}")
        #     except Exception as e_clean:
        #         print(f"Error cleaning up database file {DB_PATH}: {e_clean}")


    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Processes GeoParquet files to create a DuckDB database with spatial and HNSW vector indexes.
    Assumes Parquet files contain 'id', 'embedding' (list of floats), 
    and 'geometry_wkb' (WKB geometry as bytes) columns.
    """)
    parser.add_argument("input_dir", type=str, help="Directory containing input Parquet files.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)