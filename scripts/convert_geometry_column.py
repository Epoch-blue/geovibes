#!/usr/bin/env python3
"""Convert existing DuckDB database to match the expected schema.

Fixes:
- Converts geometry column from VARCHAR (WKT) to native GEOMETRY type
- Converts string IDs to BIGINT IDs (required for FAISS add_with_ids)
- Rebuilds FAISS index with proper ID alignment
- Optionally filters to points within a GeoJSON polygon (e.g., to remove sea points)
"""

import argparse
import sys
from pathlib import Path

import duckdb
import faiss
import numpy as np


def load_filter_geometry(conn: duckdb.DuckDBPyConnection, geojson_path: str) -> str:
    """Load GeoJSON and return WKT of the unioned geometry."""
    print(f"\nLoading filter geometry from: {geojson_path}")
    result = conn.execute(
        """
        SELECT ST_AsText(ST_Union_Agg(geom)) 
        FROM ST_Read(?)
        """,
        [geojson_path],
    ).fetchone()
    if not result or not result[0]:
        raise ValueError(f"Could not load geometry from {geojson_path}")
    wkt = result[0]
    print("  Loaded filter polygon")
    return wkt


def rebuild_faiss_index(conn: duckdb.DuckDBPyConnection, index_path: str) -> None:
    """Rebuild FAISS index from database using add_with_ids."""
    print("\nRebuilding FAISS index from database...")

    row_count = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    dim = conn.execute(
        "SELECT array_length(embedding, 1) FROM geo_embeddings LIMIT 1"
    ).fetchone()[0]
    print(f"  Rows: {row_count:,}, Embedding dim: {dim}")

    nlist = min(4096, max(1, row_count // 39))
    m = min(64, dim)
    nbits = 8

    print(f"  FAISS params: nlist={nlist}, m={m}, nbits={nbits}")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    print("  Training index...")
    train_size = min(row_count, max(row_count // 10, nlist * 256))
    train_df = conn.execute(
        f"SELECT embedding FROM geo_embeddings TABLESAMPLE RESERVOIR({train_size} ROWS)"
    ).fetchdf()
    train_array = np.vstack(train_df["embedding"].values).astype(np.float32)
    index.train(train_array)
    print(f"  Trained on {len(train_array):,} vectors")

    del train_array, train_df

    print("  Adding vectors with IDs (cursor-based for speed)...")
    batch_size = 1000000  # 1M vectors per batch
    num_batches = (row_count + batch_size - 1) // batch_size
    last_id = 0
    batch_num = 0
    total_added = 0

    while total_added < row_count:
        batch_df = conn.execute(
            f"SELECT id, embedding FROM geo_embeddings WHERE id > ? ORDER BY id LIMIT {batch_size}",
            [last_id],
        ).fetchdf()

        if batch_df.empty:
            break

        ids = batch_df["id"].values.astype(np.int64)
        vectors = np.vstack(batch_df["embedding"].values).astype(np.float32)

        index.add_with_ids(vectors, ids)

        last_id = int(ids[-1])
        total_added += len(ids)
        batch_num += 1
        print(
            f"    Batch {batch_num}/{num_batches}: added {len(ids):,} vectors (total: {total_added:,})"
        )

    faiss.write_index(index, index_path)
    print(f"  Saved FAISS index: {index_path}")


def convert_database(
    db_path: str,
    rebuild_faiss: bool = True,
    faiss_path: str | None = None,
    filter_geojson: str | None = None,
) -> None:
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    print(f"Opening database: {db_path}")
    conn = duckdb.connect(db_path)

    conn.install_extension("spatial")
    conn.load_extension("spatial")

    schema = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'geo_embeddings'"
    ).fetchall()
    schema_dict = {name: dtype for name, dtype in schema}

    if "geometry" not in schema_dict:
        print("Error: No 'geometry' column found in geo_embeddings table")
        conn.close()
        sys.exit(1)

    current_geom_type = schema_dict["geometry"]
    current_id_type = schema_dict.get("id", "UNKNOWN")
    has_tile_id = "tile_id" in schema_dict
    needs_geom_convert = current_geom_type != "GEOMETRY"
    needs_id_convert = current_id_type != "BIGINT"

    print(f"Current id column type: {current_id_type}")
    print(f"Current geometry column type: {current_geom_type}")
    print(f"Has tile_id column: {has_tile_id}")

    filter_wkt = None
    if filter_geojson:
        if not Path(filter_geojson).exists():
            print(f"Error: Filter GeoJSON not found: {filter_geojson}")
            conn.close()
            sys.exit(1)
        filter_wkt = load_filter_geometry(conn, filter_geojson)

    if not needs_geom_convert and not needs_id_convert and not filter_geojson:
        print(
            "Database already has correct schema (BIGINT id, GEOMETRY). Nothing to do."
        )
        conn.close()
        return

    row_count = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    print(f"Table has {row_count:,} rows")

    embedding_dim = conn.execute(
        "SELECT array_length(embedding, 1) FROM geo_embeddings LIMIT 1"
    ).fetchone()[0]
    print(f"Embedding dimension: {embedding_dim}")

    print("\nRecreating table with proper schema...")
    print("  Step 1: Creating sequence and new table...")

    conn.execute("DROP SEQUENCE IF EXISTS seq_geo_embeddings_id")
    conn.execute("CREATE SEQUENCE seq_geo_embeddings_id START 1")

    geom_expr = "ST_GeomFromText(geometry)" if needs_geom_convert else "geometry"

    tile_id_col = "tile_id VARCHAR," if has_tile_id else ""
    conn.execute(f"""
        CREATE TABLE geo_embeddings_new (
            id BIGINT PRIMARY KEY DEFAULT nextval('seq_geo_embeddings_id'),
            {tile_id_col}
            lon DOUBLE,
            lat DOUBLE,
            geometry GEOMETRY,
            embedding FLOAT[{embedding_dim}]
        )
    """)

    tile_id_insert = "tile_id, " if has_tile_id else ""
    tile_id_select = "tile_id, " if has_tile_id else ""

    if filter_wkt:
        print("  Filtering points to within GeoJSON polygon...")
        conn.execute(
            f"""
            INSERT INTO geo_embeddings_new ({tile_id_insert}lon, lat, geometry, embedding)
            SELECT {tile_id_select}lon, lat, {geom_expr}, embedding
            FROM geo_embeddings
            WHERE ST_Intersects({geom_expr}, ST_GeomFromText(?))
            """,
            [filter_wkt],
        )
    else:
        conn.execute(f"""
            INSERT INTO geo_embeddings_new ({tile_id_insert}lon, lat, geometry, embedding)
            SELECT {tile_id_select}lon, lat, {geom_expr}, embedding
            FROM geo_embeddings
        """)

    new_count = conn.execute("SELECT COUNT(*) FROM geo_embeddings_new").fetchone()[0]
    print(f"  Step 2: New table has {new_count:,} rows")

    if filter_wkt:
        filtered_out = row_count - new_count
        print(
            f"  Filtered out {filtered_out:,} points ({100 * filtered_out / row_count:.1f}% removed)"
        )
    elif new_count != row_count:
        print(f"Error: Row count mismatch! Original: {row_count}, New: {new_count}")
        conn.execute("DROP TABLE geo_embeddings_new")
        conn.execute("DROP SEQUENCE seq_geo_embeddings_id")
        conn.close()
        sys.exit(1)

    print("  Step 3: Dropping old table...")
    conn.execute("DROP TABLE geo_embeddings")

    print("  Step 4: Renaming new table...")
    conn.execute("ALTER TABLE geo_embeddings_new RENAME TO geo_embeddings")

    print("  Step 5: Creating indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lon ON geo_embeddings(lon)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lat ON geo_embeddings(lat)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lon_lat ON geo_embeddings(lon, lat)")

    new_schema = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'geo_embeddings'"
    ).fetchall()
    print("\nFinal schema:")
    for col, dtype in new_schema:
        print(f"  {col}: {dtype}")

    if rebuild_faiss:
        resolved_faiss_path = None
        if faiss_path and Path(faiss_path).exists():
            resolved_faiss_path = faiss_path
        else:
            auto_faiss_path = Path(db_path).with_name(
                Path(db_path).stem.replace("_metadata", "") + "_faiss.index"
            )
            if auto_faiss_path.exists():
                resolved_faiss_path = str(auto_faiss_path)
            else:
                alt_faiss_path = Path(db_path).parent / "embeddings_faiss.index"
                if alt_faiss_path.exists():
                    resolved_faiss_path = str(alt_faiss_path)

        if resolved_faiss_path:
            rebuild_faiss_index(conn, resolved_faiss_path)
        else:
            print("\nWarning: Could not find FAISS index.")
            print("Use --faiss-path to specify the correct path.")

    conn.close()

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing database to match expected schema (BIGINT id, GEOMETRY)"
    )
    parser.add_argument("db_path", help="Path to DuckDB database file")
    parser.add_argument(
        "--faiss-path",
        help="Path to FAISS index file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip rebuilding FAISS index",
    )
    parser.add_argument(
        "--filter-geojson",
        help="GeoJSON file to filter points (keeps only points within the polygon)",
    )
    args = parser.parse_args()

    convert_database(
        args.db_path,
        rebuild_faiss=not args.skip_faiss,
        faiss_path=args.faiss_path,
        filter_geojson=args.filter_geojson,
    )


if __name__ == "__main__":
    main()
