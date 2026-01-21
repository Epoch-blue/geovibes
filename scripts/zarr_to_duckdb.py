#!/usr/bin/env python3
"""Convert a zarr embedding store to DuckDB + FAISS for use in geovibes UI.

Two-phase approach:
1. Dask distributed parallel export zarr chunks to parquet files
2. Bulk load parquet into DuckDB (fast)
"""

import argparse
import shutil
import time
from pathlib import Path

import dask
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


@dask.delayed
def process_chunk_to_parquet(
    zarr_path: str,
    output_path: str,
    lon_start: int,
    lon_end: int,
    lat_start: int,
    lat_end: int,
    chunk_id: int,
    embedding_vars: list[str],
    lon_dim: str,
    lat_dim: str,
) -> int:
    """
    Load a zarr chunk and write to parquet file.

    Returns number of points written.
    """
    ds = xr.open_zarr(zarr_path)

    ds_slice = ds.isel(
        {
            lon_dim: slice(lon_start, lon_end),
            lat_dim: slice(lat_start, lat_end),
        }
    )

    if "time" in ds_slice.dims:
        ds_slice = ds_slice.isel(time=0)

    ds_computed = ds_slice.compute()

    lon_coords = ds_computed[lon_dim].values
    lat_coords = ds_computed[lat_dim].values

    embedding_stack = np.stack(
        [ds_computed[var].values for var in embedding_vars], axis=0
    )

    if embedding_stack.ndim == 4:
        embedding_stack = embedding_stack[:, 0, :, :]

    n_bands = embedding_stack.shape[0]
    embeddings_reshaped = embedding_stack.reshape(n_bands, -1).T.astype(np.float32)

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords, indexing="ij")
    lons_flat = lon_grid.ravel().astype(np.float64)
    lats_flat = lat_grid.ravel().astype(np.float64)

    valid_mask = ~np.isnan(embeddings_reshaped).any(axis=1)

    lons_valid = lons_flat[valid_mask]
    lats_valid = lats_flat[valid_mask]
    embeddings_valid = embeddings_reshaped[valid_mask]

    ds.close()

    if len(lons_valid) == 0:
        return 0

    geometries = [f"POINT({lon} {lat})" for lon, lat in zip(lons_valid, lats_valid)]
    embedding_lists = [row.tolist() for row in embeddings_valid]

    table = pa.table(
        {
            "lon": lons_valid,
            "lat": lats_valid,
            "geometry": geometries,
            "embedding": embedding_lists,
        }
    )

    pq.write_table(table, output_path)
    return len(lons_valid)


def parquet_to_duckdb(
    parquet_dir: str,
    db_path: str,
    embedding_dim: int,
) -> int:
    """Bulk load parquet files into DuckDB. Returns row count."""
    print("\n💾 Loading parquet into DuckDB...")

    conn = duckdb.connect(db_path)
    conn.install_extension("spatial")
    conn.load_extension("spatial")

    parquet_cols = conn.execute(
        f"SELECT column_name FROM parquet_schema('{parquet_dir}/*.parquet')"
    ).fetchall()
    parquet_col_names = {row[0] for row in parquet_cols}
    has_tile_id = "tile_id" in parquet_col_names

    conn.execute("DROP TABLE IF EXISTS geo_embeddings")
    conn.execute("DROP SEQUENCE IF EXISTS seq_geo_embeddings_id")
    conn.execute("CREATE SEQUENCE seq_geo_embeddings_id START 1")

    tile_id_col = "tile_id VARCHAR," if has_tile_id else ""
    conn.execute(f"""
        CREATE TABLE geo_embeddings (
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
    conn.execute(f"""
        INSERT INTO geo_embeddings ({tile_id_insert}lon, lat, geometry, embedding)
        SELECT 
            {tile_id_select}lon,
            lat,
            ST_GeomFromText(geometry),
            CAST(embedding AS FLOAT[{embedding_dim}])
        FROM read_parquet('{parquet_dir}/*.parquet')
    """)

    count_result = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()
    count = count_result[0] if count_result else 0
    print(f"   Loaded {count:,} rows")

    print("📍 Creating indexes...")
    conn.execute("CREATE INDEX idx_lon ON geo_embeddings(lon)")
    conn.execute("CREATE INDEX idx_lat ON geo_embeddings(lat)")
    conn.execute("CREATE INDEX idx_lon_lat ON geo_embeddings(lon, lat)")
    conn.close()

    print("✅ DuckDB table created with indexes")
    return count


def create_faiss_index_from_duckdb(
    db_path: str,
    index_path: str,
    nlist: int = 4096,
    m: int = 64,
    nbits: int = 8,
    batch_size: int = 50000,
) -> None:
    """Create FAISS index from DuckDB table using add_with_ids for proper ID alignment."""
    import faiss

    print("\n🔍 Creating FAISS index from DuckDB...")

    conn = duckdb.connect(db_path, read_only=True)

    total_rows = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    dim = conn.execute(
        "SELECT array_length(embedding, 1) FROM geo_embeddings LIMIT 1"
    ).fetchone()[0]

    print(f"   Total rows: {total_rows:,}, Dimension: {dim}")

    if total_rows < nlist * 39:
        nlist = max(1, total_rows // 39)
        print(f"   Adjusted nlist to {nlist} for dataset size")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    print("   Training index...")
    train_size = min(total_rows, max(total_rows // 10, nlist * 256))
    train_df = conn.execute(
        f"SELECT embedding FROM geo_embeddings TABLESAMPLE RESERVOIR({train_size} ROWS)"
    ).fetchdf()
    train_array = np.vstack(train_df["embedding"].values).astype(np.float32)
    index.train(train_array)
    print(f"   Trained on {len(train_array):,} vectors")

    del train_array, train_df

    print("   Adding vectors with IDs (cursor-based for speed)...")
    actual_batch_size = 1000000  # 1M vectors per batch
    num_batches = (total_rows + actual_batch_size - 1) // actual_batch_size
    last_id = 0
    batch_num = 0
    total_added = 0

    while total_added < total_rows:
        batch_df = conn.execute(
            f"SELECT id, embedding FROM geo_embeddings WHERE id > ? ORDER BY id LIMIT {actual_batch_size}",
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
            f"      Batch {batch_num}/{num_batches}: added {len(ids):,} vectors (total: {total_added:,})"
        )

    conn.close()

    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved: {index_path}")


def zarr_to_duckdb(
    zarr_path: str,
    output_dir: str,
    db_name: str = "embeddings",
    n_workers: int = 4,
    skip_faiss: bool = False,
    nlist: int = 4096,
    keep_parquet: bool = False,
) -> dict:
    """
    Convert zarr embedding store to DuckDB + FAISS.

    Phase 1: Dask distributed writes parquet files in parallel
    Phase 2: DuckDB bulk loads from parquet
    """
    from dask.distributed import Client, LocalCluster

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_path = str(output_path / f"{db_name}_metadata.db")
    index_path = str(output_path / f"{db_name}_faiss.index")
    parquet_dir = output_path / "parquet_chunks"

    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    parquet_dir.mkdir(parents=True)

    print("\n" + "=" * 70)
    print("🚀 Zarr to DuckDB Conversion (via Parquet)")
    print("=" * 70)
    print(f"   Input: {zarr_path}")
    print(f"   Output DB: {db_path}")
    print(f"   Workers: {n_workers}")

    print(f"\n🔧 Starting dask cluster with {n_workers} workers...")
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"   Dashboard: {client.dashboard_link}")

    start_time = time.time()

    print("\n📂 Opening zarr store...")
    ds = xr.open_zarr(zarr_path)
    print(f"   Dimensions: {dict(ds.sizes)}")

    embedding_vars = sorted([v for v in ds.data_vars if v.startswith("A")])
    embedding_dim = len(embedding_vars)
    print(f"   Embedding bands: {embedding_dim}")

    lon_dim = "lon" if "lon" in ds.dims else "X"
    lat_dim = "lat" if "lat" in ds.dims else "Y"

    chunk_size = (512, 512)
    n_lon = ds.sizes[lon_dim]
    n_lat = ds.sizes[lat_dim]

    n_lon_chunks = (n_lon + chunk_size[0] - 1) // chunk_size[0]
    n_lat_chunks = (n_lat + chunk_size[1] - 1) // chunk_size[1]
    total_chunks = n_lon_chunks * n_lat_chunks

    print(f"\n📦 Processing {total_chunks} chunks to parquet...")
    print(f"   Grid: {n_lon_chunks} x {n_lat_chunks} chunks")

    ds.close()

    delayed_tasks = []
    chunk_id = 0

    for lon_start in range(0, n_lon, chunk_size[0]):
        lon_end = min(lon_start + chunk_size[0], n_lon)
        for lat_start in range(0, n_lat, chunk_size[1]):
            lat_end = min(lat_start + chunk_size[1], n_lat)

            parquet_path = str(parquet_dir / f"chunk_{chunk_id:05d}.parquet")

            task = process_chunk_to_parquet(
                zarr_path,
                parquet_path,
                lon_start,
                lon_end,
                lat_start,
                lat_end,
                chunk_id,
                embedding_vars,
                lon_dim,
                lat_dim,
            )
            delayed_tasks.append(task)
            chunk_id += 1

    print(f"   Created {len(delayed_tasks)} delayed tasks")
    print("\n" + "-" * 70)
    print("Computing (writing parquet files)...")
    print("-" * 70)

    results = dask.compute(*delayed_tasks)
    total_points_from_tasks = sum(results)

    print(
        f"✅ Wrote {total_points_from_tasks:,} points to {len(delayed_tasks)} parquet files"
    )

    client.close()
    cluster.close()

    total_points = parquet_to_duckdb(str(parquet_dir), db_path, embedding_dim)

    if not skip_faiss and total_points > 0:
        create_faiss_index_from_duckdb(db_path, index_path, nlist=nlist)

    if not keep_parquet:
        print("\n🗑️  Cleaning up parquet files...")
        shutil.rmtree(parquet_dir)

    elapsed = time.time() - start_time
    rate = total_points / elapsed if elapsed > 0 else 0
    print(
        f"\n✅ Converted {total_points:,} points in {elapsed:.1f}s ({rate:.0f} pts/s)"
    )

    return {
        "db_path": db_path,
        "index_path": index_path if not skip_faiss else None,
        "total_points": total_points,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert zarr embedding store to DuckDB + FAISS (via parquet)"
    )
    parser.add_argument("zarr_path", help="Path to zarr store")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument(
        "--db-name", default="embeddings", help="Base name for output files"
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="Dask workers")
    parser.add_argument(
        "--skip-faiss", action="store_true", help="Skip FAISS index creation"
    )
    parser.add_argument("--nlist", type=int, default=4096, help="FAISS IVF clusters")
    parser.add_argument(
        "--keep-parquet", action="store_true", help="Keep intermediate parquet files"
    )

    args = parser.parse_args()

    result = zarr_to_duckdb(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        db_name=args.db_name,
        n_workers=args.workers,
        skip_faiss=args.skip_faiss,
        nlist=args.nlist,
        keep_parquet=args.keep_parquet,
    )

    print("\n" + "=" * 70)
    print("📋 Summary")
    print("=" * 70)
    print(f"   Database: {result['db_path']}")
    if result["index_path"]:
        print(f"   FAISS Index: {result['index_path']}")
    print(f"   Total Points: {result['total_points']:,}")
    print(f"   Time: {result['elapsed_seconds']:.1f}s")
    print("\nTo use in geovibes UI, run: uv run jupyter lab")
    print("Then open vibe_checker.ipynb and configure:")
    print(f"   duckdb_path: {result['db_path']}")


if __name__ == "__main__":
    main()
