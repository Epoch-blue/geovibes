#!/usr/bin/env python3
"""
Extract AlphaEarth embeddings for palm oil mills in Sumatra.

This script:
1. Loads palm oil mill locations from indonesia-palm-oil-mills.geojson
2. Uses Overture Maps divisions data to get Sumatra province boundaries
3. Filters mills to only those within Sumatra
4. Extracts AlphaEarth embeddings for each mill location
"""

import argparse
from pathlib import Path
from typing import List, Optional

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb

SUMATRA_PROVINCES = [
    "Aceh",
    "Sumatera Utara",
    "Sumatera Barat",
    "Riau",
    "Jambi",
    "Sumatera Selatan",
    "Bengkulu",
    "Lampung",
]

OVERTURE_DIVISIONS_PATH = "s3://overturemaps-us-west-2/release/2024-12-18.0/theme=divisions/type=division_area/*"


def get_sumatra_boundary_from_overture(
    conn: duckdb.DuckDBPyConnection, provinces: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Query Overture Maps divisions to get Sumatra province boundaries.

    Args:
        conn: DuckDB connection with spatial and httpfs extensions loaded
        provinces: List of province names to include (defaults to SUMATRA_PROVINCES)

    Returns:
        GeoDataFrame with Sumatra province boundaries
    """
    provinces = provinces or SUMATRA_PROVINCES
    province_list = ", ".join([f"'{p}'" for p in provinces])

    query = f"""
    SELECT 
        names.primary AS name,
        subtype,
        country,
        ST_GeomFromWKB(geometry) as geometry
    FROM read_parquet('{OVERTURE_DIVISIONS_PATH}', filename=true, hive_partitioning=true)
    WHERE country = 'ID'
      AND subtype = 'region'
      AND names.primary IN ({province_list})
    """

    print("🔍 Querying Overture Maps for Sumatra provinces...")
    result = conn.execute(query).fetchdf()

    if result.empty:
        print("⚠️  No provinces found with exact name match, trying fuzzy match...")
        fuzzy_query = f"""
        SELECT 
            names.primary AS name,
            subtype,
            country,
            ST_GeomFromWKB(geometry) as geometry
        FROM read_parquet('{OVERTURE_DIVISIONS_PATH}', filename=true, hive_partitioning=true)
        WHERE country = 'ID'
          AND subtype = 'region'
          AND (
            names.primary ILIKE '%sumatera%' 
            OR names.primary ILIKE '%sumatra%'
            OR names.primary ILIKE '%aceh%'
            OR names.primary ILIKE '%riau%'
            OR names.primary ILIKE '%jambi%'
            OR names.primary ILIKE '%bengkulu%'
            OR names.primary ILIKE '%lampung%'
          )
        """
        result = conn.execute(fuzzy_query).fetchdf()

    if result.empty:
        raise ValueError("Could not find Sumatra provinces in Overture Maps data")

    geometries = [
        wkb.loads(g) if isinstance(g, bytes) else g for g in result["geometry"]
    ]
    gdf = gpd.GeoDataFrame(
        result.drop(columns=["geometry"]), geometry=geometries, crs="EPSG:4326"
    )

    print(f"✅ Found {len(gdf)} Sumatra provinces:")
    for name in gdf["name"].values:
        print(f"   - {name}")

    return gdf


def load_palm_oil_mills(geojson_path: str) -> gpd.GeoDataFrame:
    """
    Load palm oil mill locations from GeoJSON.

    Args:
        geojson_path: Path to indonesia-palm-oil-mills.geojson

    Returns:
        GeoDataFrame with mill locations
    """
    print(f"📂 Loading palm oil mills from {geojson_path}...")
    mills = gpd.read_file(geojson_path)

    if mills.crs is None:
        mills = mills.set_crs("EPSG:4326")
    elif mills.crs.to_string() != "EPSG:4326":
        mills = mills.to_crs("EPSG:4326")

    print(f"✅ Loaded {len(mills)} palm oil mills")
    return mills


def filter_mills_to_sumatra(
    mills: gpd.GeoDataFrame, sumatra_boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Filter mills to only those within Sumatra provinces.

    Args:
        mills: GeoDataFrame with all mill locations
        sumatra_boundary: GeoDataFrame with Sumatra province polygons

    Returns:
        GeoDataFrame with mills in Sumatra only
    """
    print("🔄 Filtering mills to Sumatra...")

    sumatra_union = sumatra_boundary.union_all()

    mills_sumatra = mills[mills.geometry.within(sumatra_union)].copy()

    print(f"✅ Found {len(mills_sumatra)} mills in Sumatra (out of {len(mills)} total)")

    if len(mills_sumatra) > 0:
        province_counts = mills_sumatra["province_name"].value_counts()
        print("\n📊 Mills by province:")
        for province, count in province_counts.items():
            print(f"   - {province}: {count}")

    return mills_sumatra


def extract_embeddings_for_mills(
    mills: gpd.GeoDataFrame,
    output_dir: str,
    db_name: str,
    scale: int = 10,
    buffer_meters: int = 100,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    service_account_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract AlphaEarth embeddings for mill locations.

    Args:
        mills: GeoDataFrame with mill locations
        output_dir: Directory for output files
        db_name: Name for the output database
        scale: Resolution in meters
        buffer_meters: Buffer around each point to sample
        start_date: Start date for imagery
        end_date: End date for imagery
        service_account_key: Optional GEE service account key path

    Returns:
        DataFrame with mill properties and embeddings
    """
    from geovibes.database.xee_embeddings import (
        initialize_earth_engine,
    )
    import ee

    print("\n" + "=" * 70)
    print("🚀 Extracting embeddings for mill locations")
    print("=" * 70)

    initialize_earth_engine(service_account_key)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results = []
    total_mills = len(mills)

    for idx, row in mills.iterrows():
        mill_num = len(results) + 1
        lon, lat = row.geometry.x, row.geometry.y
        mill_name = row.get("mill_name", row.get("name", f"Mill_{idx}"))

        print(f"\n[{mill_num}/{total_mills}] Processing: {mill_name}")
        print(f"   Location: ({lat:.6f}, {lon:.6f})")

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_meters).bounds()

        alphaearth_collection = ee.ImageCollection(
            "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        )
        alphaearth_filtered = alphaearth_collection.filterDate(
            start_date, end_date
        ).filterBounds(region)
        alphaearth_image = alphaearth_filtered.mosaic()

        sample = alphaearth_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e6
        )

        sample_dict = sample.getInfo()

        embedding_values = []
        for i in range(64):
            band_name = f"A{i:02d}"
            value = sample_dict.get(band_name)
            if value is not None:
                embedding_values.append(float(value))
            else:
                embedding_values.append(np.nan)

        embedding_array = np.array(embedding_values, dtype=np.float32)

        if np.isnan(embedding_array).all():
            print("   ⚠️  No valid embedding data for this location")
            continue

        record = {
            "mill_name": mill_name,
            "company": row.get("company", ""),
            "group": row.get("group", ""),
            "province_name": row.get("province_name", ""),
            "kabupaten_name": row.get("kabupaten_name", ""),
            "lat": lat,
            "lon": lon,
            "embedding": embedding_array,
            "capacity_tonnes_ffb_hour": row.get("capacity_tonnes_ffb_hour"),
            "active": row.get("active"),
            "trase_code": row.get("trase_code", ""),
            "uml_id": row.get("uml_id", ""),
        }

        results.append(record)
        print(
            f"   ✅ Extracted embedding ({np.sum(~np.isnan(embedding_array))}/64 bands valid)"
        )

    results_df = pd.DataFrame(results)

    parquet_path = output_path / f"{db_name}_mill_embeddings.parquet"
    results_df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Saved embeddings to: {parquet_path}")

    geojson_path = output_path / f"{db_name}_mills.geojson"
    results_gdf = gpd.GeoDataFrame(
        results_df.drop(columns=["embedding"]),
        geometry=gpd.points_from_xy(results_df["lon"], results_df["lat"]),
        crs="EPSG:4326",
    )
    results_gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"💾 Saved mill locations to: {geojson_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract AlphaEarth embeddings for palm oil mills in Sumatra"
    )

    parser.add_argument(
        "--mills-file",
        default="geometries/indonesia-palm-oil-mills.geojson",
        help="Path to palm oil mills GeoJSON file",
    )

    parser.add_argument(
        "--output-dir",
        default="./sumatra_mill_embeddings",
        help="Output directory for embeddings",
    )

    parser.add_argument(
        "--db-name", default="sumatra_mills", help="Name prefix for output files"
    )

    parser.add_argument(
        "--scale", type=int, default=10, help="Resolution in meters (default: 10)"
    )

    parser.add_argument(
        "--buffer",
        type=int,
        default=100,
        help="Buffer around each point in meters (default: 100)",
    )

    parser.add_argument(
        "--start-date", default="2024-01-01", help="Start date for imagery (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", default="2024-12-31", help="End date for imagery (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--service-account-key", help="Path to GCP service account key JSON"
    )

    parser.add_argument(
        "--skip-overture",
        action="store_true",
        help="Skip Overture query and filter by province_name column instead",
    )

    parser.add_argument(
        "--limit", type=int, help="Limit number of mills to process (for testing)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🌴 Sumatra Palm Oil Mill Embedding Extraction")
    print("=" * 70)

    mills = load_palm_oil_mills(args.mills_file)

    if args.skip_overture:
        print("\n📋 Using province_name column to filter (--skip-overture)")
        sumatra_province_patterns = [
            "ACEH",
            "SUMATERA",
            "RIAU",
            "JAMBI",
            "BENGKULU",
            "LAMPUNG",
        ]
        mask = (
            mills["province_name"]
            .str.upper()
            .str.contains("|".join(sumatra_province_patterns), na=False)
        )
        mills_sumatra = mills[mask].copy()
        print(f"✅ Found {len(mills_sumatra)} mills in Sumatra provinces")
    else:
        print("\n🌐 Setting up DuckDB with spatial extensions...")
        conn = duckdb.connect()
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("SET s3_region='us-west-2';")

        sumatra_boundary = get_sumatra_boundary_from_overture(conn)
        mills_sumatra = filter_mills_to_sumatra(mills, sumatra_boundary)

        conn.close()

    if len(mills_sumatra) == 0:
        print("❌ No mills found in Sumatra!")
        return

    if args.limit:
        mills_sumatra = mills_sumatra.head(args.limit)
        print(f"\n⚠️  Limited to {args.limit} mills for testing")

    results = extract_embeddings_for_mills(
        mills=mills_sumatra,
        output_dir=args.output_dir,
        db_name=args.db_name,
        scale=args.scale,
        buffer_meters=args.buffer,
        start_date=args.start_date,
        end_date=args.end_date,
        service_account_key=args.service_account_key,
    )

    print("\n" + "=" * 70)
    print("✅ Extraction Complete!")
    print("=" * 70)
    print(f"📊 Total mills processed: {len(results)}")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
