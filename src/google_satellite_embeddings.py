import argparse
import ee

def aggregate_satellite_embeddings(
    asset_ids: list[str],
    year: int,
    gcs_bucket: str,
    gcs_filename: str
):
    """
    Aggregates Google Satellite Embeddings for a collection of tiles and exports the result.

    Args:
        asset_ids: A list of GEE asset IDs for the input tile feature collections.
        year: The year of the satellite embeddings to use.
        gcs_bucket: The GCS bucket to export the results to.
        gcs_filename: The name of the output file in the GCS bucket.
    """
    try:
        ee.Initialize(project='demeterlabs-gee')
        print("Earth Engine initialized successfully.")
    except Exception as e:
        raise RuntimeError("Could not initialize Earth Engine. Please ensure you have authenticated.") from e

    # Load and merge all specified feature collections into one.
    print(f"Loading {len(asset_ids)} asset(s)...")
    initial_collection = ee.FeatureCollection(asset_ids[0])
    if len(asset_ids) > 1:
        for i in range(1, len(asset_ids)):
            initial_collection = initial_collection.merge(ee.FeatureCollection(asset_ids[i]))
    
    print(f"Total features to process: {initial_collection.size().getInfo()}")

    # Load the Google Satellite Embedding collection for the specified year.
    start_date = f'{year}-01-01'
    end_date = f'{year+1}-01-01'
    embedding_image = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                        .filterDate(start_date, end_date) \
                        .mosaic()

    # Create a list of all 64 band names (A00, A01, ..., A63).
    band_names = [f'A{i:02d}' for i in range(64)]
    embedding_image = embedding_image.select(band_names)

    # Use reduceRegions to get statistics for all features at once.
    # The output features will have the properties of the input features,
    # plus new properties for the output of the reducer (e.g., 'A00', 'A01').
    def per_feature_median(feature):
        """Attach the image's per‑band median to the input feature."""
        stats = embedding_image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=feature.geometry(),
            scale=10,          # Native resolution of the image
            tileScale=8,       # Bump this higher (e.g. 16) for very large polygons
            maxPixels=1e13     # Increase if you still hit maxPixels errors
        )
        # Combine the stats dictionary with the original properties
        return feature.set(stats)

    stats_collection = initial_collection.map(per_feature_median)


    # Define and start the export task to GCS.
    task_description = f'export_{gcs_filename.split(".")[0]}'
    gcs_path = f'gs://{gcs_bucket}/{gcs_filename}'
    
    print(f"\nStarting export task to {gcs_path}...")
    task = ee.batch.Export.table.toCloudStorage(
        collection=stats_collection,
        description=task_description,
        bucket=gcs_bucket,
        fileNamePrefix=gcs_filename.split('.')[0],
        fileFormat='GeoJSON'
    )
    task.start()
    
    print(f"Export task started successfully.")
    print(f"  Task ID: {task.id}")
    print(f"  Description: {task_description}")
    print("You can monitor the task status in the GEE Code Editor's 'Tasks' tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate Google Satellite Embeddings over specified GEE tile assets."
    )
    parser.add_argument(
        "--asset_ids",
        nargs='+',
        required=True,
        help="A list of GEE FeatureCollection asset IDs (e.g., 'projects/user/assets/tile1')."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="The year for which to get the satellite embeddings (e.g., 2023)."
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        required=True,
        help="The GCS bucket to export the final GeoJSON file to."
    )
    parser.add_argument(
        "--gcs_filename",
        type=str,
        required=True,
        help="The name for the output file in the GCS bucket (e.g., 'bali_embeddings_2023.geojson')."
    )

    args = parser.parse_args()

    aggregate_satellite_embeddings(
        asset_ids=args.asset_ids,
        year=args.year,
        gcs_bucket=args.gcs_bucket,
        gcs_filename=args.gcs_filename
    ) 