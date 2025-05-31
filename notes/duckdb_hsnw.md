# Implementing HNSW Index in DuckDB for GeoParquet Embeddings

This document outlines the steps to create a DuckDB database with Hybrid SpatiAl and Nearest Neighbor Search (HSNW-like) capabilities for querying GeoParquet files containing embeddings. The goal is to integrate this with a Python-based UI (`ui.py`) for interactive spatial selection and similarity searches.

## I. DuckDB Setup and Data Ingestion

1.  **Install Necessary DuckDB Extensions:**
    *   Ensure DuckDB is installed.
    *   Within DuckDB, install and load the `spatial` and `vss` (Vector Similarity Search) extensions:
        ```sql
        INSTALL spatial;
        LOAD spatial;
        INSTALL vss;
        LOAD vss;
        ```
    *   These extensions provide GIS functions, R-tree spatial indexing, and HNSW indexing for vector similarity.

2.  **Data Source:**
    *   GeoParquet files containing at least:
        *   `id`: A unique identifier for each record.
        *   `embedding`: A numerical array (e.g., `FLOAT[384]`) representing the embedding vector.
        *   A geometry column representing the spatial footprint of the data.

3.  **Create a DuckDB Table:**
    *   Define a table schema to store the data from the GeoParquet files.
    *   Load data using `read_parquet`. Ensure correct data type mapping, especially for embeddings (e.g., `FLOAT[]`) and geometries.

    ```sql
    CREATE TABLE geo_embeddings (
        id VARCHAR PRIMARY KEY,    -- Or appropriate ID type
        embedding FLOAT[384],      -- Adjust size as per your embeddings
        geometry GEOMETRY          -- DuckDB's geometry type
    );
    ```

4.  **Load Data into the Table:**
    *   Iterate through your GeoParquet files and insert data. If geometries are in WKB or GeoJSON format in Parquet, use conversion functions.
    ```sql
    -- Example for loading from a single Parquet file, adapt for multiple files
    -- Assuming 'geometry_wkb' is the column in Parquet with WKB geometry
    INSERT INTO geo_embeddings (id, embedding, geometry)
    SELECT
        id_column AS id,
        embedding_column AS embedding,
        ST_GeomFromWKB(geometry_wkb_column) AS geometry
    FROM read_parquet('path/to/your/file.parquet');

    -- For multiple files, you can use a list in read_parquet:
    -- FROM read_parquet(['file1.parquet', 'file2.parquet', ...]);
    ```

## II. Index Creation

1.  **Create an HNSW Index on the Embedding Column:**
    *   Use the `vss` extension to build an HNSW index for fast similarity searches.
    ```sql
    -- The exact syntax for PRAGMA settings might depend on the VSS extension version
    -- PRAGMA hnsw_enable_experimental_index = true; -- If needed for older versions
    CREATE INDEX emb_hnsw_idx ON geo_embeddings USING HNSW (embedding);
    ```
    *   Consult the DuckDB `vss` extension documentation for specific HNSW parameters (e.g., `m`, `ef_construction`, `ef_search`) if customization is needed.

2.  **Create a Spatial Index on the Geometry Column:**
    *   Use the `spatial` extension to create an R-tree index for efficient spatial queries.
    ```sql
    CREATE INDEX geom_rtree_idx ON geo_embeddings USING RTREE (geometry);
    ```

## III. Querying Strategy

1.  **Spatial Query (User Selects a Tile on Map):**
    *   Input: Coordinates (longitude, latitude) from the map click.
    *   Output: `id`, `embedding`, and `geometry` of the tile containing/nearest to the point.
    *   Method: Use `ST_Point` to create a point geometry and spatial functions like `ST_Contains`, `ST_Intersects`, or `ST_Distance` (with `ORDER BY` and `LIMIT 1` for nearest).
    ```sql
    -- Example: Find the tile containing the clicked point
    -- Python would format lon, lat into this query
    -- SELECT id, embedding, geometry
    -- FROM geo_embeddings
    -- WHERE ST_Contains(geometry, ST_Point(lon, lat))
    -- LIMIT 1;

    -- Example: Find the nearest tile to the clicked point
    SELECT id, embedding, geometry
    FROM geo_embeddings
    ORDER BY ST_Distance(geometry, ST_Point(?, ?)) -- Use placeholder for lon, lat
    LIMIT 1;
    ```

2.  **Embedding Retrieval:**
    *   The spatial query directly returns the `embedding` vector for the selected tile.

3.  **Similarity Search (N Nearest Neighbors):**
    *   Input: The `embedding` vector from the selected tile, and `N` (number of neighbors).
    *   Output: `id`s, `embedding`s, and `geometry` (locations) of the `N` most similar items.
    *   Method: Use the HNSW index via distance functions provided by the `vss` extension. The common operator is `<->` for L2 distance, but check documentation for cosine similarity if needed (`<=>`).
    ```sql
    -- 'target_embedding_array' is a placeholder for the actual array
    SELECT id, embedding, geometry
    FROM geo_embeddings
    ORDER BY embedding <-> target_embedding_array -- L2 distance
    LIMIT N; -- N is the desired number of neighbors
    ```
    *   Ensure `target_embedding_array` is passed in the correct format (e.g., `ARRAY[0.1, 0.2, ...]`).

## IV. Integration with `ui.py`

1.  **Modify `GeoLabeler` (or relevant class in `ui.py`):**
    *   **Database Connection:** Establish a connection to the DuckDB database containing the `geo_embeddings` table and its indexes.
    *   **Remove Annoy:** Phase out reliance on the Annoy index if it's being replaced.
    *   **Data Source:** The `gdf` (GeoDataFrame) used for centroids might still be useful for initial map population or quick visualization, but primary queries for selection and similarity will hit DuckDB.

2.  **Update Map Interaction Logic (e.g., `label_point`):**
    *   On map click, get coordinates.
    *   Execute the spatial query (III.1) against DuckDB to retrieve the selected tile's `id` and `embedding`.
    *   Store the retrieved `embedding` for subsequent similarity search.

3.  **Implement Similarity Search UI Trigger:**
    *   Add a button or control (e.g., "Find Similar Tiles").
    *   On activation, use the stored `embedding` to execute the HNSW similarity search query (III.3) against DuckDB.
    *   Retrieve the results (IDs, embeddings, geometries of neighbors).

4.  **Visualize Results:**
    *   Display the N nearest neighbors on the map using their geometries.
    *   The `update_layer` method or similar in `ui.py` can be adapted to show these new points/polygons.

5.  **Data Preparation Script (`embeddings_to_annoy.py` or new):**
    *   Modify the existing data preparation script or create a new one.
    *   This script will be responsible for:
        *   Connecting to DuckDB.
        *   Executing `INSTALL`/`LOAD` for `spatial` and `vss` extensions.
        *   Creating the `geo_embeddings` table (I.3).
        *   Loading data from GeoParquet files into the table (I.4).
        *   Building the HNSW index (II.1).
        *   Building the spatial R-tree index (II.2).
    *   This script replaces the Annoy index generation.

## V. Considerations

*   **Performance Tuning:** Experiment with HNSW index parameters (`ef_construction`, `m`, `ef_search`) for optimal balance between build time, index size, and query speed/accuracy.
*   **Distance Metrics:** Ensure the distance metric used in HNSW (default usually L2) matches the metric your embeddings are designed for (e.g., cosine similarity). The `vss` extension may offer different metrics.
*   **Scalability:** DuckDB runs in-process. For very large datasets exceeding memory, consider strategies or if DuckDB's out-of-core capabilities are sufficient.
*   **Error Handling and UI Feedback:** Implement robust error handling for database operations and provide clear feedback to the user in the UI.
*   **DuckDB `vss` Extension Documentation:** Refer to the official DuckDB `vss` extension documentation for the most up-to-date syntax, features, and best practices, as it's an actively developed area.

This plan provides a roadmap. Detailed implementation will require referencing the specific documentation for DuckDB's `spatial` and `vss` extensions. 