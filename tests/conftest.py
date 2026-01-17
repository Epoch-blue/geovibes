"""Shared fixtures for classification integration tests."""

import json

import duckdb
import numpy as np
import pytest


@pytest.fixture(scope="session")
def duckdb_with_embeddings():
    """
    Create an in-memory DuckDB with geo_embeddings table.

    Schema matches production:
    - id: BIGINT (row id)
    - tile_id: VARCHAR (MGRS-based identifier like "16SBH0001000010")
    - embedding: FLOAT[384] (or smaller for tests)
    - geometry: GEOMETRY (Point)

    Returns tuple of (connection, metadata dict with tile info).
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")

    np.random.seed(42)

    # Create embeddings with 64 dimensions for faster tests
    embedding_dim = 64
    n_samples = 100

    # Create samples: 30 positives (cluster around embedding +2), 70 negatives (around -2)
    # This makes embeddings linearly separable for classifier tests
    n_pos = 30
    n_neg = n_samples - n_pos

    embeddings_pos = np.random.randn(n_pos, embedding_dim) + 2.0
    embeddings_neg = np.random.randn(n_neg, embedding_dim) - 2.0
    embeddings = np.vstack([embeddings_pos, embeddings_neg]).astype(np.float32)

    # Create MGRS-style tile_ids and geometries
    # Using UTM zone 16 (Alabama area) for consistent CRS handling
    tile_ids = []
    geometries = []

    # Positives in a cluster around (−86.5, 33.0) - Birmingham, AL area
    # Spread across ~0.3 degrees (about 30km) for CV fold distribution
    for i in range(n_pos):
        lon = -86.5 + (i % 6) * 0.05  # Spread across ~0.3 degrees
        lat = 33.0 + (i // 6) * 0.05
        tile_id = f"16SBH{i:05d}{i:05d}"
        tile_ids.append(tile_id)
        geometries.append(f"POINT({lon} {lat})")

    # Negatives interleaved near positives for proper CV fold distribution
    # Each negative is placed 0.01 degrees away from a positive position
    for i in range(n_neg):
        # Place near the positive positions but offset slightly
        base_i = i % n_pos  # Cycle through positive positions
        lon = -86.5 + (base_i % 6) * 0.05 + 0.01  # Offset from positive
        lat = 33.0 + (base_i // 6) * 0.05 + 0.01
        tile_id = f"16SCH{i:05d}{i:05d}"
        tile_ids.append(tile_id)
        geometries.append(f"POINT({lon} {lat})")

    # Create table
    conn.execute(f"""
        CREATE TABLE geo_embeddings (
            id BIGINT PRIMARY KEY,
            tile_id VARCHAR,
            embedding FLOAT[{embedding_dim}],
            geometry GEOMETRY
        )
    """)

    # Insert data
    for i in range(n_samples):
        emb_list = embeddings[i].tolist()
        conn.execute(
            """
            INSERT INTO geo_embeddings (id, tile_id, embedding, geometry)
            VALUES (?, ?, ?, ST_GeomFromText(?))
            """,
            [i + 1, tile_ids[i], emb_list, geometries[i]],
        )

    metadata = {
        "n_samples": n_samples,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "embedding_dim": embedding_dim,
        "tile_ids": tile_ids,
        "pos_tile_ids": tile_ids[:n_pos],
        "neg_tile_ids": tile_ids[n_pos:],
    }

    yield conn, metadata

    conn.close()


@pytest.fixture
def duckdb_connection(duckdb_with_embeddings):
    """Get just the connection from the session fixture."""
    conn, _ = duckdb_with_embeddings
    return conn


@pytest.fixture
def duckdb_metadata(duckdb_with_embeddings):
    """Get just the metadata from the session fixture."""
    _, metadata = duckdb_with_embeddings
    return metadata


@pytest.fixture
def training_geojson_with_tile_ids(duckdb_metadata, tmp_path):
    """
    Create a GeoJSON file with tile_id properties matching DuckDB.

    Returns path to the GeoJSON file.
    """
    pos_tile_ids = duckdb_metadata["pos_tile_ids"][:10]  # 10 positives
    neg_tile_ids = duckdb_metadata["neg_tile_ids"][:10]  # 10 negatives

    features = []

    for tile_id in pos_tile_ids:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, 0],
                },  # Geometry not used for tile_id match
                "properties": {
                    "tile_id": tile_id,
                    "label": 1,
                    "class": "geovibes_pos",
                },
            }
        )

    for tile_id in neg_tile_ids:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {
                    "tile_id": tile_id,
                    "label": 0,
                    "class": "geovibes_neg",
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    path = tmp_path / "training_with_tile_ids.geojson"
    path.write_text(json.dumps(geojson))

    return str(path)


@pytest.fixture
def training_geojson_with_db_ids(duckdb_metadata, tmp_path):
    """
    Create a GeoJSON file with database row IDs (id property) instead of tile_ids.

    Returns path to the GeoJSON file.
    """
    n_pos = 10
    n_neg = 10

    features = []

    # Positives have ids 1..n_pos
    for i in range(1, n_pos + 1):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {
                    "id": i,  # Database row ID
                    "label": 1,
                    "class": "geovibes_pos",
                },
            }
        )

    # Negatives have ids starting after positives
    n_total_pos = duckdb_metadata["n_positives"]
    for i in range(n_total_pos + 1, n_total_pos + n_neg + 1):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {
                    "id": i,
                    "label": 0,
                    "class": "geovibes_neg",
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    path = tmp_path / "training_with_db_ids.geojson"
    path.write_text(json.dumps(geojson))

    return str(path)


@pytest.fixture
def training_geojson_with_points(tmp_path):
    """
    Create a GeoJSON file with point geometries for spatial matching.

    Returns path to the GeoJSON file.
    """
    features = []

    # Positives spread across the Birmingham cluster area
    for i in range(5):
        lon = -86.5 + i * 0.05
        lat = 33.0 + i * 0.05
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "label": 1,
                    "class": "geovibes_pos",
                },
            }
        )

    # Negatives interleaved near positives (offset by 0.01 degrees)
    for i in range(5):
        lon = -86.5 + i * 0.05 + 0.01
        lat = 33.0 + i * 0.05 + 0.01
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "label": 0,
                    "class": "geovibes_neg",
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    path = tmp_path / "training_with_points.geojson"
    path.write_text(json.dumps(geojson))

    return str(path)


@pytest.fixture
def trained_classifier(duckdb_connection, duckdb_metadata):
    """
    Return a trained EmbeddingClassifier using embeddings from DuckDB.

    The classifier is trained on all embeddings (30 pos, 70 neg) with their
    natural separation (pos centered at +2, neg at -2).
    """
    from geovibes.classification.classifier import EmbeddingClassifier

    # Fetch all embeddings
    result = duckdb_connection.execute("""
        SELECT id, CAST(embedding AS FLOAT[]) as embedding
        FROM geo_embeddings
        ORDER BY id
    """).fetchdf()

    embeddings = np.vstack(result["embedding"].values).astype(np.float32)

    # Labels based on ID: 1-30 are positives (label=1), 31-100 are negatives (label=0)
    n_pos = duckdb_metadata["n_positives"]
    labels = np.array([1] * n_pos + [0] * (len(embeddings) - n_pos), dtype=np.int32)

    classifier = EmbeddingClassifier(n_estimators=20, max_depth=4, random_state=42)
    classifier.fit(embeddings, labels)

    return classifier
