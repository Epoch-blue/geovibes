
import duckdb
import faiss
import numpy as np
import pandas as pd

# Read the Parquet file
df = pd.read_parquet("./alphaearth_output/embeddings.parquet")
print(f"Loaded {len(df)} rows")

# Convert embeddings to tuples first (required for deduplication)
df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x))

# Remove coordinate-based duplicates (same lat/lon across different tiles)
# This fixes the search artifacts caused by sampling the same location multiple times
print(f"Before deduplication: {len(df)} rows, {df['tile_id'].nunique()} unique tiles")
df_dedup = df.drop_duplicates(subset=['lon', 'lat'], keep='first')
print(f"After deduplication: {len(df_dedup)} rows, {df_dedup['tile_id'].nunique()} unique tiles")

# Create DuckDB database
conn = duckdb.connect("./local_databases/alphaearth_riau_metadata_clean.db")

# Load spatial extension
conn.execute("INSTALL spatial;")
conn.execute("LOAD spatial;")

# Convert embeddings to tuples of regular floats (like Alabama database)
df_dedup['embedding'] = df_dedup['embedding_tuple'].apply(lambda x: tuple(float(val) for val in x))

# Drop existing table if it exists
conn.execute("DROP TABLE IF EXISTS geo_embeddings")

# Create table with embeddings in the same format as Alabama
conn.execute("""
CREATE TABLE geo_embeddings AS 
SELECT 
    CAST(id AS VARCHAR) as id,
    tile_id,
    CAST(embedding AS FLOAT[64]) as embedding,
    ST_GeomFromText('POINT(' || lon || ' ' || lat || ')') as geometry
FROM df_dedup
""")

print("Database created successfully")

# Create FAISS index (adjusted for small dataset)
embeddings = np.array(df_dedup["embedding"].tolist())
print(f"Creating FAISS index with {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}")

# Use IndexIVFPQ like Alabama (more efficient for large datasets)
# Now we have enough points after removing true duplicates
quantizer = faiss.IndexFlatL2(embeddings.shape[1])
index = faiss.IndexIVFPQ(quantizer, embeddings.shape[1], 4096, 64, 8)
index.train(embeddings.astype("float32"))
index.add(embeddings.astype("float32"))
faiss.write_index(index, "./local_databases/alphaearth_riau_faiss_clean.index")
print("FAISS index created successfully")
