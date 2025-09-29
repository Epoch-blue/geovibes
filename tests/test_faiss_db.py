import pandas as pd
import pytest

from geovibes.database import faiss_db


def test_ensure_duckdb_extension_executes_queries():
    executed = []

    class DummyConn:
        def execute(self, sql):
            executed.append(sql)

    conn = DummyConn()

    faiss_db.ensure_duckdb_extension(conn, "httpfs")

    assert executed == ["INSTALL httpfs;", "LOAD httpfs;"]


def test_infer_embedding_dim_from_file(tmp_path):
    file_path = tmp_path / "data.parquet"
    df = pd.DataFrame({"embedding": [[0.1, 0.2, 0.3, 0.4]]})
    df.to_parquet(file_path)

    dim = faiss_db.infer_embedding_dim_from_file(str(file_path), "embedding")

    assert dim == 4


def test_infer_embedding_dim_from_file_errors_on_null(tmp_path):
    file_path = tmp_path / "data.parquet"
    df = pd.DataFrame({"embedding": [None]})
    df.to_parquet(file_path)

    with pytest.raises(faiss_db.IngestParquetError):
        faiss_db.infer_embedding_dim_from_file(str(file_path), "embedding")
