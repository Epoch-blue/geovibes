from geovibes.ui.utils import (
    get_database_centroid,
    list_databases_in_directory,
    log_to_file,
    prepare_ids_for_query,
)


def test_prepare_ids_for_query_casts_to_str():
    ids = prepare_ids_for_query([1, "2", 3.5])
    assert ids == ["1", "2", "3.5"]


def test_log_to_file_appends(tmp_path):
    log_path = tmp_path / "log.txt"
    log_to_file("first", logfile=str(log_path))
    log_to_file("second", logfile=str(log_path))

    text = log_path.read_text()
    assert "first" in text
    assert "second" in text


def test_list_databases_in_directory_filters_indices(tmp_path):
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    db_one = db_dir / "sample.db"
    db_one.write_text("data")
    index_one = db_dir / "sample.index"
    index_one.write_text("idx")

    db_two = db_dir / "other.db"
    db_two.write_text("data")
    (db_dir / "other.index").write_text("idx")
    (db_dir / "other_extra.index").write_text("idx")

    databases = list_databases_in_directory(str(db_dir))

    assert databases == [{"db_path": str(db_one), "faiss_path": str(index_one)}]


def test_get_database_centroid_defaults_to_origin():
    lat, lon = get_database_centroid(None)
    assert (lat, lon) == (0.0, 0.0)
