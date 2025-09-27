from geovibes.ui_config.constants import UIConstants
from matplotlib import colormaps as mpl_colormaps
from geovibes.ui.utils import (
    get_database_centroid,
    infer_tile_spec_from_name,
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


def test_infer_tile_spec_from_name_parses_suffix():
    spec = infer_tile_spec_from_name(
        "alabama_google_satellite_embeddings_v1_2024_2025_25_0_10_metadata.db"
    )
    assert spec == {
        "tile_size_px": 25,
        "tile_overlap_px": 0,
        "meters_per_pixel": 10.0,
    }


def test_infer_tile_spec_from_name_handles_missing():
    assert infer_tile_spec_from_name("no_tile_info.db") is None


def test_distance_to_color_viridis_gradient():
    low = UIConstants.distance_to_color(1.0, 0.0, 1.0)
    high = UIConstants.distance_to_color(0.0, 0.0, 1.0)

    cmap = mpl_colormaps.get_cmap(UIConstants.SEARCH_COLORMAP)
    expected_high = cmap(1.0)
    expected_low = cmap(0.0)
    expected_high_hex = "#{:02x}{:02x}{:02x}".format(
        int(expected_high[0] * 255),
        int(expected_high[1] * 255),
        int(expected_high[2] * 255),
    )
    expected_low_hex = "#{:02x}{:02x}{:02x}".format(
        int(expected_low[0] * 255),
        int(expected_low[1] * 255),
        int(expected_low[2] * 255),
    )

    assert high == expected_high_hex
    assert low == expected_low_hex


def test_similarity_colorbar_data_uri():
    data_uri = UIConstants.similarity_colorbar_data_uri()
    assert isinstance(data_uri, str)
    assert len(data_uri) > 0
