import geopandas as gpd
import shapely.geometry
from shapely.geometry import shape

import faiss

import geovibes.ui.data_manager as data_manager_module
from geovibes.ui.data_manager import DataManager


def test_update_ee_boundary_uses_current_geometry(monkeypatch):
    dm = DataManager.__new__(DataManager)
    dm.ee_available = True
    dm.verbose = False

    polygon_a = shapely.geometry.box(0, 0, 1, 1)
    polygon_b = shapely.geometry.box(1, 1, 2, 2)
    gdf_a = gpd.GeoDataFrame({"geometry": [polygon_a]})
    gdf_b = gpd.GeoDataFrame({"geometry": [polygon_b]})

    geometries = {
        "first.geojson": gdf_a,
        "second.geojson": gdf_b,
    }

    monkeypatch.setattr(data_manager_module.gpd, "read_file", lambda path: geometries[path])

    recorded = []

    def fake_geometry(mapping):
        recorded.append(mapping)
        return mapping

    monkeypatch.setattr(data_manager_module.ee, "Geometry", fake_geometry)

    dm.effective_boundary_path = "first.geojson"
    dm._update_ee_boundary()
    first_shape = shape(dm.ee_boundary)
    assert first_shape.equals(polygon_a)

    dm.effective_boundary_path = "second.geojson"
    dm._update_ee_boundary()
    second_shape = shape(dm.ee_boundary)
    assert second_shape.equals(polygon_b)
    assert len(recorded) == 2




def test_update_ee_boundary_skips_when_disabled(monkeypatch):
    dm = DataManager.__new__(DataManager)
    dm.ee_available = False
    dm.effective_boundary_path = "first.geojson"
    dm.ee_boundary = "existing"

    dm._update_ee_boundary()

    assert dm.ee_boundary == "existing"


def test_switch_database_invokes_boundary_refresh(monkeypatch):
    dm = DataManager.__new__(DataManager)
    dm.verbose = False
    dm.ee_available = True
    dm.ee_boundary = None
    dm.current_database_path = "db1"
    dm.current_database_info = {"faiss_path": "index1", "geometry_path": "geom1"}
    dm.current_faiss_path = "index1"
    dm.current_geometry_path = "geom1"
    dm.database_info_by_path = {
        "db2": {"faiss_path": "index2", "geometry_path": "geom2"}
    }
    dm._owns_connection = False
    dm.duckdb_connection = None

    monkeypatch.setattr(DataManager, "_connect_duckdb", lambda self, path: "conn")
    monkeypatch.setattr(DataManager, "_apply_duckdb_settings", lambda self, path: None)
    monkeypatch.setattr(DataManager, "_detect_embedding_dim", lambda self: 1)
    monkeypatch.setattr(DataManager, "_warm_up_gcs_database", lambda self: None)
    monkeypatch.setattr(faiss, "read_index", lambda path: f"loaded:{path}")

    def fake_setup_boundary_and_center(self):
        return "geom2", (0.0, 0.0)

    monkeypatch.setattr(
        DataManager, "_setup_boundary_and_center", fake_setup_boundary_and_center
    )

    calls = []

    def fake_update(self):
        calls.append(True)

    monkeypatch.setattr(DataManager, "_update_ee_boundary", fake_update)

    dm.switch_database("db2")

    assert dm.current_database_path == "db2"
    assert dm.current_geometry_path == "geom2"
    assert dm.effective_boundary_path == "geom2"
    assert calls, "_update_ee_boundary was not invoked"
