from geovibes.ui.app import GeoVibes


class StubDatasetManager:
    def __init__(self, payload):
        self.payload = payload

    def save_dataset(self):
        return self.payload


def test_handle_save_dataset_reports_geojson():
    payload = {"geojson": "labeled_dataset_20250101.geojson"}
    captured = {}

    vibes = GeoVibes.__new__(GeoVibes)
    vibes.dataset_manager = StubDatasetManager(payload)
    vibes._show_operation_status = lambda message: captured.setdefault(
        "message", message
    )

    vibes._handle_save_dataset()

    assert payload["geojson"] in captured["message"]
    assert "labels" not in captured["message"]


def test_handle_save_dataset_handles_empty_payload():
    captured = {}

    vibes = GeoVibes.__new__(GeoVibes)
    vibes.dataset_manager = StubDatasetManager(None)
    vibes._show_operation_status = lambda message: captured.setdefault(
        "message", message
    )

    vibes._handle_save_dataset()

    assert "Nothing to save" in captured["message"]
