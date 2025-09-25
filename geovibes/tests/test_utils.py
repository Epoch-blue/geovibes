from geovibes.ui.utils import prepare_ids_for_query


def test_prepare_ids_for_query_casts_to_str():
    ids = prepare_ids_for_query([1, "2", 3.5])
    assert ids == ["1", "2", "3.5"]
