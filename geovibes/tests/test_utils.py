from geovibes.ui.utils import parse_env_flag, prepare_ids_for_query


def test_parse_env_flag_truthy():
    for value in ["1", "true", "YES", "On"]:
        assert parse_env_flag(value) is True


def test_parse_env_flag_falsey():
    for value in ["0", "false", "NO", "off"]:
        assert parse_env_flag(value) is False


def test_parse_env_flag_unknown():
    assert parse_env_flag("maybe") is None
    assert parse_env_flag(None) is None


def test_prepare_ids_for_query_casts_to_str():
    ids = prepare_ids_for_query([1, "2", 3.5])
    assert ids == ["1", "2", "3.5"]
