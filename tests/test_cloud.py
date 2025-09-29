from geovibes.database import cloud


def test_get_cloud_protocol_detects_supported_prefixes():
    assert cloud.get_cloud_protocol("s3://bucket") == "s3"
    assert cloud.get_cloud_protocol("gs://bucket") == "gs"
    assert cloud.get_cloud_protocol("/local/path") is None


def test_list_cloud_parquet_files_s3(monkeypatch):
    captured = {}

    class DummyFS:
        def glob(self, pattern):
            captured["pattern"] = pattern
            return ["bucket/key1.parquet", "bucket/key2.parquet"]

    def fake_filesystem(protocol, **kwargs):
        captured["protocol"] = protocol
        captured["kwargs"] = kwargs
        return DummyFS()

    monkeypatch.setattr(cloud.fsspec, "filesystem", fake_filesystem)
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://example.com")
    monkeypatch.setenv("GEOVIBES_S3_USE_ANON", "false")

    files = cloud.list_cloud_parquet_files("s3://bucket")

    assert captured["protocol"] == "s3"
    assert captured["pattern"].endswith("*.parquet")
    assert files == ["s3://bucket/key1.parquet", "s3://bucket/key2.parquet"]


def test_list_cloud_parquet_files_gs(monkeypatch):
    class DummyFS:
        def glob(self, pattern):
            return ["bucket/key.parquet"]

    def fake_filesystem(protocol):
        assert protocol == "gs"
        return DummyFS()

    monkeypatch.setattr(cloud.fsspec, "filesystem", fake_filesystem)

    files = cloud.list_cloud_parquet_files("gs://bucket")

    assert files == ["gs://bucket/key.parquet"]


def test_download_cloud_files_filters_none(monkeypatch):
    calls = []

    def fake_delayed(fn):
        def wrapper(*args, **kwargs):
            return lambda: fn(*args, **kwargs)

        return wrapper

    def fake_parallel(*args, **kwargs):
        def runner(jobs):
            return [job() for job in jobs]

        return runner

    def fake_download(path, temp_dir):
        calls.append((path, temp_dir))
        return path if "keep" in path else None

    monkeypatch.setattr(cloud, "delayed", fake_delayed)
    monkeypatch.setattr(cloud, "Parallel", fake_parallel)
    monkeypatch.setattr(cloud, "_download_single_cloud_file", fake_download)

    result = cloud.download_cloud_files([
        "keep_file",
        "drop_file",
    ], "tmp")

    assert calls == [("keep_file", "tmp"), ("drop_file", "tmp")]
    assert result == ["keep_file"]


def test_find_embedding_files_for_mgrs_ids_local(tmp_path):
    dir_path = tmp_path / "embeddings"
    dir_path.mkdir()
    file_a = dir_path / "abcd123_tile.parquet"
    file_b = dir_path / "abcd123_extra.parquet"
    file_c = dir_path / "wxyz999.parquet"
    for file in (file_a, file_b, file_c):
        file.write_text("data")

    matches = cloud.find_embedding_files_for_mgrs_ids([
        "abcd123",
        "missing",
    ], str(dir_path))

    assert set(matches) == {str(file_a), str(file_b)}
