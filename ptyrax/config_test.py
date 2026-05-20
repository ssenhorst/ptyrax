import shutil
import urllib.request

from pytest import MonkeyPatch

from ptyrax.config import load_yaml_config, resolve_config_paths


def test_import_config() -> None:
    __import__("ptyrax.config")


def test_resolve_config_paths_http_url(tmp_path: str, monkeypatch: MonkeyPatch) -> None:
    source_file = tmp_path / "remote.yaml"
    source_file.write_text("__main__:\n  seed: 7\n")

    def fake_urlretrieve(url: str, filename: str, *args, **kwargs):
        shutil.copyfile(str(source_file), filename)
        return (filename, None)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    resolved_paths, temp_dir = resolve_config_paths(["https://example.com/config.yaml"])
    try:
        assert resolved_paths is not None
        assert len(resolved_paths) == 1
        assert resolved_paths[0].endswith("config.yaml")
        cfg = load_yaml_config(resolved_paths[0])
        assert cfg.__main__.seed == 7
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_config_paths_missing_extension_defaults_yaml(tmp_path: str, monkeypatch: MonkeyPatch) -> None:
    source_file = tmp_path / "remote.yaml"
    source_file.write_text("__main__:\n  seed: 9\n")

    def fake_urlretrieve(url: str, filename: str, *args, **kwargs):
        shutil.copyfile(str(source_file), filename)
        return (filename, None)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    resolved_paths, temp_dir = resolve_config_paths(["https://example.com/config_without_extension"])
    try:
        assert resolved_paths is not None
        assert len(resolved_paths) == 1
        assert resolved_paths[0].endswith(".yaml")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_config_paths_none_input() -> None:
    resolved_paths, temp_dir = resolve_config_paths(None)
    assert resolved_paths is None
    assert temp_dir is None


def test_resolve_config_paths_non_http_url_treated_as_local() -> None:
    resolved_paths, temp_dir = resolve_config_paths(["ftp://example.com/config.yaml"])
    assert resolved_paths == ["ftp://example.com/config.yaml"]
    assert temp_dir is None
