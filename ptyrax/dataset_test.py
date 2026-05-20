import shutil
import urllib.request

import h5py
import numpy as np
import pytest
from pytest import MonkeyPatch

from ptyrax.dataset import from_hdf5


@pytest.mark.skip(reason="Lenspaper data is not currently a part of the repository")
def test_lenspaper_loadable() -> None:
    lenspaper_path = "data/lenspaper/lenspaper.hdf5"
    from ptyrax.dataset import from_hdf5

    ptychogram = from_hdf5(lenspaper_path)
    assert np.all(ptychogram.pixel_number == np.array([364, 364]))
    assert ptychogram.n == 202


def test_apply_orientation() -> None:
    from ptyrax.dataset import Ptychogram, apply_orientation

    ptychogram = Ptychogram(
        diffraction_patterns=np.random.random((20, 10, 10)),
        pixel_size=np.array([1.0, 1.0]),
        sample_positions=np.random.random((20, 2)),
        sample_orientations=np.random.random((20, 6)),
        detector_positions=np.random.random((20, 6)),
        detector_orientations=np.random.random((20, 6)),
        propagation_distance=1.0,
        wavelength=np.array([1.0]),
    )
    output_ptychogram = apply_orientation(ptychogram, 0)
    assert len(output_ptychogram.diffraction_patterns) == len(ptychogram.diffraction_patterns)


def _make_minimal_h5(path: str):
    with h5py.File(path, "w") as f:
        f.create_dataset("diffraction_patterns", data=np.zeros((2, 8, 8), dtype=np.float32))
        f.create_dataset("pixel_size", data=np.array([1.0, 1.0], dtype=np.float32))
        f.create_dataset("sample_positions", data=np.zeros((2, 3), dtype=np.float32))
        f.create_dataset("wavelength", data=np.array([500e-9], dtype=np.float32))
        # include propagation_distance so loader can infer detector positions
        # provide one value per pattern
        f.create_dataset("propagation_distance", data=np.array([100.0, 100.0], dtype=np.float32))


def test_from_hdf5_http_url_monkeypatch(tmp_path: str, monkeypatch: MonkeyPatch) -> None:
    # create local h5 file
    src = tmp_path / "local.h5"
    _make_minimal_h5(str(src))

    # monkeypatch urlretrieve to copy the local file to the destination path
    def fake_urlretrieve(url: str, filename: str, *args, **kwargs):
        shutil.copyfile(str(src), filename)
        return (filename, None)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    # call from_hdf5 with a fake http URL
    ds = from_hdf5("http://example.com/fake.h5")
    assert ds is not None
    assert hasattr(ds, "diffraction_patterns")
    assert ds.n == 2


def test_from_hdf5_http_url_falls_back_to_requests(tmp_path: str, monkeypatch: MonkeyPatch) -> None:
    src = tmp_path / "local.h5"
    _make_minimal_h5(str(src))

    def fake_urlretrieve(url: str, filename: str, *args, **kwargs):
        raise urllib.error.URLError("SSL EOF")

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int = 1024 * 1024):
            for idx in range(0, len(self.payload), chunk_size):
                yield self.payload[idx : idx + chunk_size]

    def fake_requests_get(url: str, stream: bool = True, timeout=None):  # noqa: ANN001
        assert stream is True
        return FakeResponse(src.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    import ptyrax.dataset as dataset_module

    monkeypatch.setattr(dataset_module.requests, "get", fake_requests_get)

    ds = from_hdf5("https://example.com/fake.h5")
    assert ds is not None
    assert hasattr(ds, "diffraction_patterns")
    assert ds.n == 2
