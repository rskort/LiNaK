import json

import h5py
import numpy as np
import pytest

from linak import __version__ as LINAK_VERSION
from linak.storage.hdf5_utils import (
    INCOMPATIBLE_LINAK_HDF5_MESSAGE,
    read_linak_hdf5,
)


def _write_minimal_density_hdf5(path, *, linak_version=LINAK_VERSION, metadata=None):
    payload = {
        "analysis": "density",
        "analysis_schema_version": 1,
        "profile_uid": "density-profile",
        "axis": "z",
        "species": "O",
        "bin_width_A": 1.0,
    }
    if metadata is not None:
        payload = dict(metadata)
    with h5py.File(path, "w") as handle:
        handle.attrs["linak_format"] = "linak-hdf5"
        handle.attrs["linak_format_version"] = 1
        handle.attrs["linak_version"] = linak_version
        handle.attrs["analysis"] = "density"
        handle.attrs["metadata_json"] = json.dumps(payload)
        handle.create_dataset("bin_centers_A", data=np.array([0.5], dtype=float))
        handle.create_dataset("density", data=np.array([1.0], dtype=float))


def test_linak_hdf5_reader_rejects_wrong_package_version(tmp_path):
    source = tmp_path / "wrong_version.h5"
    _write_minimal_density_hdf5(source, linak_version="0.5.0")

    with pytest.raises(ValueError, match="wrong LiNaK version") as excinfo:
        read_linak_hdf5(source, expected_analysis="density")

    assert INCOMPATIBLE_LINAK_HDF5_MESSAGE in str(excinfo.value)


def test_linak_hdf5_reader_rejects_missing_required_profile_metadata(tmp_path):
    source = tmp_path / "missing_profile_uid.h5"
    _write_minimal_density_hdf5(
        source,
        metadata={
            "analysis": "density",
            "analysis_schema_version": 1,
            "axis": "z",
            "species": "O",
            "bin_width_A": 1.0,
        },
    )

    with pytest.raises(ValueError, match="missing required key"):
        read_linak_hdf5(source, expected_analysis="density")
