import json

import h5py
import numpy as np
import pytest

from linak import __version__ as LINAK_VERSION
from linak.storage import hdf5_utils
from linak.storage.hdf5_utils import (
    read_linak_hdf5,
    read_linak_hdf5_profile_headers,
    read_linak_hdf5_profiles,
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


def test_linak_hdf5_reader_warns_but_reads_wrong_package_version(tmp_path, caplog):
    source = tmp_path / "wrong_version.h5"
    _write_minimal_density_hdf5(source, linak_version="0.5.0")
    hdf5_utils._WARNED_LINAK_VERSION_MISMATCHES.clear()

    with caplog.at_level("WARNING", logger="linak.storage.hdf5_utils"):
        datasets, metadata = read_linak_hdf5(source, expected_analysis="density")

    assert "density" in datasets
    assert metadata["analysis"] == "density"
    assert "LiNaK version mismatch (0.5.0" in caplog.text


def test_linak_hdf5_reader_warns_once_for_repeated_version_mismatch_reads(tmp_path, caplog):
    source = tmp_path / "wrong_version_repeated.h5"
    _write_minimal_density_hdf5(source, linak_version="0.5.0")
    hdf5_utils._WARNED_LINAK_VERSION_MISMATCHES.clear()

    with caplog.at_level("WARNING", logger="linak.storage.hdf5_utils"):
        read_linak_hdf5_profile_headers(source, expected_analysis="density")
        read_linak_hdf5_profiles(source, expected_analysis="density")
        read_linak_hdf5(source, expected_analysis="density")

    warning_records = [
        record
        for record in caplog.records
        if "LiNaK version mismatch (0.5.0" in record.getMessage()
    ]
    assert len(warning_records) == 1


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
