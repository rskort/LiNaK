from __future__ import annotations

import numpy as np
import pytest

from linak.analysis.binning import reconstruct_uniform_bin_edges_from_edges


@pytest.mark.parametrize("start", [0.0, -2.9716074466705322, 100.125])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reconstruct_edges_preserves_metadata_width_and_stored_origin(start, dtype):
    stored = (start + np.arange(1208) * 0.05).astype(dtype)
    edges, width = reconstruct_uniform_bin_edges_from_edges(
        stored, bin_width=0.05, source_label="grid"
    )

    assert width == 0.05
    assert edges.dtype == np.float64
    assert edges[0] == float(stored[0])
    np.testing.assert_array_equal(edges, float(stored[0]) + np.arange(1208) * 0.05)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reconstruct_edges_rejects_nonuniform_bins(dtype):
    stored = (np.arange(1208) * 0.05).astype(dtype)
    stored[800] += 0.01
    with pytest.raises(ValueError, match="non-uniform"):
        reconstruct_uniform_bin_edges_from_edges(stored, bin_width=0.05, source_label="grid")


@pytest.mark.parametrize("width", [0.1, 0.0, -0.05, float("nan"), float("inf")])
def test_reconstruct_edges_rejects_incompatible_width(width):
    stored = (np.arange(1208) * 0.05).astype(np.float32)
    with pytest.raises(ValueError, match="bin width"):
        reconstruct_uniform_bin_edges_from_edges(stored, bin_width=width, source_label="grid")


def test_reconstruct_edges_without_metadata_uses_full_span():
    stored = (100.125 + np.arange(1208) * 0.05).astype(np.float32)
    edges, width = reconstruct_uniform_bin_edges_from_edges(
        stored, bin_width=None, source_label="grid"
    )
    assert width == (float(stored[-1]) - float(stored[0])) / (stored.size - 1)
    assert edges[-1] == pytest.approx(float(stored[-1]), abs=1.0e-12)


def test_reconstruct_edges_does_not_relax_float64_validation():
    stored = (np.arange(1208) * 0.05).astype(np.float32).astype(np.float64)
    with pytest.raises(ValueError, match="non-uniform"):
        reconstruct_uniform_bin_edges_from_edges(stored, bin_width=0.05, source_label="grid")
