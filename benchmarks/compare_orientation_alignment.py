"""Cold exact-oracle versus vectorized registration, plus warm preview timing.

Run from the repository root:
    python -m benchmarks.compare_orientation_alignment --grid-size 64 --groups 4

Source construction and Matplotlib startup are excluded. No analysis files are
written. The synthetic profile is explicitly frozen as GUI-owned data so warm
measurements exercise the same accumulator cache as the GUI.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from statistics import median
from time import perf_counter
from unittest.mock import patch
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linak.cache import PREVIEW_CACHE
from linak.analysis import interface_remap, interface_density
from linak.analysis.orientation import plot_orientation_profile
from linak.analysis.orientation_cache import freeze_arrays
from linak.analysis.orientation_spatial import aligned_orientation_profile, spatial_orientation_map
from tests.support.orientation_performance import orientation_case
from tests.test_orientation_alignment import _setting


def benchmark(n=64, groups=4, repeats=3):
    profile = replace(orientation_case(n, groups), _gui_cache_key=("benchmark", n, groups))
    freeze_arrays(profile)
    settings = _setting(profile, translation=(0.173, 0.267))
    timings = {}
    oracle = None
    for mode in ("polygon_oracle", "vectorized"):
        samples = []
        for _ in range(repeats):
            PREVIEW_CACHE.clear()
            started = perf_counter()
            if mode == "polygon_oracle":
                with (
                    patch.object(interface_remap, "interval_compatible", return_value=False),
                    patch.object(interface_density, "interval_compatible", return_value=False),
                ):
                    result = aligned_orientation_profile(
                        profile, ("x", "y"), interface_alignments=settings
                    )
            else:
                result = aligned_orientation_profile(
                    profile, ("x", "y"), interface_alignments=settings
                )
            samples.append(perf_counter() - started)
        timings[f"registration_{mode}_cold_s"] = median(samples)
        if mode == "polygon_oracle":
            oracle = result
        else:
            assert oracle is not None
            for field in (
                "entity_sum",
                "top_entity_sum",
                "bottom_entity_sum",
                "cos_polar_sum",
                "cos_azimuthal_sum",
                "count_polar_valid",
                "count_azimuthal_valid",
            ):
                np.testing.assert_allclose(
                    getattr(result.sparse_grid, field),
                    getattr(oracle.sparse_grid, field),
                    rtol=1e-10,
                    atol=1e-10,
                )
            for new, old in zip(
                result.sparse_grid.density_geometry_groups,
                oracle.sparse_grid.density_geometry_groups,
            ):
                np.testing.assert_array_equal(new["flat_indices"], old["flat_indices"])
                for field in (
                    "bottom_entity_sum",
                    "bottom_cos_polar_sum",
                    "bottom_count_polar_valid",
                    "bottom_cos_azimuthal_sum",
                    "bottom_count_azimuthal_valid",
                ):
                    np.testing.assert_allclose(new[field], old[field], rtol=1e-10, atol=1e-10)
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        aligned_orientation_profile(profile, ("x", "y"), interface_alignments=settings)
        samples.append(perf_counter() - started)
    timings["registration_warm_s"] = median(samples)
    # Include canvas drawing, rather than just creating Matplotlib artists.
    figure = plt.figure()
    figure.canvas.draw()
    plt.close(figure)
    for temperature in ("cold", "warm"):
        if temperature == "cold":
            PREVIEW_CACHE.clear()
        state: dict[str, Any] = {}
        started = perf_counter()
        plot_orientation_profile(
            profile,
            orientation_heatmap_kind="spatial",
            orientation_heatmap_quantity="density-weighted",
            interface_alignments=settings,
            show=False,
            capture_state=state,
        )
        state["figure"].canvas.draw()
        timings[f"preview_including_drawing_{temperature}_s"] = perf_counter() - started
        plt.close(state["figure"])
    mean = spatial_orientation_map([profile], interface_alignments=settings)
    density = spatial_orientation_map([profile], quantity="density", interface_alignments=settings)
    weighted = spatial_orientation_map(
        [profile], quantity="density-weighted", interface_alignments=settings
    )
    np.testing.assert_allclose(weighted.values, mean.values * density.values, equal_nan=True)
    return {
        "lateral_grid": [n, n],
        "geometry_groups": groups,
        "source_geometry_rows": 2 * n * n * groups,
        **timings,
        "cold_registration_speedup": timings["registration_polygon_oracle_cold_s"]
        / timings["registration_vectorized_cold_s"],
        "oracle_agrees": True,
        "cache": PREVIEW_CACHE.stats(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.grid_size, args.groups, args.repeats) <= 0:
        parser.error("Grid size, group count and repeats must be positive.")
    logging.getLogger("linak").setLevel(logging.ERROR)
    print(json.dumps(benchmark(args.grid_size, args.groups, args.repeats), indent=2))
