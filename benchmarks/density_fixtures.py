"""Baseline-compatible density payloads for evolving geometry and topology fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.surface_fixtures import CASES, surface_frames


def density_payloads(case):
    from linak.analysis import density

    frames, options = surface_frames(case, frame_count=9, grid=3)
    profiles = density.compute_all_density_profiles(
        frames,
        species="all",
        surface_axis=options.pop("axis"),
        bin_width=1.0,
        grid_bin_width=1.0,
        min_molecule_frames=1,
        **options,
    )
    result = []
    for profile in profiles:
        payload = (
            density._density_grid_profile_hdf5_payload(profile)
            if isinstance(profile, density.DensityGridProfile)
            else density._density_profile_hdf5_payload(profile)
        )
        payload["metadata"].pop("profile_uid", None)
        result.append(payload)
    return result


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.baseline:
        sys.path.insert(0, str(args.baseline.resolve()))
    args.output.write_text(
        json.dumps(
            {case: density_payloads(case) for case in CASES},
            default=json_value,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
