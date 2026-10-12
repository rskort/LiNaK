"""Time exact sparse density heatmap projection in a fresh Python process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.conversion_performance import peak_rss_mib
from linak.analysis.density import (
    density_grids_to_averaged_heatmap_profile,
    load_density_grid_profiles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--species", default="K")
    parser.add_argument("--y-axis", default="distance")
    args = parser.parse_args()
    started = perf_counter()
    profiles = load_density_grid_profiles(args.source, species=args.species)
    loaded = perf_counter()
    result = density_grids_to_averaged_heatmap_profile(profiles, x_axis="x", y_axis=args.y_axis)
    finished = perf_counter()
    print(
        json.dumps(
            dict(
                load_s=loaded - started,
                projection_s=finished - loaded,
                total_s=finished - started,
                peak_rss_mib=peak_rss_mib(),
                geometry_groups=sum(len(profile.geometry_groups or []) for profile in profiles),
                shape=result.density.shape,
            )
        )
    )


if __name__ == "__main__":
    main()
