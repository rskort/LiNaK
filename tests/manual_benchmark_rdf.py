from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linak.analysis.rdf import compute_rdf


def _build_frames(
    *,
    n_frames: int,
    n_a: int,
    n_b: int,
    cell_length: float,
    seed: int,
) -> list[Atoms]:
    rng = np.random.default_rng(seed)
    symbols = "O" * n_a + "H" * n_b
    frames: list[Atoms] = []
    for _ in range(n_frames):
        positions = rng.uniform(0.0, float(cell_length), size=(n_a + n_b, 3))
        frames.append(
            Atoms(
                symbols,
                positions=positions,
                cell=[float(cell_length), float(cell_length), float(cell_length)],
                pbc=True,
            )
        )
    return frames


def main() -> None:
    parser = ArgumentParser(description="Manual RDF throughput benchmark for orthorhombic cells.")
    parser.add_argument("--frames", type=int, default=2_000)
    parser.add_argument("--n-a", type=int, default=64)
    parser.add_argument("--n-b", type=int, default=64)
    parser.add_argument("--cell", type=float, default=25.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    frames = _build_frames(
        n_frames=args.frames,
        n_a=args.n_a,
        n_b=args.n_b,
        cell_length=args.cell,
        seed=args.seed,
    )
    started = perf_counter()
    profile = compute_rdf(
        frames,
        species_a="O",
        species_b="H",
        r_max=float(args.r_max),
        bin_width=float(args.bin_width),
        threads=int(args.threads),
    )
    elapsed_s = perf_counter() - started
    print(
        "RDF benchmark:",
        f"frames={args.frames}",
        f"pairs/frame={args.n_a * args.n_b}",
        f"bins={profile.g_r.size}",
        f"elapsed_s={elapsed_s:.3f}",
        f"frames_per_s={args.frames / max(elapsed_s, 1.0e-12):.2f}",
    )


if __name__ == "__main__":
    main()
