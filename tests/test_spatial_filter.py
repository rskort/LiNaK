from __future__ import annotations

import numpy as np
from ase import Atoms

import linak.trajectory.spatial_filter as spatial_filter_module
from linak.trajectory.spatial_filter import SpatialFilterOptions, apply_spatial_filter


def test_distance_only_atom_filter_does_not_wrap_cartesian_positions(monkeypatch):
    frames = [
        Atoms(
            "HHH",
            positions=[[0.0, 0.0, 0.25], [0.0, 0.0, 1.25], [0.0, 0.0, 2.25]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )
    ]

    monkeypatch.setattr(
        spatial_filter_module,
        "_wrapped_positions_for_frame",
        lambda _frame: (_ for _ in ()).throw(AssertionError("unexpected Cartesian wrap")),
    )
    monkeypatch.setattr(
        spatial_filter_module,
        "_surface_distance_values_for_atoms",
        lambda *_args, **_kwargs: ([np.asarray([0.25, 1.25, 2.25])], None, "test"),
    )

    result = apply_spatial_filter(
        frames,
        options=SpatialFilterOptions(distance_range="0.5:2.0"),
    )

    assert len(result.frames[0]) == 1
    np.testing.assert_allclose(result.frames[0].positions[0], [0.0, 0.0, 1.25])
