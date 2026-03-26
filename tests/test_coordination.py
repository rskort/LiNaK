import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms

import linak.analysis.coordination as coordination_module
from linak.analysis.coordination import (
    CoordinationCutoffResolution,
    CoordinationProfile,
    compute_coordination_profile,
    load_coordination_profile,
    plot_coordination_profile,
    resolve_coordination_cutoff,
    save_coordination_profile,
)
from linak.analysis.rdf import RDFProfile, save_rdf_profile


def _coordination_test_frames() -> list[Atoms]:
    frame0 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.20],
            [1.0, 0.0, 0.20],
            [0.0, 0.0, 1.00],
            [0.70, 0.0, 1.00],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = Atoms(
        "PtPtOH",
        positions=[
            [0.0, 0.0, 0.30],
            [1.0, 0.0, 0.30],
            [0.0, 0.0, 1.20],
            [1.10, 0.0, 1.20],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return [frame0, frame1]


def _orthorhombic_cn_frames() -> list[Atoms]:
    frame0 = Atoms(
        "OOHH",
        positions=[
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 1.0],
            [9.8, 0.0, 0.0],
            [5.8, 0.0, 1.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    frame1 = frame0.copy()
    frame1.positions[2] = [9.6, 0.0, 0.0]
    frame1.positions[3] = [5.6, 0.0, 1.0]
    return [frame0, frame1]


def _triclinic_cn_frames() -> list[Atoms]:
    cell = np.array(
        [
            [8.0, 0.0, 0.0],
            [2.0, 7.5, 0.0],
            [1.0, 0.5, 9.0],
        ],
        dtype=float,
    )
    frame0 = Atoms(
        "OH",
        positions=[
            [0.3, 0.2, 0.1],
            [0.9, 0.2, 0.1],
        ],
        cell=cell,
        pbc=True,
    )
    frame1 = frame0.copy()
    frame1.positions[1] = [1.0, 0.2, 0.1]
    return [frame0, frame1]


def test_compute_coordination_profile_tracks_continuous_cn_and_surface_distance():
    profile = compute_coordination_profile(
        _coordination_test_frames(),
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="direct",
        ),
    )

    assert profile.species_a == "O"
    assert profile.species_b == "H"
    np.testing.assert_array_equal(profile.atom_indices, np.array([2]))
    np.testing.assert_allclose(profile.time_fs, np.array([0.0, 2.0]))
    np.testing.assert_allclose(profile.distance_to_surface[:, 0], np.array([0.80, 0.90]))
    assert profile.coordination_number.shape == (2, 1)
    assert profile.coordination_number[0, 0] == pytest.approx(1.0)
    assert 0.0 < profile.coordination_number[1, 0] < 1.0


def test_continuous_coordination_weights_follow_cosine_taper():
    cutoff = 2.0
    width = 0.4
    distances = np.array([1.8, 2.0, 2.2], dtype=float)

    weights = coordination_module._continuous_coordination_weights(
        distances,
        cutoff_A=cutoff,
        smoothing_width_A=width,
    )

    np.testing.assert_allclose(weights[0], 1.0)
    np.testing.assert_allclose(weights[1], 0.5)
    np.testing.assert_allclose(weights[2], 0.0)


def test_continuous_coordination_weights_are_monotonic_and_support_hard_cutoff():
    distances = np.linspace(1.8, 2.2, 9, dtype=float)
    smoothed = coordination_module._continuous_coordination_weights(
        distances,
        cutoff_A=2.0,
        smoothing_width_A=0.4,
    )
    assert np.all(np.diff(smoothed) <= 1.0e-12)

    hard = coordination_module._continuous_coordination_weights(
        np.array([1.9, 2.0, 2.1], dtype=float),
        cutoff_A=2.0,
        smoothing_width_A=0.0,
    )
    np.testing.assert_allclose(hard, np.array([1.0, 1.0, 0.0], dtype=float))


def test_continuous_coordination_weights_handle_tiny_positive_smoothing_width():
    weights = coordination_module._continuous_coordination_weights(
        np.array([1.999999999999, 2.0, 2.000000000001], dtype=float),
        cutoff_A=2.0,
        smoothing_width_A=1.0e-16,
    )
    np.testing.assert_allclose(weights, np.array([1.0, 1.0, 0.0], dtype=float))


def test_same_species_self_pairs_are_excluded_in_framewise_kernel():
    frame = Atoms(
        "OO",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    indices = np.array([0, 1], dtype=int)

    values = coordination_module._compute_coordination_frame_values(
        frame,
        center_indices=indices,
        neighbor_indices=indices,
        same_selection=True,
        cutoff_A=1.0,
        smoothing_width_A=0.0,
    )

    np.testing.assert_allclose(values, np.array([1.0, 1.0], dtype=float))


def test_same_species_self_pairs_are_excluded_in_chunked_kernel():
    frames = [
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OO",
            positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    indices = np.array([0, 1], dtype=int)

    values = coordination_module._compute_coordination_values_chunked(
        frames,
        center_indices=indices,
        neighbor_indices=indices,
        same_selection=True,
        cutoff_A=1.0,
        smoothing_width_A=0.0,
    )

    np.testing.assert_allclose(values, np.ones((2, 2), dtype=float))


def test_cross_species_zero_distance_pair_is_counted_in_framewise_kernel(monkeypatch):
    monkeypatch.setattr(
        coordination_module, "_can_vectorize_coordination_kernel", lambda *_args, **_kwargs: False
    )

    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_coordination_profile(
        [frame],
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=1.0,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=0.5,
            smoothing_width_A=0.0,
            mode="direct",
        ),
    )

    assert profile.coordination_number[0, 0] == pytest.approx(1.0)


def test_framewise_and_chunked_kernels_agree_for_orthorhombic_periodic_system():
    frames = _orthorhombic_cn_frames()
    center_indices = np.array([0, 1], dtype=int)
    neighbor_indices = np.array([2, 3], dtype=int)
    framewise = np.vstack(
        [
            coordination_module._compute_coordination_frame_values(
                frame,
                center_indices=center_indices,
                neighbor_indices=neighbor_indices,
                same_selection=False,
                cutoff_A=0.9,
                smoothing_width_A=0.2,
            )
            for frame in frames
        ]
    )
    chunked = coordination_module._compute_coordination_values_chunked(
        frames,
        center_indices=center_indices,
        neighbor_indices=neighbor_indices,
        same_selection=False,
        cutoff_A=0.9,
        smoothing_width_A=0.2,
    )

    np.testing.assert_allclose(chunked, framewise, rtol=0.0, atol=1.0e-6)


def test_periodic_minimum_image_is_applied_across_box_boundary():
    frame = Atoms(
        "OH",
        positions=[[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    values = coordination_module._compute_coordination_frame_values(
        frame,
        center_indices=np.array([0], dtype=int),
        neighbor_indices=np.array([1], dtype=int),
        same_selection=False,
        cutoff_A=0.5,
        smoothing_width_A=0.0,
    )
    np.testing.assert_allclose(values, np.array([1.0], dtype=float))


def test_triclinic_periodic_frames_fall_back_to_framewise_kernel(monkeypatch):
    monkeypatch.setattr(
        coordination_module,
        "_compute_coordination_values_chunked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("chunked kernel used")),
    )

    profile = compute_coordination_profile(
        _triclinic_cn_frames(),
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=1.0,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.0,
            mode="direct",
        ),
    )

    assert profile.coordination_number.shape == (2, 1)
    assert np.all(profile.coordination_number[:, 0] > 0.0)


def test_coordination_profile_preserves_center_column_alignment():
    frames = [
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 1.4],
                [0.3, 0.0, 0.2],
                [4.0, 0.0, 1.4],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "OOHH",
            positions=[
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 1.5],
                [0.4, 0.0, 0.3],
                [4.0, 0.0, 1.5],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]

    profile = compute_coordination_profile(
        frames,
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=1.0,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=0.6,
            smoothing_width_A=0.0,
            mode="direct",
        ),
    )

    np.testing.assert_array_equal(profile.atom_indices, np.array([0, 1], dtype=int))
    assert profile.coordination_number.shape == profile.distance_to_surface.shape
    np.testing.assert_allclose(
        profile.distance_to_surface[:, 1] - profile.distance_to_surface[:, 0],
        np.array([1.2, 1.2], dtype=float),
    )
    np.testing.assert_allclose(profile.coordination_number[:, 0], np.array([1.0, 1.0]))
    np.testing.assert_allclose(profile.coordination_number[:, 1], np.array([0.0, 0.0]))


def test_coordination_profile_fails_for_unstable_atom_layout():
    frames = [
        Atoms(
            "OH", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
        ),
        Atoms(
            "HO", positions=[[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
        ),
    ]

    with pytest.raises(ValueError, match="stable atom ordering|stable atom count"):
        compute_coordination_profile(
            frames,
            species_a="O",
            species_b="H",
            axis="z",
            timestep_fs=1.0,
            cutoff_resolution=CoordinationCutoffResolution(
                cutoff_A=1.0,
                smoothing_width_A=0.0,
                mode="direct",
            ),
        )


def test_coordination_profile_fails_for_empty_center_selection():
    frame = Atoms(
        "OH", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
    )

    with pytest.raises(ValueError, match="No atoms found for species 'Pt'"):
        compute_coordination_profile(
            [frame],
            species_a="Pt",
            species_b="H",
            axis="z",
            timestep_fs=1.0,
            cutoff_resolution=CoordinationCutoffResolution(
                cutoff_A=1.0,
                smoothing_width_A=0.0,
                mode="direct",
            ),
        )


def test_coordination_profile_returns_zero_for_empty_neighbor_contributions():
    frame = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    profile = compute_coordination_profile(
        [frame],
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=1.0,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.0,
            mode="direct",
        ),
    )

    np.testing.assert_allclose(profile.coordination_number, np.zeros((1, 1), dtype=float))


def test_same_species_single_center_has_zero_coordination():
    frame = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True)
    profile = compute_coordination_profile(
        [frame],
        species_a="O",
        species_b="O",
        axis="z",
        timestep_fs=1.0,
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.0,
            mode="direct",
        ),
    )

    np.testing.assert_allclose(profile.coordination_number, np.zeros((1, 1), dtype=float))


def test_coordination_analysis_reports_progress_for_cutoff_and_values(monkeypatch):
    events: list[tuple[str, object]] = []

    class _DummyProgressBar:
        def __init__(self, *, desc, total=None, unit="it", **_kwargs):
            self.desc = desc
            self.total = total
            self.unit = unit

        def __enter__(self):
            events.append(("enter", self.desc, self.total, self.unit))
            return self

        def update(self, n=1):
            events.append(("update", self.desc, n))

        def close(self):
            events.append(("close", self.desc))

        def __exit__(self, exc_type, exc, tb):
            self.close()

    monkeypatch.setattr(coordination_module, "ProgressBar", _DummyProgressBar)

    frames = _coordination_test_frames()
    cutoff = resolve_coordination_cutoff(
        frames=frames,
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=None,
        cutoff_from_rdf=False,
    )
    profile = compute_coordination_profile(
        frames,
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=cutoff,
    )

    assert profile.coordination_number.shape == (2, 1)
    entered = [event[1] for event in events if event[0] == "enter"]
    assert any("Coordination values" in desc for desc in entered)
    updates = [event for event in events if event[0] == "update"]
    assert updates


def test_resolve_coordination_cutoff_from_rdf_file_saves_diagnostic_plot(tmp_path):
    rdf_path = tmp_path / "reference_rdf.h5"
    diagnostic = tmp_path / "reference_rdf_cutoff.png"
    profile = RDFProfile(
        species_a="O",
        species_b="H",
        bin_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float),
        bin_centers=np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75], dtype=float),
        g_r=np.array([0.2, 1.8, 1.2, 0.35, 0.55, 0.9], dtype=float),
        n_frames=10,
    )
    save_rdf_profile(profile, rdf_path)

    resolution = resolve_coordination_cutoff(
        frames=[],
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=rdf_path,
        cutoff_from_rdf=False,
        diagnostic_plot_output=diagnostic,
    )

    assert resolution.mode == "rdf_file"
    assert resolution.rdf_g_r is not None
    assert resolution.rdf_g_r_smoothed is not None
    assert resolution.rdf_peak_A is not None
    assert resolution.rdf_minimum_A is not None
    assert diagnostic.exists()
    assert resolution.cutoff_A == pytest.approx(resolution.rdf_minimum_A)


def test_resolve_coordination_cutoff_defaults_to_sampled_rdf(monkeypatch):
    bin_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    counts = np.array([0.2, 1.8, 1.2, 0.35, 0.55, 0.9], dtype=float)
    monkeypatch.setattr(
        coordination_module,
        "_build_reference_rdf_config",
        lambda *args, **kwargs: (
            bin_edges,
            SimpleNamespace(bin_edges=bin_edges),
        ),
    )
    monkeypatch.setattr(
        coordination_module,
        "_accumulate_reference_rdf_contributions",
        lambda *args, **kwargs: (
            counts.copy(),
            np.ones_like(counts),
        ),
    )

    resolution = resolve_coordination_cutoff(
        frames=_coordination_test_frames(),
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=None,
        cutoff_from_rdf=False,
    )

    assert resolution.mode == "sampled_rdf"
    assert resolution.rdf_sampled_frame_index is not None
    assert resolution.cutoff_A == pytest.approx(resolution.rdf_minimum_A)


def test_resolve_coordination_cutoff_accepts_zero_smoothing_width():
    resolution = resolve_coordination_cutoff(
        frames=[],
        species_a="O",
        species_b="H",
        cutoff_A=2.0,
        cutoff_rdf_path=None,
        cutoff_from_rdf=False,
        cutoff_smoothing_width_A=0.0,
    )

    assert resolution.cutoff_A == pytest.approx(2.0)
    assert resolution.smoothing_width_A == pytest.approx(0.0)


def test_resolve_cutoff_from_rdf_curve_raises_when_no_post_peak_minimum_exists():
    with pytest.raises(ValueError, match="first RDF minimum"):
        coordination_module._resolve_cutoff_from_rdf_curve(
            bin_centers_A=np.array([0.25, 0.75, 1.25, 1.75], dtype=float),
            g_r=np.array([0.2, 1.5, 1.2, 1.0], dtype=float),
            smoothing_sigma_A=0.0,
        )


def test_fit_local_quadratic_minimum_falls_back_for_ill_conditioned_fit(monkeypatch):
    monkeypatch.setattr(
        coordination_module.np,
        "polyfit",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 0.0], dtype=float),
    )

    result = coordination_module._fit_local_quadratic_minimum(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([2.0, 1.0, 2.0], dtype=float),
        center_index=1,
    )

    assert result == pytest.approx(1.0)


def test_coordination_reference_rdf_rmax_uses_shared_safe_periodic_rule():
    frames = [
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            cell=[[8.0, 0.0, 0.0], [2.0, 7.5, 0.0], [1.0, 0.5, 9.0]],
            pbc=True,
        )
    ]

    resolved = coordination_module._resolve_reference_rdf_r_max(frames, r_max=None)
    expected = coordination_module._auto_r_max_from_frames(frames)
    assert resolved == pytest.approx(expected)


def test_compute_reference_rdf_uses_nan_for_zero_expected_bins(monkeypatch):
    bin_edges = np.array([0.0, 0.5, 1.0], dtype=float)
    monkeypatch.setattr(
        coordination_module,
        "_build_reference_rdf_config",
        lambda *args, **kwargs: (
            bin_edges,
            SimpleNamespace(bin_edges=bin_edges),
        ),
    )
    monkeypatch.setattr(
        coordination_module,
        "_accumulate_reference_rdf_contributions",
        lambda *args, **kwargs: (
            np.array([0.0, 2.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
        ),
    )

    centers, g_r = coordination_module._compute_reference_rdf(
        frames=_coordination_test_frames(),
        species_a="O",
        species_b="H",
        frame_indices=np.array([0, 1], dtype=int),
        r_max=None,
        bin_width=0.5,
    )

    np.testing.assert_allclose(centers, np.array([0.25, 0.75], dtype=float))
    assert np.isnan(g_r[0])
    assert g_r[1] == pytest.approx(2.0)


def test_resolve_coordination_cutoff_converges_in_random_batches(monkeypatch):
    batch_sizes: list[int] = []
    observed_cumulative_counts: list[np.ndarray] = []
    bin_edges = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    batch_size = coordination_module._DEFAULT_RDF_CONVERGENCE_BATCH_SIZE
    min_frames = coordination_module._DEFAULT_RDF_CONVERGENCE_MIN_FRAMES
    expected_steps = int(np.ceil(min_frames / batch_size))
    monkeypatch.setattr(
        coordination_module,
        "_build_reference_rdf_config",
        lambda *args, **kwargs: (
            bin_edges,
            SimpleNamespace(bin_edges=bin_edges),
        ),
    )
    monkeypatch.setattr(
        coordination_module, "_resolve_rdf_worker_count", lambda *_args, **_kwargs: 1
    )

    def _fake_accumulate(selected_frames, **_kwargs):
        batch_sizes.append(len(selected_frames))
        counts = np.full(bin_edges.size - 1, float(len(selected_frames)), dtype=float)
        expected = np.full(
            bin_edges.size - 1,
            float(batch_size) if len(batch_sizes) == 1 else 0.0,
            dtype=float,
        )
        return counts, expected

    monkeypatch.setattr(
        coordination_module,
        "_accumulate_reference_rdf_contributions",
        _fake_accumulate,
    )
    monkeypatch.setattr(
        coordination_module,
        "_resolve_cutoff_from_rdf_curve",
        lambda **_kwargs: (
            observed_cumulative_counts.append(np.asarray(_kwargs["g_r"], dtype=float).copy())
            or (
                np.zeros(bin_edges.size - 1, dtype=float),
                0.75,
                1.55
                if len(observed_cumulative_counts) == 1
                else (
                    1.50020
                    if len(observed_cumulative_counts) == expected_steps - 2
                    else (
                        1.50025
                        if len(observed_cumulative_counts) == expected_steps - 1
                        else 1.50023
                    )
                ),
            )
        ),
    )

    frames = _coordination_test_frames() * 1200
    resolution = resolve_coordination_cutoff(
        frames=frames,
        species_a="O",
        species_b="H",
        cutoff_A=None,
        cutoff_rdf_path=None,
        cutoff_from_rdf=False,
    )

    assert batch_sizes == [batch_size] * expected_steps
    np.testing.assert_allclose(observed_cumulative_counts[0], np.full(3, 1.0))
    np.testing.assert_allclose(observed_cumulative_counts[1], np.full(3, 2.0))
    np.testing.assert_allclose(
        observed_cumulative_counts[-1],
        np.full(3, float(expected_steps)),
    )
    assert resolution.rdf_sampled_frame_index is not None
    assert resolution.rdf_sampled_frame_index.size == batch_size * expected_steps
    assert resolution.mode == "sampled_rdf"


def test_resolve_coordination_cutoff_prefers_direct_cutoff_over_other_sources(tmp_path):
    rdf_path = tmp_path / "reference_rdf.h5"
    save_rdf_profile(
        RDFProfile(
            species_a="O",
            species_b="H",
            bin_edges=np.array([0.0, 0.5, 1.0, 1.5], dtype=float),
            bin_centers=np.array([0.25, 0.75, 1.25], dtype=float),
            g_r=np.array([0.2, 1.8, 0.4], dtype=float),
            n_frames=4,
        ),
        rdf_path,
    )

    resolution = resolve_coordination_cutoff(
        frames=[],
        species_a="O",
        species_b="H",
        cutoff_A=2.2,
        cutoff_rdf_path=rdf_path,
        cutoff_from_rdf=True,
    )

    assert resolution.mode == "direct"
    assert resolution.cutoff_A == pytest.approx(2.2)


def test_save_and_load_coordination_profile(tmp_path):
    profile = compute_coordination_profile(
        _coordination_test_frames(),
        species_a="O",
        species_b="H",
        axis="z",
        timestep_fs=2.0,
        surface_mode="rough",
        surface_elements=["Pt"],
        cutoff_resolution=CoordinationCutoffResolution(
            cutoff_A=1.0,
            smoothing_width_A=0.4,
            mode="direct",
        ),
    )
    output = tmp_path / "coordination.h5"

    save_coordination_profile(profile, output)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["analysis"] == "coordination"
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["species_a"] == "O"
        assert metadata["species_b"] == "H"
        assert metadata["cutoff_A"] == pytest.approx(1.0)
        assert "coordination_number" in handle

    loaded = load_coordination_profile(output, species_a="O", species_b="H", axis="z")
    np.testing.assert_allclose(loaded.coordination_number, profile.coordination_number)
    np.testing.assert_allclose(loaded.distance_to_surface, profile.distance_to_surface)
    np.testing.assert_allclose(loaded.time_fs, profile.time_fs)


def test_plot_coordination_profile_defaults_to_distance(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_line_series(x, y, **kwargs):
        captured["x"] = x
        captured["y"] = y
        captured["x_label"] = kwargs["x_label"]
        captured["y_label"] = kwargs["y_label"]
        return None

    monkeypatch.setattr("linak.analysis.coordination.plot_line_series", _fake_plot_line_series)

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        distance_to_surface=np.array([[0.8], [1.0]], dtype=float),
        coordination_number=np.array([[1.0], [0.5]], dtype=float),
        n_frames=2,
        n_atoms=1,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    plot_coordination_profile(profile, show=False)
    assert captured["x_label"] == "Distance to the surface ($\\mathrm{\\AA}$)"
    assert captured["y_label"] == "Coordination number"
    assert len(captured["x"]) > 0


def test_plot_coordination_profile_time_uses_atom_series(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_plot_multi_line_series(x_series, y_series, labels, **kwargs):
        captured["x_series"] = x_series
        captured["y_series"] = y_series
        captured["labels"] = labels
        captured["x_label"] = kwargs["x_label"]
        return None

    monkeypatch.setattr(
        "linak.analysis.coordination.plot_multi_line_series",
        _fake_plot_multi_line_series,
    )

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2, 3]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        distance_to_surface=np.array([[0.8, 1.2], [0.9, 1.3]], dtype=float),
        coordination_number=np.array([[1.0, 0.5], [0.8, 0.4]], dtype=float),
        n_frames=2,
        n_atoms=2,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    plot_coordination_profile(profile, component="time", show=False)
    assert captured["x_label"] == "Time (ps)"
    assert captured["labels"] == ["O[2]", "O[3]"]
    assert len(captured["y_series"]) == 2


def test_plot_coordination_time_distance_applies_axis_label_padding(monkeypatch):
    import matplotlib.axes

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 1.0, 2.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        distance_to_surface=np.array([[0.8], [1.0], [1.1]], dtype=float),
        coordination_number=np.array([[1.0], [0.5], [0.8]], dtype=float),
        n_frames=3,
        n_atoms=1,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )
    captured: dict[str, object] = {}

    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def _capture_set_xlabel(self, xlabel, *args, **kwargs):
        captured["x_label"] = xlabel
        captured["x_label_pad"] = kwargs.get("labelpad")
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def _capture_set_ylabel(self, ylabel, *args, **kwargs):
        captured["y_label"] = ylabel
        captured["y_label_pad"] = kwargs.get("labelpad")
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", _capture_set_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", _capture_set_ylabel)

    plot_coordination_profile(
        profile,
        component="time-distance",
        show=False,
        x_label_pad=7.0,
        y_label_pad=9.0,
    )

    assert captured["x_label"] == "Time (ps)"
    assert captured["y_label"] == "Distance to the surface ($\\mathrm{\\AA}$)"
    assert captured["x_label_pad"] == pytest.approx(7.0)
    assert captured["y_label_pad"] == pytest.approx(9.0)


def test_plot_coordination_time_distance_ignores_marker_only_line_kwargs(tmp_path):
    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2]),
        frame_index=np.array([0, 1, 2]),
        step=np.array([0.0, 1.0, 2.0]),
        time_fs=np.array([0.0, 2.0, 4.0]),
        time_ps=np.array([0.0, 0.002, 0.004]),
        distance_to_surface=np.array([[0.8], [1.0], [1.1]], dtype=float),
        coordination_number=np.array([[1.0], [0.5], [0.8]], dtype=float),
        n_frames=3,
        n_atoms=1,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )

    output = tmp_path / "coordination_time_distance.png"
    result = plot_coordination_profile(
        profile,
        component="time-distance",
        line_kwargs={"markersize": 9.0, "marker": "o", "alpha": 0.7},
        output=output,
        show=False,
    )

    assert result == output.resolve()
    assert output.exists()


def test_plot_coordination_time_distance_preserves_explicit_blank_axis_labels(monkeypatch):
    import matplotlib.axes

    profile = CoordinationProfile(
        species_a="O",
        species_b="H",
        axis="z",
        atom_indices=np.array([2]),
        frame_index=np.array([0, 1]),
        step=np.array([0.0, 1.0]),
        time_fs=np.array([0.0, 2.0]),
        time_ps=np.array([0.0, 0.002]),
        distance_to_surface=np.array([[0.8], [1.0]], dtype=float),
        coordination_number=np.array([[1.0], [0.5]], dtype=float),
        n_frames=2,
        n_atoms=1,
        coordinate_mode="distance",
        cutoff_A=1.0,
        cutoff_smoothing_width_A=0.4,
    )
    captured: dict[str, object] = {}

    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def _capture_set_xlabel(self, xlabel, *args, **kwargs):
        captured["x_label"] = xlabel
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def _capture_set_ylabel(self, ylabel, *args, **kwargs):
        captured["y_label"] = ylabel
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", _capture_set_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", _capture_set_ylabel)

    plot_coordination_profile(
        profile,
        component="time-distance",
        show=False,
        x_label="",
        y_label="",
    )

    assert captured["x_label"] == ""
    assert captured["y_label"] == ""
