"""Compare optimized trajectory reconstruction with the original ASE recurrence."""

import numpy as np
import pytest
from ase.geometry import find_mic

import linak.cell_geometry as geometry


def sequential_unwrap(positions, cells, pbc=True):
    result = np.array(positions, dtype=float, copy=True)
    for index in range(1, len(result)):
        flags = pbc if np.asarray(pbc).ndim < 2 else pbc[index - 1]
        step, _ = find_mic(positions[index] - positions[index - 1], cells[index - 1], pbc=flags)
        result[index] = result[index - 1] + step
    return result


@pytest.mark.parametrize("pbc", [False, True, [True, False, True], [False, True, False]])
@pytest.mark.parametrize("changing", [False, True])
@pytest.mark.parametrize(
    "cell",
    [
        np.diag([7.0, 9.0, 12.0]),
        np.diag([-7.0, 9.0, 12.0]),
        np.array([[7.0, 0, 0], [6.8, 0.8, 0], [3.3, 0.3, 12.0]]),
        np.array([[0, 7.0, 0], [9.0, 2.0, 0], [2.0, 3.0, -12.0]]),
    ],
)
def test_unwrap_matches_ase_for_skewed_rotated_and_changing_cells(cell, changing, pbc):
    rng = np.random.default_rng(619)
    cells = np.repeat(cell[None], 83, axis=0)
    if changing:
        cells *= np.linspace(0.8, 1.2, len(cells))[:, None, None]
    fractional = np.cumsum(rng.normal(0, 0.35, (len(cells), 127, 3)), axis=0)
    positions = np.mod(fractional, 1) @ cells
    expected = sequential_unwrap(positions, cells, pbc)
    actual = geometry.unwrap_positions(positions, cells, pbc)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)
    np.testing.assert_array_equal(positions, np.mod(fractional, 1) @ cells)


def test_half_cell_ties_and_changing_pbc_preserve_ase_image_choices():
    positions = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[5, -7, 9], [-5, 7, -9]],
            [[0, 0, 0], [0, 0, 0]],
            [[5, -7, 9], [-5, 7, -9]],
        ],
        dtype=float,
    )
    cells = np.repeat(np.diag([10.0, 14.0, 18.0])[None], len(positions), axis=0)
    pbc = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(
        geometry.unwrap_positions(positions, cells, pbc), sequential_unwrap(positions, cells, pbc)
    )


def test_constant_triclinic_mic_is_batched_with_bounded_scratch(monkeypatch):
    rng = np.random.default_rng(87)
    positions = rng.uniform(0, 1, (258, 101, 3))
    cell = np.array([[5.0, 0, 0], [4.0, 2.0, 0], [2.0, 1.0, 7.0]])
    cells = np.repeat(cell[None], len(positions), axis=0)
    expected = sequential_unwrap(positions, cells)
    original = geometry.find_mic
    counts = []

    def record(vectors, *args, **kwargs):
        counts.append(len(vectors))
        assert len(vectors) <= 8192
        return original(vectors, *args, **kwargs)

    monkeypatch.setattr(geometry, "find_mic", record)
    actual = geometry.unwrap_positions(positions, cells)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
    assert len(counts) == 4


def test_diagonal_mobile_surface_does_not_call_general_mic(monkeypatch):
    rng = np.random.default_rng(10)
    positions = rng.uniform(0, 20, (512, 97, 3))
    cells = np.repeat(np.diag([20.0, 20.0, 20.0])[None], len(positions), axis=0)
    expected = sequential_unwrap(positions, cells)

    def fail(*args, **kwargs):
        raise AssertionError("Verified diagonal geometry should use the separable kernel")

    monkeypatch.setattr(geometry, "find_mic", fail)
    np.testing.assert_allclose(
        geometry.unwrap_positions(positions, cells), expected, rtol=0, atol=1e-11
    )


@pytest.mark.parametrize("ties", [False, True])
def test_mic_scratch_is_bounded_even_for_one_large_frame(monkeypatch, ties):
    cell = (
        np.diag([10.0, 14.0, 18.0])
        if ties
        else np.array([[10.0, 0, 0], [9.0, 2.0, 0], [3.0, 1.0, 18.0]])
    )
    positions = np.zeros((2, 20000, 3))
    positions[1] = [5.0, -7.0, 9.0] if ties else [3.0, 5.0, 7.0]
    cells = [cell, cell]
    expected = sequential_unwrap(positions, cells)
    original = geometry.find_mic
    counts = []

    def record(vectors, *args, **kwargs):
        counts.append(len(vectors))
        assert len(vectors) <= 8192
        return original(vectors, *args, **kwargs)

    monkeypatch.setattr(geometry, "find_mic", record)
    np.testing.assert_allclose(
        geometry.unwrap_positions(positions, cells), expected, rtol=0, atol=1e-12
    )
    assert len(counts) == 3


@pytest.mark.parametrize("positions", [np.empty((0, 2, 3)), np.zeros((1, 2, 3))])
def test_empty_and_single_frame_reconstruction(positions):
    np.testing.assert_array_equal(geometry.unwrap_positions(positions, []), positions)


def test_invalid_cell_still_fails():
    with pytest.raises(ValueError, match="nonzero volume"):
        geometry.unwrap_positions(np.zeros((2, 1, 3)), [np.zeros((3, 3))])
