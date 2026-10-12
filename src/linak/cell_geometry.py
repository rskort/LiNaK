"""Full-cell geometry shared by trajectory, analysis and presentation code.

Cell vectors are rows, as in ASE: Cartesian positions are ``fractional @ cell``.
Lengths alone never describe a tilted or rotated cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging
from typing import Any

from ase.cell import Cell
from ase.geometry import find_mic
import numpy as np
from scipy.spatial import ConvexHull

from .cache import PREVIEW_CACHE


@dataclass(frozen=True)
class CellGeometrySummary:
    """Compact trajectory-cell classification used by logs and diagnostics."""

    classification: str
    frame_count: int
    valid: bool
    constant_cell: bool
    constant_origin: bool
    constant_pbc: bool
    lengths_A: tuple[float, float, float] | None
    angles_deg: tuple[float, float, float] | None
    matrix_A: tuple[tuple[float, float, float], ...] | None
    origin_A: tuple[float, float, float] | None
    pbc: tuple[bool, bool, bool] | None
    surface_axis: str | None
    surface_axis_supported: bool | None


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, (int, np.integer)):
        index = int(axis)
    else:
        token = str(axis).strip().lower()
        if token not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        index = {"x": 0, "y": 1, "z": 2}[token]
    if index not in {0, 1, 2}:
        raise ValueError("axis index must be 0, 1, or 2")
    return index


def is_axis_aligned_orthorhombic(cell: Any, *, atol: float = 1.0e-12) -> bool:
    matrix = cell_matrix(cell)
    return bool(
        np.allclose(matrix, np.diag(np.diag(matrix)), rtol=0.0, atol=atol)
        and np.all(np.abs(np.diag(matrix)) > atol)
    )


def is_axis_prismatic(
    cell: Any,
    axis: str | int,
    *,
    pbc: Any = True,
    atol: float | None = None,
) -> bool:
    """Return whether a Cartesian axis is a decoupled periodic prism axis."""

    matrix = cell_matrix(cell)
    axis_index = _axis_index(axis)
    flags = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    if not bool(flags[axis_index]):
        return False
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1.0e-9 * scale if atol is None else float(atol)
    others = [index for index in range(3) if index != axis_index]
    return bool(
        abs(float(matrix[axis_index, axis_index])) > tolerance
        and np.all(np.abs(matrix[axis_index, others]) <= tolerance)
        and np.all(np.abs(matrix[others, axis_index]) <= tolerance)
    )


def summarize_cell_geometry(
    frames: Any,
    *,
    surface_axis: str | None = None,
) -> CellGeometrySummary:
    """Classify per-frame cell geometry without emitting per-frame messages."""

    frame_list = list(frames)
    axis = None if surface_axis is None else str(surface_axis).strip().lower()
    if not frame_list:
        return CellGeometrySummary(
            "unavailable",
            0,
            False,
            True,
            True,
            True,
            None,
            None,
            None,
            None,
            None,
            axis,
            None,
        )

    matrices: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    pbc_values: list[np.ndarray] = []
    try:
        for frame in frame_list:
            matrices.append(cell_matrix(frame.cell))
            origins.append(frame_origin(frame))
            pbc_values.append(np.broadcast_to(np.asarray(frame.pbc, dtype=bool), (3,)).copy())
    except (TypeError, ValueError):
        return CellGeometrySummary(
            "invalid",
            len(frame_list),
            False,
            False,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            axis,
            False if axis is not None else None,
        )

    first_matrix = matrices[0]
    first_origin = origins[0]
    first_pbc = pbc_values[0]
    constant_cell = all(
        np.allclose(matrix, first_matrix, rtol=0.0, atol=1.0e-9) for matrix in matrices[1:]
    )
    constant_origin = all(
        np.allclose(origin, first_origin, rtol=0.0, atol=1.0e-9) for origin in origins[1:]
    )
    constant_pbc = all(np.array_equal(flags, first_pbc) for flags in pbc_values[1:])
    orthorhombic = all(is_axis_aligned_orthorhombic(matrix) for matrix in matrices)
    axis_supported = None
    if axis is not None:
        axis_supported = all(
            is_axis_prismatic(matrix, axis, pbc=flags)
            for matrix, flags in zip(matrices, pbc_values)
        )
    if orthorhombic:
        classification = "axis-aligned orthorhombic"
    elif axis is not None and bool(axis_supported):
        classification = "tilted prismatic"
    else:
        classification = "general triclinic/rotated"

    return CellGeometrySummary(
        classification=classification,
        frame_count=len(frame_list),
        valid=True,
        constant_cell=constant_cell,
        constant_origin=constant_origin,
        constant_pbc=constant_pbc,
        lengths_A=cell_lengths(first_matrix),
        angles_deg=cell_angles(first_matrix),
        matrix_A=tuple((float(row[0]), float(row[1]), float(row[2])) for row in first_matrix),
        origin_A=(
            float(first_origin[0]),
            float(first_origin[1]),
            float(first_origin[2]),
        ),
        pbc=(bool(first_pbc[0]), bool(first_pbc[1]), bool(first_pbc[2])),
        surface_axis=axis,
        surface_axis_supported=axis_supported,
    )


def log_cell_geometry_summary(
    frames: Any,
    *,
    logger: logging.Logger,
    context: str,
    surface_axis: str | None = None,
    source: str | None = None,
) -> CellGeometrySummary:
    """Emit one informative geometry line and detailed DEBUG provenance."""

    summary = summarize_cell_geometry(frames, surface_axis=surface_axis)
    if not summary.valid:
        logger.warning("%s cell geometry: %s.", context, summary.classification)
        return summary
    assert summary.lengths_A is not None and summary.angles_deg is not None
    pbc_label = (
        "none"
        if summary.pbc is None or not any(summary.pbc)
        else "".join(axis.upper() for axis, active in zip("xyz", summary.pbc) if active)
    )
    variability = "constant" if summary.constant_cell else "variable"
    if summary.constant_origin and summary.origin_A is not None:
        origin_state = (
            "zero origin"
            if np.allclose(summary.origin_A, 0.0, rtol=0.0, atol=1.0e-12)
            else "fixed nonzero origin="
            + "/".join(f"{value:.6g}" for value in summary.origin_A)
            + " A"
        )
    else:
        origin_state = "variable origin"
    axis_detail = ""
    if summary.surface_axis is not None:
        axis_detail = f"; {summary.surface_axis.upper()} surface axis " + (
            "supported" if summary.surface_axis_supported else "unsupported"
        )
    source_detail = "" if not source else f"; source={source}"
    logger.info(
        "%s cell geometry: %s, %s over %d frame(s); lengths=%.6g/%.6g/%.6g A, "
        "angles=%.6g/%.6g/%.6g deg, PBC=%s, %s%s%s.",
        context,
        summary.classification,
        variability,
        summary.frame_count,
        *summary.lengths_A,
        *summary.angles_deg,
        pbc_label,
        origin_state,
        axis_detail,
        source_detail,
    )
    logger.debug(
        "%s cell matrix (A)=%s; origin=%s; constant_pbc=%s.",
        context,
        summary.matrix_A,
        summary.origin_A,
        summary.constant_pbc,
    )
    if summary.surface_axis is not None and summary.surface_axis_supported is False:
        logger.warning(
            "%s selected %s surface axis is coupled to other lattice vectors; "
            "surface-relative geometry is unsupported and will use the documented raw-axis fallback.",
            context,
            summary.surface_axis.upper(),
        )
    return summary


def cell_matrix(value: Any, angles: Any = None) -> np.ndarray:
    """Validate lengths, lengths/angles, or a matrix and return a fresh matrix."""
    data = np.asarray(value, dtype=float)
    if angles is not None:
        if data.shape != (3,) or np.asarray(angles).shape != (3,):
            raise ValueError("Cell lengths and cell angles must each contain three numbers.")
        data = np.concatenate((data, np.asarray(angles, dtype=float)))
    if data.shape not in {(3,), (6,), (3, 3)} or not np.all(np.isfinite(data)):
        raise ValueError("Cell must contain finite lengths, six cell parameters, or a 3x3 matrix.")
    if data.ndim == 1:
        if np.any(data[:3] <= 0):
            raise ValueError("Cell lengths must be positive.")
        if data.size == 6:
            if np.any(data[3:] <= 0) or np.any(data[3:] >= 180):
                raise ValueError("Cell angles must be strictly between 0 and 180 degrees.")
            ca, cb, cg = np.cos(np.deg2rad(data[3:]))
            gram = 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg
            if gram <= 1e-14:
                raise ValueError("Cell angles define a singular or geometrically impossible cell.")
        data = np.asarray(Cell.new(data), dtype=float)
    lengths = np.linalg.norm(data, axis=1)
    if np.any(lengths <= 0) or abs(np.linalg.det(data)) <= 1e-12 * np.prod(lengths):
        raise ValueError("Cell must have nonzero volume and independent vectors.")
    return data.copy()


def cell_lengths(value: Any) -> tuple[float, float, float]:
    a, b, c = np.linalg.norm(cell_matrix(value), axis=1)
    return float(a), float(b), float(c)


def cell_angles(value: Any) -> tuple[float, float, float]:
    a, b, c = Cell(cell_matrix(value)).angles()
    return float(a), float(b), float(c)


def compact_cell(value: Any) -> Any:
    """Keep historical three-length returns only for positive diagonal cells."""
    matrix = cell_matrix(value)
    if np.allclose(matrix, np.diag(np.diag(matrix)), rtol=0, atol=1e-12) and np.all(
        np.diag(matrix) > 0
    ):
        return tuple(float(x) for x in np.diag(matrix))
    return tuple(tuple(float(x) for x in row) for row in matrix)


def same_cell(a: Any, b: Any, *, atol: float = 1e-6) -> bool:
    return bool(np.allclose(cell_matrix(a), cell_matrix(b), rtol=0, atol=atol))


def cell_description(value: Any) -> str:
    return (
        " ".join(f"{v:.6g}" for v in cell_lengths(value))
        + " Angstrom; angles "
        + " ".join(f"{v:.6g}" for v in cell_angles(value))
        + " deg"
    )


def frame_origin(frame: Any) -> np.ndarray:
    return np.asarray(frame.get_celldisp(), dtype=float).reshape(3)


def wrap_points(points: Any, cell: Any, pbc: Any = True, origin: Any = None) -> np.ndarray:
    matrix = cell_matrix(cell)
    offset = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    positions = np.asarray(points, dtype=float)
    fractional = np.linalg.solve(matrix.T, (positions - offset).reshape(-1, 3).T).T
    periodic = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    fractional[:, periodic] %= 1.0
    return (fractional @ matrix + offset).reshape(positions.shape)


def wrapped_frame_positions(frame: Any, points: Any = None) -> np.ndarray:
    return wrap_points(
        frame.positions if points is None else points, frame.cell, frame.pbc, frame_origin(frame)
    )


def cartesian_bounds(cell: Any, origin: Any = None) -> np.ndarray:
    matrix = cell_matrix(cell)
    offset = np.zeros(3) if origin is None else np.asarray(origin).reshape(3)
    vertices = np.asarray(list(product((0.0, 1.0), repeat=3))) @ matrix + offset
    return np.stack((vertices.min(axis=0), vertices.max(axis=0)), axis=1)


def perpendicular_heights(cell: Any) -> np.ndarray:
    matrix = cell_matrix(cell)
    return 1.0 / np.linalg.norm(np.linalg.inv(matrix), axis=0)


def unwrap_positions(positions: Any, cells: Any, pbc: Any = True) -> np.ndarray:
    """Unwrap with previous-frame cells, bounding MIC work independently of length.

    Consecutive identical cells/PBC share one ASE MIC call. Exact diagonal cells
    have a separable numerical kernel; half-cell ties retain ASE's image choice.
    Skewed cells always use ASE's geometrically correct lattice search.
    """
    result = np.array(positions, dtype=float, copy=True)
    raw = np.asarray(positions, dtype=float)
    if len(result) < 2:
        return result
    flags_array = np.asarray(pbc, dtype=bool)
    atom_count = max(1, int(np.prod(raw.shape[1:-1])))
    chunk_frames = max(1, 8192 // atom_count)
    start = 1
    while start < len(result):
        matrix = cell_matrix(cells[start - 1])
        flags = np.broadcast_to(
            flags_array if flags_array.ndim < 2 else flags_array[start - 1], (3,)
        )
        stop = start + 1
        while stop < min(len(result), start + chunk_frames):
            next_flags = flags_array if flags_array.ndim < 2 else flags_array[stop - 1]
            if not np.array_equal(flags, np.broadcast_to(next_flags, (3,))) or not np.array_equal(
                matrix, np.asarray(cells[stop - 1])
            ):
                break
            stop += 1
        steps = raw[start:stop] - raw[start - 1 : stop - 1]
        if np.array_equal(matrix, np.diag(np.diag(matrix))):
            lengths = np.diag(matrix)
            fractional = steps[..., flags] / lengths[flags]
            fractional -= np.floor(fractional + 0.5)
            steps[..., flags] = fractional * lengths[flags]
            # ASE uses its lattice search to break exact half-cell ties. Resolve
            # tie-bearing frames in bounded vector blocks.
            ties = np.any(np.abs(fractional) == 0.5, axis=tuple(range(1, fractional.ndim)))
            for offset in np.flatnonzero(ties):
                original_step = (raw[start + offset] - raw[start + offset - 1]).reshape(-1, 3)
                target = steps[offset].reshape(-1, 3)
                for atom_start in range(0, len(target), 8192):
                    target[atom_start : atom_start + 8192], _ = find_mic(
                        original_step[atom_start : atom_start + 8192], matrix, pbc=flags
                    )
        else:
            flat_steps = steps.reshape(-1, 3)
            for offset in range(0, len(flat_steps), 8192):
                flat_steps[offset : offset + 8192], _ = find_mic(
                    flat_steps[offset : offset + 8192], matrix, pbc=flags
                )
        result[start:stop] = steps
        # Include the previous unwrapped point in cumsum so additions have the
        # same order as sequential reconstruction, even across chunk boundaries.
        np.cumsum(result[start - 1 : stop], axis=0, out=result[start - 1 : stop])
        start = stop
    return result


def _intersection_volume_impl(
    matrix: np.ndarray, origin: np.ndarray, bounds: np.ndarray, extent: np.ndarray | None = None
) -> float:
    if extent is None:
        extent = cartesian_bounds(matrix, origin)
    lower = np.maximum(bounds[:, 0], extent[:, 0])
    upper = np.minimum(bounds[:, 1], extent[:, 1])
    if np.any(upper <= lower):
        return 0.0
    if np.all(lower <= extent[:, 0]) and np.all(upper >= extent[:, 1]):
        return abs(float(np.linalg.det(matrix)))
    if np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-12, rtol=0):
        return float(np.prod(upper - lower))
    # A single clipped axis whose lattice vector is decoupled from the other two
    # (a "prism" cell, e.g. a tilted in-plane cell with an orthogonal surface-normal
    # axis) has a Z-invariant cross-section, so this avoids Qhull for that common case.
    finite_axes = [
        axis for axis in range(3) if np.isfinite(bounds[axis, 0]) and np.isfinite(bounds[axis, 1])
    ]
    if len(finite_axes) == 1:
        axis = finite_axes[0]
        others = [i for i in range(3) if i != axis]
        if np.allclose(matrix[axis, others], 0.0, atol=1e-12) and np.allclose(
            matrix[others, axis], 0.0, atol=1e-12
        ):
            cross_section_area = abs(float(np.linalg.det(matrix[np.ix_(others, others)])))
            return float((upper[axis] - lower[axis]) * cross_section_area)
    # Clip the six parallelepiped faces by Cartesian half-spaces. This avoids
    # random sampling, a dense spatial grid, and ill-conditioned Qhull intersections.
    unit = np.asarray(list(product((0.0, 1.0), repeat=3)))
    vertices = unit @ matrix + origin
    faces = []
    for axis in range(3):
        for side in (0, 1):
            ids = np.flatnonzero(unit[:, axis] == side)
            ids = ids[[0, 1, 3, 2]]
            faces.append(vertices[ids])
    tolerance = 1e-10 * max(1.0, float(np.max(np.linalg.norm(matrix, axis=1))))
    for axis in range(3):
        for sign, bound in ((-1.0, lower[axis]), (1.0, upper[axis])):
            clipped, cap = [], []
            for face in faces:
                polygon = []
                for a, b in zip(face, np.roll(face, -1, axis=0)):
                    da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
                    inside_a, inside_b = da <= tolerance, db <= tolerance
                    if inside_a:
                        polygon.append(a)
                    if inside_a != inside_b:
                        point = a + (b - a) * da / (da - db)
                        polygon.append(point)
                        cap.append(point)
                if len(polygon) >= 3:
                    clipped.append(np.asarray(polygon))
            if len(cap) >= 3:
                cap_array = np.unique(np.round(cap, 12), axis=0)
                if len(cap_array) >= 3:
                    axes = [i for i in range(3) if i != axis]
                    centered = cap_array[:, axes] - cap_array[:, axes].mean(axis=0)
                    order = np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))
                    clipped.append(cap_array[order])
            faces = clipped
            if not faces:
                return 0.0
    points = np.unique(np.round(np.concatenate(faces), 12), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points[0], tol=tolerance) < 3:
        return 0.0
    return float(ConvexHull(points).volume)


def _intersection_volume(matrix_key: tuple, origin_key: tuple, bounds_key: tuple) -> float:
    cache_key = (matrix_key, origin_key, bounds_key)
    cached = PREVIEW_CACHE.get("geometry.intersection", cache_key, None)
    if cached is not None:
        return float(cached)
    matrix = np.asarray(matrix_key).reshape(3, 3)
    origin = np.asarray(origin_key)
    bounds = np.asarray(bounds_key).reshape(3, 2)
    result = _intersection_volume_impl(matrix, origin, bounds)
    PREVIEW_CACHE.put("geometry.intersection", cache_key, result)
    return result


def intersection_volume(cell: Any, bounds: Any, origin: Any = None) -> float:
    """Volume inside both a cell and a Cartesian box (infinite limits allowed)."""
    matrix = cell_matrix(cell)
    offset = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    limits = np.asarray(bounds, dtype=float).reshape(3, 2)
    return _intersection_volume(tuple(matrix.ravel()), tuple(offset), tuple(limits.ravel()))


def bin_volumes(
    cell: Any, edges: Any, axis: int, *, origin: Any = None, shift: float = 0.0
) -> np.ndarray:
    # Validate the cell/origin and compute the cell's Cartesian extent once instead of
    # once per bin. bin_volumes is invoked per frame with bin edges shifted by a
    # per-frame surface offset, so cache lookups on _intersection_volume almost
    # always miss here; for the common "prism" case (the clipped axis's lattice vector
    # decoupled from the other two, e.g. an orthogonal surface-normal axis on a tilted
    # in-plane cell) the whole bin array is computed in one vectorized pass instead of
    # a Python-level per-bin call into the general Qhull-based clipping path.
    matrix = cell_matrix(cell)
    offset = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    extent = cartesian_bounds(matrix, offset)
    edges = np.asarray(edges, dtype=float) + shift
    lower_edges, upper_edges = edges[:-1], edges[1:]

    others = [i for i in range(3) if i != axis]
    if np.allclose(matrix[axis, others], 0.0, atol=1e-12) and np.allclose(
        matrix[others, axis], 0.0, atol=1e-12
    ):
        cross_section_area = abs(float(np.linalg.det(matrix[np.ix_(others, others)])))
        lower = np.maximum(lower_edges, extent[axis, 0])
        upper = np.minimum(upper_edges, extent[axis, 1])
        return np.clip(upper - lower, 0.0, None) * cross_section_area

    result = np.empty(len(lower_edges))
    # Fall back to the general Qhull-based clipping path (cached): when this axis is not
    # a prism (e.g. an in-plane axis coupled to another tilted lattice vector), bin_volumes
    # is still normally called with the same fixed (matrix, origin, edges) every frame
    # (only the surface-relative "distance" axis shifts per frame, and that axis is a
    # prism for the common surface-normal case above), so the shared LRU turns repeated
    # frames into cache hits instead of re-running Qhull per frame per bin.
    matrix_key = tuple(matrix.ravel())
    offset_key = tuple(offset)
    bounds = np.array([[-np.inf, np.inf]] * 3)
    for i in range(len(result)):
        bounds[axis] = (lower_edges[i], upper_edges[i])
        result[i] = _intersection_volume(matrix_key, offset_key, tuple(bounds.ravel()))
    return result


class CellMetadata:
    """Dataclass mixin retaining lengths for callers and complete geometry for use."""

    cell_angstrom: tuple[float, float, float] | None
    pbc_cell_angstrom: tuple[float, float, float] | None

    def __post_init__(self):
        for prefix in ("", "pbc_"):
            name = prefix + "cell_angstrom"
            value = getattr(self, name, None)
            matrix = getattr(self, prefix + "cell_matrix_angstrom", None)
            angles = getattr(self, prefix + "cell_angles_deg", None)
            if matrix is None and value is None:
                continue
            full = cell_matrix(
                matrix if matrix is not None else value,
                angles if matrix is None and np.asarray(value).shape == (3,) else None,
            )
            object.__setattr__(self, name, cell_lengths(full))
            object.__setattr__(
                self,
                prefix + "cell_matrix_angstrom",
                tuple(tuple(float(v) for v in row) for row in full),
            )
            object.__setattr__(self, prefix + "cell_angles_deg", cell_angles(full))

    @property
    def cell_geometry(self):
        matrix = getattr(self, "cell_matrix_angstrom", None)
        value = matrix if matrix is not None else self.cell_angstrom
        return None if value is None else compact_cell(value)

    @property
    def pbc_cell_geometry(self):
        matrix = getattr(self, "pbc_cell_matrix_angstrom", None)
        value = matrix if matrix is not None else self.pbc_cell_angstrom
        return None if value is None else compact_cell(value)


def wrap_trajectory_frames(frames, fallback=None):
    """Wrap copies using each frame's own geometry and boundary flags."""
    result = []
    for source in frames:
        frame = source.copy()
        try:
            cell_matrix(frame.cell)
        except ValueError:
            if fallback is None:
                result.append(frame)
                continue
            frame.set_cell(cell_matrix(fallback))
            frame.set_pbc(True)
        frame.positions[:] = wrapped_frame_positions(frame)
        result.append(frame)
    return result


def needs_general_geometry(frames):
    return any(
        not np.allclose(np.asarray(f.cell), np.diag(np.diag(f.cell)), rtol=0, atol=1e-12)
        or np.any(frame_origin(f) != 0)
        for f in frames
    )


def normalized_bin_values(values, volumes):
    return np.divide(
        values, volumes, out=np.zeros_like(values, dtype=float), where=np.asarray(volumes) > 0
    )


class ProfileCellGeometry:
    cell_matrices_angstrom: np.ndarray | None
    cell_pbc: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        if (
            self.cell_matrices_angstrom is None
            or getattr(self, "periodic_crossings", None) is not None
            or not hasattr(self, "x")
        ):
            return
        positions = np.stack((self.x, self.y, self.z), axis=-1)
        crossings = np.zeros(positions.shape[:2], dtype=bool)
        for i in range(1, len(positions)):
            delta = positions[i] - positions[i - 1]
            finite = np.all(np.isfinite(delta), axis=-1)
            flags = True if self.cell_pbc is None else self.cell_pbc[i - 1]
            mic, _ = find_mic(delta[finite], self.cell_matrices_angstrom[i - 1], pbc=flags)
            crossings[i, finite] = np.any(np.abs(mic - delta[finite]) > 1e-7, axis=-1)
        object.__setattr__(self, "periodic_crossings", crossings)


def profile_geometry_datasets(profile):
    return {
        "cell_matrices_angstrom": profile.cell_matrices_angstrom,
        "cell_origins_angstrom": profile.cell_origins_angstrom,
        "cell_pbc": profile.cell_pbc,
        "periodic_crossings": profile.periodic_crossings,
    }


def profile_cartesian_bounds(profile):
    if profile.cell_matrices_angstrom is None:
        if profile.cell_lengths_angstrom is None:
            return None
        return cartesian_bounds(profile.cell_lengths_angstrom)
    all_bounds = np.asarray(
        [
            cartesian_bounds(
                cell,
                None if profile.cell_origins_angstrom is None else profile.cell_origins_angstrom[i],
            )
            for i, cell in enumerate(profile.cell_matrices_angstrom)
        ]
    )
    return np.stack((all_bounds[:, :, 0].min(axis=0), all_bounds[:, :, 1].max(axis=0)), axis=1)


def cell_metadata_payload(cell, prefix="resolved_"):
    return {
        prefix + "cell_angstrom": list(cell_lengths(cell)),
        prefix + "cell_matrix_angstrom": cell_matrix(cell).tolist(),
        prefix + "cell_angles_deg": list(cell_angles(cell)),
    }


def cached_frame_geometry_matches(path, frames):
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            root = handle["trajectory"] if "trajectory" in handle else handle
            if "cell" not in root:
                return False
            cells = np.asarray(root["cell"])
            current = np.asarray([frame.cell for frame in frames])
            if cells.shape != current.shape or not np.allclose(cells, current, rtol=0, atol=1e-9):
                return False
            if "pbc" in root and not np.array_equal(root["pbc"], [frame.pbc for frame in frames]):
                return False
            origins = (
                np.asarray(root["cell_origin_angstrom"])
                if "cell_origin_angstrom" in root
                else np.zeros((len(frames), 3))
            )
            return bool(
                np.allclose(origins, [frame_origin(frame) for frame in frames], rtol=0, atol=1e-9)
            )
    except (OSError, KeyError, ValueError):
        return False
