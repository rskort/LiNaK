"""Periodic registration of two slab interfaces, independent of the GUI.

Coordinates follow ASE's row-vector convention. Registration is a fixed
isometry of the lateral periodic lattice; it never changes outward distance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import logging
from time import perf_counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from ase.geometry import find_mic, minkowski_reduce
from scipy.optimize import linear_sum_assignment

from ..cancellation import check_cancelled
from ..cache import PREVIEW_CACHE, source_revision
from ..cell_geometry import frame_origin, is_axis_prismatic

LOGGER = logging.getLogger("linak")

REFERENCE_VERSION = 1
REFERENCE_GROUP = "interface_references"
DEFAULT_DEPTH_A = 5.0
DEFAULT_TOLERANCE_A = 0.3
RECOMPUTE_MESSAGE = "Recompute this density/position file to enable interface alignment."


def _matrix_tuple(matrix: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((int(matrix[0][0]), int(matrix[0][1])), (int(matrix[1][0]), int(matrix[1][1])))


@dataclass(frozen=True)
class InterfaceSnapshot:
    frame_index: int
    atom_indices: np.ndarray
    elements: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    origin: np.ndarray
    pbc: np.ndarray
    surface_top: float
    surface_bottom: float
    axis_cut: float
    axis_lower: float
    axis_period: float


@dataclass(frozen=True)
class InterfaceReference:
    reference_id: str
    axis: str
    cells: np.ndarray
    snapshots: tuple[InterfaceSnapshot, ...] = ()
    source_path: str | None = None
    unavailable_reason: str = ""
    version: int = REFERENCE_VERSION

    @property
    def lateral_indices(self) -> tuple[int, int]:
        return tuple(i for i in range(3) if i != "xyz".index(self.axis))  # type: ignore[return-value]

    def metadata(self) -> dict[str, Any]:
        return dict(
            version=self.version,
            reference_id=self.reference_id,
            axis=self.axis,
            cells=np.asarray(self.cells).tolist(),
            unavailable_reason=self.unavailable_reason,
        )

    def materialize(self) -> InterfaceReference:
        if self.unavailable_reason:
            raise ValueError(self.unavailable_reason)
        if self.snapshots:
            return self
        if not self.source_path:
            raise ValueError(RECOMPUTE_MESSAGE)
        import h5py

        with h5py.File(self.source_path, "r") as handle:
            name = f"{REFERENCE_GROUP}/{self.reference_id}"
            if name not in handle:
                raise ValueError(RECOMPUTE_MESSAGE)
            group = handle[name]
            if int(group.attrs.get("version", 0)) != REFERENCE_VERSION:
                raise ValueError(RECOMPUTE_MESSAGE)
            snapshots = []
            for key in sorted(group):
                check_cancelled()
                node = group[key]
                snapshots.append(
                    InterfaceSnapshot(
                        frame_index=int(node.attrs["frame_index"]),
                        atom_indices=np.asarray(node["atom_indices"]),
                        elements=np.asarray(node["elements"]).astype(str),
                        positions=np.asarray(node["positions_A"], dtype=float),
                        cell=np.asarray(node["cell_A"], dtype=float),
                        origin=np.asarray(node["origin_A"], dtype=float),
                        pbc=np.asarray(node["pbc"], dtype=bool),
                        **{
                            name: float(node.attrs[name])
                            for name in (
                                "surface_top",
                                "surface_bottom",
                                "axis_cut",
                                "axis_lower",
                                "axis_period",
                            )
                        },
                    )
                )
        if not snapshots:
            raise ValueError(RECOMPUTE_MESSAGE)
        return replace(self, snapshots=tuple(snapshots))

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any], path: str | Path
    ) -> InterfaceReference | None:
        record = metadata.get("interface_reference")
        if not isinstance(record, Mapping) or record.get("version") != REFERENCE_VERSION:
            return None
        try:
            axis = str(record["axis"])
            cells = np.asarray(record["cells"], dtype=float).reshape((-1, 3, 3))
            identifier = str(record["reference_id"])
            if (
                axis not in ("x", "y", "z")
                or len(identifier) != 64
                or any(c not in "0123456789abcdef" for c in identifier)
            ):
                return None
            return cls(
                identifier,
                axis,
                cells,
                source_path=str(Path(path).resolve()),
                unavailable_reason=str(record.get("unavailable_reason") or ""),
            )
        except (KeyError, ValueError, TypeError):
            return None


class InterfaceReferenceCollector:
    """Capture snapshots in an existing frame loop, without reading frames again."""

    def __init__(self, surface_reference: Any, frame_count: int):
        self.surface = surface_reference
        self.enabled = surface_reference is not None and surface_reference.is_symmetric
        self.indices = set(np.linspace(0, max(0, frame_count - 1), min(5, frame_count), dtype=int))
        self.snapshots: dict[int, InterfaceSnapshot] = {}
        self.cells: dict[bytes, np.ndarray] = {}
        self.reason = ""

    def observe(self, frame_index: int, frame: Any) -> None:
        if not self.enabled:
            return
        cell = np.asarray(frame.cell, dtype=float)
        self.cells.setdefault(cell.tobytes(), cell.copy())
        if not np.all(frame.pbc) or not is_axis_prismatic(cell, self.surface.axis, pbc=frame.pbc):
            self.reason = (
                "Interface alignment requires a fully periodic cell with a decoupled surface axis."
            )
        if frame_index not in self.indices or frame_index in self.snapshots:
            return
        from .surface import surface_substrate_mask

        reference = self.surface
        if reference.top is None or reference.bottom is None:
            self.reason = "Both surface references are required for interface alignment."
            return
        indices = np.flatnonzero(surface_substrate_mask(reference, frame))
        self.snapshots[frame_index] = InterfaceSnapshot(
            frame_index,
            indices,
            np.asarray(frame.get_chemical_symbols())[indices],
            np.asarray(frame.positions, dtype=float)[indices].copy(),
            cell.copy(),
            frame_origin(frame).copy(),
            np.asarray(frame.pbc, dtype=bool).copy(),
            float(reference.top.frame_values[frame_index]),
            float(reference.bottom.frame_values[frame_index]),
            float(reference.axis_cut_per_frame_A[frame_index]),
            float(reference.axis_lower_per_frame_A[frame_index]),
            float(reference.axis_period_per_frame_A[frame_index]),
        )

    def finish(self) -> InterfaceReference | None:
        if not self.enabled or not self.snapshots:
            return None
        snapshots = tuple(self.snapshots[i] for i in sorted(self.snapshots))
        digest = hashlib.sha256(self.surface.axis.encode())
        for snapshot in snapshots:
            digest.update(str(snapshot.frame_index).encode())
            digest.update("|".join(snapshot.elements).encode())
            for array in (
                snapshot.atom_indices,
                snapshot.positions,
                snapshot.cell,
                snapshot.origin,
                snapshot.pbc,
            ):
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(
                np.asarray(
                    [
                        snapshot.surface_top,
                        snapshot.surface_bottom,
                        snapshot.axis_cut,
                        snapshot.axis_lower,
                        snapshot.axis_period,
                    ],
                    dtype=np.float64,
                ).tobytes()
            )
        return InterfaceReference(
            digest.hexdigest(),
            self.surface.axis,
            np.asarray(list(self.cells.values())),
            snapshots,
            unavailable_reason=self.reason,
        )


def write_interface_references(
    handle: Any, references: Sequence[InterfaceReference | None]
) -> None:
    """Write once per source, inside the caller's atomic HDF5 transaction."""
    import h5py
    from ..storage.policies import dataset_create_options

    for reference in references:
        if reference is None:
            continue
        name = f"{REFERENCE_GROUP}/{reference.reference_id}"
        if name in handle:
            continue
        # An unavailable record remains advisory, like other surface diagnostics.
        if reference.unavailable_reason:
            continue
        reference = reference.materialize()
        root = handle.require_group(name)
        root.attrs["version"] = REFERENCE_VERSION
        for index, snapshot in enumerate(reference.snapshots):
            node = root.create_group(f"{index:02d}")
            for key in (
                "frame_index",
                "surface_top",
                "surface_bottom",
                "axis_cut",
                "axis_lower",
                "axis_period",
            ):
                node.attrs[key] = getattr(snapshot, key)
            for key, values in dict(
                atom_indices=snapshot.atom_indices,
                elements=np.asarray(snapshot.elements, dtype=object),
                positions_A=snapshot.positions,
                cell_A=snapshot.cell,
                origin_A=snapshot.origin,
                pbc=snapshot.pbc,
            ).items():
                options = (
                    {"dtype": h5py.string_dtype("utf-8")}
                    if key == "elements"
                    else dataset_create_options(
                        np.asarray(values),
                        compression_profile=handle.attrs.get("compression_profile", "fast"),
                    )
                )
                node.create_dataset(key, data=values, **options)


@dataclass(frozen=True)
class InterfaceAlignment:
    reference_id: str
    matrix: tuple[tuple[int, int], tuple[int, int]] = ((1, 0), (0, 1))
    translation: tuple[float, float] = (0.0, 0.0)
    depth_A: float = DEFAULT_DEPTH_A
    tolerance_A: float = DEFAULT_TOLERANCE_A
    allow_mismatches: bool = False
    diagnostics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(
            version=1,
            reference_id=self.reference_id,
            matrix=[list(row) for row in self.matrix],
            translation=list(self.translation),
            depth_A=self.depth_A,
            tolerance_A=self.tolerance_A,
            allow_mismatches=self.allow_mismatches,
            diagnostics=dict(self.diagnostics or {}),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InterfaceAlignment:
        try:
            matrix = np.asarray(value["matrix"], dtype=float)
            translation = np.asarray(value["translation"], dtype=float)
            depth, tolerance = float(value["depth_A"]), float(value["tolerance_A"])
            identifier = value.get("reference_id")
            if (
                value.get("version") != 1
                or not isinstance(identifier, str)
                or len(identifier) != 64
                or any(character not in "0123456789abcdef" for character in identifier)
                or matrix.shape != (2, 2)
                or not np.all(np.isfinite(matrix))
                or not np.array_equal(matrix, np.rint(matrix))
                or abs(round(float(np.linalg.det(matrix)))) != 1
                or translation.shape != (2,)
                or not np.all(np.isfinite(translation))
                or not np.isfinite(depth + tolerance)
                or depth <= 0
                or tolerance <= 0
                or not isinstance(value.get("allow_mismatches", False), bool)
            ):
                raise ValueError("Invalid interface alignment settings.")
            if value.get("diagnostics") is not None and not isinstance(
                value["diagnostics"], Mapping
            ):
                raise ValueError("Invalid alignment diagnostics.")
            return cls(
                str(value["reference_id"]),
                _matrix_tuple(matrix),
                (float(translation[0] % 1), float(translation[1] % 1)),
                depth,
                tolerance,
                bool(value.get("allow_mismatches", False)),
                value.get("diagnostics"),
            )
        except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError("Invalid interface alignment settings.") from exc


def validate_alignment(reference: InterfaceReference, alignment: InterfaceAlignment) -> None:
    if reference.unavailable_reason:
        raise ValueError(reference.unavailable_reason)
    if reference.reference_id != alignment.reference_id:
        raise ValueError("This alignment belongs to a different interface reference.")
    # Validate values even when constructed directly through the Python API.
    InterfaceAlignment.from_dict(alignment.to_dict())
    indices = reference.lateral_indices
    matrix = np.asarray(alignment.matrix)
    cells = np.ascontiguousarray(reference.cells, dtype=float)
    key = (reference.axis, cells.shape, hashlib.sha256(cells.tobytes()).digest(), alignment.matrix)
    if PREVIEW_CACHE.get("interface.metric", key) is not None:
        return
    for cell in reference.cells:
        check_cancelled()
        if not is_axis_prismatic(cell, reference.axis):
            raise ValueError("The cell has no supported decoupled surface axis.")
        basis = cell[list(indices)][:, list(indices)]
        metric = basis @ basis.T
        if not np.allclose(matrix @ metric @ matrix.T, metric, rtol=1e-7, atol=1e-8):
            raise ValueError("This transform is incompatible with the periodic cell geometry.")
    check_cancelled()
    PREVIEW_CACHE.put("interface.metric", key, True)


def lattice_symmetries(reference: InterfaceReference) -> list[np.ndarray]:
    """Enumerate 2-D lattice isometries in a reduced basis, including skew cells."""
    if reference.unavailable_reason:
        raise ValueError(reference.unavailable_reason)
    indices = list(reference.lateral_indices)
    cell = reference.cells[0]
    if not is_axis_prismatic(cell, reference.axis):
        raise ValueError("The cell has no supported decoupled surface axis.")
    pbc = np.ones(3, dtype=bool)
    pbc["xyz".index(reference.axis)] = False
    reduced, operation = minkowski_reduce(cell, pbc=pbc)
    transform = operation[np.ix_(indices, indices)]
    metric = reduced[indices] @ reduced[indices].T
    # In a reduced 2-D basis these bounds include every metric-preserving column/row.
    smallest = float(np.min(np.linalg.eigvalsh(metric)))
    bound = int(np.ceil(np.sqrt(np.max(np.diag(metric)) / smallest)))
    if bound > 64:
        raise ValueError("The lateral cell is too ill-conditioned for reliable alignment.")
    vectors = np.asarray(list(product(range(-bound, bound + 1), repeat=2)))
    norms = np.einsum("ij,jk,ik->i", vectors, metric, vectors)
    first = vectors[np.isclose(norms, metric[0, 0], rtol=1e-7, atol=1e-8)]
    second = vectors[np.isclose(norms, metric[1, 1], rtol=1e-7, atol=1e-8)]
    result = []
    for a, b in product(first, second):
        if not np.isclose(a @ metric @ b, metric[0, 1], rtol=1e-7, atol=1e-8):
            continue
        reduced_matrix = np.asarray([a, b])
        matrix = np.rint(np.linalg.inv(transform) @ reduced_matrix @ transform).astype(int)
        alignment = InterfaceAlignment(reference.reference_id, _matrix_tuple(matrix))
        try:
            validate_alignment(reference, alignment)
        except ValueError:
            continue
        result.append(matrix)
    result.sort(
        key=lambda m: (not np.array_equal(m, np.eye(2)), np.linalg.det(m) < 0, tuple(m.ravel()))
    )
    return result


def transform_label(reference: InterfaceReference, matrix: np.ndarray) -> str:
    indices = list(reference.lateral_indices)
    basis = reference.cells[0][np.ix_(indices, indices)]
    cartesian = np.linalg.inv(basis) @ matrix @ basis
    if np.allclose(cartesian, np.eye(2)):
        return "No rotation or reflection"
    if np.linalg.det(cartesian) > 0:
        angle = np.degrees(np.arctan2(cartesian[0, 1], cartesian[0, 0])) % 360
        return f"Rotate {angle:.0f}°"
    if np.allclose(cartesian, np.diag([-1, 1])):
        return f"Mirror {'XYZ'[indices[0]]}"
    if np.allclose(cartesian, np.diag([1, -1])):
        return f"Mirror {'XYZ'[indices[1]]}"
    angle = (0.5 * np.degrees(np.arctan2(cartesian[0, 1], cartesian[0, 0]))) % 180
    return f"Reflect about {angle:.1f}° lateral axis"


def apply_interface_alignment(
    positions: np.ndarray,
    cell: np.ndarray,
    origin: np.ndarray,
    axis: str,
    alignment: InterfaceAlignment,
) -> np.ndarray:
    indices = [i for i in range(3) if i != "xyz".index(axis)]
    basis = np.asarray(cell)[np.ix_(indices, indices)]
    result = np.array(positions, dtype=float, copy=True)
    fractional = (result[..., indices] - np.asarray(origin)[indices]) @ np.linalg.inv(basis)
    result[..., indices] = (
        (fractional @ np.asarray(alignment.matrix) + alignment.translation) % 1
    ) @ basis + np.asarray(origin)[indices]
    return result


def snapshot_sides(reference: InterfaceReference, snapshot: InterfaceSnapshot, depth_A: float):
    axis_index = "xyz".index(reference.axis)
    values = snapshot.positions[:, axis_index]
    # Match the exact canonical interval used by the surface reference.
    from .surface import _canonical_axis_values

    canonical = _canonical_axis_values(
        values, cut=snapshot.axis_cut, lower=snapshot.axis_lower, period=snapshot.axis_period
    )
    middle = 0.5 * (snapshot.surface_top + snapshot.surface_bottom)
    result = []
    for top in (True, False):
        side = canonical > middle if top else canonical < middle
        if np.any(side):
            outer = np.max(canonical[side]) if top else np.min(canonical[side])
            side &= canonical >= outer - depth_A if top else canonical <= outer + depth_A
        points = snapshot.positions[side].copy()
        lateral = list(reference.lateral_indices)
        basis = snapshot.cell[np.ix_(lateral, lateral)]
        points[:, lateral] = (
            ((points[:, lateral] - snapshot.origin[lateral]) @ np.linalg.inv(basis)) % 1
        ) @ basis + snapshot.origin[lateral]
        points[:, axis_index] = (
            canonical[side] - snapshot.surface_top
            if top
            else snapshot.surface_bottom - canonical[side]
        )
        result.append((points, snapshot.elements[side], snapshot.atom_indices[side]))
    return result


@dataclass(frozen=True)
class AlignmentDiagnostics:
    top_coverage: tuple[float, ...]
    bottom_coverage: tuple[float, ...]
    rmsd_A: tuple[float | None, ...]
    matched_indices: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    ambiguous: bool = False

    @property
    def good_match(self) -> bool:
        return bool(self.top_coverage and min(self.top_coverage + self.bottom_coverage) >= 0.95)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            top_coverage=list(self.top_coverage),
            bottom_coverage=list(self.bottom_coverage),
            rmsd_A=list(self.rmsd_A),
            good_match=self.good_match,
            ambiguous=self.ambiguous,
        )


def _match_snapshot(
    reference: InterfaceReference, snapshot: InterfaceSnapshot, alignment: InterfaceAlignment
):
    (top, top_elements, _), (bottom, bottom_elements, _) = snapshot_sides(
        reference, snapshot, alignment.depth_A
    )
    moved = apply_interface_alignment(
        bottom, snapshot.cell, snapshot.origin, reference.axis, alignment
    )
    count_top, count_bottom = len(top), len(bottom)
    if not count_top or not count_bottom:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3)), top, moved
    # Species partitions bound memory and ensure identities cannot be interchanged.
    matches_top, matches_bottom, residuals = [], [], []
    lateral_pbc = np.asarray(snapshot.pbc).copy()
    lateral_pbc["xyz".index(reference.axis)] = False
    for element in np.intersect1d(top_elements, bottom_elements):
        check_cancelled()
        ti, bi = np.flatnonzero(top_elements == element), np.flatnonzero(bottom_elements == element)
        # Use sparse neighbourhood components for large surfaces, with an exact
        # assignment (including independent unmatched nodes) in each component.
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        axes = list(reference.lateral_indices)
        pbc = np.zeros(3, bool)
        pbc[axes] = True
        reduced, _ = minkowski_reduce(snapshot.cell, pbc=pbc)
        fractional = np.linalg.solve(np.asarray(reduced).T, moved[bi].T).T
        fractional[:, axes] %= 1
        wrapped_bottom = fractional @ reduced
        top_fractional = np.linalg.solve(np.asarray(reduced).T, top[ti].T).T
        top_fractional[:, axes] %= 1
        wrapped_top = top_fractional @ reduced
        shifts = np.asarray(list(product((-1, 0, 1), repeat=2))) @ reduced[axes]
        cloud = (wrapped_bottom[None] + shifts[:, None]).reshape((-1, 3))
        tree = cKDTree(cloud)
        neighbours = tree.query_ball_point(wrapped_top, alignment.tolerance_A)
        edge_rows: list[int] = []
        edge_columns: list[int] = []
        for i, values in enumerate(neighbours):
            edge_rows.extend([i] * len(values))
            edge_columns.extend(int(j) % len(bi) for j in values)
        if not edge_rows:
            continue
        pairs = np.unique(np.column_stack([edge_rows, edge_columns]), axis=0)
        row, col = pairs.T
        delta, lengths = find_mic(top[ti[row]] - moved[bi[col]], snapshot.cell, pbc=lateral_pbc)
        valid = lengths <= alignment.tolerance_A
        row, col, delta, lengths = row[valid], col[valid], delta[valid], lengths[valid]
        graph = coo_matrix(
            (np.ones(len(row)), (row, len(ti) + col)), shape=(len(ti) + len(bi),) * 2
        )
        _, labels = connected_components(graph, directed=False)
        for component in np.unique(labels[row]):
            selected = labels[row] == component
            rr, cc = np.unique(row[selected]), np.unique(col[selected])
            # Complete geometric matching is usually a set of tiny components.
            # A pathological tolerance must not allocate an unbounded matrix.
            if len(rr) * (len(cc) + len(rr)) > 4_000_000:
                raise ValueError(
                    "Matching tolerance connects too many atoms. Reduce the tolerance."
                )
            unmatched_penalty = float(len(rr) + 1)
            cost = np.full((len(rr), len(cc) + len(rr)), 2 * unmatched_penalty)
            local_r, local_c = (
                np.searchsorted(rr, row[selected]),
                np.searchsorted(cc, col[selected]),
            )
            cost[local_r, local_c] = (lengths[selected] / alignment.tolerance_A) ** 2
            cost[:, len(cc) :] = unmatched_penalty
            rmatch, cmatch = linear_sum_assignment(cost)
            keep = cmatch < len(cc)
            rt, cb = rr[rmatch[keep]], cc[cmatch[keep]]
            d, _ = find_mic(top[ti[rt]] - moved[bi[cb]], snapshot.cell, pbc=lateral_pbc)
            matches_top.extend(ti[rt])
            matches_bottom.extend(bi[cb])
            residuals.extend(d)
    return (
        np.asarray(matches_top, int),
        np.asarray(matches_bottom, int),
        np.asarray(residuals).reshape((-1, 3)),
        top,
        moved,
    )


def evaluate_interface_alignment(
    reference: InterfaceReference, alignment: InterfaceAlignment
) -> AlignmentDiagnostics:
    validate_alignment(reference, alignment)
    reference = reference.materialize()
    top_coverage, bottom_coverage, rmsds, indices = [], [], [], []
    for snapshot in reference.snapshots:
        check_cancelled()
        ti, bi, delta, top, bottom = _match_snapshot(reference, snapshot, alignment)
        top_coverage.append(len(ti) / max(1, len(top)))
        bottom_coverage.append(len(bi) / max(1, len(bottom)))
        rmsds.append(float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))) if len(delta) else None)
        indices.append((ti, bi))
    return AlignmentDiagnostics(
        tuple(top_coverage), tuple(bottom_coverage), tuple(rmsds), tuple(indices)
    )


def suggest_interface_alignment(
    reference: InterfaceReference,
    *,
    depth_A: float = DEFAULT_DEPTH_A,
    tolerance_A: float = DEFAULT_TOLERANCE_A,
) -> list[InterfaceAlignment]:
    reference = reference.materialize()
    snapshot = reference.snapshots[0]
    (top, ts, _), (bottom, bs, _) = snapshot_sides(reference, snapshot, depth_A)
    if not len(top) or not len(bottom):
        return []
    axes = list(reference.lateral_indices)
    basis = snapshot.cell[np.ix_(axes, axes)]
    inv = np.linalg.inv(basis)
    top_fractional = ((top[:, axes] - snapshot.origin[axes]) @ inv) % 1
    bottom_fractional = ((bottom[:, axes] - snapshot.origin[axes]) @ inv) % 1
    axis_index = "xyz".index(reference.axis)
    resolution = 128
    candidates = []
    for matrix in lattice_symmetries(reference):
        check_cancelled()
        moved = (bottom_fractional @ matrix) % 1
        correlation = np.zeros((resolution, resolution))
        for element in np.intersect1d(ts, bs):
            heights = np.unique(np.floor(top[ts == element, axis_index] / tolerance_A).astype(int))
            for height in heights:
                a = (ts == element) & (
                    np.abs(top[:, axis_index] - height * tolerance_A) <= tolerance_A
                )
                b = (bs == element) & (
                    np.abs(bottom[:, axis_index] - height * tolerance_A) <= tolerance_A
                )
                ha = np.histogram2d(*top_fractional[a].T, bins=resolution, range=((0, 1), (0, 1)))[
                    0
                ]
                hb = np.histogram2d(*moved[b].T, bins=resolution, range=((0, 1), (0, 1)))[0]
                correlation += np.fft.ifft2(np.fft.fft2(ha) * np.conj(np.fft.fft2(hb))).real
        seeds = [np.zeros(2)]
        peaks = np.argsort(correlation.ravel())[-8:][::-1]
        seeds.extend(np.column_stack(np.unravel_index(peaks, correlation.shape)) / resolution)
        # A specific atom-pair offset gives subpixel starts even for large cells.
        for i in np.argsort(top[:, axis_index])[:8]:
            same = np.flatnonzero(bs == ts[i])
            same = same[np.argsort(np.abs(bottom[same, axis_index] - top[i, axis_index]))[:8]]
            seeds.extend((top_fractional[i] - moved[same]) % 1)
        seen = set()
        for seed in seeds:
            check_cancelled()
            shift = np.asarray(seed).copy()
            for _ in range(12):
                alignment = InterfaceAlignment(
                    reference.reference_id,
                    _matrix_tuple(matrix),
                    (float(shift[0] % 1), float(shift[1] % 1)),
                    depth_A,
                    tolerance_A,
                )
                ti, _, delta, _, _ = _match_snapshot(reference, snapshot, alignment)
                if not len(ti):
                    break
                correction = np.median(delta[:, axes], axis=0) @ inv
                shift = (shift + correction) % 1
                if np.linalg.norm(correction @ basis) < 1e-6:
                    break
            key = tuple(np.round(shift % 1, 5) % 1)
            if key in seen:
                continue
            seen.add(key)
            alignment = replace(alignment, translation=tuple(shift % 1))
            diagnostics = evaluate_interface_alignment(reference, alignment)
            candidates.append(replace(alignment, diagnostics=diagnostics.to_dict()))

    def rank(alignment):
        d = alignment.diagnostics
        coverage = min(d["top_coverage"] + d["bottom_coverage"])
        rmsds = [x for x in d["rmsd_A"] if x is not None]
        return (-coverage, np.mean(rmsds) if rmsds else float("inf"))

    candidates.sort(key=rank)
    if candidates:
        best_coverage, best_rmsd = rank(candidates[0])
        close = [
            c
            for c in candidates
            if abs(rank(c)[0] - best_coverage) <= 0.01 and abs(rank(c)[1] - best_rmsd) < 0.05
        ]
        candidates = [
            replace(c, diagnostics={**(c.diagnostics or {}), "ambiguous": len(close) > 1})
            for c in candidates
        ]
    return candidates[:8]


def alignment_for_reference(
    reference: InterfaceReference | None, settings: Mapping[str, Any] | None
) -> InterfaceAlignment | None:
    if not settings:
        return None
    if reference is None:
        return None
    value = settings.get(reference.reference_id)
    if value is None:
        return None
    alignment = InterfaceAlignment.from_dict(value)
    validate_alignment(reference, alignment)
    if not alignment.allow_mismatches:
        # Lazy file references have a reliable revision. Mutable Python snapshots
        # deliberately retain fresh-data fitting semantics.
        key = None
        if reference.source_path and not reference.snapshots:
            key = (
                source_revision(reference.source_path),
                reference.reference_id,
                reference.version,
                hashlib.sha256(
                    np.ascontiguousarray(reference.cells, dtype=float).tobytes()
                ).digest(),
                alignment.matrix,
                alignment.translation,
                alignment.depth_A,
                alignment.tolerance_A,
            )
        diagnostics = PREVIEW_CACHE.get("interface.fit", key) if key is not None else None
        LOGGER.debug("Interface fit cache %s", "hit" if diagnostics is not None else "miss")
        if diagnostics is None:
            from ..plot.gui_session import report_progress
            report_progress("Checking interface alignment")
            started = perf_counter()
            diagnostics = evaluate_interface_alignment(reference, alignment)
            LOGGER.debug("Interface fit validation seconds=%.6f", perf_counter() - started)
            check_cancelled()
            if key is not None and key[0] == source_revision(reference.source_path):
                PREVIEW_CACHE.put("interface.fit", key, diagnostics)
        if not diagnostics.good_match:
            raise ValueError(
                "The interfaces do not match. Review the alignment and acknowledge the mismatches."
            )
    return alignment


def alignment_metadata(reference: InterfaceReference | None) -> dict[str, Any]:
    return {} if reference is None else {"interface_reference": reference.metadata()}


def align_position_profile(profile: Any, alignment: InterfaceAlignment) -> Any:
    """Return derived coordinates and seam flags without mutating analysis arrays."""
    if profile.interface_reference is None or profile.interface_side is None:
        raise ValueError(RECOMPUTE_MESSAGE)
    validate_alignment(profile.interface_reference, alignment)
    if profile.cell_matrices_angstrom is None:
        raise ValueError(
            "Position alignment requires stored per-frame cell geometry. Recompute position."
        )
    positions = np.stack([profile.x, profile.y, profile.z], axis=-1).astype(float)
    side = np.asarray(profile.interface_side)
    crossings = np.zeros(side.shape, dtype=bool)
    if profile.periodic_crossings is not None:
        crossings |= profile.periodic_crossings
    fractional = np.full(side.shape + (2,), np.nan)
    axes = list(profile.interface_reference.lateral_indices)
    for index, cell in enumerate(profile.cell_matrices_angstrom):
        check_cancelled()
        origin = (
            np.zeros(3)
            if profile.cell_origins_angstrom is None
            else profile.cell_origins_angstrom[index]
        )
        # Validate every frame, not just the reference snapshots.
        validate_alignment(
            replace(profile.interface_reference, cells=np.asarray([cell])), alignment
        )
        bottom = side[index] < 0
        positions[index, bottom] = apply_interface_alignment(
            positions[index, bottom], cell, origin, profile.axis, alignment
        )
        positions[index, side[index] == 0] = np.nan
        basis = np.asarray(cell)[np.ix_(axes, axes)]
        fractional[index] = (positions[index][:, axes] - origin[axes]) @ np.linalg.inv(basis)
    crossings[1:] |= np.any(np.abs(np.diff(fractional, axis=0)) > 0.5, axis=-1)
    crossings[1:] |= side[1:] != side[:-1]
    crossings[1:] |= ~np.all(np.isfinite(fractional[1:]) & np.isfinite(fractional[:-1]), axis=-1)
    metadata = {
        **alignment.to_dict(),
        "excluded_ambiguous_samples": int(np.count_nonzero(side == 0)),
    }
    return replace(
        profile,
        x=positions[..., 0],
        y=positions[..., 1],
        z=positions[..., 2],
        distance_to_surface=np.where(side != 0, profile.distance_to_surface, np.nan),
        periodic_crossings=crossings,
        interface_alignment_metadata=metadata,
    )
