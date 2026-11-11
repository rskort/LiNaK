"""Geometry-only water-reference plans and independent electrostatic checks.

No Hartree values enter planning. A periodic solvent gap is one connected
reservoir, including when it crosses the cube boundary; it is not split at
an arbitrary periodic cut into two competing references.
"""

from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np
from ase import Atoms

from .surface import estimate_surface_layout_reference, surface_substrate_mask
from .water import water_molecule_triplets, water_triplet_geometry

# Subtractive water-reference calibration; see docs/potential.md.
DEFAULT_CSHE_OFFSET_EV = 0.65


@dataclass(frozen=True)
class PotentialConfig:
    water_padding_ang: float = 5.0
    cshe_offset_ev: float = DEFAULT_CSHE_OFFSET_EV
    surface_layout: str = "auto"
    surface_mode: str = "auto"
    surface_elements: tuple[str, ...] | None = None
    boundary: str = "periodic"
    bulk_window: tuple[float, float] | None = None
    window_plan: dict | None = None
    density_bin_ang: float = 1.0
    density_bandwidth_ang: float = 1.5
    min_water_density: float = 0.015
    min_width_ang: float = 4.0
    min_grid_points: int = 5
    max_gradient_ev_ang: float = 0.05
    max_std_ev: float = 0.25
    max_range_ev: float = 1.0
    sensitivity_ang: float = 1.0
    fermi_map: dict[str, float] | None = None
    snapshot_interval_fs: float | None = None


@dataclass
class ReferenceGeometry:
    origin: float
    period: float
    area: float
    start: float
    length: float
    water_s: np.ndarray
    slab_bottom: float
    slab_top: float
    layout: str
    boundary: str

    def coordinate(self, z: np.ndarray) -> np.ndarray:
        s = np.asarray(z, dtype=float) - self.start
        return np.mod(s, self.period) if self.boundary == "periodic" else s


def serialize_config(config: PotentialConfig) -> str:
    payload = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name not in {"window_plan", "fermi_map"}
    }
    payload["fermi_policy"] = (
        "explicit-source-map" if config.fermi_map is not None else "source-specific-output"
    )
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def validate_config(config: PotentialConfig) -> None:
    for name in ("water_padding_ang", "sensitivity_ang"):
        value = getattr(config, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name in (
        "density_bin_ang",
        "density_bandwidth_ang",
        "min_water_density",
        "min_width_ang",
        "max_gradient_ev_ang",
        "max_std_ev",
        "max_range_ev",
    ):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.min_grid_points < 3:
        raise ValueError("min_grid_points must be at least three.")
    if not np.isfinite(config.cshe_offset_ev):
        raise ValueError("cshe_offset_ev must be finite.")
    if config.boundary not in {"periodic", "open"}:
        raise ValueError("boundary must be periodic or open.")
    if config.surface_layout not in {"auto", "single-top", "single-bottom", "symmetric"}:
        raise ValueError("Unsupported surface_layout.")
    if config.surface_mode not in {"auto", "rough", "layered"}:
        raise ValueError("Unsupported surface_mode.")
    if config.bulk_window is not None:
        if len(config.bulk_window) != 2 or not np.all(np.isfinite(config.bulk_window)):
            raise ValueError("bulk_window requires two finite Cartesian z coordinates.")
        if config.water_padding_ang != 0:
            raise ValueError("A fixed bulk_window cannot be combined with water padding.")
    if config.snapshot_interval_fs is not None and (
        not np.isfinite(config.snapshot_interval_fs) or config.snapshot_interval_fs <= 0
    ):
        raise ValueError("snapshot_interval_fs must be finite and positive.")


def reference_geometry(atoms: Atoms, config: PotentialConfig) -> ReferenceGeometry:
    """Locate a contiguous substrate and its solvent gap, using all substrate atoms.

    The full outer envelope is conservative for terraces, steps, reconstruction,
    and fixed atoms. Shared surface detection supplies chemical membership, not
    a mean plane that might cut through a step.
    """
    cell = np.asarray(atoms.cell)
    if (
        not np.all(np.isfinite(cell))
        or np.linalg.det(cell) <= 0
        or not np.allclose(cell[:2, 2], 0, atol=1e-8)
        or not np.allclose(cell[2, :2], 0, atol=1e-8)
    ):
        raise ValueError(
            "A planar Cartesian-Z profile requires a positive cell with Z decoupled from XY."
        )
    origin = float(np.asarray(atoms.get_celldisp()).reshape(-1)[2])
    period = float(cell[2, 2])
    atoms = atoms.copy()
    atoms.pbc = [True, True, config.boundary == "periodic"]
    reference = estimate_surface_layout_reference(
        [atoms],
        axis="z",
        surface_layout=config.surface_layout,
        mode=config.surface_mode,
        surface_elements=config.surface_elements,
        _log_summary=False,
    )
    substrate = (
        np.isin(atoms.get_chemical_symbols(), config.surface_elements)
        if config.surface_elements
        else surface_substrate_mask(reference, atoms)
    )
    if np.count_nonzero(substrate) < 2:
        raise ValueError("No identifiable slab substrate; specify --surface-elements.")
    triplets = water_molecule_triplets(atoms)
    if len(triplets) < 2:
        raise ValueError("Insufficient complete H2O topology to identify a water reservoir.")
    water_z = water_triplet_geometry(atoms, triplets).com_positions[:, 2]
    metal_z = atoms.positions[substrate, 2]
    if config.boundary == "periodic":
        phase = np.sort(np.mod(metal_z - origin, period))
        gaps = np.diff(np.r_[phase, phase[0] + period])
        index = int(np.argmax(gaps))
        cut = (phase[index] + gaps[index] / 2) % period
        q = (metal_z - origin - cut) % period
        bottom, top = float(q.min()), float(q.max())
        thickness = top - bottom
        if thickness <= 0 or thickness >= period:
            raise ValueError("Degenerate slab envelope.")
        start = origin + ((top + cut) % period)
        length = period - thickness
        water_s = (water_z - start) % period
        band = min(6.0, length / 2)
        threshold = max(2, 0.1 * len(water_s))
        top_contact = np.count_nonzero((water_s > 0) & (water_s < band)) >= threshold
        bottom_contact = (
            np.count_nonzero((water_s < length) & (water_s > length - band)) >= threshold
        )
        layout = (
            "symmetric"
            if top_contact and bottom_contact
            else (
                "single-top" if top_contact else "single-bottom" if bottom_contact else "unresolved"
            )
        )
        slab_bottom = origin + ((bottom + cut) % period)
        slab_top = start
    else:
        bottom, top = float(metal_z.min()), float(metal_z.max())
        below, above = np.count_nonzero(water_z < bottom), np.count_nonzero(water_z > top)
        layout = (
            "symmetric"
            if below >= 2 and above >= 2
            else ("single-top" if above >= 2 else "single-bottom" if below >= 2 else "unresolved")
        )
        side = config.surface_layout
        if config.bulk_window is not None:
            mid = float(np.mean(config.bulk_window))
            side = "single-top" if mid > top else "single-bottom" if mid < bottom else "unresolved"
        elif layout != "symmetric":
            side = layout
        if side not in {"single-top", "single-bottom"}:
            raise ValueError(
                "Two disconnected open-boundary reservoirs: select a fixed --bulk-window."
            )
        start, length = (
            (top, origin + period - top) if side == "single-top" else (origin, bottom - origin)
        )
        water_s = water_z - start
        slab_bottom, slab_top = bottom, top
    if length <= 0 or layout == "unresolved":
        raise ValueError("No water contact with a trustworthy slab geometry.")
    if config.surface_layout != "auto" and layout != config.surface_layout:
        raise ValueError(
            f"Requested layout {config.surface_layout} conflicts with detected {layout}."
        )
    return ReferenceGeometry(
        origin,
        period,
        float(abs(np.linalg.det(cell)) / period),
        start,
        length,
        water_s[(water_s >= 0) & (water_s <= length)],
        slab_bottom,
        slab_top,
        layout,
        config.boundary,
    )


def _selection_parameters(config: PotentialConfig) -> dict:
    return {
        name: getattr(config, name)
        for name in (
            "water_padding_ang",
            "bulk_window",
            "boundary",
            "surface_layout",
            "surface_mode",
            "surface_elements",
            "density_bin_ang",
            "density_bandwidth_ang",
            "min_water_density",
            "min_width_ang",
        )
    }


def build_reference_plan(geometries: list[ReferenceGeometry], config: PotentialConfig) -> dict:
    """Create one immutable plan for a batch without inspecting any potential."""
    validate_config(config)
    if config.window_plan is not None:
        plan = dict(config.window_plan)
        if plan.get("version") != 1:
            raise ValueError("Unsupported reference-plan version.")
        if json.dumps(plan["parameters"], sort_keys=True) != json.dumps(
            _selection_parameters(config), sort_keys=True
        ):
            raise ValueError(
                "Reference-plan selection parameters differ; reuse the original parameters or make a new plan."
            )
        payload = {key: value for key, value in plan.items() if key != "id"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()[:20]
        if plan.get("id") != digest:
            raise ValueError("Reference plan checksum does not match its contents.")
        return plan
    plan = {
        "version": 1,
        "parameters": _selection_parameters(config),
        "reason": "",
        "mode": "fixed" if config.bulk_window is not None else "auto",
    }
    if config.bulk_window is not None:
        plan.update(bounds=list(config.bulk_window), coordinate="cartesian-z")
    elif not geometries:
        plan.update(bounds=None, reason="No structurally valid snapshots for window planning.")
    else:
        length = float(np.median([g.length for g in geometries]))
        n_bins = max(1, int(np.ceil(length / config.density_bin_ang)))
        edges = np.linspace(0, 1, n_bins + 1)
        densities = []
        for g in geometries:
            density = np.histogram(g.water_s / g.length, bins=edges)[0] / (
                g.area * g.length / n_bins
            )
            # A fixed structural density kernel distinguishes molecular-scale
            # sampling holes from distinct water/vacuum reservoirs. This kernel
            # NEVER acts on the Hartree profile or on the temporal trajectory.
            sigma = config.density_bandwidth_ang / (g.length / n_bins)
            radius = max(1, int(np.ceil(4 * sigma)))
            kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
            kernel /= kernel.sum()
            densities.append(np.convolve(density, kernel, mode="full")[radius : radius + n_bins])
        densities = np.asarray(densities)
        mean_density = np.round(densities.mean(axis=0), 12)
        supported = mean_density >= config.min_water_density
        transitions = np.diff(np.r_[False, supported, False].astype(int))
        runs = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
        # Multiple reservoirs are not interchangeable references. Do not rank
        # them by width, instantaneous potential, or discovery order.
        if len(runs) != 1:
            plan.update(
                bounds=None,
                reason=f"Expected one connected water reservoir; found {len(runs)}. Specify --bulk-window.",
            )
        else:
            first, last = runs[0]
            lo = float(edges[first] + config.water_padding_ang / length)
            hi = float(edges[last] - config.water_padding_ang / length)
            reason = (
                ""
                if (hi - lo) * length >= config.min_width_ang
                else (
                    "Requested padding leaves an empty or insufficiently wide bulk-water window; padding was not reduced."
                )
            )
            plan.update(
                bounds=[lo, hi],
                water_support_bounds_u=[float(edges[first]), float(edges[last])],
                reason=reason,
            )
        plan.update(
            coordinate="fractional-solvent-gap",
            reference_gap_A=length,
            density_u_edges=edges.tolist(),
            mean_water_density_A3=mean_density.tolist(),
        )
    encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
    plan["id"] = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    return plan


def evaluate_reference(
    z: np.ndarray,
    potential_ev: np.ndarray,
    geometry: ReferenceGeometry,
    plan: dict,
    config: PotentialConfig,
    *,
    perturbation: tuple[float, float] = (0, 0),
) -> tuple[dict, np.ndarray]:
    """Apply a geometric interval, returning diagnostics and integration weights."""
    reasons = []
    result = {
        "valid": False,
        "reasons": reasons,
        "requested_padding_A": config.water_padding_ang,
        "plan_id": plan["id"],
        "mean_ev": None,
        "std_ev": None,
        "range_ev": None,
        "gradient_ev_A": None,
        "grid_points": 0,
        "min_A": None,
        "max_A": None,
        "width_A": 0.0,
        "water_density_A3": None,
    }
    result["window_coordinate"] = (
        "unwrapped Cartesian z (A)"
        if plan["mode"] == "fixed"
        else "distance from positive-Z substrate envelope (A)"
    )
    if not np.all(np.isfinite(potential_ev)):
        reasons.append("Nonfinite Hartree values in the cube field.")
    empty = np.zeros(len(z), dtype=float)
    if plan.get("reason") or plan.get("bounds") is None:
        reasons.append(plan.get("reason") or "No selected window.")
        return result, empty
    if plan["mode"] == "fixed":
        lo, end = plan["bounds"]
        width = (end - lo) % geometry.period if geometry.boundary == "periodic" else end - lo
        if not np.isfinite(width) or abs(end - lo) > geometry.period:
            reasons.append("Fixed window exceeds the cell period or has nonfinite extent.")
            return result, empty
        hi = lo + width + perturbation[1]
        lo += perturbation[0]
        x = lo + ((z - lo) % geometry.period) if geometry.boundary == "periodic" else z
        water_z = geometry.start + geometry.water_s
        water_x = (
            lo + ((water_z - lo) % geometry.period) if geometry.boundary == "periodic" else water_z
        )
    else:
        support_lo, support_hi = np.asarray(plan["water_support_bounds_u"]) * geometry.length
        # Follow the same structural support, but preserve padding in Angstrom
        # when the cell/gap changes. Fractional padding would silently shrink.
        lo = support_lo + config.water_padding_ang + perturbation[0]
        hi = support_hi - config.water_padding_ang + perturbation[1]
        x = geometry.coordinate(z)
        water_x = geometry.water_s
    result.update(
        requested_min_A=float(lo), requested_max_A=float(hi), requested_width_A=float(hi - lo)
    )
    if hi <= lo:
        reasons.append("Empty or reversed averaging window.")
        return result, empty
    order = np.argsort(x)
    grid_x = x[order]
    if len(grid_x) < 2 or np.any(np.diff(grid_x) <= 0):
        reasons.append(
            "Grid coordinates do not resolve distinct samples at the selected window origin."
        )
        return result, empty
    grid_values = np.asarray(potential_ev)[order]
    grid_indices = np.asarray(order)
    if geometry.boundary == "periodic":
        grid_x = np.r_[grid_x[-1] - geometry.period, grid_x, grid_x[0] + geometry.period]
        grid_values = np.r_[grid_values[-1], grid_values, grid_values[0]]
        grid_indices = np.r_[grid_indices[-1], grid_indices, grid_indices[0]]
    if lo < grid_x[0] or hi > grid_x[-1]:
        reasons.append("Fixed interval extends beyond available nonperiodic cube samples.")
        return result, empty
    xx = np.r_[lo, grid_x[(grid_x > lo) & (grid_x < hi)], hi]
    right = np.clip(np.searchsorted(grid_x, xx, side="right"), 1, len(grid_x) - 1)
    left = right - 1
    fraction = np.clip((xx - grid_x[left]) / (grid_x[right] - grid_x[left]), 0, 1)
    vv = (1 - fraction) * grid_values[left] + fraction * grid_values[right]
    dx = np.diff(xx)
    quadrature = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2] / (hi - lo)
    weights = np.zeros(len(z), dtype=float)
    np.add.at(weights, grid_indices[left], quadrature * (1 - fraction))
    np.add.at(weights, grid_indices[right], quadrature * fraction)
    selected = weights > 1e-14
    result["grid_points"] = int(selected.sum())
    result["quadrature"] = "trapezoidal; linear interpolation only at fixed interval boundaries"
    if selected.sum() < config.min_grid_points:
        reasons.append(f"Insufficient grid points: {selected.sum()} < {config.min_grid_points}.")
    if np.any(selected):
        result.update(min_A=float(lo), max_A=float(hi), width_A=float(hi - lo))
        raw_lo = geometry.start + lo if plan["mode"] == "auto" else lo
        if geometry.boundary == "periodic":
            raw_lo = geometry.origin + (raw_lo - geometry.origin) % geometry.period
        raw_hi = raw_lo + hi - lo
        upper = geometry.origin + geometry.period
        result["raw_z_intervals_A"] = (
            [[geometry.origin, raw_hi - geometry.period], [raw_lo, upper]]
            if raw_hi > upper and geometry.boundary == "periodic"
            else [[raw_lo, raw_hi]]
        )
        if np.all(np.isfinite(vv)):
            mean = float(quadrature @ vv)
            result.update(
                mean_ev=mean,
                std_ev=float(np.sqrt(quadrature @ (vv - mean) ** 2)),
                range_ev=float(np.ptp(vv)),
            )
            if len(xx) >= 2 and np.ptp(xx) > 0:
                centered = xx - quadrature @ xx
                result["gradient_ev_A"] = float(
                    (quadrature * centered) @ (vv - mean) / (quadrature @ centered**2)
                )
        else:
            reasons.append("Nonfinite Hartree values in the selected window.")
        # Every grid point must be in solvent, including wrapped manual windows.
        s = np.r_[
            geometry.coordinate(z[(x >= lo) & (x <= hi)]),
            geometry.coordinate(np.asarray([raw_lo, raw_hi])),
        ]
        if np.any((s < 0) | (s > geometry.length)):
            reasons.append("Window overlaps the substrate or lies outside the solvent gap.")
        clearance = float(min(np.min(s), geometry.length - np.max(s)))
        result["substrate_clearance_A"] = clearance
        if clearance + 1e-8 < config.water_padding_ang:
            reasons.append(
                "Structural change violates the requested substrate clearance; window was not moved."
            )
    if result["width_A"] + 1e-8 < config.min_width_ang:
        reasons.append(
            f"Insufficient sampled width: {result['width_A']:.3g} < {config.min_width_ang:g} A."
        )
    n_water = np.count_nonzero((water_x >= lo) & (water_x <= hi))
    density = float(n_water / (geometry.area * (hi - lo)))
    result["water_density_A3"] = density
    if density < config.min_water_density:
        reasons.append(
            f"Insufficient water density: {density:.4g} < {config.min_water_density:g} molecules/A^3."
        )
    n_bins = max(1, int(np.ceil((hi - lo) / config.density_bin_ang)))
    occupancy = np.histogram(water_x, bins=np.linspace(lo, hi, n_bins + 1))[0] > 0
    result["water_occupied_fraction"] = float(occupancy.mean())
    if occupancy.mean() < 0.8:
        reasons.append("Water does not occupy at least 80% of structural bins in the fixed window.")
    for field, limit in (("std_ev", config.max_std_ev), ("range_ev", config.max_range_ev)):
        if result[field] is not None and result[field] > limit:
            reasons.append(f"Hartree {field} {result[field]:.4g} exceeds {limit:g} eV.")
    slope = result["gradient_ev_A"]
    if slope is not None and abs(slope) > config.max_gradient_ev_ang:
        reasons.append(f"Hartree gradient {slope:.4g} exceeds {config.max_gradient_ev_ang:g} eV/A.")
    result["valid"] = not reasons
    return result, weights


def sensitivity_windows(z, values, geometry, plan, config) -> dict:
    """Evaluate declared window perturbations without optimizing the reference."""
    d = config.sensitivity_ang
    return {
        name: evaluate_reference(z, values, geometry, plan, config, perturbation=shift)[0]
        for name, shift in {
            "shift_minus": (-d, -d),
            "shift_plus": (d, d),
            "contract": (d, -d),
            "expand": (-d, d),
        }.items()
    }
