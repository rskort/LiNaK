"""Standalone diagnostic figure for a persisted electrostatic reference."""

import json
from pathlib import Path

import numpy as np


def plot_potential_diagnostics(source: str | Path, output: str | Path) -> Path:
    from .potential import load_potential_records
    from ..plot.rendering import PlotFigure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    records = load_potential_records(source, include_profiles=True)
    if not records:
        raise ValueError("No potential records to diagnose.")
    fig = PlotFigure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    profile_ax, time_ax, sensitivity_ax, quality_ax = axes.flat
    u = np.linspace(0, 1, 401)
    raw_profiles, aligned_profiles, selected_profiles = [], [], []
    for record in records:
        profile = record.profile
        if profile is None or "solvent_s_A" not in profile:
            continue
        x = profile["solvent_s_A"] / profile["gap_A"][0]
        mask = (x >= 0) & (x <= 1)
        order = np.argsort(x[mask])
        xx, vv = x[mask][order], profile["hartree_ev"][mask][order]
        if len(xx) < 2:
            continue
        # Interpolate only to align changing cell geometries. No smoothing,
        # per-snapshot zeroing, or plateau optimization is applied.
        raw_profiles.append(np.interp(u, xx, vv, left=np.nan, right=np.nan))
        if record.efermi_ev is not None:
            aligned_profiles.append(
                np.interp(u, xx, vv - record.efermi_ev, left=np.nan, right=np.nan)
            )
        selected_profiles.append(
            np.interp(u, xx, profile["selected"][mask][order], left=0, right=0)
        )

    def finite_mean(values):
        values = np.asarray(values)
        count = np.isfinite(values).sum(axis=0)
        return np.divide(
            np.nansum(values, axis=0), count, out=np.full(len(u), np.nan), where=count > 0
        )

    if raw_profiles:
        profile_ax.plot(u, finite_mean(raw_profiles), label="Mean raw Hartree (CP2K gauge)")
        if aligned_profiles:
            profile_ax.plot(u, finite_mean(aligned_profiles), label="Mean Hartree − matching Fermi")
        selected_fraction = np.mean(selected_profiles, axis=0)
        profile_ax.fill_between(
            u,
            0,
            1,
            where=selected_fraction > 0,
            transform=profile_ax.get_xaxis_transform(),
            color="green",
            alpha=0.15,
            label="Selected region (union over snapshots)",
        )
    else:
        profile_ax.text(
            0.5,
            0.5,
            "No usable solvent geometry; inspect rejection reasons",
            ha="center",
            va="center",
            transform=profile_ax.transAxes,
        )
    profile_ax.set(
        xlabel="Fractional solvent-gap coordinate",
        ylabel="Energy (eV)",
        title="Time-averaged unsmoothed planar Hartree profile",
    )
    if raw_profiles:
        profile_ax.legend(fontsize=8)
    has_time = all(r.time_fs is not None for r in records)
    time = np.asarray([r.time_fs / 1000 if has_time else i + 1 for i, r in enumerate(records)])
    time_label = "Time (ps)" if has_time else "Snapshot index (input order)"
    raw = np.asarray([np.nan if r.window_mean_ev is None else r.window_mean_ev for r in records])
    accepted = np.asarray([r.reference_valid for r in records])
    time_ax.plot(time, np.where(accepted, raw, np.nan), label="Accepted water Hartree")
    time_ax.scatter(
        time[~accepted], raw[~accepted], marker="x", color="red", label="Rejected candidate"
    )
    cshe = [np.nan if r.electrode_cshe_ev is None else r.electrode_cshe_ev for r in records]
    time_ax.plot(time, cshe, label="cSHE potential")
    time_ax.set(
        xlabel=time_label,
        ylabel="Energy (eV)",
        title=f"Water reference vs time ({np.count_nonzero(~accepted)} rejected)",
    )
    time_ax.legend(fontsize=8)
    if not np.any(np.isfinite(raw)):
        time_ax.text(
            0.5,
            0.5,
            "No measurable candidate window",
            ha="center",
            va="center",
            transform=time_ax.transAxes,
        )
    for name in ("shift_minus", "shift_plus", "contract", "expand"):
        values, valid = [], []
        for record in records:
            candidate = json.loads(record.sensitivity_json or "{}").get(name, {})
            mean = candidate.get("mean_ev")
            values.append(
                np.nan
                if mean is None or record.window_mean_ev is None
                else mean - record.window_mean_ev
            )
            valid.append(candidate.get("valid", False))
        values, valid = np.asarray(values), np.asarray(valid)
        (line,) = sensitivity_ax.plot(time, np.where(valid, values, np.nan), label=name)
        sensitivity_ax.scatter(
            time[~valid], values[~valid], color=line.get_color(), marker="x", s=10
        )
    sensitivity_ax.axhline(0, color="grey", lw=0.6)
    sensitivity_ax.set(
        xlabel=time_label,
        ylabel="Δ reference / Δ cSHE (eV)",
        title="Declared window perturbations (crosses: invalid)",
    )
    sensitivity_ax.legend(fontsize=8)
    gradients = [
        np.nan if r.window_gradient_ev_A is None else r.window_gradient_ev_A for r in records
    ]
    quality_ax.plot(time, gradients, label="Gradient", color="tab:blue")
    config = json.loads(next((r.config_json for r in records if r.config_json), "{}"))
    limit = config.get("max_gradient_ev_ang")
    if limit is not None:
        for sign in (-1, 1):
            quality_ax.axhline(sign * limit, color="tab:blue", ls="--", lw=0.7)
    variation_ax = quality_ax.twinx()
    variation_ax.plot(
        time,
        [np.nan if r.window_std_ev is None else r.window_std_ev for r in records],
        color="tab:orange",
        label="Spatial std",
    )
    if config.get("max_std_ev") is not None:
        variation_ax.axhline(config["max_std_ev"], color="tab:orange", ls="--", lw=0.7)
    quality_ax.set(
        xlabel=time_label, ylabel="Gradient (eV/Å)", title="Electrostatic reference checks"
    )
    variation_ax.set_ylabel("Spatial standard deviation (eV)")
    quality_ax.legend(loc="upper left", fontsize=8)
    variation_ax.legend(loc="upper right", fontsize=8)
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return path
