# LiNaK Method Documentation

This folder explains how LiNaK computes its analysis results. It is written for
users who want to understand the numerical meaning of the stored data, the
assumptions made by each workflow, and what exactly is written into the HDF5
outputs.

This is not intended to duplicate CLI help. For command syntax and options, use
the main project [README](../README.md) or `linak --help`. This folder focuses
on method transparency.

## What This Documentation Covers

The pages in this folder describe:

- what quantity LiNaK computes
- which atoms, molecules, or grid values are selected
- which coordinate system is used
- how periodic boundary conditions are handled
- which metadata each current analysis schema requires
- which metadata and datasets are written to HDF5
- how shared helper logic such as water detection and surface estimation works

## Analysis Guides

- [Nearest-atom surface distances](nearest_surface_distances.md): optional 3D
  substrate distances, inherited conversion preferences, interface identity,
  periodic searches, and density/distribution definitions.
- [Density and distance distributions](density-distributions.md): physical units,
  nearest-distance histograms, valid Studio/CLI axes and benchmark results.

- [Density](density.md)
- [Position](position.md)
- [RDF](rdf.md)
- [Coordination](coordination.md)
- [Potential](potential.md)
- [Orientation](orientation.md)
- [Temperature](temperature.md)

## Shared Method Guides

- [Surface Estimation](surface-estimation.md)
- [Water Detection And Water Geometry](water-detection.md)
- [HDF5 Data Model And Metadata Conventions](hdf5-data-model.md)

## Common Conventions

### Units

- Distances are stored in `Angstrom` unless noted otherwise.
- Time is stored as both `fs` and `ps` when a timestep is available.
- Density-like quantities may be volumetric or linear, depending on whether a
  usable periodic cell is available.

### Profiles

LiNaK stores analysis results as HDF5 profiles. A file may contain:

- one profile, for a single analysis result
- many profiles, for example when using multi-species output or combined HDF5
  files

Each stored profile has its own metadata and stable `profile_uid`.

Schema versions belong to individual analyses: density, position, coordination,
orientation, and potential tables use v2; RDF and temperature use v1. A schema number is not the
LiNaK package version. Unsupported schemas or missing required data produce an
actionable error; recompute from the source trajectory. Existing files are not
automatically migrated, rewritten, or deleted.

The supported analysis workflow is compute first, plot second:

```bash
linak compute density /path/to/traj.h5 --species O --axis z
linak plot /path/to/traj.density.h5
```

All file extensions are invariant when resolving collisions. Counters go before
the first dot: `run.part.rdf.h5` becomes `run_1.part.rdf.h5`.
Qualifiers also belong to the base name: use `traj-all.position.h5` and `traj_1.density.h5`, never
`traj.position-all.h5` or `traj.density_1.h5`.

Repeated `compute position` commands merge compatible species/axis profiles into
the requested `.position.h5` file. For example, computing O and then H keeps both
in `traj.position.h5`; recomputing O updates its profile. Source, frame/time grid,
cell geometry, or surface incompatibilities produce `traj_1.position.h5` instead.
Saved plot settings are preserved when profiles are merged.

For repeated analysis, first convert raw text trajectories into LiNaK trajectory
HDF5:

```bash
linak apply convert /path/to/traj.xyz --input /path/to/input.inp
# Without a simulation input: supply the interval between saved frames in fs.
linak apply convert /path/to/traj.xyz --timestep 2.5
linak compute density /path/to/LiNaK_outputs/traj.traj.h5 --species O --axis z
```

Converted `*.traj.h5` files are preprocessed trajectory containers. They keep
resolved input metadata when available, store PBC-wrapped coordinates when a
valid cell is resolved, and cache the default frame-wise surface reference
(`z`, `auto`, automatic surface elements). Surface-aware analyses reuse that
cached surface when their requested surface settings match; explicit different
surface settings recompute the surface instead.

## Plot Studio Notes

When you open an analysis HDF5 with `linak plot`, LiNaK Studio edits a saved
plot profile rather than the underlying analysis arrays. In practice this means:

- `Layers` changes how each stored series is rendered
- `Annotations` adds figure-level text, line, and arrow overlays
- `Figure` changes axes, legend, styling, and export presentation
- named plot profiles round-trip through the HDF5 file

Per-layer visibility, raw-line visibility, fits, uncertainty, cumulative lines,
and annotations are all previewed live and are persisted as part of the
plot profile. Turning a raw line off hides only that base line; other derived
overlays for the same enabled layer can still render if they are turned on.

For `position` plots, Studio exposes `2D Heatmap` as a first-class view rather
than only the legacy `xy-z` controls. The view lets the user choose the X and Y
quantities, color continuously by a coordinate or time quantity, or assign one
editable categorical color per atom or molecule trajectory. In `Color mode:
species / layer`, trajectories are listed under collapsible species headings;
each has its own visibility, color, label, and line style. Search by species,
atom ID, or source using `Filter list`, which changes the list without changing
the plot. `Show matches` shows matching trajectories and hides all others;
`Hide matches` hides matching trajectories only. Layer checkboxes work on the
first click without selecting the layer. Visibility and copy actions stay above
the list. Custom aggregation groups are no longer supported, including in saved
profiles. Species checkboxes control
all their trajectories; `Isolate` shows just the selected trajectory, with Undo
restoring the previous visibility. `Apply this style to species` copies the
selected trajectory's styling to its species without changing labels or
visibility. Hiding layers preserves their order and colors.

Density layers use the same visibility and styling controls for elements and
molecules. H2O can be shown independently of elemental O and H, including after
`Hide all`. Layer colors stay attached to their IDs when layers are hidden,
reordered, or saved; the list, legend, preview and exported figure agree.
Duplicated layers retain their source identity and saved settings.

Optional value filters and the
shared colormap/colorbar controls update the preview interactively. The legacy
`xy-z` name is still accepted as a compatibility alias in the CLI and saved
profiles.

### Combined HDF5 Files

For combined HDF5 files, LiNaK does not merge the underlying arrays into one
physical dataset. It stores multiple profiles in one file. Plotting can then
render those profiles together as multiple series.

### Metadata-Driven Reproducibility

LiNaK writes:

- analysis name
- schema version
- units map
- profile metadata such as species, axis, frame count, cutoff, and coordinate
  mode

This metadata is part of the intended reproducibility story: the file should
describe not only the computed values, but also how those values should be
interpreted.

## Source Modules

The current implementations described in this folder are primarily defined in:

- `src/linak/analysis/common.py`
- `src/linak/analysis/density.py`
- `src/linak/analysis/position.py`
- `src/linak/analysis/rdf.py`
- `src/linak/analysis/coordination.py`
- `src/linak/analysis/potential.py`
- `src/linak/analysis/orientation.py`
- `src/linak/analysis/temperature.py`
- `src/linak/analysis/surface.py`
- `src/linak/analysis/water.py`
- `src/linak/analysis/schema.py`

### Export plotted data

In the plot settings window, use **Data ? Export Data** (or **Save Data** in the
preview) to save line data or a 2D density/orientation heatmap as CSV, TSV, DAT,
or TXT. Heatmap exports contain one row per grid cell with `x`, `y`, `value`,
`x_min`, `x_max`, `y_min`, and `y_max`. The coordinates are bin centers; the
boundaries preserve nonuniform bins. Rows run through Y bins for each X bin.
Missing cells retain `nan` values so the grid can be reconstructed.

The export uses the selected plot's processed grid. Color limits and logarithmic
color scaling do not change the exported numeric values. Enable **Include metadata
comments** to include grid dimensions and axis/value labels.

### Advanced plot options

The **Advanced** page groups extra settings into **Plot details**, **Canvas,
layout and export**, and **Matplotlib defaults**. Each optional JSON field
explains its purpose and shows errors next to the editor. **Format JSON**
indents valid input; **Use example** fills an empty field; **Clear** removes
its contents. These edits support Ctrl+Z. Shared settings remain synchronized
with the regular Figure controls, and saved profiles retain the same format.

### Plot window messages

The footer shows the latest update, warning, or error on one line. Click it (or
focus it and press Enter) to open **Advanced → Logs**, which keeps the full
message history for that window session. The log viewer includes timestamps,
severity, and exception details, with **Copy all** and **Clear** controls.
Terminal logging continues to follow the selected log level.

During preview rendering, the previous plot stays visible beneath a subtle
loading overlay. The indicator appears after a short delay and disappears when
the update finishes or fails, including when the preview is detached.
Changing settings during an update replaces the obsolete request with the newest
settings. Intermediate requests and late results are discarded, while the
previous plot remains visible until the replacement preview is ready.
