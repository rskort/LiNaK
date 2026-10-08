# HDF5 Data Model And Metadata Conventions

## Storage and loading policy

New files record `compression_profile` (`fast`, `balanced`, or `none`) and
`storage_precision_policy=mixed-v1` in root provenance. `mixed-v1` uses float32
for bulk trajectory coordinates, cube grids, atom-resolved matrices, profiles,
and plot arrays. Cell/origin/surface geometry, transforms, accumulated moments,
fits, and potential/work-function values remain float64.
Scientific geometry, classification, normalization, fitting, and reductions
upcast to float64 before calculation.

Numeric datasets use approximately 1 MiB chunks, with frame-major chunks
clamped to 1–256 frames. Large dtype conversions are written by chunk and final
files are atomically replaced only after HDF5 closes successfully.

Header reads expose a payload-free `dataset_inventory` for every profile. Each
entry contains dataset name, shape, dtype, and an uncompressed byte estimate.
GUI plotting uses this inventory to request only view-required datasets; public
`load_*_profiles()` functions remain eager for backward compatibility.

## Surface Schema v2

Density, position, coordination, and orientation profiles using the shared
layout model write schema v2. `surface.layout_requested`,
`surface.layout_resolved`, `surface.top`, `surface.bottom`, the automatic
selection reason, validation flags, and population asymmetry are stored in
metadata. Per-frame plane diagnostics use `surface_top_*` and
`surface_bottom_*` datasets. Partially paired symmetric metadata is rejected.

Surface-aware analysis readers require their current v2 schema, including for
single-interface cells. RDF and temperature retain their current v1 schemas.
Unsupported analysis schemas are rejected before reading numerical payloads,
with a recomputation message. Conversion caches store paired references and
layout diagnostics independently of analysis files.

Symmetric density sparse grids additionally store
`geometry_top_surface_offsets_A`, `geometry_bottom_surface_offsets_A`, and
aligned `geometry_{top,bottom}_{mass_sum_g,entity_sum}` arrays. They share the
existing geometry offsets and flat coordinate indices, so side resolution does
not duplicate sparse coordinates. These arrays are additive to schema v2.
New sparse grids also store `geometry_axis_cuts_A`,
`geometry_axis_lowers_A`, and `geometry_axis_periods_A`. One-sided layouts also
store `geometry_axis_wrap_thresholds_A`, which includes the trusted rough-plane
envelope. These values preserve the exact canonical periodic image per geometry
group, allowing filtered and tilted-prismatic distance projections to normalize
boundary-crossing slabs without treating the cell cut as a physical interface.
Legacy single-side grids and raw Cartesian projections remain readable; a
symmetric distance projection requires the complete paired set and reports a
recompute error if only part (or none) of it is present.

Coordination v2 additionally stores `axis_coordinate_A` for an unfolded
full-cell representation. Orientation v2 stores `full_bin_edges_A`, full-cell
means, density, heatmaps, valid/total counts and statistics, plus
`grid_lab_cos_polar_sum` and `grid_lab_count_polar_valid`. These datasets use a
single laboratory `+axis` polar convention. Within v2 these are additive:
plot capability metadata disables Full cell when the
required arrays are absent and reports that the analysis must be recomputed.

LiNaK stores analysis outputs in HDF5 together with analysis-specific metadata.
This page explains the general storage model shared by the analysis workflows,
including structured surface diagnostics.

## Core Design

LiNaK does not treat an HDF5 file as just a generic dump of arrays. A LiNaK
analysis file can store:

- scientific datasets
- profile metadata
- units metadata
- stable profile identifiers
- optional plot settings and plot profiles

The goal is that a LiNaK HDF5 file is self-describing enough to be inspected,
plotted, compared, and reopened without reconstructing hidden assumptions from
memory.

## Profiles

A LiNaK HDF5 file may contain one or many analysis profiles.

A profile is one logical analysis result, for example:

- one density profile for species `O`
- one RDF profile for `O-H`
- one position profile for species `Pt`
- one orientation profile for water

Each profile carries its own metadata and stable `profile_uid`.

## Single-Profile vs Multi-Profile Files

Files written by a single analysis command may contain one profile or a profile
collection.

Examples of multi-profile storage include:

- element-resolved density output
- combined HDF5 files
- files that contain several profiles of the same analysis type

LiNaK plotting and settings persistence are built around this profile model.

## Analysis Metadata

Every profile stores metadata describing how its arrays should be interpreted.
Depending on the analysis, this may include:

- species labels
- analysis axis
- frame count
- atom count
- coordinate mode
- cutoff values
- surface summary values
- provenance paths

LiNaK also stores:

- `analysis`
- `analysis_schema_version`

These identify the analysis family and its schema revision, independently of
the LiNaK package version. Readers require the current schema for that analysis
(see [Current-Schema Loading](#current-schema-loading)). Unsupported schemas or
missing required datasets/metadata are rejected with a recomputation message.
A different package version alone generates a warning, not a schema migration.

## Units Map

LiNaK stores a `units_map` inside profile metadata. This maps dataset names to
their intended units, for example:

- `Angstrom`
- `ps`
- `g/cm^3`
- `atom/nm^3`
- `dimensionless`

This makes the file self-describing enough for later plotting and export logic
to label data correctly.

## Stable Profile Identity

Each stored profile receives a stable `profile_uid`. This is important for:

- distinguishing profiles inside combined files
- persistent plot settings
- stable series identity in the GUI

The goal is that settings refer to a logical profile identity rather than only
its current position in a list.

## Combined Files

Combined LiNaK HDF5 files store multiple profiles in one physical file. They do
not flatten all underlying analysis outputs into one merged dataset.

That means:

- each original analysis result remains its own profile
- each profile keeps its own metadata
- plotting can render them together as multiple series

## Trajectory HDF5 Preprocessing Metadata

`linak apply convert` writes LiNaK trajectory HDF5 (`*.traj.h5`) as the
canonical input for repeated compute runs. These files are separate from
analysis-output HDF5 files, but they also carry metadata that compute commands
use before falling back to adjacent simulation inputs.

When available, trajectory metadata includes:

- resolved simulation input path and format
- resolved cell lengths, timestep, MD timestep, and trajectory stride
- fixed-atom indices from simulation input constraints
- whether conversion already wrote PBC-wrapped coordinates
- the PBC cell and coordinate basis used for the converted positions
- a conversion-time default surface cache for `axis=z`, `surface_mode=auto`,
  automatic surface elements, and no fixed-surface-atom inclusion

Surface-aware analyses reuse the cached conversion-time surface only when their
requested surface settings match the cache. Different settings, for example
`--axis x` or `--surface-mode rough`, deliberately recompute the surface.

## Structured Surface Diagnostics

Surface-aware analyses such as density, position, coordination, and orientation
can now persist both surface *summary metadata* and *per-frame surface
diagnostics*.

This is important because a profile may have a meaningful scalar summary
surface position while still not being trustworthy enough to use
surface-relative coordinates in the analysis itself.

### Surface Summary Metadata

Newer LiNaK files store profile-level surface summary metadata inside a nested
`surface` object in `metadata_json`.

Common nested keys can include:

- `surface.position`
- `surface.position_std`
- `surface.mode`
- `surface.side`
- `surface.selected_elements`
- `surface.candidate_indices`
- `surface.method_label`
- `surface.valid_fraction`
- `surface.median_confidence`
- `surface.composite_score`
- `surface.low_confidence_threshold`

These describe the chosen surface estimate at profile level.

### Per-Frame Surface Diagnostics

Common stored per-frame surface datasets can include:

- `surface_position_per_frame_A`
- `surface_valid_mask`
- `surface_confidence`
- `surface_provenance`
- `surface_candidate_count`
- `surface_top_layer_size`
- `surface_largest_gap_A`
- `surface_baseline_gap_A`
- `surface_reference_spread_A`
- `surface_jump_rejection_mask`
- `surface_rejection_reason`

Together, these allow a later reader to reconstruct not only the chosen surface
reference coordinate `s_t`, but also how reliable each frame was and how the
estimate was obtained.

### Effective Options

LiNaK also stores `surface.effective_options`, which captures the effective
advanced surface-estimation parameters used for that analysis.

This matters for reproducibility because the estimator behavior depends on
thresholds such as:

- gap thresholds
- top-layer size limits
- rough-reference fractions
- quantile settings
- low-confidence cutoffs
- conservative fill tolerances

## Coordinate Mode Semantics

Surface-aware analyses also store `coordinate_mode`.

- `coordinate_mode = "distance"` means the analysis actually used the stored
  frame-wise surface reference for alignment.
- `coordinate_mode = "axis"` means the analysis kept raw axis coordinates.

This distinction matters because a file may still contain rich surface metadata
and diagnostics even when the final analysis did not trust the surface estimate
enough to use distance-to-surface coordinates.

## Current-Schema Loading

Readers validate each analysis's current schema before loading its payload:
density, position, coordination, orientation, and potential tables use v2;
RDF and temperature use v1. A different version or missing required data produces an actionable
recomputation error. No analysis files are automatically migrated or rewritten.

Conversion-time surface caches are optional. Only matching current paired
`SurfaceReference` caches are reused, including references resolved to a single
interface. Missing caches and obsolete one-estimate caches are ignored, and
the analysis recomputes the reference from the trajectory. Cache decisions are
logged at DEBUG. Malformed current caches raise a clear error requesting
reconversion instead of silently guessing their geometry.

## Plot Settings

LiNaK can also store plot settings inside HDF5 files. These are separate from
the scientific datasets themselves.

In practice, one file can contain:

- the computed scientific result
- persisted plotting preferences for how to display that result

These are related but distinct layers.

## Why This Matters For Users

The HDF5 structure is part of LiNaK's reproducibility story. The output should
not just say "here are some numbers." It should also say:

- what kind of analysis these numbers represent
- what units they use
- how many frames contributed
- which species or coordinates they correspond to
- whether a surface estimate was used directly or only stored as advisory
  metadata

## Related Documentation

- [Density](density.md)
- [Position](position.md)
- [RDF](rdf.md)
- [Coordination](coordination.md)
- [Potential](potential.md)
- [Orientation](orientation.md)
- [Surface Estimation](surface-estimation.md)
