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
- what fallbacks are used when cell, timestep, or provenance data are missing
- which metadata and datasets are written to HDF5
- how shared helper logic such as water detection and surface estimation works

## Analysis Guides

- [Density](density.md)
- [MSD](msd.md)
- [Position](position.md)
- [RDF](rdf.md)
- [Coordination](coordination.md)
- [Potential](potential.md)
- [Orientation](orientation.md)

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

- `src/linak/analysis/density.py`
- `src/linak/analysis/msd.py`
- `src/linak/analysis/position.py`
- `src/linak/analysis/rdf.py`
- `src/linak/analysis/coordination.py`
- `src/linak/analysis/potential.py`
- `src/linak/analysis/orientation.py`
- `src/linak/analysis/water.py`
- `src/linak/analysis/schema.py`
