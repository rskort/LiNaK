# HDF5 Data Model And Metadata Conventions

LiNaK stores analysis outputs in HDF5 together with analysis-specific metadata.
This page explains the general storage model shared by the analysis workflows.

## Core Design

LiNaK does not treat an HDF5 file as just a generic dump of arrays. It stores:

- analysis datasets
- profile metadata
- units metadata
- stable profile identifiers
- optional plot settings and plot profiles

The goal is that a LiNaK HDF5 file is self-describing enough to be:

- inspected later
- plotted later
- combined with other LiNaK outputs
- interpreted correctly without reconstructing hidden assumptions from memory

## Profiles

A LiNaK HDF5 file may contain one or many analysis profiles.

A profile is one logical analysis result, for example:

- one density profile for species `O`
- one RDF profile for `O-H`
- one position profile for species `Pt`

Each profile carries its own metadata and stable `profile_uid`.

## Single-Profile vs Multi-Profile Files

Some workflows naturally produce one profile per file. Others can produce
multiple profiles in one file, for example:

- multi-species density output
- combined HDF5 files
- files containing multiple stored profiles of the same analysis type

LiNaK plotting code is built around this profile model.

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

These fields are used to identify the analysis type and the intended schema.

## Units Map

LiNaK stores a `units_map` inside profile metadata. This maps dataset names to
units such as:

- `Angstrom`
- `ps`
- `g/cm^3`
- `atom/nm^3`
- `dimensionless`

This makes the file more self-describing and helps later plotting and export
logic label the data correctly.

## Stable Profile Identity

Each stored profile receives a stable `profile_uid`. This is important for:

- distinguishing profiles inside combined files
- persistent plot settings
- stable series identity in the GUI

The goal is that settings can refer to a profile by identity rather than by only
its current position in a list.

## Combined Files

Combined LiNaK HDF5 files store multiple profiles in one physical file. They do
not flatten all underlying analysis outputs into one single dataset.

That means:

- each original analysis result remains its own profile
- each profile keeps its own metadata
- plotting can render them together as multiple series

This is the intended storage model for multi-source comparison.

## Plot Settings

LiNaK can also store plot settings inside HDF5 files. These are separate from
the scientific datasets themselves.

In practice this means one file can contain:

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

## Related Documentation

- [Density](density.md)
- [MSD](msd.md)
- [Position](position.md)
- [RDF](rdf.md)
- [Coordination](coordination.md)
- [Potential](potential.md)
- [Orientation](orientation.md)
