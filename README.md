# LiNaK

LiNaK is a lightweight Python toolkit for molecular dynamics trajectory analysis. The package is designed for electrochemical systems (e.g. a Pt(111)-surface with an electrolyte of water and cations), but many features are general-purpose and applicable to other MD contexts as well. It combines trajectory analysis, HDF5-based data storage, plotting, and a few practical CP2K/LAMMPS utilities behind one CLI.

LiNaK provides four top-level commands:
- `linak compute`: generate LiNaK HDF5 analysis files
- `linak plot`: plot LiNaK density, MSD, and RDF HDF5 files by auto-detecting the analysis from the HDF5 metadata
- `linak apply`: apply PBC or compress CP2K output files
- `linak hdf5` (`linak hd`, `linak h5`): inspect, combine, transform, and plot generic tabular HDF5 data

Supported inputs include:
- ASE-supported trajectory files (e.g. CP2K's `.xyz`, LAMMPS's `.dump`, etc.)
- LAMMPS and CP2K input scripts (`.lmp`, and `.inp`, respectively)
- CP2K Hartree cube files for potential analysis

LiNaK supports Python `>=3.9`.

## Installation

Clone the repository:

```bash
git clone https://github.com/rskort/LiNaK.git
cd LiNaK
```

From the project root:

```bash
pip install .
```

Install the optional GUI dependency for the interactive Plot Studio:

```bash
pip install -e .[gui]
```

Install development tools:

```bash
pip install -e .[dev]
```

## Quick Start

Trajectory to HDF5 to plot:

```bash
linak compute density traj.xyz --species H2O
linak plot traj_density_h2o_z.h5

linak compute msd traj.xyz --species O
linak plot traj_msd_o.h5
linak compute rdf traj.xyz --species-a O --species-b H
linak plot traj_rdf_o_h.h5
```

Run `linak` for information or `linak --help` for the full CLI overview.


Python API example:

```python
from linak.trajectory.io import read_trajectory
from linak.analysis.density import compute_density_profile
from linak.analysis.msd import compute_msd

frames = read_trajectory("traj.xyz")
density = compute_density_profile(frames, species="O", axis="z", bin_width=0.1)
msd = compute_msd(frames, species="O", timestep_fs=0.5)

print(density.species, density.units, density.n_frames)
print(msd.species, msd.msd[-1], "A^2")
```

## CLI Overview

### `linak compute`

`compute` commands read trajectories or CP2K Hartree cube files and write
LiNaK HDF5 outputs.

Available analyses:
- `density`: 1D density profiles
- `msd`: mean-squared displacement
- `rdf`: radial distribution function
- `potential`: CP2K cSHE-related quantities from Hartree cube files

Examples:

```bash
linak compute density traj.xyz --species H2O --axis z --bin-width 0.1
linak compute msd traj.xyz --species O --timestep-fs 0.5
linak compute rdf traj.xyz --species-a O --species-b H --bin-width 0.05
linak compute potential -f run1/*-v_hartree-1_0.cube run2/*-v_hartree-1_0.cube --output potentials.h5
```

### `linak plot`

The `plot` command reads LiNaK analysis HDF5 files and auto-detects whether the
file contains density, MSD, or RDF data.

Examples:

```bash
linak plot traj_density_o_z.h5
linak plot -f traj_density_h2o_z.h5 traj_density_li_z.h5
linak plot traj_msd_o.h5 
linak plot traj_rdf_o_h.h5 
```

When `linak plot` cannot detect a supported LiNaK analysis in the HDF5 file, it
falls back to generic HDF5 plotting via `linak hdf5 plot`.

### `linak apply`

Available apply commands:
- `pbc`: wrap atom positions into an orthorhombic periodic cell
- `compress`: extract structured files from one CP2K `.out` file

Examples:

```bash
linak apply pbc pos.xyz --cell 17.887 15.491 59.671
linak apply compress /path/to/output.out
```

### `linak hdf5`

The `hdf5` command group works with generic tabular HDF5 data. It supports:
- `info`, `preview`, `get`
- `sort`, `filter`, `dedupe`
- `combine` for LiNaK density/MSD/RDF HDF5 files
- `plot` for generic column-based plotting
- `plot-settings` for persisted plot profiles stored inside HDF5 files

Examples:

```bash
linak hdf5 info density.h5
linak hdf5 combine -f traj_density_h2o_z.h5 traj_density_li_z.h5
linak hdf5 plot table.h5 --kind line --x time_ps --y msd_A2
```

Many `linak hdf5` commands are semi-interactive: if required columns or options
are omitted, LiNaK can prompt with available choices.

## Analysis Notes

### Density

`linak compute density` supports:
- `--species O`: one element
- `--species H2O`: molecular water density
- `--species all`: one output profile per element

Cell handling for density:
- first use `--cell A B C` if provided
- then use `--input` / `--cp2k-input` / `--lammps-input` if provided
- otherwise auto-detect a single `.inp` or `.lmp` next to the trajectory
- if no periodic cell can be resolved, LiNaK falls back to a linear density profile

Density units depend on whether a usable periodic cell is available:
- mass density: `g/Angstrom` (linear) or `g/cm^3` (volumetric)
- number density: `atom/Angstrom` (linear) or `atom/nm^3` (volumetric)

Surface-aware density profiles support `--surface-mode {auto,layered,rough}`:
- `auto` (recommended default): LiNaK inspects the structure and chooses a suitable surface reference automatically.
- `layered` (recommended for flat crystalline slabs, especially metal surfaces): LiNaK identifies the top surface layer and uses its mean height as the reference plane.
- `rough` (recommended for non-flat, reconstructed, or disordered surfaces): LiNaK tracks low-mobility surface atoms frame by frame and uses their mean position as a rough surface reference.

In all surface-aware modes, the atom-surface distance used for the density profile is computed frame by frame: LiNaK finds the surface position in one frame, shifts the atoms in that frame relative to it, and then repeats this for the next frame. The single stored `surface_position` in the HDF5 output is only a summary over those frame-wise references, not the one value used for all frames.

Use `--surface-elements` when LiNaK should only use specific atoms for surface detection, for example the metal atoms of a slab and not mobile water or ions near the interface. Generally, the automatic detection should do this correctly, but `--surface-elements` can be used to override that if needed.

### MSD

For MSD, LiNaK resolves the frame timestep in this order:
1. `--timestep-fs` if provided
2. trajectory metadata when available
3. simulation input from `--input` / `--cp2k-input` / `--lammps-input` if provided
4. auto-detected `.inp` or `.lmp` next to the trajectory
5. fallback to `0.5 fs`

When a usable periodic cell is available, MSD uses periodic minimum-image
accumulation. Otherwise it falls back to direct displacement.

### RDF

`linak compute rdf` supports `--species-a`, `--species-b`, `--r-max`,
`--bin-width`, and `--threads`. RDF requires a usable cell volume, taken from
trajectory metadata when present or resolved from simulation input.

### Potential

`linak compute potential` is CP2K-focused. For each Hartree cube file, LiNaK:
- searches the same directory for a suitable CP2K output file (`output.out` preferred)
- parses the Fermi level from the output file
- computes a water-bulk potential from O/H z-bounds read from the cube header
- reports `electrode_cshe_ev = water_bulk_potential_ev - efermi_ev - 0.81` by default


Output behaviour for `compute potential`:
- appends to an existing compatible HDF5 file by default
- skips sources already present in that HDF5 file
- falls back to a safe new filename if the existing schema is incompatible
- writes rows incrementally so partial progress is retained if a later source fails

Use `--strict` when a failed or incomplete source should cause a non-zero exit code.

## Plotting and Plot Studio

For `linak plot`, the interactive Plot Studio is the default
when interactive plotting is enabled and the optional GUI dependency is
installed. Use:
- `--no-gui` for direct Matplotlib rendering
- `--no-show` for batch-style runs
- `--backend` to request a specific interactive Matplotlib backend (default is `QtAgg`, but it falls back to the best available backend if that one is not installed)

The easiest, and recommended method to change plot styles is to use the interactive controls in Plot Studio. For more advanced users, or when using `--no-gui`, LiNaK supports a wide range of CLI options to customize plot styles and settings.

Shared style controls include:
- `--figsize WIDTH HEIGHT`
- `--dpi`
- `--font-family`
- `--title-font-size`, `--label-font-size`, `--tick-font-size`
- `--line-width`, `--line-color`, `--line-colors`
- `--grid`, `--ticks`, `--markers`
- axis limits, scales, ticks, legend controls, and custom labels

Plot settings can be persisted inside HDF5 files and reused across sessions.
Use `linak hdf5 plot-settings` to inspect, edit, copy, import, or export those
saved plot profiles.

## Apply Commands

### `linak apply pbc`

PBC wrapping resolves the cell in this order:
1. `--cell A B C`
2. `--input` / `--cp2k-input` / `--lammps-input`
3. auto-detect a single `.inp` or `.lmp` in the output directory

If the trajectory file itself already contains a usable periodic cell, LiNaK uses that
without requiring an external input file.

Examples:

```bash
linak apply pbc pos.xyz
linak apply pbc pos.xyz --input input.inp --output wrapped.xyz
linak apply pbc pos.xyz --overwrite
```

### `linak apply compress`

`linak apply compress` extracts structured analysis files from one CP2K output
and then moves the raw `.out` file into a backup directory.

By default it creates:
- an output directory named after the input stem (auto-suffixed if needed)
- `README.txt`, `manifest.json`, `summary.txt`, `backup_info.txt`, CSV extracts, and text summaries
- a hidden backup directory `.linak_backups/` next to the input
- a sidecar `<backup-file>.meta.json` describing backup linkage

Useful options:
- `--backup-dir /path/to/backups`
- `--drop <section> [<section> ...]`

## Shared CLI Features

All executable subcommands support:
- `--dry-run` / `-n`
- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--log-file <path>`

Additional shared behavior:
- compute commands write HDF5 next to the input source by default
- use `-o/--output` (alias `--save-data` on compute commands) to override output paths
- use `-f/--files` when passing multiple input files; it also works for a single file
- LiNaK records resolved cell and timestep provenance in HDF5 metadata where applicable
- when available metadata sources disagree, LiNaK logs a warning instead of silently picking one without notice

## Development

```bash
ruff check .
ruff format --check .
mypy src tests
pytest -q
```

## Contributing
Contributions are always welcome! Please open an issue or submit a pull request on GitHub.

When contributing code, please follow the existing style and conventions as much as possible. The project uses `ruff` for linting and formatting, so running `ruff check .` and `ruff format --check .` before submitting can help ensure consistency.

You can also request additions or changes by sending me an email at <r.s.kort@lic.leidenuniv.nl>.