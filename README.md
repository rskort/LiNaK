# LiNaK

LiNaK is a lightweight and modular Python package for molecular dynamics trajectory analysis.

LiNaK provides a modular CLI:
- `linak plot ...` for plotting from existing LiNaK HDF5 files
- `linak compute ...` for generating HDF5 analysis outputs
- `linak apply ...` for trajectory transformations and output post-processing

## Installation

From the project root:

```bash
pip install .
```

For development (includes lint, type-check, and test tools):

```bash
pip install -e .[dev]
```

For GUI plot controls (`--gui`, PySide6/Qt):

```bash
pip install -e .[gui]
```

## Quick Start

CLI quick example:

```bash
linak compute density traj.xyz --species O --axis z --bin-width 0.1 --output density_o.h5
linak plot density density_o.h5 --no-show --output density_o.png
```

Python API quick example:

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

## Developer Commands

```bash
ruff check .
ruff format --check .
mypy src tests
pytest -q
```

## CLI Overview

Show top-level help:

```bash
linak --help
```

Explore plotting commands:

```bash
linak plot --help
linak plot density --help
linak plot msd --help
linak plot rdf --help
```

Explore compute commands:

```bash
linak compute --help
linak compute density --help
linak compute msd --help
linak compute rdf --help
linak compute potential --help
```

Explore apply commands:

```bash
linak apply --help
linak apply pbc --help
linak apply compress --help
```

## Compute Commands (HDF5 Generation)

`compute` commands read trajectories and write HDF5 outputs.
Trajectory sources can be:
- regular ASE-supported trajectories (for example `.xyz`)
- LAMMPS text dump trajectories (`.dump`)
- LAMMPS input scripts (`.lmp`) that reference a dump file

Density (mass density):

```bash
linak compute density traj.xyz --species O --axis z --bin-width 0.1
```

MSD:

```bash
linak compute msd traj.xyz --species O
# or explicitly:
linak compute msd traj.xyz --species O --timestep-fs 0.5
```

RDF:

```bash
linak compute rdf traj.xyz --species-a O --species-b H --bin-width 0.05
```

Potential (CP2K Hartree cube workflow):

```bash
# one Hartree cube file
linak compute potential /path/to/Au111_H2O-v_hartree-1_0.cube

# many Hartree cube files in one HDF5 file
linak compute potential -f run1/*-v_hartree-1_0.cube run2/*-v_hartree-1_0.cube --output potentials.h5
```

`compute potential` is currently CP2K-focused and expects Hartree cube file paths as input.
For each cube file, LiNaK auto-discovers the CP2K output in the same directory:
- prefers `output.out`
- otherwise checks other `*.out` files and picks one with a parseable Fermi level when possible

By default LiNaK computes:
- `efermi_ev`
- `water_bulk_potential_ev`
- `electrode_cshe_ev = water_bulk_potential_ev - efermi_ev - 0.81`

Water-bulk averaging is built from O/H atom z-positions read from the cube header.
The vacuum potential is not part of this command.

HDF5 behavior for `compute potential`:
- default mode is append when the existing HDF5 schema is compatible
- when appending, LiNaK pre-checks existing `source` entries and skips already computed files
- if an existing file is incompatible, LiNaK writes to a safe fallback filename automatically
- rows are persisted incrementally during compute (with flush/sync), so partial progress is retained if a later step fails
- run-level averages with standard deviations are logged for Fermi, water-bulk, and cSHE values

Dry-run notes:
- all commands accept `--dry-run` and `-n`
- for `compute potential`, dry-run validates that each Hartree cube path exists and is readable
- dry-run inspects HDF5 append/fallback behavior and existing-row skips, but does not compute or write

For trajectory-based density/MSD/RDF analyses, LiNaK resolves PBC cell dimensions in this order:
1. `--cell A B C`
2. `--input /path/to/input.inp` or `--input /path/to/input.lmp`
   (aliases: `--cp2k-input`, `--lammps-input`)
3. auto-detected single simulation input (`.inp` or `.lmp`) in the trajectory directory
If multiple available sources disagree (explicit vs input metadata), LiNaK logs a warning.

By default, `compute` writes HDF5 output next to the input trajectory file.
Use `-o/--output` (alias: `--save-data`) on each compute subcommand to override the destination.

Density species policy:
- `--species all`: one separate density dataset per element (for example `H`, `O`, `Au`)
- `--species O`: only oxygen
- `--species H2O`: only molecular water density (single combined dataset)

Density outputs are mass-weighted using atomic masses (or H2O molecular mass for `--species H2O`).
HDF5 stores mass density in `g/Angstrom` or `g/Angstrom^3` plus run metadata.

For MSD, LiNaK resolves timestep in this order:
1. `--timestep-fs`
2. trajectory metadata (when available, for example `time_fs` / `time_ps`)
3. simulation input from `--input` / `--cp2k-input` / `--lammps-input`:
   - CP2K `.inp`: `TIMESTEP [fs] * &TRAJECTORY / &EACH / MD`
   - LAMMPS `.lmp`: `timestep * dump every` (converted to fs via `units`)
4. auto-detected `.inp`/`.lmp` in the trajectory directory (same formulas)
5. fallback to `1.0 fs` if no source is available

LiNaK logs warnings when available timestep sources disagree.
Resolved cell/timestep provenance is written to output HDF5 metadata.

## Plot Commands (HDF5 Input Only)

`plot` commands read HDF5 inputs only.  
Use `compute` commands first when starting from trajectories.

```bash
# 1) compute from trajectory to HDF5
linak compute density traj.xyz --species O --axis z
linak compute msd traj.xyz --species O
linak compute rdf traj.xyz --species-a O --species-b H

# 2) plot from HDF5
linak plot density traj_density_o_z.h5 --no-show --output density.png
linak plot msd traj_msd_o.h5 --no-show --output msd.png
linak plot rdf traj_rdf_o_h.h5 --no-show --output rdf.png

# overlay multiple HDF5 files (use -f for multiple files)
linak plot density -f run1_density_o_z.h5 run2_density_o_z.h5
```

`linak plot /path/to/file.h5` is shorthand for `linak hdf5 plot /path/to/file.h5`.

LAMMPS examples:

```bash
# direct dump trajectory + LAMMPS input metadata
linak compute msd lammps.dump --species O --input input.lmp

# or use .lmp directly as source (LiNaK resolves and reads the referenced dump)
linak compute msd input.lmp --species O
```

## Plot Style Controls

Plot commands support shared style options:
- `--figsize WIDTH HEIGHT`
- `--dpi`
- `--font-family`
- `--title-font-size`
- `--label-font-size`
- `--tick-font-size`
- `--line-width`
- `--line-color`
- `--grid` / `--no-grid`

Example:

```bash
linak plot density traj_density_o_z.h5 \
  --species O \
  --figsize 8 4 \
  --title-font-size 16 \
  --line-color "#003049"
```

## Apply Commands

Apply periodic boundary conditions to a trajectory:

```bash
linak apply pbc in.xyz --cell 17.887 15.491 59.671
```

For CP2K and LAMMPS workflows, `linak apply pbc` resolves cell dimensions in this order:
1. `--cell A B C`
2. `--input /path/to/input.inp` or `--input /path/to/input.lmp`
   (aliases: `--cp2k-input`, `--lammps-input`)
3. auto-detect a single simulation input (`.inp` or `.lmp`) in the output directory

Example with CP2K auto-detection:

```bash
# LiNaK writes pos_pbc.xyz by default and searches for one .inp/.lmp in that output directory
linak apply pbc pos.xyz

# explicit output path
linak apply pbc pos.xyz --output ./run/wrapped.xyz

# overwrite original file in place
linak apply pbc pos.xyz --overwrite
```

If auto-detection fails (no `.inp`/`.lmp`, multiple candidates, or invalid input metadata), LiNaK returns a clear error telling you to provide `--input` or `--cell`.

Compress one CP2K output into structured, smaller analysis files:

```bash
linak apply compress /path/to/output.out
```

By default, `linak apply compress`:
- writes extracted outputs to `/path/to/output/` (auto-suffixed if that directory already exists)
- moves the original raw output to `/path/to/.linak_backups/` using a unique filename
- writes backup linkage metadata as `<backup-file>.meta.json`

Main generated files in the output directory:
- `README.txt`: generated/skipped file report
- `manifest.json`: machine-readable metadata and row counts
- `summary.txt`: compact CP2K run summary
- `*.csv`: extracted tables (for example SCF iterations, charges, forces, MD steps)
- `*.txt`: setup, warnings, timing, performance, and run snippets

Useful options:
- `--backup-dir /path/to/private_backups`: override the backup location
- `--drop <section> [<section> ...]`: skip optional outputs
  (sections: `coordinates`, `mulliken`, `hirshfeld`, `forces`, `scf-iterations`, `md-steps`, `thermostat`, `timing`, `performance`, `grid`)

## Logging

All commands support:
- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--log-file <path>`
- `--dry-run` (available on each executable subcommand, e.g. `linak compute density ... --dry-run`)

At `INFO` level, LiNaK now prints a cleaner run banner with a LiNaK watermark and structured dry-run plans for easier terminal scanning.

Long-running stages (trajectory read/write and frame-based analyses) show a built-in LiNaK progress bar in interactive terminals.

Example:

```bash
linak --log-level DEBUG --log-file linak.log compute density traj.xyz
```

## Python Version Notes

LiNaK supports Python `>=3.9`.

Interactive plotting defaults to Qt (`QtAgg`) and GUI plot controls use PySide6.
Backend availability remains environment-dependent on HPC systems.

## Package Layout

```text
src/linak/
  __init__.py
  resolution.py
  cli.py
  progress.py
  pbc.py
  utils.py
  analysis/
    __init__.py
    density.py
    msd.py
    rdf.py
    potential.py
  trajectory/
    __init__.py
    io.py
    lammps.py
  plot/
    __init__.py
    plotting.py
    plot_settings.py
    plot_gui.py
  storage/
    __init__.py
    hdf5_utils.py
    hdf5_table.py
    csv_tools.py
    compress.py
tests/
  test_cli.py
  test_density.py
  test_resolution.py
  test_io.py
  test_msd.py
  test_rdf.py
  test_pbc.py
```

