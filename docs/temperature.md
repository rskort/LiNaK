# Temperature

`linak compute temperature` writes temperature time series from CP2K temperature
tables or velocity trajectories into LiNaK HDF5.

Supported inputs:

- `.temp`: CP2K per-kind temperatures. LiNaK labels columns from sibling
  trajectory atom order when available, then from `input.inp` `&KIND` blocks,
  and finally with generic kind labels.
- `.tregion`: CP2K thermal-region temperatures. LiNaK labels columns from
  `&THERMAL_REGION/&DEFINE_REGION` blocks in `input.inp`; fixed atoms not in a
  defined region are stored as the "Unassigned" region when resolvable.
  When a sibling velocity/position XYZ supplies atom symbols, region labels are
  enriched with composition, such as `Region 2 [O215 H430]`.
- `*-vel-1.xyz`: CP2K velocity XYZ. LiNaK recomputes kinetic temperatures for
  elements and, when region metadata is available, thermal regions.

Velocity temperatures use:

```text
T = 2 K / (3 N kB)
K = 0.5 * sum_i(m_i * |v_i|^2)
```

For CP2K velocity XYZ files the default velocity unit is atomic velocity units.
This matches the CP2K `.temp` and `.tregion` values for the examined
`Au111_K6` AIMD files. `--velocity-unit angstrom/fs` is available for files
whose velocity columns are already in Angstrom/fs. `--remove-com` removes the
center-of-mass velocity per selected element or region and changes the effective
degrees of freedom from `3N` to `3N-3`.

Velocity XYZ atom labels follow the normal LiNaK XYZ rules. Labels that imply an
element, such as `Pt_top`, are resolved automatically to `Pt`, and truly custom
labels can be mapped with `--atom-alias RAW=ELEMENT`.

Metadata discovery is best effort. LiNaK first uses `--input`, then sibling
`input.inp`, then sibling velocity/position XYZ files where useful. If labels
still cannot be resolved, LiNaK warns and writes generic labels rather than
failing.

The HDF5 output stores one profile per element or region with:

- `frame_index`, `step`, `time_fs`, `time_ps`
- `temperature_K`
- optional zero-based `atom_indices`

Profile metadata records the source type, element or region identity, CP2K
`LIST` expression, atom count, target temperature, velocity unit, DOF mode,
label-resolution status, and region composition. Region profiles preserve the
raw CP2K/logical `region_name` separately from the plotted `default_label`; the
composition fields are `region_elements` and `region_formula`.

Examples:

```bash
linak compute temperature Au111_K6-1.temp
linak compute temperature Au111_K6-1.tregion --input input.inp
linak compute temperature Au111_K6-vel-1.xyz --input input.inp
linak compute temperature Au111_K6-vel-1.xyz --group-by elements
linak plot LiNaK_outputs/Au111_K6.temperature.h5
```
