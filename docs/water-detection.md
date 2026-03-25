# Water Detection And Water Geometry

LiNaK uses shared water-specific helper logic in analyses such as density and
orientation. This page explains how water molecules are identified and how
water geometry is made consistent under periodic boundary conditions.

The current implementation lives in `src/linak/analysis/water.py`.

## Why This Exists

Water should usually not be treated as three unrelated atoms when the analysis
target is molecular rather than atomic. For example:

- water density should often be based on molecular center of mass
- water orientation requires a chemically meaningful O-H geometry

LiNaK therefore provides shared water-detection and water-geometry helpers.

## Water Identification Rule

LiNaK identifies a valid water molecule as one oxygen bonded to exactly two
hydrogen atoms within the internal O-H cutoff, with the additional requirement
that each of those hydrogens is bonded to exactly one oxygen.

This prevents overcounting ambiguous environments.

The current default cutoff is:

- `H2O_OH_CUTOFF_A = 1.25 Angstrom`

## Algorithm

For one frame, LiNaK:

1. builds an O-H neighbor list using the cutoff
2. constructs an oxygen-to-hydrogen mapping
3. constructs a hydrogen-to-oxygen mapping
4. accepts only oxygen atoms bonded to exactly two hydrogens
5. requires each accepted hydrogen to belong to exactly one oxygen

The output is a list of unique `(O, H1, H2)` triplets.

In set notation, an oxygen `O_k` is accepted only if:

- `|N_H(O_k)| = 2`

and each hydrogen `H_j` in that bonded set satisfies:

- `|N_O(H_j)| = 1`

where `N_H(O_k)` is the set of hydrogen atoms within the O-H cutoff of oxygen
`O_k`, and `N_O(H_j)` is the set of oxygens within the cutoff of hydrogen
`H_j`.

## Why The Extra Hydrogen Check Matters

The second rule, that each hydrogen must belong to exactly one oxygen, is there
to avoid classifying ambiguous proton-sharing or unusual close-contact
geometries as ordinary water molecules.

The method is therefore deliberately conservative rather than permissive.

## Periodic Geometry Handling

Once a water triplet is known, LiNaK computes PBC-aware water geometry:

- oxygen position
- hydrogen 1 position
- hydrogen 2 position
- molecular center of mass

The hydrogens are first reconstructed relative to the oxygen using
minimum-image vectors. This avoids the common problem where one hydrogen has
wrapped across the periodic boundary and appears spuriously far away in raw
Cartesian coordinates.

If `r_O`, `r_H1`, and `r_H2` are the raw coordinates, LiNaK first computes
minimum-image bond vectors:

- `v_1 = MIC(r_H1 - r_O)`
- `v_2 = MIC(r_H2 - r_O)`

and then reconstructs PBC-consistent hydrogen positions:

- `r'_H1 = r_O + v_1`
- `r'_H2 = r_O + v_2`

## Center Of Mass

LiNaK computes the water center of mass from the PBC-corrected O/H positions and
the atomic masses returned by ASE.

The center of mass is:

`r_COM = (m_O r_O + m_H r'_H1 + m_H r'_H2) / (m_O + 2 m_H)`

This COM is then used in molecular analyses such as water density and
water-orientation distance binning.

## Topology Caching Across Frames

For multi-frame workflows, LiNaK does not necessarily recompute the entire water
topology from scratch in every frame.

Instead, it caches the detected triplets and periodically revalidates them. The
current validation stride is:

- `H2O_VALIDATION_STRIDE = 100 frames`

If revalidation shows that the water topology changed, LiNaK refreshes the
cached triplets and logs a warning.

This is a performance optimization with a correctness safeguard.

## What This Means For Users

The shared water logic assumes that the system behaves like ordinary molecular
water for the purposes of selection and geometry. It is a good fit for:

- liquid water
- interfacial water
- solvated systems with conventional O-H connectivity

It may be a poor fit for:

- strong proton transfer chemistry
- reactive trajectories with changing water identities
- unusual hydrogen-bonding environments that violate the simple connectivity
  model

## Which Analyses Use This

The shared water logic is currently used by at least:

- [Density](density.md), for `H2O` density
- [Orientation](orientation.md), for water orientation

## Related Documentation

- [Density](density.md)
- [Orientation](orientation.md)
