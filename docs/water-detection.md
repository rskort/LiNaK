# Water Detection And Water Geometry

LiNaK uses shared water-specific helper logic in analyses such as density and
orientation. This page explains how water molecules are identified and how
water geometry is made consistent under periodic boundary conditions.

The shared geometry and classifier live in `src/linak/analysis/water.py`;
`src/linak/analysis/topology.py` manages exact frame assignments and persistence.

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

- `H2O_OH_CUTOFF_A = 1.27 Angstrom`

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

LiNaK now resolves O/H membership exactly for every frame. Assignments include
free H and O, OH, H2O, H3O, and ambiguous/overcoordinated atom diagnostics.
Short-lived proton transfers are therefore captured even between the former
100-frame validation intervals. The existing Python stride argument remains
accepted for compatibility but no longer samples molecular assignments.

`linak apply convert` builds this topology after cell/selection preparation and
stores it in `.traj.h5`. Assignments use the coordinates at the file's actual
storage precision. Surface populations reuse those assignments. Stable states
are deduplicated, so an unchanged molecule layout need not be stored 8,000 times.

Subsequent molecular analyses read memberships without neighbor searches. Each
frame's geometry, element/order, cell, and PBC are fingerprinted; changing them
invalidates the assignment. Molecular centers, angles, and distances are always
computed from the current geometry. Different cutoffs or requested detection
backends use separate profiles.

Older `.traj.h5` files remain usable. Their first molecular analysis builds a
full-source profile and saves it atomically after read handles close. A read-only
file or failed save produces a warning and continues with the same topology in
memory. Existing scientific datasets and the filename are preserved.

Computing directly from `.xyz` also works: topology is collected during that
run and shared across species and analysis passes. Conversion is not required;
the XYZ is unchanged, and no permanent sidecar is created. Streaming uses the
existing temporary trajectory storage and removes it when the command finishes.

## What This Means For Users

The shared water logic assumes that the system behaves like ordinary molecular
water for the purposes of selection and geometry. It is a good fit for:

- liquid water
- interfacial water
- solvated systems with conventional O-H connectivity

Changing molecular identities are supported through exact per-frame assignments.
Environments that violate the simple O/H connectivity model are excluded through
the ambiguity and overcoordination rules; caching does not change those rules.

## Which Analyses Use This

The shared water logic is currently used by at least:

- [Density](density.md), for molecular density
- [Position](position.md), for molecular positions
- [Coordination](coordination.md), for molecular coordination
- [Orientation](orientation.md), for water orientation
- [Surface Estimation](surface-estimation.md), for water populations

## Related Documentation

- [Density](density.md)
- [Orientation](orientation.md)
