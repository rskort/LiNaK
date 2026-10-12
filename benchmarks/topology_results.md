# Persisted molecular topology benchmark

Measured on the supplied `Pt110_Na4_H_cell1-pos-1.xyz` (315,033,300 bytes,
8,010 frames, 604 atoms/frame), Windows 11, Python 3.12, NumPy 2.3.2,
h5py 3.14, ASE 3.24. Runs were sequential in fresh processes without profiling
or concurrent tests. OS file caches were not flushed.

Default conversion retained surface caching, float32 positions, and fast/LZF
compression. One full cold conversion took 74.95 s and peaked at 647.66 MiB.

| Cold conversion stage | Time |
|---|---:|
| Read trajectory | 6.74 s |
| Prepare cell/selection | 2.92 s |
| Build molecular topology | 17.99 s |
| Build surface cache | 46.30 s |
| Write HDF5 | 0.88 s |

Exactly 8,010 geometric classifications ran, one per frame. All frames in this
particular trajectory share one topology state; other trajectories can retain
many states and transient changes. The complete output is 44,066,508 bytes.
Compared with a fresh HDF5 copy containing identical scientific datasets without
the topology group, the cache adds 441,435 bytes (431.1 KiB). Container allocation
order contributes to this physical-size comparison. Compared with the previous
conversion artifact, the new file is 347,821 bytes larger.

For repeated analysis, both paths read the same trajectory and compute molecular
density on Z with 0.1 Angstrom bins and surface alignment disabled. The control
uses the unchanged geometric detector exactly on every frame; it does not use
the previous sampled topology policy, which can miss transient changes.

| Three-run median | Exact live detection | Persisted topology |
|---|---:|---:|
| Read | 4.33 s | 4.30 s |
| Analysis | 24.07 s | 6.72 s |
| Read + analysis | 28.40 s | 10.99 s |
| Detector calls | 8,010 | 0 |
| Peak resident memory | 422.24 MiB | 435.45 MiB |

Analysis time fell 72.1% (3.58 times faster); total read + analysis fell 61.3%.
Median resident memory increased 13.21 MiB with the shared memberships retained.
Individual analysis timings were 23.53/24.07/24.07 s live and
6.83/6.72/6.68 s cached. Density mass/count arrays matched exactly.

Comparison with the previous conversion found all 41 pre-existing datasets have
identical shapes, dtypes, chunking, and compression. Stored positions, labels,
indices, cell, PBC, and surface selection/validity masks match exactly. Surface
reference positions differ by at most 3.82e-7 Angstrom because conversion now
prepares surface references at the stored coordinate precision. Surface layout
decisions remain unchanged.

Reproduce with:

```text
python benchmarks/topology_performance.py SOURCE ARTIFACT_DIRECTORY
```

The script checks equality of live/cached density arrays, records every run's
time, detector count, peak resident memory, and cold conversion stage times in
`results.json`, and retains ordinary stage logs. Use `--reuse-conversion` to
repeat the analysis measurements against its existing cold conversion artifact.
