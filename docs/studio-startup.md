# Opening results in Studio

`linak plot file.orientation.h5` opens the Studio window first. The same
workflow applies to density, position, RDF, coordination, potential, and
temperature, including multi-file sessions opened with `--files`.

Startup and preview updates share one compact activity indicator. Short captions
such as **Reading your results**, **Lining up the interfaces**, and **Building
your plot** follow the work being performed. Source names and file counts appear
when useful; the full source detail remains available in the tooltip and startup
Details. Elapsed time appears quietly after ten seconds. Quick preview updates
avoid flashing the indicator.

Plot controls become available after the source catalog and saved settings are
ready, and the selected plot is then rendered in the background. Only one initial
preview is calculated.

You can cancel preparation or close the window while loading. Preparation
errors appear in the window with **Retry**, **Details**, and **Close**. Closing
before the results are ready does not prompt to save plot settings. Cancelled
attempts discard their own temporary combined results and preserve input files.

Source catalogs read metadata and, where necessary, atom identities. Numerical
plot arrays are loaded for the selected view. Multi-file sessions retain their
existing combined-file workflow, performed after the window becomes visible.
Existing memory caches and limits still apply. No recomputation is needed.

Startup avoids rendering an invisible plot and keeps Matplotlib figures,
artists, and plotted data out of settings, undo snapshots, and saved profiles.
This also reduces the cost of switching views and restoring settings. Saved
manual choices and the separate settings for each representation are retained.

For diagnostics, use the existing `--log-level DEBUG` option. Debug messages
separate GUI imports, preparation, first paint, control construction, data
loading, and preview rendering. Window opening and preview completion are
different measurements; a visible window does not mean a large map has finished
loading.

Developers can measure fresh-process and warm startup, including the first
canvas draw, without saving settings or modifying a single input file:

```console
python -m benchmarks.measure_studio_startup path/to/file.orientation.h5 --repeats 3
```

This benchmark uses Qt's offscreen platform. It reports imports, first paint,
and first preview separately; desktop timing can vary with the operating
system, fonts, file size, and selected view.
