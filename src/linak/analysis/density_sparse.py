"""Bounded, balanced numeric reductions for sparse density accumulators."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..runtime_resources import MIB, resolve_resource_policy


def row_dtype(channels: int) -> np.dtype:
    return np.dtype([("group", "<u8"), ("bin", "<u8"), ("values", "<f8", (channels,))])


def compact_rows(rows: np.ndarray) -> np.ndarray:
    """Sort and reduce a bounded batch, preserving integer keys exactly."""
    if not rows.size:
        return rows.copy()
    order = np.lexsort((rows["bin"], rows["group"]))
    sorted_rows = rows[order]
    starts = np.r_[
        0,
        1
        + np.flatnonzero(
            (sorted_rows["group"][1:] != sorted_rows["group"][:-1])
            | (sorted_rows["bin"][1:] != sorted_rows["bin"][:-1])
        ),
    ]
    result = sorted_rows[starts].copy()
    result["values"] = np.add.reduceat(sorted_rows["values"], starts, axis=0)
    return result


class SparseWorkspace:
    """One shared budget and temporary-resource owner for a density computation."""

    def __init__(self, budget_bytes: int | None = None):
        if budget_bytes is None:
            policy = resolve_resource_policy()
            self.budget = min(256 * MIB, policy.memory_budget_bytes // 8)
        else:
            self.budget = int(budget_bytes)
        if self.budget < 1024:
            raise ValueError(
                "Density accumulation requires at least 1 KiB of workspace; free memory "
                "or increase LINAK_MEMORY_BUDGET_MIB."
            )
        self.buffer_limit = min(8 * MIB, self.budget // 4)
        self.block_bytes = max(64, self.budget // 64)
        self.pending_bytes = 0
        self.resident_bytes = 0
        self.runs: list[_Run] = []
        self.reducers: list[SparseReducer] = []
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._serial = 0
        self.closed = False
        self.metrics: dict[str, int] = {
            "budget_bytes": self.budget,
            "peak_pending_bytes": 0,
            "peak_resident_bytes": 0,
            "peak_spill_bytes": 0,
            "spill_bytes_written": 0,
            "merged_rows": 0,
            "merges": 0,
        }
        self._spill_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def path(self) -> Path:
        try:
            if self._temporary is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="linak-density-")
            self._serial += 1
            return Path(self._temporary.name) / f"run_{self._serial}.bin"
        except OSError as exc:
            raise OSError(
                "Density temporary storage is unavailable; free space in the system temporary directory."
            ) from exc

    def reserve_pending(self, size: int) -> None:
        while self.pending_bytes + size > self.buffer_limit:
            largest = max(self.reducers, key=lambda reducer: reducer.pending_bytes)
            largest.flush()

    def register(self, run: _Run) -> None:
        self.runs.append(run)
        if run.data is not None:
            self.resident_bytes += run.data.nbytes
            while self.resident_bytes > self.budget // 2:
                next(candidate for candidate in self.runs if candidate.data is not None).spill()
        self.metrics["peak_resident_bytes"] = max(
            self.metrics["peak_resident_bytes"], self.resident_bytes
        )

    def spill_written(self, size: int) -> None:
        self._spill_bytes += size
        self.metrics["spill_bytes_written"] += size
        self.metrics["peak_spill_bytes"] = max(self.metrics["peak_spill_bytes"], self._spill_bytes)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for reducer in self.reducers:
            reducer.pending.clear()
            reducer.levels.clear()
        for run in self.runs:
            run.data = None
        self.runs.clear()
        self.pending_bytes = self.resident_bytes = 0
        if self._temporary is not None:
            self._temporary.cleanup()


class _Run:
    def __init__(
        self,
        workspace: SparseWorkspace,
        data: np.ndarray | None,
        dtype: np.dtype,
        count: int,
        path: Path | None = None,
    ):
        self.workspace, self.data, self.dtype = workspace, data, dtype
        self.count, self.path = count, path
        workspace.register(self)

    def spill(self) -> None:
        assert self.data is not None
        path = self.workspace.path()
        try:
            with path.open("wb") as handle:
                self.data.tofile(handle)
        except OSError as exc:
            raise OSError(
                "Density temporary write failed; check free disk space and temporary-directory access."
            ) from exc
        self.workspace.spill_written(self.data.nbytes)
        self.workspace.resident_bytes -= self.data.nbytes
        self.path, self.data = path, None

    def release(self) -> None:
        if self not in self.workspace.runs:
            return
        self.workspace.runs.remove(self)
        if self.data is not None:
            self.workspace.resident_bytes -= self.data.nbytes
            self.data = None
        if self.path is not None:
            self.workspace._spill_bytes -= self.count * self.dtype.itemsize
            self.path.unlink()
            self.path = None

    def materialize(self) -> np.ndarray:
        if self.data is not None:
            return self.data
        assert self.path is not None
        with self.path.open("rb") as handle:
            result = np.fromfile(handle, dtype=self.dtype, count=self.count)
        if result.size != self.count:
            raise OSError("Density temporary run is truncated.")
        return result

    def unpack(self, value_dtype: Any, group_count: int | None = None):
        """Fill returned profile arrays without materializing a second full run."""
        bins = np.empty(self.count, dtype=np.uint64)
        channels = self.dtype["values"].shape[0]
        values = [np.empty(self.count, dtype=value_dtype) for _ in range(channels)]
        offsets = (
            np.full(group_count + 1, self.count, dtype=np.int64)
            if group_count is not None
            else None
        )
        offset = 0
        previous_group = None
        with ExitStack() as stack:
            reader = _Reader(self, stack, max(1, self.workspace.block_bytes // self.dtype.itemsize))
            block = reader.read()
            while block.size:
                end = offset + block.size
                bins[offset:end] = block["bin"]
                for channel, destination in enumerate(values):
                    destination[offset:end] = block["values"][:, channel]
                if offsets is not None:
                    starts = np.r_[0, 1 + np.flatnonzero(block["group"][1:] != block["group"][:-1])]
                    if previous_group == int(block["group"][0]):
                        starts = starts[1:]
                    offsets[block["group"][starts].astype(np.int64)] = offset + starts
                    previous_group = int(block["group"][-1])
                offset = end
                block = reader.read()
        if offsets is not None:
            offsets = np.minimum.accumulate(offsets[::-1])[::-1]
        return bins, values, offsets


class _Reader:
    def __init__(self, run: _Run, stack: ExitStack, block_rows: int):
        self.run, self.block_rows, self.offset = run, block_rows, 0
        self.handle: Any = (
            stack.enter_context(run.path.open("rb")) if run.path is not None else None
        )

    def read(self) -> np.ndarray:
        count = min(self.block_rows, self.run.count - self.offset)
        if self.handle is None:
            assert self.run.data is not None
            result = self.run.data[self.offset : self.offset + count]
        else:
            result = np.fromfile(self.handle, dtype=self.run.dtype, count=count)
            if result.size != count:
                raise OSError("Density temporary run is truncated.")
        self.offset += count
        return result


class SparseReducer:
    """Balanced sorted runs; no repeated merging of a growing accumulated prefix."""

    def __init__(
        self,
        workspace: SparseWorkspace,
        channels: int,
        max_bins: int | None = None,
        species: str = "",
    ):
        self.workspace, self.dtype = workspace, row_dtype(channels)
        self.max_bins, self.species = max_bins, species
        self.pending: list[np.ndarray] = []
        self.pending_bytes = 0
        self.levels: dict[int, _Run] = {}
        workspace.reducers.append(self)

    def _check_limit(self, count: int) -> None:
        if self.max_bins is not None and count > self.max_bins:
            raise ValueError(
                "Density sparse grid exceeded the configured nonzero-bin limit: "
                f"species='{self.species}', nonzero bins={count:,}, limit={self.max_bins:,}. "
                "Increase --grid-bin-width or request fewer species."
            )

    def add(self, rows: np.ndarray) -> None:
        block_rows = max(1, self.workspace.block_bytes // self.dtype.itemsize)
        for start in range(0, rows.size, block_rows):
            size = min(block_rows, rows.size - start) * self.dtype.itemsize
            self.workspace.reserve_pending(size)
            self.pending.append(rows[start : start + block_rows].copy())
            self.pending_bytes += size
            self.workspace.pending_bytes += size
            self.workspace.metrics["peak_pending_bytes"] = max(
                self.workspace.metrics["peak_pending_bytes"], self.workspace.pending_bytes
            )

    def flush(self) -> None:
        pending, self.pending = self.pending, []
        block_rows = max(1, self.workspace.block_bytes // self.dtype.itemsize)
        batch: list[np.ndarray] = []
        size = 0

        def consume() -> None:
            nonlocal batch, size
            rows = np.concatenate(batch)
            batch, size = [], 0
            self.pending_bytes -= rows.nbytes
            self.workspace.pending_bytes -= rows.nbytes
            data = compact_rows(rows)
            self._check_limit(data.size)
            run = _Run(self.workspace, data, self.dtype, data.size)
            level = 0
            while level in self.levels:
                run = self._merge(self.levels.pop(level), run)
                level += 1
            self.levels[level] = run

        for rows in pending:
            for start in range(0, rows.size, block_rows):
                part = rows[start : start + block_rows]
                if size + part.size > block_rows:
                    consume()
                batch.append(part)
                size += part.size
        if batch:
            consume()

    def _merge(self, left: _Run, right: _Run) -> _Run:
        workspace = self.workspace
        workspace.metrics["merged_rows"] += left.count + right.count
        workspace.metrics["merges"] += 1
        capacity = left.count + right.count
        # Reserve half the workspace for pending inputs and reduction scratch,
        # including the exact-size copy when overlapping keys reduce capacity.
        in_memory = (
            workspace.resident_bytes + 2 * capacity * self.dtype.itemsize <= workspace.budget // 2
        )
        output = np.empty(capacity, dtype=self.dtype) if in_memory else None
        path = None if in_memory else workspace.path()
        count = 0
        try:
            with ExitStack() as stack:
                handle = stack.enter_context(path.open("wb")) if path is not None else None
                block_rows = max(1, workspace.block_bytes // self.dtype.itemsize)
                a_reader, b_reader = (
                    _Reader(left, stack, block_rows),
                    _Reader(right, stack, block_rows),
                )
                a, b = a_reader.read(), b_reader.read()
                while a.size or b.size:
                    if not a.size:
                        piece, b = b, b_reader.read()
                    elif not b.size:
                        piece, a = a, a_reader.read()
                    else:
                        pivot = min(
                            (int(a["group"][-1]), int(a["bin"][-1])),
                            (int(b["group"][-1]), int(b["bin"][-1])),
                        )

                        def prefix(block, pivot=pivot):
                            return int(
                                np.count_nonzero(
                                    (block["group"] < pivot[0])
                                    | ((block["group"] == pivot[0]) & (block["bin"] <= pivot[1]))
                                )
                            )

                        na, nb = prefix(a), prefix(b)
                        if not nb:
                            piece = a[:na]
                        elif not na:
                            piece = b[:nb]
                        else:
                            piece = compact_rows(np.concatenate((a[:na], b[:nb])))
                        a, b = a[na:], b[nb:]
                        if not a.size:
                            a = a_reader.read()
                        if not b.size:
                            b = b_reader.read()
                    self._check_limit(count + piece.size)
                    if output is not None:
                        output[count : count + piece.size] = piece
                    else:
                        piece.tofile(handle)
                        workspace.spill_written(piece.nbytes)
                    count += piece.size
        except OSError as exc:
            raise OSError(
                "Density temporary merge failed; check free disk space and temporary-directory access."
            ) from exc
        left.release()
        right.release()
        # Compact a substantially reduced allocation while it is still bounded.
        if output is not None:
            output = output[:count].copy() if count != capacity else output
        return _Run(workspace, output, self.dtype, count, path)

    def finish(self) -> _Run:
        self.flush()
        runs = [self.levels[level] for level in sorted(self.levels)]
        self.levels.clear()
        if not runs:
            return _Run(self.workspace, np.empty(0, self.dtype), self.dtype, 0)
        result = runs[0]
        for run in runs[1:]:
            result = self._merge(result, run)
        return result
