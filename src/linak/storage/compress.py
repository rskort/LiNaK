"""CP2K output compression utilities used by ``linak apply compress``."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
from collections.abc import Iterable
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


LOG = logging.getLogger("linak.compress")
DEFAULT_BACKUP_SUBDIR = ".linak_backups"
READ_BUFFER_SIZE = 1024 * 1024

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
INT_RE = re.compile(r"\b\d+\b")
OUTER_SCF_ITER_RE = re.compile(r"outer SCF iter =\s*(\d+)")
CELL_PREFIX_RE = re.compile(r"^\s*(CELL(?:_TOP|_REF)?\|)\s*(.*)$")
CELL_VECTOR_RE = re.compile(
    r"Vector\s+([abc])\s+\[angstrom\]?:?\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+).*?\|[abc]\|\s*=\s*([+-]?\d+\.\d+)"
)
ATOMIC_KIND_RE = re.compile(r"^\s*(\d+)\.\s+Atomic kind:\s+([A-Za-z0-9_+-]+)\s+Number of atoms:\s+(\d+)\s*$")
MD_INI_RE = re.compile(rf"^\s*MD_INI\|\s+(.*?)\s+({FLOAT_PATTERN})\s*$")

SCF_ITERATION_FIELDS = [
    "scf_block_index",
    "md_step",
    "md_time_fs",
    "iteration",
    "update_method",
    "time",
    "convergence",
    "total_energy_hartree",
    "change_hartree",
]
MULLIKEN_FIELDS = [
    "scf_block_index",
    "md_step",
    "md_time_fs",
    "atom",
    "element",
    "kind",
    "x_ang",
    "y_ang",
    "z_ang",
    "population_alpha",
    "population_beta",
    "population_total",
    "net_charge",
    "spin_moment",
]
HIRSHFELD_FIELDS = [
    "scf_block_index",
    "md_step",
    "md_time_fs",
    "atom",
    "element",
    "kind",
    "x_ang",
    "y_ang",
    "z_ang",
    "ref_charge",
    "population_alpha",
    "population_beta",
    "population_total",
    "spin_moment",
    "net_charge",
]
FORCES_FIELDS = [
    "scf_block_index",
    "md_step",
    "md_time_fs",
    "atom",
    "element",
    "kind",
    "x_ang",
    "y_ang",
    "z_ang",
    "fx_hartree_per_bohr",
    "fy_hartree_per_bohr",
    "fz_hartree_per_bohr",
    "force_norm_hartree_per_bohr",
]
MD_STEPS_FIELDS = [
    "step",
    "time_fs",
    "conserved_quantity_hartree",
    "cpu_time_inst_s",
    "cpu_time_avg_s",
    "energy_drift_inst_k",
    "energy_drift_avg_k",
    "potential_energy_inst_hartree",
    "potential_energy_avg_hartree",
    "kinetic_energy_inst_hartree",
    "kinetic_energy_avg_hartree",
    "temperature_inst_k",
    "temperature_avg_k",
    "estimated_peak_process_memory_mib",
]
BLOCK_LINKED_STREAM_FILES = (
    "scf_iterations.csv",
    "mulliken.csv",
    "hirshfeld.csv",
    "forces.csv",
)


def local_now() -> datetime:
    return datetime.now().astimezone()


def iso_now() -> str:
    return local_now().isoformat(timespec="seconds")


def parse_float(value: str) -> float | None:
    try:
        return float(value.replace("D", "E"))
    except Exception:
        return None


def make_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.with_name(f"{base_dir.name}_{timestamp}")
    idx = 1
    while candidate.exists():
        candidate = base_dir.with_name(f"{base_dir.name}_{timestamp}_{idx}")
        idx += 1
    return candidate


def sanitize_filename_part(text: str, maxlen: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    safe = safe.strip("._")
    if not safe:
        safe = "file"
    return safe[:maxlen]


def build_backup_name(src: Path) -> str:
    st = src.stat()
    payload = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = sanitize_filename_part(src.stem)
    suffix = src.suffix if src.suffix else ".out"
    return f"compress_output__{stamp}__{stem}__{digest}{suffix}"


def default_backup_dir_for_input(input_path: Path) -> Path:
    return input_path.parent / DEFAULT_BACKUP_SUBDIR


def choose_backup_paths(src: Path, backup_dir: Path) -> tuple[Path, Path]:
    backup_name = build_backup_name(src)
    backup_path = backup_dir / backup_name
    idx = 1
    while backup_path.exists() or (backup_dir / f"{backup_path.name}.meta.json").exists():
        backup_path = backup_dir / f"{Path(backup_name).stem}_{idx}{Path(backup_name).suffix}"
        idx += 1
    meta_path = backup_dir / f"{backup_path.name}.meta.json"
    return backup_path, meta_path


def move_to_backup(src: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.debug("Moving input file to backup: %s -> %s", src, backup_path)
    shutil.move(str(src), str(backup_path))


def write_text(path: Path, text: str) -> None:
    make_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: Path, obj: Any) -> None:
    make_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    make_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def unique_preserve(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            out.append(line)
            seen.add(line)
    return out


@dataclass
class ParserOptions:
    drop_coordinates: bool = False
    drop_mulliken: bool = False
    drop_hirshfeld: bool = False
    drop_forces: bool = False
    drop_scf_iterations: bool = False
    drop_md_steps: bool = False
    drop_thermostat: bool = False
    drop_timing: bool = False
    drop_performance: bool = False
    drop_grid: bool = False


DROP_SECTION_TO_OPTION = {
    "coordinates": "drop_coordinates",
    "mulliken": "drop_mulliken",
    "hirshfeld": "drop_hirshfeld",
    "forces": "drop_forces",
    "scf-iterations": "drop_scf_iterations",
    "md-steps": "drop_md_steps",
    "thermostat": "drop_thermostat",
    "timing": "drop_timing",
    "performance": "drop_performance",
    "grid": "drop_grid",
}

DROP_SECTION_CHOICES = tuple(DROP_SECTION_TO_OPTION.keys())


@dataclass
class FileSpec:
    name: str
    format: str
    description: str
    generated: bool = False
    reason: str = "Not generated."
    size_bytes: int = 0


@dataclass
class ParseResult:
    original_output_path: str
    backup_path: str
    output_dir: str
    run_info_lines: list[str] = field(default_factory=list)
    parallel_setup_lines: list[str] = field(default_factory=list)
    restart_lines: list[str] = field(default_factory=list)
    constants_lines: list[str] = field(default_factory=list)
    cell_lines: list[str] = field(default_factory=list)
    dft_lines: list[str] = field(default_factory=list)
    functional_lines: list[str] = field(default_factory=list)
    vdw_lines: list[str] = field(default_factory=list)
    qs_lines: list[str] = field(default_factory=list)
    poisson_lines: list[str] = field(default_factory=list)
    ld_lines: list[str] = field(default_factory=list)
    md_par_lines: list[str] = field(default_factory=list)
    grid_lines: list[str] = field(default_factory=list)
    rot_dof_lines: list[str] = field(default_factory=list)
    electronic_summary_lines: list[str] = field(default_factory=list)
    scf_settings_blocks: list[list[str]] = field(default_factory=list)
    atomic_guess_lines: list[str] = field(default_factory=list)
    system_summary_lines: list[str] = field(default_factory=list)
    references_lines: list[str] = field(default_factory=list)
    cube_files: list[str] = field(default_factory=list)
    performance_summary_lines: list[str] = field(default_factory=list)
    warnings_counter: Counter[str] = field(default_factory=Counter)

    cell_rows: list[dict[str, Any]] = field(default_factory=list)
    atomic_kinds: list[dict[str, Any]] = field(default_factory=list)
    coordinates: list[dict[str, Any]] = field(default_factory=list)
    thermostat_rows: list[dict[str, Any]] = field(default_factory=list)
    scf_iterations: list[dict[str, Any]] = field(default_factory=list)
    scf_blocks: list[dict[str, Any]] = field(default_factory=list)
    mulliken_rows: list[dict[str, Any]] = field(default_factory=list)
    hirshfeld_rows: list[dict[str, Any]] = field(default_factory=list)
    forces_rows: list[dict[str, Any]] = field(default_factory=list)
    md_init_rows: list[dict[str, Any]] = field(default_factory=list)
    md_steps: list[dict[str, Any]] = field(default_factory=list)
    timing_rows: list[dict[str, Any]] = field(default_factory=list)
    stream_row_counts: dict[str, int] = field(default_factory=dict)
    parsed_row_counts: dict[str, int] = field(default_factory=dict)
    last_md_step: dict[str, Any] | None = None


def parsed_count(result: ParseResult, key: str, rows: list[dict[str, Any]]) -> int:
    value = result.parsed_row_counts.get(key)
    if value is not None:
        return int(value)
    return len(rows)


class StreamingCSVTable:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.handle: Any | None = None
        self.writer: csv.DictWriter[Any] | None = None
        self.rows = 0

    def write_row(self, row: dict[str, Any]) -> None:
        if self.handle is None:
            make_parent(self.path)
            self.handle = self.path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames, extrasaction="ignore")
            self.writer.writeheader()
        assert self.writer is not None
        self.writer.writerow(row)
        self.rows += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.writer = None


class StreamingCSVManager:
    def __init__(self, output_dir: Path, options: ParserOptions) -> None:
        self.tables: dict[str, StreamingCSVTable] = {}
        self._register(output_dir, "scf_iterations.csv", SCF_ITERATION_FIELDS, enabled=not options.drop_scf_iterations)
        self._register(output_dir, "mulliken.csv", MULLIKEN_FIELDS, enabled=not options.drop_mulliken)
        self._register(output_dir, "hirshfeld.csv", HIRSHFELD_FIELDS, enabled=not options.drop_hirshfeld)
        self._register(output_dir, "forces.csv", FORCES_FIELDS, enabled=not options.drop_forces)
        self._register(output_dir, "md_steps.csv", MD_STEPS_FIELDS, enabled=not options.drop_md_steps)

    def _register(self, output_dir: Path, filename: str, fieldnames: list[str], enabled: bool) -> None:
        if enabled:
            self.tables[filename] = StreamingCSVTable(output_dir / filename, fieldnames)

    def is_enabled(self, filename: str) -> bool:
        return filename in self.tables

    def write_row(self, filename: str, row: dict[str, Any]) -> None:
        table = self.tables.get(filename)
        if table is None:
            return
        table.write_row(row)

    def row_count(self, filename: str) -> int:
        table = self.tables.get(filename)
        if table is None:
            return 0
        return table.rows

    def row_counts(self) -> dict[str, int]:
        return {filename: table.rows for filename, table in self.tables.items()}

    def close(self) -> None:
        for table in self.tables.values():
            table.close()


class CP2KOutputParser:
    def __init__(self, source_path: Path, backup_path: Path | None, output_dir: Path, options: ParserOptions) -> None:
        self.source_path = source_path
        self.backup_path = backup_path
        self.output_dir = output_dir
        self.options = options

        self.result = ParseResult(
            original_output_path=str(source_path),
            backup_path=str(backup_path) if backup_path is not None else "",
            output_dir=str(output_dir),
        )
        self.stream_manager = StreamingCSVManager(output_dir, options)
        self.pending_stream_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
            filename: {} for filename in BLOCK_LINKED_STREAM_FILES if self.stream_manager.is_enabled(filename)
        }
        self.parsed_row_counts: Counter[str] = Counter()
        self.coords_by_atom: dict[int, dict[str, Any]] = {}

        self.pending_line: str | None = None

        self.in_restart = False
        self.restart_lines: list[str] = []

        self.in_constants = False

        self.in_atomic_kinds = False
        self.current_atomic_kind: dict[str, Any] | None = None

        self.in_coordinates = False

        self.in_scf_params = False
        self.current_scf_settings_lines: list[str] = []

        self.in_atomic_guess = False

        self.in_scf = False
        self.current_scf_block: dict[str, Any] | None = None

        self.in_mulliken = False
        self.current_mulliken_block_index: int | None = None

        self.in_hirshfeld = False
        self.current_hirshfeld_block_index: int | None = None

        self.in_forces = False
        self.current_forces_block_index: int | None = None

        self.in_warning = False
        self.current_warning_lines: list[str] = []

        self.in_references = False

        self.in_timing = False

        self.in_performance_section: str | None = None

        self.latest_scf_block_index: int | None = None
        self.last_unassigned_scf_block_index: int | None = None

        self.pending_cube_next_line = False

        self.current_md_step: dict[str, Any] | None = None

    def parse(self) -> ParseResult:
        LOG.info("Parsing input file: %s", self.source_path)

        # determine total number of lines for progress reporting
        LOG.info("Counting total number of lines...")
        with self.source_path.open("r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

        LOG.info("Total lines: %d", total_lines)

        line_count = 0
        progress_interval = 10000  # update every 10k lines

        with self.source_path.open("r", encoding="utf-8", errors="replace", buffering=READ_BUFFER_SIZE) as handle:
            while True:
                if self.pending_line is not None:
                    line = self.pending_line
                    self.pending_line = None
                else:
                    line = handle.readline()

                if not line:
                    break
                
                line_count += 1

                if total_lines > 0 and line_count % progress_interval == 0:
                    percent = 100.0 * line_count / total_lines
                    LOG.debug("Parsing progress: %.1f%% (%d/%d)", percent, line_count, total_lines)

                if self.in_warning:
                    if line.lstrip().startswith("***") or line.startswith(" ***"):
                        self.current_warning_lines.append(" ".join(line.strip().split()))
                        continue
                    self._finalize_warning_block()
                    self.pending_line = line
                    continue

                if self.in_restart:
                    if line.startswith(" DBCSR|") or line.startswith("  **** ****"):
                        self.result.restart_lines.extend(self.restart_lines)
                        self.restart_lines = []
                        self.in_restart = False
                        self.pending_line = line
                        continue
                    self.restart_lines.append(line.rstrip("\n"))
                    continue

                if self.in_constants:
                    if line.startswith(" CELL_TOP|") or line.startswith(" CELL|") or line.startswith(" CELL_REF|"):
                        self.in_constants = False
                        self.pending_line = line
                        continue
                    self.result.constants_lines.append(line.rstrip("\n"))
                    continue

                if self.in_atomic_kinds:
                    if "MOLECULE KIND INFORMATION" in line:
                        self._finalize_atomic_kind()
                        self.in_atomic_kinds = False
                        self.pending_line = line
                        continue
                    self._consume_atomic_kind_line(line)
                    continue

                if self.in_coordinates:
                    if line.startswith(" SCF PARAMETERS") or line.startswith(" Number of electrons:") or line.startswith(" Spin 1"):
                        self.in_coordinates = False
                        self.pending_line = line
                        continue
                    self._consume_coordinate_line(line)
                    continue

                if self.in_scf_params:
                    if (
                        line.startswith(" PW_GRID|")
                        or line.startswith(" RS_GRID|")
                        or line.startswith(" POISSON|")
                        or line.startswith(" Number of electrons:")
                        or line.startswith(" Spin 1")
                        or line.startswith(" Extrapolation method:")
                        or line.startswith(" SCF WAVEFUNCTION OPTIMIZATION")
                    ):
                        if self.current_scf_settings_lines:
                            self.result.scf_settings_blocks.append(self.current_scf_settings_lines[:])
                        self.current_scf_settings_lines = []
                        self.in_scf_params = False
                        self.pending_line = line
                        continue
                    if line.strip():
                        self.current_scf_settings_lines.append(line.rstrip("\n"))
                    continue

                if self.in_atomic_guess:
                    if line.startswith(" SCF WAVEFUNCTION OPTIMIZATION"):
                        self.in_atomic_guess = False
                        self.pending_line = line
                        continue
                    self.result.atomic_guess_lines.append(line.rstrip("\n"))
                    continue

                if self.in_scf:
                    if self._is_scf_end(line):
                        self._finalize_scf_block()
                        self.pending_line = line
                        continue
                    self._consume_scf_line(line)
                    continue

                if self.in_mulliken:
                    if line.strip().startswith("!-----------------------------------------------------------------------------!"):
                        self.in_mulliken = False
                        continue
                    self._consume_mulliken_line(line)
                    continue

                if self.in_hirshfeld:
                    if line.strip().startswith("!-----------------------------------------------------------------------------!"):
                        self.in_hirshfeld = False
                        continue
                    self._consume_hirshfeld_line(line)
                    continue

                if self.in_forces:
                    if not line.startswith(" FORCES|"):
                        self.in_forces = False
                        self.pending_line = line
                        continue
                    self._consume_forces_line(line)
                    continue

                if self.in_references:
                    if "T I M I N G" in line:
                        self.in_references = False
                        self.pending_line = line
                        continue
                    self.result.references_lines.append(line.rstrip("\n"))
                    continue

                if self.in_timing:
                    if line.startswith(" The number of warnings for this run is"):
                        self.in_timing = False
                        self.result.performance_summary_lines.append(line.rstrip("\n"))
                        continue
                    self._consume_timing_line(line)
                    continue

                self._consume_line(line)

        if self.in_warning:
            self._finalize_warning_block()

        if self.in_restart and self.restart_lines:
            self.result.restart_lines.extend(self.restart_lines)

        if self.in_scf and self.current_scf_block is not None:
            self._finalize_scf_block()

        if self.in_scf_params and self.current_scf_settings_lines:
            self.result.scf_settings_blocks.append(self.current_scf_settings_lines[:])

        if self.last_unassigned_scf_block_index is not None:
            self._flush_pending_stream_rows_for_block(self.last_unassigned_scf_block_index, None)
            self.last_unassigned_scf_block_index = None
        self._flush_all_pending_stream_rows()
        self.stream_manager.close()
        self.result.stream_row_counts = self.stream_manager.row_counts()
        self.result.parsed_row_counts = dict(self.parsed_row_counts)

        self._deduplicate_simple_text_sections()
        LOG.info(
            "Parsing complete: %d atoms, %d SCF blocks, %d MD steps, %d warnings",
            len(self.result.coordinates),
            len(self.result.scf_blocks),
            parsed_count(self.result, "md_steps.csv", self.result.md_steps),
            sum(self.result.warnings_counter.values()),
        )
        return self.result

    def _deduplicate_simple_text_sections(self) -> None:
        self.result.run_info_lines = unique_preserve(self.result.run_info_lines)
        self.result.parallel_setup_lines = unique_preserve(self.result.parallel_setup_lines)
        self.result.restart_lines = unique_preserve(self.result.restart_lines)
        self.result.constants_lines = unique_preserve(self.result.constants_lines)
        self.result.cell_lines = unique_preserve(self.result.cell_lines)
        self.result.dft_lines = unique_preserve(self.result.dft_lines)
        self.result.functional_lines = unique_preserve(self.result.functional_lines)
        self.result.vdw_lines = unique_preserve(self.result.vdw_lines)
        self.result.qs_lines = unique_preserve(self.result.qs_lines)
        self.result.poisson_lines = unique_preserve(self.result.poisson_lines)
        self.result.ld_lines = unique_preserve(self.result.ld_lines)
        self.result.md_par_lines = unique_preserve(self.result.md_par_lines)
        self.result.grid_lines = unique_preserve(self.result.grid_lines)
        self.result.rot_dof_lines = unique_preserve(self.result.rot_dof_lines)
        self.result.electronic_summary_lines = unique_preserve(self.result.electronic_summary_lines)
        self.result.system_summary_lines = unique_preserve(self.result.system_summary_lines)
        self.result.references_lines = unique_preserve(self.result.references_lines)
        self.result.performance_summary_lines = unique_preserve(self.result.performance_summary_lines)
        self.result.cube_files = unique_preserve(self.result.cube_files)

    def _increment_row_count(self, filename: str) -> None:
        self.parsed_row_counts[filename] += 1

    def _apply_coordinate_context(self, row: dict[str, Any]) -> None:
        atom = row.get("atom")
        if not isinstance(atom, int):
            return
        coord = self.coords_by_atom.get(atom)
        if coord is None:
            return
        row.setdefault("kind", coord.get("kind"))
        row.setdefault("element", coord.get("element"))
        row["x_ang"] = coord.get("x_ang")
        row["y_ang"] = coord.get("y_ang")
        row["z_ang"] = coord.get("z_ang")

    def _queue_stream_row_for_block(self, filename: str, block_index: int | None, row: dict[str, Any]) -> None:
        if not self.stream_manager.is_enabled(filename):
            return
        if block_index is None:
            self.stream_manager.write_row(filename, row)
            return
        rows_by_block = self.pending_stream_rows.setdefault(filename, {})
        rows_by_block.setdefault(block_index, []).append(row)

    def _flush_pending_stream_rows_for_block(self, block_index: int, md_row: dict[str, Any] | None) -> None:
        for filename in BLOCK_LINKED_STREAM_FILES:
            if not self.stream_manager.is_enabled(filename):
                continue
            rows_by_block = self.pending_stream_rows.get(filename)
            if rows_by_block is None:
                continue
            rows = rows_by_block.pop(block_index, [])
            for row in rows:
                if md_row is not None:
                    row["md_step"] = md_row.get("step")
                    row["md_time_fs"] = md_row.get("time_fs")
                self.stream_manager.write_row(filename, row)

    def _flush_all_pending_stream_rows(self) -> None:
        pending_block_indices: set[int] = set()
        for rows_by_block in self.pending_stream_rows.values():
            pending_block_indices.update(rows_by_block.keys())
        for block_index in sorted(pending_block_indices):
            self._flush_pending_stream_rows_for_block(block_index, None)

    def _finalize_warning_block(self) -> None:
        if self.current_warning_lines:
            key = "\n".join(self.current_warning_lines)
            self.result.warnings_counter[key] += 1
        self.current_warning_lines = []
        self.in_warning = False

    def _finalize_atomic_kind(self) -> None:
        if self.current_atomic_kind is not None:
            self.result.atomic_kinds.append(self.current_atomic_kind)
            self.current_atomic_kind = None

    def _finalize_scf_block(self) -> None:
        if self.current_scf_block is None:
            self.in_scf = False
            return

        steps = self.current_scf_block.pop("_steps")
        if steps:
            self.current_scf_block["n_iterations"] = len(steps)
            self.current_scf_block["first_energy_hartree"] = steps[0]["total_energy_hartree"]
            self.current_scf_block["final_energy_hartree"] = steps[-1]["total_energy_hartree"]
            self.current_scf_block["final_convergence"] = steps[-1]["convergence"]
            self.current_scf_block["final_change_hartree"] = steps[-1]["change_hartree"]
        else:
            self.current_scf_block["n_iterations"] = 0

        previous_unassigned = self.last_unassigned_scf_block_index
        block_index = int(self.current_scf_block["scf_block_index"])
        self.result.scf_blocks.append(self.current_scf_block)
        self.latest_scf_block_index = block_index
        if previous_unassigned is not None and previous_unassigned != block_index:
            self._flush_pending_stream_rows_for_block(previous_unassigned, None)
        self.last_unassigned_scf_block_index = block_index
        self.current_scf_block = None
        self.in_scf = False

    def _is_scf_end(self, line: str) -> bool:
        starts = (
            "  Electronic density on regular grids:",
            "  Core density on regular grids:",
            "  Total charge density on r-space grids:",
            "  Total charge density g-space grids:",
            "  Total dipole moment perpendicular to",
            "  the slab [electrons-Angstroem]:",
            "  Overlap energy of the core charge distribution:",
            "  Self energy of the core charge distribution:",
            "  Core Hamiltonian energy:",
            "  Hartree energy:",
            "  Exchange-correlation energy:",
            "  Dispersion energy:",
            "  Electronic entropic energy:",
            "  Fermi energy:",
            "  Total energy:",
            "  outer SCF iter =",
            "  Integrated spin density:",
            "  Integrated absolute spin density:",
            "  Ideal and single determinant S**2 :",
            " The electron density is written in cube file format to the file:",
            " !-----------------------------------------------------------------------------!",
            " ENERGY|",
            " MD_INI|",
            " MD|",
            " FORCES|",
            " -------------------------------------------------------------------------------",
        )
        return line.startswith(starts)

    def _consume_line(self, line: str) -> None:
        stripped = line.strip()

        if self.pending_cube_next_line and stripped:
            self.result.cube_files.append(stripped)
            self.pending_cube_next_line = False
            return

        if "RESTART INFORMATION" in line:
            self.in_restart = True
            self.restart_lines = [line.rstrip("\n")]
            return

        if line.startswith(" DBCSR|"):
            self.result.parallel_setup_lines.append(line.rstrip("\n"))
            return

        if line.startswith("  **** ****") or line.startswith(" ***** **") or line.startswith(" **    ****"):
            self.result.run_info_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" CP2K|") or line.startswith(" GLOBAL|"):
            self.result.run_info_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" MEMORY|") and "Estimated peak process memory" not in line:
            self.result.run_info_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" *** Fundamental physical constants"):
            self.in_constants = True
            self.result.constants_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" CELL_TOP|") or line.startswith(" CELL|") or line.startswith(" CELL_REF|"):
            self.result.cell_lines.append(line.rstrip("\n"))
            parsed = parse_cell_line(line)
            if parsed is not None:
                self.result.cell_rows.append(parsed)
            return

        if "ATOMIC KIND INFORMATION" in line:
            self.in_atomic_kinds = True
            self.current_atomic_kind = None
            return

        if "TOTAL NUMBERS AND MAXIMUM NUMBERS" in line:
            self.result.system_summary_lines.append(line.rstrip("\n"))
            return

        if line.startswith("  Total number of") or line.startswith("  Maximum angular momentum of"):
            self.result.system_summary_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"):
            self.in_coordinates = True
            return

        if line.startswith(" SCF PARAMETERS"):
            self.in_scf_params = True
            self.current_scf_settings_lines = [line.rstrip("\n")]
            return

        if line.startswith(" DFT|"):
            self.result.dft_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" FUNCTIONAL|"):
            self.result.functional_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" vdW POTENTIAL|"):
            self.result.vdw_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" QS|"):
            self.result.qs_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" POISSON|"):
            self.result.poisson_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" PW_GRID|") or line.startswith(" RS_GRID|"):
            self.result.grid_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" LD|"):
            self.result.ld_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" MD_PAR|"):
            self.result.md_par_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" ROT|") or line.startswith(" DOF|"):
            self.result.rot_dof_lines.append(line.rstrip("\n"))
            return

        thermostat_row = parse_thermostat_line(line)
        if thermostat_row is not None:
            self.result.thermostat_rows.append(thermostat_row)
            return

        if line.startswith(" MD_VEL|"):
            self.result.md_par_lines.append(line.rstrip("\n"))
            return

        if (
            line.startswith(" Spin 1")
            or line.startswith(" Spin 2")
            or line.startswith(" Number of electrons:")
            or line.startswith(" Number of occupied orbitals:")
            or line.startswith(" Number of molecular orbitals:")
            or line.startswith(" Number of orbital functions:")
            or line.startswith(" Number of independent orbital functions:")
            or line.startswith(" Extrapolation method:")
        ):
            self.result.electronic_summary_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" Parameters for the always stable predictor-corrector (ASPC) method:") or line.startswith("  ASPC order:") or line.startswith("  B(1) ="):
            self.result.md_par_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" Atomic guess:") or line.startswith(" Guess for atomic kind:"):
            self.in_atomic_guess = True
            self.result.atomic_guess_lines.append(line.rstrip("\n"))
            return

        if line.startswith(" SCF WAVEFUNCTION OPTIMIZATION"):
            self.in_scf = True
            next_index = len(self.result.scf_blocks) + 1
            self.current_scf_block = OrderedDict(
                [
                    ("scf_block_index", next_index),
                    ("md_step", None),
                    ("md_time_fs", None),
                    ("scf_converged_steps_reported", None),
                    ("first_energy_hartree", None),
                    ("final_energy_hartree", None),
                    ("final_convergence", None),
                    ("final_change_hartree", None),
                    ("outer_scf_iter", None),
                    ("outer_rms_gradient", None),
                    ("outer_energy_hartree", None),
                    ("electronic_density_regular_grid", None),
                    ("core_density_regular_grid", None),
                    ("charge_density_rspace", None),
                    ("charge_density_gspace", None),
                    ("dipole_perpendicular_eA", None),
                    ("overlap_energy_core_charge", None),
                    ("self_energy_core_charge", None),
                    ("core_hamiltonian_energy", None),
                    ("hartree_energy", None),
                    ("exchange_correlation_energy", None),
                    ("dispersion_energy", None),
                    ("electronic_entropic_energy", None),
                    ("fermi_energy", None),
                    ("total_energy_hartree", None),
                    ("energy_force_eval_hartree", None),
                    ("integrated_spin_density", None),
                    ("integrated_absolute_spin_density", None),
                    ("ideal_s2", None),
                    ("single_determinant_s2", None),
                    ("_steps", []),
                ]
            )
            return

        if line.startswith("  Electronic density on regular grids:"):
            pair = parse_two_floats_from_tail(line)
            block = self._latest_block()
            if block is not None and pair is not None:
                block["electronic_density_regular_grid"] = pair[0]
            return

        if line.startswith("  Core density on regular grids:"):
            pair = parse_two_floats_from_tail(line)
            block = self._latest_block()
            if block is not None and pair is not None:
                block["core_density_regular_grid"] = pair[0]
            return

        if line.startswith("  Total charge density on r-space grids:"):
            value = parse_last_float(line)
            block = self._latest_block()
            if block is not None:
                block["charge_density_rspace"] = value
            return

        if line.startswith("  Total charge density g-space grids:"):
            value = parse_last_float(line)
            block = self._latest_block()
            if block is not None:
                block["charge_density_gspace"] = value
            return

        if line.startswith("  the slab [electrons-Angstroem]:"):
            value = parse_last_float(line)
            block = self._latest_block()
            if block is not None:
                block["dipole_perpendicular_eA"] = value
            return

        if line.startswith("  Overlap energy of the core charge distribution:"):
            block = self._latest_block()
            if block is not None:
                block["overlap_energy_core_charge"] = parse_last_float(line)
            return

        if line.startswith("  Self energy of the core charge distribution:"):
            block = self._latest_block()
            if block is not None:
                block["self_energy_core_charge"] = parse_last_float(line)
            return

        if line.startswith("  Core Hamiltonian energy:"):
            block = self._latest_block()
            if block is not None:
                block["core_hamiltonian_energy"] = parse_last_float(line)
            return

        if line.startswith("  Hartree energy:"):
            block = self._latest_block()
            if block is not None:
                block["hartree_energy"] = parse_last_float(line)
            return

        if line.startswith("  Exchange-correlation energy:"):
            block = self._latest_block()
            if block is not None:
                block["exchange_correlation_energy"] = parse_last_float(line)
            return

        if line.startswith("  Dispersion energy:"):
            block = self._latest_block()
            if block is not None:
                block["dispersion_energy"] = parse_last_float(line)
            return

        if line.startswith("  Electronic entropic energy:"):
            block = self._latest_block()
            if block is not None:
                block["electronic_entropic_energy"] = parse_last_float(line)
            return

        if line.startswith("  Fermi energy:"):
            block = self._latest_block()
            if block is not None:
                block["fermi_energy"] = parse_last_float(line)
            return

        if line.startswith("  Total energy:"):
            block = self._latest_block()
            if block is not None:
                block["total_energy_hartree"] = parse_last_float(line)
            return

        if line.startswith("  outer SCF iter ="):
            block = self._latest_block()
            if block is not None:
                parsed = parse_outer_scf_line(line)
                if parsed is not None:
                    block.update(parsed)
            return

        if line.startswith("  Integrated spin density:"):
            block = self._latest_block()
            if block is not None:
                block["integrated_spin_density"] = parse_last_float(line)
            return

        if line.startswith("  Integrated absolute spin density:"):
            block = self._latest_block()
            if block is not None:
                block["integrated_absolute_spin_density"] = parse_last_float(line)
            return

        if line.startswith("  Ideal and single determinant S**2 :"):
            block = self._latest_block()
            if block is not None:
                vals = parse_all_floats(line)
                if len(vals) >= 2:
                    block["ideal_s2"] = vals[-2]
                    block["single_determinant_s2"] = vals[-1]
            return

        if line.startswith(" The electron density is written in cube file format to the file:"):
            self.pending_cube_next_line = True
            return

        if "Mulliken Population Analysis" in line:
            self.in_mulliken = True
            self.current_mulliken_block_index = self.latest_scf_block_index
            return

        if "Hirshfeld Charges" in line:
            self.in_hirshfeld = True
            self.current_hirshfeld_block_index = self.latest_scf_block_index
            return

        if line.startswith(" ENERGY| Total FORCE_EVAL"):
            block = self._latest_block()
            if block is not None:
                block["energy_force_eval_hartree"] = parse_last_float(line)
            return

        if line.startswith(" MD_INI|"):
            row = parse_md_ini_line(line)
            if row is not None:
                self.result.md_init_rows.append(row)
            return

        if line.startswith(" MD| Step number"):
            self.current_md_step = {"step": parse_last_int(line)}
            return

        if line.startswith(" MD| Time [fs]"):
            self._ensure_current_md_step()
            if self.current_md_step is not None:
                self.current_md_step["time_fs"] = parse_last_float(line)
            return

        if line.startswith(" MD| Conserved quantity [hartree]"):
            self._ensure_current_md_step()
            if self.current_md_step is not None:
                self.current_md_step["conserved_quantity_hartree"] = parse_last_float(line)
            return

        if line.startswith(" MD| CPU time per MD step [s]"):
            self._ensure_current_md_step()
            vals = parse_two_floats_from_tail(line)
            if self.current_md_step is not None and vals is not None:
                self.current_md_step["cpu_time_inst_s"] = vals[0]
                self.current_md_step["cpu_time_avg_s"] = vals[1]
            return

        if line.startswith(" MD| Energy drift per atom [K]"):
            self._ensure_current_md_step()
            vals = parse_two_floats_from_tail(line)
            if self.current_md_step is not None and vals is not None:
                self.current_md_step["energy_drift_inst_k"] = vals[0]
                self.current_md_step["energy_drift_avg_k"] = vals[1]
            return

        if line.startswith(" MD| Potential energy [hartree]"):
            self._ensure_current_md_step()
            vals = parse_two_floats_from_tail(line)
            if self.current_md_step is not None and vals is not None:
                self.current_md_step["potential_energy_inst_hartree"] = vals[0]
                self.current_md_step["potential_energy_avg_hartree"] = vals[1]
            return

        if line.startswith(" MD| Kinetic energy [hartree]"):
            self._ensure_current_md_step()
            vals = parse_two_floats_from_tail(line)
            if self.current_md_step is not None and vals is not None:
                self.current_md_step["kinetic_energy_inst_hartree"] = vals[0]
                self.current_md_step["kinetic_energy_avg_hartree"] = vals[1]
            return

        if line.startswith(" MD| Temperature [K]"):
            self._ensure_current_md_step()
            vals = parse_two_floats_from_tail(line)
            if self.current_md_step is not None and vals is not None:
                self.current_md_step["temperature_inst_k"] = vals[0]
                self.current_md_step["temperature_avg_k"] = vals[1]
            return

        if line.startswith(" MD| Estimated peak process memory after this step [MiB]"):
            self._ensure_current_md_step()
            if self.current_md_step is not None:
                self.current_md_step["estimated_peak_process_memory_mib"] = parse_last_float(line)
                completed_md_step = self.current_md_step
                self._increment_row_count("md_steps.csv")
                if self.stream_manager.is_enabled("md_steps.csv"):
                    self.stream_manager.write_row("md_steps.csv", completed_md_step)
                self.result.last_md_step = completed_md_step
                self._assign_latest_md_to_pending_block(completed_md_step)
                self.current_md_step = None
            return

        if line.startswith(" FORCES| Atomic forces"):
            self.in_forces = True
            self.current_forces_block_index = self.latest_scf_block_index
            return

        if line.lstrip().startswith("*** WARNING"):
            self.in_warning = True
            self.current_warning_lines = [" ".join(line.strip().split())]
            return

        if "DBCSR STATISTICS" in line:
            self.in_performance_section = "dbcsr_statistics"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if "DBCSR MESSAGE PASSING PERFORMANCE" in line:
            self.in_performance_section = "dbcsr_msg"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if "DBM STATISTICS" in line:
            self.in_performance_section = "dbm_statistics"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if "GRID STATISTICS" in line:
            self.in_performance_section = "grid_statistics"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if "MULTIGRID INFO" in line:
            self.in_performance_section = "multigrid"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if "MESSAGE PASSING PERFORMANCE" in line and "DBCSR" not in line:
            self.in_performance_section = "message_passing"
            self.result.performance_summary_lines.append(line.rstrip("\n"))
            return

        if self.in_performance_section is not None:
            if stripped == "" or set(stripped) == {"-"}:
                self.in_performance_section = None
                return

            if self._should_keep_performance_line(line, self.in_performance_section):
                self.result.performance_summary_lines.append(line.rstrip("\n"))
                return

            self.in_performance_section = None
            self.pending_line = line
            return

        if "R E F E R E N C E S" in line:
            self.in_references = True
            self.result.references_lines.append(line.rstrip("\n"))
            return

        if "T I M I N G" in line:
            self.in_timing = True
            return

        if "PROGRAM ENDED AT" in line or "PROGRAM RAN ON" in line or "PROGRAM RAN BY" in line or "PROGRAM STOPPED IN" in line:
            self.result.run_info_lines.append(line.rstrip("\n"))
            return

    def _should_keep_performance_line(self, line: str, section: str) -> bool:
        stripped = line.strip()
        if section == "dbcsr_statistics":
            keys = (
                "flops total",
                "flops max/rank",
                "matmuls total",
                "number of processed stacks",
                "average stack size",
                "marketing flops",
                "# multiplications",
                "max memory usage/rank",
                "# max total images/rank",
                "# max 3D layers",
                "# MPI messages exchanged",
                "MPI messages size (bytes):",
                "total size",
                "min size",
                "max size",
                "average size",
            )
            return any(k in stripped for k in keys)
        if section in {"dbcsr_msg", "message_passing"}:
            return stripped.startswith("ROUTINE") or stripped.startswith("MP_")
        if section == "dbm_statistics":
            return stripped.startswith("M") or stripped.startswith("COUNT")
        if section == "grid_statistics":
            return stripped.startswith("LP") or re.match(r"^\d+\s+\w+", stripped) is not None
        if section == "multigrid":
            return "count for grid" in stripped or "total gridlevel count" in stripped
        return False

    def _ensure_current_md_step(self) -> None:
        if self.current_md_step is None:
            self.current_md_step = {}

    def _assign_latest_md_to_pending_block(self, md_row: dict[str, Any]) -> None:
        if self.last_unassigned_scf_block_index is None:
            return
        block_index = self.last_unassigned_scf_block_index
        for block in reversed(self.result.scf_blocks):
            if int(block["scf_block_index"]) == block_index:
                block["md_step"] = md_row.get("step")
                block["md_time_fs"] = md_row.get("time_fs")
                break
        self._flush_pending_stream_rows_for_block(block_index, md_row)
        self.last_unassigned_scf_block_index = None

    def _latest_block(self) -> dict[str, Any] | None:
        if self.current_scf_block is not None:
            return self.current_scf_block
        if self.result.scf_blocks:
            return self.result.scf_blocks[-1]
        return None

    def _consume_atomic_kind_line(self, line: str) -> None:
        m = ATOMIC_KIND_RE.match(line)
        if m:
            self._finalize_atomic_kind()
            self.current_atomic_kind = OrderedDict(
                [
                    ("kind_index", int(m.group(1))),
                    ("element", m.group(2)),
                    ("number_of_atoms", int(m.group(3))),
                    ("basis_set", None),
                    ("potential", None),
                    ("covalent_radius_ang", None),
                    ("vdw_radius_ang", None),
                ]
            )
            return

        if self.current_atomic_kind is None:
            return

        if "Orbital Basis Set" in line:
            self.current_atomic_kind["basis_set"] = line.split()[-1]
            return

        if "GTH Potential information for" in line:
            self.current_atomic_kind["potential"] = line.split()[-1]
            return

        if "Atomic covalent radius [Angstrom]:" in line:
            self.current_atomic_kind["covalent_radius_ang"] = parse_last_float(line)
            return

        if "Atomic van der Waals radius [Angstrom]:" in line:
            self.current_atomic_kind["vdw_radius_ang"] = parse_last_float(line)
            return

    def _consume_coordinate_line(self, line: str) -> None:
        parts = line.split()
        if len(parts) < 9:
            return
        if not parts[0].isdigit():
            return
        row = {
            "atom": int(parts[0]),
            "kind": int(parts[1]),
            "element": parts[2],
            "atomic_number": int(parts[3]),
            "x_ang": parse_float(parts[4]),
            "y_ang": parse_float(parts[5]),
            "z_ang": parse_float(parts[6]),
            "z_eff": parse_float(parts[7]),
            "mass_amu": parse_float(parts[8]),
        }
        self.result.coordinates.append(row)
        self.coords_by_atom[row["atom"]] = row

    def _consume_scf_line(self, line: str) -> None:
        if self.current_scf_block is None:
            return
        parts = line.split()
        if len(parts) < 6:
            if line.strip().startswith("*** SCF run converged in"):
                m = re.search(r"converged in\s+(\d+)\s+steps", line)
                if m:
                    self.current_scf_block["scf_converged_steps_reported"] = int(m.group(1))
            return
        if not parts[0].isdigit():
            return
        tail_values = [parse_float(tok) for tok in parts[-4:]]
        if any(value is None for value in tail_values):
            return
        time_value, convergence_value, total_energy_value, change_value = tail_values

        step = int(parts[0])
        method = " ".join(parts[1:-4])
        row = {
            "scf_block_index": int(self.current_scf_block["scf_block_index"]),
            "md_step": None,
            "md_time_fs": None,
            "iteration": step,
            "update_method": method,
            "time": time_value,
            "convergence": convergence_value,
            "total_energy_hartree": total_energy_value,
            "change_hartree": change_value,
        }
        self.current_scf_block["_steps"].append(row)
        self._increment_row_count("scf_iterations.csv")
        self._queue_stream_row_for_block("scf_iterations.csv", int(row["scf_block_index"]), row)

    def _consume_mulliken_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return
        parts = stripped.split()
        if len(parts) < 5 or not parts[0].isdigit():
            return
        atom = int(parts[0])
        element = parts[1]
        kind = int(parts[2])
        rest = parts[3:]
        row: dict[str, Any] = {
            "scf_block_index": self.current_mulliken_block_index,
            "md_step": None,
            "md_time_fs": None,
            "atom": atom,
            "element": element,
            "kind": kind,
            "population_alpha": None,
            "population_beta": None,
            "population_total": None,
            "net_charge": None,
            "spin_moment": None,
        }

        if len(rest) >= 4:
            numeric = [parse_float(tok) for tok in rest[:4]]
            if any(value is None for value in numeric):
                numeric = []
        else:
            numeric = []
        if numeric:
            alpha, beta, net_charge, spin_moment = numeric
            row["population_alpha"] = alpha
            row["population_beta"] = beta
            row["population_total"] = (alpha if alpha is not None else 0.0) + (beta if beta is not None else 0.0)
            row["net_charge"] = net_charge
            row["spin_moment"] = spin_moment
        elif len(rest) >= 2:
            numeric_pair = [parse_float(tok) for tok in rest[:2]]
            if any(value is None for value in numeric_pair):
                return
            total, net_charge = numeric_pair
            row["population_total"] = total
            row["net_charge"] = net_charge
        else:
            return

        self._apply_coordinate_context(row)
        self._increment_row_count("mulliken.csv")
        self._queue_stream_row_for_block("mulliken.csv", self.current_mulliken_block_index, row)

    def _consume_hirshfeld_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return
        parts = stripped.split()
        if len(parts) < 6 or not parts[0].isdigit():
            return
        atom = int(parts[0])
        element = parts[1]
        kind = int(parts[2])
        rest = parts[3:]

        row: dict[str, Any] = {
            "scf_block_index": self.current_hirshfeld_block_index,
            "md_step": None,
            "md_time_fs": None,
            "atom": atom,
            "element": element,
            "kind": kind,
            "ref_charge": None,
            "population_alpha": None,
            "population_beta": None,
            "population_total": None,
            "spin_moment": None,
            "net_charge": None,
        }

        if len(rest) >= 5:
            numeric = [parse_float(tok) for tok in rest[:5]]
            if any(value is None for value in numeric):
                numeric = []
        else:
            numeric = []
        if numeric:
            ref_charge, pop_alpha, pop_beta, spin_moment, net_charge = numeric
            row["ref_charge"] = ref_charge
            row["population_alpha"] = pop_alpha
            row["population_beta"] = pop_beta
            row["population_total"] = (pop_alpha if pop_alpha is not None else 0.0) + (pop_beta if pop_beta is not None else 0.0)
            row["spin_moment"] = spin_moment
            row["net_charge"] = net_charge
        elif len(rest) >= 3:
            numeric_triplet = [parse_float(tok) for tok in rest[:3]]
            if any(value is None for value in numeric_triplet):
                return
            ref_charge, pop_total, net_charge = numeric_triplet
            row["ref_charge"] = ref_charge
            row["population_total"] = pop_total
            row["net_charge"] = net_charge
        else:
            return

        self._apply_coordinate_context(row)
        self._increment_row_count("hirshfeld.csv")
        self._queue_stream_row_for_block("hirshfeld.csv", self.current_hirshfeld_block_index, row)

    def _consume_forces_line(self, line: str) -> None:
        parts = line.split()
        if len(parts) >= 6 and parts[1].isdigit():
            row = {
                "scf_block_index": self.current_forces_block_index,
                "md_step": None,
                "md_time_fs": None,
                "atom": int(parts[1]),
                "fx_hartree_per_bohr": parse_float(parts[2]),
                "fy_hartree_per_bohr": parse_float(parts[3]),
                "fz_hartree_per_bohr": parse_float(parts[4]),
                "force_norm_hartree_per_bohr": parse_float(parts[5]),
            }
            self._apply_coordinate_context(row)
            self._increment_row_count("forces.csv")
            self._queue_stream_row_for_block("forces.csv", self.current_forces_block_index, row)
            return

        stripped = line.strip()
        if stripped.startswith("FORCES| Sum") or stripped.startswith("FORCES| Total atomic force"):
            self.result.performance_summary_lines.append(line.rstrip("\n"))

    def _consume_timing_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        if stripped.startswith("SUBROUTINE") or stripped.startswith("MAXIMUM") or set(stripped) == {"-"}:
            return
        parts = stripped.split()
        if len(parts) < 7:
            return
        asd_value = parse_float(parts[2])
        if asd_value is None:
            return
        tail_values = [parse_float(tok) for tok in parts[-4:]]
        if any(value is None for value in tail_values):
            return
        self_time_max, self_time_avg, total_time_max, total_time_avg = tail_values

        row = {
            "subroutine": parts[0],
            "calls": parts[1],
            "asd": asd_value,
            "self_time_max": self_time_max,
            "self_time_avg": self_time_avg,
            "total_time_max": total_time_max,
            "total_time_avg": total_time_avg,
        }
        self.result.timing_rows.append(row)


def parse_all_floats(line: str) -> list[float]:
    return [float(tok.replace("D", "E")) for tok in FLOAT_RE.findall(line)]


def parse_last_float(line: str) -> float | None:
    last_token: str | None = None
    for match in FLOAT_RE.finditer(line):
        last_token = match.group(0)
    if last_token is None:
        return None
    return float(last_token.replace("D", "E"))


def parse_last_int(line: str) -> int | None:
    nums = INT_RE.findall(line)
    if not nums:
        return None
    return int(nums[-1])


def parse_two_floats_from_tail(line: str) -> tuple[float, float] | None:
    prev_token: str | None = None
    last_token: str | None = None
    for match in FLOAT_RE.finditer(line):
        prev_token = last_token
        last_token = match.group(0)
    if prev_token is None or last_token is None:
        return None
    return float(prev_token.replace("D", "E")), float(last_token.replace("D", "E"))


def parse_cell_line(line: str) -> dict[str, Any] | None:
    prefix_match = CELL_PREFIX_RE.match(line)
    if not prefix_match:
        return None
    section = prefix_match.group(1).rstrip("|")
    body = prefix_match.group(2)

    if "Volume [angstrom^3]:" in body:
        return {
            "section": section,
            "property": "volume_angstrom3",
            "value_1": parse_last_float(line),
            "value_2": None,
            "value_3": None,
            "extra": None,
        }

    vector_match = CELL_VECTOR_RE.search(line)
    if vector_match:
        return {
            "section": section,
            "property": f"vector_{vector_match.group(1)}",
            "value_1": parse_float(vector_match.group(2)),
            "value_2": parse_float(vector_match.group(3)),
            "value_3": parse_float(vector_match.group(4)),
            "extra": parse_float(vector_match.group(5)),
        }

    if "alpha [degree]" in body:
        return {"section": section, "property": "alpha_deg", "value_1": parse_last_float(line), "value_2": None, "value_3": None, "extra": None}
    if "beta  [degree]" in body or "beta [degree]" in body:
        return {"section": section, "property": "beta_deg", "value_1": parse_last_float(line), "value_2": None, "value_3": None, "extra": None}
    if "gamma [degree]" in body:
        return {"section": section, "property": "gamma_deg", "value_1": parse_last_float(line), "value_2": None, "value_3": None, "extra": None}
    if "Numerically orthorhombic" in body:
        value = body.split(":")[-1].strip()
        return {"section": section, "property": "numerically_orthorhombic", "value_1": None, "value_2": None, "value_3": None, "extra": value}
    if "Periodicity" in body:
        value = body.split()[-1]
        return {"section": section, "property": "periodicity", "value_1": None, "value_2": None, "value_3": None, "extra": value}
    return None


def parse_outer_scf_line(line: str) -> dict[str, Any] | None:
    vals = parse_all_floats(line)
    match = OUTER_SCF_ITER_RE.search(line)
    if match is None:
        return None
    out: dict[str, Any] = {"outer_scf_iter": int(match.group(1))}
    if len(vals) >= 2:
        out["outer_rms_gradient"] = vals[-2]
        out["outer_energy_hartree"] = vals[-1]
    return out


def parse_thermostat_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    parts = stripped.split()
    if len(parts) != 5:
        return None
    if not parts[0].isdigit():
        return None
    if not parts[1].isdigit():
        return None
    target_temperature = parse_float(parts[3])
    coupling = parse_float(parts[4])
    if target_temperature is None or coupling is None:
        return None
    return {
        "atom": int(parts[0]),
        "group": int(parts[1]),
        "type": parts[2],
        "target_temperature_k": target_temperature,
        "coupling": coupling,
    }


def parse_md_ini_line(line: str) -> dict[str, Any] | None:
    m = MD_INI_RE.match(line)
    if not m:
        return None
    label = m.group(1).strip()
    value = parse_float(m.group(2))
    key = normalize_text_key(label)
    return {"key": key, "label": label, "value": value}


def normalize_text_key(text: str) -> str:
    key = text.lower()
    key = re.sub(r"\[[^\]]+\]", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = key.strip("_")
    return key


def build_summary(result: ParseResult) -> str:
    last_block = result.scf_blocks[-1] if result.scf_blocks else None
    last_md = result.last_md_step if result.last_md_step is not None else (result.md_steps[-1] if result.md_steps else None)
    scf_iteration_count = parsed_count(result, "scf_iterations.csv", result.scf_iterations)
    mulliken_count = parsed_count(result, "mulliken.csv", result.mulliken_rows)
    hirshfeld_count = parsed_count(result, "hirshfeld.csv", result.hirshfeld_rows)
    forces_count = parsed_count(result, "forces.csv", result.forces_rows)
    md_steps_count = parsed_count(result, "md_steps.csv", result.md_steps)
    lines = [
        "CP2K output compression summary",
        "",
        f"Original output path: {result.original_output_path}",
        f"Backup path:          {result.backup_path}",
        f"Output directory:     {result.output_dir}",
        "",
        f"Atomic kinds:         {len(result.atomic_kinds)}",
        f"Atoms in coordinates: {len(result.coordinates)}",
        f"SCF blocks:           {len(result.scf_blocks)}",
        f"SCF iterations:       {scf_iteration_count}",
        f"Mulliken rows:        {mulliken_count}",
        f"Hirshfeld rows:       {hirshfeld_count}",
        f"Forces rows:          {forces_count}",
        f"MD steps:             {md_steps_count}",
        f"Timing rows:          {len(result.timing_rows)}",
        f"Warnings:             {sum(result.warnings_counter.values())}",
        "",
    ]
    if last_block is not None:
        lines.extend(
            [
                "Final SCF block",
                f"  scf_block_index:      {last_block.get('scf_block_index')}",
                f"  md_step:              {last_block.get('md_step')}",
                f"  md_time_fs:           {last_block.get('md_time_fs')}",
                f"  n_iterations:         {last_block.get('n_iterations')}",
                f"  total_energy_hartree: {last_block.get('total_energy_hartree')}",
                f"  force_eval_hartree:   {last_block.get('energy_force_eval_hartree')}",
                f"  fermi_energy:         {last_block.get('fermi_energy')}",
                "",
            ]
        )
    if last_md is not None:
        lines.extend(
            [
                "Final MD step",
                f"  step:                 {last_md.get('step')}",
                f"  time_fs:              {last_md.get('time_fs')}",
                f"  temperature_inst_k:   {last_md.get('temperature_inst_k')}",
                f"  temperature_avg_k:    {last_md.get('temperature_avg_k')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def create_file_specs() -> OrderedDict[str, FileSpec]:
    specs = OrderedDict()
    specs["README.txt"] = FileSpec(
        name="README.txt",
        format="txt",
        description="Human-readable overview of every possible file, including files that were not generated.",
        generated=True,
        reason="Generated.",
    )
    specs["manifest.json"] = FileSpec(
        name="manifest.json",
        format="json",
        description="Machine-readable metadata about the extraction, input path, backup path, and generated files.",
    )
    specs["backup_info.txt"] = FileSpec(
        name="backup_info.txt",
        format="txt",
        description="Original output path, backup path, timestamps, and retention note.",
    )
    specs["summary.txt"] = FileSpec(
        name="summary.txt",
        format="txt",
        description="Compact overview of the extracted calculation data.",
    )
    specs["run_info.txt"] = FileSpec(
        name="run_info.txt",
        format="txt",
        description="Program start and end information, CP2K version information, and GLOBAL metadata.",
    )
    specs["parallel_setup.txt"] = FileSpec(
        name="parallel_setup.txt",
        format="txt",
        description="Top-level DBCSR and parallel setup lines.",
    )
    specs["setup_overview.txt"] = FileSpec(
        name="setup_overview.txt",
        format="txt",
        description="Combined restart info, system summary, and SCF settings in one file.",
    )
    specs["cell.csv"] = FileSpec(
        name="cell.csv",
        format="csv",
        description="Parsed simulation cell information from CELL, CELL_TOP, and CELL_REF lines.",
    )
    specs["atomic_kinds.csv"] = FileSpec(
        name="atomic_kinds.csv",
        format="csv",
        description="Compressed summary of atomic kinds, basis sets, and pseudopotentials.",
    )
    specs["coordinates.csv"] = FileSpec(
        name="coordinates.csv",
        format="csv",
        description="Initial coordinates with atom index, kind, element, Z, Z_eff, and mass.",
    )
    specs["settings_summary.txt"] = FileSpec(
        name="settings_summary.txt",
        format="txt",
        description="DFT, XC, vdW, QS, POISSON, LD, MD_PAR, and related settings.",
    )
    specs["grid_info.txt"] = FileSpec(
        name="grid_info.txt",
        format="txt",
        description="PW_GRID and RS_GRID information if kept.",
    )
    specs["rot_dof.txt"] = FileSpec(
        name="rot_dof.txt",
        format="txt",
        description="ROT and DOF text sections in compact form.",
    )
    specs["electronic_summary.txt"] = FileSpec(
        name="electronic_summary.txt",
        format="txt",
        description="Orbital counts, electron counts, extrapolation notes, and related electronic metadata.",
    )
    specs["atomic_guess.txt"] = FileSpec(
        name="atomic_guess.txt",
        format="txt",
        description="Atomic guess section printed before the main SCF loop if present.",
    )
    specs["thermostat.csv"] = FileSpec(
        name="thermostat.csv",
        format="csv",
        description="Per-atom thermostat assignment table extracted from the DOF section.",
    )
    specs["scf_iterations.csv"] = FileSpec(
        name="scf_iterations.csv",
        format="csv",
        description="All SCF iteration rows in structured form.",
    )
    specs["scf_blocks.csv"] = FileSpec(
        name="scf_blocks.csv",
        format="csv",
        description="One-row summary per SCF block with energy and convergence fields.",
    )
    specs["energy_components.csv"] = FileSpec(
        name="energy_components.csv",
        format="csv",
        description="Energy decomposition and spin diagnostics per SCF block.",
    )
    specs["mulliken.csv"] = FileSpec(
        name="mulliken.csv",
        format="csv",
        description="Mulliken populations and charges.",
    )
    specs["hirshfeld.csv"] = FileSpec(
        name="hirshfeld.csv",
        format="csv",
        description="Hirshfeld populations and charges.",
    )
    specs["forces.csv"] = FileSpec(
        name="forces.csv",
        format="csv",
        description="Atomic forces per SCF block.",
    )
    specs["md_steps.csv"] = FileSpec(
        name="md_steps.csv",
        format="csv",
        description="Per-step MD summary values from MD blocks.",
    )
    specs["warnings.txt"] = FileSpec(
        name="warnings.txt",
        format="txt",
        description="Unique warning blocks with counts.",
    )
    specs["timing.csv"] = FileSpec(
        name="timing.csv",
        format="csv",
        description="Timing table extracted into CSV format.",
    )
    specs["performance_summary.txt"] = FileSpec(
        name="performance_summary.txt",
        format="txt",
        description="Compressed summary of non-trivial performance-related sections.",
    )
    specs["cube_files.txt"] = FileSpec(
        name="cube_files.txt",
        format="txt",
        description="List of cube files mentioned in the output.",
    )
    return specs


def write_outputs(result: ParseResult, options: ParserOptions) -> dict[str, FileSpec]:
    output_dir = Path(result.output_dir)
    specs = create_file_specs()
    LOG.info("Writing compressed outputs to %s", output_dir)

    def mark_generated(filename: str, path: Path) -> None:
        spec = specs[filename]
        spec.generated = True
        spec.reason = "Generated."
        spec.size_bytes = path.stat().st_size if path.exists() else 0
        LOG.debug("Generated %s (%s)", path, format_size(spec.size_bytes))

    def mark_not_generated(filename: str, reason: str) -> None:
        spec = specs[filename]
        spec.generated = False
        spec.reason = reason
        LOG.debug("Skipped %s: %s", filename, reason)

    backup_info_text = (
        f"Original output path: {result.original_output_path}\n"
        f"Backup path:          {result.backup_path}\n"
        f"Output directory:     {result.output_dir}\n"
        f"Compressed at:        {iso_now()}\n"
        "Retention note:       Backup cleanup policy is environment-specific and not managed by LiNaK.\n"
    )
    path = output_dir / "backup_info.txt"
    write_text(path, backup_info_text)
    mark_generated("backup_info.txt", path)

    path = output_dir / "summary.txt"
    write_text(path, build_summary(result))
    mark_generated("summary.txt", path)

    if result.run_info_lines:
        path = output_dir / "run_info.txt"
        write_text(path, "\n".join(result.run_info_lines) + "\n")
        mark_generated("run_info.txt", path)
    else:
        mark_not_generated("run_info.txt", "No run info lines were detected.")

    if result.parallel_setup_lines:
        path = output_dir / "parallel_setup.txt"
        write_text(path, "\n".join(result.parallel_setup_lines) + "\n")
        mark_generated("parallel_setup.txt", path)
    else:
        mark_not_generated("parallel_setup.txt", "No DBCSR setup lines were detected.")

    setup_overview_parts: list[str] = []
    if result.restart_lines:
        setup_overview_parts.append("[RESTART]")
        setup_overview_parts.extend(result.restart_lines)
        setup_overview_parts.append("")
    if result.system_summary_lines:
        setup_overview_parts.append("[SYSTEM_SUMMARY]")
        setup_overview_parts.extend(result.system_summary_lines)
        setup_overview_parts.append("")
    if result.scf_settings_blocks:
        setup_overview_parts.append("[SCF_SETTINGS]")
        for idx, block in enumerate(result.scf_settings_blocks, start=1):
            setup_overview_parts.append(f"SCF_SETTINGS_BLOCK_{idx}")
            setup_overview_parts.extend(block)
            setup_overview_parts.append("")
    if setup_overview_parts:
        path = output_dir / "setup_overview.txt"
        write_text(path, "\n".join(setup_overview_parts).rstrip() + "\n")
        mark_generated("setup_overview.txt", path)
    else:
        mark_not_generated("setup_overview.txt", "No restart/system summary/SCF settings were detected.")

    if result.cell_rows:
        path = output_dir / "cell.csv"
        write_csv(
            path,
            result.cell_rows,
            ["section", "property", "value_1", "value_2", "value_3", "extra"],
        )
        mark_generated("cell.csv", path)
    else:
        mark_not_generated("cell.csv", "No parsed cell information was detected.")

    if result.atomic_kinds:
        path = output_dir / "atomic_kinds.csv"
        write_csv(
            path,
            result.atomic_kinds,
            ["kind_index", "element", "number_of_atoms", "basis_set", "potential", "covalent_radius_ang", "vdw_radius_ang"],
        )
        mark_generated("atomic_kinds.csv", path)
    else:
        mark_not_generated("atomic_kinds.csv", "No atomic kind information was detected.")

    if result.coordinates and not options.drop_coordinates:
        csv_path = output_dir / "coordinates.csv"
        write_csv(
            csv_path,
            result.coordinates,
            ["atom", "kind", "element", "atomic_number", "x_ang", "y_ang", "z_ang", "z_eff", "mass_amu"],
        )
        mark_generated("coordinates.csv", csv_path)
    elif options.drop_coordinates:
        mark_not_generated("coordinates.csv", "Dropped by command line option.")
    else:
        mark_not_generated("coordinates.csv", "No coordinates were detected.")

    settings_lines: list[str] = []
    for group_title, lines in [
        ("DFT", result.dft_lines),
        ("FUNCTIONAL", result.functional_lines),
        ("VDW", result.vdw_lines),
        ("QS", result.qs_lines),
        ("POISSON", result.poisson_lines),
        ("LD", result.ld_lines),
        ("MD_PAR", result.md_par_lines),
    ]:
        if lines:
            settings_lines.append(f"[{group_title}]")
            settings_lines.extend(lines)
            settings_lines.append("")

    if settings_lines:
        path = output_dir / "settings_summary.txt"
        write_text(path, "\n".join(settings_lines).rstrip() + "\n")
        mark_generated("settings_summary.txt", path)
    else:
        mark_not_generated("settings_summary.txt", "No settings lines were detected.")

    if result.grid_lines and not options.drop_grid:
        path = output_dir / "grid_info.txt"
        write_text(path, "\n".join(result.grid_lines) + "\n")
        mark_generated("grid_info.txt", path)
    elif options.drop_grid:
        mark_not_generated("grid_info.txt", "Dropped by command line option.")
    else:
        mark_not_generated("grid_info.txt", "No grid information was detected.")

    if result.rot_dof_lines:
        path = output_dir / "rot_dof.txt"
        write_text(path, "\n".join(result.rot_dof_lines) + "\n")
        mark_generated("rot_dof.txt", path)
    else:
        mark_not_generated("rot_dof.txt", "No ROT or DOF section was detected.")

    if result.electronic_summary_lines:
        path = output_dir / "electronic_summary.txt"
        write_text(path, "\n".join(result.electronic_summary_lines) + "\n")
        mark_generated("electronic_summary.txt", path)
    else:
        mark_not_generated("electronic_summary.txt", "No electronic summary lines were detected.")

    if result.atomic_guess_lines:
        path = output_dir / "atomic_guess.txt"
        write_text(path, "\n".join(result.atomic_guess_lines) + "\n")
        mark_generated("atomic_guess.txt", path)
    else:
        mark_not_generated("atomic_guess.txt", "No atomic guess section was detected.")

    if result.thermostat_rows and not options.drop_thermostat:
        path = output_dir / "thermostat.csv"
        write_csv(
            path,
            result.thermostat_rows,
            ["atom", "group", "type", "target_temperature_k", "coupling"],
        )
        mark_generated("thermostat.csv", path)
    elif options.drop_thermostat:
        mark_not_generated("thermostat.csv", "Dropped by command line option.")
    else:
        mark_not_generated("thermostat.csv", "No thermostat assignment table was detected.")

    scf_iterations_streamed_rows = result.stream_row_counts.get("scf_iterations.csv")
    if options.drop_scf_iterations:
        mark_not_generated("scf_iterations.csv", "Dropped by command line option.")
    elif scf_iterations_streamed_rows is not None:
        path = output_dir / "scf_iterations.csv"
        if scf_iterations_streamed_rows > 0:
            mark_generated("scf_iterations.csv", path)
        else:
            mark_not_generated("scf_iterations.csv", "No SCF iteration rows were detected.")
    elif result.scf_iterations:
        path = output_dir / "scf_iterations.csv"
        write_csv(path, result.scf_iterations, SCF_ITERATION_FIELDS)
        mark_generated("scf_iterations.csv", path)
    else:
        mark_not_generated("scf_iterations.csv", "No SCF iteration rows were detected.")

    if result.scf_blocks:
        path = output_dir / "scf_blocks.csv"
        write_csv(
            path,
            result.scf_blocks,
            [
                "scf_block_index",
                "md_step",
                "md_time_fs",
                "n_iterations",
                "scf_converged_steps_reported",
                "first_energy_hartree",
                "final_energy_hartree",
                "final_convergence",
                "final_change_hartree",
                "outer_scf_iter",
                "outer_rms_gradient",
                "outer_energy_hartree",
                "total_energy_hartree",
                "energy_force_eval_hartree",
                "fermi_energy",
            ],
        )
        mark_generated("scf_blocks.csv", path)

        energy_rows: list[dict[str, Any]] = []
        for row in result.scf_blocks:
            energy_rows.append(
                OrderedDict(
                    [
                        ("scf_block_index", row.get("scf_block_index")),
                        ("md_step", row.get("md_step")),
                        ("md_time_fs", row.get("md_time_fs")),
                        ("electronic_density_regular_grid", row.get("electronic_density_regular_grid")),
                        ("core_density_regular_grid", row.get("core_density_regular_grid")),
                        ("charge_density_rspace", row.get("charge_density_rspace")),
                        ("charge_density_gspace", row.get("charge_density_gspace")),
                        ("dipole_perpendicular_eA", row.get("dipole_perpendicular_eA")),
                        ("overlap_energy_core_charge", row.get("overlap_energy_core_charge")),
                        ("self_energy_core_charge", row.get("self_energy_core_charge")),
                        ("core_hamiltonian_energy", row.get("core_hamiltonian_energy")),
                        ("hartree_energy", row.get("hartree_energy")),
                        ("exchange_correlation_energy", row.get("exchange_correlation_energy")),
                        ("dispersion_energy", row.get("dispersion_energy")),
                        ("electronic_entropic_energy", row.get("electronic_entropic_energy")),
                        ("fermi_energy", row.get("fermi_energy")),
                        ("total_energy_hartree", row.get("total_energy_hartree")),
                        ("energy_force_eval_hartree", row.get("energy_force_eval_hartree")),
                        ("integrated_spin_density", row.get("integrated_spin_density")),
                        ("integrated_absolute_spin_density", row.get("integrated_absolute_spin_density")),
                        ("ideal_s2", row.get("ideal_s2")),
                        ("single_determinant_s2", row.get("single_determinant_s2")),
                    ]
                )
            )
        energy_path = output_dir / "energy_components.csv"
        write_csv(
            energy_path,
            energy_rows,
            list(energy_rows[0].keys()),
        )
        mark_generated("energy_components.csv", energy_path)
    else:
        mark_not_generated("scf_blocks.csv", "No SCF blocks were detected.")
        mark_not_generated("energy_components.csv", "No SCF blocks were detected.")

    mulliken_streamed_rows = result.stream_row_counts.get("mulliken.csv")
    if options.drop_mulliken:
        mark_not_generated("mulliken.csv", "Dropped by command line option.")
    elif mulliken_streamed_rows is not None:
        path = output_dir / "mulliken.csv"
        if mulliken_streamed_rows > 0:
            mark_generated("mulliken.csv", path)
        else:
            mark_not_generated("mulliken.csv", "No Mulliken section was detected.")
    elif result.mulliken_rows:
        path = output_dir / "mulliken.csv"
        write_csv(path, result.mulliken_rows, MULLIKEN_FIELDS)
        mark_generated("mulliken.csv", path)
    else:
        mark_not_generated("mulliken.csv", "No Mulliken section was detected.")

    hirshfeld_streamed_rows = result.stream_row_counts.get("hirshfeld.csv")
    if options.drop_hirshfeld:
        mark_not_generated("hirshfeld.csv", "Dropped by command line option.")
    elif hirshfeld_streamed_rows is not None:
        path = output_dir / "hirshfeld.csv"
        if hirshfeld_streamed_rows > 0:
            mark_generated("hirshfeld.csv", path)
        else:
            mark_not_generated("hirshfeld.csv", "No Hirshfeld section was detected.")
    elif result.hirshfeld_rows:
        path = output_dir / "hirshfeld.csv"
        write_csv(path, result.hirshfeld_rows, HIRSHFELD_FIELDS)
        mark_generated("hirshfeld.csv", path)
    else:
        mark_not_generated("hirshfeld.csv", "No Hirshfeld section was detected.")

    forces_streamed_rows = result.stream_row_counts.get("forces.csv")
    if options.drop_forces:
        mark_not_generated("forces.csv", "Dropped by command line option.")
    elif forces_streamed_rows is not None:
        path = output_dir / "forces.csv"
        if forces_streamed_rows > 0:
            mark_generated("forces.csv", path)
        else:
            mark_not_generated("forces.csv", "No forces section was detected.")
    elif result.forces_rows:
        path = output_dir / "forces.csv"
        write_csv(path, result.forces_rows, FORCES_FIELDS)
        mark_generated("forces.csv", path)
    else:
        mark_not_generated("forces.csv", "No forces section was detected.")

    md_steps_streamed_rows = result.stream_row_counts.get("md_steps.csv")
    if options.drop_md_steps:
        mark_not_generated("md_steps.csv", "Dropped by command line option.")
    elif md_steps_streamed_rows is not None:
        path = output_dir / "md_steps.csv"
        if md_steps_streamed_rows > 0:
            mark_generated("md_steps.csv", path)
        else:
            mark_not_generated("md_steps.csv", "No MD step summaries were detected.")
    elif result.md_steps:
        path = output_dir / "md_steps.csv"
        write_csv(path, result.md_steps, MD_STEPS_FIELDS)
        mark_generated("md_steps.csv", path)
    else:
        mark_not_generated("md_steps.csv", "No MD step summaries were detected.")

    if result.warnings_counter:
        lines = ["Unique warning blocks with counts", ""]
        total = sum(result.warnings_counter.values())
        lines.append(f"Total warning blocks: {total}")
        lines.append(f"Unique warning blocks: {len(result.warnings_counter)}")
        lines.append("")
        for warning_text, count in result.warnings_counter.items():
            lines.append(f"Count: {count}")
            lines.extend(warning_text.splitlines())
            lines.append("")
        path = output_dir / "warnings.txt"
        write_text(path, "\n".join(lines).rstrip() + "\n")
        mark_generated("warnings.txt", path)
    else:
        mark_not_generated("warnings.txt", "No warning block was detected.")

    if result.timing_rows and not options.drop_timing:
        path = output_dir / "timing.csv"
        write_csv(
            path,
            result.timing_rows,
            ["subroutine", "calls", "asd", "self_time_max", "self_time_avg", "total_time_max", "total_time_avg"],
        )
        mark_generated("timing.csv", path)
    elif options.drop_timing:
        mark_not_generated("timing.csv", "Dropped by command line option.")
    else:
        mark_not_generated("timing.csv", "No timing table was detected.")

    if result.performance_summary_lines and not options.drop_performance:
        path = output_dir / "performance_summary.txt"
        write_text(path, "\n".join(result.performance_summary_lines) + "\n")
        mark_generated("performance_summary.txt", path)
    elif options.drop_performance:
        mark_not_generated("performance_summary.txt", "Dropped by command line option.")
    else:
        mark_not_generated("performance_summary.txt", "No performance summary lines were detected.")

    if result.cube_files:
        path = output_dir / "cube_files.txt"
        write_text(path, "\n".join(result.cube_files) + "\n")
        mark_generated("cube_files.txt", path)
    else:
        mark_not_generated("cube_files.txt", "No cube file output was detected.")

    manifest = OrderedDict(
        [
            ("created_at", iso_now()),
            ("original_output_path", result.original_output_path),
            ("backup_path", result.backup_path),
            ("output_dir", result.output_dir),
            ("counts", OrderedDict(
                [
                    ("atomic_kinds", len(result.atomic_kinds)),
                    ("coordinates", len(result.coordinates)),
                    ("thermostat_rows", len(result.thermostat_rows)),
                    ("scf_blocks", len(result.scf_blocks)),
                    ("scf_iterations", parsed_count(result, "scf_iterations.csv", result.scf_iterations)),
                    ("mulliken_rows", parsed_count(result, "mulliken.csv", result.mulliken_rows)),
                    ("hirshfeld_rows", parsed_count(result, "hirshfeld.csv", result.hirshfeld_rows)),
                    ("forces_rows", parsed_count(result, "forces.csv", result.forces_rows)),
                    ("md_steps", parsed_count(result, "md_steps.csv", result.md_steps)),
                    ("timing_rows", len(result.timing_rows)),
                    ("warnings_total", sum(result.warnings_counter.values())),
                ]
            )),
            ("files", [
                OrderedDict(
                    [
                        ("name", spec.name),
                        ("format", spec.format),
                        ("description", spec.description),
                        ("generated", spec.generated),
                        ("reason", spec.reason),
                        ("size_bytes", spec.size_bytes),
                    ]
                )
                for spec in specs.values()
            ]),
        ]
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    mark_generated("manifest.json", manifest_path)

    readme_lines = [
        "Compressed CP2K output directory",
        "",
        f"Original output path: {result.original_output_path}",
        f"Backup path:          {result.backup_path}",
        f"Output directory:     {result.output_dir}",
        f"Created at:           {iso_now()}",
        "",
        "Files",
        "",
    ]
    for spec in specs.values():
        status = "GENERATED" if spec.generated else "NOT GENERATED"
        size_text = f" | size={format_size(spec.size_bytes)}" if spec.generated else ""
        readme_lines.append(f"{spec.name} | {status} | format={spec.format}{size_text}")
        readme_lines.append(f"  Description: {spec.description}")
        readme_lines.append(f"  Status note: {spec.reason}")
        readme_lines.append("")
    readme_path = output_dir / "README.txt"
    write_text(readme_path, "\n".join(readme_lines).rstrip() + "\n")
    mark_generated("README.txt", readme_path)

    generated_count = sum(1 for spec in specs.values() if spec.generated)
    skipped_count = sum(1 for spec in specs.values() if not spec.generated)
    LOG.info("Output writing complete: %d generated, %d skipped", generated_count, skipped_count)

    return specs


def validate_input_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a regular file: {path}")
    resolved = path.resolve()
    LOG.debug("Validated input path: %s", resolved)
    return resolved


def choose_output_dir_from_original(original_path: Path) -> Path:
    base_dir = original_path.with_suffix("")
    output_dir = make_unique_dir(base_dir)
    if output_dir != base_dir:
        LOG.info("Output directory already existed; using unique directory %s", output_dir)
    else:
        LOG.debug("Selected output directory %s", output_dir)
    return output_dir


@dataclass
class CompressionRunResult:
    input_path: Path
    output_dir: Path
    backup_path: Path
    metadata_path: Path
    generated_count: int
    skipped_count: int


def build_parser_options_from_drop_sections(
    drop_sections: Iterable[str] | None = None,
) -> ParserOptions:
    selected = set(drop_sections or [])
    unknown = sorted(selected - set(DROP_SECTION_CHOICES))
    if unknown:
        choices = ", ".join(DROP_SECTION_CHOICES)
        raise ValueError(f"Unknown drop section(s): {', '.join(unknown)}. Valid choices: {choices}")

    kwargs = {option_name: False for option_name in DROP_SECTION_TO_OPTION.values()}
    for section in selected:
        kwargs[DROP_SECTION_TO_OPTION[section]] = True
    return ParserOptions(**kwargs)


def write_backup_metadata(meta_path: Path, original_path: Path, backup_path: Path, output_dir: Path) -> None:
    meta = OrderedDict(
        [
            ("created_at", iso_now()),
            ("original_output_path", str(original_path)),
            ("backup_path", str(backup_path)),
            ("output_dir", str(output_dir)),
            (
                "retention_note",
                "Backup cleanup policy is environment-specific and not managed by LiNaK.",
            ),
        ]
    )
    write_json(meta_path, meta)
    LOG.debug("Wrote backup metadata: %s", meta_path)


def compress_cp2k_output(
    output_file: str | Path,
    *,
    backup_dir: str | Path | None = None,
    options: ParserOptions | None = None,
) -> CompressionRunResult:
    input_path = validate_input_path(Path(output_file).expanduser())
    if backup_dir is None:
        resolved_backup_dir = default_backup_dir_for_input(input_path).resolve()
    else:
        resolved_backup_dir = Path(backup_dir).expanduser().resolve()
    resolved_options = options or ParserOptions()

    LOG.info("Compressing %s", input_path)
    LOG.info("Backup directory: %s", resolved_backup_dir)

    output_dir = choose_output_dir_from_original(input_path)
    backup_path, meta_path = choose_backup_paths(input_path, resolved_backup_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    LOG.debug("Created output directory: %s", output_dir)

    parser = CP2KOutputParser(input_path, backup_path, output_dir, resolved_options)
    result = parser.parse()
    specs = write_outputs(result, resolved_options)
    LOG.info("Output processing complete; moving original file to backup")
    move_to_backup(input_path, backup_path)
    write_backup_metadata(meta_path, input_path, backup_path, output_dir)

    generated_count = sum(1 for spec in specs.values() if spec.generated)
    not_generated_count = sum(1 for spec in specs.values() if not spec.generated)

    LOG.info("Original output moved to backup: %s", backup_path)
    LOG.info("Compressed output directory:     %s", output_dir)
    LOG.info("Generated files: %d | Not generated: %d", generated_count, not_generated_count)
    return CompressionRunResult(
        input_path=input_path,
        output_dir=output_dir,
        backup_path=backup_path,
        metadata_path=meta_path,
        generated_count=generated_count,
        skipped_count=not_generated_count,
    )
