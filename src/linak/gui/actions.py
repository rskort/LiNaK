"""Scalable workspace action registry and backend adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field as dataclass_field
import io
from pathlib import Path
from typing import Any, Literal

from .model import ProjectItem

ActionCategory = Literal["Compute", "Apply", "Convert", "Open"]


@dataclass(frozen=True)
class SettingField:
    """One typed settings field rendered by the shared settings dialog."""

    key: str
    label: str
    kind: Literal["text", "float", "int", "bool", "choice", "path"]
    default: Any = None
    choices: tuple[str, ...] = ()
    required: bool = False
    group: str = "General"
    help_text: str = ""
    minimum: float | None = None
    widget: Literal["auto", "species", "axis", "float", "int", "text", "path", "choice", "bool"] = "auto"
    unit: str = ""
    placeholder: str = ""
    auto_value: Any = None
    description: str = ""
    multi: bool = False
    advanced: bool = False


class ActionCanceled(RuntimeError):
    """Raised by cooperative GUI actions when cancellation is requested."""


@dataclass(frozen=True)
class ActionExecutionResult:
    """Result emitted by an action backend."""

    output_paths: tuple[Path, ...] = ()
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionContext:
    """Execution context passed to action backends."""

    project_dir: Path
    item: ProjectItem
    settings: dict[str, Any]
    log: Callable[[str, str], None]
    progress: Callable[[str, int | None, int | None], None]
    cancel_requested: Callable[[], bool] = dataclass_field(
        default_factory=lambda: (lambda: False)
    )


ActionBackend = Callable[[ActionContext], ActionExecutionResult]
SettingsFactory = Callable[[ProjectItem], list[SettingField]]
ExpectedOutputsFactory = Callable[[Path, ProjectItem, dict[str, Any]], tuple[Path, ...]]
SummaryFactory = Callable[[Path, ProjectItem, dict[str, Any]], str]


@dataclass(frozen=True)
class Action:
    """A compute, apply, convert, or open operation."""

    action_id: str
    category: ActionCategory
    name: str
    description: str
    supported_input_types: frozenset[str]
    output_type: str | None
    settings_factory: SettingsFactory
    backend: ActionBackend | None = None
    expected_outputs_factory: ExpectedOutputsFactory | None = None
    summary_factory: SummaryFactory | None = None

    def supports(self, item: ProjectItem) -> bool:
        if item.validation.state == "invalid":
            return False
        return item.item_type in self.supported_input_types

    def settings_schema(self, item: ProjectItem) -> list[SettingField]:
        return self.settings_factory(item)

    def expected_outputs(
        self,
        *,
        project_dir: Path,
        item: ProjectItem,
        settings: dict[str, Any],
    ) -> tuple[Path, ...]:
        if self.expected_outputs_factory is None:
            return ()
        return self.expected_outputs_factory(project_dir, item, settings)

    def summary(
        self,
        *,
        project_dir: Path,
        item: ProjectItem,
        settings: dict[str, Any],
    ) -> str:
        if self.summary_factory is not None:
            return self.summary_factory(project_dir, item, settings)
        outputs = self.expected_outputs(project_dir=project_dir, item=item, settings=settings)
        if outputs:
            return "Output: " + ", ".join(path.name for path in outputs)
        return "Ready to run."


def _no_settings(_item: ProjectItem) -> list[SettingField]:
    return []


def _trajectory_settings() -> list[SettingField]:
    return [
        SettingField("input", "Simulation input", "path", group="Cell / Metadata", widget="path"),
        SettingField(
            "cell",
            "Cell A B C",
            "text",
            group="Cell / Metadata",
            help_text="Optional explicit orthorhombic cell lengths in Angstrom.",
            placeholder="auto from .out.h5 when available",
        ),
    ]


def _surface_settings() -> list[SettingField]:
    return [
        SettingField("surface_mode", "Surface mode", "choice", "auto", ("auto", "layered", "rough"), group="Surface", widget="choice"),
        SettingField("surface_elements", "Surface elements", "text", group="Surface", help_text="Space-separated element symbols. Leave blank for automatic detection.", widget="species", multi=True),
        SettingField("include_fixed_surface_atoms", "Include fixed surface atoms", "bool", False, group="Surface"),
        SettingField("rough_surface_envelope", "Rough surface envelope", "float", None, group="Surface", minimum=0.0, widget="float", unit="A"),
    ]


def _convert_settings(item: ProjectItem) -> list[SettingField]:
    target_default = {
        "raw_trajectory": "traj.h5",
        "trajectory_hdf5": "xyz",
        "cube_file": "cube.h5",
        "cube_hdf5": "cube",
    }.get(item.item_type, "traj.h5")
    choices = ("traj.h5", "xyz") if item.item_type in {"raw_trajectory", "trajectory_hdf5"} else ("cube.h5", "cube")
    return [
        SettingField("target_file_type", "Target format", "choice", target_default, choices, True, "Output", widget="choice"),
        SettingField("select", "Frame selection", "text", group="Trajectory selection", help_text="Optional selector such as first:1000f, last:5ps, or range:100f:500f."),
        SettingField("input", "Simulation input", "path", group="Cell / Metadata", widget="path"),
        SettingField("cell", "Cell A B C", "text", group="Cell / Metadata", placeholder="auto from .out.h5 when available"),
    ]


def _pack_settings(item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("output_name", "Output name", "text", f"{item.path.name}.out.h5", True, "Output", placeholder="run.out.h5"),
        SettingField("overwrite", "Overwrite output", "bool", False, group="Output"),
        SettingField("include", "Include patterns", "text", group="Discovery", help_text="Optional glob patterns separated by spaces."),
        SettingField("exclude", "Exclude patterns", "text", group="Discovery", help_text="Optional glob patterns separated by spaces."),
        SettingField("drop", "Drop CP2K sections", "text", group="CP2K", help_text="Optional sections such as mulliken hirshfeld forces."),
    ]


def _density_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("species", "Species", "text", "all", True, group="Selection", widget="species", multi=True),
        SettingField("axis", "Axis", "choice", "z", ("x", "y", "z"), True, "Binning", widget="axis"),
        SettingField("bin_width", "Bin width", "float", 0.05, True, "Binning", minimum=0.0, widget="float", unit="A"),
        SettingField("outputs", "Outputs", "choice", "line", ("line", "heatmap", "all"), True, "Binning", widget="choice"),
        SettingField("heatmap_planes", "Heatmap planes", "text", group="Binning", help_text="Space-separated xy, xz, yz."),
        *_surface_settings(),
        *_trajectory_settings(),
    ]


def _msd_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("species", "Species", "text", "all", True, group="Selection", widget="species", multi=True),
        SettingField("timestep_fs", "Timestep fs", "float", None, group="Time", minimum=0.0, widget="float", unit="fs"),
        *_trajectory_settings(),
    ]


def _temperature_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("group_by", "Group by", "choice", "auto", ("auto", "elements", "regions", "both"), True, "Selection", widget="choice"),
        SettingField("input", "CP2K input", "path", group="Metadata", widget="path"),
        SettingField("velocity_unit", "Velocity unit", "choice", "auto", ("auto", "atomic", "angstrom/fs"), True, "Velocity", widget="choice"),
        SettingField("remove_com", "Remove COM velocity", "bool", False, group="Velocity"),
    ]


def _rdf_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("species_a", "Species A", "text", "all", True, group="Selection", widget="species"),
        SettingField("species_b", "Species B", "text", group="Selection", widget="species"),
        SettingField("r_max", "R max", "float", None, group="Binning", minimum=0.0, widget="float", unit="A"),
        SettingField("bin_width", "Bin width", "float", 0.05, True, "Binning", minimum=0.0, widget="float", unit="A"),
        SettingField("threads", "Threads", "int", None, group="Execution", minimum=1.0, widget="int"),
        *_surface_settings(),
        *_trajectory_settings(),
    ]


def _position_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("species", "Species", "text", "all", True, group="Selection", widget="species", multi=True),
        SettingField("axis", "Axis", "choice", "z", ("x", "y", "z"), True, "Geometry", widget="axis"),
        SettingField("timestep_fs", "Timestep fs", "float", None, group="Time", minimum=0.0, widget="float", unit="fs"),
        *_surface_settings(),
        *_trajectory_settings(),
    ]


def _coordination_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("species_a", "Species A", "text", "", True, group="Selection", widget="species"),
        SettingField("species_b", "Species B", "text", "", True, group="Selection", widget="species"),
        SettingField("axis", "Axis", "choice", "z", ("x", "y", "z"), True, "Geometry", widget="axis"),
        SettingField("timestep_fs", "Timestep fs", "float", None, group="Time", minimum=0.0, widget="float", unit="fs"),
        SettingField("cutoff", "Direct cutoff", "float", None, group="Cutoff", minimum=0.0, widget="float", unit="A"),
        SettingField("cutoff_rdf", "Cutoff RDF file", "path", group="Cutoff", widget="path"),
        SettingField("cutoff_smoothing_width", "Smoothing width", "float", 0.20, True, "Cutoff", minimum=0.0, widget="float", unit="A"),
        *_surface_settings(),
        *_trajectory_settings(),
    ]


def _orientation_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("axis", "Distance axis", "choice", "z", ("x", "y", "z"), True, "Geometry", widget="axis"),
        SettingField("reference_axis", "Reference axis", "choice", "z", ("x", "y", "z"), True, "Geometry", widget="axis"),
        SettingField("bin_width", "Bin width", "float", 0.01, True, "Binning", minimum=0.0, widget="float", unit="A"),
        SettingField("angle_bins", "Angle bins", "int", 100, True, "Binning", minimum=1.0, widget="int"),
        SettingField("oh_cutoff", "O-H cutoff", "float", 1.25, True, "Water detection", minimum=0.0, widget="float", unit="A"),
        *_surface_settings(),
        *_trajectory_settings(),
    ]


def _pbc_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("cell", "Cell A B C", "text", group="Cell", help_text="Optional explicit orthorhombic cell lengths in Angstrom.", placeholder="auto from .out.h5 when available"),
        SettingField("input", "Simulation input", "path", group="Cell", widget="path"),
    ]


def _potential_settings(_item: ProjectItem) -> list[SettingField]:
    return [
        SettingField("water_padding_ang", "Water padding", "float", 5.0, True, "Analysis", minimum=0.0, widget="float", unit="A"),
        SettingField("cshe_offset_ev", "cSHE offset", "float", 0.81, True, "Analysis", widget="float", unit="eV"),
        SettingField("threads", "Threads", "int", None, group="Execution", minimum=1.0, widget="int"),
        SettingField("include_failures", "Include failures", "bool", True, group="Execution"),
    ]


def _stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".out.hdf5", ".out.h5", ".traj.hdf5", ".traj.h5", ".cube.hdf5", ".cube.h5", ".hdf5", ".h5"):
        if lower.endswith(suffix):
            return name[: -len(suffix)] or "linak"
    return path.stem or "linak"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    lower = str(path).lower()
    for suffix in (".traj.hdf5", ".traj.h5", ".cube.hdf5", ".cube.h5", ".hdf5", ".h5"):
        if lower.endswith(suffix):
            base = str(path)[: -len(suffix)]
            counter = 1
            while True:
                candidate = Path(f"{base}_{counter}{str(path)[-len(suffix):]}")
                if not candidate.exists():
                    return candidate
                counter += 1
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _project_output(ctx: ActionContext, suffix: str) -> Path:
    return _unique_path(ctx.project_dir / f"{_stem(ctx.item.path)}{suffix}")


def _planned_output(project_dir: Path, item: ProjectItem, suffix: str) -> Path:
    return _unique_path(Path(project_dir).expanduser().resolve() / f"{_stem(item.path)}{suffix}")


def _analysis_output(project_dir: Path, source: Path, analysis: str) -> Path:
    from ..analysis.output_naming import analysis_hdf5_filename

    return _unique_path(Path(project_dir).expanduser().resolve() / analysis_hdf5_filename(source, analysis))


def _project_analysis_output(ctx: ActionContext, analysis: str) -> Path:
    return _analysis_output(ctx.project_dir, ctx.item.path, analysis)


def _split_words(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [token for token in str(value).replace(",", " ").split() if token.strip()]


def _add_optional(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    argv.extend([flag, str(value)])


def _add_cell(argv: list[str], value: Any) -> None:
    tokens = _split_words(value)
    if not tokens:
        return
    if len(tokens) != 3:
        raise ValueError("Cell must contain exactly three numbers: A B C.")
    argv.extend(["--cell", *tokens])


def validate_action_settings(action: Action, item: ProjectItem, settings: dict[str, Any]) -> None:
    """Validate settings independently of Qt widgets."""

    for field in action.settings_schema(item):
        value = settings.get(field.key)
        if field.required and (value is None or str(value).strip() == ""):
            raise ValueError(f"{field.label} is required.")
        if value is None or str(value).strip() == "":
            continue
        if field.kind == "float":
            value = float(value)
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{field.label} must be >= {field.minimum}.")
        elif field.kind == "int":
            value = int(value)
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{field.label} must be >= {int(field.minimum)}.")
        elif field.kind == "choice":
            if str(value) not in field.choices:
                raise ValueError(
                    f"{field.label} must be one of: {', '.join(field.choices)}."
                )
        elif field.kind == "path" and value:
            path = Path(str(value)).expanduser()
            if not path.exists():
                raise ValueError(f"{field.label} does not exist: {path}.")

    cell_value = settings.get("cell")
    if cell_value not in (None, ""):
        tokens = _split_words(cell_value)
        if len(tokens) != 3:
            raise ValueError("Cell must contain exactly three numbers: A B C.")
        for token in tokens:
            value = float(token)
            if value <= 0:
                raise ValueError("Cell lengths must be positive.")


def _add_surface(argv: list[str], settings: dict[str, Any]) -> None:
    _add_optional(argv, "--surface-mode", settings.get("surface_mode"))
    elements = _split_words(settings.get("surface_elements"))
    if elements:
        argv.extend(["--surface-elements", *elements])
    if bool(settings.get("include_fixed_surface_atoms")):
        argv.append("--include-fixed-surface-atoms")
    _add_optional(argv, "--rough-surface-envelope", settings.get("rough_surface_envelope"))


def _run_cli(argv: Sequence[str], log: Callable[[str, str], None]) -> int:
    from .. import cli as cli_mod

    parser = cli_mod.build_parser()
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        raise ValueError(f"Invalid LiNaK command arguments: {' '.join(argv)}") from exc
    args._runtime_argv = tuple(argv)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = int(args.handler(args) or 0)
    for line in stdout.getvalue().splitlines():
        if line.strip():
            log("INFO", line)
    for line in stderr.getvalue().splitlines():
        if line.strip():
            log("INFO", line)
    if rc != 0:
        raise RuntimeError(f"LiNaK backend returned exit code {rc}.")
    return rc


def _run_cli_with_expected_outputs(
    ctx: ActionContext,
    argv: Sequence[str],
    expected_outputs: Sequence[Path],
) -> ActionExecutionResult:
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled before backend command started.")
    ctx.log("INFO", "Running: linak " + " ".join(str(token) for token in argv))
    ctx.progress("Running LiNaK command", 1, 3)
    _run_cli(argv, ctx.log)
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled after backend command completed.")
    ctx.progress("Checking outputs", 2, 3)
    outputs = tuple(path for path in expected_outputs if path.exists())
    ctx.progress("Finished", 3, 3)
    return ActionExecutionResult(output_paths=outputs)


def _convert_backend(ctx: ActionContext) -> ActionExecutionResult:
    from ..conversion import (
        CONVERSION_REGISTRY,
        CubeConversionOptions,
        TrajectoryConversionOptions,
    )

    target_selector = str(ctx.settings.get("target_file_type") or "").strip()
    target_type = CONVERSION_REGISTRY.resolve_target_file_type(
        ctx.item.path,
        target_selector=target_selector or None,
    )
    default_path = CONVERSION_REGISTRY.default_output_path(
        ctx.item.path,
        target_file_type=target_type.id,
    )
    target_path = _unique_path(ctx.project_dir / default_path.name)
    request = CONVERSION_REGISTRY.build_request(
        ctx.item.path,
        output_path=target_path,
        target_selector=target_selector or None,
    )
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled before conversion started.")
    if request.family == "trajectory":
        cell_tokens = _split_words(ctx.settings.get("cell"))
        if cell_tokens and len(cell_tokens) != 3:
            raise ValueError("Cell must contain exactly three numbers: A B C.")
        options: Any = TrajectoryConversionOptions(
            input_path=ctx.settings.get("input") or None,
            select=ctx.settings.get("select") or None,
            cell=tuple(float(token) for token in cell_tokens) if cell_tokens else None,
            output_was_default=False,
        )
    else:
        options = CubeConversionOptions()
    ctx.log("INFO", f"Converting {request.source_path.name} to {request.target_file_type}.")
    ctx.progress("Converting file", 1, 2)
    result = CONVERSION_REGISTRY.execute(request, options=options)
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled after conversion completed.")
    ctx.progress("Finished", 2, 2)
    return ActionExecutionResult(output_paths=(result.output_path,), messages=result.metadata_notes)


def _pack_backend(ctx: ActionContext) -> ActionExecutionResult:
    from ..out_h5 import OutH5PackOptions, pack_simulation_directory

    output_name = str(ctx.settings.get("output_name") or f"{ctx.item.path.name}.out.h5").strip()
    output_path = Path(output_name).expanduser()
    if not output_path.is_absolute():
        output_path = ctx.project_dir / output_path
    options = OutH5PackOptions(
        include=tuple(_split_words(ctx.settings.get("include"))),
        exclude=tuple(_split_words(ctx.settings.get("exclude"))),
        overwrite=bool(ctx.settings.get("overwrite")),
        drop_sections=tuple(_split_words(ctx.settings.get("drop"))),
    )
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled before packing started.")
    result = pack_simulation_directory(
        ctx.item.path,
        output_path,
        options=options,
        progress=ctx.progress,
        logger=ctx.log,
        cancel_requested=ctx.cancel_requested,
    )
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled after packing completed.")
    ctx.progress("Finished", 1, 1)
    return ActionExecutionResult(output_paths=(result.output_path,), messages=result.messages)


def _compute_density_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "density")
    argv = ["compute", "density", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--species", ctx.settings.get("species"))
    _add_optional(argv, "--axis", ctx.settings.get("axis"))
    _add_optional(argv, "--bin-width", ctx.settings.get("bin_width"))
    _add_optional(argv, "--outputs", ctx.settings.get("outputs"))
    planes = _split_words(ctx.settings.get("heatmap_planes"))
    if planes:
        argv.extend(["--heatmap-planes", *planes])
    _add_surface(argv, ctx.settings)
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_msd_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "msd")
    argv = ["compute", "msd", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--species", ctx.settings.get("species"))
    _add_optional(argv, "--timestep-fs", ctx.settings.get("timestep_fs"))
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_temperature_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "temperature")
    argv = ["compute", "temperature", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--group-by", ctx.settings.get("group_by"))
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_optional(argv, "--velocity-unit", ctx.settings.get("velocity_unit"))
    if bool(ctx.settings.get("remove_com", False)):
        argv.append("--remove-com")
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_rdf_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "rdf")
    argv = ["compute", "rdf", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--species-a", ctx.settings.get("species_a"))
    _add_optional(argv, "--species-b", ctx.settings.get("species_b"))
    _add_optional(argv, "--r-max", ctx.settings.get("r_max"))
    _add_optional(argv, "--bin-width", ctx.settings.get("bin_width"))
    _add_optional(argv, "--threads", ctx.settings.get("threads"))
    _add_surface(argv, ctx.settings)
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_position_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "position")
    argv = ["compute", "position", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--species", ctx.settings.get("species"))
    _add_optional(argv, "--axis", ctx.settings.get("axis"))
    _add_optional(argv, "--timestep-fs", ctx.settings.get("timestep_fs"))
    _add_surface(argv, ctx.settings)
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_coordination_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "coordination")
    argv = ["compute", "coordination", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--species-a", ctx.settings.get("species_a"))
    _add_optional(argv, "--species-b", ctx.settings.get("species_b"))
    _add_optional(argv, "--axis", ctx.settings.get("axis"))
    _add_optional(argv, "--timestep-fs", ctx.settings.get("timestep_fs"))
    _add_optional(argv, "--cutoff", ctx.settings.get("cutoff"))
    _add_optional(argv, "--cutoff-rdf", ctx.settings.get("cutoff_rdf"))
    _add_optional(argv, "--cutoff-smoothing-width", ctx.settings.get("cutoff_smoothing_width"))
    _add_surface(argv, ctx.settings)
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_orientation_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "orientation")
    argv = ["compute", "orientation", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--axis", ctx.settings.get("axis"))
    _add_optional(argv, "--reference-axis", ctx.settings.get("reference_axis"))
    _add_optional(argv, "--bin-width", ctx.settings.get("bin_width"))
    _add_optional(argv, "--angle-bins", ctx.settings.get("angle_bins"))
    _add_optional(argv, "--oh-cutoff", ctx.settings.get("oh_cutoff"))
    _add_surface(argv, ctx.settings)
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _apply_pbc_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_output(ctx, "_pbc.xyz")
    argv = ["apply", "pbc", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--input", ctx.settings.get("input"))
    _add_cell(argv, ctx.settings.get("cell"))
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _compute_potential_backend(ctx: ActionContext) -> ActionExecutionResult:
    output = _project_analysis_output(ctx, "potential")
    argv = ["compute", "potential", str(ctx.item.path), "--output", str(output)]
    _add_optional(argv, "--water-padding-ang", ctx.settings.get("water_padding_ang"))
    _add_optional(argv, "--cshe-offset-ev", ctx.settings.get("cshe_offset_ev"))
    _add_optional(argv, "--threads", ctx.settings.get("threads"))
    if not bool(ctx.settings.get("include_failures", True)):
        argv.append("--no-include-failures")
    return _run_cli_with_expected_outputs(ctx, argv, (output,))


def _export_out_trajectory_backend(ctx: ActionContext) -> ActionExecutionResult:
    from ..out_h5 import export_out_h5_component

    output = _project_output(ctx, ".traj.h5")
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled before export started.")
    ctx.progress("Exporting trajectory", 1, 2)
    written = export_out_h5_component(ctx.item.path, "trajectory", output)
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled after export completed.")
    ctx.progress("Finished", 2, 2)
    return ActionExecutionResult(output_paths=(written,))


def _export_out_cube_backend(ctx: ActionContext) -> ActionExecutionResult:
    from ..out_h5 import export_out_h5_component

    output = _project_output(ctx, ".cube.h5")
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled before export started.")
    ctx.progress("Exporting cubes", 1, 2)
    written = export_out_h5_component(ctx.item.path, "cube", output)
    if ctx.cancel_requested():
        raise ActionCanceled("Task canceled after export completed.")
    ctx.progress("Finished", 2, 2)
    return ActionExecutionResult(output_paths=(written,))


def _expected_pack_outputs(
    project_dir: Path,
    item: ProjectItem,
    settings: dict[str, Any],
) -> tuple[Path, ...]:
    output_name = str(settings.get("output_name") or f"{item.path.name}.out.h5").strip()
    output = Path(output_name).expanduser()
    if not output.is_absolute():
        output = Path(project_dir).expanduser().resolve() / output
    if bool(settings.get("overwrite")):
        return (output,)
    return (_unique_path(output),)


def _expected_convert_outputs(
    project_dir: Path,
    item: ProjectItem,
    settings: dict[str, Any],
) -> tuple[Path, ...]:
    from ..conversion import CONVERSION_REGISTRY

    target_selector = str(settings.get("target_file_type") or "").strip()
    target_type = CONVERSION_REGISTRY.resolve_target_file_type(
        item.path,
        target_selector=target_selector or None,
    )
    default_path = CONVERSION_REGISTRY.default_output_path(
        item.path,
        target_file_type=target_type.id,
    )
    return (_unique_path(Path(project_dir).expanduser().resolve() / default_path.name),)


def _expected_suffix(suffix: str) -> ExpectedOutputsFactory:
    def _factory(project_dir: Path, item: ProjectItem, _settings: dict[str, Any]) -> tuple[Path, ...]:
        return (_planned_output(project_dir, item, suffix),)

    return _factory


def _expected_analysis(analysis: str) -> ExpectedOutputsFactory:
    def _factory(project_dir: Path, item: ProjectItem, _settings: dict[str, Any]) -> tuple[Path, ...]:
        return (_analysis_output(project_dir, item.path, analysis),)

    return _factory


def _action_summary(
    project_dir: Path,
    item: ProjectItem,
    settings: dict[str, Any],
    action_label: str,
    expected_outputs: ExpectedOutputsFactory,
) -> str:
    outputs = expected_outputs(project_dir, item, settings)
    output_text = ", ".join(path.name for path in outputs) if outputs else "no file output"
    return f"{action_label} will write to {Path(project_dir).name}: {output_text}"


def _summary(label: str, expected: ExpectedOutputsFactory) -> SummaryFactory:
    return lambda project_dir, item, settings: _action_summary(
        project_dir,
        item,
        settings,
        label,
        expected,
    )


def build_action_registry() -> list[Action]:
    trajectory_inputs = frozenset({"raw_trajectory", "trajectory_hdf5", "out_hdf5"})
    temperature_inputs = frozenset({"temperature_file", "raw_trajectory"})
    convertible_inputs = frozenset({"raw_trajectory", "trajectory_hdf5", "cube_file", "cube_hdf5"})
    analysis_outputs = frozenset({"analysis_hdf5", "cube_hdf5"})
    return [
        Action("pack_out_h5", "Convert", "Convert directory to .out.h5", "Pack a simulation output directory into one LiNaK output container.", frozenset({"simulation_directory"}), "out_hdf5", _pack_settings, _pack_backend, _expected_pack_outputs, _summary("Directory pack", _expected_pack_outputs)),
        Action("convert", "Convert", "Convert file", "Convert to another LiNaK-compatible working format.", convertible_inputs, None, _convert_settings, _convert_backend, _expected_convert_outputs, _summary("Conversion", _expected_convert_outputs)),
        Action("export_out_trajectory", "Convert", "Export trajectory", "Export trajectory data from this output container as .traj.h5.", frozenset({"out_hdf5"}), "trajectory_hdf5", _no_settings, _export_out_trajectory_backend, _expected_suffix(".traj.h5"), _summary("Trajectory export", _expected_suffix(".traj.h5"))),
        Action("export_out_cube", "Convert", "Export cubes", "Export cube data from this output container as .cube.h5.", frozenset({"out_hdf5"}), "cube_hdf5", _no_settings, _export_out_cube_backend, _expected_suffix(".cube.h5"), _summary("Cube export", _expected_suffix(".cube.h5"))),
        Action("density", "Compute", "Density", "Compute density profiles and heatmaps from a trajectory.", trajectory_inputs, "analysis_hdf5", _density_settings, _compute_density_backend, _expected_analysis("density"), _summary("Density", _expected_analysis("density"))),
        Action("msd", "Compute", "MSD", "Compute mean-squared displacement from a trajectory.", trajectory_inputs, "analysis_hdf5", _msd_settings, _compute_msd_backend, _expected_analysis("msd"), _summary("MSD", _expected_analysis("msd"))),
        Action("temperature", "Compute", "Temperature", "Compute temperature profiles from .temp, .tregion, or velocity XYZ.", temperature_inputs, "analysis_hdf5", _temperature_settings, _compute_temperature_backend, _expected_analysis("temperature"), _summary("Temperature", _expected_analysis("temperature"))),
        Action("rdf", "Compute", "RDF", "Compute radial distribution functions from a trajectory.", trajectory_inputs, "analysis_hdf5", _rdf_settings, _compute_rdf_backend, _expected_analysis("rdf"), _summary("RDF", _expected_analysis("rdf"))),
        Action("position", "Compute", "Position", "Compute atom-resolved positions and distance-to-surface profiles.", trajectory_inputs, "analysis_hdf5", _position_settings, _compute_position_backend, _expected_analysis("position"), _summary("Position", _expected_analysis("position"))),
        Action("coordination", "Compute", "Coordination", "Compute continuous coordination numbers.", trajectory_inputs, "analysis_hdf5", _coordination_settings, _compute_coordination_backend, _expected_analysis("coordination"), _summary("Coordination", _expected_analysis("coordination"))),
        Action("orientation", "Compute", "Orientation", "Compute water orientation versus distance to surface.", trajectory_inputs, "analysis_hdf5", _orientation_settings, _compute_orientation_backend, _expected_analysis("orientation"), _summary("Orientation", _expected_analysis("orientation"))),
        Action("pbc", "Apply", "Apply PBC", "Wrap trajectory atoms into the resolved periodic cell.", trajectory_inputs, "raw_trajectory", _pbc_settings, _apply_pbc_backend, _expected_suffix("_pbc.xyz"), _summary("PBC", _expected_suffix("_pbc.xyz"))),
        Action("potential", "Compute", "Potential", "Compute CP2K electrode cSHE potential from cube data.", frozenset({"cube_file", "cube_hdf5", "out_hdf5"}), "analysis_hdf5", _potential_settings, _compute_potential_backend, _expected_analysis("potential"), _summary("Potential", _expected_analysis("potential"))),
        Action("open_plot", "Open", "Open plot viewer", "Open this output in the LiNaK plotting GUI.", analysis_outputs, None, _no_settings, None),
    ]


class ActionRegistry:
    """Central lookup and filtering service for workspace actions."""

    def __init__(self, actions: Sequence[Action] | None = None) -> None:
        self._actions = list(build_action_registry() if actions is None else actions)

    @property
    def actions(self) -> list[Action]:
        return list(self._actions)

    def by_id(self, action_id: str) -> Action:
        for action in self._actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)

    def available_for(self, item: ProjectItem | None) -> list[Action]:
        if item is None:
            return []
        return [action for action in self._actions if action.supports(item)]
