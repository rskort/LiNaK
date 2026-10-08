import pytest

from linak.analysis.output_naming import append_hdf5_name_suffix, numbered_hdf5_path
from linak.analysis.potential import _fallback_csv_path
from linak.analysis.rdf import _resolve_non_overwriting_rdf_output_path
from linak.cli import _unique_path_with_numeric_suffix as cli_unique_path
from linak.cli import _default_pbc_output_path
from linak.conversion import _unique_path_with_numeric_suffix as conversion_unique_path
from linak.gui.actions import _unique_path as gui_unique_path
from linak.gui.defaults import _unique_name as gui_default_unique_path
from linak.out_h5 import unique_out_h5_output_path
from linak.storage import compress


@pytest.mark.parametrize(
    "name,expected",
    [
        ("run.rdf.h5", "run_1.rdf.h5"),
        ("run.part.rdf.h5", "run_1.part.rdf.h5"),
        ("run.traj.hdf5", "run_1.traj.hdf5"),
        ("run.out.h5", "run_1.out.h5"),
        ("run.cube.h5", "run_1.cube.h5"),
        ("run.position.h5", "run_1.position.h5"),
        ("run.coordination.h5", "run_1.coordination.h5"),
        ("run.density.h5", "run_1.density.h5"),
        ("run.potential.h5", "run_1.potential.h5"),
        ("run.temperature.h5", "run_1.temperature.h5"),
        ("run.orientation.h5", "run_1.orientation.h5"),
        ("run.xyz.gz", "run_1.xyz.gz"),
        ("run.tar.gz", "run_1.tar.gz"),
        ("run.custom.EXT", "run_1.custom.EXT"),
        ("run.png", "run_1.png"),
        ("run", "run_1"),
    ],
)
def test_numbering_preserves_everything_from_first_dot(tmp_path, name, expected):
    source = tmp_path / "parent.with.dots" / name
    assert numbered_hdf5_path(source, 1) == source.with_name(expected)


@pytest.mark.parametrize(
    "resolver",
    [
        cli_unique_path,
        conversion_unique_path,
        gui_unique_path,
        gui_default_unique_path,
        unique_out_h5_output_path,
        _resolve_non_overwriting_rdf_output_path,
        _fallback_csv_path,
    ],
)
def test_all_output_resolvers_preserve_full_extension_and_skip_occupied_names(tmp_path, resolver):
    source = tmp_path / "run.part.rdf.h5"
    first = tmp_path / "run_1.part.rdf.h5"
    source.write_text("original", encoding="utf-8")
    first.write_text("first", encoding="utf-8")
    assert resolver(source) == tmp_path / "run_2.part.rdf.h5"
    assert source.read_text(encoding="utf-8") == "original"
    assert first.read_text(encoding="utf-8") == "first"


def test_selection_suffix_preserves_full_extension(tmp_path):
    assert append_hdf5_name_suffix(tmp_path / "run.part.traj.h5", "_selected") == (
        tmp_path / "run_selected.part.traj.h5"
    )


def test_pbc_output_preserves_compound_extension(tmp_path):
    assert _default_pbc_output_path(tmp_path / "run.part.traj.h5") == (
        tmp_path / "run_pbc.part.traj.h5"
    )


def test_backup_collision_preserves_compound_extension(tmp_path, monkeypatch):
    source = tmp_path / "run.part.out.gz"
    source.write_text("original", encoding="utf-8")
    assert compress.build_backup_name(source).endswith(".part.out.gz")
    monkeypatch.setattr(compress, "build_backup_name", lambda _: "backup.part.out.gz")
    (tmp_path / "backup.part.out.gz").write_text("occupied", encoding="utf-8")
    # A metadata-only collision must also reserve its matching backup name.
    (tmp_path / "backup_1.part.out.gz.meta.json").write_text("{}", encoding="utf-8")
    backup, metadata = compress.choose_backup_paths(source, tmp_path)
    assert backup.name == "backup_2.part.out.gz"
    assert metadata.name == "backup_2.part.out.gz.meta.json"
