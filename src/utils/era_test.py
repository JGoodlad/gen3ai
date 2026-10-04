"""The ERA label (`utils.era`): the table, the default / explicit run NAME, the immutable
`metadata.json` record, and the readers (`main.lineage`, the ELO headline, the cross-era warning)."""
import json
import re

import pytest

from utils import era as era_mod


# --- the table ---------------------------------------------------------------------------------

def test_the_table_is_ordered_unique_and_starts_at_rustboro():
    eras = era_mod.ERAS
    assert [e.order for e in eras] == list(range(1, len(eras) + 1))
    assert len({e.code for e in eras}) == len(eras) == len({e.name for e in eras})
    assert all(re.fullmatch(r"[a-z]{2}", e.code) for e in eras)
    assert eras[0].name == "Rustboro" and eras[0].code == "rb"
    assert era_mod.CURRENT_ERA is eras[0]


# --- run names ---------------------------------------------------------------------------------

def test_the_default_run_name_carries_the_era_prefix(run_archive):
    from main.train.run_io import _resolve_fresh_model_dir
    leaf = _resolve_fresh_model_dir(None, None, None).rsplit("/", 1)[1]
    assert re.fullmatch(r"rb_run_\d{8}_\d{6}", leaf)
    leaf = _resolve_fresh_model_dir(None, "ext_some_target", None).rsplit("/", 1)[1]
    assert leaf == "rb_exploiter_vs_some_target"


def test_the_launcher_default_run_name_carries_the_era_prefix(run_archive):
    from main.launcher.checkpoint import _resolve_fresh_run_dir
    assert _resolve_fresh_run_dir(["--debug"], "20261003_120000") == str(
        run_archive / "rb_run_20261003_120000")


def test_the_default_name_prefix_follows_current_era(run_archive, monkeypatch):
    # one edit (CURRENT_ERA) moves every default name to the next era
    monkeypatch.setattr(era_mod, "CURRENT_ERA", era_mod.ERAS[1])
    from main.train.run_io import _resolve_fresh_model_dir
    assert _resolve_fresh_model_dir(None, None, None).rsplit("/", 1)[1].startswith("dw_run_")


def test_an_explicit_name_is_accepted_unchanged_and_warns_without_the_prefix(run_archive, capsys):
    from main.train.run_io import _resolve_fresh_model_dir
    assert _resolve_fresh_model_dir("rb_x26_s1001", None, None) == str(run_archive / "rb_x26_s1001")
    assert "[Era]" not in capsys.readouterr().out            # prefixed: silent
    assert _resolve_fresh_model_dir("x26_s1001", None, None) == str(run_archive / "x26_s1001")
    out = capsys.readouterr().out
    assert "[Era]" in out and "no era prefix" in out and "rb_x26_s1001" in out   # unchanged, warned
    assert _resolve_fresh_model_dir("dw_x1", None, None) == str(run_archive / "dw_x1")
    assert "Dewford" in capsys.readouterr().out               # another era's prefix: warned too


def test_the_launcher_accepts_an_explicit_name_unchanged(run_archive):
    from main.launcher.checkpoint import _resolve_fresh_run_dir
    assert _resolve_fresh_run_dir(["--run-name", "plain"], "T") == str(run_archive / "plain")
    assert _resolve_fresh_run_dir(["--run-name", "rb_x26_s1001"], "T") == str(
        run_archive / "rb_x26_s1001")


def test_era_of_name_reads_only_a_leading_code_token():
    assert era_mod.era_of_name("rb_x26_s1001").name == "Rustboro"
    assert era_mod.era_of_name("rbx26") is None
    assert era_mod.era_of_name("x26_rb") is None
    assert era_mod.era_of_name("run_2026") is None
    assert era_mod.prefixed("rb_already") == "rb_already"      # never doubled


# --- the run record ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def version():
    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [64, 64]})


def _meta(d):
    return json.loads((d / "metadata.json").read_text())


def test_a_new_runs_metadata_records_the_era(tmp_path, version):
    from agents.model.snapshot import save_model_snapshot
    save_model_snapshot(str(tmp_path), version, git_hash="abc", original_command="x --steps 5")
    assert _meta(tmp_path)["era"] == {"code": "rb", "name": "Rustboro", "order": 1}
    assert era_mod.read_run_era(str(tmp_path))["code"] == "rb"


def test_the_era_is_stamped_even_if_an_early_eval_made_the_file_first(tmp_path, version):
    from agents.model.snapshot import record_eval_results, save_model_snapshot
    record_eval_results(str(tmp_path), 100, {"win_rate": 0.5})       # metadata.json exists, no era
    save_model_snapshot(str(tmp_path), version, git_hash="abc", original_command="x")
    assert _meta(tmp_path)["era"]["code"] == "rb"


def test_a_resume_never_changes_the_recorded_era(tmp_path, version, monkeypatch):
    from agents.model.snapshot import save_model_snapshot
    save_model_snapshot(str(tmp_path), version, git_hash="abc", original_command="x")
    before = _meta(tmp_path)["era"]
    monkeypatch.setattr(era_mod, "CURRENT_ERA", era_mod.ERAS[1])     # the project moves on...
    for _ in range(2):                                                # ...restarts / resumes save again
        save_model_snapshot(str(tmp_path), version, git_hash="def", cli_args={"steps": 9})
    assert _meta(tmp_path)["era"] == before == era_mod.ERAS[0].block()


def test_a_pre_era_run_reads_pre_era_and_a_resume_never_stamps_it(tmp_path, version):
    from agents.model.snapshot import save_model_snapshot
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"original_command": "old --steps 5", "cli_args": {"steps": 5}}))
    assert era_mod.read_run_era(str(tmp_path)) is None
    assert era_mod.era_label(era_mod.read_run_era(str(tmp_path))) == "pre-era"
    save_model_snapshot(str(tmp_path), version, git_hash="abc", cli_args={"steps": 6})   # a resume
    assert "era" not in _meta(tmp_path)
    assert era_mod.read_run_era(str(tmp_path)) is None


def test_a_missing_or_malformed_record_reads_pre_era(tmp_path):
    assert era_mod.read_run_era(str(tmp_path)) is None                # no metadata.json at all
    (tmp_path / "metadata.json").write_text(json.dumps({"era": "rb"}))   # a bare string is not a block
    assert era_mod.read_run_era(str(tmp_path)) is None


# --- the readers -------------------------------------------------------------------------------

def _run(root, name, era=None):
    d = root / name
    d.mkdir()
    meta = {"original_command": "x"}
    if era:
        meta["era"] = era.block()
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def test_main_lineage_prints_the_era(tmp_path, capsys):
    from main.lineage import main as lineage_main
    a = _run(tmp_path, "rb_a", era_mod.ERAS[0])
    b = _run(tmp_path, "old_b")
    lineage_main([str(a)])
    assert "era=rb Rustboro" in capsys.readouterr().out
    lineage_main([str(b)])
    assert "era=pre-era" in capsys.readouterr().out
    from main.lineage import read_run
    assert read_run(str(a))["era"]["name"] == "Rustboro" and read_run(str(b))["era"] is None


def test_the_cross_era_warning(tmp_path):
    a = _run(tmp_path, "rb_a", era_mod.ERAS[0])
    a2 = _run(tmp_path, "rb_a2", era_mod.ERAS[0])
    old = _run(tmp_path, "old")
    nxt = _run(tmp_path, "dw_n", era_mod.ERAS[1])
    assert era_mod.cross_era_warning(str(a), str(a2)) is None
    assert era_mod.cross_era_warning(str(old), str(_run(tmp_path, "old2"))) is None   # same: both pre-era
    w = era_mod.cross_era_warning(str(a), str(old))
    assert "CROSS-ERA" in w and "pre-era" in w and "rb Rustboro" in w
    assert "CROSS-ERA" in era_mod.cross_era_warning(str(a), str(nxt))


def test_the_critic_gate_ladder_section_carries_the_cross_era_warning():
    # the wiring: ladder_section's result has an `era_warning` key fed by cross_era_warning
    import inspect
    from main import critic_gate
    src = inspect.getsource(critic_gate.ladder_section)
    assert '"era_warning": cross_era_warning(run["run_dir"], parent["run_dir"])' in src


def test_the_elo_headline_names_the_era(tmp_path):
    from main import elo
    a = _run(tmp_path, "rb_a", era_mod.ERAS[0])
    # no ladder.json => the early-return text; the era tag rides the headline line proper
    import inspect
    assert "era_label(read_run_era(run_dir))" in inspect.getsource(elo.ladder_headline)
    assert elo.ladder_headline(str(a)).startswith("[ladder] no snapshot_ladder/ladder.json")
