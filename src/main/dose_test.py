"""`python -m main.dose` — the offline reader, on synthetic run dirs.

Model-free and torch-free by construction, so these tests build the artifacts a run writes rather
than a run. The one thing worth being pedantic about is the SOURCE PRECEDENCE: the sidecars are
preferred over `snapshot_history` because the history is CAPPED (a long run keeps ~15 rows while
its sidecars keep every un-groomed checkpoint), and a median over the capped set is a median over
a different run than the one being asked about.
"""
import json

import pytest

from main.dose import read_run, render


def _run(tmp_path, name, *, lrs, batch_size=2048, grad_accum_steps=16, n_epochs=7,
         history=None, dose=None):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    for i, lr in enumerate(lrs):
        step = (i + 1) * 1000
        (run / "checkpoints" / f"checkpoint_{step}_steps.json").write_text(json.dumps({
            "lr": lr, "n_epochs": n_epochs, "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
        }))
    meta = {}
    if history is not None:
        meta["snapshot_history"] = history
    if dose is not None:
        meta["dose"] = dose
    (run / "metadata.json").write_text(json.dumps(meta))
    return run


def test_it_computes_the_dose_from_the_SIDECARS(tmp_path):
    run = _run(tmp_path, "r", lrs=[1e-4, 1.2e-4, 0.8e-4])
    row = read_run(str(run))
    assert row["source"] == "sidecars"
    assert row["n_lr"] == 3
    assert row["lr_median"] == pytest.approx(1e-4)
    assert row["effective_batch"] == 32768
    assert row["dose_rate"] == pytest.approx(1e-4 * 7 / 32768)
    assert row["shape_stable"] is True and row["error"] is None


def test_the_median_is_over_the_TRAJECTORY_not_the_endpoints(tmp_path):
    run = _run(tmp_path, "r", lrs=[1e-4] * 9 + [1.0])
    assert read_run(str(run))["lr_median"] == pytest.approx(1e-4)


def test_checkpoints_are_ordered_by_STEP_not_by_filename(tmp_path):
    """`checkpoint_9_steps` must not sort after `checkpoint_10_steps` — the LAST row sets the shape."""
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint_9_steps.json").write_text(json.dumps(
        {"lr": 1e-4, "n_epochs": 7, "batch_size": 2048, "grad_accum_steps": 2}))
    (run / "checkpoints" / "checkpoint_10_steps.json").write_text(json.dumps(
        {"lr": 1e-4, "n_epochs": 7, "batch_size": 2048, "grad_accum_steps": 16}))
    row = read_run(str(run))
    assert row["grad_accum_steps"] == 16 and row["shape_stable"] is False


def test_it_FALLS_BACK_to_snapshot_history_when_the_sidecars_are_groomed(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"snapshot_history": {
        "a.zip": {"lr": 2e-5, "n_epochs": 10, "batch_size": 4096, "grad_accum_steps": 1},
        "b.zip": {"lr": 4e-5, "n_epochs": 10, "batch_size": 4096, "grad_accum_steps": 1},
    }}))
    row = read_run(str(run))
    assert row["source"] == "snapshot_history"
    assert row["lr_median"] == pytest.approx(3e-5)
    assert row["dose_rate"] == pytest.approx(3e-5 * 10 / 4096)


def test_the_sidecars_WIN_over_a_capped_history(tmp_path):
    run = _run(tmp_path, "r", lrs=[1e-4] * 5,
               history={"z.zip": {"lr": 9e-9, "n_epochs": 7, "batch_size": 2048,
                                  "grad_accum_steps": 16}})
    assert read_run(str(run))["source"] == "sidecars"


def test_a_run_with_only_a_run_level_lr_reports_ONE_point(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps(
        {"current_lr": 6e-5, "n_epochs": 10, "batch_size": 2048, "grad_accum_steps": 8}))
    row = read_run(str(run))
    assert row["source"] == "metadata" and row["n_lr"] == 1
    assert row["dose_rate"] == pytest.approx(6e-5 * 10 / 16384)


def test_a_run_that_recorded_NOTHING_reads_as_UNKNOWN_never_as_a_dose_of_zero(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text("{}")
    row = read_run(str(run))
    assert row["dose_rate"] is None and row["error"]


def test_a_missing_run_says_so(tmp_path):
    row = read_run(str(tmp_path / "nope"))
    assert row["dose_rate"] is None and "no such run" in row["error"]


def test_a_recorded_dose_block_is_reported_BESIDE_the_derived_one(tmp_path):
    run = _run(tmp_path, "r", lrs=[1e-4],
               dose={"dose_rate_now": 1.23e-8, "fork_lr": 7e-5, "lr_frozen": True})
    row = read_run(str(run))
    assert row["recorded_dose"] == pytest.approx(1.23e-8)
    assert row["fork_lr"] == pytest.approx(7e-5) and row["lr_frozen"] is True
    assert row["dose_rate"] == pytest.approx(1e-4 * 7 / 32768)   # still derived independently


def test_a_shape_that_MOVED_mid_run_is_flagged_and_uses_the_LAST_row(tmp_path):
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    for step, accum in ((1000, 2), (2000, 16)):
        (run / "checkpoints" / f"checkpoint_{step}_steps.json").write_text(json.dumps(
            {"lr": 1e-4, "n_epochs": 7, "batch_size": 2048, "grad_accum_steps": accum}))
    row = read_run(str(run))
    assert row["shape_stable"] is False and row["effective_batch"] == 32768
    assert "SHAPE MOVED" in render([row], None)


def test_the_ratio_column_is_against_the_reference_run(tmp_path):
    ref = read_run(str(_run(tmp_path, "ref", lrs=[1e-4])))
    arm = read_run(str(_run(tmp_path, "arm", lrs=[1e-4], grad_accum_steps=2)))
    table = render([ref, arm], ref)
    assert "1.00x" in table and "8.00x" in table


def test_the_ratio_is_omitted_rather_than_faked_when_there_is_no_reference(tmp_path):
    row = read_run(str(_run(tmp_path, "r", lrs=[1e-4])))
    assert "—" in render([row], None)


def test_the_markdown_form_is_a_table(tmp_path):
    row = read_run(str(_run(tmp_path, "r", lrs=[1e-4])))
    md = render([row], None, markdown=True)
    assert md.startswith("| run |") and "\n|---" in md


def test_the_cli_runs_end_to_end_and_emits_json(tmp_path, capsys):
    from main.dose import main

    a = _run(tmp_path, "a", lrs=[1e-4])
    b = _run(tmp_path, "b", lrs=[1e-4], grad_accum_steps=2)
    assert main([str(a), str(b), "--reference", str(a), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["run"] for r in payload["runs"]] == ["a", "b"]
    assert payload["reference"]["run"] == "a"
    assert payload["runs"][1]["dose_rate"] / payload["reference"]["dose_rate"] == pytest.approx(8.0)


def test_reference_none_omits_the_column(tmp_path, capsys):
    from main.dose import main

    a = _run(tmp_path, "a", lrs=[1e-4])
    assert main([str(a), "--reference", "none"]) == 0
    assert "reference:" not in capsys.readouterr().out


def test_the_module_imports_no_torch():
    """It must read a run whose architecture drifted past the current code — most of `models/`."""
    import subprocess
    import sys

    from utils.paths import src_root
    out = subprocess.run(
        [sys.executable, "-c",
         "import main.dose, sys; print('torch' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": str(src_root()), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "main.dose must stay importable without torch"
