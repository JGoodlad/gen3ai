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


# ------------------------------------------------------------------------------------
# num_timesteps — the RECORDED step, preferred over the step INFERRED from the filename
# ------------------------------------------------------------------------------------

def test_the_recorded_num_timesteps_orders_the_rows_over_the_filename(tmp_path):
    """The filename's number is an inference that only holds for the `checkpoint_<N>_steps`
    convention; the recorded key is what the save path wrote down. Where they disagree the
    record wins — here the name says 9 is oldest, the record says it is newest, and the LAST
    row's shape is what the dose is computed from."""
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint_9_steps.json").write_text(json.dumps(
        {"lr": 1e-4, "n_epochs": 7, "batch_size": 2048, "grad_accum_steps": 16,
         "num_timesteps": 900}))
    (run / "checkpoints" / "checkpoint_10_steps.json").write_text(json.dumps(
        {"lr": 1e-4, "n_epochs": 7, "batch_size": 2048, "grad_accum_steps": 2,
         "num_timesteps": 100}))
    row = read_run(str(run))
    assert row["grad_accum_steps"] == 16, "the row recording step 900 is the LAST one"
    assert row["num_timesteps"] == 900


def test_the_step_count_is_reported_and_rendered(tmp_path):
    run = _run(tmp_path, "r", lrs=[1e-4])
    (run / "metadata.json").write_text(json.dumps({"num_timesteps": 12_345_678}))
    row = read_run(str(run))
    assert row["num_timesteps"] == 12_345_678
    assert "12,345,678" in render([row], None)


def test_a_run_that_recorded_no_step_reads_UNKNOWN_not_zero(tmp_path):
    """Same rule the dose itself follows: a run that recorded nothing must not read as 0."""
    row = read_run(str(_run(tmp_path, "r", lrs=[1e-4])))
    assert row["num_timesteps"] is None
    assert "—" in render([row], None)


def test_the_rollout_rows_come_from_the_recorded_block_then_the_original_command(tmp_path):
    """K10(c): a ragged last step is a FULL optimizer step, so the dose needs the rollout size."""
    run = _run(tmp_path, "ragged", lrs=[2.8e-5], batch_size=2048, grad_accum_steps=32, n_epochs=10)
    row = read_run(str(run))
    assert row["rollout_rows"] is None and row["dose_rate"] == pytest.approx(2.8e-5 * 10 / 65_536)
    assert "ROWS UNKNOWN" in render([row], None)
    meta = {"original_command": "x --n-envs 48 --steps 1"}
    (run / "metadata.json").write_text(json.dumps(meta))
    side = run / "checkpoints" / "checkpoint_1000_steps.json"
    d = json.loads(side.read_text())
    side.write_text(json.dumps({**d, "n_steps": 2048}))
    row = read_run(str(run))
    assert (row["rollout_rows"], row["optimizer_steps_per_epoch"]) == (98_304, 2)
    assert row["rollout_source"].startswith("n_steps") and row["dose_rate"] == pytest.approx(2.8e-5 * 20 / 98_304)
    side.write_text(json.dumps({**d, "n_steps": 2048, "dose": {"rollout_rows": 131_072}}))
    row = read_run(str(run))
    assert (row["rollout_rows"], row["rollout_source"], row["optimizer_steps_per_epoch"]) == (131_072, "recorded", 2)
    assert "ROWS UNKNOWN" not in render([row], None)


# ------------------------------------------------------------------------------------
# The TB LR curve — the durable record the 2026-10-09 pre-Rustboro cleanup left (it deleted every
# intermediate checkpoint, sidecars included)
# ------------------------------------------------------------------------------------

def _history_only_run(tmp_path, name="r", *, history_lrs=(2e-5, 4e-5), n_epochs=10):
    """A run whose sidecars are gone: `metadata.json`'s capped `snapshot_history` is all that is left."""
    run = tmp_path / name
    run.mkdir()
    hist = {f"checkpoint_{(i + 1) * 1000}_steps.zip": {
        "lr": lr, "n_epochs": n_epochs, "batch_size": 4096, "grad_accum_steps": 1,
        "num_timesteps": (i + 1) * 1000} for i, lr in enumerate(history_lrs)}
    (run / "metadata.json").write_text(json.dumps({"snapshot_history": hist}))
    return run


def _tb(run, entries, name="events.out.tfevents.1700000000.host.1.0"):
    from agents.training.tb_inherit_test import _write_events

    _write_events(str(run / "tb"), entries, name=name)


def test_with_the_sidecars_gone_the_LR_comes_from_the_TB_curve_not_the_capped_history(tmp_path):
    """The archive's pre-Rustboro runs: the capped history keeps 3 rows (median 6.97e-5 on an
    exploiter whose LR climbed 2.8e-5 -> 8.4e-5); the TB curve holds every update."""
    run = _history_only_run(tmp_path, history_lrs=(4e-5, 8e-5, 9e-5))
    _tb(run, [(s, "train/learning_rate", lr) for s, lr in
              ((100, 2e-5), (200, 2e-5), (300, 2e-5), (400, 6e-5), (500, 9e-5))])
    row = read_run(str(run))
    assert row["source"] == "tb" and row["shape_source"] == "snapshot_history"
    assert row["n_lr"] == 5
    assert row["lr_median"] == pytest.approx(2e-5)            # the history's median would be 8e-5
    assert row["dose_rate"] == pytest.approx(2e-5 * 10 / 4096)
    assert "lr from tb (5 pts)" in render([row], None)


def test_a_forks_INHERITED_PREFIX_is_not_its_own_trajectory(tmp_path):
    """`tb_inherit` copies the parent's curve into the fork's tb/ as one file — the parent's LR."""
    from agents.training.tb_inherit import INHERITED_EVENTS_BASENAME

    run = _history_only_run(tmp_path)
    _tb(run, [(s, "train/learning_rate", 3e-4) for s in range(10, 60, 10)],
        name=INHERITED_EVENTS_BASENAME)                        # the PARENT's (high) curve
    _tb(run, [(s, "train/learning_rate", 3e-5) for s in range(100, 160, 10)])
    row = read_run(str(run))
    assert row["source"] == "tb" and row["n_lr"] == 6
    assert row["lr_median"] == pytest.approx(3e-5)


def test_a_step_two_event_files_both_carry_counts_once_and_the_later_file_wins(tmp_path):
    run = _history_only_run(tmp_path)
    _tb(run, [(100, "train/learning_rate", 1e-5), (200, "train/learning_rate", 1e-5)],
        name="events.out.tfevents.1700000001.host.1.0")
    _tb(run, [(200, "train/learning_rate", 5e-5), (300, "train/learning_rate", 5e-5)],
        name="events.out.tfevents.1700000002.host.1.0")        # a restart re-logging step 200
    row = read_run(str(run))
    assert row["n_lr"] == 3 and row["lr_median"] == pytest.approx(5e-5)


def test_the_sidecars_still_WIN_over_the_TB_curve(tmp_path):
    """Preference 1 is unchanged for every run that still has its checkpoints (every rb_ run)."""
    run = _run(tmp_path, "r", lrs=[1e-4] * 3)
    _tb(run, [(s, "train/learning_rate", 9e-9) for s in (1, 2, 3)])
    row = read_run(str(run))
    assert row["source"] == "sidecars" and row["lr_median"] == pytest.approx(1e-4)


def test_a_tb_dir_without_the_LR_tag_or_with_a_foreign_file_falls_through_to_the_history(tmp_path):
    run = _history_only_run(tmp_path, history_lrs=(2e-5, 4e-5))
    _tb(run, [(1, "train/other", 1.0)])
    (run / "tb" / "events.out.tfevents.1800000000.garbage").write_bytes(b"not an event file")
    row = read_run(str(run))
    assert row["source"] == "snapshot_history" and row["lr_median"] == pytest.approx(3e-5)
    assert "CAPPED/ONE-POINT" in render([row], None)


def test_a_run_with_a_TB_curve_but_no_shape_record_still_reads_UNKNOWN(tmp_path):
    """The TB curve supplies the LR only; the shape (batch, accumulation, epochs) still needs a record."""
    run = tmp_path / "r"
    run.mkdir()
    (run / "metadata.json").write_text("{}")
    _tb(run, [(1, "train/learning_rate", 1e-4)])
    row = read_run(str(run))
    assert row["dose_rate"] is None and row["error"]


def test_the_full_resolution_sources_are_the_ones_best_response_gap_compares_on():
    from agents.training import best_response_gap as brg
    from main import dose

    assert set(dose.FULL_RESOLUTION_SOURCES) <= set(dose.LR_SOURCES)
    assert tuple(brg.DOSE_FULL_RESOLUTION) == tuple(dose.FULL_RESOLUTION_SOURCES)
