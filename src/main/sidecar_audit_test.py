"""`python -m main.sidecar_audit` — does every sidecar agree with its run about the commit?

The tool exists because the 2026-09-05 defect (a checkpoint sidecar stamped with the ambient
main HEAD instead of the run's pin) left an unknown number of historical sidecars
misattributing their code, and the fix says nothing about how many. This is the counter.

Run: python -m pytest src/main/sidecar_audit_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import json

from main import sidecar_audit as sa


def _run_dir(tmp_path, name, *, meta, sidecars):
    """A minimal run dir: metadata.json + `<ckpt>.zip` / `<ckpt>.json` pairs."""
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps(meta))
    for ckpt_name, entry in sidecars.items():
        (run / "checkpoints" / f"{ckpt_name}.zip").write_text("")
        (run / "checkpoints" / f"{ckpt_name}.json").write_text(json.dumps(entry))
    return run


def test_one_matching_and_one_mismatching_sidecar(tmp_path, capsys):
    run = _run_dir(
        tmp_path, "run_x",
        meta={"git_hash": "aaaa1111bbbb", "pin_source": "checkpoint"},
        sidecars={
            "checkpoint_100_steps": {"git_hash": "aaaa1111bbbb", "lr": 3e-4},
            "checkpoint_200_steps": {"git_hash": "ffff9999cccc", "lr": 3e-4},
        },
    )
    rec = sa.audit_run(str(run))
    assert rec["n_sidecars"] == 2
    assert rec["n_mismatched"] == 1
    # Nothing in this run's records mentions ffff9999 ⇒ the sidecar misattributes its code.
    assert rec["n_unexplained"] == 1
    bad = [s for s in rec["sidecars"] if not s["matches_run"]]
    assert bad[0]["git_hash"] == "ffff9999cccc"
    assert bad[0]["explained_by_history"] is False

    assert sa.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "run_x" in out and "ffff9999" in out
    assert "1 differing" in out and "1 UNEXPLAINED" in out
    assert sa.main([str(run), "--strict"]) == 1, "--strict must fail on an unexplained hash"


def test_a_mismatch_covered_by_pin_history_is_EXPLAINED_not_a_defect(tmp_path, capsys):
    """A run that restarted onto new code legitimately has sidecars from the earlier span."""
    run = _run_dir(
        tmp_path, "run_split",
        meta={
            "git_hash": "bbbb2222", "pin_source": "head",
            "pin_history": [
                {"git_hash": "aaaa1111", "pin_source": "pin_commit",
                 "first_step": 0, "last_step": 1000},
                {"git_hash": "bbbb2222", "pin_source": "head",
                 "first_step": 1500, "last_step": 2000},
            ],
        },
        sidecars={
            "checkpoint_1000_steps": {"git_hash": "aaaa1111"},
            "checkpoint_2000_steps": {"git_hash": "bbbb2222"},
        },
    )
    rec = sa.audit_run(str(run))
    assert rec["pin_split"] is True
    assert rec["n_mismatched"] == 1 and rec["n_unexplained"] == 0

    assert sa.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "PIN-SPLIT" in out
    assert "steps 0 → 1000" in out and "steps 1500 → 2000" in out
    assert sa.main([str(run), "--strict"]) == 0, "an explained span is not a --strict failure"


def test_a_models_dir_is_expanded_into_its_runs(tmp_path, capsys):
    _run_dir(tmp_path, "run_a", meta={"git_hash": "aaaa"},
             sidecars={"checkpoint_1_steps": {"git_hash": "aaaa"}})
    _run_dir(tmp_path, "run_b", meta={"git_hash": "bbbb"},
             sidecars={"checkpoint_1_steps": {"git_hash": "zzzz"}})
    (tmp_path / "not_a_run").mkdir()          # no metadata.json → not a run

    runs = sa.discover_runs([str(tmp_path)])
    assert sorted(r.rsplit("/", 1)[-1] for r in runs) == ["run_a", "run_b"]

    assert sa.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "SUMMARY: 2 run(s) · 2 sidecar(s)" in out
    assert "1 differing from their run hash in 1 run(s)" in out


def test_a_run_with_no_pin_history_says_so_rather_than_implying_one_commit(tmp_path, capsys):
    run = _run_dir(tmp_path, "run_legacy", meta={"git_hash": "aaaa"},
                   sidecars={"checkpoint_1_steps": {"git_hash": "aaaa"}})
    sa.main([str(run)])
    out = capsys.readouterr().out
    assert "pin_history        : (absent" in out
    assert "not the only one" in out, "the absence must be labelled, not silently benign"


def test_json_mode_is_machine_readable(tmp_path, capsys):
    run = _run_dir(tmp_path, "run_j", meta={"git_hash": "aaaa"},
                   sidecars={"checkpoint_1_steps": {"git_hash": "bbbb"}})
    assert sa.main([str(run), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["n_mismatched"] == 1


def test_it_is_torch_free():
    """Model-free by construction: it must read a run whose architecture no longer loads.

    Asserted by IMPORTING it in a clean interpreter and looking at ``sys.modules`` — a source
    scan would be fooled by an import three modules deep (and by this file's own prose).
    """
    import subprocess
    import sys
    from utils.paths import src_root
    probe = (
        "import sys; import main.sidecar_audit; "
        "bad=[m for m in ('torch','stable_baselines3','sb3_contrib') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    env = {"PYTHONPATH": str(src_root()), "PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env=env, check=True).stdout.strip()
    assert out == "", f"main.sidecar_audit dragged in heavy modules: {out}"


# ------------------------------------------------------------------------------------
# num_timesteps — the step count, which used to be readable only by opening the .zip
# ------------------------------------------------------------------------------------

def test_it_reports_each_sidecars_step_count(tmp_path, capsys):
    """The tool is JSON-only by design (no torch, no zip), so before `num_timesteps` existed
    it could not say how far a run had trained. It reports the run's own and each sidecar's."""
    run = _run_dir(
        tmp_path, "run_steps",
        meta={"git_hash": "aaaa1111", "num_timesteps": 278_664_287},
        sidecars={
            "checkpoint_100_steps": {"git_hash": "aaaa1111", "num_timesteps": 100},
            "checkpoint_200_steps": {"git_hash": "ffff9999", "num_timesteps": 200},
        },
    )
    rec = sa.audit_run(str(run))
    assert rec["num_timesteps"] == 278_664_287
    assert sorted(s["num_timesteps"] for s in rec["sidecars"]) == [100, 200]

    assert sa.main([str(run), "-v"]) == 0
    out = capsys.readouterr().out
    assert "278,664,287" in out, "the run-level step count must be on the report"
    assert "steps=100" in out and "steps=200" in out


def test_a_legacy_run_shows_an_UNKNOWN_step_count_never_zero(tmp_path, capsys):
    """Every sidecar written before 2026-09-05 carries no `num_timesteps`. `?` says so; a 0
    would read as 'this checkpoint is at step zero'."""
    run = _run_dir(tmp_path, "run_legacy", meta={"git_hash": "aaaa1111"},
                   sidecars={"checkpoint_100_steps": {"git_hash": "aaaa1111", "lr": 3e-4}})
    rec = sa.audit_run(str(run))
    assert rec["num_timesteps"] is None and rec["sidecars"][0]["num_timesteps"] is None

    assert sa.main([str(run), "-v"]) == 0
    out = capsys.readouterr().out
    assert "num_timesteps      : ?" in out
    assert "steps=?" in out
    assert "steps=0" not in out
