"""`python -m main.critic_gate` — the composed pre-registered read.

The subject is a SYNTHETIC run tree: a ladder, a metadata block, `eval_results.jsonl`, an
`eval_manifest.json` with a selection record, per-battle trace summaries and `states.npz` files
carrying a win-prob column whose calibration this suite CHOOSES. Choosing it is the point — a test
that reads a real run can only assert "it ran", whereas here the arm is built to clear or to miss
G1 and the tool's verdict is checked against a known answer.

Nothing here writes into `models/`, plays a battle, or loads a checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from main import critic_gate as cg
from main.critic_gate_design import FALSIFICATION_CLAUSE
from utils.paths import repo_path, src_root

# ---------------------------------------------------------------------------------- the fixture

_OPPONENTS = ("heuristic", "aggressive", "sentinel_0", "sentinel_1")
#: The true per-opponent win rates the CYCLE recorded (the population the reweighting targets).
_TRUE_WR = {"heuristic": 0.90, "aggressive": 0.90, "sentinel_0": 0.70, "sentinel_1": 0.70}
#: What the CAPTURE QUOTA kept — deliberately loss-enriched, like the real recorder's.
_CAPTURED_WR = 0.5


def _write_zip(path: str) -> None:
    """A minimal file that `resolve_model_ref` will accept as a checkpoint (never opened)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data", "{}")


def _write_trace(step_dir: str, opponent: str, outcome: str, idx: int, *, p: float, turns: int,
                 n_states: int = 6, result: str | None = None) -> None:
    """One captured battle: a `_summary.json` (turns + result) and a `_states.npz` (the head)."""
    d = os.path.join(step_dir, opponent)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{outcome}_{idx:03d}")
    with open(stem + "_summary.json", "w") as fh:
        json.dump({"meta": {"step": 1, "battle_id": f"b{idx}", "turns": turns,
                            "result": result or outcome.upper(), "invocations": n_states}}, fh)
    np.savez(stem + "_states.npz",
             values=np.full(n_states, 0.5, dtype=np.float32),
             win_probs=np.full(n_states, p, dtype=np.float32),
             has_state=np.ones(n_states, dtype=bool))


def build_run(root: str, name: str, *, steps=(1_000_000, 2_000_000), sharpness: float = 0.49,
              ladder_elo=(1900.0, 1950.0), converged: bool = True, finished: bool = True,
              stall_turns: int | None = None, ep_len_bots: float = 18.0,
              ep_len_pool: float = 29.0, n_battles_per_opp: int = 8) -> str:
    """A run tree complete enough for every section of the gate.

    ``sharpness`` sets how far the head's forecast moves off the base rate on each outcome, which
    is what MOVES RESOLUTION: at 0.49 the head separates wins from losses almost perfectly, at 0.0
    it is a constant and resolution is ~0. That is the dial G1 is tested on.
    """
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    _write_zip(os.path.join(run_dir, "checkpoints", f"checkpoint_{max(steps)}_steps.zip"))
    with open(os.path.join(run_dir, "model_config.json"), "w") as fh:
        json.dump({"version": 1}, fh)
    if finished:
        _write_zip(os.path.join(run_dir, "final_model.zip"))

    # ---- the ladder
    os.makedirs(os.path.join(run_dir, "snapshot_ladder"), exist_ok=True)
    with open(os.path.join(run_dir, "snapshot_ladder", "ladder.json"), "w") as fh:
        json.dump({"version": 1, "base": 1500.0, "anchored_to_bots": True,
                   "converged": converged,
                   "ratings": {str(s): e for s, e in zip(steps, ladder_elo)},
                   "se": {str(s): 10.0 for s in steps},
                   "fit_quality": {"mean_abs_err": 0.03}}, fh)

    # ---- the eval record + metadata (episode length lives here, not in the jsonl)
    hist = {}
    with open(os.path.join(run_dir, "eval_results.jsonl"), "w") as fh:
        for s in steps:
            fh.write(json.dumps({
                "step": s, "n_games": 100,
                "bots": {o: _TRUE_WR[o] for o in _OPPONENTS if not o.startswith("sentinel")},
                "sentinels": [{"step": s, "win_rate": _TRUE_WR[o]}
                              for o in _OPPONENTS if o.startswith("sentinel")],
            }) + "\n")
            hist[f"checkpoint_{s}_steps.zip"] = {"latest_eval": {
                "step": s, "win_rate_vs_bots": 0.9, "mean_ep_len_vs_bots": ep_len_bots,
                "pool": {"win_rate": 0.7, "mean_ep_len": ep_len_pool}}}
    with open(os.path.join(run_dir, "metadata.json"), "w") as fh:
        json.dump({"git_hash": "deadbeef", "num_timesteps": max(steps),
                   "latest_eval": {"step": max(steps), "win_rate_vs_bots": 0.9,
                                   "mean_ep_len_vs_bots": ep_len_bots,
                                   "pool": {"win_rate": 0.7, "mean_ep_len": ep_len_pool}},
                   "snapshot_history": hist}, fh)

    # ---- the traces
    for s in steps:
        step_dir = os.path.join(run_dir, "eval_traces", f"step_{s}")
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "eval_manifest.json"), "w") as fh:
            json.dump({"step": s, "opponents": list(_OPPONENTS), "n_games": 100}, fh)
        for opp in _OPPONENTS:
            n_win = int(round(n_battles_per_opp * _CAPTURED_WR))
            for i in range(n_battles_per_opp):
                win = i < n_win
                # A sharp head forecasts high on the wins it saw and low on the losses.
                p = 0.5 + sharpness if win else 0.5 - sharpness
                turns = stall_turns if (stall_turns and i == 0) else 20
                _write_trace(step_dir, opp, "win" if win else "loss", i, p=p, turns=turns)
    return run_dir


def build_baseline(dirpath: str, *, resolution: float = 0.02) -> str:
    """A committed-baseline-shaped artifact whose `resolution` is the bar under test."""
    os.makedirs(dirpath, exist_ok=True)
    strata = [{"kind": "all", "name": "all"}] + [{"kind": "class", "name": c}
                                                 for c in ("bot", "pool")]
    blocks = [{"step": 26_000_016, "bins": 10, "reweighted": True,
               "strata": [dict(s, resolution=resolution, reliability=0.001, ece=0.02,
                               skill=0.30) for s in strata]}]
    path = os.path.join(dirpath, cg.BASELINE_ARTIFACT)
    with open(path, "w") as fh:
        json.dump({"tool": "scaffolding_gauge", "meta": {"run_name": "SYNTH_BASELINE"},
                   "reliability": blocks}, fh)
    return dirpath


@pytest.fixture()
def tree(tmp_path):
    """arm + parent + one continuation control + a baseline artifact."""
    root = str(tmp_path / "models")
    arm = build_run(root, "ARM", sharpness=0.49)
    parent = build_run(root, "PARENT", sharpness=0.05, ladder_elo=(1850.0, 1880.0))
    control = build_run(root, "CONTROL", sharpness=0.05, ladder_elo=(1860.0, 1890.0))
    baseline = build_baseline(str(tmp_path / "baseline"))
    return {"root": root, "arm": arm, "parent": parent, "control": control,
            "baseline": baseline, "tmp": str(tmp_path)}


def _run(tree, *extra, meter=False):
    argv = [tree["arm"], "--parent", tree["parent"], "--baseline-dir", tree["baseline"],
            "--boot", "60", "--quiet"]
    if not meter:
        argv.append("--skip-meter")
    return list(argv) + list(extra)


# ---------------------------------------------------------------------------------- the report

def test_the_report_has_every_section(tree, tmp_path, capsys):
    out_json, out_md = str(tmp_path / "g.json"), str(tmp_path / "g.md")
    rc = cg.main(_run(tree, "--json", out_json, "--md", out_md))
    doc = json.load(open(out_json))
    md = open(out_md).read()

    assert set(doc) >= {"ladder", "calibration", "kill", "verdict", "_meta", "not_runnable"}
    # (1) the ladder, at matched snapshot COUNT
    assert doc["ladder"]["at_snapshots"] == 2
    assert doc["ladder"]["delta_elo"] == pytest.approx(1950.0 - 1880.0)
    assert doc["ladder"]["rating_final"] is True
    # (2) calibration: bot and pool are gated, `all` is not
    strata = {(c["step"], s["stratum"]): s
              for c in doc["calibration"]["checkpoints"] for s in c["strata"]}
    assert {k[1] for k in strata} == {"all", "bot", "pool"}
    assert strata[(1_000_000, "all")]["gated"] is False
    assert strata[(1_000_000, "bot")]["gated"] is True
    # (3) the kill condition read both halves
    assert doc["kill"]["cycles"][0]["ep_len_bots"] == 18.0
    assert doc["kill"]["cycles"][0]["stall_rate_captured"] == 0.0
    assert doc["kill"]["kill"] is False
    # (4) the meter was skipped, and the report SAYS so rather than omitting the section
    assert doc["untaught_meter"] is None
    assert "skipped" in md

    for heading in ("## 1. Anchored ladder", "## 2. Calibration gate", "## 3. G7",
                    "## 4. Untaught meter", "## Not runnable here"):
        assert heading in md
    # every threshold used, and every input path, ride in the JSON
    assert doc["_meta"]["thresholds"]["G7_max_stall_rate"] == cg.DEFAULT_MAX_STALL_RATE
    assert doc["_meta"]["run"]["resolved_file"].endswith(".zip")
    assert doc["_meta"]["run"]["resolution_rung"] in ("highest_checkpoint", "latest_txt",
                                                      "final_model")
    assert rc in (0, 1)   # the verdict decides; the report is complete either way


def test_a_sharp_head_passes_G1_against_a_low_bar(tree):
    doc_path = os.path.join(tree["tmp"], "pass.json")
    cg.main(_run(tree, "--json", doc_path))
    doc = json.load(open(doc_path))
    assert doc["calibration"]["verdict"]["G1"] is True
    assert doc["verdict"]["verdict"] == "PASS"
    assert doc["verdict"]["falsified"] is False


def test_resolution_is_measured_against_the_artifact_not_a_hardcoded_number(tree):
    """Move the committed bar and the verdict must move with it. G1's bar is READ, never baked in."""
    build_baseline(tree["baseline"], resolution=0.99)          # an unreachable bar
    doc_path = os.path.join(tree["tmp"], "hi.json")
    cg.main(_run(tree, "--json", doc_path))
    doc = json.load(open(doc_path))
    assert doc["calibration"]["verdict"]["G1"] is False
    for c in doc["calibration"]["checkpoints"]:
        for s in c["strata"]:
            assert s["baseline_resolution"] == 0.99


# ---------------------------------------------------------------------- the falsification clause

def test_the_falsification_clause_is_the_designs_own_words(tree):
    """VERBATIM, asserted against the design file — a paraphrase would put words in its mouth."""
    text = open(repo_path(cg.DESIGN_DOC)).read()
    normalized = " ".join(text.split())
    wanted = " ".join(FALSIFICATION_CLAUSE.replace(
        "What would falsify the design, stated before the data:",
        "**What would falsify the design, stated before the data:**").split())
    assert wanted in normalized


def test_G1_flat_with_G2_to_G4_passing_reports_the_clause_as_the_verdict(tree):
    """The design's own falsification path: a calibrated head that does not SEPARATE any better."""
    build_baseline(tree["baseline"], resolution=0.99)     # G1 cannot clear it
    doc_path = os.path.join(tree["tmp"], "falsified.json")
    md_path = os.path.join(tree["tmp"], "falsified.md")
    rc = cg.main(_run(tree, "--json", doc_path, "--md", md_path))
    doc = json.load(open(doc_path))
    cv = doc["calibration"]["verdict"]
    assert (cv["G1"], cv["G2"], cv["G3"], cv["G4"]) == (False, True, True, True)
    assert doc["verdict"]["verdict"].startswith("FALSIFIED")
    assert doc["verdict"]["why"] == FALSIFICATION_CLAUSE
    assert doc["verdict"]["falsified"] is True
    assert rc != 0                                       # a falsification is never a success exit
    assert FALSIFICATION_CLAUSE in open(md_path).read()


def test_a_G7_breach_outranks_the_falsification_clause(tree):
    """A KILL is a KILL. The clause is for a run that is otherwise fine, not one running the clock."""
    build_baseline(tree["baseline"], resolution=0.99)
    root = os.path.join(tree["tmp"], "stalled")
    arm = build_run(root, "ARM", sharpness=0.49, stall_turns=250)
    doc_path = os.path.join(tree["tmp"], "kill.json")
    cg.main([arm, "--parent", tree["parent"], "--baseline-dir", tree["baseline"],
             "--boot", "60", "--quiet", "--skip-meter", "--json", doc_path])
    doc = json.load(open(doc_path))
    assert doc["kill"]["kill"] is True
    assert doc["verdict"]["verdict"] == "KILL"
    assert "stall rate" in doc["verdict"]["why"]


def test_a_long_episode_against_the_era_is_a_kill_too(tree):
    root = os.path.join(tree["tmp"], "slow")
    arm = build_run(root, "ARM", sharpness=0.49, ep_len_bots=40.0)
    doc_path = os.path.join(tree["tmp"], "slow.json")
    cg.main([arm, "--parent", tree["parent"], "--baseline-dir", tree["baseline"],
             "--boot", "60", "--quiet", "--skip-meter", "--json", doc_path])
    doc = json.load(open(doc_path))
    assert doc["kill"]["kill"] is True
    assert any("ep_len_bots" in b for c in doc["kill"]["cycles"] for b in c["breaches"])


# ------------------------------------------------------------------------------- the refusals

def _refusal(argv) -> str:
    with pytest.raises(SystemExit) as exc:
        cg.main(argv)
    assert exc.value.code == 2
    return str(exc.value)


def test_a_missing_ladder_refuses_naming_the_path(tree):
    os.remove(os.path.join(tree["arm"], "snapshot_ladder", "ladder.json"))
    msg = _refusal(_run(tree))
    assert "REFUSAL" in msg and "snapshot_ladder" in msg and "--backfill" in msg


def test_an_unconverged_ladder_refuses_naming_the_key(tree):
    path = os.path.join(tree["arm"], "snapshot_ladder", "ladder.json")
    doc = json.load(open(path))
    doc["converged"] = False
    json.dump(doc, open(path, "w"))
    assert "'converged' is false" in _refusal(_run(tree))


def test_a_ladder_missing_a_key_refuses_naming_it(tree):
    path = os.path.join(tree["parent"], "snapshot_ladder", "ladder.json")
    doc = json.load(open(path))
    del doc["se"]
    json.dump(doc, open(path, "w"))
    assert "'se'" in _refusal(_run(tree))


def test_an_empty_ratings_block_refuses(tree):
    path = os.path.join(tree["arm"], "snapshot_ladder", "ladder.json")
    doc = json.load(open(path))
    doc["ratings"] = {}
    json.dump(doc, open(path, "w"))
    assert "EMPTY" in _refusal(_run(tree))


def test_at_snapshots_beyond_the_matched_count_refuses(tree):
    msg = _refusal(_run(tree, "--at-snapshots", "5"))
    assert "matched count can be at most 2" in msg


def test_a_missing_baseline_artifact_refuses(tree, tmp_path):
    msg = _refusal([tree["arm"], "--parent", tree["parent"], "--skip-meter", "--quiet",
                    "--baseline-dir", str(tmp_path / "nope")])
    assert "baseline artifact" in msg and cg.BASELINE_ARTIFACT in msg


def test_a_RAW_baseline_artifact_refuses_rather_than_becoming_the_bar(tree):
    """§4.2: the un-reweighted table inverts the verdict, so it is not a bar at all."""
    path = os.path.join(tree["baseline"], cg.BASELINE_ARTIFACT)
    doc = json.load(open(path))
    doc["reliability"][0]["reweighted"] = False
    json.dump(doc, open(path, "w"))
    msg = _refusal(_run(tree))
    assert "NOT selection-reweighted" in msg


def test_a_baseline_missing_the_matched_stratum_refuses(tree):
    path = os.path.join(tree["baseline"], cg.BASELINE_ARTIFACT)
    doc = json.load(open(path))
    doc["reliability"][0]["strata"] = [s for s in doc["reliability"][0]["strata"]
                                       if s["name"] != "pool"]
    json.dump(doc, open(path, "w"))
    assert "'pool'" in _refusal(_run(tree))


def test_an_unresolvable_reweighting_surfaces_the_gauges_own_refusal(tree):
    """The gauge REFUSES rather than falling back to unweighted; the gate must not soften that."""
    os.remove(os.path.join(tree["arm"], "eval_results.jsonl"))
    for name in sorted(os.listdir(os.path.join(tree["arm"], "eval_traces"))):
        os.remove(os.path.join(tree["arm"], "eval_traces", name, "eval_manifest.json"))
    msg = _refusal(_run(tree))
    assert "reliability-reweight" in msg and "loss-enriched" in msg


# ------------------------------------------------------------------------------------- --check

def test_check_resolves_everything_and_computes_nothing(tree, capsys, monkeypatch):
    def _boom(*_a, **_k):                       # any computation here is a bug
        raise AssertionError("--check must not compute the calibration section")

    monkeypatch.setattr(cg, "calibration_section", _boom)
    monkeypatch.setattr(cg, "kill_section", _boom)
    rc = cg.main([tree["arm"], "--parent", tree["parent"], "--control", tree["control"],
                  "--baseline-dir", tree["baseline"], "--check"])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "every input resolved — OK" in printed
    for expect in ("run ", "parent ", "control ", "ladder ", "baseline ", "traces ",
                   "reweight ", "ep_len ", "meter "):
        assert expect in printed


def test_check_fails_non_zero_and_names_every_miss(tree, capsys):
    os.remove(os.path.join(tree["arm"], "snapshot_ladder", "ladder.json"))
    os.remove(os.path.join(tree["arm"], "metadata.json"))
    rc = cg.main([tree["arm"], "--parent", tree["parent"], "--baseline-dir", tree["baseline"],
                  "--check"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "--check FAILED" in err
    assert "ladder.json" in err          # both misses named, not just the first
    assert "episode length" in err


def test_check_reports_a_missing_continuation_control(tree, capsys):
    cg.main([tree["arm"], "--parent", tree["parent"], "--baseline-dir", tree["baseline"],
             "--check"])
    assert "OVERSTATE" in capsys.readouterr().out


# ------------------------------------------------------------------------- the meter invocation

def test_the_meter_argv_carries_the_baseline_and_every_control(tree):
    run = cg._resolve_ref(tree["arm"], what="run")
    parent = cg._resolve_ref(tree["parent"], what="--parent")
    control = cg._resolve_ref(tree["control"], what="--control")
    argv = cg.meter_argv(run, parent, [control], games_per_team=25, rows=[], from_rows=False,
                         workers=2, json_out="/tmp/x.json", check=True, dry_run=False)
    assert argv[1:3] == ["-m", "main.untaught_meter"]
    assert f"ARM={tree['arm']}" in argv
    assert argv[argv.index("--baseline") + 1] == tree["parent"]
    assert argv[argv.index("--control") + 1] == tree["control"]
    assert argv[argv.index("--games-per-team") + 1] == "25"
    assert "--check" in argv and "--dry-run" not in argv


def test_from_rows_passes_the_artifacts_through_and_plays_nothing(tree):
    run = cg._resolve_ref(tree["arm"], what="run")
    parent = cg._resolve_ref(tree["parent"], what="--parent")
    argv = cg.meter_argv(run, parent, [], games_per_team=200,
                         rows=["A=rows_a.json", "B=rows_b.json"], from_rows=True, workers=1,
                         json_out=None, check=False, dry_run=False)
    assert "--from-rows" in argv and "A=rows_a.json" in argv
    assert "--games-per-team" not in argv          # a rows read has no battle count
    assert not any(a.startswith("ARM=") for a in argv)


def test_the_no_control_case_is_stated_in_the_report_not_silently_omitted(tree):
    run = cg._resolve_ref(tree["arm"], what="run")
    parent = cg._resolve_ref(tree["parent"], what="--parent")
    argv = cg.meter_argv(run, parent, [], games_per_team=1, rows=[], from_rows=False, workers=1,
                         json_out=None, check=True, dry_run=False)
    section = cg.meter_section(argv, None)
    assert "OVERSTATES" in section["control_note"]


@pytest.mark.integration
def test_the_cli_is_reachable_as_a_module(tree):
    """`python -m main.critic_gate --help` — the entry point, in a real subprocess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [env.get("PYTHONPATH", ""), str(src_root())] if p)
    proc = subprocess.run([sys.executable, "-m", "main.critic_gate", "--help"],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0
    assert "PRE-REGISTERED READ" in proc.stdout
    assert "--control" in proc.stdout and "--at-snapshots" in proc.stdout
