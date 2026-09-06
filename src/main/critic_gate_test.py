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
from main.critic_gate_design import (FALSIFICATION_CLAUSE, G2_MAX_RELIABILITY, G3_MAX_ECE,
                                     OWNER_RULING_2026_09_06, RELATIVE_RULE)
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


def build_baseline(dirpath: str, *, resolution: float = 0.02, reliability: float = 0.001,
                   ece: float = 0.02, per_stratum: dict | None = None) -> str:
    """A committed-baseline-shaped artifact whose values are the bars under test.

    ``per_stratum`` overrides any of `resolution` / `reliability` / `ece` for one named stratum —
    the shape the REAL artifact has and the reason G2/G3 are per-stratum bars: on the committed
    baseline `pool` breaches §4.3's absolutes while `bot` does not.
    """
    os.makedirs(dirpath, exist_ok=True)
    strata = [{"kind": "all", "name": "all"}] + [{"kind": "class", "name": c}
                                                 for c in ("bot", "pool")]
    rows = []
    for st in strata:
        row = dict(st, resolution=resolution, reliability=reliability, ece=ece, skill=0.30)
        row.update((per_stratum or {}).get(st["name"], {}))
        rows.append(row)
    blocks = [{"step": 26_000_016, "bins": 10, "reweighted": True, "strata": rows}]
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


# ------------------------------------------------- G2/G3: the per-stratum RELATIVE bars (ruling)

def test_moving_the_artifacts_pool_reliability_moves_G2s_verdict(tree):
    """The sibling of the resolution test: G2's bar is READ per stratum, never baked in.

    The arm is unchanged between the two halves — only the committed artifact's `pool`
    reliability moves — so a verdict that moves proves the comparison is against the artifact and
    against the MATCHED stratum.
    """
    # (a) a pool baseline the arm is comfortably inside → PASS
    build_baseline(tree["baseline"], per_stratum={"pool": {"reliability": 0.05}})
    doc_a = os.path.join(tree["tmp"], "g2_pass.json")
    cg.main(_run(tree, "--json", doc_a))
    a = json.load(open(doc_a))
    assert a["calibration"]["verdict"]["G2"] is True

    # (b) the SAME arm against a pool baseline below it → FAIL, and only on `pool`
    build_baseline(tree["baseline"], per_stratum={"pool": {"reliability": 1e-9}})
    doc_b = os.path.join(tree["tmp"], "g2_fail.json")
    cg.main(_run(tree, "--json", doc_b))
    b = json.load(open(doc_b))
    assert b["calibration"]["verdict"]["G2"] is False
    by_name = {(c["step"], s["stratum"]): s
               for c in b["calibration"]["checkpoints"] for s in c["strata"]}
    assert by_name[(1_000_000, "pool")]["G2_reliability"] is False
    assert by_name[(1_000_000, "bot")]["G2_reliability"] is True          # matched stratum only
    assert by_name[(1_000_000, "pool")]["relative"]["G2"]["baseline"] == 1e-9
    assert b["verdict"]["verdict"] != "PASS"


def test_moving_the_artifacts_pool_ece_moves_G3s_verdict(tree):
    build_baseline(tree["baseline"], per_stratum={"pool": {"ece": 1e-9}})
    path = os.path.join(tree["tmp"], "g3_fail.json")
    cg.main(_run(tree, "--json", path))
    doc = json.load(open(path))
    assert doc["calibration"]["verdict"]["G3"] is False
    by_name = {(c["step"], s["stratum"]): s
               for c in doc["calibration"]["checkpoints"] for s in c["strata"]}
    assert by_name[(2_000_000, "pool")]["G3_ece"] is False
    assert by_name[(2_000_000, "bot")]["G3_ece"] is True


def test_the_absolute_4_3_numbers_are_ASPIRATIONAL_and_gate_nothing(tree):
    """§4.3's 0.005 / 0.05 are printed and never gated (owner ruling 2026-09-06).

    Driven with an unreachable absolute (0.0): under the OLD rule G2/G3 would both fail; under the
    ruling they pass on the relative bar while the report says the aspirational target is MISSED.
    """
    path = os.path.join(tree["tmp"], "aspirational.json")
    md_path = os.path.join(tree["tmp"], "aspirational.md")
    cg.main(_run(tree, "--max-reliability", "0.0", "--max-ece", "0.0",
                 "--json", path, "--md", md_path))
    doc = json.load(open(path))
    cv = doc["calibration"]["verdict"]
    assert (cv["G2"], cv["G3"]) == (True, True)
    rows = [s for c in doc["calibration"]["checkpoints"] for s in c["strata"] if s["gated"]]
    assert rows
    for s in rows:
        assert s["relative"]["G2"]["aspirational"] == 0.0
        assert s["relative"]["G2"]["aspirational_met"] is False
        assert s["relative"]["G3"]["aspirational_met"] is False
    # and the numbers ride in the report as targets, under a key that says so
    assert doc["_meta"]["thresholds"]["aspirational_only"]["G2_max_reliability"] == 0.0
    assert "aspirational" in open(md_path).read().lower()


def test_the_relative_rule_and_the_owner_ruling_are_printed_in_both_renderings(tree):
    path = os.path.join(tree["tmp"], "ruling.json")
    md_path = os.path.join(tree["tmp"], "ruling.md")
    cg.main(_run(tree, "--json", path, "--md", md_path))
    doc = json.load(open(path))
    md, txt = open(md_path).read(), cg.render_text(doc)
    assert doc["calibration"]["owner_ruling"] == OWNER_RULING_2026_09_06
    assert doc["calibration"]["relative_rule"] == RELATIVE_RULE
    for text in (md, txt):
        assert OWNER_RULING_2026_09_06 in text
        assert "NON-INFERIORITY" in text.upper()
    # the baseline value sits BESIDE the arm's, and a column names the deciding clause
    for s in (s for c in doc["calibration"]["checkpoints"] for s in c["strata"]):
        assert s["baseline_reliability"] == s["relative"]["G2"]["baseline"]
        assert s["baseline_ece"] == s["relative"]["G3"]["baseline"]
        assert s["G2_decided_by"] in (cg.RULE_BETTER, cg.RULE_NONINFERIOR, cg.RULE_WORSE,
                                      cg.RULE_NO_CI)
    assert cg.RULE_BETTER in md and cg.RULE_BETTER in txt


def test_a_generation_is_NEVER_reported_as_inferior_to_ITSELF(tree):
    """The soundness property the matched-checkpoint half exists for.

    A baseline artifact carrying the arm's OWN per-step values is a self-comparison: every delta
    must be exactly zero and every gated row must pass. Reduced-only, the 1M row would be judged
    against the 2M value and this can — and on the real committed artifact does — report FAIL.
    """
    # the arm's own values, per step, taken from a first pass
    first = os.path.join(tree["tmp"], "measure.json")
    cg.main(_run(tree, "--json", first))
    rows = {(c["step"], s["stratum"]): s
            for c in json.load(open(first))["calibration"]["checkpoints"] for s in c["strata"]}
    steps = sorted({k[0] for k in rows})
    # ... rebuilt as a two-step artifact whose every cell IS the arm's own measurement
    strata = [{"kind": "all", "name": "all"}] + [{"kind": "class", "name": c}
                                                 for c in ("bot", "pool")]
    blocks = [{"step": st, "bins": 10, "reweighted": True,
               "strata": [dict(x, resolution=rows[(st, x["name"])]["resolution"],
                               reliability=rows[(st, x["name"])]["reliability"],
                               ece=rows[(st, x["name"])]["ece"], skill=0.30) for x in strata]}
              for st in steps]
    with open(os.path.join(tree["baseline"], cg.BASELINE_ARTIFACT), "w") as fh:
        json.dump({"tool": "scaffolding_gauge", "meta": {"run_name": "SELF"},
                   "reliability": blocks}, fh)

    path = os.path.join(tree["tmp"], "self.json")
    cg.main(_run(tree, "--json", path))
    doc = json.load(open(path))
    cv = doc["calibration"]["verdict"]
    assert (cv["G2"], cv["G3"]) == (True, True)
    for s in (s for c in doc["calibration"]["checkpoints"] for s in c["strata"]):
        for gate in ("G2", "G3"):
            r = s["relative"][gate]
            assert r["baseline_matched_step"] is True
            assert r["delta"] == pytest.approx(0.0, abs=1e-12)
            assert (r["pass"], r["decided_by"]) == (True, cg.RULE_BETTER)
    assert cv["G1"] is False          # ... and it does not out-RESOLVE itself either


def test_a_step_the_artifact_does_not_carry_falls_back_to_the_reduction_and_says_so(tree):
    """The ordinary case — a new arm's steps do not coincide with the baseline's at all."""
    build_baseline(tree["baseline"])          # one block, at 26,000,016; the arm is at 1M / 2M
    path = os.path.join(tree["tmp"], "reduced.json")
    md_path = os.path.join(tree["tmp"], "reduced.md")
    cg.main(_run(tree, "--json", path, "--md", md_path))
    doc = json.load(open(path))
    for s in (s for c in doc["calibration"]["checkpoints"] for s in c["strata"]):
        r = s["relative"]["G2"]
        assert r["baseline_matched_step"] is False
        assert r["baseline_from_step"] == 26_000_016
        assert "reduced" in r["baseline_source"]
    assert "reduced" in open(md_path).read()


@pytest.mark.parametrize("point, ci, base, expect_pass, expect_rule", [
    (0.004, (0.002, 0.006), 0.005, True, cg.RULE_BETTER),        # better outright
    (0.005, (0.004, 0.006), 0.005, True, cg.RULE_BETTER),        # equal counts as no worse
    (0.006, (0.004, 0.008), 0.005, True, cg.RULE_NONINFERIOR),   # above, but CI covers the base
    (0.006, (0.0055, 0.008), 0.005, False, cg.RULE_WORSE),       # the WHOLE CI sits above
    (0.006, (float("nan"), float("nan")), 0.005, False, cg.RULE_NO_CI),   # no interval to lean on
])
def test_the_non_inferiority_clauses_each_decide_their_own_case(point, ci, base, expect_pass,
                                                                expect_rule):
    """The rule is three clauses, and each one is the reason for exactly one shape of row."""
    ok, rule = cg.relative_verdict(point, ci, base)
    assert (ok, rule) == (expect_pass, expect_rule)


def test_a_higher_arm_inside_its_own_CI_is_never_reported_as_BETTER(tree):
    """Non-inferiority is not a direction claim — the label must say which clause carried it."""
    ok, rule = cg.relative_verdict(0.010, (0.001, 0.020), 0.005)
    assert ok is True and rule == cg.RULE_NONINFERIOR and rule != cg.RULE_BETTER


def test_the_committed_baseline_really_does_breach_4_3s_absolutes_on_pool():
    """The ruling's PREMISE, pinned to the artifact rather than to prose.

    If a future re-measurement brings `pool` under the §4.3 absolutes, this fails and the ruling's
    justification should be re-read — the finding is a measurement, not a belief.
    """
    baseline = cg.load_baseline(cg.DEFAULT_BASELINE_DIR)
    pool = baseline["per_stratum"]["pool"]
    assert pool, "the committed baseline must carry a `pool` stratum"
    assert any(float(r["reliability"]) > G2_MAX_RELIABILITY for r in pool)
    assert any(float(r["ece"]) > G3_MAX_ECE for r in pool)
    # ... while `bot` does not, which is precisely why a POOLED reading hid this.
    assert all(float(r["ece"]) <= G3_MAX_ECE for r in baseline["per_stratum"]["bot"])


def test_the_owner_ruling_is_recorded_in_the_design_doc(tree):
    """The tool prints a ruling; §4.3 must carry the same dated paragraph, or the two drift."""
    text = " ".join(open(repo_path(cg.DESIGN_DOC)).read().split())
    assert "OWNER RULING 2026-09-06" in text
    assert "PER-STRATUM RELATIVE" in text.upper()
    for phrase in ("aspirational", "non-inferiority"):
        assert phrase in text.lower()


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
