"""Regression tests from the 2026-10-26 hands-on health audit of the prober against TODAY's runs.

Every case here is a defect the audit hit on a current run (the win-prob critic, the 2845-dim
re-homed observation, core traces) and that the suites written for earlier eras could not see,
because their fixtures were shaped like those eras (a shaped critic, a layout that still carried
the matchup / incoming-damage / turn-history blocks). Each test fails with its fix reverted.
"""

import json
import os

import numpy as np
import pytest

from main.prober.engine import attribute_turning_point
from main.prober.model import ObsOffsets
from main.prober.session import ProbeSession
from main.prober.session_test import (_FakeFeatureModel, _FakeModel, _build_probe_run, _build_run,
                                      _build_triage_run, _loss_inv, _write_battle)


def _winprob_run(tmp_path):
    """The `_build_run` fixture, declared a win-prob-critic run (`model_config.json: critic`)."""
    run, summ = _build_run(tmp_path)
    with open(f"{run}/model_config.json", "w") as f:
        json.dump({"critic": "winprob"}, f)
    return run, summ


def test_td_phrase_reads_a_win_prob_swing_as_a_swing(tmp_path):
    """Under the P(win) critic a turn's TD residual lives in [-1, 1]; the 0.5 / 3.0 bars of the
    phrase were written for a shaped return (span 60), so a -0.40 P(win) collapse read "about what
    the critic expected". The bars are fractions of the critic's span now."""
    run, summ = _winprob_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    sess._td = lambda reward, v, v_next: -0.40          # the swing, whatever the fixture's V is
    d = sess.analyze(summ, 0)
    assert d["critic_currency"]["span"] == 1.0
    assert d["value"]["td_residual"] == -0.40
    assert d["value"]["td_phrase"] == "much worse than the critic expected"
    sess._td = lambda reward, v, v_next: -0.004         # noise on the same scale stays noise
    assert sess.analyze(summ, 0)["value"]["td_phrase"] == "about what the critic expected"


def test_td_phrase_on_a_shaped_run_keeps_its_old_bars(tmp_path):
    run, summ = _build_run(tmp_path)                    # no model_config.json: shaped (span 60)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    sess._td = lambda reward, v, v_next: -0.40
    assert sess.analyze(summ, 0)["value"]["td_phrase"] == "about what the critic expected"
    sess._td = lambda reward, v, v_next: -4.0
    assert sess.analyze(summ, 0)["value"]["td_phrase"] == "much worse than the critic expected"


# --- triage on a run whose observation has no incoming-damage belief (every current run) -------

def test_a_death_with_no_belief_is_not_filed_as_a_partial_belief():
    """`attrition_death` says "the belief only PARTLY fired". With `active_pko` unknown (the block is
    deleted from the observation) the belief did not fire at all, and 44 % of rb_x35_n1p_s3501's
    losses were filed there. The unread buckets say only what the outcome shows."""
    healthy = attribute_turning_point({"faint": True, "our_hp": 0.9, "active_pko": None})
    chipped = attribute_turning_point({"faint": True, "our_hp": 0.3, "active_pko": None})
    assert healthy["category"] == "death_healthy_unread"
    assert chipped["category"] == "death_chipped_unread"
    # a KNOWN mid belief still means attrition: the original bucket keeps its meaning
    assert attribute_turning_point(
        {"faint": True, "our_hp": 0.45, "active_pko": 0.4, "n_healthy_bench": 2})["category"] == "attrition_death"


def test_triage_says_when_no_loss_carries_the_belief_and_prints_no_none_examples(tmp_path):
    run = _build_triage_run(tmp_path)                    # four losses, no belief block in any obs
    out = ProbeSession(run).triage()
    assert out["belief_coverage"] == {"n": 0, "d": 4}
    assert any("incoming-damage belief" in c and "cannot fire" in c for c in out["caveats"])
    assert "attrition_death" not in {c["category"] for c in out["categories"]}
    examples = [e for c in out["categories"] for e in c["examples"]]
    assert examples and not any("None" in e for e in examples), examples


def test_triage_on_a_win_prob_run_reads_v_as_p_win_not_as_a_missing_head(tmp_path):
    """A core trace stores NaN in `win_probs`, so `wp_coverage` was 0 on every current run and the
    triage page told the reader "this run recorded no win-prob head ... V has a -6.5 offset". The
    win-prob critic's V IS P(win): the split reads it and says so."""
    run = tmp_path / "run"
    _write_battle(run, "heuristic", "loss_001",
                  [_loss_inv(1, "thunderbolt", hp=0.8, hp_delta=-0.1)], [0.8, 0.1])   # winning, then not
    _write_battle(run, "heuristic", "loss_002",
                  [_loss_inv(1, "protect", hp=0.8, hp_delta=-0.1)], [0.2, 0.05])       # already behind
    (run / "model_config.json").write_text(json.dumps({"critic": "winprob"}))
    out = ProbeSession(str(run)).triage()
    cats = {c["category"] for c in out["categories"]}
    assert cats == {"critic_blindspot", "positional_grind"}
    assert out["winning_split"]["wp_coverage"] == 1.0
    split = next(c for c in out["caveats"] if c.startswith("grind-vs-throw"))
    assert "IS the win-prob head" in split and "OVER-counted" not in split


def test_triage_on_a_shaped_run_keeps_the_v_fallback_wording(tmp_path):
    run = _build_triage_run(tmp_path)                    # no model_config.json: shaped
    out = ProbeSession(run).triage()
    assert out["winning_split"]["wp_coverage"] == 0.0
    assert any("OVER-counted" in c for c in out["caveats"])


# --- loops: a core trace's NaN win_probs ---------------------------------------------------------

_LOOP_LINES = ["|player|p1|RLEval|", "|player|p2|Sentinel|", "|start|",
               "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Snorlax|Snorlax|100/100",
               "|turn|1", "|switch|p2a: Salamence|Salamence, F|100/100",
               "|move|p1a: Zapdos|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence", "|upkeep",
               "|turn|2", "|move|p1a: Zapdos|Thunderbolt|p2a: Salamence", "|upkeep"]


def _loops_run(tmp_path, *, winprob):
    """The `_build_run` battle with the npz a CORE trace writes: `win_probs` all NaN, `values` = V."""
    run, summ = _build_run(tmp_path)
    npz = os.path.join(os.path.dirname(summ), "win_001_states.npz")
    np.savez(npz, obs=np.zeros((2, 256), dtype=np.float32), has_state=np.ones(2, dtype=np.int8),
             values=np.array([0.8, 0.3], dtype=np.float32), win_probs=np.full(2, np.nan, np.float32))
    if winprob:
        with open(f"{run}/model_config.json", "w") as f:
            json.dump({"critic": "winprob"}, f)
    sess = ProbeSession(run)
    sess._protocol_lines = lambda b: list(_LOOP_LINES)
    return sess


def test_loops_on_a_win_prob_run_reads_the_p_win_swing_from_v(tmp_path):
    """`critic_delta.median_delta_win_prob` was NaN on every current run (a core trace stores NaN in
    `win_probs`), so the one comparison with the gen-15 baseline's -0.096 was unreadable. V IS P(win)."""
    out = _loops_run(tmp_path, winprob=True).loops()
    assert out["aggregate"]["coverage"]["n_matched"] == 1
    bait = out["aggregate"]["critic_delta"]["other_bait"]
    assert bait["median_delta_win_prob"] == pytest.approx(-0.5, abs=1e-6)
    assert bait["median_delta_v"] == pytest.approx(-0.5, abs=1e-6)


def test_loops_never_emits_a_nan_even_where_there_is_no_win_prob(tmp_path):
    """A shaped run with a NaN `win_probs` array has no P(win) series: the median is absent (None),
    not NaN — a NaN poisons the median and is not JSON."""
    out = _loops_run(tmp_path, winprob=False).loops()
    bait = out["aggregate"]["critic_delta"]["other_bait"]
    assert bait["median_delta_win_prob"] is None
    assert bait["median_delta_v"] == pytest.approx(-0.5, abs=1e-6)
    json.dumps(out, allow_nan=False)                    # raises on any NaN anywhere in the result


# --- the CLI: strict JSON and an honest exit code ------------------------------------------------

def _run_cli(monkeypatch, capsys, argv):
    import main.prober.query as q
    monkeypatch.setattr("sys.argv", ["query", *argv])
    code = None
    try:
        q.main()
    except SystemExit as e:                             # noqa: PERF203 — the exit code IS the datum
        code = e.code
    return code, capsys.readouterr().out


def test_the_cli_prints_strict_json_and_maps_nan_to_null(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ProbeSession, "battles", lambda self, **kw: [
        {"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": float("-inf")}}])
    code, out = _run_cli(monkeypatch, capsys, ["list", str(tmp_path)])
    assert code is None
    assert json.loads(out, parse_constant=lambda tok: pytest.fail(f"non-JSON token {tok}")) == [
        {"a": None, "b": [None, 1.5], "c": {"d": None}}]


def test_the_cli_exits_1_when_a_session_method_returns_its_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ProbeSession, "history_saliency",
                        lambda self, **kw: {"error": "no decisions", "step": 3})
    code, out = _run_cli(monkeypatch, capsys, ["history-saliency", str(tmp_path)])
    assert code == 1 and json.loads(out)["error"] == "no decisions"


# --- history-saliency: refuse from the layout, before any gradient --------------------------------

class _NoHistoryModel(_FakeModel):
    """A current-layout model: no turn-history block. Any gradient taken is the bug."""
    offsets = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5)
    grads = 0

    def logit_grad(self, obs, mask, idx):
        type(self).grads += 1
        return np.zeros(len(obs))

    def value_grad(self, obs, mask):
        type(self).grads += 1
        return np.zeros(len(obs))


def test_history_saliency_refuses_at_once_on_a_layout_with_no_turn_history(tmp_path):
    """The loop `continue`d on a missing block without counting the decision, so on a current run it
    took two backward passes over EVERY decision of the step (15 min on rb_x35_n1p_s3501) to return
    "no decisions with a turn-history block"."""
    run, _ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _NoHistoryModel())
    _NoHistoryModel.grads = 0
    out = sess.history_saliency()
    assert out["kind"] == "no_turn_history_block" and "error" in out
    assert _NoHistoryModel.grads == 0, "a gradient was computed before the layout was consulted"


# --- probe: say when the provided feature / the group split is absent -----------------------------

def test_a_probe_names_the_missing_baseline_instead_of_returning_a_bare_null(tmp_path):
    """`provided_feature_baseline: null` and a lone `unknown` group are the belief features
    (active_pko / active_exp / active_outspeed) missing from the current observation; the result
    read like a measurement. Now it carries a `warnings` list saying what it could not do."""
    run = _build_probe_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda p: _FakeFeatureModel())
    out = sess.probe("is_faster", max_decisions=200)          # base-speed groups exist, baseline does not
    assert out["provided_feature_baseline"] is None
    assert any("active_outspeed" in w and "absent" in w for w in out["warnings"]), out["warnings"]
    assert not any("`unknown` group" in w for w in out["warnings"])
    out = sess.probe("damage_taken", max_decisions=200)       # grouped by the (absent) belief
    assert set(out["representation_probe"]["by_group"]) == {"unknown"}
    assert any("`unknown` group" in w for w in out["warnings"]), out["warnings"]


# --- the per-battle CLI and the web's short ids ---------------------------------------------------

def test_a_short_id_given_to_a_per_battle_command_names_the_way_out():
    """`list` / `scan` print both `id` and `short_id`; the per-battle commands take the first. Handed
    the second, the CLI said only "no trace found" — and that is the id the web's URLs carry."""
    short = "step_2000000/staller/win_001"
    sess = ProbeSession(short)                          # the CLI builds its session FROM the battle id
    with pytest.raises(FileNotFoundError) as e:
        sess.battle_overview(short)
    assert "`id` field" in str(e.value) and "*_summary.json" in str(e.value)
