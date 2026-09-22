"""gen3_anchor_regime_split_v1 — the REGIME question and the PROCESS question are two fields.

THE DEFECT THIS STANDS FOR. `regime_verified` was one composite AND over "did the regime take
effect?" and "did the peer exit cleanly?". Metamon raises a `RecursionError` in its post-game
teardown when our side forfeits at turn 250 — AFTER its last decision, after every game has been
played and recorded — so the peer exits nonzero and the composite went FALSE on **31 of the 84
sub-cells** of the 2026-09-20 continuation campaign while every one of those cells'
`argmax_match_rate` values was **1.0000**. A verification flag that reads FALSE on a clean regime
is a flag a reader learns to ignore, which is worse than not having one.

So: `regime_verified_decisions` (both halves, per-decision, over ≥1 decision) is what the SOP
reads; `peer_clean` (every peer rc == 0) is the separate operational fact; `regime_verified`
survives one release as their AND, carrying a deprecation note.
"""
from __future__ import annotations

import json

import pytest

from main.anchors.peers import REGIME_VERIFIED_DEPRECATION, FoulPlayPeer, MetamonPeer
from main.anchors.results import REQUIRED_ROW_FIELDS, GameRow, render, summarize
from main.anchors import results as results_mod
from main.anchors import runner as runner_mod
from main.anchors.runner import finalize_regime_fields

from main.anchors.peers_test import _foulplay_plan, _metamon_plan


@pytest.fixture
def cfg():
    """The same `AnchorsConfig` `peers_test` builds. Defined here rather than star-imported: a
    fixture imported by name shadows the parameter of every test that takes it, which pyflakes
    reads as a redefinition and the ruff gate refuses."""
    from main.anchors.config import load_config

    return load_config()


# ------------------------------------------------------------------ the peer-report read side
def _metamon_report(tmp_path, cfg_, **blob):
    plan = _metamon_plan(cfg_, tmp_path)
    plan.report_path.write_text(json.dumps(blob))
    return MetamonPeer.read_report(plan)


def test_a_clean_regime_with_a_dirty_exit_reads_VERIFIED_on_decisions(cfg, tmp_path) -> None:
    """🚨 THE 31-of-84 CASE. argmax 1.0000 throughout, peer raised afterwards."""
    r = _metamon_report(tmp_path, cfg, sample_kwarg_values=[False], argmax_match_rate=1.0,
                        n_decisions=2533, regime_verified_decisions=True, peer_error_free=False,
                        regime_check_ok=False)
    assert r["regime_verified_decisions"] is True
    assert r["peer_error_free"] is False
    assert r["regime_verified"] is False, "the deprecated composite still ANDs them"


def test_an_argmax_rate_below_one_in_greedy_reads_NOT_verified(cfg, tmp_path) -> None:
    r = _metamon_report(tmp_path, cfg, sample_kwarg_values=[False], argmax_match_rate=0.9922,
                        n_decisions=2533, regime_verified_decisions=False, peer_error_free=True,
                        regime_check_ok=False)
    assert r["regime_verified_decisions"] is False
    assert r["peer_error_free"] is True
    assert r["regime_verified"] is False


def test_a_clean_run_reads_true_on_both(cfg, tmp_path) -> None:
    r = _metamon_report(tmp_path, cfg, sample_kwarg_values=[False], argmax_match_rate=1.0,
                        n_decisions=152, regime_verified_decisions=True, peer_error_free=True,
                        regime_check_ok=True)
    assert (r["regime_verified_decisions"], r["peer_error_free"], r["regime_verified"]) == (
        True, True, True)


def test_a_LEGACY_report_falls_back_to_the_composite_on_BOTH_halves(cfg, tmp_path) -> None:
    """A report written before the split genuinely cannot tell the two apart. Saying so — by
    giving both halves the composite's value — is the honest answer; inventing a decision-level
    pass would be a silent upgrade of an old measurement."""
    r = _metamon_report(tmp_path, cfg, sample_kwarg_values=[False], argmax_match_rate=1.0,
                        n_decisions=9, regime_check_ok=False)
    assert r["regime_verified_decisions"] is False and r["peer_error_free"] is False
    r = _metamon_report(tmp_path, cfg, sample_kwarg_values=[False], argmax_match_rate=1.0,
                        n_decisions=9, regime_check_ok=True)
    assert r["regime_verified_decisions"] is True and r["peer_error_free"] is True


def test_a_missing_report_is_false_on_both_and_never_a_pass(cfg, tmp_path) -> None:
    r = MetamonPeer.read_report(_metamon_plan(cfg, tmp_path))
    assert r["regime_verified_decisions"] is False
    assert r["peer_error_free"] is False
    assert r["regime_verified"] is False


def test_foulplay_carries_the_same_two_fields(cfg, tmp_path) -> None:
    """Foul Play has no sampling knob, so its per-decision evidence is the realized WIDTH — the
    same question through a different instrument. Its ERROR count is the process half."""
    plan = _foulplay_plan(cfg, tmp_path)
    plan.log_path.write_text("Iterations 41233: 1400000\nIterations 39112: 1200000\n")
    r = FoulPlayPeer.read_report(plan)
    assert r["regime_verified_decisions"] is True and r["peer_error_free"] is True
    plan.log_path.write_text("Iterations 41233: 1400000\nERROR something broke\n")
    r = FoulPlayPeer.read_report(plan)
    assert r["regime_verified_decisions"] is True, "the width was still recorded"
    assert r["peer_error_free"] is False and r["regime_verified"] is False


# ------------------------------------------------------------------------- the rc half
def test_peer_clean_is_the_RETURN_CODE_and_the_regime_is_not(cfg, tmp_path) -> None:
    report = {"regime_verified_decisions": True, "peer_error_free": True}
    finalize_regime_fields(report, peer_rcs=[1])
    assert report["regime_verified_decisions"] is True
    assert report["peer_clean"] is False
    assert report["regime_verified"] is False
    assert report["peer_rcs"] == [1]

    report = {"regime_verified_decisions": True, "peer_error_free": True}
    finalize_regime_fields(report, peer_rcs=[0, 0])
    assert report["peer_clean"] is True and report["regime_verified"] is True


def test_an_unknown_return_code_does_not_by_itself_make_a_peer_dirty(cfg, tmp_path) -> None:
    """`None` means the process was never started or never reaped — it is recorded as None and
    does not silently count as a nonzero exit."""
    report = {"regime_verified_decisions": True, "peer_error_free": True}
    finalize_regime_fields(report, peer_rcs=[None])
    assert report["peer_clean"] is True and report["peer_rcs"] == [None]


# ------------------------------------------------------------------------ rows and the summary
from main.anchors.results_test import _cell, _row as _base_row


def _row(**kw) -> GameRow:
    """The `results_test` factory plus this module's three fields. Reusing that factory is the
    point: a CellSpec field added there must not need a second copy here to keep passing."""
    over = dict(regime_verified_decisions=True, peer_clean=True, regime_verified=True)
    over.update(kw)
    return _base_row(_cell(), over.pop("result", "win"), index=over.pop("index", 1), **over)


def test_every_row_carries_the_two_fields_and_the_required_list_demands_them() -> None:
    row = _row().to_json()
    for f in ("regime_verified_decisions", "peer_clean", "regime_verified"):
        assert f in row, f
        assert f in REQUIRED_ROW_FIELDS, f
    assert not [f for f in REQUIRED_ROW_FIELDS if f not in row]


def test_the_summary_separates_a_verified_regime_from_a_dirty_exit() -> None:
    rows = [_row(index=i, peer_clean=False, regime_verified=False) for i in range(1, 6)]
    s = summarize(_cell(), rows, status="OK")
    assert s["regime_verified_decisions"] is True, "the 31-of-84 shape reads VERIFIED now"
    assert s["peer_clean"] is False
    assert s["regime_verified"] is False, "the deprecated composite is still their AND"


def test_a_summary_with_no_rows_verifies_NOTHING_rather_than_everything() -> None:
    """`all([])` is True, and that is the wrong answer: an empty cell has verified nothing."""
    s = summarize(_cell(), [], status="FAILED")
    assert s["regime_verified_decisions"] is False
    assert s["peer_clean"] is False
    assert s["regime_verified"] is False


def test_one_unverified_row_makes_the_cell_unverified() -> None:
    rows = [_row(index=1), _row(index=2, regime_verified_decisions=False, regime_verified=False)]
    s = summarize(_cell(), rows, status="OK")
    assert s["regime_verified_decisions"] is False and s["peer_clean"] is True


def test_the_rendered_block_prints_the_two_facts_on_SEPARATE_lines() -> None:
    rows = [_row(index=i, peer_clean=False, regime_verified=False) for i in range(1, 4)]
    text = render(summarize(_cell(), rows, status="OK"))
    assert "regime      per-decision VERIFIED: True" in text
    assert "peers       exited cleanly: False" in text
    assert "does NOT invalidate a verified regime" in text


def test_the_deprecation_note_exists_and_names_the_replacement() -> None:
    assert "DEPRECATED" in REGIME_VERIFIED_DEPRECATION
    assert "regime_verified_decisions" in REGIME_VERIFIED_DEPRECATION
    assert "peer_clean" in REGIME_VERIFIED_DEPRECATION


def test_the_peer_script_computes_the_split_in_its_own_interpreter() -> None:
    """The driver runs in ANOTHER interpreter (it needs `metamon`/`amago`), so the split has to
    be inside it too — read as text, exactly as the reserved-port guard is."""
    from utils.paths import src_path

    src = src_path("main", "anchors", "peer_scripts", "metamon_side.py").read_text()
    assert 'timing["regime_verified_decisions"]' in src
    assert 'timing["peer_error_free"] = error is None' in src
    # and the vacuous pass is closed: no decisions is no longer a pass
    assert "rate is not None and n_dec >= 1" in src


@pytest.mark.parametrize("regime,rate,ok", [("greedy", 1.0, True), ("greedy", 0.9922, False),
                                            ("t1", 0.647, True), ("t1", 1.0, False)])
def test_the_regime_appropriate_rule_is_what_decisions_means(regime, rate, ok) -> None:
    """🚨 NOT literally "== 1.0". Greedy demands exactly 1.0; t1 demands materially below it, and
    that asymmetry IS the positive control that the instrument has power. Re-derived here so the
    rule is pinned outside the peer script's own source text."""
    n_dec = 100
    got = (rate is not None and n_dec >= 1
           and (rate == 1.0 if regime == "greedy" else rate < 1.0))
    assert got is ok


# ------------------------------------------ Metamon's POST-GAME recursion, named (hazard H-H)
def _report(error: str) -> dict:
    return {"error": error, "argmax_match_rate": 1.0, "n_decisions": 3052}


def test_a_complete_challenger_half_with_the_upstream_recursion_is_NAMED() -> None:
    """It costs no games, and a reader should not have to re-derive that from an rc."""
    note = runner_mod.classify_peer_error(
        _report("RecursionError: maximum recursion depth exceeded while calling a Python object"),
        "peer_challenge", "metamon", 50, 50)
    assert note is not None
    assert note["cause"] == "peer_recursion_upstream"
    # the detail must carry the CAUSE, not just the symptom — that is the whole point
    assert "start_challenging" in note["detail"]
    assert "_accept_challenge_loop" in note["detail"]


def test_an_INCOMPLETE_half_is_not_excused() -> None:
    """🚨 The excuse is 'post-game'. A half that lost games to the recursion is a real failure and
    must keep reading as one — otherwise this classifier launders a short series."""
    assert runner_mod.classify_peer_error(
        _report("RecursionError: maximum recursion depth exceeded"),
        "peer_challenge", "metamon", 37, 50) is None


def test_the_half_where_WE_challenge_is_not_excused() -> None:
    """All four recorded occurrences are in the half where METAMON challenges. A recursion in the
    other half is something we have never seen and must not be pre-labelled as known."""
    assert runner_mod.classify_peer_error(
        _report("RecursionError: maximum recursion depth exceeded"),
        "ours_challenge", "metamon", 50, 50) is None


def test_another_error_is_not_excused() -> None:
    assert runner_mod.classify_peer_error(
        _report("RuntimeError: Agent is not challenging"),
        "peer_challenge", "metamon", 50, 50) is None


def test_a_clean_peer_gets_no_note() -> None:
    assert runner_mod.classify_peer_error(
        {"error": None, "argmax_match_rate": 1.0}, "peer_challenge", "metamon", 50, 50) is None


def test_foulplay_is_not_given_metamons_excuse() -> None:
    assert runner_mod.classify_peer_error(
        _report("RecursionError: boom"), "peer_challenge", "foulplay", 50, 50) is None


def test_the_note_reaches_the_summary_and_is_PRINTED() -> None:
    """A named cause nobody can see is a named cause nobody uses."""
    note = {"cause": "peer_recursion_upstream", "detail": "Metamon's own post-game teardown"}
    summary = results_mod.summarize(_cell(), [], status="OK", peer_exit_notes=[note])
    assert summary["peer_exit_notes"] == [note]
    text = results_mod.render(summary)
    assert "peer_recursion_upstream" in text
