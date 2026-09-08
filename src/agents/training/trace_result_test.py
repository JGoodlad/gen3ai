"""The DRAW bucket — one test per seam, with a PLANTED tie and a PLANTED timeout.

`gen3_trace_result_v2`. The defect these pin (2026-09-07): `meta.result` was WIN/LOSS only,
filenames were `win_`/`loss_`, a 250-turn TIMEOUT was written as an ordinary LOSS, and a true TIE
matched no quota branch and was dropped without a file. 145,173 archived traces, of which — verified
by scanning the whole `models/` tree — exactly **zero** carry any result but WIN or LOSS. "0 draws"
was what the instrument could express, not what happened.

Every test here is pure stdlib + tmp_path. Nothing reads `models/`.
"""

import json
import os

import pytest

from agents.training.trace_result import (
    DRAW, DRAW_TIE, DRAW_TIMEOUT, LOSS, OUTCOMES, PRE_DRAW_BUCKET_NOTE, RESULT_VOCABULARY,
    RESULT_VOCABULARY_V1, UnknownTraceResult, WIN, check_draw_kind, check_result, classify_result,
    era_note, is_pre_draw_bucket, outcome_prefix, result_era, result_of,
)

CAP = 250


# ── the classification itself ────────────────────────────────────────────────────────────────

def test_a_timeout_is_a_DRAW_even_though_the_battle_layer_calls_it_a_loss():
    """THE defect, in one assertion.

    At the turn cap the trainee forfeits (`inference/player._handle_stall` →
    `ForfeitBattleOrder`), so poke-env reports `lost=True` — which is why a timeout wore a loss's
    clothes for five months. The training reward never agreed: `reward_manager`'s terminal fold
    pays `draw_penalty` for exactly this state, detected by the TURN COUNT, not by won/lost."""
    assert classify_result(won=False, lost=True, finished=True, turn=CAP, turn_cap=CAP) \
        == (DRAW, DRAW_TIMEOUT)
    assert classify_result(won=False, lost=True, finished=True, turn=CAP + 9, turn_cap=CAP) \
        == (DRAW, DRAW_TIMEOUT)


def test_a_tie_before_the_cap_is_a_DRAW_and_is_told_APART_from_a_timeout():
    """The sim's `|tie|` leaves `_won` None, so won/lost are both falsy. Under the old two-branch
    quota that matched NEITHER and the buffered capture was discarded — no file, no count."""
    assert classify_result(won=False, lost=False, finished=True, turn=41, turn_cap=CAP) \
        == (DRAW, DRAW_TIE)
    # …and the two draws are DISTINGUISHABLE, which is the whole reason `draw_kind` exists.
    tie = classify_result(won=False, lost=False, finished=True, turn=41, turn_cap=CAP)
    timeout = classify_result(won=False, lost=True, finished=True, turn=CAP, turn_cap=CAP)
    assert tie[0] == timeout[0] == DRAW
    assert tie[1] != timeout[1]


def test_a_decisive_win_and_a_decisive_loss_are_unchanged():
    assert classify_result(won=True, lost=False, finished=True, turn=30, turn_cap=CAP) == (WIN, None)
    assert classify_result(won=False, lost=True, finished=True, turn=30, turn_cap=CAP) == (LOSS, None)


def test_a_win_at_the_cap_is_still_a_WIN():
    """Ordering guard: the timeout test must not swallow a win that happened to land on turn 250."""
    assert classify_result(won=True, lost=False, finished=True, turn=CAP, turn_cap=CAP)[0] == WIN


def test_the_classification_matches_the_training_rewards_own_timeout_rule():
    """Parity with the code that PAYS for the outcome — the same `turn >= cap` test, on the same
    constant. If these two ever disagree, a trace's label contradicts the reward that shaped the
    behaviour it records."""
    from agents.training.reward_weights import _TIMEOUT_TURN_CAP
    from agents.observation.constants import MAX_TURNS
    assert _TIMEOUT_TURN_CAP == MAX_TURNS
    res, kind = classify_result(won=False, lost=True, finished=True,
                                turn=_TIMEOUT_TURN_CAP, turn_cap=_TIMEOUT_TURN_CAP)
    assert (res, kind) == (DRAW, DRAW_TIMEOUT)


# ── the throwing guard ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bogus", ["TIE", "tied", "draw?", "", "  ", "UNKNOWN", "0"])
def test_an_unknown_result_is_REFUSED_never_coerced(bogus):
    """The GIGO rule. A coerced unknown is a silent misclassification — and the last one cost five
    months of invisible timeouts, so this raises rather than falling back to LOSS.

    "TIE" is in the list deliberately: it is the token the PREVIOUS writer would have emitted for
    a tie, and it is NOT part of this vocabulary. (It never reached disk — a tie was dropped before
    the summary was written — but a reader must refuse it, not translate it.)"""
    with pytest.raises(UnknownTraceResult):
        check_result(bogus)


def test_check_result_accepts_the_vocabulary_and_normalises_case():
    for r in (WIN, LOSS, DRAW):
        assert check_result(r) == r
        assert check_result(r.lower()) == r


def test_an_unknown_draw_kind_is_refused_too():
    assert check_draw_kind(None) is None
    assert check_draw_kind("timeout") == DRAW_TIMEOUT
    with pytest.raises(UnknownTraceResult):
        check_draw_kind("stall")


def test_outcome_prefix_is_the_filename_contract():
    assert [outcome_prefix(r) for r in (WIN, LOSS, DRAW)] == ["win", "loss", "draw"]
    assert OUTCOMES == ("win", "loss", "draw")
    with pytest.raises(UnknownTraceResult):
        outcome_prefix("TIE")


# ── reading an OLD trace (backward compatibility) ────────────────────────────────────────────

def test_a_pre_draw_bucket_trace_is_readable_and_reported_AS_IT_WAS_WRITTEN():
    """The ~145k archived traces stay readable. A LOSS written in the old era is still a LOSS —
    nothing rewrites history — and the era is identified by the ABSENCE of the capture-version
    field, never by guessing from the content."""
    legacy = {"step": 1000, "result": "LOSS", "turns": 250, "invocations": 40}
    assert result_of(legacy) == LOSS
    assert result_era(legacy) == RESULT_VOCABULARY_V1
    assert is_pre_draw_bucket(legacy)
    note = era_note([legacy])
    assert note == PRE_DRAW_BUCKET_NOTE
    # and the note SAYS the two things a reader would otherwise get wrong
    assert "TIMEOUT" in note and "NOT MEASURABLE" in note


def test_a_current_trace_names_its_vocabulary_and_needs_no_note():
    current = {"result": "DRAW", "draw_kind": DRAW_TIMEOUT, "turns": 250,
               "result_vocabulary": RESULT_VOCABULARY}
    assert result_era(current) == RESULT_VOCABULARY
    assert not is_pre_draw_bucket(current)
    assert era_note([current]) is None


def test_a_MIXED_tree_still_gets_the_note():
    """A run that restarted onto new code mid-flight has both eras. The pre-bucket half is still
    unsplittable, so the note stays — and says the sample is mixed."""
    note = era_note([{"result": "LOSS"},
                     {"result": "WIN", "result_vocabulary": RESULT_VOCABULARY}])
    assert note is not None and "MIXED" in note


def test_an_unrecognised_vocabulary_tag_is_reported_verbatim_not_assumed_current():
    meta = {"result": "WIN", "result_vocabulary": "gen3_trace_result_v99"}
    assert result_era(meta) == "gen3_trace_result_v99"
    assert "UNKNOWN RESULT VOCABULARY" in (era_note([meta]) or "")


def test_an_absent_result_is_absent_not_a_loss():
    assert result_of({}) is None
    assert result_of(None) is None


# ── the FILENAME seam (producer -> prober) ───────────────────────────────────────────────────

def test_the_prober_parses_a_planted_draw_filename():
    """`trace_filename_stem` (producer) → `discovery._parse_trace` (consumer), for the NEW bucket.

    The same contract that silently drifted when sharding added the `s<shard>_` infix and every
    sharded trace parsed as outcome '?' — blinding the whole prober."""
    from agents.training.eval_callback import trace_filename_stem
    from main.prober.discovery import _parse_trace
    for tag in ("", "s0_", "s3_"):
        stem = trace_filename_stem(outcome_prefix(DRAW), tag, 7)
        t = _parse_trace(f"/run/eval_traces/step_100/heuristic/{stem}_summary.json")
        assert t.outcome == "draw", tag
        assert t.opponent == "heuristic"


def test_a_draw_trace_sorts_after_wins_and_losses_but_before_the_unparsed():
    from main.prober.discovery import _outcome_sort_key
    keys = [_outcome_sort_key(o) for o in ("win", "loss", "draw", "?")]
    assert keys == sorted(keys) and len(set(keys)) == 4


# ── the QUOTA seam ───────────────────────────────────────────────────────────────────────────

def test_draws_have_their_OWN_quota_and_cannot_evict_a_loss():
    """WHERE DRAWS SIT, asserted rather than left to a comment: an independent bucket.

    Folding draws into the loss quota is what the old code did, and it means a stall storm evicts
    the decisive losses the prober exists to study."""
    from agents.training.eval_callback import (
        _FORENSIC_DRAW_QUOTA, _FORENSIC_LOSS_QUOTA, _FORENSIC_WIN_QUOTA,
    )
    assert _FORENSIC_DRAW_QUOTA > 0
    # independent: the three constants are three numbers, and the recorded RULE names all three
    from agents.training.eval_callback import forensic_selection_rule
    rule = forensic_selection_rule()
    for q in (_FORENSIC_WIN_QUOTA, _FORENSIC_LOSS_QUOTA, _FORENSIC_DRAW_QUOTA):
        assert str(q) in rule
    assert "DRAWS" in rule and "timeout" in rule


def test_the_manifest_records_the_draw_counts_and_its_capture_rate(tmp_path):
    """`gen3_trace_selection_manifest_v1` at schema 2 — the draw denominator is a real count from
    the player, because poke-env supplies none (a tie is neither a win nor a loss to it, and a
    timeout is booked as our forfeit)."""
    from agents.training.eval_callback import record_eval_selection, write_eval_manifest
    from agents.training.trace_selection import read_selection
    run = str(tmp_path / "run")
    os.makedirs(run, exist_ok=True)
    (tmp_path / "run" / "metadata.json").write_text(json.dumps({"git_hash": "abc123"}))
    write_eval_manifest(run, 1000, opponents=["staller"], n_games=100)
    merged = {"counts": {"staller": (40, 100)},
              "draws": {"staller": 12},
              "traces": {"staller": (5, 20, 4)}}       # (won, written, drawn)
    block = record_eval_selection(run, 1000, merged)
    e = block["opponents"]["staller"]
    assert e["battles_drawn"] == 12 and e["traces_drawn"] == 4
    assert e["capture_rate_draw"] == pytest.approx(4 / 12)
    # 🚨 A DRAW IS SUBTRACTED FROM THE LOSSES, not added to the played count: 100 played,
    # 40 won, 12 drawn ⇒ 48 DECISIVE losses, and 20−5−4 = 11 of them traced.
    assert e["capture_rate_loss"] == pytest.approx(11 / 48)
    on_disk = json.loads((tmp_path / "run" / "eval_traces" / "step_1000"
                          / "eval_manifest.json").read_text())
    assert read_selection(on_disk)["draw_quota"] > 0


def test_a_schema_1_manifest_still_READS_and_its_draws_are_absent_not_zero(tmp_path):
    """Backward compatibility for every cycle recorded before today. Demoting schema 1 to
    SELECTION UNKNOWN would silently throw away a record that is perfectly good — it simply says
    nothing about draws, which is different from saying there were none."""
    from agents.training.trace_selection import (
        UNKNOWN_LABEL, describe_selection, read_selection,
    )
    legacy = {"selection": {"schema": 1, "win_quota": 5, "loss_quota": 10, "opponents": {
        "heuristic": {"battles_played": 100, "battles_won": 90, "traces_written": 15,
                      "traces_won": 5, "capture_rate_win": 5 / 90, "capture_rate_loss": 1.0},
    }}}
    sel = read_selection(legacy)
    assert sel is not None and sel["schema"] == 1
    e = sel["opponents"]["heuristic"]
    assert "battles_drawn" not in e          # ABSENT
    desc = describe_selection(legacy)
    assert desc != UNKNOWN_LABEL
    assert "DRAWS NOT COUNTED" in desc and "would be a claim" in desc

    # an UNKNOWN schema is still refused — SELECTION UNKNOWN, never guessed
    assert read_selection({"selection": dict(legacy["selection"], schema=99)}) is None


def test_a_two_tuple_traces_entry_still_records_with_draws_left_absent(tmp_path):
    """The `traces` tuple grew a third element. Unpacking BY LENGTH means an older parent (or a
    caller written before the bucket) still records, with a 0 draw count that is honest only
    because that build could not persist a draw at all."""
    from agents.training.eval_callback import record_eval_selection, write_eval_manifest
    run = str(tmp_path / "run")
    os.makedirs(run, exist_ok=True)
    (tmp_path / "run" / "metadata.json").write_text(json.dumps({"git_hash": "abc123"}))
    write_eval_manifest(run, 1000, opponents=["h"], n_games=10)
    block = record_eval_selection(run, 1000,
                                  {"counts": {"h": (7, 10)}, "traces": {"h": (2, 5)}})
    e = block["opponents"]["h"]
    assert e["traces_drawn"] == 0 and e["battles_drawn"] == 0


def test_the_shard_layer_pools_draw_counts_and_reads_a_legacy_shard(tmp_path):
    """`n_drawn` / `traces_drawn` are additive like every other ShardResult field, and defaulted
    so a shard written before today still deserializes."""
    from agents.training.eval_sharding import EvalItem, ShardResult, ShardUnit, BOT
    from agents.training.eval_sharding.results import aggregate, to_merged, write_shard_result
    item = EvalItem(key="h", kind=BOT, n_games=10)
    units = [ShardUnit(item=item, shard_index=0, n_games=5),
             ShardUnit(item=item, shard_index=1, n_games=5)]
    d = str(tmp_path)
    write_shard_result(d, ShardResult(unit_id=units[0].unit_id, item_key="h", worker_id=0,
                                      n_won=3, n_finished=5, sum_reward=1.0, n_episodes=5,
                                      sum_ep_len=50.0, duration_sec=1.0,
                                      traces_written=3, traces_won=1,
                                      n_drawn=2, traces_drawn=2))
    legacy = {"unit_id": units[1].unit_id, "item_key": "h", "worker_id": 1, "n_won": 3,
              "n_finished": 5, "sum_reward": 1.0, "n_episodes": 5, "sum_ep_len": 50.0,
              "duration_sec": 1.0, "td_residuals": []}
    with open(os.path.join(d, f"shard__{units[1].unit_id}.json"), "w") as f:
        json.dump(legacy, f)
    pooled = aggregate(units, d)["h"]
    assert (pooled.n_drawn, pooled.traces_drawn) == (2, 2)
    merged = to_merged({"h": pooled})
    assert merged["draws"]["h"] == 2
    assert merged["traces"]["h"] == (1, 3, 2)
