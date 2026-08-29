"""`gen3_winprob_oneply_teacher_v1` — the win-prob one-ply teacher mode (ai_v12 routes 2+3).

What each group pins:

* **THE GATES, as pure functions** — contested / ranking / margin / confirmation. These are the
  filters the design doc calls route 3, and every one of them is the difference between a target
  and a winner's-curse artifact, so they are tested without a session, a model or a battle.
* **THE MODE SEAM** — that `crater` is the default, that both call sites dispatch through ONE
  validator, and that an unknown mode RAISES instead of falling back (a silent fall-back runs a
  different teacher under the same flag at the same coefficient).
* **THE CONSUMER CONTRACT** — a `winprob_oneply` target is a `Correction` the EXISTING AWR loss
  accepts unchanged. If that ever stops being true, this mode has become a second pipeline.
* **OFF byte-identity** — `crater` reaches exactly the code it reached before the flag existed.
* **THE CONFIG GATES** — for an operational flag the `parser.error` is the only gate there is.

Run:
    python -m pytest src/agents/training/teacher/winprob_oneply_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.training.teacher.buffer import Correction, CorrectionBuffer
from agents.training.teacher.modes import (
    MODE_CRATER,
    MODE_WINPROB_ONEPLY,
    TEACHER_MODES,
    produce_for_mode,
    select_for_mode,
    validate_mode,
)
from agents.training.teacher.winprob_oneply import (
    DEFAULT_MARGIN_MIN,
    clears_margin,
    confirmed_better,
    is_contested,
    rank_by_win_prob,
    wilson_lower,
)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1. THE CONTESTED GATE (route 3, filter 1) — the H rule
# ──────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_legal,wp,expected", [
    (3, 0.50, True),      # dead even, several options — the case the mode exists for
    (3, 0.60, True),      # inside the 0.15 band
    (3, 0.66, False),     # outside it: the head is already sure
    (3, 0.34, False),     # symmetric on the losing side
    (1, 0.50, False),     # FORCED — nothing to prefer, so nothing to teach
    (2, 0.50, True),      # two options is enough
])
def test_the_contested_gate_is_the_H_rule(n_legal, wp, expected):
    assert is_contested(n_legal, wp) is expected


def test_a_decision_with_no_recorded_win_prob_is_never_contested_and_is_never_imputed():
    """A run with --win-prob-mode none writes NaN into `win_probs`. A decision we cannot judge is
    one we do not teach from — imputing 0.5 there would make every unjudgeable decision maximally
    attractive to the selector, which is exactly backwards."""
    assert is_contested(3, None) is False
    assert is_contested(3, float("nan")) is False


def test_the_gate_delegates_to_the_searchers_own_gate_rather_than_re_typing_it():
    """Two definitions of 'contested' that can drift apart while both look right is a failure this
    tree has paid for. The teacher's band IS `DefensiveConfig.wp_margin`."""
    from main.search_dividend.defensive import DEFAULT_WP_MARGIN, GATE_SEARCH, DefensiveConfig, gate
    for n_legal, wp in [(3, 0.5), (3, 0.7), (1, 0.5), (2, 0.42)]:
        expect = gate(n_legal, wp, DefensiveConfig(wp_margin=DEFAULT_WP_MARGIN)) == GATE_SEARCH
        assert is_contested(n_legal, wp, DEFAULT_WP_MARGIN) is expect


def test_widening_the_band_admits_strictly_more():
    narrow = [wp for wp in np.linspace(0.0, 1.0, 101) if is_contested(3, wp, 0.05)]
    wide = [wp for wp in np.linspace(0.0, 1.0, 101) if is_contested(3, wp, 0.25)]
    assert set(narrow) < set(wide)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 2. THE RANKING (filter 2) — and the leaf it refuses to substitute
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_actions_are_ranked_by_win_prob_best_first_with_a_deterministic_tie_break():
    ranked = rank_by_win_prob([
        {"action": 5, "win_prob": 0.4}, {"action": 2, "win_prob": 0.7},
        {"action": 9, "win_prob": 0.7}, {"action": 1, "win_prob": 0.1}])
    assert ranked == [(2, 0.7), (9, 0.7), (5, 0.4), (1, 0.1)]


def test_a_candidate_with_no_win_prob_read_is_DROPPED_not_scored_from_the_critic():
    """Falling back to `value` would silently run a DIFFERENT teacher — the critic's shaped-return
    ranking — under the same flag. That is the confusion `defensive.check_leaf` exists to prevent,
    and here the equivalent is simply refusing to rank what the head did not read."""
    ranked = rank_by_win_prob([
        {"action": 0, "win_prob": None, "value": 99.0},
        {"action": 1, "value": 42.0},
        {"action": 2, "win_prob": float("nan")},
        {"action": 3, "win_prob": 0.3}])
    assert ranked == [(3, 0.3)]


def test_an_empty_or_malformed_lookahead_ranks_to_nothing_rather_than_raising():
    assert rank_by_win_prob([]) == []
    assert rank_by_win_prob(None) == []
    assert rank_by_win_prob([{"action": None, "win_prob": 0.5}]) == []
    assert rank_by_win_prob([{"action": 1, "win_prob": "not a number"}]) == []


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 3. THE MARGIN GATE (filter 3)
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_clear_preference_over_the_played_action_passes():
    a_star, margin = clears_margin([(2, 0.70), (5, 0.40)], played_action=5, margin_min=0.02)
    assert (a_star, round(margin, 6)) == (2, 0.30)


def test_a_preference_BELOW_the_floor_is_refused_and_says_so_via_the_margin():
    a_star, margin = clears_margin([(2, 0.51), (5, 0.50)], played_action=5, margin_min=0.02)
    assert a_star is None and margin == pytest.approx(0.01)


def test_the_policy_already_playing_the_heads_preference_is_not_a_target():
    """There is nothing to teach, and the row would only sharpen an existing peak."""
    a_star, _ = clears_margin([(5, 0.70), (2, 0.40)], played_action=5, margin_min=0.0)
    assert a_star is None


def test_the_margin_is_against_the_PLAYED_action_not_the_runner_up():
    """The target exists to move probability OFF what the policy did. Here A* beats the runner-up by
    0.01 but the played action by 0.30 — a runner-up comparison would reject a real correction."""
    a_star, margin = clears_margin([(2, 0.70), (7, 0.69), (5, 0.40)], played_action=5,
                                   margin_min=0.05)
    assert a_star == 2 and margin == pytest.approx(0.30)


def test_an_unscored_played_action_yields_no_target_rather_than_a_fabricated_contrast():
    a_star, margin = clears_margin([(2, 0.70)], played_action=9, margin_min=0.0)
    assert a_star is None and margin == 0.0


def test_the_default_floor_is_the_WORKING_value_not_the_measured_bias_rms():
    """0.122 is the leaf's differential-bias RMS from defensive-search iter 2. Shipping THAT as the
    default would collapse target volume by ~an order of magnitude before any arm had asked whether
    it should — the design doc's E4 is where that question gets measured."""
    assert DEFAULT_MARGIN_MIN == 0.02
    assert DEFAULT_MARGIN_MIN < 0.122


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. CONFIRMATION (route 3 proper) — the winner's-curse defence
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_wilson_lower_bound_is_below_the_point_rate_and_rises_with_evidence():
    assert wilson_lower(8, 8) < 1.0
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(80, 100) > wilson_lower(8, 10), "more evidence, same rate ⇒ a tighter bound"
    assert wilson_lower(5, 10) < 0.5


def test_a_below_the_floor_preference_is_REJECTED_by_confirmation_even_when_it_leads_on_points():
    """The synthetic winner's curse: A* is nominally ahead, and 8 paired rollouts cannot certify it.
    This is the filter that separates a target from an artifact of a biased reader."""
    ok, adv = confirmed_better(star_wins=5, played_wins=4, n=8)
    assert ok is False
    assert adv == pytest.approx(0.125), "the advantage is still REPORTED — it just doesn't pass"


def test_a_decisive_preference_is_confirmed():
    ok, adv = confirmed_better(star_wins=16, played_wins=2, n=16)
    assert ok is True and adv == pytest.approx(0.875)


def test_no_rollouts_is_not_evidence_of_superiority():
    assert confirmed_better(0, 0, 0) == (False, 0.0)


def test_the_confirmation_test_is_ASYMMETRIC_on_purpose():
    """A* is gated on its Wilson LOWER bound against the played action's POINT rate. The failure
    this filter catches is a flattering estimate of A*, not an unflattering one of what was played,
    so the conservatism is spent entirely on the challenger."""
    n = 20
    # identical rates: the lower bound on A* must lose to the played point estimate
    ok, _ = confirmed_better(star_wins=10, played_wins=10, n=n)
    assert ok is False
    assert wilson_lower(10, n) < 10 / n


def test_raising_the_rollout_count_lets_a_REAL_edge_through_that_a_small_sample_refuses():
    small, _ = confirmed_better(star_wins=6, played_wins=4, n=10)      # 0.60 vs 0.40 at n=10
    large, _ = confirmed_better(star_wins=60, played_wins=40, n=100)   # the SAME rates at n=100
    assert small is False and large is True, (
        "the gate is evidence-limited, not effect-limited — the same 20pp edge is refused at n=10 "
        "and admitted at n=100, which is why --teacher-confirm-rollouts is the real knob")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 5. THE MODE SEAM
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_crater_is_the_default_and_the_first_choice():
    assert TEACHER_MODES[0] == MODE_CRATER == "crater"
    assert MODE_WINPROB_ONEPLY in TEACHER_MODES
    assert validate_mode(None) == MODE_CRATER


def test_an_unknown_mode_RAISES_rather_than_falling_back_to_crater():
    """A silent fall-back would run a different teacher than the operator asked for, at the same
    coefficient, with nothing in the logs to say so."""
    with pytest.raises(ValueError, match="unknown --search-teacher-mode"):
        validate_mode("winprob")          # a plausible near-miss


def test_the_dispatcher_routes_to_the_right_pair_of_functions(monkeypatch):
    seen = {}
    monkeypatch.setattr("agents.training.teacher.selection.select_candidates",
                        lambda *a, **k: seen.setdefault("sel", "crater") or [])
    monkeypatch.setattr("agents.training.teacher.winprob_oneply.select_winprob_candidates",
                        lambda *a, **k: seen.setdefault("sel", "wp") or [])
    select_for_mode(MODE_CRATER, "/tmp/x", budget=1, scan_limit=1, falsify_gate=False,
                    window=1, wp_band=0.15)
    assert seen.pop("sel") == "crater"
    select_for_mode(MODE_WINPROB_ONEPLY, "/tmp/x", budget=1, scan_limit=1, falsify_gate=False,
                    window=1, wp_band=0.15)
    assert seen.pop("sel") == "wp"

    monkeypatch.setattr("agents.training.teacher.produce.produce_correction",
                        lambda *a, **k: ("crater", "ok"))
    monkeypatch.setattr("agents.training.teacher.winprob_oneply.produce_winprob_correction",
                        lambda *a, **k: ("wp", "ok"))
    common = dict(opponent_ckpt=None, opponent_source="bot", confirm_rollouts=1, depth=2,
                  beam=3, top_k=4, margin_min=0.0, wp_margin=0.02)
    assert produce_for_mode(MODE_CRATER, None, None, **common)[0] == "crater"
    assert produce_for_mode(MODE_WINPROB_ONEPLY, None, None, **common)[0] == "wp"


def test_the_two_margins_stay_SEPARATE_parameters(monkeypatch):
    """`margin_min` is the crater mode's Wilson gate in win-RATE units; `wp_margin` is the one-ply
    Δφ in win-PROBABILITY units. Collapsing them would silently re-purpose whichever a run set."""
    got = {}
    monkeypatch.setattr("agents.training.teacher.winprob_oneply.produce_winprob_correction",
                        lambda *a, **k: (got.update(k) or (None, "ok")))
    produce_for_mode(MODE_WINPROB_ONEPLY, None, None, opponent_ckpt=None, opponent_source="bot",
                     confirm_rollouts=8, depth=2, beam=3, top_k=4,
                     margin_min=0.99, wp_margin=0.05)
    assert got["margin_min"] == 0.05, "the win-prob producer must receive wp_margin, not margin_min"


def test_both_workers_dispatch_through_the_mode_and_default_to_crater():
    """A config written by an older parent (no `mode` key) must still run exactly as it did."""
    import inspect
    import main.search_teacher_persistent_worker as pw
    import main.search_teacher_worker as w
    for mod in (w, pw):
        src = inspect.getsource(mod)
        assert 'cfg.get("mode", "crater")' in src
        assert "produce_for_mode" in src
    assert "select_for_mode" in inspect.getsource(pw)


def test_the_callback_validates_the_mode_at_construction_not_in_a_subprocess():
    import inspect
    from agents.training.teacher.callback import SearchTeacherCallback
    src = inspect.getsource(SearchTeacherCallback.__init__)
    assert "validate_mode(mode)" in src
    assert inspect.signature(SearchTeacherCallback.__init__).parameters["mode"].default == "crater"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 6. THE CONSUMER CONTRACT — a winprob target is a Correction the EXISTING loss accepts
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_a_winprob_target_is_the_same_record_the_existing_AWR_loss_already_consumes():
    """If this ever stops holding, `winprob_oneply` has become a second pipeline rather than a new
    supply on the existing one."""
    import torch as th
    from agents.training.instrumented_ppo.distill_terms import DistillTerms

    corr = Correction(obs=np.zeros(8, np.float32), action_mask=np.array([1, 1, 0] + [0] * 8, np.int8),
                      better_action=1, advantage=0.4, confirmed_value=0.7,
                      step_produced=1000, opponent="sentinel_1")
    buf = CorrectionBuffer(capacity=4)
    buf.add(corr)
    td = CorrectionBuffer.to_tensors(buf.sample(1), th.device("cpu"))
    assert set(td) >= {"obs_dict", "action_mask", "better_action", "advantage", "confirmed_value"}
    out = DistillTerms._searchteacher_loss(
        th.zeros((1, 11)), td["action_mask"], td["better_action"], td["advantage"])
    assert out is not None and th.isfinite(out[0])


def test_the_shard_format_is_unchanged_so_the_parents_ingest_needs_no_mode_awareness():
    """The worker packs obs/mask (+ optional pi_target) regardless of mode; a winprob correction
    simply never carries a pi_target (no beam ⇒ no per-action backed-up values to soften)."""
    corr = Correction(obs=np.zeros(4, np.float32), action_mask=np.ones(11, np.int8),
                      better_action=0, advantage=0.1, confirmed_value=0.5,
                      step_produced=0, opponent="x")
    assert corr.pi_target is None
    rec = corr.as_record()
    assert "obs" not in rec and rec["better_action"] == 0


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 7. OFF byte-identity + the config gates
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_crater_path_reaches_exactly_the_functions_it_reached_before_the_flag(monkeypatch):
    calls = []
    monkeypatch.setattr("agents.training.teacher.selection.select_candidates",
                        lambda run_dir, **k: calls.append(("select", k)) or [])
    select_for_mode(MODE_CRATER, "/tmp/x", budget=7, scan_limit=9, falsify_gate=True,
                    window=2, wp_band=0.15, step=None)
    (_, kw), = calls
    assert kw == {"budget": 7, "scan_limit": 9, "falsify_gate": True, "window": 2, "step": None}, (
        "the crater selector must see its own arguments and nothing about the win-prob mode")


def test_the_default_argv_selects_the_crater_mode():
    from main.train.parser import build_parser
    args = build_parser().parse_args(["--steps", "1"])
    assert args.search_teacher_mode == "crater"
    assert args.winprob_teacher_band == 0.15
    assert args.winprob_teacher_margin == 0.02


@pytest.mark.parametrize("argv,needle", [
    (["--search-teacher-mode", "winprob_oneply"], "requires --search-teacher"),
    (["--search-teacher-mode", "winprob_oneply", "--search-teacher"],
     "requires --win-prob-mode read_only|shaping"),
    (["--winprob-teacher-band", "0.0"], "must be in (0, 0.5]"),
    (["--winprob-teacher-band", "0.9"], "must be in (0, 0.5]"),
    (["--winprob-teacher-margin", "-0.1"], "must be in [0, 1)"),
])
def test_the_config_gates_refuse_every_way_the_mode_can_be_asked_for_and_not_run(argv, needle,
                                                                                capsys):
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    parser = build_parser()
    args = parser.parse_args(["--steps", "1", "--debug", *argv])
    with pytest.raises(SystemExit):
        resolve_config(args, parser)
    assert needle in capsys.readouterr().err


def test_a_fully_specified_winprob_teacher_argv_is_accepted():
    from main.train.parser import build_parser
    args = build_parser().parse_args([
        "--steps", "1", "--search-teacher", "--search-teacher-mode", "winprob_oneply",
        "--win-prob-mode", "shaping", "--winprob-teacher-band", "0.2",
        "--winprob-teacher-margin", "0.05", "--teacher-confirm-rollouts", "16"])
    assert args.search_teacher_mode == "winprob_oneply"
    assert (args.winprob_teacher_band, args.winprob_teacher_margin) == (0.2, 0.05)
    assert args.teacher_confirm_rollouts == 16
