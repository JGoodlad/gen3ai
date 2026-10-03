"""Unit tests for MaskableAgentWrapper's live per-episode opponent selection.

Tests `_select_episode_opponent` / `set_self_play_target` / `opponent_default_stats` directly
with stub env + pool + players (no Showdown server, no full reset)."""

from unittest.mock import MagicMock

import pytest

from agents.training.wrappers import MaskableAgentWrapper, STABLE_CHALLENGE_SHARE


def _stub_env():
    env = MagicMock()
    env.agent1.username = "a1"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    return env


def _make_wrapper(*, fraction=0.0, pool=None, pool_player=None, n_heuristics=2, rng_seed=0,
                  heuristic_weights=None, stable_players=None, stable_labels=None,
                  exploiter_player=None, exploiter_keep_bots=False, exploiter_bot_fraction=0.5):
    heuristics = [MagicMock(name=f"heur{i}") for i in range(n_heuristics)]
    w = MaskableAgentWrapper(
        _stub_env(), heuristic_opponents=heuristics, pool=pool, pool_player=pool_player,
        self_play_fraction=fraction, rng_seed=rng_seed, heuristic_weights=heuristic_weights,
        stable_players=stable_players, stable_labels=stable_labels,
        exploiter_player=exploiter_player, exploiter_keep_bots=exploiter_keep_bots,
        exploiter_bot_fraction=exploiter_bot_fraction,
    )
    return w, heuristics


def _stable(n):
    return ([MagicMock(name=f"stable{i}") for i in range(n)], [f"ext_run{i}" for i in range(n)])


def _stub_pool(empty=False, model="MODEL_X"):
    pool = MagicMock()
    pool.is_empty.return_value = empty
    pool.sample.return_value = MagicMock(name="entry")
    pool.load_model.return_value = model
    return pool


# ── back-compat + construction ────────────────────────────────────────────────

def test_legacy_single_opponent_form():
    opp = MagicMock(name="opp")
    w = MaskableAgentWrapper(_stub_env(), opp)
    assert w._heuristic_opponents == [opp]
    w._select_episode_opponent()
    assert w.opponent is opp          # no pool → always the single opponent


def test_exploiter_player_is_the_sole_opponent():
    # --exploiter mode: the fixed target is ALWAYS the opponent, short-circuiting the pool / stable /
    # heuristic selection entirely — even with fraction=1.0, a ready pool, AND stable players present.
    exploiter = MagicMock(name="exploiter")
    stable_p, stable_l = _stable(2)
    w, heuristics = _make_wrapper(fraction=1.0, pool=_stub_pool(), pool_player=MagicMock(),
                                  stable_players=stable_p, stable_labels=stable_l,
                                  exploiter_player=exploiter)
    for _ in range(50):
        w._select_episode_opponent()
        assert w.opponent is exploiter            # never the pool/stable/heuristic
    # and it never touched the pool (no scan/sample/load) — the exploiter branch returns first
    w._pool.sample.assert_not_called()


def test_exploiter_none_is_unchanged():
    # exploiter_player=None (the default) leaves the normal selection byte-identical.
    w, heuristics = _make_wrapper(fraction=0.0, exploiter_player=None)
    w._select_episode_opponent()
    assert w.opponent in heuristics


def test_exploiter_keep_bots_mixes_target_and_bots():
    # --exploiter-keep-bots: per episode, the opponent is SOMETIMES the exploiter target and
    # SOMETIMES a heuristic bot (via _pick_floor_opponent). Off → ALWAYS the target.
    exploiter = MagicMock(name="exploiter")

    # keep-bots ON at a middling fraction: over many resets we see BOTH the target and bots.
    w, heuristics = _make_wrapper(exploiter_player=exploiter, exploiter_keep_bots=True,
                                  exploiter_bot_fraction=0.5, rng_seed=1)
    seen_target = seen_bot = 0
    for _ in range(400):
        w._select_episode_opponent()
        if w.opponent is exploiter:
            seen_target += 1
        else:
            assert w.opponent in heuristics    # a keep-bots non-target pick is always a floor bot
            seen_bot += 1
    assert seen_target > 0 and seen_bot > 0     # the mix actually mixes both ways
    # ~50/50 at fraction 0.5 (loose bound — this is a randomness sanity check, not an exact ratio)
    assert 0.30 < seen_bot / 400 < 0.70

    # keep-bots OFF (the default) → the target is the SOLE opponent, never a bot.
    w_off, _ = _make_wrapper(exploiter_player=exploiter, exploiter_keep_bots=False, rng_seed=1)
    for _ in range(100):
        w_off._select_episode_opponent()
        assert w_off.opponent is exploiter


def test_exploiter_bot_fraction_zero_is_all_target():
    # keep-bots ON but bot_fraction=0.0 → still always the target (byte-identical to sole-target).
    exploiter = MagicMock(name="exploiter")
    w, _ = _make_wrapper(exploiter_player=exploiter, exploiter_keep_bots=True,
                         exploiter_bot_fraction=0.0, rng_seed=3)
    for _ in range(100):
        w._select_episode_opponent()
        assert w.opponent is exploiter


def test_exploiter_bot_fraction_one_is_all_bots():
    # keep-bots ON with bot_fraction=1.0 → always a floor bot, never the target.
    exploiter = MagicMock(name="exploiter")
    w, heuristics = _make_wrapper(exploiter_player=exploiter, exploiter_keep_bots=True,
                                  exploiter_bot_fraction=1.0, rng_seed=4)
    for _ in range(100):
        w._select_episode_opponent()
        assert w.opponent in heuristics


def test_set_exploiter_temperature_updates_player():
    # gen3_exploiter_temp_anneal_v1: the env_method push sets the exploiter RLPlayer's temperature.
    exploiter = MagicMock(name="exploiter")
    exploiter._temperature = 1.0
    w, _ = _make_wrapper(exploiter_player=exploiter)
    w.set_exploiter_temperature(2.5)
    assert exploiter._temperature == 2.5


def test_set_exploiter_temperature_noop_without_exploiter():
    # A non-exploiter env never has a target — the push must be a harmless no-op (never raises).
    w, _ = _make_wrapper(exploiter_player=None)
    w.set_exploiter_temperature(2.5)


def test_exploiter_winrate_totals_counts_only_target_games():
    # gen3_exploiter_temp_anneal_v1 ratchet signal: outcomes are counted ONLY when the episode's
    # opponent was the exploiter target (bot episodes excluded), so the WR measures the target.
    exploiter = MagicMock(name="exploiter")
    w, heuristics = _make_wrapper(exploiter_player=exploiter, exploiter_keep_bots=True)
    assert w.exploiter_winrate_totals() == (0, 0.0)
    w.opponent = exploiter
    w._record_exploiter_outcome(1.0)      # target win
    w._record_exploiter_outcome(0.0)      # target loss
    w.opponent = heuristics[0]
    w._record_exploiter_outcome(1.0)      # a BOT win — must NOT count
    assert w.exploiter_winrate_totals() == (2, 1.0)


def test_exploiter_winrate_totals_zero_without_exploiter():
    w, heuristics = _make_wrapper(exploiter_player=None)
    w.opponent = heuristics[0]
    w._record_exploiter_outcome(1.0)      # no exploiter → never counts
    assert w.exploiter_winrate_totals() == (0, 0.0)


def test_requires_an_opponent_or_roster():
    with pytest.raises(ValueError):
        MaskableAgentWrapper(_stub_env())


# ── fraction-driven selection ───────────────────────────────────────────────

def test_fraction_zero_always_heuristic():
    pool = _stub_pool()
    w, heuristics = _make_wrapper(fraction=0.0, pool=pool, pool_player=MagicMock())
    for _ in range(50):
        w._select_episode_opponent()
        assert w.opponent in heuristics
    pool.sample.assert_not_called()   # never even consults the pool


def test_fraction_one_uses_pool_and_swaps_model():
    pool = _stub_pool(model="SAMPLED_MODEL")
    pp = MagicMock(name="pool_player")
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp)
    w._select_episode_opponent()
    assert w.opponent is pp
    assert pp.model == "SAMPLED_MODEL"   # swapped in the sampled snapshot's model


def test_fraction_one_but_empty_pool_falls_back_to_heuristic():
    pool = _stub_pool(empty=True)
    w, heuristics = _make_wrapper(fraction=1.0, pool=pool, pool_player=MagicMock())
    w._select_episode_opponent()
    assert w.opponent in heuristics
    pool._scan.assert_called()          # tried a re-scan to discover a (not-yet-written) seed


def test_no_pool_player_always_heuristic():
    w, heuristics = _make_wrapper(fraction=1.0, pool=_stub_pool(), pool_player=None)
    w._select_episode_opponent()
    assert w.opponent in heuristics


# ── live update + generation re-scan ─────────────────────────────────────────

def test_set_self_play_target_updates_fraction():
    w, heuristics = _make_wrapper(fraction=0.0, pool=_stub_pool(), pool_player=MagicMock())
    w.set_self_play_target(1.0, generation=1)
    assert w._self_play_fraction == 1.0
    w._select_episode_opponent()
    assert w.opponent is w._pool_player   # now uses the pool


def test_generation_bump_triggers_pool_rescan():
    pool = _stub_pool()
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=MagicMock())
    w._select_episode_opponent()          # first pool use → initial scan (gen -1 → 0)
    first = pool._scan.call_count
    w._select_episode_opponent()          # same generation → no re-scan
    assert pool._scan.call_count == first
    w.set_self_play_target(1.0, generation=5)
    w._select_episode_opponent()          # new generation → re-scan
    assert pool._scan.call_count == first + 1


def test_pool_model_loaded_once_per_generation():
    """FPS-regression guard: the ~27MB snapshot is (re)loaded once per generation, NOT every
    episode. Per-episode load_model thrashed the workers (blocked in reset() on a 27MB
    deserialize → CPU ~40%, FPS ~1400→~500). Within a generation the opponent must be reused."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp)
    for _ in range(50):
        w._select_episode_opponent()
        assert w.opponent is pp           # always the pool, but...
    pool.load_model.assert_called_once()  # ...one load across 50 same-generation episodes
    # A new generation re-samples + loads exactly once more (not per episode).
    w.set_self_play_target(1.0, generation=7)
    for _ in range(50):
        w._select_episode_opponent()
    assert pool.load_model.call_count == 2


# ── telemetry reads the persistent pool player ───────────────────────────────

def test_opponent_default_stats_reads_pool_player():
    pp = MagicMock(_n_decisions=100, _n_defaults=5, _n_redecides=2)
    w, _ = _make_wrapper(fraction=1.0, pool=_stub_pool(), pool_player=pp)
    # Even with a heuristic currently selected, stats come from the persistent pool player.
    w.opponent = w._heuristic_opponents[0]
    assert w.opponent_default_stats() == (100, 5, 2)


def test_opponent_default_stats_zero_without_pool_player():
    w, _ = _make_wrapper(fraction=0.0, pool=None, pool_player=None)
    assert w.opponent_default_stats() == (0, 0, 0)


# ── #2: weighted heuristic pick (--bot-weights) ──────────────────────────────

def test_bot_weights_none_is_uniform():
    """No weights → every heuristic appears over many draws (the original uniform behavior)."""
    w, heuristics = _make_wrapper(fraction=0.0, n_heuristics=4, rng_seed=1)
    assert w._heuristic_weights is None
    seen = set()
    for _ in range(500):
        w._select_episode_opponent()
        seen.add(id(w.opponent))
    assert seen == {id(h) for h in heuristics}   # all four picked at least once


def test_bot_weights_bias_distribution():
    """weights=[3,1] → the 3-weighted heuristic is drawn ~3x as often (seeded RNG, tolerant)."""
    w, heuristics = _make_wrapper(fraction=0.0, n_heuristics=2, rng_seed=7,
                                  heuristic_weights=[3, 1])
    counts = {id(heuristics[0]): 0, id(heuristics[1]): 0}
    N = 8000
    for _ in range(N):
        w._select_episode_opponent()
        counts[id(w.opponent)] += 1
    share0 = counts[id(heuristics[0])] / N
    assert share0 == pytest.approx(0.75, abs=0.03)   # 3/(3+1)


def test_bot_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        _make_wrapper(n_heuristics=3, heuristic_weights=[1, 1])   # 2 != 3


def test_bot_weights_negative_or_zero_sum_raises():
    with pytest.raises(ValueError):
        _make_wrapper(n_heuristics=2, heuristic_weights=[1, -1])
    with pytest.raises(ValueError):
        _make_wrapper(n_heuristics=2, heuristic_weights=[0, 0])


def test_bot_weights_ignored_on_pool_branch():
    """Weights bias only the heuristic branch — a pool episode still plays the pool player."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp, heuristic_weights=[5, 1])
    w._select_episode_opponent()
    assert w.opponent is pp


# ── stable cross-run opponents: challenge until mastered, then floor ──────────

def test_unmastered_stable_is_a_challenge_peer_without_a_pool():
    """fraction=1 (challenge), empty pool, 1 un-mastered stable → the stable opponent plays — it's
    a challenge candidate in its own right, not gated on a seeded pool."""
    sp, sl = _stable(1)
    w, _ = _make_wrapper(fraction=1.0, pool=_stub_pool(empty=True), pool_player=MagicMock(),
                         stable_players=sp, stable_labels=sl)
    w._select_episode_opponent()
    assert w.opponent is sp[0]


def test_unmastered_stable_is_a_capped_minority_of_challenge():
    """fraction=1 with a seeded pool + 1 un-mastered stable → the stable opponent plays, but only a
    CAPPED minority (~STABLE_CHALLENGE_SHARE) of challenge episodes; the pool gets the bulk. A
    single fixed opponent must never dominate training."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    sp, sl = _stable(1)
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp,
                         stable_players=sp, stable_labels=sl, rng_seed=7)
    N, n_stable = 8000, 0
    for _ in range(N):
        w._select_episode_opponent()
        if w.opponent is sp[0]:
            n_stable += 1
    assert n_stable / N == pytest.approx(STABLE_CHALLENGE_SHARE, abs=0.03)   # ~20%, not ~50%


def test_two_unmastered_stable_share_the_capped_slice():
    """Multiple un-mastered stable opponents SHARE the cap — total stable share stays ≤ the cap
    (each ~cap/2), so adding opponents never grows the stable footprint."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    sp, sl = _stable(2)
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp,
                         stable_players=sp, stable_labels=sl, rng_seed=2)
    N, n_stable = 8000, 0
    for _ in range(N):
        w._select_episode_opponent()
        if w.opponent in sp:
            n_stable += 1
    assert n_stable / N == pytest.approx(STABLE_CHALLENGE_SHARE, abs=0.03)   # total still ~20%


def test_unmastered_stable_not_in_floor():
    """fraction=0 (floor only) → an UN-mastered stable opponent never plays (challenge-only)."""
    sp, sl = _stable(1)
    w, heuristics = _make_wrapper(fraction=0.0, stable_players=sp, stable_labels=sl)
    for _ in range(100):
        w._select_episode_opponent()
        assert w.opponent in heuristics


def test_mastered_stable_leaves_challenge_and_joins_floor():
    """Once mastered: it LEAVES the challenge bucket (fraction=1 plays only the pool) and JOINS the
    floor (fraction=0 plays it alongside the bots) — it "becomes another bot"."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    sp, sl = _stable(1)
    w, heuristics = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp,
                                  stable_players=sp, stable_labels=sl, rng_seed=1)
    w.set_stable_mastered(sl)

    for _ in range(50):                       # challenge bucket: mastered stable is excluded
        w._select_episode_opponent()
        assert w.opponent is pp

    w.set_self_play_target(0.0, generation=0)  # floor only
    seen = set()
    for _ in range(300):
        w._select_episode_opponent()
        assert w.opponent in heuristics or w.opponent is sp[0]
        seen.add(id(w.opponent))
    assert id(sp[0]) in seen                   # the mastered opponent now plays as a floor peer


def test_set_stable_mastered_reflects_the_pushed_set():
    sp, sl = _stable(2)
    w, _ = _make_wrapper(stable_players=sp, stable_labels=sl)
    assert w._stable_mastered == {"ext_run0": False, "ext_run1": False}
    w.set_stable_mastered(["ext_run1"])
    assert w._stable_mastered == {"ext_run0": False, "ext_run1": True}


# ── cross-check: the reporting mirror matches the ACTUAL selection sampling ────
# SelfPlayCallback._opponent_mix_fractions and its per-case unit tests are both hand-derived from
# the SAME model of the rules below — so neither catches the two implementations DRIFTING. This
# runs the REAL _select_episode_opponent many times and asserts its empirical pool/stable shares
# match the analytic fractions: the one guard that fails if a future change to selection here isn't
# mirrored in the reporting (or vice-versa). Seeded → deterministic, not flaky.

def test_mix_fractions_match_actual_sampling():
    from types import SimpleNamespace
    from agents.training.selfplay_callback import SelfPlayCallback

    # (label, sf, pool_empty, n_stable, mastered_labels)
    configs = [
        ("pool-only",            0.9, False, 0, []),
        ("unmastered-caps",      0.9, False, 1, []),
        ("no-pool-unmastered",   0.6, True,  1, []),
        ("mastered-in-floor",    0.8, False, 1, ["ext_run0"]),
    ]
    N = 8000
    for label, sf, pool_empty, n_stable, mastered in configs:
        stable_players, stable_labels = _stable(n_stable)
        pool_player = MagicMock(name="pool")
        w, heuristics = _make_wrapper(
            fraction=sf, pool=_stub_pool(empty=pool_empty), pool_player=pool_player,
            n_heuristics=2, rng_seed=7,
            stable_players=stable_players or None, stable_labels=stable_labels or None)
        w.set_stable_mastered(mastered)

        counts = {"pool": 0, "stable": 0, "bot": 0}
        for _ in range(N):
            w._select_episode_opponent()
            o = w.opponent
            counts["pool" if o is pool_player
                   else "stable" if o in stable_players
                   else "bot"] += 1

        # The reporting mirror, configured to the SAME state (called unbound on a stub `self`).
        mirror = SimpleNamespace(
            _fixed_opponents=[SimpleNamespace(label=lab) for lab in stable_labels],
            _stable_mastered=set(mastered),
            _stable_challenge_share=STABLE_CHALLENGE_SHARE,
            _bot_weight_vec=None,
            _floor_roster_count=len(heuristics),
        )
        sp, st, nb = SelfPlayCallback._opponent_mix_fractions(mirror, sf, pool_ready=not pool_empty)
        assert counts["pool"] / N == pytest.approx(sp, abs=0.03), label
        assert counts["stable"] / N == pytest.approx(st, abs=0.03), label
        assert (counts["pool"] + counts["stable"]) / N == pytest.approx(nb, abs=0.03), label


# ── fold-back: per-opponent pinned teams ──────────────────────────────────────

def _make_pinned_wrapper(*, stable=None, stable_teams=None, exploiter=None, exploiter_team=None,
                         pool_team="POOL_TB", rng_seed=0, exploiter_keep_bots=False,
                         exploiter_bot_fraction=0.5):
    heuristics = [MagicMock(name="heur0"), MagicMock(name="heur1")]
    env = _stub_env()
    env.agent2._team = pool_team          # what Gen3Env(opponent_team=) set at construction
    players, labels = stable or ([], [])
    w = MaskableAgentWrapper(
        env, heuristic_opponents=heuristics, rng_seed=rng_seed,
        stable_players=players, stable_labels=labels, stable_teams=stable_teams,
        exploiter_player=exploiter, exploiter_team=exploiter_team,
        exploiter_keep_bots=exploiter_keep_bots, exploiter_bot_fraction=exploiter_bot_fraction,
        opponent_pool_team=pool_team,
    )
    return w, env


def test_pinned_exploiter_episode_sets_agent2_team_to_pin():
    exploiter = MagicMock(name="exploiter")
    w, env = _make_pinned_wrapper(exploiter=exploiter, exploiter_team="PIN_TB")
    w._select_episode_opponent()
    assert w.opponent is exploiter
    w._apply_opponent_team()
    assert env.agent2._team == "PIN_TB"


def test_bot_episode_restores_pool_team():
    # keep-bots at fraction 1.0 → every episode is a bot → the pool builder must be restored
    # after a pinned-opponent episode set the pin.
    exploiter = MagicMock(name="exploiter")
    w, env = _make_pinned_wrapper(exploiter=exploiter, exploiter_team="PIN_TB",
                                  exploiter_keep_bots=True, exploiter_bot_fraction=1.0)
    env.agent2._team = "PIN_TB"           # as if the previous episode faced the pinned target
    w._select_episode_opponent()
    assert w.opponent is not exploiter    # a bot
    w._apply_opponent_team()
    assert env.agent2._team == "POOL_TB"


def test_pinned_stable_opponent_uses_its_own_pin_others_pool():
    players, labels = _stable(2)
    w, env = _make_pinned_wrapper(stable=(players, labels), stable_teams=["PIN0", None])
    w.opponent = players[0]
    w._apply_opponent_team()
    assert env.agent2._team == "PIN0"     # the pinned specialist brings ITS OWN team
    w.opponent = players[1]
    w._apply_opponent_team()
    assert env.agent2._team == "POOL_TB"  # the unpinned generalist stays a pool pilot


def test_no_pins_never_touches_agent2_team():
    # Byte-identical guard: with no pinned team anywhere the wrapper must never write agent2._team.
    players, labels = _stable(1)
    w, env = _make_pinned_wrapper(stable=(players, labels), stable_teams=[None])
    sentinel = object()
    env.agent2._team = sentinel
    for opp in (players[0], w._heuristic_opponents[0]):
        w.opponent = opp
        w._apply_opponent_team()
        assert env.agent2._team is sentinel


def test_stable_teams_length_mismatch_raises():
    players, labels = _stable(2)
    with pytest.raises(ValueError, match="stable_teams len"):
        _make_pinned_wrapper(stable=(players, labels), stable_teams=["PIN0"])


def test_pins_require_pool_team():
    exploiter = MagicMock(name="exploiter")
    with pytest.raises(ValueError, match="opponent_pool_team"):
        _make_pinned_wrapper(exploiter=exploiter, exploiter_team="PIN_TB", pool_team=None)


# ── dynamic stable-opponent selection (--stable-opponent-pfsp) ────────────────

def _stable_wrapper(pfsp, n=3):
    players = [MagicMock(name=f"ext{i}") for i in range(n)]
    labels = [f"ext_run{i}" for i in range(n)]
    w = MaskableAgentWrapper(_stub_env(), heuristic_opponents=[MagicMock()],
                             stable_players=players, stable_labels=labels,
                             stable_pfsp=pfsp, rng_seed=0)
    return w, players, labels


def test_stable_pfsp_off_is_uniform():
    # OFF (default): _pick_stable == _rng.choice, uniform over the 3 — byte-identical behavior.
    w, players, _ = _stable_wrapper(pfsp=False)
    w.set_stable_win_rates({"ext_run0": 0.1, "ext_run1": 0.9, "ext_run2": 0.9})  # ignored when off
    from collections import Counter
    counts = Counter(id(w._pick_stable(players)) for _ in range(6000))
    fracs = sorted(c / 6000 for c in counts.values())
    assert all(0.28 < f < 0.39 for f in fracs), fracs   # ~1/3 each


def test_stable_pfsp_on_oversamples_the_loser():
    # ON: the opponent we're LOSING to most (lowest win-rate) is picked far more often.
    w, players, labels = _stable_wrapper(pfsp=True)
    w.set_stable_win_rates({labels[0]: 0.10, labels[1]: 0.90, labels[2]: 0.90})  # losing badly to #0
    from collections import Counter
    counts = Counter()
    for _ in range(6000):
        counts[id(w._pick_stable(players))] += 1
    f0 = counts[id(players[0])] / 6000
    f1 = counts[id(players[1])] / 6000
    # weights: (1-.1)=0.9 vs (1-.9)=0.1 → #0 should get ~0.9/(0.9+0.1+0.1)=0.818
    assert f0 > 0.7 and f1 < 0.2, (f0, f1)


def test_stable_pfsp_on_without_rates_is_uniform():
    # ON but no win-rates pushed yet (cold start) → falls back to uniform, no crash.
    w, players, _ = _stable_wrapper(pfsp=True)
    picks = {id(w._pick_stable(players)) for _ in range(200)}
    assert len(picks) == 3                                 # all three reachable


def test_stable_pfsp_mastered_weight_floored():
    # A just-mastered opponent (win_rate ~1.0) keeps a small floor weight (0.05), never zero-division.
    w, players, labels = _stable_wrapper(pfsp=True, n=2)
    w.set_stable_win_rates({labels[0]: 1.0, labels[1]: 1.0})
    picks = [id(w._pick_stable(players)) for _ in range(200)]
    assert set(picks) == {id(players[0]), id(players[1])}  # both still reachable (floor > 0)
