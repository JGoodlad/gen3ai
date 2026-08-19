"""Unit tests for per-team win-rate TRACKING (``--team-wr-tracking``).

Three layers, because the thing being built spans three files: the teambuilder's per-draw
accumulator (incl. the byte-identity claim on the default uniform draw), the wrapper's episode-end
hook, and the callback's aggregation / emission / restart-safe artifact.

The Node team-validator is mocked (no bridge subprocess) but the team strings are REAL gen3ou
exports, so parse/pack/HP-IV-fix and the ``team_sha`` fingerprints are real.
"""
import json
import random
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agents.model.opp_intent import OPP_CLASS_NAMES
from agents.training.team_archetypes import team_sha
from agents.training.team_winrate_callback import TeamWinRateCallback
from agents.training.wrappers import MaskableAgentWrapper
from utils.teambuilder_test import TEAM_A, TEAM_B, _make_builder

BOT = MaskableAgentWrapper.OPP_CLASS_BOT
POOL = MaskableAgentWrapper.OPP_CLASS_POOL
NCLS = MaskableAgentWrapper.N_OPP_CLASSES


# ── the sha convention: ONE hash, not two that happen to agree today ─────────

def test_pool_keys_are_exactly_team_archetypes_team_sha():
    """The builder inlines the sha (it cannot import agents.* without a cycle), so the agreement is
    a CONVENTION that must be pinned — a drift would silently unjoin every archetype label."""
    tb = _make_builder([TEAM_A, TEAM_B])
    assert tb.get_pool_team_keys() == [team_sha(TEAM_A), team_sha(TEAM_B)]


def test_pool_keys_are_strip_normalized_like_the_artifact():
    """team_sha strips; a trailing newline must NOT produce a different key (the artifact keys are
    shas of what the samplers actually draw)."""
    assert _make_builder([TEAM_A + "\n\n"]).get_pool_team_keys() == [team_sha(TEAM_A)]


def test_get_team_pfsp_keys_is_the_same_list():
    """The PFSP-named accessor is kept as an alias — one key list, two readers."""
    tb = _make_builder([TEAM_A, TEAM_B])
    assert tb.get_team_pfsp_keys() == tb.get_pool_team_keys()


# ── the builder accumulator ───────────────────────────────────────────────────

def test_default_uniform_draw_is_rng_identical_with_tracking():
    """The index is recovered by DICT LOOKUP, never by re-drawing — so the default (team_pfsp off)
    path consumes exactly the RNG it always did. This is the byte-identity claim."""
    tb = _make_builder([TEAM_A, TEAM_B])
    random.seed(4242)
    got = [tb.yield_team() for _ in range(60)]
    random.seed(4242)
    want = [random.choice(tb.packed_teams) for _ in range(60)]
    assert got == want


def test_the_off_path_still_knows_which_team_it_yielded():
    """...and the lookup actually resolves — the whole point of the reverse map."""
    tb = _make_builder([TEAM_A, TEAM_B])
    for _ in range(20):
        team = tb.yield_team()
        assert tb.packed_teams[tb._last_pool_idx] == team


def test_record_and_drain_counts_per_team_and_per_class():
    tb = _make_builder([TEAM_A, TEAM_B])
    tb._last_pool_idx = 0
    tb.record_team_wr_outcome(1.0, POOL, NCLS)
    tb.record_team_wr_outcome(0.0, POOL, NCLS)
    tb.record_team_wr_outcome(1.0, BOT, NCLS)
    tb._last_pool_idx = 1
    tb.record_team_wr_outcome(0.0, POOL, NCLS)

    counts, keys = tb.drain_team_wr_counts()
    assert keys == tb.get_pool_team_keys()
    assert counts[0][1][POOL] == 2.0 and counts[0][0][POOL] == 1.0     # 2 games, 1 win
    assert counts[0][1][BOT] == 1.0 and counts[0][0][BOT] == 1.0
    assert counts[1][1][POOL] == 1.0 and counts[1][0][POOL] == 0.0
    # Drain ZEROES: each pull is one window, so a second pull double-counts nothing.
    assert tb.drain_team_wr_counts()[0] == {}


def test_team_block_episodes_attributes_the_whole_block_to_its_team():
    """--team-block-episodes holds one team for K yields; all K outcomes must land on THAT team.
    (Before the reverse-map the off path left `_block_cached_idx` None, so a blocked default run
    would have attributed nothing at all.)"""
    tb = _make_builder([TEAM_A, TEAM_B])
    tb.set_block_episodes(4)
    seen = set()
    for _ in range(4):
        tb.yield_team()
        seen.add(tb._last_pool_idx)
        tb.record_team_wr_outcome(1.0, POOL, NCLS)
    assert len(seen) == 1 and None not in seen          # one team held across the whole block
    counts, _ = tb.drain_team_wr_counts()
    assert list(counts) == list(seen)
    assert counts[seen.pop()][1][POOL] == 4.0


def test_a_bias_team_yield_is_never_attributed_to_a_pool_team():
    """A distill/bias-pinned team is not a pool member; recording it against the last POOL index
    would corrupt that team's rate."""
    tb = _make_builder([TEAM_A], bias_teams=[TEAM_B], bias_prob=1.0)
    tb.yield_team()
    assert tb._last_pool_idx is None
    tb.record_team_wr_outcome(1.0, POOL, NCLS)
    assert tb.drain_team_wr_counts()[0] == {}


def test_out_of_range_opp_class_is_dropped_not_mis_slotted():
    tb = _make_builder([TEAM_A])
    tb._last_pool_idx = 0
    tb.record_team_wr_outcome(1.0, 99, NCLS)
    assert tb.drain_team_wr_counts()[0] == {}


def test_the_pfsp_table_and_the_tracking_table_are_independent():
    """The two share the draw INDEX and nothing else — PFSP stays off-gated while tracking counts."""
    tb = _make_builder([TEAM_A], team_pfsp="off")
    tb._last_pool_idx = 0
    tb.record_team_wr_outcome(1.0, POOL, NCLS)
    tb.record_team_pfsp_outcome(1.0)                       # off ⇒ PFSP records nothing
    assert tb.drain_team_pfsp_counts()[0] == [0.0]
    assert tb.drain_team_wr_counts()[0][0][1][POOL] == 1.0  # tracking recorded it anyway


# ── the wrapper hook ──────────────────────────────────────────────────────────

def _wrapper(tb, *, tracking=True, opp_class=POOL):
    w = MaskableAgentWrapper.__new__(MaskableAgentWrapper)
    w._team_wr_tracking = tracking
    w._opponent_class = opp_class
    w.env = MagicMock()
    w.env.agent1._team = tb
    return w


def test_wrapper_records_the_episode_outcome_under_this_episodes_class():
    tb = _make_builder([TEAM_A])
    tb._last_pool_idx = 0
    w = _wrapper(tb, opp_class=BOT)
    w._maybe_record_team_wr(1.0)
    counts, _ = tb.drain_team_wr_counts()
    assert counts[0][1][BOT] == 1.0 and counts[0][1][POOL] == 0.0


def test_wrapper_off_records_nothing_and_drains_none():
    tb = _make_builder([TEAM_A])
    tb._last_pool_idx = 0
    w = _wrapper(tb, tracking=False)
    w._maybe_record_team_wr(1.0)
    assert tb.drain_team_wr_counts()[0] == {}
    assert w.drain_team_wr_counts() is None      # off ⇒ the callback sees nothing to aggregate


def test_wrapper_drain_is_none_on_a_non_tracking_builder():
    w = _wrapper(object(), tracking=True)
    assert w.drain_team_wr_counts() is None


# ── the callback ──────────────────────────────────────────────────────────────

def _cb(tmp_path, workers, *, update_every=1, num_timesteps=12345, **kw):
    """A callback shell wired to `workers` (each a `(counts, keys)` drain payload or None).

    ``BaseCallback.training_env`` / ``.logger`` / ``.num_timesteps`` are getter-only properties over
    ``self.model``, so the seam is the model stub."""
    cb = TeamWinRateCallback(run_dir=(str(tmp_path) if tmp_path else None),
                             update_every=update_every, **kw)
    env = MagicMock()
    env.env_method.return_value = list(workers)
    cb.model = MagicMock(logger=MagicMock(), num_timesteps=num_timesteps)
    cb.model.get_env.return_value = env
    cb.num_timesteps = num_timesteps    # a plain attribute on BaseCallback, synced in on_step()
    return cb


def _drain(keys, per_idx):
    """Build one worker's drain payload: {idx: ([wins/class], [games/class])}."""
    counts = {}
    for idx, cells in per_idx.items():
        wins, games = [0.0] * NCLS, [0.0] * NCLS
        for c, (w, g) in cells.items():
            wins[c], games[c] = float(w), float(g)
        counts[idx] = (wins, games)
    return (counts, list(keys))


KEYS = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]


def test_running_math_sums_across_workers_and_windows(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (3, 10)}}),
                        _drain(KEYS, {0: {POOL: (1, 10)}, 1: {POOL: (5, 10)}})])
    cb._on_rollout_end()
    assert cb.rates(min_games=1) == {"aaaaaaaaaa": 0.2, "bbbbbbbbbb": 0.5}

    # A second window ACCUMULATES (raw counts, not a smoothed rate): team a → 4+6 of 20+10.
    cb.training_env.env_method.return_value = [_drain(KEYS, {0: {POOL: (6, 10)}})]
    cb._on_rollout_end()
    assert cb.rates(min_games=1)["aaaaaaaaaa"] == pytest.approx(10 / 30)
    assert cb.table()["aaaaaaaaaa"]["n"] == 30 and cb.table()["aaaaaaaaaa"]["wins"] == 10


def test_rates_can_be_restricted_to_one_opponent_class(tmp_path):
    """The stratification that makes a raw rate readable: 0.9 vs bots and 0.3 vs the pool must not
    average into one uninterpretable 0.6."""
    cb = _cb(tmp_path, [_drain(KEYS, {0: {BOT: (9, 10), POOL: (3, 10)}})])
    cb._on_rollout_end()
    assert cb.rates(min_games=1) == {"aaaaaaaaaa": pytest.approx(0.6)}
    assert cb.rates(min_games=1, opp_classes=[POOL]) == {"aaaaaaaaaa": pytest.approx(0.3)}
    assert cb.rates(min_games=1, opp_classes=[BOT]) == {"aaaaaaaaaa": pytest.approx(0.9)}


def test_min_games_keeps_one_game_teams_out_of_the_summaries(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 1)}, 1: {POOL: (5, 10)}})], min_games=5)
    cb._on_rollout_end()
    assert set(cb.rates(min_games=5)) == {"bbbbbbbbbb"}     # a 1-game 1.000 would own the top-k


def test_update_every_throttles_the_pull(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})], update_every=3)
    for _ in range(2):
        cb._on_rollout_end()
    assert cb.training_env.env_method.call_count == 0
    cb._on_rollout_end()
    assert cb.training_env.env_method.call_count == 1


def test_workers_returning_none_are_filtered(tmp_path):
    cb = _cb(tmp_path, [None, _drain(KEYS, {0: {POOL: (1, 2)}}), None])
    cb._on_rollout_end()
    assert cb.rates(min_games=1) == {"aaaaaaaaaa": 0.5}


def test_all_none_is_a_clean_noop(tmp_path):
    cb = _cb(tmp_path, [None, None])
    cb._on_rollout_end()
    assert cb.rates(min_games=1) == {}


def _block(tmp_path):
    """The table now rides metadata.json as the top-level team_win_rates block (owner rule:
    no TB, ride the existing metadata channel — design_flywheel_tick_tock.md §6b)."""
    return json.loads((tmp_path / "metadata.json").read_text())["team_win_rates"]


def test_diverged_pool_ORDER_across_workers_throws(tmp_path):
    """Same pool SIZE, different ORDER: a per-index count can no longer be keyed to a sha, so every
    per-team number would be attributed to the wrong team. THROW, never average."""
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}}),
                        _drain([KEYS[1], KEYS[0], KEYS[2]], {0: {POOL: (1, 2)}})])
    with pytest.raises(RuntimeError, match="IDENTITY mismatch"):
        cb._on_rollout_end()


# ── emission ──────────────────────────────────────────────────────────────────

def test_NOTHING_is_emitted_to_tensorboard(tmp_path):
    """Owner rule (design_flywheel_tick_tock.md §6b): per-team win rates do not touch TB at all —
    the table rides metadata.json. A future 'just one scalar' regression fails here."""
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (2, 10)}, 1: {POOL: (5, 10)}, 2: {POOL: (9, 10)}})],
             top_k=1, min_games=1)
    cb._on_rollout_end()
    assert cb.logger.record.call_args_list == []


def test_artifact_shape_carries_counts_classes_and_the_confound_note(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {BOT: (9, 10), POOL: (1, 5)}})])
    cb._on_rollout_end()
    doc = _block(tmp_path)
    assert doc["step"] == 12345 and doc["n_teams_seen"] == 1 and doc["n_games"] == 15
    row = doc["teams"]["aaaaaaaaaa"]
    assert row["n"] == 15 and row["wins"] == 10 and row["wr"] == pytest.approx(10 / 15, abs=1e-4)
    assert row["by_class"] == {"bot": {"n": 10, "wins": 9}, "pool": {"n": 5, "wins": 1}}
    assert doc["opp_classes"] == {str(k): v for k, v in OPP_CLASS_NAMES.items()}
    # THE CONFOUND rides with the data, not only in a doc nobody opens beside the file.
    assert "PILOT COMPETENCE" in doc["notes"] and "TEAM STRENGTH" in doc["notes"]


def test_archetype_is_joined_by_team_sha(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    with mock.patch("agents.training.team_archetypes.load_team_archetypes",
                    return_value={"teams": {"aaaaaaaaaa": {"archetype": "stall"}}}):
        cb._on_rollout_end()
    doc = _block(tmp_path)
    assert doc["teams"]["aaaaaaaaaa"]["archetype"] == "stall"


def test_a_missing_archetype_artifact_is_not_fatal(tmp_path):
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    with mock.patch("agents.training.team_archetypes.load_team_archetypes",
                    side_effect=FileNotFoundError):
        cb._on_rollout_end()
    assert _block(tmp_path)["teams"]["aaaaaaaaaa"]["archetype"] is None


# ── restart safety ────────────────────────────────────────────────────────────

def test_a_restart_loads_and_CONTINUES_the_counts(tmp_path):
    first = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (3, 10)}})])
    first._on_rollout_end()

    second = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (7, 10)}})])   # a fresh process, same run dir
    second._on_rollout_end()
    assert second.table()["aaaaaaaaaa"]["n"] == 20                 # not 10 — it continued
    assert second.rates(min_games=1)["aaaaaaaaaa"] == pytest.approx(0.5)


def test_reload_is_keyed_by_sha_so_a_REORDERED_pool_still_joins(tmp_path):
    """The reason to key the artifact by sha rather than pool index: between runs the pool may be
    reordered (or resized) and the reloaded history must still land on the right team."""
    first = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (3, 10)}})])
    first._on_rollout_end()
    reordered = [KEYS[2], KEYS[1], KEYS[0]]                        # team a is now index 2
    second = _cb(tmp_path, [_drain(reordered, {2: {POOL: (7, 10)}})])
    second._on_rollout_end()
    assert second.table()["aaaaaaaaaa"]["n"] == 20


def test_a_corrupt_metadata_file_starts_fresh_rather_than_crashing(tmp_path):
    (tmp_path / "metadata.json").write_text("{not json")
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    cb._on_rollout_end()
    assert cb.table()["aaaaaaaaaa"]["n"] == 2


def test_no_run_dir_still_tracks_and_emits(tmp_path):
    cb = _cb(None, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    cb._on_rollout_end()
    assert cb.rates(min_games=1) == {"aaaaaaaaaa": 0.5}


# ── the seam claim: this is an env_method PULL, which is what makes it async-safe ──

def test_aggregation_reads_env_method_and_never_the_info_dicts(tmp_path):
    """The design claim, pinned: the callback's ONLY input is the drain-safe `env_method` RPC.

    An info-dict route would have to know which buffer row a terminal landed on, which only the
    async collector knows — so a callback reading `self.locals["infos"]` is the thing that breaks
    under `--async-rollout`. Nothing here reads `locals`, so both collectors are covered by the
    same code path (`AsyncSubprocVecEnv.env_method` stashes in-flight steps before the barrier)."""
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    cb.locals = {"infos": [{"win_outcome": 1.0}]}   # present, and deliberately contradictory
    cb._on_rollout_end()
    cb.training_env.env_method.assert_called_once_with("drain_team_wr_counts")
    assert cb.rates(min_games=1) == {"aaaaaaaaaa": 0.5}     # the info dict was NOT consulted


# ── the flag: default ON, an opt-out, and NOT a versioned/arch knob ──────────

@pytest.mark.parametrize("argv,want", [([], True), (["--team-wr-tracking"], True),
                                       (["--no-team-wr-tracking"], False),
                                       (["--team_wr_tracking"], True)])
def test_flag_defaults_on_and_has_a_no_opt_out(argv, want):
    from main.train_rl_agent import build_parser
    assert build_parser().parse_args(argv).team_wr_tracking is want


def test_flag_is_training_runtime_class_not_an_arch_toggle():
    """It never reaches the extractor and scales no loss ⇒ it must NOT be in the extractor flag
    registry and must not appear in a ModelVersion. A drift here would make a pure instrument
    resume-FATAL against every existing checkpoint."""
    from agents.model import flag_registry
    assert not any("team_wr" in str(n) for n in dir(flag_registry))
    from agents.model.model_version import ModelVersion
    assert not any("team_wr" in f for f in getattr(ModelVersion, "__dataclass_fields__", {}))


def test_on_step_does_no_work(tmp_path):
    """Per-step is the hot path; all the work is at the rollout boundary."""
    cb = _cb(tmp_path, [_drain(KEYS, {0: {POOL: (1, 2)}})])
    assert cb._on_step() is True
    cb.training_env.env_method.assert_not_called()
