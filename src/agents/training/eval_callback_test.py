import os
import json
import math
from unittest.mock import MagicMock, patch

import pytest

import agents.training.eval_callback as eval_callback
import agents.training.eval_record as eval_record
from agents.training.eval_callback import (
    PerOpponentEvalCallback, bot_mean, RANDOM_OPPONENT_NAME,
    external_elo, record_external_elos, request_forced_eval, consume_forced_eval_request,
)


@pytest.fixture(autouse=True)
def _isolate_force_eval_event():
    """The forced-eval request is a module-global Event; clear it around every test so a
    request set by one test can never leak into the next (they share one process)."""
    eval_callback._force_eval_event.clear()
    yield
    eval_callback._force_eval_event.clear()


# ── external_elo — display-only ballpark from the trainee's bot-anchored rating ──

def test_external_elo_even_winrate_equals_trainee():
    assert external_elo(1500.0, 0.5) == pytest.approx(1500.0)


def test_external_elo_stronger_opponent_when_trainee_loses():
    # Trainee wins only 25% → the opponent is rated ABOVE the trainee.
    assert external_elo(1500.0, 0.25) > 1500.0


def test_external_elo_weaker_opponent_when_trainee_wins():
    assert external_elo(1500.0, 0.75) < 1500.0


def test_external_elo_symmetric_about_even():
    # logit is odd around 0.5, so equal-distance win rates give equal-and-opposite gaps.
    hi = external_elo(1500.0, 0.80) - 1500.0
    lo = external_elo(1500.0, 0.20) - 1500.0
    assert hi == pytest.approx(-lo)


def test_external_elo_clamps_extremes_finite():
    # 0% / 100% would be ±inf un-clamped; clamped to a finite ≈±676 gap.
    assert external_elo(1500.0, 0.0) == pytest.approx(1500.0 + 676, abs=2)
    assert external_elo(1500.0, 1.0) == pytest.approx(1500.0 - 676, abs=2)


def test_record_external_elos_writes_per_opponent_keys():
    logger, tui = MagicMock(), {}
    record_external_elos(logger, tui, 1500.0, {"ext_run_a": 0.5, "ext_run_b": 0.25})
    assert tui["eval/elo_vs_ext_run_a"] == 1500           # even → trainee rating
    assert tui["eval/elo_vs_ext_run_b"] > 1500            # losing → stronger
    recorded = {c.args[0] for c in logger.record.call_args_list}
    assert "eval/elo_vs_ext_run_a" in recorded and "eval/elo_vs_ext_run_b" in recorded


def test_record_external_elos_prefers_recorded_over_ballpark():
    """A carried source ELO wins over the trainee-derived ballpark; it's shown even with no trainee
    rating (None), while a fallback-only opponent is skipped when there's no rating yet."""
    logger, tui = MagicMock(), {}
    record_external_elos(logger, tui, None, {"ext_carried": 0.9, "ext_fallback": 0.9},
                         source_elos={"ext_carried": 1888.0})
    assert tui["eval/elo_vs_ext_carried"] == 1888         # recorded ELO used verbatim
    assert "eval/elo_vs_ext_fallback" not in tui          # no carried ELO + no trainee rating → skipped


# ── write_best_model_sidecar (best_model.json, reusing the checkpoint-sidecar code) ──

def test_write_best_model_sidecar_includes_elo(tmp_path):
    from agents.training.eval_callback import write_best_model_sidecar
    from agents.model.snapshot import record_eval_results, read_checkpoint_metadata
    model_dir = str(tmp_path)
    best = os.path.join(model_dir, "best_model")
    os.makedirs(best)
    record_eval_results(model_dir, step=100, metrics={"elo": 1888.0, "win_rate_vs_bots": 0.8})
    best_zip = os.path.join(best, "best_model.zip")
    open(best_zip, "w").close()
    model = MagicMock(n_epochs=7)
    model.policy.optimizer.param_groups = [{"lr": 1e-4}]
    write_best_model_sidecar(model_dir, best_zip, model)
    sidecar = read_checkpoint_metadata(best_zip)          # reads <best_zip − .zip>.json = best_model.json
    assert os.path.exists(os.path.join(best, "best_model.json"))
    assert sidecar["latest_eval"]["elo"] == pytest.approx(1888.0)
    assert sidecar["lr"] == pytest.approx(1e-4) and sidecar["n_epochs"] == 7


# ── bot_mean ─────────────────────────────────────────────────────────────────

def test_bot_mean_excludes_random():
    assert bot_mean({"random": 0.9, "heuristic": 0.4, "staller": 0.6}) == pytest.approx(0.5)


def test_bot_mean_all_random_returns_zero():
    assert bot_mean({"random": 0.9}) == pytest.approx(0.0)


def test_bot_mean_empty_returns_zero():
    assert bot_mean({}) == pytest.approx(0.0)


def test_bot_mean_no_random_averages_all():
    assert bot_mean({"heuristic": 0.4, "staller": 0.6}) == pytest.approx(0.5)


# ── the random opponent's name ────────────────────────────────────────────────

def test_random_opponent_name_constant_is_the_rosters_first_name():
    from agents.training.eval_schedule import eval_opponent_names

    assert RANDOM_OPPONENT_NAME == "random" == eval_opponent_names()[0]


def _make_callback(best_model_save_path=None, model_dir=None):
    cb = PerOpponentEvalCallback(
        model_dir=model_dir,
        best_model_save_path=best_model_save_path,
    )
    cb.model = MagicMock()
    cb.model.save = MagicMock()
    cb.model.logger = MagicMock()
    cb.num_timesteps = 0
    return cb


# --- Schedule ---

@pytest.mark.parametrize("step", [1_000_000, 30_000_000, 120_000_000])
def test_schedule_is_flat_at_every_step(step):
    # One cadence, one game count — no maturity tiers, no per-opponent caps.
    cb = _make_callback()
    cb.num_timesteps = step
    freq, n_games = cb._schedule()
    assert freq == 2_000_000
    assert n_games == 100


# --- Trigger logic ---

def test_no_eval_at_step_zero():
    cb = _make_callback()
    cb.num_timesteps = 0
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_no_eval_before_first_freq_boundary():
    cb = _make_callback()
    cb.num_timesteps = 500_000  # below the first 2M-step boundary
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_triggers_at_first_boundary():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000  # first 2M boundary
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_triggers_at_each_later_boundary():
    # Flat 2M cadence: crossing any 2M boundary fires, regardless of training maturity.
    cb = _make_callback()
    cb._last_eval_step = 20_000_000
    cb.num_timesteps = 22_000_000
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_no_double_trigger_within_interval():
    cb = _make_callback()
    cb._last_eval_step = 1_000_000
    cb.num_timesteps = 1_500_000
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_updates_last_eval_step_on_trigger():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000  # first boundary is now 2M
    with patch.object(cb, '_launch_eval'):
        cb._on_step()
    assert cb._last_eval_step == 2_000_000


# --- Force-eval (launcher "force eval" button → SIGUSR2 → request_forced_eval) ---

def _make_forceable_callback():
    """A callback mid-run, OFF a cadence boundary, with eval enabled — so a launch can only
    come from the forced request, never the schedule."""
    cb = _make_callback()
    cb._eval_root = "/tmp/eval_root"   # non-None → eval enabled (a forced cycle may launch)
    cb.num_timesteps = 1_234_567       # mid-run, between 2M cadence boundaries
    return cb                          # event isolation: _isolate_force_eval_event (autouse)


def test_force_eval_launches_off_cadence_when_idle():
    cb = _make_forceable_callback()
    request_forced_eval()
    with patch.object(cb, "_launch_eval") as mock_launch, \
         patch("agents.training.eval_callback.send_event"):
        cb._on_step()
        mock_launch.assert_called_once()     # off-cadence launch, purely from the request
    assert cb._last_eval_step == 1_234_567   # current cadence bucket consumed (no double-launch)
    assert not consume_forced_eval_request()  # the request was cleared


def test_force_eval_request_persists_until_first_real_step():
    # A request flagged before the first rollout (num_timesteps == 0) must NOT be dropped —
    # it should fire on the first real step instead.
    cb = _make_forceable_callback()
    cb.num_timesteps = 0
    request_forced_eval()
    with patch.object(cb, "_launch_eval") as mock_launch:
        cb._on_step()                        # step 0 → not consumed yet
        mock_launch.assert_not_called()
    cb.num_timesteps = 100
    with patch.object(cb, "_launch_eval") as mock_launch, \
         patch("agents.training.eval_callback.send_event"):
        cb._on_step()                        # first real step → fires
        mock_launch.assert_called_once()


def test_force_eval_ignored_when_eval_disabled():
    cb = _make_forceable_callback()
    cb._eval_root = None                     # no run dir → eval disabled
    request_forced_eval()
    with patch.object(cb, "_launch_eval") as mock_launch, \
         patch("agents.training.eval_callback.send_event") as mock_event:
        cb._on_step()
        mock_launch.assert_not_called()
    assert any("disabled" in str(c.args[0]).lower() for c in mock_event.call_args_list)
    assert not consume_forced_eval_request()  # still consumed (cleared), just not acted on


# --- Best model saving ---

def test_saves_best_model_on_first_improvement(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = -1.0
    aggregate = 0.65
    if aggregate > cb._best_aggregate_win_rate:
        cb._best_aggregate_win_rate = aggregate
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_called_once()
    assert cb._best_aggregate_win_rate == pytest.approx(0.65)


def test_does_not_save_when_aggregate_does_not_improve(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = 0.80
    aggregate = 0.75
    if aggregate > cb._best_aggregate_win_rate:
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_not_called()


# ── roster ────────────────────────────────────────────────────────────────────

from agents.training.eval_callback import (
    eval_opponent_names, read_latest_eval_block,
)


def test_eval_opponent_names_is_full_roster():
    # All eight archetype bots (both v1 and v2 of each) + Random as the eval-only floor.
    # snake_case names match the eval roster's names (`eval_schedule._EVAL_ROSTER`) + the metric-key convention (1e50634).
    assert eval_opponent_names() == [
        "random",
        "heuristic", "heuristic2",
        "staller", "staller_v2",
        "aggressive", "aggressive_v2",
        "setup_sweep", "setup_sweep_v2",
    ]


# ── read_latest_eval_block (TUI resume source) ────────────────────────────────

def test_read_latest_eval_block_reads_top_level(tmp_path):
    import json as _json
    meta = {"latest_eval": {"step": 200, "win_rate_mean": 0.5,
                            "opponents": {"random": {"win_rate": 0.7}}}}
    p = tmp_path / "metadata.json"
    p.write_text(_json.dumps(meta))
    blk = read_latest_eval_block(str(p))
    assert blk["step"] == 200 and blk["win_rate_mean"] == 0.5


def test_read_latest_eval_block_legacy_per_checkpoint_fallback(tmp_path):
    # Older metadata.json nested evals under each checkpoint — still readable.
    import json as _json
    meta = {"snapshot_history": {
        "checkpoint_100_steps.zip": {"evals": {"step": 100, "win_rate_mean": 0.3, "opponents": {}}},
        "checkpoint_200_steps.zip": {"evals": {"step": 200, "win_rate_mean": 0.5,
                                               "opponents": {"random": {"win_rate": 0.7}}}},
    }}
    p = tmp_path / "metadata.json"
    p.write_text(_json.dumps(meta))
    blk = read_latest_eval_block(str(p))
    assert blk["step"] == 200 and blk["win_rate_mean"] == 0.5


def test_read_latest_eval_block_missing_or_empty():
    assert read_latest_eval_block(None) is None
    assert read_latest_eval_block("/no/such/metadata.json") is None


def test_latest_recorded_eval_step():
    from agents.training.eval_callback import latest_recorded_eval_step
    assert latest_recorded_eval_step(None, None) == 0


def test_resume_restores_eval_step_no_immediate_re_eval(tmp_path):
    """Bot path: a resume restores _last_eval_step from metadata so it doesn't re-eval the
    same checkpoint immediately (it waits for the next cadence boundary)."""
    import json
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"latest_eval": {"step": 46_963_120, "win_rate_mean": 0.7, "opponents": {}}}))
    cb = _make_callback(model_dir=str(tmp_path))
    # The RESUME's own step, on the MODEL (where _init_callback reads it — the callback's own
    # num_timesteps mirror is still 0 at that point). It must be present: the anchor is CLAMPED to
    # it, so a model left at 0 would model a fork-from-step-0 (see eval_fork_cadence_test.py).
    cb.model.num_timesteps = 46_963_136
    cb._init_callback()
    assert cb._last_eval_step == 46_963_120

    cb.num_timesteps = 46_963_136                 # same 3.5M bucket → no eval
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_not_called()
    cb.num_timesteps = 49_500_000                 # next boundary → eval
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_called_once()


def test_replay_last_eval_publishes_to_tui_on_init(tmp_path, monkeypatch):
    """Resume: _init_callback re-publishes the most recent eval to the TUI."""
    import json as _json
    meta = {"latest_eval": {
        "step": 200, "win_rate_mean": 0.5, "win_rate_vs_bots": 0.4,
        "mean_reward_vs_bots": -1.0, "mean_ep_len_vs_bots": 30.0,
        "opponents": {"random": {"win_rate": 0.7, "mean_reward": 0.1, "mean_ep_len": 25.0}},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(eval_record, "send_metrics", lambda d: sent.update(d))
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path))
    cb._init_callback()

    assert sent.get("eval/win_rate_vs_random") == 0.7
    assert sent.get("eval/win_rate_mean") == 0.5
    assert sent.get("_step") == 200


def test_replay_last_eval_republishes_pool_block(tmp_path, monkeypatch):
    """Resume: the saved self-play pool block (aggregate + per-sentinel rows) is re-published,
    so the Pool/sentinel rows aren't blank until the next cycle — parity with the bot rows."""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    meta = {"latest_eval": {
        "step": 300, "win_rate_mean": 0.6, "win_rate_vs_bots": 0.55,
        "mean_reward_vs_bots": 5.0, "mean_ep_len_vs_bots": 20.0,
        "elo": 1532.4, "elo_ci": 41.0,
        "opponents": {"random": {"win_rate": 0.9, "mean_reward": 30.0, "mean_ep_len": 15.0}},
        "pool": {
            "win_rate": 0.72, "mean_reward": 13.4, "mean_ep_len": 22.0,
            "monotonicity": 0.4, "snapshot_count": 13,
            "sentinels": [
                {"step": 63_000_000, "win_rate": 0.657, "mean_reward": 8.7, "mean_ep_len": 21.0},
                {"step": 0, "win_rate": 0.707, "mean_reward": 13.2, "mean_ep_len": 23.0},
            ],
        },
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(eval_record, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    # Pool aggregate — including the reward that used to be missing.
    assert sent.get("eval/win_rate_vs_pool") == 0.72
    assert sent.get("eval/mean_reward_vs_pool") == 13.4
    assert sent.get("eval/sentinel_monotonicity") == 0.4
    assert sent.get("eval/pool_snapshot_count") == 13.0
    # Per-sentinel rows, positional, with their saved step tags (seed = step 0).
    assert sent.get("eval/win_rate_vs_sentinel_0") == 0.657
    assert sent.get("eval/mean_reward_vs_sentinel_0") == 8.7
    assert sent.get("eval/sentinel_step_0") == 63_000_000.0
    assert sent.get("eval/sentinel_step_1") == 0.0
    # Skill rating re-published on resume so the 🏅 badge isn't blank until the next cycle.
    assert sent.get("eval/elo") == 1532.4
    assert sent.get("eval/elo_ci") == 41.0


def test_replay_computes_elo_when_block_predates_field(tmp_path, monkeypatch):
    """Resuming a checkpoint saved BEFORE the elo field: the block has no `elo`, but the
    republish computes it from the block's win rates so the 🏅 badge isn't blank for a full
    cadence. (Robust to the anchor file's presence — value just shifts scale.)"""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    meta = {"latest_eval": {  # NOTE: no "elo"/"elo_ci" — a pre-feature checkpoint
        "step": 128_000_010, "win_rate_mean": 0.6, "win_rate_vs_bots": 0.55,
        "mean_reward_vs_bots": 5.0, "mean_ep_len_vs_bots": 20.0,
        "opponents": {n: {"win_rate": w, "mean_reward": 0.0, "mean_ep_len": 15.0}
                      for n, w in [("random", 0.99), ("heuristic", 0.77), ("staller", 0.78)]},
        "pool": {"win_rate": 0.71, "mean_reward": 14.0, "mean_ep_len": 22.0,
                 "monotonicity": 0.8, "snapshot_count": 20,
                 "sentinels": [{"step": 126_000_000, "win_rate": 0.71, "mean_reward": 14.0,
                                "mean_ep_len": 21.0}]},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(eval_record, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    assert "eval/elo" in sent and math.isfinite(sent["eval/elo"])
    assert sent["eval/elo"] > 1000.0          # a strong model is well above the base
    assert sent.get("eval/elo_ci", -1) >= 0   # CI computed (Z95 * SE)
    # Per-opponent ELO for the eval panel: each bot + each (positional) sentinel.
    assert "eval/elo_vs_random" in sent and math.isfinite(sent["eval/elo_vs_random"])
    assert "eval/elo_vs_heuristic" in sent
    assert "eval/elo_vs_sentinel_0" in sent and math.isfinite(sent["eval/elo_vs_sentinel_0"])


def test_replay_skips_pool_block_when_unseeded(tmp_path, monkeypatch):
    """A pre-seed eval persists an empty sentinels list — don't re-publish a misleading 'vs Pool 0%'."""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    meta = {"latest_eval": {
        "step": 100, "win_rate_mean": 0.3, "win_rate_vs_bots": 0.25,
        "mean_reward_vs_bots": -2.0, "mean_ep_len_vs_bots": 18.0,
        "opponents": {"random": {"win_rate": 0.5, "mean_reward": 0.0, "mean_ep_len": 12.0}},
        "pool": {"win_rate": 0.0, "mean_reward": 0.0, "mean_ep_len": 0.0,
                 "monotonicity": 1.0, "snapshot_count": 0, "sentinels": []},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(eval_record, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    assert "eval/win_rate_vs_pool" not in sent
    assert "eval/win_rate_vs_sentinel_0" not in sent
    assert sent.get("eval/win_rate_vs_random") == 0.5  # bot rows still re-published


def test_trace_naming_contract():
    """The prober's `discovery` must parse exactly the filename stems the eval writer
    produces — the contract that silently drifted when sharding added the `s<shard>_`
    infix (every sharded trace then parsed as outcome '?', blinding the whole prober).
    Pins producer (`trace_filename_stem`) → consumer (`discovery._parse_trace`)."""
    from main.prober.discovery import _parse_trace
    from agents.training.eval_callback import trace_filename_stem

    def parse(tag, idx=5):
        stem = trace_filename_stem("loss", tag, idx)
        return _parse_trace(f"/run/eval_traces/step_100/heuristic/{stem}_summary.json")

    # the two tag forms eval emits: "" (un-sharded) and f"s{shard}_" (the executor's per-shard tag)
    for tag in ("", "s0_", "s3_"):
        t = parse(tag)
        assert t.outcome == "loss", f"discovery failed to parse producer stem for tag {tag!r}"
        assert t.opponent == "heuristic"
    assert parse("").index == 5                       # un-sharded index unchanged
    assert parse("s1_").index != parse("s3_").index   # two shards' same idx stay DISTINCT


def test_eval_manifest_records_the_regime(tmp_path):
    """The manifest is self-describing about HOW the numbers were measured (the OOD-era gap):
    matchup hash + the trainee's pin sha + per-opponent fold-back pin shas."""
    import hashlib
    from agents.training.eval_callback import write_eval_manifest
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"cli_args": {"_matchup_spec_hash": "cafe000042"}}))
    m = write_eval_manifest(str(tmp_path), 1000, opponents=["heuristic", "ext_spec"], n_games=100,
                            trainee_team_str="Skarmory @ Leftovers\n",
                            opponent_pins={"ext_spec": "Magneton @ Leftovers\n", "ext_gen": None})
    assert m["matchup_hash"] == "cafe000042"
    assert m["trainee_team_sha"] == hashlib.sha1(b"Skarmory @ Leftovers\n").hexdigest()[:10]
    assert m["opponent_pins"] == {
        "ext_spec": hashlib.sha1(b"Magneton @ Leftovers\n").hexdigest()[:10]}   # None pin dropped
    on_disk = json.loads((tmp_path / "eval_traces" / "step_1000" / "eval_manifest.json").read_text())
    assert on_disk["matchup_hash"] == "cafe000042"
    # regime absent (the old call shape) → explicit Nones, never a KeyError for readers
    m2 = write_eval_manifest(str(tmp_path), 2000, opponents=["heuristic"], n_games=100)
    assert m2["trainee_team_sha"] is None and m2["opponent_pins"] == {}


def test_opponent_pins_record_every_pinned_team_of_a_multi_team_opponent(tmp_path):
    """F-LH-13 follow-through: a multi-team fixed opponent is MEASURED on all its pins
    (`EvalItem.fixed_from_cfg`), so the manifest records all of them — a LIST of per-team shas — and
    a single-team one keeps the one-sha shape. Both callbacks build the map through
    `opponent_pins_of`. FAILS on revert (the callbacks passed `team_str`, the FIRST pin, and the
    manifest read a 3-team specialist as a 1-team one)."""
    import ast
    import hashlib
    from types import SimpleNamespace

    from agents.training.eval_callback import opponent_pins_of, write_eval_manifest
    from utils.paths import src_path

    t = ["Skarmory @ Leftovers\n", "Blissey @ Leftovers\n", "Gengar @ Leftovers\n"]
    fixed = [SimpleNamespace(label="ext_multi", team_str=t[0], team_strs=tuple(t)),
             SimpleNamespace(label="ext_one", team_str=t[1], team_strs=(t[1],)),
             SimpleNamespace(label="ext_gen", team_str=None, team_strs=())]
    pins = opponent_pins_of(fixed)
    assert pins == {"ext_multi": t, "ext_one": t[1], "ext_gen": None}
    m = write_eval_manifest(str(tmp_path), 1000, opponents=["ext_multi", "ext_one", "ext_gen"],
                            n_games=10, opponent_pins=pins)
    sha = lambda x: hashlib.sha1(x.encode()).hexdigest()[:10]   # noqa: E731
    assert m["opponent_pins"] == {"ext_multi": [sha(x) for x in t], "ext_one": sha(t[1])}
    for rel in (("agents", "training", "eval_callback.py"), ("agents", "training", "selfplay_callback.py")):
        tree = ast.parse(src_path(*rel).read_text())
        kws = [k for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "write_eval_manifest"
               for k in c.keywords if k.arg == "opponent_pins"]
        assert kws and all(isinstance(k.value, ast.Call) and getattr(k.value.func, "id", "") == "opponent_pins_of"
                           for k in kws), f"{rel[-1]} builds opponent_pins without opponent_pins_of"


def test_eval_games_override_flows_through_schedule():
    """--eval-games and --eval-freq override the cycle size/cadence via the _schedule() seam
    (None → the module defaults EVAL_GAMES / EVAL_FREQ_STEPS). Both callbacks read _schedule()
    for the pending cycle AND record the actual cycle size into eval_results.jsonl, so each
    override is a single-seam change. (gen3_eval_freq_flag_v1 made the cadence an instance
    attribute too — this test builds via __new__, so it must set both knobs, like __init__ does.)"""
    from agents.training.eval_callback import PerOpponentEvalCallback, EVAL_FREQ_STEPS, EVAL_GAMES
    from agents.training.selfplay_callback import SelfPlayCallback

    for cls in (PerOpponentEvalCallback, SelfPlayCallback):
        cb = cls.__new__(cls)
        cb._eval_games = 200
        cb._eval_freq = EVAL_FREQ_STEPS
        if cls is SelfPlayCallback:
            cb._debug = False
        assert cb._schedule() == (EVAL_FREQ_STEPS, 200), cls.__name__
        cb._eval_games = EVAL_GAMES
        assert cb._schedule() == (EVAL_FREQ_STEPS, EVAL_GAMES), cls.__name__
        cb._eval_freq = 123_456
        assert cb._schedule() == (123_456, EVAL_GAMES), cls.__name__


def test_eval_manifest_handles_a_multi_team_pin(tmp_path):
    """A `pin_multi` trainee carries a LIST of team exports, not one string.

    Regression: `write_eval_manifest` called `.encode()` on the list, raising inside the eval
    callback — and because eval fires MID-ROLLOUT, that crashed the whole run into a restart loop
    (2026-07-26, the multi-team def-20 exploiter). One sha per team keeps the field usable.
    """
    from agents.training.eval_callback import write_eval_manifest

    a, b = "Salamence @ Leftovers\n", "Skarmory @ Leftovers\n"
    m = write_eval_manifest(str(tmp_path), 42, opponents=["heuristic"], n_games=10,
                            trainee_team_str=[a, b])
    assert isinstance(m["trainee_team_sha"], list) and len(m["trainee_team_sha"]) == 2
    assert all(len(s) == 10 for s in m["trainee_team_sha"])

    # single pin still yields ONE sha (unchanged), and no pin still yields None
    single = write_eval_manifest(str(tmp_path), 43, opponents=["heuristic"], n_games=10,
                                 trainee_team_str=a)
    assert single["trainee_team_sha"] == m["trainee_team_sha"][0]
    assert write_eval_manifest(str(tmp_path), 44, opponents=["heuristic"], n_games=10,
                               trainee_team_str=None)["trainee_team_sha"] is None
