"""Unit tests for the training-side value sidecar (`gen3_value_sidecar_v1`) — writer and reader."""

import json
import types

import numpy as np
import pytest

from agents.training.value_sidecar import (
    CLOCK_LINEAR_INDEX, DEFAULT_SIDECAR_FRACTION, SIDECAR_SCHEMA, MixedSchemaError,
    ValueSidecarCallback, read_sidecar, read_sidecar_segments, sidecar_path,
    turn_from_observation,
)
from agents.observation.constants import MAX_TURNS

# Wide enough to hold the clock channel, and NOT the real observation width. The sidecar reads
# exactly one index out of the flat vector, so pinning the true 2501 here would couple this test
# to a dim the root CLAUDE.md says never to hardcode — and would fail on every layout change for a
# reason that has nothing to do with the sidecar. The INDEX is imported; only the padding is local.
_OBS_W = CLOCK_LINEAR_INDEX + 1


class _FakeBuffer:
    pass


def _model(n_steps=8, n_envs=4, *, filled=True, turns=None, opp=True, num_timesteps=1_000_000,
           values=None, targets=None):
    """A rollout buffer shaped exactly like the real one at `_on_rollout_end`.

    Every array is the shape SB3 hands the callback — `[n_steps, n_envs]` for the scalars and
    `[n_steps, n_envs, *shape]` for the obs dict — so a shape change in the trainer breaks this
    test rather than silently producing rows about the wrong axis.
    """
    rng = np.random.default_rng(0)
    obs = {
        "observation": np.zeros((n_steps, n_envs, _OBS_W), dtype=np.float32),
        "win_target": np.zeros((n_steps, n_envs, 1), dtype=np.float32),
        "win_mask": np.zeros((n_steps, n_envs, 1), dtype=np.float32),
        "win_margin": np.zeros((n_steps, n_envs, 1), dtype=np.float32),
    }
    if opp:
        obs["opp_class"] = np.ones((n_steps, n_envs, 1), dtype=np.int64)  # POOL
    if filled:
        obs["win_mask"][:] = 1.0
        obs["win_target"][:] = (targets if targets is not None
                                else rng.integers(0, 2, (n_steps, n_envs, 1)).astype(np.float32))
    # The deadline clock: turn t encoded as the LINEAR remaining channel.
    t_grid = (np.tile(np.arange(n_steps, dtype=np.float32)[:, None], (1, n_envs))
              if turns is None else np.asarray(turns, dtype=np.float32))
    obs["observation"][:, :, CLOCK_LINEAR_INDEX] = 1.0 - (t_grid / MAX_TURNS)

    buf = _FakeBuffer()
    buf.observations = obs
    buf.values = (np.asarray(values, dtype=np.float32) if values is not None
                  else rng.uniform(0, 1, (n_steps, n_envs)).astype(np.float32))
    buf.episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf.episode_starts[0, :] = 1.0
    m = types.SimpleNamespace(rollout_buffer=buf, num_timesteps=num_timesteps)
    return m


def _run(cb, model):
    cb.model = model
    cb._on_rollout_end()


def _rows(run):
    return read_sidecar(str(run))[1]


# ── the observation → turn inversion ───────────────────────────────────────────────────────────
def test_the_turn_is_inverted_from_the_deadline_clock_exactly():
    obs = np.zeros((3, _OBS_W), dtype=np.float32)
    for i, turn in enumerate((0.0, 37.0, 249.0)):
        obs[i, CLOCK_LINEAR_INDEX] = 1.0 - (turn / MAX_TURNS)
    got = turn_from_observation(obs)
    assert got == pytest.approx([0.0, 37.0, 249.0], abs=1e-2)


def test_the_clock_SATURATES_at_the_cap_rather_than_running_past_it():
    """`global_env` clamps `remaining` at 0, so every turn at or past the cap reads as the cap.

    That saturation is exactly what the `timeout` column is built on, and a test that did not pin
    it would let a layout change turn every long episode into a silent non-timeout.
    """
    obs = np.zeros((1, _OBS_W), dtype=np.float32)
    obs[0, CLOCK_LINEAR_INDEX] = 0.0          # remaining == 0
    assert float(turn_from_observation(obs)[0]) == pytest.approx(MAX_TURNS)


# ── the writer ─────────────────────────────────────────────────────────────────────────────────
def test_the_row_schema_is_complete_and_the_header_declares_the_currency(tmp_path):
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model())
    header, rows = read_sidecar(str(tmp_path))

    assert header["schema"] == SIDECAR_SCHEMA and header["kind"] == "header"
    # 🚨 Without this the reader cannot tell a probability from a shaped return.
    assert header["critic_mode"] == "winprob" and header["v_is_probability"] is True
    assert header["fraction"] == 1.0 and header["max_turns"] == MAX_TURNS

    assert len(rows) == 8 * 4
    # 🚨 `outcome` / `outcome_known` are their OWN columns and not a copy of `target` for the sake
    # of it: under `--win-prob-lambda < 1` the λ recursion overwrites `win_target` in place, and
    # the terminal bit exists nowhere else in the file afterwards.
    expected = {"step", "rollout", "env", "episode", "t", "turn", "v", "win_logit", "target",
                "target_known", "outcome", "outcome_known", "opp_class", "win_margin",
                "ep_len", "ep_complete", "timeout"}
    for r in rows:
        assert set(r) == expected, set(r) ^ expected
        assert 0.0 <= r["v"] <= 1.0
        assert r["opp_class"] == 1               # POOL, as the fake env emitted
        assert r["ep_complete"] is True and r["ep_len"] == 8
        # At λ = 1.0 the target IS the outcome, so the two columns agree exactly.
        assert r["outcome"] == r["target"] and r["outcome_known"] is True
    # The header states the λ regime even when the flag is off — its ABSENCE was the defect.
    assert header["win_prob_lambda"] == 1.0
    assert header["win_prob_lambda_truncated"] == "bootstrap"
    assert header["resumed"] is False


def test_the_win_logit_is_the_EXACT_inverse_link_under_winprob_and_NULL_under_shaped(tmp_path):
    """`v = sigmoid(logit)`, so the logit inverts exactly — and under `shaped` there is no link."""
    vals = np.full((4, 2), 0.75, dtype=np.float32)
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(4, 2, values=vals))
    for r in _rows(tmp_path):
        assert r["win_logit"] == pytest.approx(np.log(0.75 / 0.25), abs=1e-5)

    shaped = tmp_path / "shaped"
    cb2 = ValueSidecarCallback(str(shaped), fraction=1.0, critic_mode="shaped")
    _run(cb2, _model(4, 2, values=vals))
    header, rows = read_sidecar(str(shaped))
    assert header["v_is_probability"] is False
    # A number here would be a category error: under `shaped` v is a shaped return, not a
    # probability, so there is no link to invert.
    assert all(r["win_logit"] is None for r in rows)


def test_a_saturated_value_stays_valid_JSON(tmp_path):
    """logit(1.0) is +inf and `json.dumps` writes `Infinity`, which is not valid JSON."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(2, 2, values=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)))
    raw = open(sidecar_path(str(tmp_path))).read()
    assert "Infinity" not in raw and "NaN" not in raw
    for line in raw.strip().splitlines():
        json.loads(line)                       # a strict parser must accept every line


def test_the_sampling_fraction_is_honoured_and_the_seed_is_DETERMINISTIC(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d, seed in ((a, 3), (b, 3), (c, 4)):
        cb = ValueSidecarCallback(str(d), fraction=0.25, seed=seed, critic_mode="winprob")
        _run(cb, _model(16, 8))                # 128 cells → 32 rows
    ra, rb, rc = _rows(a), _rows(b), _rows(c)
    assert len(ra) == 32
    key = lambda rs: sorted((r["t"], r["env"]) for r in rs)    # noqa: E731
    assert key(ra) == key(rb), "same seed must draw the same states"
    assert key(ra) != key(rc), "a different seed must draw different states"


def test_the_sample_for_a_rollout_does_not_depend_on_HOW_MANY_ran_before_it(tmp_path):
    """Seeded on (seed, rollout index), not as one stream — so a RESTART re-draws the same states.

    A stream-seeded sampler would silently score different states after every launcher restart,
    and a three-hour restart interval means that is most of a run.
    """
    fresh = ValueSidecarCallback(str(tmp_path / "fresh"), fraction=0.5, critic_mode="winprob")
    fresh._rollout_index = 2                    # as if resuming at rollout 2
    _run(fresh, _model(8, 4))

    cont = ValueSidecarCallback(str(tmp_path / "cont"), fraction=0.5, critic_mode="winprob")
    for _ in range(3):                          # rollouts 0, 1, then 2
        _run(cont, _model(8, 4))

    third = [r for r in _rows(tmp_path / "cont") if r["rollout"] == 2]
    assert sorted((r["t"], r["env"]) for r in third) == \
           sorted((r["t"], r["env"]) for r in _rows(tmp_path / "fresh"))


def test_the_sidecar_refuses_a_rollout_whose_labels_were_never_filled(tmp_path):
    """🚨 THE CALLBACK-ORDER GUARD.

    Registered before `WinProbLabelCallback`, the sidecar reads placeholder ZEROS and would write a
    file full of `target: 0.0` — indistinguishable from a critic facing an unbroken run of losses.
    That is the failure this whole guard exists for: a plausible wrong number, not a crash.
    """
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(filled=False))
    assert cb.labels_unfilled == 1 and cb.rows_written == 0
    with pytest.raises(FileNotFoundError):
        read_sidecar(str(tmp_path))            # nothing written at all, not an empty file


def test_a_missing_opp_class_key_is_NULL_and_never_the_bot_code(tmp_path):
    """`opp_class` 0 is a REAL class, so defaulting to it would invent a curriculum."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(4, 2, opp=False))
    assert all(r["opp_class"] is None for r in _rows(tmp_path))


def test_a_timeout_is_inferred_from_the_CLOCK_not_from_a_draw_flag(tmp_path):
    """🚨 A 250-turn timeout arrives as a plain LOSS; `win_draw` counts ties only.

    So the only signal that separates a timeout from a loss is the deadline clock reaching the cap.
    """
    turns = np.array([[10.0, MAX_TURNS], [11.0, MAX_TURNS]], dtype=np.float32)
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(2, 2, turns=turns))
    rows = _rows(tmp_path)
    assert {r["timeout"] for r in rows if r["env"] == 1} == {True}
    assert {r["timeout"] for r in rows if r["env"] == 0} == {False}


def test_an_incomplete_trailing_episode_is_MARKED_not_dropped(tmp_path):
    """An episode straddling the rollout boundary keeps its rows and loses its length claim."""
    m = _model(6, 1, filled=True)
    m.rollout_buffer.observations["win_mask"][4:] = 0.0    # the tail is still in progress
    m.rollout_buffer.episode_starts[4, 0] = 1.0
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, m)
    rows = _rows(tmp_path)
    assert len(rows) == 6
    tail = [r for r in rows if r["t"] >= 4]
    assert all(r["ep_complete"] is False and r["target_known"] is False for r in tail)
    assert all(r["ep_complete"] is True for r in rows if r["t"] < 4)


def test_the_write_is_ONE_append_per_rollout_not_one_per_row(tmp_path, monkeypatch):
    """Cost gate: the hot path pays a single `open` per rollout regardless of row count."""
    import agents.training.value_sidecar as mod

    opens = []
    real_open = mod.open if hasattr(mod, "open") else open
    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: (opens.append(a[0]), real_open(*a, **k))[1])
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(32, 16))                   # 512 rows
    assert cb.rows_written == 512
    # one header write + one row-block write; NEVER 512.
    assert len([o for o in opens if str(o).endswith("rows.jsonl")]) == 2


def test_a_disabled_sidecar_writes_NOTHING(tmp_path):
    for cb in (ValueSidecarCallback(None, critic_mode="winprob"),
               ValueSidecarCallback(str(tmp_path), fraction=0.0, critic_mode="winprob")):
        _run(cb, _model())
        assert cb.rows_written == 0
    assert not (tmp_path / "value_sidecar").exists()


def test_the_default_fraction_is_the_documented_one():
    assert DEFAULT_SIDECAR_FRACTION == 1.0 / 64.0


# ── the registration + ordering contract ───────────────────────────────────────────────────────
def test_the_sidecar_is_ON_BY_DEFAULT_for_a_winprob_run():
    from main.train.callbacks import _value_sidecar_on
    from main.train.parser import build_parser

    p = build_parser()
    assert _value_sidecar_on(p.parse_args(["--critic", "winprob"])) is True
    assert _value_sidecar_on(p.parse_args([])) is False           # shaped stays byte-identical
    assert _value_sidecar_on(p.parse_args(["--value-sidecar", "on"])) is True
    assert _value_sidecar_on(
        p.parse_args(["--critic", "winprob", "--value-sidecar", "off"])) is False


def test_the_sidecar_is_registered_AFTER_the_win_prob_backfill():
    """🚨 SB3 runs `_on_rollout_end` in list order, and reading before the back-fill is SILENT.

    Pinned by reading the source: the two registrations must appear in this order in the builder,
    because there is no runtime error to catch the other one — only a file of plausible zeros.
    """
    import inspect

    import main.train.callbacks as mod

    src = inspect.getsource(mod.build_callbacks)
    # The constructor now takes arguments (gen3_winprob_rollout_target_v1 threads the cf_records
    # ring and the bridge impl into it), so the probe is the NAME, not a bare call.
    i_win = src.index("WinProbLabelCallback(")
    i_side = src.index("ValueSidecarCallback(")
    assert i_win < i_side, (
        "ValueSidecarCallback must be appended AFTER WinProbLabelCallback — before it, the "
        "win_target/win_mask placeholders are still zeros and the sidecar records a critic "
        "apparently facing an unbroken run of losses")


# ── the reader ─────────────────────────────────────────────────────────────────────────────────
def test_the_reader_RECOVERS_a_known_calibration(tmp_path):
    """A synthetic sidecar with a KNOWN calibration must read back as that calibration.

    Two forecasters on the same states: a perfectly calibrated one (reliability ~0, positive
    resolution, positive skill) and a base-rate one (also reliability ~0, but resolution ~0 and
    skill ~0). 🚨 That pair is the point — it is the case where reliability cannot tell the two
    apart and only `critic_resolution` can.
    """
    from main.ops.value_sidecar_read import murphy

    rng = np.random.default_rng(11)
    p = rng.uniform(0.0, 1.0, 20_000)
    y = (rng.uniform(0.0, 1.0, 20_000) < p).astype(float)

    good = murphy(list(p), list(y))
    assert good["reliability"] == pytest.approx(0.0, abs=0.005)
    assert good["resolution"] > 0.05
    assert good["skill"] > 0.25
    assert good["decomp_residual"] == pytest.approx(0.0, abs=0.01)

    base = murphy([float(y.mean())] * len(y), list(y))
    assert base["reliability"] == pytest.approx(0.0, abs=0.005)   # indistinguishable here…
    assert base["resolution"] == pytest.approx(0.0, abs=0.005)    # …and separated only here
    assert base["skill"] == pytest.approx(0.0, abs=0.01)

    # A deliberately OPTIMISTIC forecaster must read as optimistic, with the sign convention
    # POSITIVE = optimistic.
    opt = murphy(list(np.clip(p + 0.2, 0, 1)), list(y))
    assert opt["mean_v"] > opt["mean_target"]
    assert opt["reliability"] > good["reliability"]


def test_the_reader_slices_and_clusters_by_EPISODE(tmp_path):
    from main.ops.value_sidecar_read import _cell, _slices

    rng = np.random.default_rng(3)
    rows = []
    for ep in range(60):
        target = float(ep % 2)
        for t in range(30):
            rows.append({"episode": f"e{ep}", "step": 1_000_000 * (ep // 30), "turn": float(t),
                         "v": float(np.clip(target * 0.8 + 0.1 + rng.normal(0, 0.05), 0, 1)),
                         "target": target, "target_known": True,
                         "opp_class": ep % 2, "ep_complete": True})
    cell = _cell(rows, draws=200, seed=1)
    assert cell["n"] == 1800 and cell["skill"] > 0.5
    lo, hi = cell["mean_error_ci"]
    assert lo is not None and lo < cell["mean_error"] < hi

    s = _slices(rows, draws=200, seed=1)
    assert set(s["by_outcome"]) == {"won", "lost"}
    assert set(s["by_opponent_class"]) == {"bot", "pool"}
    assert s["by_turn"] and s["by_step"]


def test_a_cell_under_the_floor_is_MARKED_never_reported():
    from agents.training.stats import MIN_CELL_N
    from main.ops.value_sidecar_read import _cell

    rows = [{"episode": "e0", "v": 0.5, "target": 1.0, "target_known": True, "turn": 1.0,
             "step": 0}] * (MIN_CELL_N - 1)
    assert _cell(rows, draws=50, seed=1)["under_floor"] is True


def test_the_reader_REFUSES_a_headerless_sidecar(tmp_path):
    """Without the header the critic mode is unknown, so `v` cannot be read as anything."""
    d = tmp_path / "value_sidecar"
    d.mkdir()
    (d / "rows.jsonl").write_text(json.dumps({"v": 0.5, "target": 1.0}) + "\n")
    with pytest.raises(ValueError, match="no header"):
        read_sidecar(str(tmp_path))


def test_the_reader_REFUSES_a_run_with_no_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="no value sidecar"):
        read_sidecar(str(tmp_path))


def test_a_torn_final_line_is_SKIPPED_not_guessed(tmp_path):
    """A run killed mid-append leaves a partial line; it must not take down the read."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(4, 2))
    with open(sidecar_path(str(tmp_path)), "a") as f:
        f.write('{"step": 1, "v": 0.5, "targ')
    header, rows = read_sidecar(str(tmp_path))
    assert header is not None and len(rows) == 8


# ── THE λ REGIME: what `target` means, and where the OUTCOME lives ─────────────────────────────
def _lambda_model(n_steps=8, n_envs=4, *, lam=0.9, stash=True, unmask_tail=False):
    """A rollout AFTER `WinProbLabelCallback._apply_lambda` has run.

    That callback overwrites `win_target` / `win_mask` in place with the λ-return and its
    (possibly wider) mask, then publishes the pre-overwrite pair on the model. This fixture
    reproduces both halves, because the sidecar's whole job here is to read the second one.
    """
    m = _model(n_steps, n_envs)
    obs = m.rollout_buffer.observations
    y = obs["win_target"][:, :, 0].copy()          # the terminal bit, as back-filled
    mask = obs["win_mask"][:, :, 0].copy()
    if unmask_tail:
        # `--win-prob-lambda-truncated bootstrap`: the trailing in-progress episode gains a
        # λ target and keeps NO outcome. Here the last two steps are that tail.
        mask[-2:, :] = 0.0                         # never finished → no outcome
        obs["win_mask"][-2:, :, 0] = 1.0           # …but the λ target covers them
    # The λ-return: a soft blend, deliberately different from every terminal bit.
    obs["win_target"][:, :, 0] = 0.5 * y + 0.25
    m.win_prob_lambda = lam
    m.win_prob_lambda_truncated = "bootstrap"
    if stash:
        m._win_prob_terminal_outcome = (y, mask)
    return m, y, mask


def test_under_LAMBDA_the_target_is_the_RETURN_and_the_OUTCOME_is_its_own_column(tmp_path):
    """🚨 The defect in one assertion: `target` and `outcome` are DIFFERENT numbers here."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    model, y, _ = _lambda_model()
    _run(cb, model)
    header, rows = read_sidecar(str(tmp_path))
    assert header["win_prob_lambda"] == 0.9
    for r in rows:
        assert r["target"] == pytest.approx(0.5 * r["outcome"] + 0.25)
        assert r["outcome"] in (0.0, 1.0)          # the terminal bit, not a blend
        assert r["outcome"] != r["target"]


def test_a_BOOTSTRAP_UNMASKED_row_has_a_TARGET_but_NO_OUTCOME(tmp_path):
    """The row sets differ on purpose — a trailing episode never produced an outcome at all."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    model, _, _ = _lambda_model(unmask_tail=True)
    _run(cb, model)
    rows = _rows(tmp_path)
    tail = [r for r in rows if r["t"] >= 6]
    assert tail and all(r["target_known"] and not r["outcome_known"] for r in tail)
    # 🚨 …and the episode is NOT reported complete. Reading completeness off the λ-widened
    # `win_mask` would call a straddling episode finished.
    assert all(r["ep_complete"] is False for r in tail)
    head = [r for r in rows if r["t"] < 6]
    assert head and all(r["outcome_known"] for r in head)


def test_with_NO_stash_under_LAMBDA_the_outcome_is_NULL_rather_than_GUESSED(tmp_path):
    """The λ-return holds the outcome at weight λ^d for a `d` nothing records. So: nothing."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    model, _, _ = _lambda_model(stash=False)
    _run(cb, model)
    for r in _rows(tmp_path):
        assert r["outcome"] is None and r["outcome_known"] is False


def test_a_STALE_stash_from_a_DIFFERENT_shaped_rollout_is_REFUSED_not_used(tmp_path):
    """A wrong-shape stash would label these states with another rollout's outcomes."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    model, _, _ = _lambda_model(stash=False)
    model._win_prob_terminal_outcome = (np.zeros((3, 3)), np.ones((3, 3)))
    _run(cb, model)
    assert all(r["outcome"] is None for r in _rows(tmp_path))


def test_the_callback_CLEARS_the_stash_at_ROLLOUT_START(tmp_path):
    """🚨 A stash surviving into the next rollout is the plausible wrong number, exactly."""
    from agents.training.win_prob_callback import WinProbLabelCallback

    cb = WinProbLabelCallback()
    cb.model = types.SimpleNamespace(n_steps=4, n_envs=2, _win_terminal_scratch=None,
                                     _win_prob_terminal_outcome=("stale", "stale"))
    cb._on_rollout_start()
    assert cb.model._win_prob_terminal_outcome is None


# ── the header is per WRITER SESSION, and a mixed file is refused by ROW INDEX ─────────────────
def test_a_RESUME_writes_its_OWN_header_rather_than_riding_the_first_one(tmp_path):
    """🚨 The silent defect: a resumed run's rows used to sit under the first process's header."""
    for _ in range(2):                             # two processes, one file
        cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
        _run(cb, _model(4, 2))
    segments = read_sidecar_segments(str(tmp_path))
    assert len(segments) == 2
    assert segments[0]["header"]["resumed"] is False
    assert segments[1]["header"]["resumed"] is True
    assert segments[0]["row_index"] == 0 and segments[1]["row_index"] == 8


def test_the_header_is_written_ONCE_per_process_not_once_per_ROLLOUT(tmp_path):
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    for _ in range(3):
        _run(cb, _model(4, 2))
    assert len(read_sidecar_segments(str(tmp_path))) == 1


def test_read_sidecar_REFUSES_a_file_whose_target_CHANGES_MEANING_mid_way(tmp_path):
    """A resume across the `gen3_winprob_lambda_v1` boundary — outcomes, then λ-returns."""
    cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    _run(cb, _model(4, 2))                         # λ = 1.0: targets are the outcome
    cb2 = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
    model, _, _ = _lambda_model(4, 2)
    _run(cb2, model)                               # λ = 0.9: targets are λ-returns
    with pytest.raises(MixedSchemaError) as e:
        read_sidecar(str(tmp_path))
    assert e.value.change_at == 8                  # the exact row the meaning changes at
    assert len(e.value.segments) == 2


def test_a_RESUME_at_the_SAME_lambda_is_NOT_refused(tmp_path):
    """🚨 Not a false alarm: a plain restart must still read as one file."""
    for _ in range(2):
        cb = ValueSidecarCallback(str(tmp_path), fraction=1.0, critic_mode="winprob")
        _run(cb, _model(4, 2))
    header, rows = read_sidecar(str(tmp_path))
    assert len(rows) == 16 and header["resumed"] is False
