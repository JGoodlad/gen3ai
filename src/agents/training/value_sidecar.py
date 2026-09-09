"""THE TRAINING-SIDE VALUE SIDECAR (`gen3_value_sidecar_v1`, 2026-09-08).

**WHY THIS EXISTS.** Every instrument this project owns that reads the critic reads EVAL battles —
`main.critic_gate`, `main.ops.critic_read`, the prober's `calibration`, the scaffolding gauge, the
whole win-prob ladder. Nothing has ever logged the value estimate against **its own training
target**. That is a real hole and not a cosmetic one, for three reasons:

1. **The eval slice is not the training distribution.** Eval plays a greedy trainee against a fixed
   roster under an outcome quota that PREFERS LOSSES (`trace_selection`, rule 17). Training plays a
   stochastic policy against a self-play curriculum whose mix moves with `self_play_fraction`. A
   critic can be well calibrated on one and badly calibrated on the other, and until now the second
   was unmeasurable.
2. **The eval read is 8 cycles wide.** A 10M arm evaluates 5 times. The training side produces a
   labelled state every step of every episode, so the same question can be asked per rollout.
3. **The training target is the thing the loss actually minimises.** `--critic winprob` fits
   `V = sigmoid(win logit)` by BCE against `win_target`, back-filled to every state by
   `WinProbLabelCallback`. Calibration against THAT is the objective's own residual; calibration
   against an eval battle is a generalisation question. They are different measurements and only
   the second was ever taken.

**WHAT IT IS.** Once per rollout, at `_on_rollout_end`, a seeded fixed fraction of the buffer's
states is sampled and appended to `<run>/value_sidecar/rows.jsonl` as one JSON object per state.
No forward pass, no extra battle, no env call — every column is read out of arrays the rollout
buffer already holds, so the cost is a slice and a `json.dumps` per sampled row.

🚨 **ORDERING IS LOAD-BEARING AND SILENT IF WRONG.** This callback MUST run after
`WinProbLabelCallback`, whose own `_on_rollout_end` is what overwrites the `win_target` /
`win_mask` placeholders with the Monte-Carlo label. Registered earlier, this reads placeholder
ZEROS and writes a file full of `target: 0.0` that looks exactly like a critic scoring a run of
losses. `main.train.callbacks` appends them in that order and
`value_sidecar_test.py::test_the_sidecar_refuses_a_rollout_whose_labels_were_never_filled` pins it;
at runtime an all-zero mask over a whole rollout is REPORTED (`labels_unfilled`) rather than
written as data, because a plausible wrong number is worse than a gap.

**WHAT IS DERIVED AND WHAT IS READ** — the distinction a consumer needs, so nothing here is
mistaken for a measurement it is not:

| column | provenance |
|---|---|
| `v` | READ — `rollout_buffer.values`, the value PPO actually used |
| `win_logit` | DERIVED — the exact inverse link of `v` under `--critic winprob`, `null` otherwise |
| `target` / `target_known` | READ — the back-filled `win_target` / `win_mask` obs keys |
| `opp_class` | READ — the `opp_class` obs key (bot / pool / stable / exploiter), `null` if absent |
| `turn` | DERIVED — inverted from the observation's linear deadline-clock channel |
| `win_margin` | READ — the `win_margin` obs key, `null` when absent |
| `episode` / `ep_len` / `ep_complete` | DERIVED — folded from `episode_starts` WITHIN this rollout |
| `timeout` | DERIVED — a completed episode whose last turn reached `MAX_TURNS` |

⚠️ **THE OPPONENT IS IDENTIFIED BY CLASS, AND THAT IS ALL THE ENV KNOWS PER STEP.** `opp_class`
is one of four codes — bot / pool / stable / exploiter. The finer identities a reader will reach
for are deliberately NOT here, and each for a reason rather than an omission:

* the **bot's archetype name** and the **pool snapshot's step** are chosen per EPISODE inside
  `MaskableAgentWrapper._select_episode_opponent` and never reach the observation; adding them
  would mean a new obs key per identity, and the class is what every existing consumer
  (`opp_intent/*`, the per-class calibration split) is already keyed on.
* an opponent **ladder rating** does not exist at training time at all. Nothing in the training
  loop carries one: bots are unrated by construction and a snapshot's Elo is a POST-HOC quantity
  that `main.elo` derives from the finished run's `snapshot_ladder/ladder.json`. A column here
  would be null on every row of every run, which is a worse artifact than its absence — so the
  rating is looked up by SNAPSHOT afterwards, never logged per step.

🚨 **`opp_class` RIDES A GATE THIS FILE WIDENED.** It used to be declared only under the
opponent-intent labels, and a win-prob arm normally runs with no intent loss — so before
`gen3_value_sidecar_v1` the by-opponent-class slice was empty on exactly the runs the sidecar
exists for. `gen3_env` now declares it under the win-prob label gate as well. It stays a LABEL key
the network never reads, and `train()`'s one-ahead intent SHIFT is still gated on
`opp_intent_coef > 0` **and** runs after every `_on_rollout_end`, so what is read here is the env's
own per-episode value in both regimes.

⚠️ **`target` IS THE EPISODE'S FINAL OUTCOME, not a per-state quantity.** `WinProbLabelCallback`
back-fills the terminal win/loss to every step of the episode that produced it, so `target` is
constant within an episode and `target_known` marks exactly the episodes that FINISHED inside the
buffer. There is no separate outcome column because it would be the same number.

⚠️ **`win_logit` IS NOT AN INDEPENDENT MEASUREMENT.** Under `--critic winprob` the head IS the
critic, so `v = sigmoid(logit)` exactly and the logit is recoverable by inverting it — but it
carries no information `v` does not. It is written because logit-space is where a calibration
residual is linear, not because a second quantity was observed. Under `--critic shaped` `v` is a
shaped return in raw-reward units and there is no link at all, so the column is `null`: a number
there would be a category error, and this file's whole job is to not produce one.

⚠️ **`ep_len` IS THE LENGTH WITHIN THIS ROLLOUT, NOT THE EPISODE'S LENGTH.** An episode straddling
a rollout boundary has its head in the previous buffer, so its count starts at the buffer edge.
`ep_complete` is false for exactly those, and every length statistic must filter on it. The
alternative — carrying per-env state across rollouts — was rejected because a restart would silently
reset it and produce a short-episode spike that nothing could distinguish from a stall regression.

⚠️ **A 250-TURN TIMEOUT IS NOT A DRAW AND `win_draw` DOES NOT FLAG ONE.** Measured 2026-09-06: the
cap forfeits, Showdown answers `|win|<opponent>`, `won_by` sets `_won = False` and it arrives as a
plain LOSS. `signal/draw_rate` counts TIES only. So `timeout` here is inferred from the deadline
clock reaching `MAX_TURNS` on a completed episode, which is the only signal that separates the two.

**COST.** The default fraction is 1/64 of states. At the production `--n-steps 2048 --n-envs 64`
that is 2048×64 = 131,072 states per rollout, of which ~2,048 are written — about 0.6 MB of JSONL
per rollout. The write is a single buffered `open(..., "a")` per rollout, never per row, and every
column is a numpy slice taken before the loop.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
from agents.observation.constants import (
    CLOCK_OFFSET_IN_GLOBAL, MAX_TURNS, OFFSET_GLOBAL,
)

#: The run-relative directory and file the sidecar writes. One file, append-only, never rotated:
#: a rotated sidecar would need the reader to know the rotation rule, and at the measured ~0.6 MB
#: per rollout a 10M-step arm produces a few hundred MB — bounded enough to keep as one file.
SIDECAR_DIRNAME = "value_sidecar"
SIDECAR_FILENAME = "rows.jsonl"
#: The header line's schema tag. Bump when a column changes MEANING; a purely additive column does
#: not, so an older reader keeps working on a newer file (the `trace_selection` convention).
SIDECAR_SCHEMA = 1

#: Sample one state in this many, by default. 1/64 of a production rollout is ~2,048 rows —
#: enough for a per-turn-bucket calibration table within a single rollout, small enough that the
#: serialization is far below the noise floor of a training step.
DEFAULT_SIDECAR_FRACTION = 1.0 / 64.0

#: The absolute index of the LINEAR remaining-turns channel in the flat observation.
#: `global_env.py` writes three clock channels at `CLOCK_OFFSET_IN_GLOBAL`: [0] log-elapsed,
#: [1] remaining LINEAR, [2] log-remaining. The linear one inverts exactly (`turn = MAX_TURNS *
#: (1 - v)`) where the log ones lose precision at small turn counts, so it is the one read.
#: 🚨 COMPUTED from the layout constants, never hardcoded — the root CLAUDE.md's rule is that an
#: observation index is asked of the code, and this offset has moved before.
CLOCK_LINEAR_INDEX = OFFSET_GLOBAL + CLOCK_OFFSET_IN_GLOBAL + 1


def sidecar_path(run_dir: str) -> str:
    """The one path the writer appends to and the reader reads."""
    return os.path.join(run_dir, SIDECAR_DIRNAME, SIDECAR_FILENAME)


def turn_from_observation(flat_obs: np.ndarray) -> np.ndarray:
    """Battle turn, inverted from the observation's LINEAR deadline-clock channel.

    ``flat_obs`` is ``[..., obs_dim]``; the return has its leading shape.

    ⚠️ **THIS SATURATES AT THE DEADLINE.** `global_env` clamps `remaining` at 0, so every turn at
    or past `MAX_TURNS` encodes identically and reads back as exactly `MAX_TURNS`. That is the
    property `timeout` below is built on, and it is also why a turn statistic must never be read as
    a mean over the tail.

    ⚠️ **A BATTLE TURN IS NOT A DECISION INDEX.** One turn can carry several decisions (a forced
    switch after a KO), so consecutive rows can share a turn value. Bucketing by it is a statement
    about game phase, not about decision count.
    """
    remaining = np.asarray(flat_obs)[..., CLOCK_LINEAR_INDEX]
    return np.clip(MAX_TURNS * (1.0 - remaining), 0.0, MAX_TURNS)


def _logit(p: np.ndarray) -> np.ndarray:
    """The exact inverse of the sigmoid link, clipped off the asymptotes.

    A saturated head produces `v` at exactly 0.0 or 1.0 in float32, whose logit is ±inf — and
    `json.dumps` writes `Infinity`, which is not valid JSON and which `json.loads` in another
    language will refuse. Clipping at float32's resolution keeps the file loadable and keeps a
    saturated forecast readable as "very confident" rather than as a parse error.
    """
    eps = 1e-7
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


class ValueSidecarCallback(BaseCallback):
    """Append a seeded sample of the rollout buffer's value/target pairs to the run's sidecar.

    Args:
        run_dir: the run directory. ``None`` disables the callback entirely (it stays in the list
            and does nothing) — used by the smoke path and by tests.
        fraction: share of buffer states to sample per rollout.
        seed: the sampler's seed. A run's sample is reproducible from (seed, rollout index), so a
            re-read of the same run scores the same states.
        critic_mode: the run's `--critic` value. Decides whether `win_logit` is meaningful.
    """

    def __init__(self, run_dir: "str | None", *,
                 fraction: float = DEFAULT_SIDECAR_FRACTION,
                 seed: int = 0,
                 critic_mode: str = CRITIC_DEFAULT,
                 verbose: int = 0):
        super().__init__(verbose)
        self._run_dir = run_dir
        self._fraction = float(fraction)
        self._seed = int(seed)
        self._critic_mode = str(critic_mode)
        self._rollout_index = 0
        self._header_written = False
        #: Rollouts whose labels were never filled in — reported, never written as data.
        self.labels_unfilled = 0
        #: Rows actually written. Read by the test and by the run's own reporting.
        self.rows_written = 0

    def _on_step(self) -> bool:
        """Nothing per step — and that is the design.

        Every column this sidecar writes is already in the rollout buffer at `_on_rollout_end`, so
        there is no per-step capture to do. `WinProbLabelCallback` needs an `_on_step` because it
        catches a TERMINAL fact (the outcome) that exists only at the done step and would be gone
        by rollout end; nothing here is like that. Keeping the hot path empty is what makes the
        cost a per-rollout slice rather than a per-step branch on 131,072 steps — and it is why
        this callback needs no inline hook in `collect_rollouts_async` the way that one does.
        """
        return True

    # ── the write ────────────────────────────────────────────────────────────────────────────
    def _ensure_header(self) -> None:
        """Write the one-line header describing the file, if this is a fresh sidecar.

        The header is a ROW like any other (`{"kind": "header", ...}`) rather than a separate
        format, so the file stays a plain JSONL a consumer can `for line in f` without a special
        case for line 1. It records the schema, the critic mode and the sampling parameters —
        without the critic mode a reader cannot tell whether `v` is a probability, and without the
        fraction it cannot turn a row count back into a state count.
        """
        path = sidecar_path(self._run_dir)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._header_written = True
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "kind": "header",
            "schema": SIDECAR_SCHEMA,
            "tag": "gen3_value_sidecar_v1",
            "critic_mode": self._critic_mode,
            "v_is_probability": bool(is_winprob(self._critic_mode)),
            "fraction": self._fraction,
            "seed": self._seed,
            "max_turns": int(MAX_TURNS),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "a") as f:
            f.write(json.dumps(header) + "\n")
        self._header_written = True

    def _on_rollout_end(self) -> None:
        if not self._run_dir or self._fraction <= 0.0:
            return
        buf = getattr(self.model, "rollout_buffer", None)
        if buf is None:
            return
        obs = getattr(buf, "observations", None)
        # Same guard shape as WinProbLabelCallback: a config mismatch SKIPS rather than crashing a
        # 24-hour run for the sake of a diagnostic.
        if not isinstance(obs, dict) or "win_target" not in obs or "win_mask" not in obs:
            return

        rollout = self._rollout_index
        self._rollout_index += 1

        wm = np.asarray(obs["win_mask"])[..., 0]          # [n_steps, n_envs]
        # 🚨 A rollout in which NOTHING is labelled means this callback ran before the back-fill,
        # or the win-prob head is off. Either way the rows would all read target=0.0 — a critic
        # apparently facing an unbroken run of losses. Count it and write nothing.
        if not np.any(wm >= 0.5):
            self.labels_unfilled += 1
            return

        wt = np.asarray(obs["win_target"])[..., 0]        # [n_steps, n_envs]
        values = np.asarray(buf.values)                   # [n_steps, n_envs]
        starts = np.asarray(buf.episode_starts)           # [n_steps, n_envs]
        n_steps, n_envs = wm.shape

        flat = np.asarray(obs["observation"])             # [n_steps, n_envs, obs_dim]
        turns = turn_from_observation(flat)               # [n_steps, n_envs]

        # OPTIONAL columns — present only when the env emits them. Absent is null, never zero:
        # `opp_class` 0 is a REAL class (a bot), so defaulting to it would invent a curriculum.
        # Present on every win-prob run since the gate widened (see the module docstring); still
        # optional, because a pre-widening checkpoint resumed under this code has a buffer built
        # from its own saved observation space.
        opp = (np.asarray(obs["opp_class"])[..., 0] if "opp_class" in obs else None)
        margin = (np.asarray(obs["win_margin"])[..., 0] if "win_margin" in obs else None)

        # Per-env episode index within this rollout, and each episode's extent. `cumsum` over the
        # episode-start flags is the same boundary signal the win-target back-fill uses.
        ep_index = np.cumsum(starts >= 0.5, axis=0)       # [n_steps, n_envs]
        ep_len, ep_complete = _episode_extents(starts, wm)

        rows = self._sample(rollout, n_steps, n_envs)
        if rows[0].size == 0:
            return

        self._ensure_header()
        winprob = is_winprob(self._critic_mode)
        ts, es = rows
        v_sel = values[ts, es].astype(np.float64)
        logits = _logit(v_sel) if winprob else None
        base_step = int(getattr(self.model, "num_timesteps", 0))

        out = []
        for k in range(ts.size):
            t, e = int(ts[k]), int(es[k])
            turn = float(turns[t, e])
            complete = bool(ep_complete[t, e])
            row = {
                "step": base_step,
                "rollout": rollout,
                "env": e,
                # Unique WITHIN THE RUN: an episode never spans two rollouts under one id, which
                # is exactly the honesty `ep_complete` records.
                "episode": f"{base_step}:{e}:{int(ep_index[t, e])}",
                "t": t,
                "turn": turn,
                "v": float(v_sel[k]),
                "win_logit": (float(logits[k]) if winprob else None),
                "target": float(wt[t, e]),
                "target_known": bool(wm[t, e] >= 0.5),
                "opp_class": (int(opp[t, e]) if opp is not None else None),
                "win_margin": (float(margin[t, e]) if margin is not None else None),
                "ep_len": int(ep_len[t, e]),
                "ep_complete": complete,
                # A completed episode whose clock reached the cap. NOT `win_draw` — see the module
                # docstring: a timeout arrives as a plain loss and `win_draw` counts ties only.
                "timeout": bool(complete and turn >= MAX_TURNS - 0.5),
            }
            out.append(json.dumps(row, separators=(",", ":")))

        # ONE write per rollout, off the hot path — never a write per row.
        with open(sidecar_path(self._run_dir), "a") as f:
            f.write("\n".join(out) + "\n")
        self.rows_written += len(out)

    # ── the sampler ──────────────────────────────────────────────────────────────────────────
    def _sample(self, rollout: int, n_steps: int, n_envs: int):
        """Which (step, env) cells this rollout writes.

        Seeded on ``(seed, rollout)`` rather than on a single stream, so the sample for rollout N
        does not depend on how many rollouts ran before it. That is what makes a resumed run's
        sample reproducible: a restart re-enters at rollout 0 of the new process, and a
        stream-seeded sampler would silently draw different states than an uninterrupted run.

        ⚠️ **Sampling is UNIFORM over buffer cells, which is not uniform over episodes**: a long
        episode contributes proportionally more rows. That is the right weighting for a
        per-decision calibration question and the wrong one for a per-episode rate, which is why
        the reader clusters its bootstrap by episode rather than by row.
        """
        total = n_steps * n_envs
        n = int(round(total * self._fraction))
        if n <= 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        n = min(n, total)
        rng = np.random.default_rng((self._seed, rollout))
        flat = rng.choice(total, size=n, replace=False)
        flat.sort()
        return (flat // n_envs).astype(np.int64), (flat % n_envs).astype(np.int64)


def _episode_extents(starts: np.ndarray, mask: np.ndarray):
    """Per-cell episode length within this rollout, and whether that episode COMPLETED in it.

    ``starts`` is ``episode_starts`` `[n_steps, n_envs]`; ``mask`` is the filled ``win_mask``,
    whose 1.0 marks exactly the steps whose episode terminated inside the buffer — the same
    definition `WinProbLabelCallback` used to decide the label was real, reused rather than
    re-derived so the two can never disagree about which episodes finished.
    """
    n_steps, n_envs = starts.shape
    length = np.zeros((n_steps, n_envs), dtype=np.int64)
    complete = mask >= 0.5
    for e in range(n_envs):
        bounds = [0] + [t for t in range(1, n_steps) if starts[t, e] >= 0.5] + [n_steps]
        for a, b in zip(bounds[:-1], bounds[1:]):
            length[a:b, e] = b - a
    return length, complete


def read_sidecar(run_dir: str):
    """Load a run's sidecar as ``(header, rows)``.

    🚨 **REFUSES rather than returning an empty result** — a run with no sidecar and a run whose
    sidecar is empty are different facts from a run whose critic was well calibrated, and every
    caller here is a measurement.
    """
    path = sidecar_path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no value sidecar at {path} — this run was trained without one "
            "(--value-sidecar off, or before gen3_value_sidecar_v1). Nothing is read and "
            "nothing is concluded.")
    header, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # a torn final line from a killed run — skip it, never guess it
            if obj.get("kind") == "header":
                header = obj
            else:
                rows.append(obj)
    if header is None:
        raise ValueError(
            f"{path} has no header row, so the critic mode and sampling fraction are UNKNOWN "
            "and `v` cannot be read as a probability or as a shaped return. Refusing.")
    return header, rows
