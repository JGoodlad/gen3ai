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
| `outcome` / `outcome_known` | READ — the PRE-λ terminal bit and its mask; identical to `target` at λ = 1.0, `null` when λ < 1 and the writer's stash is missing |
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

⚠️ **`target` IS THE EPISODE'S FINAL OUTCOME, not a per-state quantity — UNLESS `--win-prob-lambda`
IS LIVE.** `WinProbLabelCallback` back-fills the terminal win/loss to every step of the episode that
produced it, so `target` is constant within an episode and `target_known` marks exactly the episodes
that FINISHED inside the buffer. There is no separate outcome column because it would be the same
number. 🚨 Under `--win-prob-lambda < 1` (`gen3_winprob_lambda_v1`) that same callback then
OVERWRITES `win_target` with the λ-RETURN — a per-state probability that varies within the episode —
and, under the default `bootstrap` truncation branch, sets `win_mask = 1` on the trailing in-progress
episode as well. This callback runs immediately after it and therefore reads the λ-return. The
header's **`win_prob_lambda`** field says which quantity a file holds; a reader that assumes the
outcome on a λ file is measuring the critic against a moving target and will not know it.

🚨 **SO THE OUTCOME IS WRITTEN AS ITS OWN COLUMN, and it has to be, because it is otherwise gone.**
`_apply_lambda` overwrites `win_target` IN PLACE; the λ-return carries the outcome at weight `λ^d`
for a distance `d` nothing records, and `win_margin` is a per-turn MATERIAL margin (a by-product of
Φ_mat) whose sign is a material lead, not a win. Nothing downstream can invert either. So that
callback publishes the pre-overwrite `(y, mask)` on the model and this one reads it into
`outcome` / `outcome_known` — which is what lets the reader score a λ file against BOTH quantities
and label each. When the stash is absent under λ < 1 the columns are `null`, never inferred.

🚨 **`ep_complete` FOLLOWS THE TERMINAL MASK, NOT `target_known`.** Under the default `bootstrap`
truncation the λ recursion UNMASKS the trailing in-progress episode, so `win_mask` stops meaning
"this episode finished inside the buffer". Read through it, a straddling episode would read
COMPLETE and every length statistic that filters on `ep_complete` would quietly include a
truncated head.

🚨 **THE HEADER IS WRITTEN ONCE PER PROCESS, NOT ONCE PER FILE.** It used to be skipped whenever
the file was non-empty, so a RESUME appended its rows under the first process's header — a run
resumed across a flag change then held one header saying `win_prob_lambda: 1.0` above a tail of
λ-returns, with nothing on disk to say so. A header per writer session makes the change visible at
the exact row it happens, which is what `read_sidecar_segments` refuses by index.

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
#: v2 (gen3_winprob_lambda_v1): the header gained `win_prob_lambda` / `win_prob_lambda_truncated`,
#: and with them the `target` column's meaning became HEADER-DEPENDENT — the episode outcome at
#: λ = 1.0 (every v1 file), the λ-return below it. That is a meaning change under this file's own
#: rule, so the tag moves: a v1 reader meeting a v2 file can refuse rather than quietly average a
#: quantity it thinks is a 0/1 outcome. A v2 file at λ = 1.0 is byte-identical to a v1 one apart
#: from the two header fields.
#: v3 (gen3_winprob_rollout_target_v1): the header gained `win_prob_rollout_target` /
#: `win_prob_rollout_r` / `win_prob_rollout_mode`, and with them `target` acquired a THIRD meaning —
#: on a SUBSAMPLE of the rows it is now an R-rollout Monte-Carlo win fraction measured by replaying
#: that state and playing it forward, not a copied bit and not a λ-return. That is a meaning change
#: under this file's own rule (and a nastier one than λ's, because it applies to SOME rows and not
#: all), so the tag moves again. A v3 file at target 0.0 is byte-identical to a v2 one apart from
#: the three header fields.
SIDECAR_SCHEMA = 3

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
        """Write this WRITER SESSION's header row, once per process.

        The header is a ROW like any other (`{"kind": "header", ...}`) rather than a separate
        format, so the file stays a plain JSONL a consumer can `for line in f` without a special
        case for line 1. It records the schema, the critic mode and the sampling parameters —
        without the critic mode a reader cannot tell whether `v` is a probability, and without the
        fraction it cannot turn a row count back into a state count.

        🚨 **ONCE PER PROCESS, NOT ONCE PER FILE — and the difference is a whole class of silent
        defect.** This used to skip whenever the file was non-empty, so a RESUME appended its rows
        under the FIRST process's header. A run resumed across a flag change then held one header
        saying `win_prob_lambda: 1.0` above a tail of rows whose `target` is a λ-RETURN, and
        nothing on disk said so: every consumer would pool a 0/1 outcome with a soft return and
        report the average as a calibration. A header per session makes the change VISIBLE at the
        exact row it happens, which is what lets `read_sidecar_segments` refuse it by index.
        A restart appends a few hundred bytes; the cost is not a consideration.
        """
        if self._header_written:
            return
        path = sidecar_path(self._run_dir)
        resumed = os.path.exists(path) and os.path.getsize(path) > 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "kind": "header",
            "schema": SIDECAR_SCHEMA,
            "tag": "gen3_value_sidecar_v1",
            "critic_mode": self._critic_mode,
            # gen3_winprob_lambda_v1: WHICH quantity the `target` column holds. 1.0 = the episode's
            # terminal outcome (every file before this flag); below 1.0 = the λ-return, which
            # varies within an episode and covers the truncated rows the bootstrap branch unmasks.
            # Written from the live model so it cannot disagree with what actually ran.
            "win_prob_lambda": float(getattr(self.model, "win_prob_lambda", 1.0) or 1.0),
            "win_prob_lambda_truncated": str(
                getattr(self.model, "win_prob_lambda_truncated", "bootstrap")),
            # gen3_winprob_rollout_target_v1: whether SOME of the `target` column is a MEASURED
            # win fraction rather than the copied bit / λ-return the two fields above describe.
            # 0.0 = none of it. The per-row flag is deliberately NOT written: the sidecar samples
            # its own 1/64 of the buffer and the labeller samples its own, so the two rarely
            # intersect, and a column that is "usually" one quantity is exactly what this header
            # exists to warn about.
            "win_prob_rollout_target": float(
                getattr(self.model, "win_prob_rollout_target", 0.0) or 0.0),
            "win_prob_rollout_r": int(getattr(self.model, "win_prob_rollout_r", 8) or 8),
            "win_prob_rollout_mode": str(
                getattr(self.model, "win_prob_rollout_mode", "replace") or "replace"),
            "v_is_probability": bool(is_winprob(self._critic_mode)),
            "fraction": self._fraction,
            "seed": self._seed,
            "max_turns": int(MAX_TURNS),
            # This SEGMENT's start. `resumed` is True for every header after the first, so a
            # reader can say "the file changed here" rather than "the file is inconsistent".
            "started_at": datetime.now(timezone.utc).isoformat(),
            "resumed": bool(resumed),
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

        # 🚨 THE OUTCOME, WHICH UNDER λ < 1 IS NO LONGER `target`. `WinProbLabelCallback` publishes
        # the PRE-λ terminal bit and its mask before overwriting `win_target` in place; at λ = 1.0
        # the recursion is skipped whole and `target` IS the outcome, so the two arrays are the
        # ones already read. When λ < 1 and the stash is absent or the wrong shape (a hand-built
        # model, a callback-order defect), the columns are `null` — the λ-return carries the
        # outcome at weight λ^d for a `d` nothing records, so there is nothing to recover and a
        # guess would be exactly the plausible wrong number this file exists to avoid.
        y_arr, y_mask = self._terminal_outcome(wt, wm, (n_steps, n_envs))

        # Per-env episode index within this rollout, and each episode's extent. `cumsum` over the
        # episode-start flags is the same boundary signal the win-target back-fill uses.
        ep_index = np.cumsum(starts >= 0.5, axis=0)       # [n_steps, n_envs]
        # 🚨 COMPLETENESS IS THE TERMINAL MASK, NEVER THE TARGET MASK. Under
        # `--win-prob-lambda-truncated bootstrap` the λ recursion UNMASKS the trailing in-progress
        # episode, so `win_mask` stops meaning "this episode finished inside the buffer" — read
        # through it, `ep_complete` would call a straddling episode complete and every length
        # statistic that filters on it would silently include a truncated head.
        ep_len, ep_complete = _episode_extents(starts, y_mask if y_mask is not None else wm)

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
                # The episode's 0/1 OUTCOME, always — identical to `target` at λ = 1.0 and the
                # only route to it below 1.0. `null` when unrecoverable; never inferred.
                "outcome": (float(y_arr[t, e]) if y_arr is not None else None),
                "outcome_known": (bool(y_mask[t, e] >= 0.5) if y_mask is not None else False),
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

    # ── the outcome, under either λ regime ───────────────────────────────────────────────────
    def _terminal_outcome(self, wt, wm, shape):
        """``(outcome, outcome_mask)`` as ``[n_steps, n_envs]`` arrays, or ``(None, None)``.

        Three cases, and only the first two produce a number:

        * **λ = 1.0** (every file before `gen3_winprob_lambda_v1`, and every unflagged run since):
          the back-fill wrote the terminal bit into `win_target` and nothing overwrote it, so the
          target IS the outcome and the arrays are returned as they are.
        * **λ < 1.0 with the stash present**: `WinProbLabelCallback._apply_lambda` published the
          pre-overwrite `(y, mask)` on the model immediately before replacing them.
        * **λ < 1.0 with no stash**: ``(None, None)``. The λ-return is
          ``G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]``, so the outcome enters row `t` at weight `λ^d`
          for a distance `d` this file does not record, and `win_margin` is a per-turn MATERIAL
          margin (a by-product of Φ_mat), not an outcome. Nothing here can be inverted.
        """
        lam = float(getattr(self.model, "win_prob_lambda", 1.0) or 1.0)
        stash = getattr(self.model, "_win_prob_terminal_outcome", None)
        if stash is not None:
            y, mask = stash
            y, mask = np.asarray(y), np.asarray(mask)
            if y.shape == shape and mask.shape == shape:
                return y, mask
            return None, None
        if lam >= 1.0:
            return wt, wm
        return None, None

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


#: The header fields that decide WHAT `target` IS. Two files (or two segments of one file) that
#: disagree on any of them hold two different quantities under one column name, and pooling them
#: is the defect this tuple exists to make detectable. `critic_mode` is here because it decides
#: whether `v` is a probability at all; the two λ fields because they decide whether `target` is
#: the terminal outcome or a λ-return over the collector's own values.
TARGET_IDENTITY_FIELDS = ("schema", "critic_mode", "win_prob_lambda", "win_prob_lambda_truncated",
                          "win_prob_rollout_target", "win_prob_rollout_r",
                          "win_prob_rollout_mode")


def target_identity(header) -> dict:
    """The subset of a header that says what `target` MEANS, with the pre-λ defaults filled in.

    A schema-1 header has neither λ field. Both defaults are the values the flag's OFF position
    writes, so a v1 file and a v2 file at λ = 1.0 compare EQUAL here — which is exactly right:
    `gen3_winprob_lambda_v1` states that a v2 file at λ = 1.0 is byte-identical to a v1 one apart
    from the two header fields, so refusing to compare them would be a false alarm.
    """
    h = header or {}
    return {
        "schema": int(h.get("schema", 1)),
        "critic_mode": str(h.get("critic_mode")),
        "win_prob_lambda": float(h.get("win_prob_lambda", 1.0) or 1.0),
        "win_prob_lambda_truncated": str(h.get("win_prob_lambda_truncated", "bootstrap")),
        "win_prob_rollout_target": float(h.get("win_prob_rollout_target", 0.0) or 0.0),
        "win_prob_rollout_r": int(h.get("win_prob_rollout_r", 8) or 8),
        "win_prob_rollout_mode": str(h.get("win_prob_rollout_mode", "replace") or "replace"),
    }


#: The schema versions whose `target` column holds the SAME quantity whenever the three λ/critic
#: fields agree — i.e. between which a difference in the version NUMBER alone is not a difference
#: in meaning. `SIDECAR_SCHEMA`'s own comment states this for the 1↔2 pair: a v2 file at λ = 1.0 is
#: byte-identical to a v1 one apart from the two header fields, so refusing to compare them would
#: be a false alarm on every arm before arm 8.
#: 🚨 **A NEW SCHEMA IS NOT ADDED HERE BY DEFAULT.** Leaving it out means a v3 file refuses to be
#: pooled with a v2 one until somebody states, here, why the two columns are the same quantity —
#: which is the direction this subsystem errs in everywhere else.
#: 3 JOINS the set for the same measured reason 2 did: a v3 file at `win_prob_rollout_target` 0.0
#: is byte-identical to a v2 one apart from the three header fields, and the QUANTITY_FIELDS below
#: carry the actual distinction — so refusing on the NUMBER would be a false alarm on every arm
#: before arm 10, exactly as it would have been on every arm before arm 8.
SCHEMA_EQUIVALENCE = frozenset({1, 2, 3})

#: The header fields that decide what QUANTITY `target` holds, independent of the version number.
#: 🚨 **`win_prob_rollout_weight` IS DELIBERATELY ABSENT, and so is a schema bump for it.** The
#: rule for this tuple is "does it change what the `target` COLUMN HOLDS?", not "is it a new flag".
#: `gen3_winprob_rollout_weight_v1` changes how much a row COUNTS IN THE LOSS; it changes no row's
#: target, and this file records targets, not loss weights — a v3 file written at weight 64 holds
#: the same per-row quantity as one written at weight 1.0, row for row. Adding it here would refuse
#: to pool two files that are genuinely poolable, which is the false alarm `SCHEMA_EQUIVALENCE`
#: exists to name. (Were the sidecar ever to record a per-row WEIGHT column, that column's identity
#: would need its own field here — this reasoning is about `target` and nothing else.)
QUANTITY_FIELDS = ("critic_mode", "win_prob_lambda", "win_prob_lambda_truncated",
                   "win_prob_rollout_target", "win_prob_rollout_r", "win_prob_rollout_mode")


def same_quantity(a_header, b_header) -> bool:
    """Do these two headers' `target` columns hold the SAME quantity?

    The three λ/critic fields must agree, AND both schemas must be in the declared
    :data:`SCHEMA_EQUIVALENCE` set (or be equal). That is the split the rest of this module rests
    on: a version NUMBER that moved without the meaning moving is not a reason to refuse, and a
    version number nobody has reasoned about is.
    """
    a, b = target_identity(a_header), target_identity(b_header)
    if any(a[k] != b[k] for k in QUANTITY_FIELDS):
        return False
    if a["schema"] == b["schema"]:
        return True
    return a["schema"] in SCHEMA_EQUIVALENCE and b["schema"] in SCHEMA_EQUIVALENCE


def target_is_outcome(header) -> bool:
    """Does this file's `target` column hold the episode's terminal 0/1 outcome?

    True at λ = 1.0 — including every schema-1 file, which predates the flag — and False below it,
    where the column holds a λ-return that varies WITHIN an episode.
    """
    idn = target_identity(header)
    # A rollout-labelled file's column is the outcome on MOST rows and a measured win fraction on
    # the sampled ones. "Mostly the outcome" is not the outcome, and a reader that treats it as one
    # is doing the exact pooling this module refuses everywhere else.
    return idn["win_prob_lambda"] >= 1.0 and idn["win_prob_rollout_target"] <= 0.0


def describe_target(header) -> str:
    """What `target` IS, in words, for THIS file. One sentence, for a report header.

    🚨 The whole defect this exists for is that the column has one NAME and two MEANINGS. A reader
    that prints a calibration table without saying which one it scored has produced a number
    nobody can interpret and everybody can pool.
    """
    idn = target_identity(header)
    roll = ""
    if idn["win_prob_rollout_target"] > 0.0:
        roll = (f" — EXCEPT on a seeded ~{idn['win_prob_rollout_target']:.4g} subsample of the "
                f"buffer, where it is an R = {idn['win_prob_rollout_r']} Monte-Carlo win fraction "
                f"measured by replaying that state and playing it forward "
                f"(`{idn['win_prob_rollout_mode']}`)")
    if idn["win_prob_lambda"] >= 1.0:
        return ("the episode's TERMINAL 0/1 OUTCOME, back-filled to every state of the episode "
                "that produced it (constant within an episode)" + roll)
    return (f"the λ-RETURN, λ = {idn['win_prob_lambda']:g} — a per-state SOFT probability that "
            f"varies within an episode, blending the outcome with the collector's own recorded "
            f"V(s); truncated episodes are handled `{idn['win_prob_lambda_truncated']}`"
            + (", so trailing in-progress rows are BOOTSTRAPPED and carry no outcome at all"
               if idn["win_prob_lambda_truncated"] == "bootstrap" else "") + roll)


class MixedSchemaError(ValueError):
    """A single `rows.jsonl` whose `target` column changes MEANING part-way through.

    Carries the segments so a caller can name both headers and the exact row index of the change
    rather than reporting "inconsistent file".
    """

    def __init__(self, message, segments, change_at):
        super().__init__(message)
        self.segments = segments
        self.change_at = change_at


def read_sidecar_segments(run_dir: str):
    """Load a run's sidecar as a list of ``{"header", "rows", "row_index"}`` SEGMENTS.

    One segment per writer session — the sidecar writes a header row per process, so a resumed run
    contributes one segment per restart. ``row_index`` is the segment's first row's index in the
    POOLED row list, which is what a refusal quotes.

    🚨 **REFUSES rather than returning an empty result** — a run with no sidecar and a run whose
    sidecar is empty are different facts from a run whose critic was well calibrated, and every
    caller here is a measurement.

    ⚠️ Rows appearing BEFORE any header (a file written by a pre-`gen3_value_sidecar_v1` writer, or
    a truncated head) are refused rather than attributed to the first header that follows them:
    guessing which config produced them is the exact move this whole subsystem forbids.
    """
    path = sidecar_path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no value sidecar at {path} — this run was trained without one "
            "(--value-sidecar off, or before gen3_value_sidecar_v1). Nothing is read and "
            "nothing is concluded.")
    segments: list = []
    n_rows = 0
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
                segments.append({"header": obj, "rows": [], "row_index": n_rows})
            else:
                if not segments:
                    raise ValueError(
                        f"{path} begins with DATA rows and no header row precedes them, so the "
                        "critic mode and the λ regime that produced them are UNKNOWN. Refusing.")
                segments[-1]["rows"].append(obj)
                n_rows += 1
    if not segments:
        raise ValueError(
            f"{path} has no header row, so the critic mode and sampling fraction are UNKNOWN "
            "and `v` cannot be read as a probability or as a shaped return. Refusing.")
    return segments


def read_sidecar(run_dir: str):
    """Load a run's sidecar as ``(header, rows)`` — the pooled read, which REFUSES a mixed file.

    🚨 **POOLING IS ONLY LEGAL WHEN EVERY SEGMENT MEANS THE SAME THING BY `target`.** A run resumed
    across the `gen3_winprob_lambda_v1` boundary holds terminal 0/1 outcomes in its head and
    λ-returns in its tail under one column name; averaging the two produces a number that is
    neither. ``MixedSchemaError`` names both headers and the row index of the change.
    """
    segments = read_sidecar_segments(run_dir)
    first = segments[0]["header"]
    for seg in segments[1:]:
        if not same_quantity(first, seg["header"]):
            raise MixedSchemaError(
                f"{sidecar_path(run_dir)} changes what `target` MEANS at row "
                f"{seg['row_index']:,}", segments, seg["row_index"])
    rows = [r for seg in segments for r in seg["rows"]]
    return segments[0]["header"], rows
