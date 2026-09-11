"""Win-probability label plumbing (`--win-prob-mode read_only|shaping`).

The auxiliary win-probability head (`WinProbHead`, `features_extractor.py`) is supervised by the
Monte-Carlo episode OUTCOME (win=1 / loss=0) — but unlike the belief labels (privileged info known
each step), the outcome is a FUTURE quantity, only known when the battle ends. So it cannot ride as a
per-step obs key the way belief labels do. This callback bridges that gap with **zero new buffer
internals** by reusing the proven obs-dict-label storage path:

1. `Gen3Env` declares two TRAINING-ONLY obs keys — `win_target` and `win_mask` ([1] float32 each) —
   and emits PLACEHOLDER zeros every step (`emit_win_target`). The rollout buffer therefore stores +
   shuffles them automatically, exactly like the belief labels.
2. The terminal OUTCOME of each episode is captured during collection (the env puts it in
   ``info["win_outcome"]`` at the done step; `MaskableAgentWrapper`): this callback records it for the
   SYNC path (in `_on_step`, using ``rollout_buffer.pos``), and the async collector records it inline
   (it owns the per-env buffer row) — both into a shared ``model._win_terminal_scratch`` [n_steps, n_envs].
3. After collection (`_on_rollout_end`, before `train()`), the per-episode outcome is propagated
   backward to every step of its episode and the buffer's ``observations["win_target"]`` /
   ``["win_mask"]`` placeholders are OVERWRITTEN with the MC label + a known-mask. `train()` then reads
   ``rollout_data.observations["win_target"]`` / ``["win_mask"]`` (shuffle-aligned for free) and folds
   the BCE at ``win_prob_coef``.

Transitions in the trailing IN-PROGRESS episode (no terminal yet within the buffer) get ``win_mask=0``
and are excluded from the loss — never trained toward a fabricated label. The win label is **undiscounted**
(γ_win = 1): every step of an episode carries that episode's actual outcome, so P(win|s) is directly
"the probability this state leads to a win" — the interpretable quantity the head exists to give.

Only added to the callback list when the win-prob head is on, so a default run pays nothing.

**`--win-prob-lambda` (`gen3_winprob_lambda_v1`) replaces that copied bit with a λ-RETURN**, still
written into the same two obs keys and read by the same BCE — see `lambda_return_targets` below for
the recursion, the buffer-boundary convention and why the values it blends are the RECORDED ones.
At the default `1.0` the recursion is skipped entirely and everything above is unchanged.

**`--win-prob-rollout-target` (`gen3_winprob_rollout_target_v1`) replaces the copied bit on a
SUBSAMPLE of the buffer's own states with an R-ROLLOUT Monte-Carlo win fraction** — new bits bought
by playing the state forward, rather than the network's own later estimates moved backward. It runs
between the back-fill and the λ recursion, writes the same two obs keys, and the labelled rows
become ANCHORS the recursion passes through (`_apply_rollout`, and the precedence note there).
At the default `0.0` the whole path is skipped and everything above is unchanged. The selection,
the cost identity and the ecology approximation live in
:mod:`agents.training.win_prob_rollout`.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
from agents.training.win_prob_rollout import (BANKED_CONTINUATION_DECISIONS, DEFAULT_ROLLOUT_R,
                                              ROLLOUT_MODES, ROLLOUT_OFF, apply_rollout_labels,
                                              eligible_mask, index_records, n_states_for,
                                              rollout_metrics, select_rows)


class WinProbLabelCallback(BaseCallback):
    """Captures per-episode win/loss outcomes during rollout collection and back-fills the rollout
    buffer's ``win_target`` / ``win_mask`` obs keys with the Monte-Carlo label before ``train()``."""

    def __init__(self, records_dir=None, impl: str = "rust") -> None:
        """``records_dir`` is the `cf_records` ring (``<run>/cf_records``) the rollout labeller
        resolves a sampled state's reconstruction record out of; ``None`` (the default, and every
        run without ``--cf-records``) disables the rollout-target path with a message rather than
        silently labelling nothing. ``impl`` is the sim transport the continuations play on — the
        same ``--use-bridge`` the training battles use, so a label is measured on the engine the
        run is being trained on."""
        super().__init__()
        self._records_dir = records_dir
        self._impl = str(impl or "rust")
        self._rollout_calls = 0
        self._said_no_records = False

    def _scratch(self) -> np.ndarray:
        n_steps = self.model.n_steps
        n_envs = self.model.n_envs
        scr = getattr(self.model, "_win_terminal_scratch", None)
        if scr is None or scr.shape != (n_steps, n_envs):
            scr = np.full((n_steps, n_envs), np.nan, dtype=np.float32)
            self.model._win_terminal_scratch = scr
        return scr

    def _handle_scratch(self):
        """``(keys, turns)`` — the per-row RECONSTRUCTION HANDLE, [n_steps, n_envs] each.

        ``keys`` is an object array of ``"<pid>_<battle_tag>"`` (or None) and ``turns`` an int array
        (-1 = none). Captured at DECISION time: the env publishes the turn it was ASKED at, not the
        turn the step landed on, because the buffer row holds the observation the decision was made
        from. Allocated only when the rollout target is on — a default run never touches this."""
        n_steps = self.model.n_steps
        n_envs = self.model.n_envs
        keys = getattr(self.model, "_win_handle_keys", None)
        turns = getattr(self.model, "_win_handle_turns", None)
        if keys is None or keys.shape != (n_steps, n_envs):
            keys = np.empty((n_steps, n_envs), dtype=object)
            turns = np.full((n_steps, n_envs), -1, dtype=np.int64)
            self.model._win_handle_keys = keys
            self.model._win_handle_turns = turns
        return keys, turns

    def _on_rollout_start(self) -> None:
        # Fresh scratch each rollout: NaN = "no terminal captured at this (step, env)". The async
        # collector writes into this same array inline (it runs after on_rollout_start).
        self._scratch().fill(np.nan)
        # A STALE `win_prob/lambda_*` family would read as a live measurement of the rollout that
        # did not produce it. Cleared here, published only by a recursion that actually ran.
        self.model._win_prob_lambda_metrics = None
        # 🚨 THE PRE-λ TERMINAL BIT, cleared for the same reason and published by the same
        # recursion. `_apply_lambda` OVERWRITES `win_target` in place, so once it has run the
        # episode's 0/1 outcome is gone from the buffer and no downstream reader can recover it
        # (the λ-return carries it at weight λ^d for a `d` nothing records). The value sidecar is
        # the reader that needs it, and a stale array from the PREVIOUS rollout would label this
        # rollout's states with another rollout's outcomes — a plausible wrong number, which is
        # the one thing this subsystem refuses to emit.
        self.model._win_prob_terminal_outcome = None
        # The same staleness rule for the rollout family and the handles it selects from: a handle
        # left over from the PREVIOUS rollout names an episode this buffer never contained.
        self.model._win_prob_rollout_metrics = None
        if self._rollout_on():
            keys, turns = self._handle_scratch()
            keys.fill(None)
            turns.fill(-1)

    def _rollout_on(self) -> bool:
        """True when `--win-prob-rollout-target` is live for this run. Checked before ANY of the
        handle capture allocates or runs, so an unflagged run pays exactly nothing."""
        return float(getattr(self.model, "win_prob_rollout_target", 0.0) or 0.0) > ROLLOUT_OFF

    def _on_step(self) -> bool:
        # SYNC capture only — the async collector records terminals inline (it owns the per-env buffer
        # row; the wave-batched on_step here can't recover each env's row). on_step fires BEFORE
        # rollout_buffer.add() in the stock loop, so buffer.pos is the row about to be written for THIS
        # transition, and infos/dones in locals describe it.
        if getattr(self.model, "_async_rollout", False):
            return True
        buf = self.locals.get("rollout_buffer")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if buf is None or infos is None or dones is None:
            return True
        t = int(buf.pos)
        if t >= self.model.n_steps:
            return True  # defensive: never index past the buffer
        scratch = self._scratch()
        keys = turns = None
        if self._rollout_on():
            keys, turns = self._handle_scratch()
        for env_i, done in enumerate(dones):
            info = infos[env_i]
            if info is None:
                continue
            if done and "win_outcome" in info:
                scratch[t, env_i] = float(info["win_outcome"])
            if keys is not None and info.get("wp_handle"):
                keys[t, env_i] = str(info["wp_handle"])
                turns[t, env_i] = int(info.get("wp_turn", -1))
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations
        if not isinstance(obs, dict) or "win_target" not in obs or "win_mask" not in obs:
            return  # win-prob head on but env not emitting the keys (config mismatch) — skip, don't crash
        scratch = getattr(self.model, "_win_terminal_scratch", None)
        if scratch is None:
            return
        es = buf.episode_starts                       # [n_steps, n_envs] (1.0 = start of a new episode)
        wt = obs["win_target"]                        # [n_steps, n_envs, 1]
        wm = obs["win_mask"]                          # [n_steps, n_envs, 1]
        n_steps, n_envs = scratch.shape
        # Backward scan per timestep (vectorised over envs): carry each episode's terminal outcome back
        # to its earlier steps, resetting at an episode boundary. `known` flags steps whose episode
        # finished within the buffer (so the label is real); the trailing in-progress episode stays 0.
        known = np.zeros(n_envs, dtype=bool)
        val = np.zeros(n_envs, dtype=np.float32)
        for t in range(n_steps - 1, -1, -1):
            s = scratch[t]                            # [n_envs]
            has_terminal = ~np.isnan(s)
            known = known | has_terminal
            val = np.where(has_terminal, s, val).astype(np.float32)
            wt[t, :, 0] = val
            wm[t, :, 0] = known.astype(np.float32)
            # An episode start at step t means the EARLIER (t-1) step belongs to a prior episode.
            starts = es[t] >= 0.5
            known = known & ~starts
            val = np.where(starts, 0.0, val).astype(np.float32)
        # 🚨 THE TERMINAL BIT, COPIED BEFORE ANYTHING OVERWRITES IT. Two consumers need the
        # pre-treatment label — the value sidecar (which must not describe a λ-return as an
        # outcome) and the rollout dose meter (`|rollout label − terminal bit|`) — and both of the
        # treatments below write `wt`/`wm` IN PLACE.
        terminal_y = wt[:, :, 0].copy()
        terminal_m = wm[:, :, 0].copy()
        anchor_mask, anchor_value = self._apply_rollout(buf, wt, wm, terminal_y)
        self._apply_lambda(buf, wt, wm, terminal_y, terminal_m,
                           anchor_mask=anchor_mask, anchor_value=anchor_value)

    # ── R-ROLLOUT MC TARGETS (`gen3_winprob_rollout_target_v1`) ────────────────────────────────
    def _rollout_config(self):
        """``(fraction, R, mode)`` for this run, or ``None`` when the labelling must NOT run.

        ``None`` means BIT-IDENTICAL, and there are three ways to get it: the default fraction
        ``0.0``; a critic that is not ``winprob`` (`combination_checks` refuses that argv — this is
        belt-and-braces for a hand-built model, and the reason is the same as λ's: under ``shaped``
        the win-prob BCE is a diagnostic readout, so a new target there re-aims nothing); and no
        ``cf_records`` ring, which is where a sampled state's replayable episode lives. The last one
        ANNOUNCES itself once rather than labelling zero states in silence."""
        frac = float(getattr(self.model, "win_prob_rollout_target", 0.0) or 0.0)
        if frac <= ROLLOUT_OFF:
            return None
        if not is_winprob(getattr(getattr(self.model, "policy", None), "_critic_mode",
                                  CRITIC_DEFAULT)):
            return None
        if not self._records_dir:
            if not self._said_no_records:
                self._said_no_records = True
                print("⚠️  [win_prob_rollout] --win-prob-rollout-target is set but this run has no "
                      "cf_records ring — a sampled state's replayable episode lives there, so NO "
                      "state can be labelled. Pass --cf-records. Said once.", flush=True)
            return None
        mode = str(getattr(self.model, "win_prob_rollout_mode", "replace") or "replace")
        if mode not in ROLLOUT_MODES:
            mode = "replace"
        return frac, int(getattr(self.model, "win_prob_rollout_r", DEFAULT_ROLLOUT_R)
                         or DEFAULT_ROLLOUT_R), mode

    def _apply_rollout(self, buf, wt, wm, terminal_y):
        """Label a seeded subsample of THIS buffer's states with an R-rollout MC win fraction.

        Returns the ``(anchor_mask, anchor_value)`` pair the λ recursion consumes, or
        ``(None, None)`` when the path is off.

        🚨 **PRECEDENCE WITH `--win-prob-lambda`.** A labelled row is an ANCHOR: its target IS the
        rollout fraction and the recursion runs THROUGH it, so the states before it in the same
        episode blend toward a measured win probability instead of toward a copied bit. That is the
        composition that makes the two levers additive rather than rival — λ moves information
        backward, this puts better information there to move. The anchor carries outcome-weight
        1.0 in `lambda_return_targets`' accounting, exactly as a terminal row does, because it IS a
        measurement of that state and not a bootstrap off the network.
        """
        cfg = self._rollout_config()
        if cfg is None:
            return None, None
        frac, R, mode = cfg
        obs = buf.observations
        self._rollout_calls += 1
        keys, turns = self._handle_scratch()
        n_steps, n_envs = wt.shape[0], wt.shape[1]
        has_handle = np.array([[keys[t, e] is not None for e in range(n_envs)]
                               for t in range(n_steps)], dtype=bool)
        el = eligible_mask(wm[:, :, 0], turns, has_handle, obs["action_mask"])
        want = n_states_for(frac, n_steps, n_envs)
        # Seeded from the RUN's seed and the rollout INDEX, so the same argv replays the same
        # sample — and so two different rollouts of one run never draw the same stream.
        rng = np.random.default_rng(
            [int(getattr(self.model, "seed", 0) or 0), int(self._rollout_calls)])
        picks = select_rows(el, buf.episode_starts, want, rng)
        index = index_records(self._records_dir)
        states, kept, missing = [], [], 0
        for (t, e) in picks:
            path = index.get(str(keys[t, e]))
            if path is None:
                missing += 1
                continue
            states.append({"record": path, "turn": int(turns[t, e]),
                           "salt": f"{keys[t, e]}:{int(turns[t, e])}:{self._rollout_calls}"})
            kept.append((t, e))
        labels, stats = ([], {"seconds": 0.0, "arms": 0.0, "arms_capped": 0.0,
                              "arm_decisions": 0.0})
        if states:
            from agents.training.win_prob_rollout_labeller import label_states
            labels, stats = label_states(model=self.model, states=states, rollouts=R,
                                         impl=self._impl)
        anchor_mask, anchor_value, applied = apply_rollout_labels(
            wt, wm, kept, labels, mode, terminal_y)
        opp = obs.get("opp_class")
        opp_cls = (np.asarray(opp)[:, :, 0] if opp is not None
                   else np.zeros((n_steps, n_envs), dtype=np.int64))
        self.model._win_prob_rollout_metrics = rollout_metrics(
            applied=applied, picks=picks, labels=labels, eligible=el, terminal_y=terminal_y,
            new_mask=wm[:, :, 0], opp_class=opp_cls, fraction=frac, rollouts=R, mode=mode,
            seconds=float(stats.get("seconds", 0.0)), arms_played=int(stats.get("arms", 0)),
            arms_capped=int(stats.get("arms_capped", 0)),
            # The MEASURED mean live decisions per continuation when there were any, and the
            # banked `cf_producer` profile when there were none — never 0, which would render the
            # cost meter as "free".
            arm_decisions=float(stats.get("arm_decisions") or BANKED_CONTINUATION_DECISIONS),
            records_missing=missing, n_steps=n_steps, n_envs=n_envs)
        return anchor_mask, anchor_value

    # ── λ-RETURN TARGETS (`gen3_winprob_lambda_v1`) ────────────────────────────────────────────
    def _lambda_config(self):
        """`(λ, truncated_mode)` for this run, or `None` when the recursion must NOT run.

        `None` means BIT-IDENTICAL, and there are exactly two ways to get it:

        * **λ = 1.0** — the default, and the identity: every state's target is its outcome, which
          is what the back-fill above already wrote. Skipping rather than computing it is what
          also keeps the TRUNCATION convention unchanged, which `bootstrap` would otherwise move
          even at λ = 1.
        * **a critic that is not `winprob`** — the buffer's `values` are then a shaped return in
          PopArt units, not a probability, so blending them into a BCE target is a category error.
          `combination_checks` refuses that argv, so this is belt-and-braces for a hand-built model.
        """
        lam = float(getattr(self.model, "win_prob_lambda", LAMBDA_OFF) or LAMBDA_OFF)
        if lam >= LAMBDA_OFF:
            return None
        if not is_winprob(getattr(getattr(self.model, "policy", None), "_critic_mode",
                                  CRITIC_DEFAULT)):
            return None
        return lam, str(getattr(self.model, "win_prob_lambda_truncated", "bootstrap"))

    def _bootstrap_values(self, n_envs):
        """`V(s_T)` per env — one no-grad forward on `model._last_obs`, the SAME post-rollout
        observation SB3's own GAE bootstrap and `winprob_pbrs` use, so the λ-return's boundary and
        the advantage's boundary cannot drift apart. `None` when the forward is unavailable (no
        `_last_obs` yet, or a policy without a critic), which the caller turns into `V(s[last])`."""
        import torch as th
        from stable_baselines3.common.utils import obs_as_tensor
        last_obs = getattr(self.model, "_last_obs", None)
        if last_obs is None:
            return None
        try:
            with th.no_grad():
                v = self.model.policy.predict_values(obs_as_tensor(last_obs, self.model.device))
            return np.asarray(v.detach().cpu().numpy(), dtype=np.float64).reshape(-1)[:n_envs]
        except Exception as exc:                       # pragma: no cover - defensive
            print(f"⚠️  [win_prob_lambda] bootstrap forward failed ({exc}); falling back to "
                  f"V(s_last) at the buffer boundary", flush=True)
            return None

    def _apply_lambda(self, buf, wt, wm, terminal_y=None, terminal_m=None, *,
                      anchor_mask=None, anchor_value=None):
        """Overwrite the just-back-filled `win_target` / `win_mask` with the λ-return, in place.

        ``terminal_y`` / ``terminal_m`` are the PRE-treatment terminal bit and its mask, copied by
        the caller before the rollout labels were written: they are what the sidecar is published
        with and what `lambda_target_shift` is measured against, so the two readings stay about the
        OUTCOME even on a run where the rollout target has already moved some rows. ``anchor_*``
        are the rollout-labelled rows, which the recursion terminates on — see `_apply_rollout`.
        """
        cfg = self._lambda_config()
        if cfg is None:
            return
        lam, mode = cfg
        values = np.asarray(buf.values, dtype=np.float64)
        n_steps, n_envs = values.shape
        last_dones = np.asarray(
            getattr(self.model, "_last_episode_starts", np.zeros(n_envs)),
            dtype=np.float64).reshape(-1)
        last_values = self._bootstrap_values(n_envs)
        if last_values is None or last_values.shape != (n_envs,):
            # `V(s[last])` — the spec's stated fallback. Worse than the true bootstrap (it is the
            # value of s_{T-1}, one step stale) but never a fabricated 0/1 label, and it is
            # REPORTED via `lambda_bootstrap_fallback` rather than absorbed.
            last_values = values[-1].copy()
            fallback = 1.0
        else:
            fallback = 0.0
        y_pre = wt[:, :, 0] if terminal_y is None else np.asarray(terminal_y, dtype=np.float64)
        m_pre = wm[:, :, 0] if terminal_m is None else np.asarray(terminal_m, dtype=np.float64)
        target, weight, new_mask, n_unmasked = lambda_return_targets(
            values, wt[:, :, 0], wm[:, :, 0], np.asarray(buf.episode_starts, dtype=np.float64),
            last_values, last_dones, lam, mode,
            anchor_mask=anchor_mask, anchor_value=anchor_value)
        metrics = lambda_metrics(values, y_pre, m_pre, target, weight, new_mask,
                                 n_unmasked, lam, mode)
        metrics["lambda_bootstrap_fallback"] = fallback
        # 🚨 PUBLISH THE OUTCOME BEFORE DESTROYING IT. `wt`/`wm` still hold the back-filled
        # terminal bit and the "this episode finished inside the buffer" mask; the two lines below
        # replace both with the λ-return and its (possibly UNMASKED) coverage. Copied, never
        # aliased — the very next statement writes through these same arrays.
        self.model._win_prob_terminal_outcome = (np.asarray(y_pre).copy(),
                                                 np.asarray(m_pre).copy())
        wt[:, :, 0] = target.astype(wt.dtype)
        wm[:, :, 0] = new_mask.astype(wm.dtype)
        self.model._win_prob_lambda_metrics = metrics


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE λ-RETURN TARGET (`gen3_winprob_lambda_v1`, 2026-09-09 — the critic ladder's arm 8)
# ──────────────────────────────────────────────────────────────────────────────────────────────

#: How a TRUNCATED episode (still running when the rollout buffer filled) is handled under λ < 1.
#: `bootstrap` gives its states the bootstrap target V(s_T) and UNMASKS them; `mask` leaves them
#: excluded exactly as they are today. Declared so the read can attribute an effect to the target
#: change rather than to the extra rows.
LAMBDA_TRUNCATED_MODES = ("bootstrap", "mask")

#: λ = 1.0 is the identity: every state's target is its episode's outcome `y`, which is precisely
#: what the back-fill above already wrote. The whole recursion is SKIPPED at this value, so an
#: unflagged run (and an explicit `--win-prob-lambda 1.0`) is BIT-identical to the pre-flag tree —
#: including the truncation convention, which `bootstrap` would otherwise change even at λ = 1.
LAMBDA_OFF = 1.0


def lambda_return_targets(values, y, mask, episode_starts, last_values, last_dones,
                          lam, truncated="bootstrap", *, anchor_mask=None, anchor_value=None):
    """The per-state λ-return target for the win-prob BCE. Pure numpy — no torch, no PPO.

    **THE DEFECT THIS EXISTS FOR.** Under `--critic winprob` every state of an episode is trained
    against the SAME terminal bit `y` (`WinProbLabelCallback._on_rollout_end`, above). One bit
    copied to ~30 states is a very noisy regression target, and the head refit
    (`designs/research_state/measurements/winprob_head_refit_2026-09-09/` §6) showed the
    consequence is not a head defect but a TARGET defect: only 10.2 % / 14.4 % of that label's
    variance lies BETWEEN (cycle, opponent) cells, so a learner minimising a proper scoring rule
    shrinks the weak axes toward the marginal and the turn-1 value barely separates opponents
    (spread ratio ~0.1 at turn 1 against ~0.5–0.8 over all states). A λ-return hands an early state
    a blend of the network's OWN later estimates, which move opponent information backward WITHIN
    the episode along a channel with far less noise than the terminal draw.

    **THE RECURSION.** γ = 1 and the clean-world reward stream is terminal-only, so an n-step
    return has no intermediate reward term at all and IS just `V(s[t+n])`. The λ-weighted average
    of those collapses to one backward pass:

        row t ENDS its episode      ⇒  G[t] = y[t]                        (the outcome, exactly)
        otherwise                   ⇒  G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]

    with `V` the RECORDED, pre-update value the collector stored (`rollout_buffer.values`, which
    under this critic IS `sigmoid(win logit) ∈ [0,1]` — `policy._critic_value`). Recorded and not
    re-forwarded on purpose: a target recomputed from the CURRENT weights inside `train()` would
    move under its own gradient across epochs, which is the classic self-referential-target
    divergence; the collection-time values are a fixed point of this rollout by construction.

    Expanding the recursion, a state `d` steps from its terminal carries weight **λ^d on `y`** and
    the rest on later `V`s — returned as `weight` so the read can say how much of the objective is
    still the outcome rather than assume it.

    **THE BUFFER BOUNDARY.** The last row of a TRUNCATED episode has no terminal inside the buffer.
    Its successor is `s_T`, whose value is `last_values` (the same bootstrap SB3's own GAE uses, and
    the same `model._last_obs` forward `winprob_pbrs` takes), so the branch above needs no special
    case: `G = (1−λ)·V(s_T) + λ·V(s_T) = V(s_T)`. Those rows are `mask = 0` TODAY — they have no
    outcome — so `truncated` decides whether they now carry that bootstrap target (`bootstrap`,
    which UNMASKS them and returns the count) or stay excluded (`mask`, byte-identical to today).

    **ANCHORS** (`gen3_winprob_rollout_target_v1`, default `None` ⇒ byte-identical). A row flagged
    in `anchor_mask` takes `anchor_value` as its target and STOPS the recursion there, exactly as a
    row that ends its episode does — with outcome weight 1.0, because an R-rollout Monte-Carlo win
    fraction IS a measurement of that state and not a bootstrap off the network. Earlier rows of the
    same episode then blend toward the measured probability instead of toward the copied terminal
    bit, which is the whole composition between the two levers.

    Args are all `[n_steps, n_envs]` float arrays except `last_values` / `last_dones`, `[n_envs]`.
    `last_dones` is `model._last_episode_starts` — 1.0 where the buffer's final row ENDED its
    episode. Returns `(target, weight, new_mask, n_unmasked)`.
    """
    lam = float(lam)
    if truncated not in LAMBDA_TRUNCATED_MODES:
        raise ValueError(f"truncated must be one of {LAMBDA_TRUNCATED_MODES}, got {truncated!r}")
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    episode_starts = np.asarray(episode_starts, dtype=np.float64)
    n_steps, n_envs = values.shape
    for name, arr in (("y", y), ("mask", mask), ("episode_starts", episode_starts)):
        if arr.shape != (n_steps, n_envs):
            raise ValueError(f"{name} {arr.shape} must match values {values.shape}")
    last_values = np.asarray(last_values, dtype=np.float64).reshape(-1)
    last_dones = np.asarray(last_dones, dtype=np.float64).reshape(-1)
    if last_values.shape != (n_envs,) or last_dones.shape != (n_envs,):
        raise ValueError("last_values and last_dones must both be [n_envs]")

    anchors = (np.zeros((n_steps, n_envs), dtype=bool) if anchor_mask is None
               else np.asarray(anchor_mask, dtype=bool))
    anchor_v = (np.zeros((n_steps, n_envs), dtype=np.float64) if anchor_value is None
                else np.asarray(anchor_value, dtype=np.float64))
    if anchors.shape != (n_steps, n_envs) or anchor_v.shape != (n_steps, n_envs):
        raise ValueError("anchor_mask and anchor_value must both match values' shape")

    target = np.zeros((n_steps, n_envs), dtype=np.float64)
    weight = np.zeros((n_steps, n_envs), dtype=np.float64)
    known = mask >= 0.5
    for t in range(n_steps - 1, -1, -1):
        if t == n_steps - 1:
            ends = last_dones >= 0.5
            v_succ = last_values
            g_succ = last_values          # beyond the buffer: the bootstrap IS the successor return
            w_succ = np.zeros(n_envs)     # ...and it carries no weight on any outcome
        else:
            ends = episode_starts[t + 1] >= 0.5
            v_succ = values[t + 1]
            g_succ = target[t + 1]
            w_succ = weight[t + 1]
        target[t] = np.where(ends, y[t], (1.0 - lam) * v_succ + lam * g_succ)
        weight[t] = np.where(ends, 1.0, lam * w_succ)
        # An ANCHOR overrides both branches: it is a measurement OF this state, so the recursion
        # terminates on it rather than passing through it.
        target[t] = np.where(anchors[t], anchor_v[t], target[t])
        weight[t] = np.where(anchors[t], 1.0, weight[t])

    # THE TRAILING IN-PROGRESS EPISODE, per env — the rows the back-fill left unlabelled BECAUSE
    # the rollout ended, and the only rows a bootstrap may legitimately unmask. Computed rather
    # than inferred from `mask == 0`: an episode that ended without a `win_outcome` in its info is
    # ALSO unlabelled, its rows anchor at `y = 0`, and unmasking those would train the head against
    # a fabricated loss — the exact failure the mask exists to prevent.
    trailing = np.zeros((n_steps, n_envs), dtype=bool)
    in_trail = last_dones < 0.5                  # the final row is mid-episode ⇒ its episode trails
    for t in range(n_steps - 1, -1, -1):
        trailing[t] = in_trail
        in_trail = in_trail & (episode_starts[t] < 0.5)   # row t STARTS it ⇒ t−1 is a prior episode

    n_unmasked = 0
    if truncated == "bootstrap":
        new_mask = np.where(trailing, 1.0, mask)
        n_unmasked = int((trailing & ~known).sum())
    else:
        # `mask`: the unlabelled rows stay excluded, exactly as today.
        new_mask = mask.copy()
    # Rows that are STILL unscored keep the placeholder 0.0 rather than a live-looking number
    # behind a zero mask — the sidecar writes this column, and a plausible wrong value there is
    # worse than the zero a reader already knows to filter on `target_known`.
    scored = new_mask >= 0.5
    target = np.where(scored, target, 0.0)
    weight = np.where(scored, weight, 0.0)
    return target, weight, new_mask, n_unmasked


def _bce(p, t, sel):
    """Mean binary cross-entropy of probabilities `p` against soft targets `t` over `sel` rows."""
    n = float(sel.sum())
    if n <= 0.0:
        return float("nan")
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    per = -(t * np.log(q) + (1.0 - t) * np.log(1.0 - q))
    return float((per * sel).sum() / n)


def lambda_metrics(values, y, old_mask, target, weight, new_mask, n_unmasked, lam, truncated):
    """The `win_prob/lambda_*` TB family: what the recursion DID, priced on this rollout.

    🚨 `lambda_loss` / `lambda_loss_terminal` are scored on the RECORDED, pre-update `V` — the same
    states and the SAME predictions under both targets — so their difference isolates the target
    change and nothing else. They are deliberately not the post-update per-minibatch BCE: that
    would need `y` to survive the buffer's shuffle (it does not — the recursion overwrites
    `win_target` in place) and would confound the target change with a step of learning.
    """
    old = np.asarray(old_mask, dtype=np.float64) >= 0.5
    new = np.asarray(new_mask, dtype=np.float64) >= 0.5
    shift = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    n_old = float(old.sum())
    n_new = float(new.sum())
    return {
        "lambda": float(lam),
        # 1.0 = truncated episodes were bootstrapped INTO the loss; 0.0 = left masked out.
        "lambda_truncated_bootstrap": 1.0 if truncated == "bootstrap" else 0.0,
        "lambda_rows": n_new,
        # How many states the truncation choice ADDED to the objective this rollout. Read it before
        # attributing anything to the λ target itself.
        "lambda_unmasked": float(n_unmasked),
        # Fraction of scored rows whose target is MOSTLY the network's own later estimates
        # (weight on the outcome λ^d < 0.5). 0.0 ⇒ the lever is not reaching far back.
        "lambda_bootstrap_frac": (float((np.asarray(weight)[new] < 0.5).mean()) if n_new > 0
                                  else float("nan")),
        "lambda_weight_mean": (float(np.asarray(weight)[new].mean()) if n_new > 0 else float("nan")),
        # How far the targets moved from the outcome, on the rows that HAVE an outcome.
        "lambda_target_shift": (float(shift[old].mean()) if n_old > 0 else float("nan")),
        "lambda_loss": _bce(values, target, new.astype(np.float64)),
        "lambda_loss_terminal": _bce(values, y, old.astype(np.float64)),
    }
