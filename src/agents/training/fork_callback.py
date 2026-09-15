"""`ForkArmCallback` — the training-loop hook for `--fork-fraction` (`gen3_fork_v1`).

Runs ONCE per rollout, in ``_on_rollout_end``, between `compute_returns_and_advantage` and
``train()``. Six steps, and each one is a separate module so this file is the SEQUENCE and nothing
else:

1. **eligibility** — `fork_arm.eligible_mask` over the buffer's own planes.
2. **a candidate POOL** — `fork_arm.candidate_pool`, uniform over episode slices.
3. **the contested test** — one batched policy forward over the pool, then
   `fork_arm.contested_threshold` / `contested_select`. This is the paired-refit's selector: a
   move round, turn in band, >= 3 legal actions, top-2 masked-logit gap under the shard's own
   ``--fork-contested-gap`` quantile.
4. **the branches** — `fork_driver.play_forks` fans out to children that replay each fork's
   episode to its turn and play the branches to a terminal under common random numbers.
5. **the rows** — `fork_driver.assemble` scores them with the LIVE policy and builds the block;
   `fork_buffer` turns each branch into ordinary PPO rows with its own GAE/lambda-return.
6. **the meters** — `fork_arm.fork_metrics`, published under ``fork/``.

🚨 **ORDER WITH `WinProbLabelCallback`.** This callback MUST be registered AFTER it. Step 1 reads
``win_mask`` to mean "this episode terminated inside the buffer", and that plane is a PLACEHOLDER
of zeros until the win-prob callback back-fills it — so a fork callback that ran first would find
nothing eligible, every rollout, in silence. `build_callbacks` appends in that order and
:meth:`_on_rollout_end` says so out loud the first time it finds an empty mask.

**Nothing here raises into the training loop.** Every failure path leaves the buffer exactly as
collection made it. A fabricated transition in the PPO objective is the one outcome this subsystem
must never produce, and "no forks this rollout" is always available as the answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
from agents.training.fork_arm import (
    DEFAULT_BRANCHES, DEFAULT_CONTESTED_ABSV, DEFAULT_CONTESTED_GAP,
    DEFAULT_CRN, DEFAULT_MAX_PER_BATTLE, FORK_OFF, MAX_FORKS_PER_ROLLOUT, branch_actions,
    candidate_pool, contested_select, contested_threshold, eligible_mask, fork_metrics,
    forks_per_battle, n_forks_for, pool_size_for, top2_gaps,
)

#: The row budget, as a multiple of the collected buffer. A branch row is a FULL observation
#: (2,501 float32 plus every label key), so an unbounded injection is an unbounded memory bill on a
#: box that is also carrying a GPU run. 1.0 = the arm may at most DOUBLE the buffer; at
#: ``--fork-fraction 0.02`` with 3 branches the expected injection is ~1.5x the collected rows, so
#: this binds and drops whole forks — which is why `fork/dropped_forks` is published rather than
#: left to be inferred from a row count.
ROW_BUDGET_MULTIPLE = 1.0


class ForkArmCallback(BaseCallback):
    """Fork contested decisions and inject the branches' transitions into this rollout's buffer."""

    def __init__(self, records_dir=None, impl: str = "rust") -> None:
        """``records_dir`` is the `cf_records` ring (``<run>/cf_records``) a fork's replayable
        episode is resolved out of; ``None`` (every run without ``--cf-records``) disables the arm
        with a message rather than silently forking nothing. ``impl`` is the sim transport the
        branches play on — the same ``--use-bridge`` the training battles use, so a branch is
        measured on the engine the run is trained on."""
        super().__init__()
        self._records_dir = records_dir
        self._impl = str(impl or "rust")
        self._calls = 0
        self._said_no_records = False
        self._said_no_mask = False
        self._said_unfillable = False
        self._disabled = False
        # gen3_fork_v1 — the MEASURED mean rows a fork contributes, carried across rollouts so the
        # NEXT one can stop asking for branches the row budget would drop. None until a pass has
        # measured one; see `_budgeted_forks`.
        self._rows_per_fork = None

    # ── configuration ────────────────────────────────────────────────────────────────────────
    def _fraction(self) -> float:
        return float(getattr(self.model, "fork_fraction", 0.0) or 0.0)

    def _on(self) -> bool:
        """True when the arm is live. Checked before ANY allocation, so an unflagged run pays
        exactly nothing — the discipline every OFF-is-bit-identical flag in this tree keeps."""
        if self._disabled or self._fraction() <= FORK_OFF:
            return False
        if not is_winprob(getattr(getattr(self.model, "policy", None), "_critic_mode",
                                  CRITIC_DEFAULT)):
            # `combination_checks` refuses that argv; this is belt-and-braces for a hand-built
            # model, and the reason is `fork_buffer.branch_rewards`: outside `winprob` a branch's
            # reward stream is not reconstructible without the env that computed it.
            return False
        if not self._records_dir:
            if not self._said_no_records:
                self._said_no_records = True
                print("⚠️  [fork_arm] --fork-fraction is set but this run has no cf_records ring — "
                      "a fork's replayable episode lives there, so NO decision can be forked. "
                      "Pass --cf-records. Said once.", flush=True)
            return False
        return True

    def _on_rollout_start(self) -> None:
        # A STALE `fork/*` family would read as a live measurement of the rollout that did not
        # produce it. Cleared here, published only by a pass that actually ran.
        self.model._fork_metrics = None

    # ── the pass ─────────────────────────────────────────────────────────────────────────────
    def _on_rollout_end(self) -> None:
        if not self._on():
            return
        buf = self.model.rollout_buffer
        if not hasattr(buf, "add_fork_rows"):
            self._disable("this run's rollout buffer is not a ForkRolloutBuffer, so there is "
                          "nowhere to inject a branch row. `model_build.install_fork_buffer` is "
                          "what installs it")
            return
        obs = buf.observations
        if not isinstance(obs, dict) or "win_mask" not in obs or "action_mask" not in obs:
            self._disable("the rollout buffer's obs Dict is missing `win_mask`/`action_mask` — "
                          "the arm reads both to decide what may be forked")
            return
        bad = self._unfillable(list(obs))
        if bad:
            return
        self._calls += 1
        try:
            self._pass(buf, obs)
        except Exception as exc:                                        # noqa: BLE001
            # The training loop must survive anything this path can do.
            print(f"⚠️  [fork_arm] pass failed ({type(exc).__name__}: {str(exc)[:200]}) — this "
                  f"rollout's buffer is exactly what collection made it", flush=True)
            try:
                buf.add_fork_rows(None)
            except Exception:                                           # noqa: BLE001
                pass

    def _unfillable(self, keys) -> List[str]:
        from agents.training.fork_buffer import refusal_text, unfillable_keys
        bad = unfillable_keys(keys)
        if bad and not self._said_unfillable:
            self._said_unfillable = True
            self._disabled = True
            print(refusal_text(bad), flush=True)
            print("   The arm is DISABLED for this process; the run continues unforked.",
                  flush=True)
        return bad

    def _disable(self, why: str) -> None:
        self._disabled = True
        print(f"⚠️  [fork_arm] DISABLED: {why}. The run continues unforked.", flush=True)

    def _pass(self, buf, obs) -> None:
        from agents.training.fork_buffer import concat_blocks  # noqa: F401  (contract pin)
        from agents.training.fork_driver import assemble, play_forks
        from agents.training.win_prob_rollout import index_records

        n_steps, n_envs = int(self.model.n_steps), int(self.model.n_envs)
        keys = getattr(self.model, "_win_handle_keys", None)
        turns = getattr(self.model, "_win_handle_turns", None)
        if keys is None or turns is None:
            self._disable("no per-decision reconstruction HANDLE was captured. "
                          "`WinProbLabelCallback` allocates it, and `env_factory` must set "
                          "`env._emit_wp_rollout_handle`")
            return

        wm = np.asarray(obs["win_mask"])[:, :, 0]
        el = eligible_mask(wm, turns, obs["action_mask"])
        if not el.any() and not self._said_no_mask:
            self._said_no_mask = True
            print("⚠️  [fork_arm] no eligible rows this rollout. If that persists, check that "
                  "ForkArmCallback is registered AFTER WinProbLabelCallback — `win_mask` is a "
                  "placeholder of zeros until that callback back-fills it. Said once.", flush=True)

        frac = self._fraction()
        budget = int(ROW_BUDGET_MULTIPLE * n_steps * n_envs)
        want = self._budgeted_forks(n_forks_for(frac, n_steps, n_envs, cap=int(
            getattr(self.model, "fork_max_per_rollout", MAX_FORKS_PER_ROLLOUT))), budget)
        gap_q = float(getattr(self.model, "fork_contested_gap", DEFAULT_CONTESTED_GAP))
        absv_band = float(getattr(self.model, "fork_contested_absv", DEFAULT_CONTESTED_ABSV))
        n_branches = int(getattr(self.model, "fork_branches", DEFAULT_BRANCHES))
        per_battle = int(getattr(self.model, "fork_max_per_battle", DEFAULT_MAX_PER_BATTLE))
        crn = str(getattr(self.model, "fork_crn", DEFAULT_CRN))

        # Seeded from the RUN's seed and `num_timesteps` — `win_prob_rollout`'s rule and its
        # reason: `num_timesteps` is MONOTONIC ACROSS A LAUNCHER RESTART, so the first rollouts
        # after every restart do not redraw the first rollouts of the run.
        rng = np.random.default_rng(
            [int(getattr(self.model, "seed", 0) or 0),
             int(getattr(self.model, "num_timesteps", 0) or 0), int(self._calls)])
        pool = candidate_pool(el, buf.episode_starts, pool_size_for(want, gap_q), per_battle, rng)
        threshold = float("-inf")
        chosen: List[int] = []
        gaps = np.zeros(0)
        a1 = a2 = np.zeros(0, dtype=np.int64)
        if pool and want > 0:
            gaps, a1, a2, vals = self._score_pool(obs, pool)
            threshold = contested_threshold(gaps, gap_q)
            chosen = contested_select(pool, gaps, np.abs(vals - 0.5), threshold, absv_band, want)

        index = index_records(self._records_dir)
        forks: List[Dict[str, Any]] = []
        missing = 0
        for j in chosen:
            t, e = pool[j]
            path = index.get(str(keys[t, e]))
            if path is None:
                missing += 1
                continue
            legal = np.flatnonzero(np.asarray(obs["action_mask"])[t, e] > 0).astype(int).tolist()
            salt = f"{keys[t, e]}:{int(turns[t, e])}:{self._calls}"
            acts = branch_actions(legal, int(a1[j]), int(a2[j]), n_branches, salt=salt,
                                  seed=int(getattr(self.model, "seed", 0) or 0))
            forks.append({"id": len(forks), "record": path, "turn": int(turns[t, e]),
                          "actions": acts, "salt": salt, "row": (int(t), int(e))})

        results, stats = ([], {"seconds": 0.0})
        block, dropped, injected, masked = None, 0, 0, 0
        if forks:
            results, stats = play_forks(model=self.model, forks=forks, impl=self._impl, crn=crn)
            block, dropped = assemble(model=self.model, results=results, obs_keys=list(obs),
                                      mask_dims=int(buf.mask_dims), row_budget=budget)
            injected = buf.add_fork_rows(block)
            masked = int(block["n_masked"]) if block else 0
            self._observe_rows_per_fork(stats, len(forks))

        opp = obs.get("opp_class")
        cls = ([int(np.asarray(opp)[f["row"][0], f["row"][1], 0]) for f in forks]
               if opp is not None else None)
        m = fork_metrics(
            forks=results, requested=len(forks), eligible=int(el.sum()), pool=len(pool),
            threshold=threshold, injected_rows=injected, buffer_rows=n_steps * n_envs,
            masked_rows=masked, fraction=frac, branches=n_branches, crn=crn,
            seconds=float(stats.get("seconds", 0.0)), records_missing=missing,
            n_steps=n_steps, n_envs=n_envs, opp_class=cls)
        m["rate"] = forks_per_battle(int(m["forks"]), buf.episode_starts)
        m["dropped_forks"] = float(dropped)
        m["worker_failures"] = float(stats.get("worker_failures", 0.0))
        m["rows_per_fork"] = float(self._rows_per_fork or 0.0)
        m["row_budget"] = float(budget)
        m["branches_capped_frac"] = (float(stats.get("branches_capped", 0.0))
                                     / float(stats.get("branches", 0.0))
                                     if float(stats.get("branches", 0.0)) > 0 else 0.0)
        self.model._fork_metrics = m

    def _budgeted_forks(self, want: int, row_budget: int) -> int:
        """Cap the ASK at what the row budget can actually take.

        🚨 **A fork dropped at the budget has ALREADY BEEN PLAYED.** The branches are simulated
        before their rows can be counted, so the row cap alone bounds MEMORY and not COST — a run
        at a fraction the budget cannot carry would pay the full simulation bill for rows it then
        throws away. Measured in the `--debug` smoke: 30 of 85 forks dropped in one rollout.

        So the previous rollout's MEASURED mean rows-per-fork bounds this one's ask. Measured and
        not modelled: a branch's length is the run's own episode length, which moves over a run and
        is exactly the quantity a banked constant would get wrong. Until a pass has measured one
        the ask is unchanged — the first rollout pays the drop, once, and says so.

        The drop path in `fork_driver.assemble` stays as the BACKSTOP: this is an estimate from the
        previous rollout, and an estimate must never be the only thing standing between the buffer
        and an unbounded injection.
        """
        rpf = self._rows_per_fork
        if not rpf or rpf <= 0:
            return int(want)
        return int(max(1, min(int(want), int(row_budget // rpf))))

    def _observe_rows_per_fork(self, stats, n_forks: int) -> None:
        """Fold this rollout's realised rows-per-fork into the estimate the next one uses."""
        rows = float(stats.get("rows", 0.0))
        if n_forks <= 0 or rows <= 0.0:
            return
        seen = rows / float(n_forks)
        prev = self._rows_per_fork
        # A plain EMA, and biased UPWARD on the first update by taking the max: under-estimating
        # rows-per-fork is the failure that costs simulation, over-estimating only costs forks.
        self._rows_per_fork = seen if prev is None else 0.5 * (prev + seen)

    def _score_pool(self, obs, pool: List[Tuple[int, int]]):
        """``(gaps, top1, top2, values)`` for the candidate pool, from ONE batched forward.

        Every obs key the buffer holds is passed through, not just ``observation`` — the extractor
        may READ a flag-gated key (`agents.model.extra_obs_keys`), and a one-key dict is the
        `377a5aa1` forkserver death in miniature.
        """
        import torch as th

        ts = np.asarray([t for t, _e in pool], dtype=np.int64)
        es = np.asarray([e for _t, e in pool], dtype=np.int64)
        batch = {k: np.asarray(v)[ts, es] for k, v in obs.items()}
        masks = np.asarray(batch["action_mask"], dtype=np.float32)
        dev = self.model.device
        d = {k: th.as_tensor(np.ascontiguousarray(v)).to(dev) for k, v in batch.items()}
        was_training = self.model.policy.training
        self.model.policy.set_training_mode(False)
        try:
            with th.no_grad():
                dist = self.model.policy.get_distribution(d)
                logits = dist.distribution.logits.float().cpu().numpy()
                values = self.model.policy.predict_values(d).reshape(-1).float().cpu().numpy()
        finally:
            self.model.policy.set_training_mode(was_training)
        gaps, a1, a2 = top2_gaps(logits, masks)
        return gaps, a1, a2, values

    def _on_step(self) -> bool:
        return True
