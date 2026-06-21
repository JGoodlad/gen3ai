import random

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

# Stable cross-run opponents are a CAPPED minority of the self-play (challenge) bucket — the pool
# keeps the bulk, so no single fixed opponent can dominate training. Multiple un-mastered stable
# opponents SHARE this slice (so the total stable share stays ≤ this regardless of count). Mastered
# ones leave the challenge bucket entirely for the floor.
STABLE_CHALLENGE_SHARE = 0.20


class MaskableAgentWrapper(SingleAgentWrapper):
    """
    Bridges poke-env's SingleAgentWrapper to the MaskablePPO interface, and selects the
    episode's opponent at reset() from a LIVE self-play fraction.

    Five responsibilities:

    1. action_masks() — MaskablePPO calls this each step to retrieve the valid action mask.
       SingleAgentWrapper doesn't implement it, so we delegate to the inner Gen3Env.

    2. Wait-turn absorption — poke-env returns a step whenever a server round-trip completes,
       even when agent1 had no decision (e.g. the opponent is choosing a forced switch). Those
       ghost steps would pollute the rollout buffer, so we absorb them (loop until agent1 has a
       pending decision, accumulating reward).

    3. Per-episode opponent selection (the self-play curriculum). The wrapper holds the heuristic
       bot roster + one reusable pool ``RLPlayer`` + a ``SnapshotPool`` handle. At reset(), with
       probability ``self_play_fraction`` it plays the pool, else a random heuristic — the coin
       flip is per-episode so the live fraction is honored exactly. The pool *snapshot* itself is
       (re)sampled+loaded only once per generation (see ``_select_episode_opponent`` — loading is
       a ~27MB deserialize, too expensive to do per-episode), held in the reusable pool player.
       The opponent is a pure decision function over ``env.battle2`` (env.agent1/agent2 do all the
       networking), so swapping it between episodes is free and safe — the in-episode
       stale-decision path (SingleAgentWrapper.step) is untouched.

    4. Live curriculum updates — ``set_self_play_target(fraction, generation)`` is pushed by the
       eval callback via ``VecEnv.env_method`` after every eval, so the heuristic-vs-pool ratio
       tracks the model's measured strength *during* a run (no restart needed). A new generation
       triggers a pool re-scan to pick up freshly seeded/promoted snapshots.

    5. Opponent default-stat telemetry — read from the persistent pool player (not the rotating
       ``self.opponent``), so the counts aggregate across all self-play episodes.
    """

    def __init__(self, env, opponent=None, *, heuristic_opponents=None, pool=None,
                 pool_player=None, self_play_fraction=0.0, rng_seed=0,
                 heuristic_weights=None, stable_players=None, stable_labels=None,
                 stable_challenge_share=STABLE_CHALLENGE_SHARE):
        # Back-compat: a single positional `opponent` (legacy / tests) becomes a 1-bot roster.
        roster = list(heuristic_opponents) if heuristic_opponents else (
            [opponent] if opponent is not None else [])
        if not roster:
            raise ValueError(
                "MaskableAgentWrapper needs an `opponent` or a `heuristic_opponents` roster")
        super().__init__(env, roster[0])
        self._heuristic_opponents = roster
        # Per-bot sampling weights aligned index-for-index to `roster`, biasing WHICH heuristic
        # an episode draws (e.g. the coverage-punishing aggressive_v2/heuristic2). None → uniform
        # (the original behavior, byte-for-byte). Validated at construction so a misaligned vector
        # fails loudly here, never silently per-episode. Applies only to the heuristic branch of
        # _select_episode_opponent; the pool-vs-heuristic fraction and pool sampling are untouched.
        if heuristic_weights is None:
            self._heuristic_weights = None
        else:
            w = [float(x) for x in heuristic_weights]
            if len(w) != len(roster):
                raise ValueError(
                    f"heuristic_weights len {len(w)} != heuristic roster len {len(roster)}")
            if any(x < 0 for x in w) or sum(w) <= 0:
                raise ValueError("heuristic_weights must be non-negative with a positive sum")
            self._heuristic_weights = w
        self._pool = pool
        self._pool_player = pool_player
        # Stable cross-run opponents (already-loaded RLPlayers + their labels): UN-mastered ones are
        # peers of the pool in the CHALLENGE bucket ("tossed in like another sentinel"); once the
        # trainee MASTERS one (pushed via set_stable_mastered) it joins the FLOOR bucket alongside
        # the bots ("becomes another bot"). Built once per worker, so no per-episode reload.
        self._stable_players = list(stable_players) if stable_players else []
        self._stable_labels = list(stable_labels) if stable_labels else []
        self._stable_mastered = {lab: False for lab in self._stable_labels}
        self._stable_challenge_share = float(stable_challenge_share)
        self._self_play_fraction = float(self_play_fraction)
        self._target_generation = 0
        self._scanned_generation = -1   # -1 forces a pool re-scan on the first pool selection
        self._has_pool_model = False    # a snapshot is loaded into the pool player (this gen)
        self._rng = random.Random(rng_seed)  # per-env seed → envs don't pick in lockstep
        # All-or-nothing distillation (distill_integration.md §8): when active, EVERY pool opponent
        # is the cheap distilled variant (sampled from the deployable set); when not, all full.
        self._distill_active = False
        self._distill_steps: set[int] = set()
        self._loaded_distilled = False  # whether the currently-loaded pool model is distilled

    def set_self_play_target(self, fraction: float, generation: int) -> None:
        """Live curriculum update (called via ``VecEnv.env_method`` after each eval): the
        per-episode probability of facing a pool opponent, and a generation counter whose change
        triggers a pool re-scan (to pick up newly seeded/promoted snapshots)."""
        self._self_play_fraction = float(fraction)
        self._target_generation = int(generation)

    def set_opponent_win_rates(self, rates) -> None:
        """Live PFSP update (pushed via ``VecEnv.env_method`` each eval): the trainee's per-snapshot
        win-rates (``{step: P(win)}``, EMA-smoothed in the callback) that weight pool sampling toward
        the selves we're losing to. Forwarded to the worker's ``SnapshotPool``; a no-op on sampling
        unless the pool was built with ``pfsp_scale > 0``. Pushed only when PFSP is on, so an off run
        never even makes this IPC call (byte-identical)."""
        if self._pool is not None:
            self._pool.set_win_rates(rates)

    def set_distill_active(self, active: bool, steps=None) -> None:
        """Pushed by the trainer's reconcile each eval (via ``env_method``): whether to use the
        cheap distilled opponents pool-wide, and which snapshot steps are deployable (gate-passed).
        All-or-nothing — ``active`` is true only when the *whole* deployable set is distilled."""
        self._distill_active = bool(active)
        self._distill_steps = set(steps or ())

    def set_stable_mastered(self, mastered_labels) -> None:
        """Pushed by the eval callback each cycle (via ``env_method``): the labels of stable
        cross-run opponents the trainee has MASTERED (win_rate ≥ threshold). A mastered opponent
        moves from the CHALLENGE bucket (peer of the pool) to the FLOOR bucket (peer of the bots) —
        it "becomes another bot", kept for coverage but no longer a thing to master. The callback's
        set only grows, so this is monotonic (no flapping on eval noise) and resume-safe (recomputed
        from eval each cycle — no stored env state)."""
        s = set(mastered_labels or ())
        self._stable_mastered = {lab: (lab in s) for lab in self._stable_labels}

    def _ensure_pool_model(self) -> bool:
        """Load/refresh the per-generation pool snapshot into the reusable pool player; return True
        if a pool model is ready.

        The snapshot is (re)sampled+loaded ONLY when the trainer signals a new generation (a seed or
        promotion — i.e. every eval), the first time a model is needed, or when the distilled/full
        mode flips (the atomic all-or-nothing switch) — NOT per episode. ``load_model`` deserializes
        a ~27MB MaskablePPO; doing it per-episode against an N-deep pool with a small LRU thrashed
        the workers (they block in ``reset()`` on disk I/O → CPU ~40%, FPS ~1400→~500). Loading once
        per generation restores the "load once, reuse" throughput while keeping diversity: the envs
        sample independently and every env rotates to a fresh sample each generation."""
        if self._pool is None or self._pool_player is None:
            return False
        if (self._target_generation != self._scanned_generation or not self._has_pool_model
                or self._distill_active != self._loaded_distilled):
            self._pool._scan()
            if not self._pool.is_empty():
                if self._distill_active and self._distill_steps:
                    entry = self._pool.sample_from(self._distill_steps)  # deployable distilled only
                    if entry is not None:
                        self._pool_player.model = self._pool.load_distilled_opponent(entry)
                        self._has_pool_model = True
                        self._loaded_distilled = True
                        self._scanned_generation = self._target_generation
                if not (self._distill_active and self._loaded_distilled):
                    entry = self._pool.sample()                          # full model
                    self._pool_player.model = self._pool.load_model(entry)  # LRU-cached
                    self._has_pool_model = True
                    self._loaded_distilled = False
                    self._scanned_generation = self._target_generation
        return self._has_pool_model

    def _stable_in(self, mastered: bool) -> list:
        """The stable-opponent players whose mastery state matches ``mastered``."""
        return [p for p, lab in zip(self._stable_players, self._stable_labels)
                if self._stable_mastered.get(lab, False) == mastered]

    def _pick_floor_opponent(self):
        """The always-on coverage bucket: the heuristic bot roster + any MASTERED stable opponents
        (each weighted like an unlisted bot, 1.0). Honors ``--bot-weights`` for the bots. Stable
        opponents are excluded while distillation is active (see ``_pick_challenge_opponent``)."""
        mastered = [] if self._distill_active else self._stable_in(mastered=True)
        candidates = self._heuristic_opponents + mastered
        # This floor weighting (Σ bot weights, +1.0 per mastered stable) — like the whole selection
        # split here — is mirrored for REPORTING by SelfPlayCallback._opponent_mix_fractions; keep
        # the two in sync (pinned by wrappers_test.py::test_mix_fractions_match_actual_sampling).
        if self._heuristic_weights is None:
            return self._rng.choice(candidates)
        weights = self._heuristic_weights + [1.0] * len(mastered)
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_challenge_opponent(self):
        """The CHALLENGE bucket — what the model is actively trying to master. The self-play pool
        gets the BULK; any UN-mastered stable cross-run opponents share a CAPPED minority slice
        (``_stable_challenge_share``, default 20%), so a single fixed opponent can never dominate
        training. Returns ``None`` if neither has a ready opponent (→ fall through to the floor).

        While **distillation is active** the pool is 100% cheap distilled models (all-or-nothing,
        since one full-model worker straggles and gates the per-step ``SubprocVecEnv`` barrier). A
        FULL foreign stable opponent would re-introduce exactly that straggler, so stable opponents
        drop OUT of the training mix entirely while distill is on (they stay eval-only that period);
        the pool gets the whole challenge bucket."""
        stable = [] if self._distill_active else self._stable_in(mastered=False)
        pool_ready = self._ensure_pool_model()
        if stable and pool_ready:
            if self._rng.random() < self._stable_challenge_share:
                return self._rng.choice(stable)       # the capped stable slice
            return self._pool_player                  # the bulk → the pool
        if pool_ready:
            return self._pool_player
        if stable:
            return self._rng.choice(stable)           # no pool seeded yet → stable IS the challenge
        return None

    def _select_episode_opponent(self) -> None:
        """Pick this episode's opponent.

        CHALLENGE bucket (competence-gated by ``self_play_fraction``, ``_pick_challenge_opponent``) =
        the self-play pool (the bulk) + a capped minority of un-mastered stable opponents. FLOOR
        bucket (always-on coverage) = the heuristic bots + any MASTERED stable opponents. The
        per-episode coin flip honors the live ``self_play_fraction`` exactly; the pool snapshot is
        loaded once per generation (``_ensure_pool_model``), not per episode."""
        if self._rng.random() < self._self_play_fraction:
            opp = self._pick_challenge_opponent()
            if opp is not None:
                self.opponent = opp
                return
        # Floor bucket — also the fallthrough when the challenge bucket has no ready opponent.
        self.opponent = self._pick_floor_opponent()

    def reset(self, *, seed=None, options=None):
        self._select_episode_opponent()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        while not self.env.agent1_to_move and not term and not trunc:
            obs, r, term, trunc, info = super().step(0)
            reward += r
        if term or trunc:
            # Expose the battle OUTCOME for the win-probability label plumbing (win=1 / loss-or-tie=0).
            # The trainee's battle is finished here (before the VecEnv auto-resets), so battle1.won is
            # set. Consumed ONLY when the win-prob head is on (WinProbLabelCallback / the async
            # collector); harmless otherwise. A tie (won is None) counts as not-a-win → 0.0.
            b = getattr(self.env, "battle1", None)
            info["win_outcome"] = 1.0 if (b is not None and b.won is True) else 0.0
        return obs, reward, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()

    def opponent_default_stats(self):
        """(#decisions, #defaults, #re-decides) for the self-play POOL opponent — read from the
        persistent pool player (NOT ``self.opponent``, which rotates per episode), so the counts
        aggregate across all self-play episodes. Heuristic-only envs report zeros. #re-decides
        counts stale-decision races RESOLVED by re-deciding on the now-current request."""
        op = self._pool_player
        if op is None:
            return (0, 0, 0)
        return (
            getattr(op, "_n_decisions", 0),
            getattr(op, "_n_defaults", 0),
            getattr(op, "_n_redecides", 0),
        )
