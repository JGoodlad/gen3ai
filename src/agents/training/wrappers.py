import random

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper


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
                 pool_player=None, self_play_fraction=0.0, rng_seed=0):
        # Back-compat: a single positional `opponent` (legacy / tests) becomes a 1-bot roster.
        roster = list(heuristic_opponents) if heuristic_opponents else (
            [opponent] if opponent is not None else [])
        if not roster:
            raise ValueError(
                "MaskableAgentWrapper needs an `opponent` or a `heuristic_opponents` roster")
        super().__init__(env, roster[0])
        self._heuristic_opponents = roster
        self._pool = pool
        self._pool_player = pool_player
        self._self_play_fraction = float(self_play_fraction)
        self._target_generation = 0
        self._scanned_generation = -1   # -1 forces a pool re-scan on the first pool selection
        self._has_pool_model = False    # a snapshot is loaded into the pool player (this gen)
        self._rng = random.Random(rng_seed)  # per-env seed → envs don't pick in lockstep

    def set_self_play_target(self, fraction: float, generation: int) -> None:
        """Live curriculum update (called via ``VecEnv.env_method`` after each eval): the
        per-episode probability of facing a pool opponent, and a generation counter whose change
        triggers a pool re-scan (to pick up newly seeded/promoted snapshots)."""
        self._self_play_fraction = float(fraction)
        self._target_generation = int(generation)

    def _select_episode_opponent(self) -> None:
        """Pick this episode's opponent from the live fraction. Pool → use the per-generation
        snapshot held in the reusable pool player; else → a random heuristic.

        The pool snapshot is (re)sampled+loaded ONLY when the trainer signals a new generation
        (a seed or promotion — i.e. every eval), not every episode. ``load_model`` deserializes a
        ~27MB MaskablePPO; doing it per-episode against an N-deep pool with a small LRU thrashed
        the workers (they block in ``reset()`` on disk I/O → CPU ~40%, FPS ~1400→~500). Loading
        once per generation restores the old "load once, reuse" throughput while keeping
        diversity: 48 envs sample independently (≈ many distinct snapshots in flight at once) and
        every env rotates to a fresh sample each generation. The per-episode pool-vs-heuristic
        coin flip is unchanged, so the live ``self_play_fraction`` is still honored exactly."""
        if (self._pool is not None and self._pool_player is not None
                and self._rng.random() < self._self_play_fraction):
            # Re-sample+load on a new generation, or until we first have a model loaded (the pool
            # may still be empty right after the seed crosses the threshold). Steady state within
            # a generation: neither branch runs → no scan, no load → the opponent is reused.
            if self._target_generation != self._scanned_generation or not self._has_pool_model:
                self._pool._scan()
                if not self._pool.is_empty():
                    entry = self._pool.sample()
                    self._pool_player.model = self._pool.load_model(entry)  # LRU-cached
                    self._has_pool_model = True
                    self._scanned_generation = self._target_generation
            if self._has_pool_model:
                self.opponent = self._pool_player
                return
        self.opponent = self._rng.choice(self._heuristic_opponents)

    def reset(self, *, seed=None, options=None):
        self._select_episode_opponent()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        while not self.env.agent1_to_move and not term and not trunc:
            obs, r, term, trunc, info = super().step(0)
            reward += r
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
