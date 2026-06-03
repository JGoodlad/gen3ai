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
       bot roster + one reusable pool ``RLPlayer`` (model swapped on demand) + a ``SnapshotPool``
       handle. At reset(), with probability ``self_play_fraction`` it plays a pool snapshot
       (sampled recency-weighted from the *current* pool); otherwise a random heuristic. The
       opponent is a pure decision function over ``env.battle2`` (env.agent1/agent2 do all the
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
        self._rng = random.Random(rng_seed)  # per-env seed → envs don't pick in lockstep

    def set_self_play_target(self, fraction: float, generation: int) -> None:
        """Live curriculum update (called via ``VecEnv.env_method`` after each eval): the
        per-episode probability of facing a pool opponent, and a generation counter whose change
        triggers a pool re-scan (to pick up newly seeded/promoted snapshots)."""
        self._self_play_fraction = float(fraction)
        self._target_generation = int(generation)

    def _select_episode_opponent(self) -> None:
        """Pick this episode's opponent from the live fraction. Pool → sample a snapshot and
        swap it into the reusable pool player; else → a random heuristic."""
        if (self._pool is not None and self._pool_player is not None
                and self._rng.random() < self._self_play_fraction):
            # Re-scan to see new snapshots when the trainer signals a new generation, or while
            # the pool still looks empty (e.g. just after the seed first crossed the threshold).
            if self._target_generation != self._scanned_generation or self._pool.is_empty():
                self._pool._scan()
                self._scanned_generation = self._target_generation
            if not self._pool.is_empty():
                entry = self._pool.sample()
                self._pool_player.model = self._pool.load_model(entry)  # LRU-cached in the pool
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
