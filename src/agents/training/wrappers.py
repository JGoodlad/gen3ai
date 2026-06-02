from poke_env.environment.single_agent_wrapper import SingleAgentWrapper


class MaskableAgentWrapper(SingleAgentWrapper):
    """
    Bridges poke-env's SingleAgentWrapper to the MaskablePPO interface.

    Two responsibilities:

    1. action_masks() — MaskablePPO calls this each step to retrieve the valid
       action mask. SingleAgentWrapper doesn't implement it, so we delegate to
       the inner Gen3Env which owns the current BattleContext.

    2. Wait-turn absorption — poke-env returns a step to the caller whenever a
       server round-trip completes, even if agent1 had no decision to make (e.g.
       the opponent is choosing a forced-switch replacement). In those cases
       env.agent1_to_move is False and the action we pass is silently discarded.
       Letting these ghost steps surface to SB3 pollutes the rollout buffer and
       breaks credit assignment. We absorb them here by looping until agent1
       actually has a pending decision, accumulating any rewards along the way.
    """
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        while not self.env.agent1_to_move and not term and not trunc:
            obs, r, term, trunc, info = super().step(0)
            reward += r
        return obs, reward, term, trunc, info

    def action_masks(self):
        return self.env.action_masks()

    def opponent_default_stats(self):
        """(#decisions, #defaults, #re-decides) for the opponent — self-play telemetry, read
        via VecEnv.env_method. #re-decides counts stale-decision races RESOLVED by re-deciding
        on the now-current request (the foundational fix); a nonzero rate confirms the race is
        firing and being handled rather than crashing. Heuristic opponents lack these counters,
        so they report zeros and don't pollute the rate."""
        op = self.opponent
        return (
            getattr(op, "_n_decisions", 0),
            getattr(op, "_n_defaults", 0),
            getattr(op, "_n_redecides", 0),
        )
