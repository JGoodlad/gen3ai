import asyncio
from typing import Any, Awaitable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.envs.registration import EnvSpec

from poke_env.environment.env import ActionType, PokeEnv
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.player.player import Player

# Opponent-poll settle (the self-play stale-decision race fix). Yields are cheap (µs); the
# common case is already-settled → one re-check and return. The bound + timeout are backstops
# against a pathological loop; the strict StaleDecisionError guard catches any residual.
_SETTLE_MAX_YIELDS = 64
_SETTLE_TIMEOUT_S = 2.0


def _battle_decision_signature(battle):
    """Cheap, value-only fingerprint of a battle's decision-relevant state — the axes a
    legal-actions snapshot keys on. Used to detect that POKE_LOOP has stopped mutating the
    battle (the request has settled) before we read it on the training thread."""
    return (
        battle.force_switch,
        battle.maybe_trapped,
        tuple(m.id for m in (battle.available_moves or [])),
        tuple(p.species for p in (battle.available_switches or [])),
    )


class SingleAgentWrapper(Env[Dict[str, Any], ActionType]):
    def __init__(self, env: PokeEnv[ActionType], opponent: Player):
        self.env = env
        self.opponent = opponent
        self.spec = EnvSpec("poke-env-v0", nondeterministic=True)
        self.observation_space = env.observation_spaces[env.agent1.username]
        self.action_space = env.action_spaces[env.agent1.username]
        self.second_teampreview_action: npt.NDArray[np.int64] | None = None

    def step(
        self, action: ActionType
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        Takes an action from the main agent and automatically generates an action
        for the opponent using the opponent's choose_move method.

        :param action: Action from the main agent.
        :type action: ActionType
        :return: Tuple of (observation, reward, terminated, truncated, info) for the main agent.
        :rtype: Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]
        """
        assert self.env.battle2 is not None
        # Settle POKE_LOOP's in-flight work for the opponent's battle BEFORE we read it for a
        # decision. The opponent is a self-play RLPlayer polled out-of-band on the training
        # thread: it latches a legal-actions snapshot and serializes it against the live
        # battle. poke-env's request drain (199936d) gates its OWN players' decisions, but not
        # this one, so without this a faint→force-switch parse can land between the snapshot
        # and the serialize (the stale-decision race). See _settle_opponent_battle.
        self._settle_opponent_battle()
        if self.env.battle2.wait:
            opp_action = self.env.order_to_action(
                DefaultBattleOrder(),
                self.env.battle2,
                fake=self.env._fake,
                strict=self.env._strict,
            )
        elif not self.env.battle2.teampreview:
            opp_order = self.opponent.choose_move(self.env.battle2)
            assert not isinstance(opp_order, Awaitable)
            opp_action = self.env.order_to_action(
                opp_order,
                self.env.battle2,
                fake=self.env._fake,
                strict=self.env._strict,
            )
        elif self.env.battle2.format is None or "vgc" not in self.env.battle2.format:
            raise NotImplementedError(
                "Teampreview is only supported for VGC formats in SingleAgentWrapper."
            )
        elif self.second_teampreview_action is None:
            tp_order = self.opponent.teampreview(self.env.battle2)
            assert not isinstance(tp_order, Awaitable)
            assert len(tp_order) == 10, f"{tp_order} must specify 4 slots in VGC!"
            teampreview_order_list = [int(i) for i in tp_order[-4:]]
            opp_action = np.array(teampreview_order_list[:2])  # type: ignore
            self.second_teampreview_action = np.array(teampreview_order_list[2:])
            # only the first two pokemon are selected in teampreview for now
            for i, pokemon in enumerate(self.env.battle2.team.values(), start=1):
                pokemon._selected_in_teampreview = i in teampreview_order_list[:2]
        else:
            opp_action = self.second_teampreview_action  # type: ignore
            # now the second two pokemon are selected in teampreview
            for i in opp_action:  # type: ignore
                mon = list(self.env.battle2.team.values())[i - 1]
                mon._selected_in_teampreview = True
            self.second_teampreview_action = None
        actions = {
            self.env.agent1.username: action,
            self.env.agent2.username: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )

    def _settle_opponent_battle(self) -> None:
        """Drain POKE_LOOP's pending message processing for the opponent's battle so the
        opponent's snapshot→serialize can't race a background parse_request.

        Runs a coroutine ON POKE_LOOP that yields until the battle's decision signature is
        stable across a yield (the server is waiting on our move, so once nothing changes
        across a scheduler tick, nothing else will land before we act). We *yield* rather than
        await the in-flight tasks directly: one of those tasks is the opponent player's own
        ``_handle_battle_request`` coroutine, blocked awaiting the order we're about to
        produce — awaiting it here would deadlock. Best-effort and bounded; if a message still
        slips in, the strict ``StaleDecisionError`` guard crashes loudly (no garbage move).

        The settle is unconditional in production — there is no "disable" switch (that would be
        a footgun on a correctness fix). Reproducing the race is a test concern: the fuzz catches
        a settle regression by finding divergences that shouldn't exist, and to demonstrate the
        crash on purpose, revert this method's call site (one line) and re-run."""
        battle = self.env.battle2
        client = getattr(self.env.agent2, "ps_client", None)
        loop = getattr(client, "loop", None)
        if battle is None or loop is None:
            return

        async def _settle():
            prev = None
            for _ in range(_SETTLE_MAX_YIELDS):
                cur = _battle_decision_signature(battle)
                if cur == prev:
                    return
                prev = cur
                await asyncio.sleep(0)  # let POKE_LOOP apply any in-flight parse_request

        try:
            asyncio.run_coroutine_threadsafe(_settle(), loop).result(
                timeout=_SETTLE_TIMEOUT_S
            )
        except Exception:
            # Best-effort settle; the strict StaleDecisionError guard is the backstop.
            pass

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, infos = self.env.reset(seed, options)
        self.opponent.reset_battles()
        assert self.env.battle2 is not None
        self.opponent._battles[self.env.battle2.battle_tag] = self.env.battle2
        self._np_random = self.env._np_random
        return obs[self.env.agent1.username], infos[self.env.agent1.username]

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()
