import asyncio
import numpy as np
import torch
from typing import Dict, Any, Optional
from poke_env.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder


class ShowdownConnectionError(RuntimeError):
    """A Gen3 player could not log in to the Showdown server within the connect
    deadline. Raised instead of hanging: poke-env's ``PSClient.listen()`` swallows
    the underlying ``ConnectionRefused`` (it logs the exception and returns), which
    leaves ``logged_in`` forever unset, so every ``await logged_in.wait()`` in
    ``battle_against`` / ``accept_challenges`` would otherwise block indefinitely.
    Surfacing it lets replay / eval / self-play FAIL FAST (crash → the launcher
    detects and restarts) rather than silently stalling the whole training run."""

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.training.episode_tracker import EpisodeTracker
from agents.training.stall import StallConfig, StallLogger
from agents.model.features_extractor import N_HISTORY_TURNS


class Gen3Player(Player):
    """Base player: embeds battle state and maps actions using Gen 3 logic.

    Defaults ``battle_class=Gen3Battle`` so every Gen3 player (RL, eval, replay,
    stat-tracking) gets the revealed-order event log + ``live_view()`` for free.
    Callers can still override (e.g. a plain-Battle baseline) via the kwarg.
    """

    def __init__(self, observation_encoder=None, mappings=None,
                 stall_config: Optional[StallConfig] = None,
                 battle_class=Gen3Battle, **kwargs):
        super().__init__(battle_class=battle_class, **kwargs)
        self.observation_encoder = observation_encoder
        self.mappings = mappings
        self._stall_config = stall_config or StallConfig()
        self._stall_loggers: dict[str, StallLogger] = {}
        self._trackers: dict[str, EpisodeTracker] = {}
        self._turn_delta_encoder: Optional[TurnDeltaEncoder] = None

    async def _battle_against(self, *opponents, n_battles: int):
        """FUNDAMENTAL connect-or-raise guard around poke-env's battle flow.

        poke-env's ``listen()`` swallows a failed connection (it logs the exception
        and returns), leaving every participant's ``logged_in`` event unset — so the
        ``await logged_in.wait()`` inside the real ``_battle_against`` would hang
        forever. We run on POKE_LOOP here (battle_against wrapped us via
        handle_threaded_coroutines) and every client's ``logged_in`` / listen task
        lives on POKE_LOOP, so we can convert a dead connection into a loud
        :class:`ShowdownConnectionError` before delegating. Covers every participant
        — including plain poke-env opponents — since we check their ``ps_client`` too."""
        for participant in (self, *opponents):
            await self._await_connected(participant.ps_client)
        return await super()._battle_against(*opponents, n_battles=n_battles)

    @staticmethod
    async def _await_connected(client) -> None:
        """Block until ``client`` is logged in, or raise the moment its listen() task
        finishes WITHOUT logging in. This is DETERMINISTIC — no timeout guess: a
        refused connection makes poke-env's listen() return immediately, and a
        dead-but-open socket is closed by the websocket ping; both finish the listen
        task. So 'listen task done && not logged in' is an exact signal that the
        connection failed, which we surface instead of hanging on it forever."""
        if client.logged_in.is_set():
            return
        listen_fut = getattr(client, "_listening_coroutine", None)
        if listen_fut is None:
            # No listening task ⇒ the client can never log in on its own. A battling
            # player must listen; fail loudly rather than wait forever.
            raise ShowdownConnectionError(
                f"{client.websocket_url}: client has no listening task "
                f"(start_listening=False?) — it can never log in."
            )
        while not client.logged_in.is_set():
            if listen_fut.done():
                err = None
                if not listen_fut.cancelled():
                    err = listen_fut.exception()  # usually None — listen() swallows it
                raise ShowdownConnectionError(
                    f"{client.websocket_url}: listen task exited without logging in — "
                    f"Showdown server unreachable? "
                    + (f"({err!r})" if err else "(poke-env swallowed the connection error)")
                )
            await asyncio.sleep(0.05)

    def _handle_stall(self, battle, suffix: str) -> Optional[ForfeitBattleOrder]:
        """Returns ForfeitBattleOrder if the battle exceeds the stall threshold, else None.
        Each battle gets its own StallLogger, so concurrent battles don't interfere."""
        tag = battle.battle_tag
        if tag not in self._stall_loggers:
            self._stall_loggers[tag] = StallLogger(self._stall_config)
        stall_logger = self._stall_loggers[tag]
        if battle.turn >= stall_logger.threshold:
            stall_logger.log_once(battle, suffix=suffix)
            return ForfeitBattleOrder()
        return None

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)
        self._stall_loggers.pop(battle.battle_tag, None)
        self._trackers.pop(battle.battle_tag, None)

    def _get_tracker(self, battle) -> EpisodeTracker:
        tag = battle.battle_tag
        if tag not in self._trackers:
            self._trackers[tag] = EpisodeTracker()
        return self._trackers[tag]

    def embed_battle(self, battle) -> Dict[str, Any]:
        if self.observation_encoder is None:
            from agents.observation.state_encoder import load_mappings, get_observation_encoder
            if self.mappings is None:
                self.mappings = load_mappings()
            self.observation_encoder = get_observation_encoder(self.mappings)
        if self._turn_delta_encoder is None:
            self._turn_delta_encoder = TurnDeltaEncoder(
                self.mappings.get("moves", {}) if self.mappings else {},
                self.mappings.get("species", {}) if self.mappings else {},
            )

        # Record first so the tracker's HP candidates are up-to-date BEFORE
        # we encode the obs (mirrors gen3_env.embed_battle ordering). The legality
        # snapshot is captured once and threaded to both the mask and the context.
        legal = LegalActions.from_battle(battle)
        mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)
        tracker = self._get_tracker(battle)
        if not battle.finished and mask.sum() > 0:
            tracker.record(battle, mask, legal=legal)

        obs = self.observation_encoder.encode(
            battle, hp_tracker=tracker.hidden_power_tracker
        )

        prev_mask = tracker.prev_mask
        history_vecs = tracker.prev_N_delta_vecs(N_HISTORY_TURNS, self._turn_delta_encoder, battle=battle)

        return {
            "observation": np.concatenate([obs, prev_mask, history_vecs.flatten()]),
            "action_mask": mask,
        }

    def action_to_order(self, action_idx, battle):
        ctx = self._get_tracker(battle).last_ctx
        Gen3ActionMapper.assert_decision_current(ctx, battle)
        return Gen3ActionMapper.action_to_order(
            action_idx, battle, legal=ctx.legal, mask=ctx.mask,
        )


class RLPlayer(Gen3Player):
    """Runs a trained SB3 model to choose moves during evaluation."""

    def __init__(self, model, team, battle_format, server_configuration,
                 mappings=None, account_configuration=None,
                 stall_config: Optional[StallConfig] = None,
                 max_concurrent_battles=10, **kwargs):
        super().__init__(
            observation_encoder=None,
            mappings=mappings,
            stall_config=stall_config,
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
            **kwargs,
        )
        self.model = model

    def _predict_best_action(self, battle, stochastic=False, need_aux=True):
        """Pick the best legal action for `battle`.

        `need_aux` gates the work that ONLY the forensic recorder consumes:
        the critic forward pass (`predict_values`), the softmax→CPU `probs`
        transfer, and the `_last_prediction` snapshot. With `need_aux=False`
        (eval / plain inference, which use only the chosen action) this skips a
        whole second forward through the transformer body — roughly halving the
        GPU work per decision — plus two GPU→CPU copies. `probs` is then None.
        """
        obs_dict = self.embed_battle(battle)
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]

        obs_batched = np.expand_dims(obs, axis=0)
        mask_batched = np.expand_dims(mask, axis=0)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batched).to(self.model.device)
            mask_tensor = torch.as_tensor(mask_batched).to(self.model.device)
            policy_in = {"observation": obs_tensor, "action_mask": mask_tensor}
            dist = self.model.policy.get_distribution(policy_in)
            logits = dist.distribution.logits
            masked_logits = logits + (mask_tensor - 1.0) * 1e9

            if stochastic:
                idx = torch.distributions.Categorical(logits=masked_logits).sample().item()
            else:
                idx = torch.argmax(masked_logits, dim=1).item()

            probs = None
            if need_aux:
                probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
                value = float(self.model.policy.predict_values(policy_in)[0].item())

        if mask[idx] == 0:
            raise ValueError(f"Illegal action {idx} selected. Mask: {mask}")

        self._get_tracker(battle).advance(idx)
        if need_aux:
            # Raw inputs/outputs for offline forensic replay (probe_replay.py). Kept
            # as the last-prediction snapshot so callers that want it can read it
            # without changing the (idx, probs, mask) return contract.
            self._last_prediction = {
                "obs": np.asarray(obs, dtype=np.float32),
                "logits": logits[0].cpu().numpy(),
                "value": value,
            }
        return idx, probs, mask

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "INFERENCE_STALL")
        if forfeit:
            return forfeit
        idx, _, _ = self._predict_best_action(battle, need_aux=False)
        return self.action_to_order(idx, battle)
