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

from agents.action.mapper import Gen3ActionMapper, StaleDecisionError
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.training.episode_tracker import EpisodeTracker
from agents.training.stall import StallConfig, StallLogger
from agents.inference.belief_decode import decode_species_belief
from utils import race_trace  # debug ring buffer (GEN3_RACE_TRACE); no-op when off

# Self-play opponent re-decide budget. When the live battle advances under the opponent's
# decision (POKE_LOOP parses an in-flight turn-resolution during the model forward), the
# opponent re-decides on the now-current request instead of raising — the server is waiting on
# our move, so the battle settles within a couple of re-decides. See choose_move().
_OPP_REDECIDE_MAX = 8
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
        view = battle.strict_view()
        tag = view.battle_tag
        if tag not in self._stall_loggers:
            self._stall_loggers[tag] = StallLogger(self._stall_config)
        stall_logger = self._stall_loggers[tag]
        if view.turn >= stall_logger.threshold:
            stall_logger.log_once(battle, suffix=suffix)
            return ForfeitBattleOrder()
        return None

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)
        tag = battle.strict_view().battle_tag
        self._stall_loggers.pop(tag, None)
        self._trackers.pop(tag, None)

    def reset_battles(self) -> None:
        """Clear poke-env's battle tracker AND our per-battle-tag caches.

        ``_battle_finished_callback`` is the normal eviction point for ``_trackers`` /
        ``_stall_loggers``, but it only fires for a *networked* player (one with a live
        message loop). A self-play OPPONENT is built ``start_listening=False`` and used as
        a pure decision function over the env's battle — it never receives that callback,
        so without this override every battle would leave a permanent ``EpisodeTracker``
        (+ its obs/turn-delta history) and ``StallLogger`` behind, keyed by the bridge's
        per-battle-unique tag. The wrapper already calls ``reset_battles()`` on the
        opponent every episode, so pruning here is the correct, cheap fix. We keep only
        entries whose tag is still live in ``_battles`` (``super()`` empties it, so this
        clears everything in the per-episode case, while staying safe for any caller that
        resets with battles still in flight)."""
        super().reset_battles()
        live = set(self._battles)
        self._trackers = {t: v for t, v in self._trackers.items() if t in live}
        self._stall_loggers = {t: v for t, v in self._stall_loggers.items() if t in live}

    def _get_tracker(self, battle) -> EpisodeTracker:
        tag = battle.strict_view().battle_tag
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
        if not battle.strict_view().finished and mask.sum() > 0:
            tracker.record(battle, mask, legal=legal)
            # Advance the ProgressClock for the just-completed window BEFORE encode reads it (same
            # helper gen3_env uses), so the turns_since_progress obs scalar matches what the model saw
            # during TRAINING — without this, eval / self-play opponents / play.py would always read 0.
            tracker.update_progress_clock(battle, legal)

        # Thread the same legality snapshot into the encoder for its trapped / maybe_trapped
        # reactive bits (avoids a second LegalActions.from_battle this decision).
        obs = self.observation_encoder.encode(
            battle, hp_tracker=tracker.hidden_power_tracker, legal=legal,
            progress_clock=tracker.progress_clock,
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
                 max_concurrent_battles=10,
                 stochastic: bool = True, temperature: float = 1.0, **kwargs):
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
        # Stochastic (temperature-sampled) by DEFAULT — opponents and ordinary play
        # sample the policy's full action distribution (richer, less exploitable than
        # always-greedy). Eval/measurement players pass stochastic=False explicitly so
        # win_rate_vs_bots / win_rate_vs_pool stay stable, comparable control signals.
        # `temperature` scales the sampling logits: 1.0 = the policy's own distribution,
        # >1 flatter (more random), <1 sharper (toward argmax). Ignored when not stochastic.
        self._stochastic = stochastic
        self._temperature = temperature
        # Telemetry (read via the env wrapper for self-play opponents): total decision calls,
        # total default deferrals (idx None → no legal action, or re-decide budget exhausted),
        # and total RE-DECIDES (the live battle advanced under a decision and we re-decided on
        # the now-current request — the self-play stale-decision race, resolved not crashed).
        self._n_decisions = 0
        self._n_defaults = 0
        self._n_redecides = 0

    def _predict_best_action(self, battle, stochastic=False, need_aux=True, temperature=1.0):
        """Pick the best legal action for `battle`.

        `need_aux` gates the work that ONLY the forensic recorder consumes:
        the critic forward pass (`predict_values`), the softmax→CPU `probs`
        transfer, and the `_last_prediction` snapshot. With `need_aux=False`
        (eval / plain inference, which use only the chosen action) this skips a
        whole second forward through the transformer body — roughly halving the
        GPU work per decision — plus two GPU→CPU copies. `probs` is then None.
        """
        obs_dict = self.embed_battle(battle)
        race_trace.trace(
            getattr(battle, "battle_tag", ""),
            f"EMBED(opp) battle.turn={getattr(battle, 'turn', '?')} mask_sum={int(obs_dict['action_mask'].sum())}",
        )
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]

        # No legal action this call — e.g. SingleAgentWrapper polls the opponent during
        # the OTHER agent's forced-switch window, so this battle's request is momentarily
        # empty. embed_battle() skips recording a decision context when the mask is all-zero,
        # so acting would assert on a stale ctx. Signal 'no decision' (idx=None) → the caller
        # defers to poke-env's default order, exactly as the heuristic opponents do.
        if int(mask.sum()) == 0:
            return None, None, mask

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
                # Temperature-scaled sampling. The -1e9 mask offset stays hugely
                # negative after dividing by any positive temperature, so illegal
                # actions remain masked regardless of T.
                sample_logits = masked_logits / temperature if temperature != 1.0 else masked_logits
                idx = torch.distributions.Categorical(logits=sample_logits).sample().item()
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
                # The model's top-k species guess for each still-HIDDEN opp slot (None unless the
                # hidden-opponent belief is enabled) — "what does it think the unrevealed mons are?".
                "belief": self._decode_belief(),
                # Calibrated P(win) from the win-probability head (None unless --win-prob-mode != none).
                # The prober shows it + ΔP(win) beside the CRITIC line — "how a move moved the win odds".
                "win_prob": self._win_prob(),
                # The distributional value head's predicted RETURN DISTRIBUTION (per-atom probs; None
                # unless --value-dist-mode != none) — the prober renders the histogram + spread/PIT.
                "value_dist": self._value_dist(),
            }
        return idx, probs, mask

    def _win_prob(self) -> Optional[float]:
        """P(win) from the win-probability head (forensic trace only). Reads the logit the extractor
        stashed on this same forward (``last_win_prob_logits``); None when the head is off
        (``--win-prob-mode none``). sigmoid(logit) ⇒ probability the current state leads to a win."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = getattr(extractor, "last_win_prob_logits", None)
        if logits is None:
            return None
        return float(torch.sigmoid(logits[0, 0]).item())

    def _value_dist(self) -> Optional[list]:
        """The distributional value head's predicted return distribution (forensic trace only) — the
        per-atom softmax over ``last_value_dist_logits`` stashed on this same forward; None when the head
        is off (``--value-dist-mode none``). The prober reconstructs the atom support from
        model_config.json's value_dist_vmin/vmax/bins to render the histogram + mean/std/entropy/PIT."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = getattr(extractor, "last_value_dist_logits", None)
        if logits is None:
            return None
        return torch.softmax(logits[0], dim=-1).cpu().numpy().tolist()

    def _decode_belief(self) -> Optional[list]:
        """Decode the belief head's per-slot species prediction for the still-hidden opponent slots
        (forensic trace only). Reads the logits the extractor stashed on this same forward
        (``last_belief_logits["species"]``) + the believed-slot mask (``last_opp_believed_mask``);
        returns None when the hidden-opponent belief is off / no slot is hidden. See belief_decode.py.
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = getattr(extractor, "last_belief_logits", None)
        mask = getattr(extractor, "last_opp_believed_mask", None)
        if logits is None or mask is None or "species" not in logits:
            return None
        species_logits = logits["species"][0].detach().cpu().numpy()   # [n_slots, n_species]
        believed = mask[0].detach().cpu().numpy()                      # [n_slots] bool
        return decode_species_belief(species_logits, believed) or None

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "INFERENCE_STALL")
        if forfeit:
            return forfeit
        self._n_decisions += 1
        # The live battle can advance under us between embed_battle (the obs / legal snapshot)
        # and action_to_order (the serialize): POKE_LOOP parses an in-flight turn-resolution
        # during our model forward, so assert_decision_current sees battle.turn one ahead of
        # ctx.turn (proven by the race trace — both Dugtrios mutually Arena-Trap, the turn
        # resolves while the opponent computes). Our decision is INTERNAL to
        # SingleAgentWrapper.step — SB3 never sees it — so we RE-DECIDE on the now-current
        # request rather than raise (a raise kills the SB3 worker; SB3 has no failed-step path).
        # Bounded: the server is waiting on our move, so the battle settles within a couple of
        # re-decides. The TRAINEE cannot do this (its action is SB3's, computed outside the env)
        # and stays strict in gen3_env — by design, since acting on a stale trainee snapshot
        # would corrupt its (obs, action) → (reward, next_obs) transition.
        #
        # Each attempt's embed_battle() records its (would-be) decision into the tracker's
        # rolling turn-history. On a stale re-decide we restore() the snapshot taken here, so a
        # superseded attempt leaves no phantom turn behind — only the committed decision survives
        # in the opponent's turn-history obs. (The trainee never re-decides, so it needs none.)
        tracker = self._get_tracker(battle)
        history_snapshot = tracker.snapshot()
        for attempt in range(_OPP_REDECIDE_MAX):
            idx, _, _ = self._predict_best_action(
                battle, stochastic=self._stochastic, need_aux=False, temperature=self._temperature
            )
            if idx is None:
                self._n_defaults += 1
                tracker.restore(history_snapshot)
                return self.choose_default_move()
            try:
                return self.action_to_order(idx, battle)
            except StaleDecisionError:
                self._n_redecides += 1
                tracker.restore(history_snapshot)
                race_trace.trace(
                    getattr(battle, "battle_tag", ""),
                    f"OPP-REDECIDE attempt={attempt + 1} battle.turn={getattr(battle, 'turn', '?')}",
                )
        # The battle never settled across the budget (pathological — the server kept advancing
        # it under us). A valid default on the current state (the env re-requests next step)
        # beats killing the worker; this branch is rare by construction.
        self._n_defaults += 1
        race_trace.trace(getattr(battle, "battle_tag", ""), "OPP-REDECIDE EXHAUSTED -> default")
        return self.choose_default_move()
