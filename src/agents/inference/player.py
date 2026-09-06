import asyncio
import os
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


#: How long a player waits for its own login before giving up (seconds). Generous on purpose —
#: it is a WEDGE detector, not a latency budget: a healthy local login lands in well under a
#: second and the official server in a few, so 30 s never fires on a working connection.
#:
#: 🚨 A DEADLINE IS REQUIRED, and the listen-task signal alone is NOT enough. `localhost_server_
#: configuration` authenticates against the real Smogon `action.php`, so a username REGISTERED
#: upstream is refused even on a `--no-security` local server — and that refusal leaves the
#: socket OPEN and the listen task RUNNING with `logged_in` never set. The deterministic
#: "listen task finished" signal below therefore never fires, and the wait sat the whole battle
#: deadline in silence, reporting phantom timeouts (measured 2026-09-05, cross-era head-to-head).
DEFAULT_CONNECT_TIMEOUT_S = 30.0

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions

from agents.action.mapper import Gen3ActionMapper, StaleDecisionError
from agents.action.mask_generator import Gen3ActionMasker
from agents.training.episode_tracker import EpisodeTracker
from agents.training.stall import StallConfig, StallLogger
from agents.inference.belief_decode import decode_species_belief
from utils import race_trace  # debug ring buffer (GEN3_RACE_TRACE); no-op when off

# Self-play opponent re-decide budget. When the live battle advances under the opponent's
# decision (POKE_LOOP parses an in-flight turn-resolution during the model forward), the
# opponent re-decides on the now-current request instead of raising — the server is waiting on
# our move, so the battle settles within a couple of re-decides. See choose_move().
_OPP_REDECIDE_MAX = 8

# The policy's action sample — per-instance torch generator (OPT-IN). See
# `gen3_policy_sample_rng_v1` in `RLPlayer.__init__` for the coupling this closes.
_POLICY_SEED_ENV = "GEN3AI_POLICY_SEED"


def _resolve_policy_seed(policy_seed: Optional[int]) -> Optional[int]:
    """The seed for a player's private sampling generator, or ``None`` for "use torch's shared
    default generator" — which is what keeps the default byte-identical.

    An unparseable ``$GEN3AI_POLICY_SEED`` raises rather than silently falling back: a seed that
    was meant to be set and silently was not would make a paired arm look reproducible while it
    is not."""
    if policy_seed is not None:
        return int(policy_seed)
    env = os.environ.get(_POLICY_SEED_ENV)
    if env is None or env == "":
        return None
    try:
        return int(env)
    except ValueError as exc:
        raise ValueError(f"${_POLICY_SEED_ENV}={env!r} is not an integer seed") from exc


class Gen3Player(Player):
    """Base player: embeds battle state and maps actions using Gen 3 logic.

    Defaults ``battle_class=Gen3Battle`` so every Gen3 player (RL, eval, replay,
    stat-tracking) gets the revealed-order event log + ``live_view()`` for free.
    Callers can still override (e.g. a plain-Battle baseline) via the kwarg.
    """

    #: The connect-or-raise deadline, in seconds. A CLASS attribute so a test double built
    #: without ``Player.__init__`` still has one, and so a caller can raise it for a slow link
    #: (``player.connect_timeout_s = 90``) or per subclass. See ``DEFAULT_CONNECT_TIMEOUT_S``.
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S

    def __init__(self, observation_encoder=None, mappings=None,
                 stall_config: Optional[StallConfig] = None,
                 battle_class=Gen3Battle,
                 connect_timeout_s: Optional[float] = None, **kwargs):
        super().__init__(battle_class=battle_class, **kwargs)
        if connect_timeout_s is not None:
            self.connect_timeout_s = float(connect_timeout_s)
        self.observation_encoder = observation_encoder
        self.mappings = mappings
        self._stall_config = stall_config or StallConfig()
        self._stall_loggers: dict[str, StallLogger] = {}
        self._trackers: dict[str, EpisodeTracker] = {}

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
            await self._await_connected(
                participant.ps_client, timeout_s=self.connect_timeout_s,
                username=getattr(participant, "username", None))
        return await super()._battle_against(*opponents, n_battles=n_battles)

    # ---- the SAME guard on the challenge / accept / ladder paths -------------------------
    # These are the three the LADDER entry point (`src/main/play.py`) actually calls, and they
    # were unguarded until 2026-09-05: `Player._{send_challenges,accept_challenges,ladder}` all
    # open with a bare `await self.ps_client.logged_in.wait()`, so a refused login sat there in
    # silence for the whole session. Overridden at the UNDERSCORE method, exactly like
    # `_battle_against`, because that is the half `handle_threaded_coroutines` runs on POKE_LOOP
    # — where the `logged_in` event and the listen task live.

    async def _send_challenges(self, opponent, n_challenges, to_wait=None):
        await self._await_connected(self.ps_client, timeout_s=self.connect_timeout_s,
                                    username=getattr(self, "username", None))
        return await super()._send_challenges(opponent, n_challenges, to_wait)

    async def _accept_challenges(self, opponent, n_challenges, packed_team=None):
        await self._await_connected(self.ps_client, timeout_s=self.connect_timeout_s,
                                    username=getattr(self, "username", None))
        return await super()._accept_challenges(opponent, n_challenges, packed_team)

    async def _ladder(self, n_games: int):
        await self._await_connected(self.ps_client, timeout_s=self.connect_timeout_s,
                                    username=getattr(self, "username", None))
        return await super()._ladder(n_games)

    @staticmethod
    async def _await_connected(client, *, timeout_s: Optional[float] = None,
                               username: Optional[str] = None) -> None:
        """Block until ``client`` is logged in, or raise — on either of TWO signals.

        1. DETERMINISTIC, and it fires instantly: the listen() task finishes WITHOUT logging
           in. A refused *connection* makes poke-env's listen() return immediately, and a
           dead-but-open socket is closed by the websocket ping; both finish the listen task.
           So 'listen task done && not logged in' is an exact signal that the connection
           failed, and it needs no timeout guess.
        2. The DEADLINE, ``timeout_s`` (default :data:`DEFAULT_CONNECT_TIMEOUT_S`). Signal 1
           alone is not sufficient and the gap is not hypothetical: a login REFUSED by
           `action.php` (the `--no-security` local server still authenticates a username that
           is registered upstream) leaves the socket open and the listen task RUNNING with
           `logged_in` never set, so this loop would spin until the caller's own battle
           deadline expired and report a phantom timeout instead of a bad login.

        The raise names the USERNAME and the SERVER, because the whole class of failure here
        is "this account cannot log in *here*" and a message without both sends the reader to
        the network. ``timeout_s=None``/``<= 0`` restores the unbounded pre-2026-09-05
        behaviour and is only for a caller that has its own bound.
        """
        if client.logged_in.is_set():
            return
        who = f"{username!r} " if username else ""
        listen_fut = getattr(client, "_listening_coroutine", None)
        if listen_fut is None:
            # No listening task ⇒ the client can never log in on its own. A battling
            # player must listen; fail loudly rather than wait forever.
            raise ShowdownConnectionError(
                f"{client.websocket_url}: {who}client has no listening task "
                f"(start_listening=False?) — it can never log in."
            )
        loop = asyncio.get_event_loop()
        deadline = (None if not timeout_s or timeout_s <= 0
                    else loop.time() + float(timeout_s))
        while not client.logged_in.is_set():
            if deadline is not None and loop.time() >= deadline:
                login_err = getattr(client, "_last_login_error", None)
                raise ShowdownConnectionError(
                    f"{client.websocket_url}: {who}did not finish logging in within "
                    f"{float(timeout_s):.0f}s"
                    + (f" — {login_err}" if login_err else
                       " — the socket is still open and the listen task is still running, so "
                       "this is a login the server never completed (a username registered on "
                       "the official ladder is REFUSED even by a --no-security local server, "
                       "which authenticates against Smogon's action.php). Try an unregistered "
                       "username, or supply the account's password.")
                )
            if listen_fut.done():
                err = None
                if not listen_fut.cancelled():
                    err = listen_fut.exception()  # usually None — listen() swallows it
                # A REFUSED LOGIN retires the socket the same way an unreachable server
                # does, so without this the message would blame the network for a wrong
                # password. `_last_login_error` is the client's own record of the cause.
                login_err = getattr(client, "_last_login_error", None)
                if login_err is not None:
                    raise ShowdownConnectionError(
                        f"{client.websocket_url}: {who}authentication was refused — {login_err}"
                    )
                raise ShowdownConnectionError(
                    f"{client.websocket_url}: {who}listen task exited without logging in — "
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

    def track_decision(self, battle) -> "tuple":
        """The faithful per-decision TRACKING side-effects — record the decision context + advance the
        HiddenPowerTracker and the ProgressClock — WITHOUT the (expensive) obs encode. ``embed_battle``
        does exactly this then encodes; a replay that only needs the obs at SOME decisions can call this
        at the rest to keep tracker state faithful for later decisions while skipping ~80% of the cost
        (the encode). Returns ``(legal, mask, tracker)``. The tracker WRITES here (record /
        update_progress_clock) are identical to ``embed_battle``'s, and the encode it skips is read-only
        on the tracker — so a later decision's encode reads identical state ⇒ a bit-for-bit identical obs
        (pinned by the search clone-parity fuzz)."""
        # Record first so the tracker's HP candidates are up-to-date BEFORE we encode the obs (mirrors
        # gen3_env.embed_battle ordering). The legality snapshot is captured once and threaded to both
        # the mask and the context.
        legal = LegalActions.from_battle(battle)
        mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)
        tracker = self._get_tracker(battle)
        if not battle.strict_view().finished and mask.sum() > 0:
            tracker.record(battle, mask, legal=legal)
            # Advance the ProgressClock for the just-completed window BEFORE encode reads it (same
            # helper gen3_env uses), so the turns_since_progress obs scalar matches what the model saw
            # during TRAINING — without this, eval / self-play opponents / play.py would always read 0.
            tracker.update_progress_clock(battle, legal)
        return legal, mask, tracker

    def embed_battle(self, battle) -> Dict[str, Any]:
        if self.observation_encoder is None:
            from agents.observation.state_encoder import load_mappings, get_observation_encoder
            if self.mappings is None:
                self.mappings = load_mappings()
            self.observation_encoder = get_observation_encoder(self.mappings)
        legal, mask, tracker = self.track_decision(battle)

        # Thread the same legality snapshot into the encoder for its trapped / maybe_trapped
        # reactive bits (avoids a second LegalActions.from_battle this decision).
        obs = self.observation_encoder.encode(
            battle, hp_tracker=tracker.hidden_power_tracker, legal=legal,
            progress_clock=tracker.progress_clock,
            recency=tracker.recency,
            pair_history=tracker.pair_history,
            event_window=tracker.event_window,
            # gen3_obs_assembler_v1 — the per-battle tracker owns the cache, so concurrent
            # battles on one player never share one.
            assembler=tracker.obs_assembler(self.observation_encoder.dimension),
        )


        return {
            # gen3_frame_deletion_v1: the obs IS the encoder output (no appended tail).
            "observation": obs,
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

    #: Sampling-stream state (`gen3_policy_sample_rng_v1`). The CLASS defaults say "unseeded" —
    #: i.e. use torch's shared default generator, which is what this has always done — so a player
    #: built by bypassing ``__init__`` behaves exactly as before instead of raising.
    _policy_seed = None
    _policy_gens: dict = {}

    def __init__(self, model, team, battle_format, server_configuration,
                 mappings=None, account_configuration=None,
                 stall_config: Optional[StallConfig] = None,
                 max_concurrent_battles=10,
                 stochastic: bool = True, temperature: float = 1.0,
                 policy_seed: Optional[int] = None, **kwargs):
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
        # ── The action sample — per-instance torch generator (OPT-IN) [gen3_policy_sample_rng_v1] ──
        # GLOBAL-RANDOM COUPLING, in torch rather than in `random`. `Categorical.sample()` draws
        # from torch's process-wide DEFAULT generator, which every RLPlayer in the process shares
        # — and self-play puts TWO of them in one battle, interleaved by the bridge, while a
        # paired eval puts two ARMS in one process. Same genre as the staller's Protect coin
        # (`agents/opponents.py`, 4437c85) and it bites harder: this is not a conditional coin,
        # it is EVERY stochastic decision, and `stochastic=True` is the default for the pool
        # opponents and the stable cross-run opponents.
        # DEFAULT IS UNCHANGED: with no seed, `_policy_generator` returns None and the sampling
        # call is the same `Categorical.sample()` on the same shared stream.
        self._policy_seed = _resolve_policy_seed(policy_seed)
        self._policy_gens: dict = {}     # torch device → that device's private generator

    def _policy_generator(self, device):
        """This player's private sampling generator for `device`, or None when unseeded (which
        means "use torch's shared default generator", i.e. today's behaviour).

        Per-device because the generator must live on the tensor's device and an opponent may be
        constructed on cpu while a trainee samples on cuda; built lazily so an unseeded player
        allocates nothing. See `gen3_policy_sample_rng_v1` in ``__init__``."""
        if self._policy_seed is None:
            return None
        # `setdefault` on the INSTANCE dict, so the class-level empty default above can never be
        # mutated into shared state — the usual mutable-class-attribute footgun, closed here
        # rather than argued about.
        gens = self.__dict__.setdefault("_policy_gens", {})
        key = str(device)
        gen = gens.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self._policy_seed))
            gens[key] = gen
        return gen

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
                cat = torch.distributions.Categorical(logits=sample_logits)
                gen = self._policy_generator(sample_logits.device)
                if gen is None:
                    # DEFAULT — the shared torch global generator, exactly as before.
                    idx = cat.sample().item()
                else:
                    # `cat.probs` is the SAME tensor `cat.sample()` would feed to
                    # `torch.multinomial`, so the only difference between the branches is which
                    # generator is drawn from — no re-derived softmax, no last-bit drift.
                    idx = torch.multinomial(cat.probs, 1, True, generator=gen).item()
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
                # `battle` is passed so a β slot that is ALREADY REVEALED can be named from the
                # board rather than from the species posterior — see `_opp_intent`.
                "opp_intent": self._opp_intent(battle),
                # The distributional value head's predicted RETURN DISTRIBUTION (per-atom probs; None
                # unless --value-dist-mode != none) — the prober renders the histogram + spread/PIT.
                "value_dist": self._value_dist(),
                # The opp-ACTIVE move-belief posterior (sigmoid; None unless --move-belief-mode != off) +
                # the believed opp DERIVED stats [6,5] (None unless --spread-belief) — so the prober's
                # across-battle (axis B) belief trajectory decodes WITHOUT re-running the model on old runs.
                "move_logits": self._move_belief_active_row(),
                "spread_belief": self._spread_belief(),
            }
        return idx, probs, mask

    def _move_belief_active_row(self) -> Optional[list]:
        """The opp-ACTIVE slot's move-belief posterior (forensic trace only) — `sigmoid` over the
        extractor's stashed `last_move_belief_logits` at `last_opp_active_local` (the active opp mon, whose
        believed moveset the across-battle trajectory tracks). None when the move-belief head is off
        (`--move-belief-mode none`). A lean per-decision row (n_moves floats), parallel to `value_dist`."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = extractor.last_move_belief_logits if extractor is not None else None
        if logits is None:
            return None
        active = extractor.last_opp_active_local
        a = int(active[0].item()) if active is not None else 0
        a = max(0, min(a, int(logits.shape[1]) - 1))
        return torch.sigmoid(logits[0, a]).detach().cpu().numpy().tolist()

    def _spread_belief(self) -> Optional[list]:
        """The believed opp-ACTIVE DERIVED stats `[atk,def,spa,spd,spe]` (forensic trace only) — the
        opp-active row of the extractor's stashed `last_spread_belief` (the DamageOperator's stat input).
        Active-row (not the full [6,5]) so the across-battle trajectory can read the believed opp-active
        Atk/Spe WITHOUT a separate active-index array — parallel to `_move_belief_active_row`. None when
        `--spread-belief` is off. (The prober's per-decision spread PANEL re-runs the model for the full
        [6,5] vs the privileged true spread; this lean capture is for the offline axis-B trajectory.)"""
        extractor = getattr(self.model.policy, "features_extractor", None)
        sb = extractor.last_spread_belief if extractor is not None else None
        if sb is None:
            return None
        active = extractor.last_opp_active_local
        a = int(active[0].item()) if active is not None else 0
        a = max(0, min(a, int(sb.shape[1]) - 1))
        return sb[0, a].detach().cpu().numpy().tolist()

    def _win_prob(self) -> Optional[float]:
        """P(win) from the win-probability head (forensic trace only). Reads the logit the extractor
        stashed on this same forward (``last_win_prob_logits``); None when the head is off
        (``--win-prob-mode none``). sigmoid(logit) ⇒ probability the current state leads to a win."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = extractor.last_win_prob_logits if extractor is not None else None
        if logits is None:
            return None
        return float(torch.sigmoid(logits[0, 0]).item())

    def _opp_intent(self, battle=None) -> Optional[dict]:
        """ALPHA/BETA as NAMED, ranked options (forensic trace only) — the interpretability payload.

        This is the deliverable the whole opponent-intent design is justified on independently of
        Elo: a turn where the model played around a Fire Blast and one where it never saw the move
        coming look IDENTICAL in every existing view. Reads the logits + seat move-nums the
        extractor stashed on this same forward; None when the heads are off.

        The owner constraint — the model may only ever point at options it can NAME — is enforced
        here by construction: every entry is rendered through the move dex, and a seat the belief
        did not fill is dropped rather than shown as an anonymous index.

        **β's naming rule (`gen3_beta_revealed_naming_v1`), and it has exactly two branches:**

        - a slot the board has already REVEALED is named from `battle.opponent_team` (the encoder's
          own opp-slot order, read through `ObservationEncoder.get_team_list` so the mapping cannot
          drift from the obs), and carries ``"revealed": true``;
        - a still-HIDDEN slot is named by the model's OWN species posterior
          (`belief_decode.top_species_per_slot`) — the same content-addressing `β`'s target uses —
          and carries ``"revealed": false`` (`species: None` when there is no species head at all).

        The posterior used to name BOTH, and that was a display defect that produced a wrong
        research conclusion. `β`'s candidate mask is alive-and-not-active, which INCLUDES revealed
        bench mons, and the species aux only supervises the *believed* slots — so on a revealed
        slot the posterior is un-trained. Measured over a 843-battle sentinel sweep (2026-08-19):
        the rendered name was a mon not on the opponent's team at all in **73.3% of 6,876 pivots**
        (88.3% on revealed slots), which read as "β predicts porygon2" on a turn where β's slot was
        the revealed Salamence and β was CORRECT. Naming a revealed slot from the board is not a
        nicety; it is the difference between reading the pointer and reading a different head.

        The block lands in the trace verbatim (`BattleRecorder.record` → the summary invocation's
        `opp_intent`), which is what the prober's `/battle` replay and `analyze` read.
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None:
            return None
        alogits = extractor.last_alpha_logits
        seat_nums = extractor.last_alpha_seat_nums
        if alogits is None or seat_nums is None:
            return None
        from agents.model.opp_intent import render_alpha
        from agents.gen3_data import moves as _gm
        try:
            _by_num = {int(rec["num"]): rec.get("name") or mid
                       for mid, rec in _gm.raw().items() if "num" in rec}
        except Exception:
            _by_num = {}
        probs = torch.softmax(alogits[0], dim=-1)
        alpha = render_alpha(probs, seat_nums[0], lambda n: _by_num.get(int(n)))
        # Rounded on the way to disk: these ride EVERY captured decision of every captured battle,
        # and 15 significant figures of a display probability is trace weight with no reader.
        out = {"alpha": [{"name": r["name"], "p": round(float(r["p"]), 4)} for r in alpha]}
        blogits = extractor.last_beta_logits
        if blogits is not None:
            bp = torch.softmax(blogits[0], dim=-1)
            named = self._slot_species()
            board = self._revealed_opp_species(battle)
            rows = []
            for i in range(bp.shape[0]):
                if not torch.isfinite(blogits[0, i]):
                    continue
                seen = board[i] if i < len(board) else None
                rows.append({"slot": i, "p": round(float(bp[i]), 4),
                             "species": seen if seen is not None else named.get(i),
                             # ADDED, never substituted: a reader that predates this key sees the
                             # old shape unchanged, and its absence is what marks an OLD trace as
                             # posterior-named (see `engine.build_opp_intent`).
                             "revealed": seen is not None})
            rows.sort(key=lambda r: -r["p"])
            out["beta"] = rows[:4]
        return out

    @staticmethod
    def _revealed_opp_species(battle) -> list:
        """`[species | None]` in OBS-SLOT order for the opponent mons the board has revealed.

        Read through `ObservationEncoder.get_team_list` — the encoder's OWN opp-team iteration —
        rather than re-deriving `list(battle.opponent_team.values())` here, so obs slot `k` and
        `β` candidate `k` cannot drift apart by a later change to one of them. Validated
        7560/7560 against the belief block's hidden-slot set (2026-08-19).

        Empty on any path with no battle to read (the trace then carries only posterior names,
        flagged as such), and defensive against a partially-built battle: a name that cannot be
        read is `None`, which falls back to the posterior rather than raising inside a
        trace-capture helper."""
        if battle is None:
            return []
        try:
            from agents.observation.base import ObservationEncoder
            team = ObservationEncoder.get_team_list(battle, is_opponent=True)
        except Exception:      # noqa: BLE001 — a forensic label is never worth crashing a battle for
            return []
        return [(getattr(m, "species", None) or None) for m in team]

    def _slot_species(self) -> dict:
        """`{opp slot -> species name}` from the model's OWN species posterior — what NAMES a
        still-HIDDEN `β` slot. Empty when the checkpoint has no species-belief head (then `β` rows
        carry a bare index, which is honest: nothing in the model can say what that slot holds).

        NOT the namer for a REVEALED slot — the posterior is un-supervised there, so it names a mon
        that is usually not even on the opponent's team. See `_opp_intent`."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = extractor.last_belief_logits if extractor is not None else None
        if logits is None or "species" not in logits:
            return {}
        from agents.inference.belief_decode import top_species_per_slot
        rows = top_species_per_slot(logits["species"][0].detach().cpu().numpy())
        return {int(r["slot"]): r["species"] for r in rows}

    def _value_dist(self) -> Optional[list]:
        """The distributional value head's predicted return distribution (forensic trace only) — the
        per-atom softmax over ``last_value_dist_logits`` stashed on this same forward; None when the head
        is off (``--value-dist-mode none``). The prober reconstructs the atom support from
        model_config.json's value_dist_vmin/vmax/bins to render the histogram + mean/std/entropy/PIT."""
        extractor = getattr(self.model.policy, "features_extractor", None)
        logits = extractor.last_value_dist_logits if extractor is not None else None
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
        if extractor is None:
            return None
        logits = extractor.last_belief_logits
        mask = extractor.last_opp_believed_mask
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
