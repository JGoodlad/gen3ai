"""The search-wrapped eval player — a depth-1 search sitting on top of the trained policy.

:class:`SearchDividendPlayer` is an ordinary :class:`~agents.inference.player.RLPlayer` in every
respect that the battery measures: same obs, same mask, same action mapper, same greedy policy.
The ONLY difference is what it does with the decision — the ``base`` arm plays the policy's
argmax (so it is the literal control, not a re-implementation of one) and the search arms replace
it with :meth:`SearchEngine.choose`.

Three wiring facts, each of which is load-bearing:

* **``choose_move`` is ``async``.** ``materialize_branches`` refuses to run on ``POKE_LOOP``, and
  a sync ``choose_move`` runs there — so the search executes in a worker thread and this method
  awaits it. poke-env supports that natively (``player.py``: ``if isinstance(choice, Awaitable):
  choice = await choice``), and awaiting frees POKE_LOOP for the replay feed the search needs.
* **The α publication is read BEFORE the search.** Every critic forward the search runs clobbers
  the extractor's per-forward stash, so α is captured off the live decision's own forward and
  never re-read. Same discipline ``RLPlayer._opp_intent`` follows.
* **Our legal choice strings come from the REAL action mapper** (``action_to_order``), not from a
  re-derivation of legality. The search therefore branches on exactly the actions the policy could
  have taken, and an action-mapping change cannot silently desynchronize the two — the
  ``op move-order`` bug class in miniature.

The player also owns the LIVE reconstruction record (:mod:`record`), because the record's three
inputs — the pinned seed, the two packed teams, and the committed choices — are only all visible
from inside the battle.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from agents.inference.player import RLPlayer
from main.search_dividend.alpha import alpha_publication
from main.search_dividend.record import LiveRecordBuilder, set_active_builder
from main.search_dividend.search import DecisionResult, SearchEngine


class SearchDividendPlayer(RLPlayer):
    """An ``RLPlayer`` whose decision is (optionally) chosen by a depth-1 search.

    ``opp_player`` is the opponent object this player is matched against — used ONLY on the
    ORACLE arm, to read its ``_current_packed_team``. It is passed explicitly rather than
    discovered so that an honest-arm player literally cannot reach the truth: the engine raises if
    handed a team on any arm but ``oracle``.
    """

    def __init__(self, *args, engine: SearchEngine, battle_format: str,
                 opp_player=None, **kwargs):
        kwargs.setdefault("stochastic", False)     # the battery measures the GREEDY policy
        super().__init__(*args, battle_format=battle_format, **kwargs)
        self.engine = engine
        self._fmt = battle_format
        self._opp_player = opp_player
        self._builder_for: Dict[str, LiveRecordBuilder] = {}
        self._pending: Optional[LiveRecordBuilder] = None
        self._history: Dict[str, List[int]] = {}
        # Battles whose OUR-ACTION history can no longer be reconstructed (a `/choose default`
        # went out, or an order mapping failed). The search cannot branch from a state it cannot
        # replay, so it declines for the rest of the battle — loudly, with its own reason.
        self._desynced: set = set()
        self.decisions: List[dict] = []            # one row per decision, for the results file

    # -- per-battle bookkeeping ---------------------------------------------

    def open_battle(self, seed: str, chunk_sink, our_side: str = "p1") -> LiveRecordBuilder:
        """Start a record for the battle about to be played.

        The battle TAG is minted inside ``run_local_battles``, so the builder is created here and
        adopted by whichever battle's first decision arrives — safe because the driver plays one
        battle at a time (:func:`record.set_active_builder` enforces that)."""
        b = LiveRecordBuilder(battle_format=self._fmt, seed=seed,
                              trainee_username=self.username,
                              chunk_sink=chunk_sink, our_side=our_side)
        self._pending = b
        return b

    def _builder(self, battle) -> Optional[LiveRecordBuilder]:
        tag = battle.battle_tag
        b = self._builder_for.get(tag)
        if b is None:
            b = self._pending
            if b is None:
                return None
            b.battle_tag = tag
            b.our_side = battle.player_role or b.our_side
            self._builder_for[tag] = b
        if not b.ready():
            side = battle.player_role
            other = "p2" if side == "p1" else "p1"
            mine = getattr(self, "_current_packed_team", None)
            theirs = getattr(self._opp_player, "_current_packed_team", None)
            if not mine or not theirs:
                return None
            b.set_player(side, self.username, mine)
            b.set_player(other, self._opp_player.username, theirs)
        return b

    # -- the decision --------------------------------------------------------

    async def choose_move(self, battle):            # type: ignore[override]
        forfeit = self._handle_stall(battle, "SEARCH_DIVIDEND_STALL")
        if forfeit:
            return forfeit
        side = battle.player_role
        tag = battle.battle_tag
        builder = self._builder(battle)
        idx, _probs, mask = self._predict_best_action(battle, stochastic=False, need_aux=False)
        if idx is None:
            # poke-env will send `/choose default`, whose action INDEX we do not know — so our
            # action history can no longer be reconstructed and every later search in this battle
            # would branch from the wrong state. Poison the battle rather than let the
            # materializer discover it as an opaque desync.
            self._n_defaults += 1
            self._desynced.add(tag)
            self._log_decision(battle, None, None, "policy_default")
            return self.choose_default_move()
        # Read α off THIS forward, before any search forward clobbers the stash.
        pub = _safe_alpha(self.model)

        history = self._history.setdefault(tag, [])
        if self.engine.cfg.arm == "base":
            result = self._search(battle, side, builder, history, mask, int(idx), pub)
        else:
            # OFF POKE_LOOP. `materialize_branches` drives a replay player THROUGH this loop and
            # blocks on the result, so running it here would deadlock the loop against itself
            # (`obs_materializer._refuse_poke_loop` refuses loudly rather than hanging). Awaiting
            # an executor hands the loop back for exactly the duration of the search.
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._search, battle, side, builder, history, mask, int(idx), pub)
        chosen = int(result.action)
        try:
            order = self.action_to_order(chosen, battle)
        except Exception:                            # noqa: BLE001 — a stale/illegal mapping
            self._n_defaults += 1
            self._desynced.add(tag)
            self._log_decision(battle, result, None, "order_failed")
            return self.choose_default_move()
        # ⚠️ `_predict_best_action` already advanced the tracker with the POLICY's index, and the
        # search may have picked a different one. `_last_action` feeds OUR OWN turn-history obs,
        # so leaving it would make every subsequent observation claim we played a move we did not
        # — a silent GIGO of exactly the shape this project has eaten before. Overwrite it with
        # what we are actually about to send.
        self._get_tracker(battle).advance(chosen)
        history.append(chosen)
        self._log_decision(battle, result, chosen, None)
        return order

    def _search(self, battle, side, builder, history, mask, policy_action,
                pub) -> DecisionResult:
        from main.search_dividend.budget import RealizedWidths

        cfg = self.engine.cfg
        if cfg.arm == "base":
            return self.engine.choose(record=None, side=side, turn=battle.turn,
                                      our_history=history, our_tokens={},
                                      observed_our_lines=(), pub=pub,
                                      policy_action=policy_action)
        if battle.battle_tag in self._desynced:
            return DecisionResult(policy_action, "history_desync",
                                  RealizedWidths(planned={}, n_our_actions=0),
                                  policy_action=policy_action)
        if builder is None or not builder.ready() or battle.force_switch:
            reason = "record_unavailable" if builder is None or not builder.ready() \
                else "not_move_selection"
            return DecisionResult(policy_action, reason,
                                  RealizedWidths(planned={}, n_our_actions=0),
                                  policy_action=policy_action)
        tokens = self._legal_tokens(battle, mask)
        if len(tokens) < 2:
            return DecisionResult(policy_action, "not_move_selection",
                                  RealizedWidths(planned={}, n_our_actions=len(tokens)),
                                  policy_action=policy_action)
        record = builder.build()
        opp_true = (getattr(self._opp_player, "_current_packed_team", None)
                    if cfg.arm == "oracle" else None)
        return self.engine.choose(
            record=record, side=side, turn=int(battle.turn), our_history=history,
            our_tokens=tokens, observed_our_lines=builder.our_lines, pub=pub,
            policy_action=policy_action, opp_true_packed=opp_true)

    def _legal_tokens(self, battle, mask) -> Dict[int, str]:
        """``{action_index: sim choice string}`` for every legal action, via the REAL mapper.

        A mapping that raises for an index the mask calls legal is dropped rather than guessed —
        the search then simply does not branch on it, and the policy's own action is still in the
        set (it came from the same mask)."""
        out: Dict[int, str] = {}
        for i, ok in enumerate(mask):
            if not ok:
                continue
            try:
                msg = self.action_to_order(int(i), battle).message
            except Exception:                        # noqa: BLE001
                continue
            if msg.startswith("/choose "):
                out[int(i)] = msg[len("/choose "):]
        return out

    def _log_decision(self, battle, result: Optional[DecisionResult], chosen, note) -> None:
        row = {"battle": battle.battle_tag, "turn": int(battle.turn), "chosen": chosen}
        if note:
            row["note"] = note
        if result is not None:
            row["fallback"] = result.fallback
            row["policy_action"] = result.policy_action
            row["changed"] = result.changed
            row["widths"] = result.widths.as_dict()
            # A counted fallback whose REASON cannot be read is only half the discipline: it says
            # the search declined without saying what to fix. Carry the message for the two
            # reasons that have one.
            err = (result.diagnostics or {}).get("error")
            if err:
                row["error_detail"] = err
        self.decisions.append(row)


def _safe_alpha(model):
    """α, or ``None`` — an α-off checkpoint and a width mismatch are different failures.

    A width mismatch RAISES inside :func:`alpha_publication` by design (clause 3: fail loud), and
    that is a configuration error worth surfacing, so it is re-raised. Anything else degrades to
    the uniform absence fallback."""
    extractor = getattr(model.policy, "features_extractor", None)
    return alpha_publication(extractor)


async def play_one_battle(player: SearchDividendPlayer, opponent, *, battle_format: str,
                          seed: str, impl: str) -> dict:
    """Play ONE battle with the live record wired up, and return its outcome row.

    One battle per call, deliberately: ``chunk_sink`` is not side-deduped across concurrent
    battles (``local_battle_runner``'s own docstring says so) and the record's command order is
    only unambiguous for a single battle in flight. The seed is PINNED because the record needs
    the RESOLVED seed at decision time, not at ``__RECON__`` time.
    """
    from utils.bridge.local_battle_runner import run_local_battles

    chunk_sink: list = []
    builder = player.open_battle(seed, chunk_sink, our_side="p1")
    set_active_builder(builder)
    won_before, fin_before = player.n_won_battles, player.n_finished_battles
    n_dec_before = len(player.decisions)
    try:
        await run_local_battles(player, opponent, 1, battle_format=battle_format,
                                seed=None, concurrency=1, chunk_sink=chunk_sink, impl=impl,
                                start_extra={"seed": seed})
    finally:
        set_active_builder(None)
    return {
        "seed": seed,
        "won": player.n_won_battles - won_before,
        "finished": player.n_finished_battles - fin_before,
        "battle_tag": builder.battle_tag,
        "n_commands": builder.n_commands,
        "decisions": player.decisions[n_dec_before:],
    }
