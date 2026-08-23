from typing import Callable, Optional
import numpy as np

from agents.battle.live_view import LivePokemon, LiveView
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.reward_tracker import RewardTracker
from agents.training.slot_registry import SlotRegistry
from agents.training.reward_manager import Gen3RewardManager
from agents.gen3_mechanics import boosts_str as _boosts_str_fn


class BattleRecorder:
    """
    Records every model invocation for a single battle and exports a
    human-readable summary JSON.

    Call record() once per choose_move(), finalize() at battle end,
    then to_summary() to get the exportable dict.

    Each invocation entry is self-contained — action labels are inlined,
    not referenced from a legend — so a human can read any turn in isolation.

    Reward computation is delegated to RewardTracker, which shares the same
    SlotRegistries so BattleContext slot lookups stay consistent.
    """

    def __init__(self, battle_tag: str, reward_fn_factory: Callable[[], RewardFunction] = Gen3RewardManager,
                 gamma: float = 0.99):
        self.battle_tag = battle_tag
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._invocations: list[dict] = []
        self._pending_entry: Optional[dict] = None
        self._tracker = RewardTracker(reward_fn_factory, self._our_slots, self._opp_slots)
        # Optional raw model I/O per invocation, parallel to _invocations
        # (obs/logits/value the model saw). Populated only when record() gets them;
        # exported via states_arrays() for offline forensic replay.
        self._states: list[dict] = []
        # The chosen action index per invocation (agent-side, always available).
        # Exported in states_arrays() so an offline replay can re-advance the
        # turn-history tracker with the exact actions the live player took —
        # required for bit-faithful obs materialization (obs_materializer.py).
        self._actions_taken: list[int] = []
        # The LEGAL-ACTION MASK per invocation (agent-side, always available — it is what the
        # player masked its own logits with). Exported in states_arrays() as `action_mask`.
        # `gen3_audit_mask_recovery_v1`: the stored `logits` are PRE-mask (inference/player.py
        # keeps the -1e9 offset in a local), so every offline consumer that "recovered" the mask
        # as `logits > -1e8` silently got ALL-LEGAL. The mask was only ever on disk as the
        # summary's per-label `valid` flags — recoverable, but by a label-keyed dict whose
        # index alignment is implicit. Writing it as an array makes the npz self-contained.
        self._action_masks: list[np.ndarray] = []
        # Per-decision TD residual δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the SAME formula the
        # prober uses (main/prober/session/core.py::ProbeSession._td), the single source of truth. Computed live at
        # the NEXT record() (when complete_pending finalizes reward(t) and the current state
        # carries V(s_{t+1})), so the last decision yields no δ (no next state — matches the
        # prober leaving td_residual(last)=None). The left tail of these is the "critic got
        # blindsided" signal (#4) the eval cycle folds into eval/td_resid_tail_*.
        self._gamma = float(gamma)
        self._td_residuals: list[float] = []
        self._prev_value: Optional[float] = None  # V(s_t) carried across decisions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, battle, action_idx: int, probs: np.ndarray, mask: np.ndarray,
               state: Optional[dict] = None) -> None:
        """Record a model invocation. Call from choose_move() after prediction.

        `state` (optional): {"obs", "logits", "value"} as captured by
        RLPlayer._predict_best_action — the raw model I/O for this turn. Stored
        parallel to the invocation so a saved battle can be replayed exactly.
        """
        # Build the strict view ONCE; the recorder reads all current-board state
        # through its LiveView (and legality through .legal) — never the raw battle.
        # The raw `battle` is still passed to BattleContext/RewardTracker below, which
        # own their poke-env reads behind their own boundaries.
        view = battle.strict_view()
        live = view.live
        legal = view.legal
        curr_ctx = self._build_ctx(battle, mask, legal)
        self._states.append(state or {})
        self._actions_taken.append(int(action_idx))
        self._action_masks.append(np.asarray(mask).astype(bool).reshape(-1))

        if self._pending_entry is not None:
            prev_ctx = self._tracker.pending_ctx
            delta, reward = self._tracker.complete_pending(curr_ctx, battle)
            self._fill_pending_outcome(prev_ctx, curr_ctx, delta, reward, live)
            # δ(prev) = r(prev) + γ·V(s_now) − V(s_prev): the critic's surprise on the just-closed
            # transition. `reward` is the scalar total for it (== outcome.reward.total the prober
            # reads); `_prev_value` is V(s_prev) stashed last call; this call's value is V(s_now).
            v_next = (state or {}).get("value")
            if self._prev_value is not None and v_next is not None and reward is not None:
                self._td_residuals.append(reward + self._gamma * float(v_next) - self._prev_value)

        chosen = self._action_label(action_idx, live, legal)
        our_mon = live.ours.active
        opp_mon = live.opp.active
        our_status = self._mon_display_status(our_mon)
        opp_status = self._mon_display_status(opp_mon)
        our_boosts = _boosts_str_fn(our_mon)
        opp_boosts = _boosts_str_fn(opp_mon)

        our_section: dict = {"species": curr_ctx.our_active, "hp": self._our_hp_pct(curr_ctx)}
        if our_status:
            our_section["status"] = our_status
        if our_boosts:
            our_section["boosts"] = our_boosts
        our_section["bench"] = self._our_bench_summary(live)

        opp_section: dict = {"species": curr_ctx.opp_active, "hp": self._opp_hp_pct(curr_ctx)}
        if opp_status:
            opp_section["status"] = opp_status
        if opp_boosts:
            opp_section["boosts"] = opp_boosts
        opp_section["bench"] = self._opp_bench_summary(live)

        # The model's top-k species guess for each still-hidden opp slot (only when the hidden-opponent
        # belief is on AND a slot is unrevealed; see RLPlayer._decode_belief). Sits right after `opp`
        # — "the board, then what we believe is still hidden" — and is omitted entirely otherwise.
        belief = (state or {}).get("belief")
        # What the model expects the opponent to DO this turn — `α` (a ranked, NAMED distribution over
        # their believed moves + SWITCH) and `β` (given a switch, which mon comes in), from the v67
        # opponent-intent heads (`RLPlayer._opp_intent`; only when `--opp-intent-coef>0`). It sits
        # after `belief` because that is the reading order: the board, what we think is hidden, then
        # what we think they will do with it. Omitted entirely when the heads are off, so an
        # intent-off run's trace is byte-unchanged.
        opp_intent = (state or {}).get("opp_intent")
        entry = {
            "i": len(self._invocations) + 1,
            "turn": curr_ctx.turn,
            "phase": curr_ctx.phase,
            "chosen": chosen,
            "our": our_section,
            "opp": opp_section,
            **({"belief": belief} if belief else {}),
            **({"opp_intent": opp_intent} if opp_intent else {}),
            "outcome": None,
            "actions": self._all_action_labels(live, probs, mask, legal),
        }

        self._tracker.begin_turn(curr_ctx, action_idx, view.event_cursor)
        self._pending_entry = entry
        # Carry V(s_now) so the NEXT record() can close δ for this decision.
        self._prev_value = (state or {}).get("value")

    def td_residuals(self) -> list[float]:
        """The per-decision TD residuals δ closed so far (one per non-terminal decision).

        Empty when no value was captured (the cheap fast path) or fewer than two decisions ran.
        The eval cycle pools these across an opponent's captured battles → a tail statistic
        (``eval/td_resid_tail_*``). Read it BEFORE the recorder is discarded at battle end."""
        return list(self._td_residuals)

    def states_arrays(self) -> dict:
        """Stack the per-invocation raw model I/O into npz-ready arrays, aligned
        index-for-index with to_summary()['invocations']. Returns {} if states
        were never captured. `has_state` flags which turns carry real data.
        `actions` is the chosen action index per invocation (always real data —
        it's the agent-side input an offline obs replay needs to re-advance the
        turn-history tracker; see obs_materializer.py)."""
        if not any(self._states):
            return {}
        obs_dim = next(len(s["obs"]) for s in self._states if s)
        n_act = next(len(s["logits"]) for s in self._states if s)
        T = len(self._states)
        obs = np.zeros((T, obs_dim), dtype=np.float32)
        logits = np.zeros((T, n_act), dtype=np.float32)
        values = np.zeros(T, dtype=np.float32)
        # P(win) from the win-probability head, parallel to `values`. NaN = no head (--win-prob-mode
        # none) / not captured, so the prober can distinguish "unavailable" from a real P(win)=0.0.
        win_probs = np.full(T, np.nan, dtype=np.float32)
        # Distributional value head's per-atom return distribution [T, bins], parallel to `values`. The
        # key is OMITTED entirely when the head is off (no state carried a distribution) so the prober's
        # KeyError guard reads "unavailable"; a captured-but-headless row stays all-NaN.
        vd_bins = next((len(s["value_dist"]) for s in self._states
                        if s and s.get("value_dist") is not None), 0)
        value_dist = np.full((T, vd_bins), np.nan, dtype=np.float32) if vd_bins else None
        # Move-belief posterior (opp-active row, [T, n_moves]) + believed opp-active spread ([T, 5]), parallel
        # to value_dist — for the prober's across-battle belief trajectory (axis B) WITHOUT re-running the
        # model (move-belief entropy decay + believed opp-active Atk/Spe). Each key is OMITTED when its head
        # is off (no state carried it); a captured-but-headless row = NaN.
        mb_n = next((len(s["move_logits"]) for s in self._states
                     if s and s.get("move_logits") is not None), 0)
        move_logits = np.full((T, mb_n), np.nan, dtype=np.float32) if mb_n else None
        sb_shape = next((np.asarray(s["spread_belief"]).shape for s in self._states
                         if s and s.get("spread_belief") is not None), None)
        spread_belief = np.full((T,) + tuple(sb_shape), np.nan, dtype=np.float32) if sb_shape else None
        has_state = np.zeros(T, dtype=np.int8)
        for i, s in enumerate(self._states):
            if not s:
                continue
            obs[i] = s["obs"]
            logits[i] = s["logits"]
            values[i] = s.get("value", 0.0)
            wp = s.get("win_prob")
            if wp is not None:
                win_probs[i] = float(wp)
            if value_dist is not None:
                vd = s.get("value_dist")
                if vd is not None:
                    value_dist[i] = np.asarray(vd, dtype=np.float32)
            if move_logits is not None:
                ml = s.get("move_logits")
                if ml is not None:
                    move_logits[i] = np.asarray(ml, dtype=np.float32)
            if spread_belief is not None:
                sb = s.get("spread_belief")
                if sb is not None:
                    spread_belief[i] = np.asarray(sb, dtype=np.float32)
            has_state[i] = 1
        actions = np.asarray(self._actions_taken, dtype=np.int16)
        # `action_mask` [T, n_act] bool — the legality the player actually masked with, so an
        # offline audit never has to infer it. See __init__ (gen3_audit_mask_recovery_v1).
        action_mask = np.zeros((T, n_act), dtype=bool)
        for i, m in enumerate(self._action_masks):
            action_mask[i, :min(n_act, len(m))] = m[:n_act]
        out = {"obs": obs, "logits": logits, "values": values, "win_probs": win_probs,
               "has_state": has_state, "actions": actions, "action_mask": action_mask}
        if value_dist is not None:
            out["value_dist"] = value_dist
        if move_logits is not None:
            out["move_logits"] = move_logits
        if spread_belief is not None:
            out["spread_belief"] = spread_belief
        return out

    def finalize(self, battle) -> None:
        """Complete the last pending invocation at battle end."""
        if self._pending_entry is None:
            return

        view = battle.strict_view()
        live = view.live
        prev_ctx = self._tracker.pending_ctx
        our_hp, opp_hp = self._terminal_hp(live)
        terminal_ctx, delta, reward = self._tracker.finalize(battle)

        # WHICH mon fainted, from the live board rather than from the decision-time active — the
        # last turn of a battle is the one most likely to end on a switch-in dying, and it is the
        # turn a reader looks at first. Same rule as the per-turn path (`_newly_fainted`).
        final_our = frozenset(m.species for m in live.ours.mons if m.fainted)
        final_opp = frozenset(m.species for m in live.opp.mons if m.fainted)
        our_fainted = self._newly_fainted(
            prev_ctx.our_fainted_species, final_our, prev_ctx.our_active)
        opp_fainted = self._newly_fainted(
            prev_ctx.opp_fainted_species, final_opp, prev_ctx.opp_active)
        our_fainted_species = our_fainted[0] if our_fainted else prev_ctx.our_active
        opp_fainted_species = opp_fainted[0] if opp_fainted else prev_ctx.opp_active

        final_our_fainted = len(final_our)
        final_opp_fainted = len(final_opp)
        we_newly_fainted = final_our_fainted > prev_ctx.our_fainted_count
        opp_newly_fainted = final_opp_fainted > prev_ctx.opp_fainted_count

        # HP delta: for faint turns use the fainted mon's slot, not the forced switch-in.
        our_ref = our_fainted_species if we_newly_fainted else (delta.our_switch_to or prev_ctx.our_active)
        opp_ref = opp_fainted_species if opp_newly_fainted else (delta.opp_switch_to or prev_ctx.opp_active)
        our_slot = prev_ctx.our_slot_map.get(our_ref, 0)
        opp_slot = prev_ctx.opp_slot_map.get(opp_ref, 0)
        our_delta = (our_hp[our_slot] - prev_ctx.our_hp[our_slot]) * 100
        opp_delta = (opp_hp[opp_slot] - prev_ctx.opp_hp[opp_slot]) * 100

        events = []
        if we_newly_fainted:
            for sp in our_fainted:
                events.append(f"our:{sp}:fainted")
        if opp_newly_fainted:
            for sp in opp_fainted:
                events.append(f"opp:{sp}:fainted")

        self._append_status_events(events, prev_ctx, delta, live)

        if view.won:
            events.append("result:win")
        elif view.lost:
            events.append("result:loss")
        else:
            events.append("result:tie")

        breakdown = getattr(self._tracker._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": self._pending_entry["chosen"], "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": "unknown",                     "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None

    def to_summary(self, battle, step: int) -> dict:
        """Export the full battle summary as a JSON-serializable dict."""
        view = battle.strict_view()
        live = view.live
        if view.won:
            result = "WIN"
        elif view.lost:
            result = "LOSS"
        else:
            result = "TIE"

        return {
            "meta": {
                "step": step,
                "battle_id": self.battle_tag,
                "result": result,
                "turns": view.turn,
                "invocations": len(self._invocations),
            },
            "teams": {
                "ours": [
                    {
                        "species": m.species,
                        "item": m.item or "none",
                        "final_hp": f"{m.hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in live.ours.mons
                ],
                "opponent": [
                    {
                        "species": m.species,
                        "final_hp": f"{m.hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in live.opp.mons
                ],
            },
            "invocations": self._invocations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ctx(self, battle, mask: np.ndarray, legal=None) -> BattleContext:
        return BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots, legal)

    def _action_label(self, action_idx: int, live: LiveView, legal) -> str:
        team_list = live.ours.mons
        # display_move_ids (not move_ids): shows OUR Hidden Power with its TYPED id
        # ("hiddenpowergrass") instead of the wire-bare "hiddenpower" — we always know our own HP
        # type, and these are human-/prober-facing labels (the mask/mapper use the wire-truth ids).
        move_ids = list(legal.display_move_ids)

        if action_idx < 6:
            return f"switch:{team_list[action_idx].species}" if action_idx < len(team_list) else f"switch:slot{action_idx}"
        elif action_idx < 10:
            m = action_idx - 6
            return move_ids[m] if m < len(move_ids) else f"move{m}"
        return "struggle"

    def _all_action_labels(self, live: LiveView, probs: np.ndarray, mask: np.ndarray, legal) -> dict:
        team_list = live.ours.mons
        move_ids = list(legal.display_move_ids)  # typed own HP — see _action_label

        result = {}
        for i in range(11):
            if i < 6:
                label = f"switch:{team_list[i].species}" if i < len(team_list) else f"switch:slot{i}"
            elif i < 10:
                m = i - 6
                label = move_ids[m] if m < len(move_ids) else f"move{m}"
            else:
                label = "struggle"
            result[label] = {"prob": f"{probs[i] * 100:.1f}%", "valid": bool(mask[i])}
        return result

    @staticmethod
    def _mon_display_status(mon: LivePokemon | None) -> str | None:
        """Rich status string including counters and volatiles, read from a
        :class:`LivePokemon` (id-form status + ``{volatile_id: counter}``).

        Examples: "SLP(3)", "TOX(5)", "BRN", "PAR|TAUNT", "PERISH(2)|CONF"
        """
        if mon is None:
            return None
        parts = []
        status = mon.status  # id form: 'slp'/'tox'/'brn'/'par'/'frz'/'psn'/'fnt' or None
        ctr = mon.status_counter or 0
        vol = mon.volatiles  # {volatile_id: counter}
        if status == "slp":
            parts.append(f"SLP({ctr})" if ctr else "SLP")
        elif status == "tox":
            parts.append(f"TOX({ctr})" if ctr else "TOX")
        elif status is not None:
            name_map = {"brn": "BRN", "par": "PAR", "frz": "FRZ", "psn": "PSN"}
            if name := name_map.get(status):
                parts.append(name)
        effect_names = {
            "taunt": "TAUNT", "confusion": "CONF", "encore": "ENCORE",
            "attract": "ATTRACT", "disable": "DISABLE", "substitute": "SUB",
        }
        for vid, name in effect_names.items():
            if vid in vol:
                parts.append(name)
        for n, vid in [(3, "perish3"), (2, "perish2"), (1, "perish1"), (0, "perish0")]:
            if vid in vol:
                parts.append(f"PERISH({n})")
                break
        return "|".join(parts) if parts else None

    @staticmethod
    def _status_key(status_str: str | None) -> str | None:
        """Normalize for change detection — strips counter values, sorts parts."""
        if not status_str:
            return None
        return "|".join(sorted(p.split("(")[0] for p in status_str.split("|")))

    def _our_bench_summary(self, live: LiveView) -> str:
        return self._bench_summary(live.ours.active, live.ours.mons)

    def _opp_bench_summary(self, live: LiveView) -> str:
        return self._bench_summary(live.opp.active, live.opp.mons)

    def _bench_summary(self, active: LivePokemon | None, mons) -> str:
        parts = []
        for mon in mons:
            if active and mon.species == active.species:
                continue
            if mon.fainted:
                parts.append(f"{mon.species}(faint)")
            else:
                pct = f"{mon.hp_fraction * 100:.0f}%"
                status = self._mon_display_status(mon)
                parts.append(f"{mon.species}({pct},{status})" if status else f"{mon.species}({pct})")
        return ", ".join(parts)

    def _our_hp_pct(self, ctx: BattleContext) -> str:
        if ctx.our_active == "NONE":
            return "0%"
        return f"{ctx.our_hp[ctx.our_slot_map.get(ctx.our_active, 0)] * 100:.0f}%"

    def _opp_hp_pct(self, ctx: BattleContext) -> str:
        if ctx.opp_active == "NONE":
            return "?%"
        slot = ctx.opp_slot_map.get(ctx.opp_active)
        return f"{ctx.opp_hp[slot] * 100:.0f}%" if slot is not None else "?%"

    @staticmethod
    def _newly_fainted(prev_fainted, now_fainted, fallback: str) -> "list[str]":
        """EVERY species that actually fainted this turn, as a set difference.

        A faint used to be detected by COUNT and then labelled with `prev_ctx.*_active` — the mon
        that was active when the decision was made. That is the wrong mon whenever a switch
        happened on the same turn, which is not a corner case:

          * we switch Cloyster → Jolteon, the opponent's Explosion kills JOLTEON, and the trace
            records `our:cloyster:fainted` — while its own battle log, two lines above, says the
            Explosion hit Jolteon;
          * the opponent switches Claydol → Dugtrio and our Ice Beam kills DUGTRIO, recorded as
            `opp:claydol:fainted`.

        Measured on ai_v9_17_tdaux_lam3: **25 of 466 turns** named a mon that did not faint.

        The set difference also gets the case an HP-transition check would miss — a mon REVEALED
        and killed on the same turn (Dugtrio above) has no previous HP to fall from.

        `fallback` keeps the old behaviour when the sets cannot answer (a snapshot without the
        species sets, or a faint the tracker saw but neither set names): a slightly wrong label is
        still better than an empty one, and a forensic recorder must never raise into training.
        """
        gained = [sp for sp in (now_fainted or ()) if sp not in (prev_fainted or ())]
        if gained:
            # ONE SIDE CAN LOSE TWO MONS IN A TURN — measured: an opponent mon is KO'd, its forced
            # replacement switches in and dies to Spikes, both inside turn 34. The old
            # `if delta.opp_fainted:` shape could only ever emit one event per side, so the second
            # faint was silently unreported (1 of 36 faints in a 4-battle fuzz). Return them all
            # and let the caller emit one event each.
            return gained
        return [fallback] if fallback else []

    def _append_status_events(self, events: list, prev_ctx: BattleContext,
                              delta: TurnDelta, live: LiveView) -> None:
        """Append events for any status conditions newly applied this turn."""
        prev_our_status = self._pending_entry["our"].get("status")
        prev_opp_status = self._pending_entry["opp"].get("status")

        # Only track the mon that was active at decision time; skip if they fainted.
        if not delta.we_fainted:
            our_mon = live.ours.get(prev_ctx.our_active)
            if our_mon:
                new_status = self._mon_display_status(our_mon)
                if self._status_key(new_status) != self._status_key(prev_our_status):
                    if new_status:
                        events.append(f"our:{prev_ctx.our_active}:{new_status}")

        if not delta.opp_fainted:
            opp_mon = live.opp.get(prev_ctx.opp_active)
            if opp_mon:
                new_status = self._mon_display_status(opp_mon)
                if self._status_key(new_status) != self._status_key(prev_opp_status):
                    if new_status:
                        events.append(f"opp:{prev_ctx.opp_active}:{new_status}")

    def _fill_pending_outcome(self, prev_ctx: BattleContext, curr_ctx: BattleContext,
                              delta: TurnDelta, reward: float, live: LiveView) -> None:
        """Build and commit the JSON outcome for the pending entry using already-computed delta/reward."""
        # Our action
        if delta.our_switch_to:
            we_action = f"switched_to:{delta.our_switch_to}"
        elif prev_ctx.phase == "forced_switch" and curr_ctx.our_active not in ("NONE", prev_ctx.our_active):
            we_action = f"forced_switch_to:{curr_ctx.our_active}"
        else:
            we_action = self._pending_entry["chosen"]

        # Their action — distinguish: voluntary switch / faint+forced-switch / phaze / moved.
        # `delta.opp_resolved_move_id` prefers the protocol-confirmed event when
        # a damaging move resolved (protects against the stale-last_move bug
        # surfaced in step 3 smoke tests), and falls back to the inferred
        # opp_move_id for non-damaging moves (status, Roar, BP).
        opp_move_id = delta.opp_resolved_move_id
        if prev_ctx.phase == "forced_switch":
            they_action = "none"
        elif delta.opp_switch_to:
            if delta.opp_fainted and opp_move_id:
                they_action = f"{opp_move_id} → {delta.opp_switch_to}_sent_in"
            elif delta.opp_fainted:
                they_action = f"{delta.opp_switch_to}_sent_in"
            elif opp_move_id:
                they_action = f"{opp_move_id} → phazed_to:{delta.opp_switch_to}"
            else:
                they_action = f"switched_to:{delta.opp_switch_to}"
        elif opp_move_id is not None:
            they_action = opp_move_id
        else:
            # No protocol-confirmed (opp_resolved_move_id) or inferred (opp_move_id)
            # move this window. The old fallback read poke-env's stale `last_move` —
            # exactly the past-turn-field footgun this migration removes. History now
            # comes from the event-log TurnDelta, so an empty window is just "unknown".
            they_action = "unknown"

        # WHICH mon fainted — the set difference, not the decision-time active (see
        # `_newly_fainted`). This names the label AND picks the HP-delta slot, because both were
        # wrong in the same way: a switch-in that dies is neither the mon we were piloting when we
        # chose, nor the one whose HP row the delta was read from.
        our_fainted = (self._newly_fainted(
            prev_ctx.our_fainted_species, curr_ctx.our_fainted_species, prev_ctx.our_active)
            if delta.we_fainted else [])
        opp_fainted = (self._newly_fainted(
            prev_ctx.opp_fainted_species, curr_ctx.opp_fainted_species, prev_ctx.opp_active)
            if delta.opp_fainted else [])
        our_fainted_species = our_fainted[0] if our_fainted else None
        opp_fainted_species = opp_fainted[0] if opp_fainted else None

        # HP delta: for faint turns use the fainted mon's slot, not the forced switch-in.
        our_ref = our_fainted_species or (delta.our_switch_to or prev_ctx.our_active)
        opp_ref = opp_fainted_species or (delta.opp_switch_to or prev_ctx.opp_active)
        our_slot = prev_ctx.our_slot_map.get(our_ref, 0)
        opp_slot = prev_ctx.opp_slot_map.get(opp_ref, 0)
        our_delta = delta.our_hp_delta[our_slot] * 100
        opp_delta = delta.opp_hp_delta[opp_slot] * 100

        events = []
        for sp in our_fainted:
            events.append(f"our:{sp}:fainted")
        for sp in opp_fainted:
            events.append(f"opp:{sp}:fainted")

        self._append_status_events(events, prev_ctx, delta, live)

        breakdown = getattr(self._tracker._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": we_action,   "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": they_action, "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None

    def _terminal_hp(self, live: LiveView) -> tuple[np.ndarray, np.ndarray]:
        our_hp = np.zeros(6, dtype=np.float32)
        for mon in live.ours.mons:
            slot = self._our_slots.get(mon.species)
            if slot is not None:
                our_hp[slot] = mon.hp_fraction
        opp_hp = np.zeros(6, dtype=np.float32)
        for mon in live.opp.mons:
            slot = self._opp_slots.get(mon.species)
            if slot is not None:
                opp_hp[slot] = mon.hp_fraction
        return our_hp, opp_hp


def write_battle_record(out_prefix: str, recorder: "BattleRecorder", battle, step: int) -> None:
    """Finalize a recorder and write its forensic trace to disk.

    Writes three co-located artifacts for the same battle:

    * ``<out_prefix>_summary.json`` — the human-readable per-invocation summary.
    * ``<out_prefix>_states.npz`` — obs/logits/values/action_mask aligned with the summary's
      invocations (only when raw model I/O was captured), for ``probe_replay.py``. The stored
      ``logits`` are PRE-mask; ``action_mask`` is the legality the player used (see
      ``states_arrays``) — an offline consumer must read it, never infer legality from logits.
    * ``<out_prefix>_replay.html`` — a self-contained, **browser-watchable** Showdown
      replay of the battle. The two files above are prober-only forensic dumps; this
      one lets a human (no checkout, no prober) just open the game in a browser.

    Shared by the eval forensic capture so the on-disk format stays identical.
    """
    import os
    import re
    import json
    import sys

    recorder.finalize(battle)
    summary = recorder.to_summary(battle, step)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with open(f"{out_prefix}_summary.json", "w") as f:
        text = json.dumps(summary, indent=2)
        # Collapse the tiny leaf objects onto one line so a turn reads top-to-bottom.
        text = re.sub(
            r'\{\s*"prob":\s*"([^"]+)",\s*"valid":\s*(true|false)\s*\}',
            r'{"prob": "\1", "valid": \2}', text,
        )
        text = re.sub(
            r'\{\s*"species":\s*"([^"]+)",\s*"hp":\s*"([^"]+)"\s*\}',
            r'{"species": "\1", "hp": "\2"}', text,
        )
        text = re.sub(
            r'\{\s*"action":\s*"([^"]+)",\s*"hp_delta":\s*"([^"]+)"\s*\}',
            r'{"action": "\1", "hp_delta": "\2"}', text,
        )
        # Belief leaves: one species-guess per line ({"species": "tyranitar", "prob": "41.2%"}).
        text = re.sub(
            r'\{\s*"species":\s*"([^"]+)",\s*"prob":\s*"([^"]+)"\s*\}',
            r'{"species": "\1", "prob": "\2"}', text,
        )
        # Opponent-intent leaves: one expected option per line ({"name": "fireblast", "p": 0.41}),
        # and one β candidate per line ({"slot": 4, "p": 0.22, "species": "blissey"}).
        text = re.sub(
            r'\{\s*"name":\s*"([^"]+)",\s*"p":\s*([-0-9.eE]+)\s*\}',
            r'{"name": "\1", "p": \2}', text,
        )
        text = re.sub(
            r'\{\s*"slot":\s*(\d+),\s*"p":\s*([-0-9.eE]+),\s*"species":\s*("[^"]*"|null)\s*\}',
            r'{"slot": \1, "p": \2, "species": \3}', text,
        )
        f.write(text)

    states = recorder.states_arrays()
    if states:
        np.savez_compressed(f"{out_prefix}_states.npz", **states)

    # Human-watchable replay alongside the forensic dump. `save_replay` is a poke-env
    # method (renders the accumulated protocol stream — `_replay_data`, populated on every
    # parsed line regardless of the fast/heavy path — into a standalone HTML page), NOT
    # battle state, so it's an allowed raw seam, same as stall.py. Guarded so a replay
    # failure never costs us the forensic trace we already wrote.
    try:
        battle.save_replay(f"{out_prefix}_replay.html")
    except Exception as e:
        sys.stderr.write(f"Failed to save replay HTML for {out_prefix}: {e}\n")
        sys.stderr.flush()
