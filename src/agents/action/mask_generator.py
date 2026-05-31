import numpy as np

from agents.action.constants import ACTION_SPACE_SIZE, SWITCH_END, MOVE_START, N_MOVE_SLOTS, STRUGGLE
from agents.action.ordering_integrity import check_switch_ordering_alignment
from agents.battle.live_view import LegalActions, LiveView


class Gen3ActionMasker:
    """
    Generates 11-dimensional action masks for Gen 3 OU from the server-authoritative
    :class:`~agents.battle.live_view.LegalActions` snapshot.

    Indices:
      0–5: Switches (team slots 0–5)
      6–9: Moves (request slots 0–3)
      10:  Struggle (single-sourced via ``legal.struggle`` — never a move slot)

    The mask is built purely from ``LegalActions`` (:meth:`mask_from_legal`); the
    battle-taking :meth:`get_mask` is a thin wrapper that snapshots the legal surface and
    runs the team-ordering integrity guards. There is no longer a
    ``battle._gen3_decision_context`` stash — the immutable ``LegalActions`` captured here
    IS the per-decision snapshot, carried forward on the BattleContext to the mapper.
    """

    @staticmethod
    def mask_from_legal(legal: LegalActions) -> np.ndarray:
        """PURE: build the 11-dim binary mask from a :class:`LegalActions` snapshot
        alone (no battle). Fully testable with a stub.

        Struggle is already excluded from ``legal.move_slots`` (it is the
        :attr:`~agents.battle.live_view.LegalActions.struggle` flag), so a struggle entry
        can never set a move-slot bit — the single-source contract that prevents the
        historical "struggle double-enabling" bug.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        # 0–5: switches the server says we may make, by team slot.
        for sw in legal.switches:
            if sw.slot < SWITCH_END:
                mask[sw.slot] = 1

        # 6–9: request move slots that aren't disabled.
        for i, m in enumerate(legal.move_slots):
            if i < N_MOVE_SLOTS and not m.disabled:
                mask[i + MOVE_START] = 1

        # 10: struggle — enabled only when the server forces it (all PP gone).
        if legal.struggle:
            mask[STRUGGLE] = 1

        return mask

    @staticmethod
    def get_mask(battle, legal: LegalActions = None, live: LiveView = None) -> np.ndarray:
        """Build the action mask for ``battle``. STRICT MODE: crashes on ambiguity.

        Reads only through the strict boundary — the server-authoritative
        :class:`LegalActions` (legality + ``last_request``) and the current-board
        :class:`LiveView` (the own-team roster the integrity guards check). No raw
        ``battle.<attr>`` reads remain.

        The env / player pass the ``legal`` snapshot they captured this decision so the
        mask and the mapper share one immutable source; standalone callers omit it and a
        fresh snapshot is taken here. ``live`` is likewise optional: a caller that already
        built a :class:`LiveView` this decision can pass it to avoid a second build,
        otherwise one is built from ``battle`` (``LiveView.from_battle`` works on any
        poke-env battle, ``Gen3Battle`` or plain ``Battle``).
        """
        if legal is None:
            legal = LegalActions.from_battle(battle)

        if not legal.last_request:
            # We crash if we are asked to mask but have no request context.
            # This is a 'junk in junk out' prevention measure.
            raise RuntimeError("STRICT MODE FAILURE: No last_request found in battle. Cannot mask.")

        if live is None:
            live = LiveView.from_battle(battle)

        # Integrity Check: No duplicate species allowed (Gen 3 OU standard). The own-team
        # roster is read through the LiveView boundary; ``live.ours.mons`` preserves the
        # ``list(battle.team.values())`` order the encoder's get_team_list also uses.
        species_list = [m.species for m in live.ours.mons]
        if len(species_list) != len(set(species_list)):
            raise RuntimeError(f"STRICT MODE FAILURE: Duplicate species detected in team: {species_list}")

        mask = Gen3ActionMasker.mask_from_legal(legal)

        # Integrity: the model's team ordering must match the action space — switch
        # action index i, switch-validity bit i, and per-Pokémon obs slot i must all
        # refer to the same Pokémon. (Move-validity ordering is fixed + validated at
        # prev_mask construction, EpisodeTracker.prev_mask.)
        check_switch_ordering_alignment(live, mask, legal)

        return mask
