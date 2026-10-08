"""The forensic trace's LABELS — pure functions of the read-models (``LiveView`` / ``LegalActions``).

Lifted out of :class:`~agents.training.battle_recorder.BattleRecorder` (poke-env retirement P5,
``T27``) so the two writers of the prober's summary shape share ONE rendering: the live eval
recorder (a poke-env battle) and the prober's core-trace recorder (``main.prober.core_recorder``,
the Rust core's views). Nothing here imports poke-env, and nothing reads a battle — every input is a
read-model, which is what makes the two writers byte-comparable.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

#: The volatiles a status string names, in the order it names them.
_EFFECT_NAMES = (("taunt", "TAUNT"), ("confusion", "CONF"), ("encore", "ENCORE"),
                 ("attract", "ATTRACT"), ("disable", "DISABLE"), ("substitute", "SUB"))
_STATUS_NAMES = {"brn": "BRN", "par": "PAR", "frz": "FRZ", "psn": "PSN"}


def mon_display_status(mon) -> Optional[str]:
    """Rich status string including counters and volatiles, read from a ``LivePokemon`` (id-form
    status + ``{volatile_id: counter}``). Examples: "SLP(3)", "TOX(5)", "BRN", "PAR|TAUNT",
    "PERISH(2)|CONF"."""
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
        if name := _STATUS_NAMES.get(status):
            parts.append(name)
    for vid, name in _EFFECT_NAMES:
        if vid in vol:
            parts.append(name)
    for n, vid in [(3, "perish3"), (2, "perish2"), (1, "perish1"), (0, "perish0")]:
        if vid in vol:
            parts.append(f"PERISH({n})")
            break
    return "|".join(parts) if parts else None


def status_key(status_str: Optional[str]) -> Optional[str]:
    """Normalize for change detection — strips counter values, sorts parts."""
    if not status_str:
        return None
    return "|".join(sorted(p.split("(")[0] for p in status_str.split("|")))


def bench_summary(active, mons) -> str:
    parts = []
    for mon in mons:
        if active and mon.species == active.species:
            continue
        if mon.fainted:
            parts.append(f"{mon.species}(faint)")
        else:
            pct = f"{mon.hp_fraction * 100:.0f}%"
            status = mon_display_status(mon)
            parts.append(f"{mon.species}({pct},{status})" if status else f"{mon.species}({pct})")
    return ", ".join(parts)


def _label(i: int, team_list, move_ids) -> str:
    if i < 6:
        return f"switch:{team_list[i].species}" if i < len(team_list) else f"switch:slot{i}"
    if i < 10:
        m = i - 6
        return move_ids[m] if m < len(move_ids) else f"move{m}"
    return "struggle"


def action_label(action_idx: int, live, legal) -> str:
    """The label of one action index. ``display_move_ids`` (not ``move_ids``): OUR Hidden Power shows
    its TYPED id ("hiddenpowergrass") — we always know our own HP type, and these are human-/prober-
    facing labels (the mask / mapper use the wire-truth ids)."""
    return _label(action_idx, live.ours.mons, list(legal.display_move_ids))


def all_action_labels(live, probs: np.ndarray, mask: np.ndarray, legal) -> dict:
    team_list = live.ours.mons
    move_ids = list(legal.display_move_ids)  # typed own HP — see action_label
    return {_label(i, team_list, move_ids): {"prob": f"{probs[i] * 100:.1f}%", "valid": bool(mask[i])}
            for i in range(11)}


def newly_fainted(prev_fainted: Optional[Iterable[str]], now_fainted: Optional[Iterable[str]],
                  fallback: str) -> List[str]:
    """EVERY species that actually fainted this turn, as a set difference.

    A faint used to be detected by COUNT and then labelled with the mon that was active when the
    decision was made. That is the wrong mon whenever a switch happened on the same turn, which is
    not a corner case: we switch Cloyster → Jolteon and the opponent's Explosion kills JOLTEON, yet
    the trace recorded `our:cloyster:fainted`; the opponent switches Claydol → Dugtrio and our Ice
    Beam kills DUGTRIO, recorded as `opp:claydol:fainted`. Measured on ai_v9_17_tdaux_lam3: **25 of
    466 turns** named a mon that did not faint. The set difference also gets a mon REVEALED and killed
    on the same turn (no previous HP to fall from), and ONE SIDE CAN LOSE TWO MONS IN A TURN (a KO,
    then the forced replacement dies to Spikes) — every one is returned, one event each.

    ``fallback`` keeps the old behaviour when the sets cannot answer: a slightly wrong label is still
    better than an empty one, and a forensic recorder must never raise into training.

    ⚠️ The ORDER of a multi-faint list is ``now_fainted``'s iteration order. The live recorder passes
    a ``frozenset`` (so two faints in one turn come out in hash order, which varies with
    ``PYTHONHASHSEED``); the core-trace recorder passes the board's mon order (deterministic)."""
    gained = [sp for sp in (now_fainted or ()) if sp not in (prev_fainted or ())]
    if gained:
        return gained
    return [fallback] if fallback else []


def snapshot_statuses(live) -> dict:
    """Every mon's normalized status key on both sides, keyed ``"our:<species>"`` /
    ``"opp:<species>"``. A fainted mon reads ``None`` — ``fnt`` is not a status a move inflicted,
    and the faint already has its own event."""
    out: dict = {}
    for side, team in (("our", live.ours), ("opp", live.opp)):
        for mon in team.mons:
            out[f"{side}:{mon.species}"] = (None if mon.fainted else status_key(mon_display_status(mon)))
    return out


def status_events(before: Optional[dict], live) -> "tuple[list, dict]":
    """``(events, now)``: an ``"<side>:<species>:<STATUS>"`` event for every status NEWLY applied
    since ``before`` (on either side, to ANY mon — see ``BattleRecorder._append_status_events``),
    and the snapshot that becomes the next ``before``. A cure says nothing."""
    before = before or {}
    now = snapshot_statuses(live)
    events = []
    for key, new_key in now.items():
        if not new_key or new_key == before.get(key):
            continue
        side, _, species = key.partition(":")
        mon = (live.ours if side == "our" else live.opp).get(species)
        display = mon_display_status(mon) if mon is not None else None
        if display:
            events.append(f"{side}:{species}:{display}")
    return events, now
