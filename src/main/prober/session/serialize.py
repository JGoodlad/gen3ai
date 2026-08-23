"""JSON-shaping leaves: a board side, a model choice, a battle's short id, round-or-None.

Everything `ProbeSession` returns is JSON-serializable, and these are the functions that make it
so. Pure — no session, no IO.
"""

from __future__ import annotations

from dataclasses import asdict

from main.prober.discovery import BattleTrace, ModelChoice
from main.prober.engine import _pct as _hp_pct, build_opp_intent, opp_intent_text, parse_pct


def _active_str(side) -> str:
    s = f"{side.active_species} {side.active_hp}"
    return f"{s} {side.status}" if side.status else s


def _chosen_prob(inv: dict) -> "float | None":
    """The policy's recorded probability on the action it actually took, or None when the trace
    didn't record that action's row."""
    acts, chosen = inv.get("actions", {}), inv.get("chosen", "")
    return parse_pct(acts[chosen]["prob"]) if chosen in acts else None


def _mon_dict(m) -> dict:
    """One benched `MonState` as JSON. `hp_pct` is the NUMERIC form of the recorded `hp` string
    ('28%' → 28.0, 'faint' → 0.0) so a surface can size an HP bar without parsing a display string —
    the parse belongs here, next to the recorder's format, not in six renderers."""
    return {"species": m.species, "hp": m.hp, "hp_pct": _hp_pct(m.hp), "fainted": m.fainted,
            "status": m.status, "item": m.item, "moves": list(m.moves)}


def _side_dict(side) -> dict:
    """One side's board at a decision as JSON: the active (species/hp/status/boosts/item/moves) plus
    the revealed bench. Opponent movesets/items are revealed-only by construction — `build_board`
    reads the recorder's one-sided view, so this cannot leak hidden information."""
    return {
        "species": side.active_species, "hp": side.active_hp,
        "hp_pct": _hp_pct(side.active_hp), "status": side.status, "boosts": side.boosts,
        "item": side.item, "moves": list(side.moves),
        "bench": [_mon_dict(m) for m in side.bench],
    }


def _opp_intent_dict(inv: dict) -> "dict | None":
    """The decision's α/β block as JSON — `{alpha, beta, top, switch_p, text}` — or `None` when the
    trace carries no `opp_intent` (the heads were off, i.e. every trace before v67).

    The probabilities are passed through as the recorder wrote them (already rounded at capture);
    `text` is `engine.opp_intent_text`, so the sentence the web replay prints is the one the TUI and
    the CLI print.

    The sequences are LISTS, not the view's tuples: this dict is served as JSON and also compared
    against the API's response, and a tuple survives `json.dumps` but comes back a list — so an
    `asdict` here would make the session and its own HTTP endpoint disagree on a value they share."""
    view = build_opp_intent(inv)
    if view is None:
        return None
    out = asdict(view)
    out["alpha"] = list(out["alpha"])
    out["beta"] = list(out["beta"])
    out["text"] = opp_intent_text(view)
    return out


def _r(x, n=3):
    """round-or-None — compact numbers in JSON output."""
    return round(float(x), n) if isinstance(x, (int, float)) else None


def _choice_dict(c: ModelChoice) -> dict:
    return {"path": c.path, "tier": c.tier, "detail": c.detail, "manifest": c.manifest}


def _short_id(b: BattleTrace) -> str:
    return f"step_{b.step}/{b.opponent}/{b.outcome}_{b.index:03d}"
