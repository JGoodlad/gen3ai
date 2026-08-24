"""Which OPPONENT actions the depth-1 search marginalizes over, and with what weights.

The registered design spends the first marginal second of budget on the OPPONENT axis, and the
weights come from the v67 `α` head. This module is the seam between the head's PUBLICATION and
the sim's legal-choice surface, kept separate from :mod:`search` so its rules are unit-testable
without a sim.

**The α-consumer contract** (`src/agents/model/CLAUDE.md` -> *The rules an α CONSUMER follows*),
and how each clause lands here:

1. *Read ``last_alpha_logits`` — the PUBLICATION — never a raw stash.* :func:`alpha_publication`
   is the only reader.
2. *Take the UNRENORMALIZED move slice; the missing mass is SWITCH, and renormalizing asserts
   they attacked.* Honoured literally: the move seats keep their unrenormalized weights and
   ``α_SWITCH`` is spent as switch mass on the opponent's actual switch targets. The final
   renormalization is over a candidate set that CONTAINS the switch options, so it never
   re-attributes switch mass to attacking — the thing the clause forbids.
3. *Align by CONSTRUCTION and fail loud on a width mismatch.* The seats are matched to the
   opponent's real choices by canonical move NUM (the same key ``match_seats_to_move_num`` uses
   at loss time), never by position. A seat the opponent does not actually have is DROPPED and
   its mass REPORTED (``unmatched_move_mass``) rather than smeared — that number is the BELIEF's
   coverage failure and folding it into the weights would hide which component to fix.

**The no-head fallback is an ABSENCE, not a CLAIM.** With no α head the candidates are UNIFORM
over the opponent's legal actions. That is the honest statement "I have no opinion". The rejected
alternative — deriving weights from the R1 ``belief_mean`` rung — would set ``α_SWITCH ≡ 0`` and
thereby assert *"they never switch"*, which is the v94 `SwitchBranchMoveCell` lesson: a fallback
that silently states something false is worse than no fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

# A move seat whose belief did not fire, or a choice α never names, still has to be reachable —
# otherwise the search would refuse to consider an action the opponent can legally take. This is
# the weight such a choice gets BEFORE renormalization. Small enough not to compete with a named
# seat, large enough that a surprise is not structurally invisible.
UNNAMED_FLOOR = 1e-3


@dataclass(frozen=True)
class OppCandidate:
    """One opponent action the search will branch on."""

    token: str          # the sim choice string, e.g. "move earthquake" / "switch 3"
    weight: float       # normalized over the retained candidate set; sums to 1
    kind: str           # "move" | "switch"
    label: str          # human name, for the results row
    source: str         # "alpha" | "alpha_switch" | "floor" | "uniform"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# the sim's legal-choice surface
# ---------------------------------------------------------------------------


def legal_choices_from_request(request: Optional[dict]) -> List[dict]:
    """The opponent's legal choices at a move-selection request, as sim choice strings.

    Reads Showdown's own request JSON — the SERVER's answer to "what may this side do" — rather
    than re-deriving legality from a board model. Switch targets are addressed by 1-based INDEX
    into ``side.pokemon`` (canonical and unambiguous; a species name collides on a mirror match).

    Returns ``[]`` for anything that is not an open move request (``forceSwitch``, ``teamPreview``,
    ``wait``), which is the caller's signal that there is nothing to marginalize over.
    """
    if not request or request.get("wait") or request.get("teamPreview"):
        return []
    if request.get("forceSwitch"):
        return []
    out: List[dict] = []
    active = (request.get("active") or [None])[0] or {}
    trapped = bool(active.get("trapped") or active.get("maybeTrapped"))
    for slot in active.get("moves") or []:
        if slot.get("disabled"):
            continue
        if slot.get("pp") is not None and int(slot["pp"]) <= 0:
            continue
        mid = slot.get("id") or slot.get("move") or ""
        if not mid:
            continue
        out.append({"kind": "move", "token": f"move {mid}",
                    "label": slot.get("move") or mid, "move_id": mid})
    if not trapped:
        for i, mon in enumerate(((request.get("side") or {}).get("pokemon") or [])):
            if mon.get("active"):
                continue
            cond = mon.get("condition") or ""
            if cond.endswith(" fnt") or cond == "0 fnt":
                continue
            species = (mon.get("details") or mon.get("ident") or "").split(",")[0]
            out.append({"kind": "switch", "token": f"switch {i + 1}",
                        "label": species or f"slot{i + 1}", "slot": i})
    return out


# ---------------------------------------------------------------------------
# the publication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaPublication:
    """The α head's read for ONE decision. ``None``-safe: an α-off checkpoint yields ``None``."""

    move_p: Dict[int, float]     # {move_num: unrenormalized probability}
    switch_p: float              # α_SWITCH
    beta_p: Dict[int, float]     # {opp team slot index: P(this mon comes in | switch)}


def alpha_publication(extractor) -> Optional[AlphaPublication]:
    """Read α (and β) off the extractor's per-forward stash. ``None`` when the heads are off.

    Called immediately after the policy forward that produced the live decision, so the stash is
    the one belonging to THIS state — the same discipline `RLPlayer._opp_intent` follows. Any
    later forward (the search's own critic passes) clobbers it, which is why the caller reads it
    first and never again."""
    import torch

    if extractor is None:
        return None
    alogits = getattr(extractor, "last_alpha_logits", None)
    seat_nums = getattr(extractor, "last_alpha_seat_nums", None)
    if alogits is None or seat_nums is None:
        return None
    probs = torch.softmax(alogits[0].float(), dim=-1)
    k = int(probs.shape[-1]) - 1
    nums = seat_nums[0]
    if int(nums.shape[-1]) != k:
        # Clause 3 — align by CONSTRUCTION, fail loud on a width mismatch. Broadcasting here
        # would pair each α weight with the wrong opponent move while every shape check passed.
        raise ValueError(
            f"α seat width mismatch: {k} move seats in last_alpha_logits but "
            f"{int(nums.shape[-1])} seat nums — the axes must be the same by construction")
    move_p: Dict[int, float] = {}
    for i in range(k):
        num = int(nums[i])
        if num <= 0:                       # a seat the belief never filled
            continue
        move_p[num] = move_p.get(num, 0.0) + float(probs[i])
    beta_p: Dict[int, float] = {}
    blogits = getattr(extractor, "last_beta_logits", None)
    if blogits is not None:
        bp = torch.softmax(blogits[0].float(), dim=-1)
        for i in range(int(bp.shape[0])):
            if torch.isfinite(blogits[0, i]):
                beta_p[i] = float(bp[i])
    return AlphaPublication(move_p=move_p, switch_p=float(probs[k]), beta_p=beta_p)


def _move_num_lookup() -> Dict[str, int]:
    from agents.gen3_data import moves as _gm

    return {mid: int(rec["num"]) for mid, rec in _gm.raw().items() if "num" in rec}


# ---------------------------------------------------------------------------
# the candidate set
# ---------------------------------------------------------------------------


def build_candidates(legal: Sequence[dict], pub: Optional[AlphaPublication], *, m_opp: int,
                     num_by_id: Optional[Dict[str, int]] = None) -> tuple:
    """``(candidates, diagnostics)`` — the top ``m_opp`` opponent actions, weights summing to 1.

    ``legal`` is :func:`legal_choices_from_request` output; ``pub`` is :func:`alpha_publication`
    (``None`` -> the uniform ABSENCE fallback). Pruning happens BEFORE renormalization, so the
    retained weights are the head's own relative opinion among the branches actually explored.
    """
    if not legal:
        return [], {"n_legal": 0, "alpha_used": False, "unmatched_move_mass": 0.0,
                    "retained_mass": 0.0}
    if pub is None:
        w = 1.0 / len(legal)
        cands = [OppCandidate(token=c["token"], weight=w, kind=c["kind"], label=c["label"],
                              source="uniform") for c in legal[:m_opp]]
        return _renormalize(cands), {"n_legal": len(legal), "alpha_used": False,
                                     "unmatched_move_mass": 0.0,
                                     "retained_mass": sum(c.weight for c in cands)}

    num_by_id = num_by_id if num_by_id is not None else _move_num_lookup()
    scored: List[OppCandidate] = []
    matched_nums: set = set()
    switch_slots = [c for c in legal if c["kind"] == "switch"]
    for c in legal:
        if c["kind"] == "move":
            num = num_by_id.get(c["move_id"])
            p = pub.move_p.get(num) if num is not None else None
            if p is None:
                scored.append(OppCandidate(c["token"], UNNAMED_FLOOR, "move", c["label"], "floor"))
            else:
                matched_nums.add(num)
                scored.append(OppCandidate(c["token"], max(p, 0.0), "move", c["label"], "alpha"))
        else:
            # Spend α_SWITCH across the real switch targets, weighted by β when it exists and
            # UNIFORMLY when it does not. β's slot axis is the opponent's obs-team order, which
            # is the same order `side.pokemon` is published in.
            if pub.beta_p:
                share = pub.beta_p.get(c["slot"], 0.0)
                tot = sum(pub.beta_p.get(s["slot"], 0.0) for s in switch_slots)
                frac = (share / tot) if tot > 0 else (1.0 / max(1, len(switch_slots)))
            else:
                frac = 1.0 / max(1, len(switch_slots))
            scored.append(OppCandidate(c["token"], pub.switch_p * frac, "switch", c["label"],
                                       "alpha_switch"))
    unmatched = sum(p for n, p in pub.move_p.items() if n not in matched_nums)
    scored.sort(key=lambda c: -c.weight)
    kept = scored[:max(1, int(m_opp))]
    retained = sum(c.weight for c in kept)
    return _renormalize(kept), {
        "n_legal": len(legal), "alpha_used": True,
        # The BELIEF's coverage failure, reported not smeared: α mass on moves the opponent does
        # not actually have in this world.
        "unmatched_move_mass": round(unmatched, 4),
        # How much of α's opinion survived the top-m prune — the honest cost of the width bound.
        "retained_mass": round(retained, 4),
    }


def _renormalize(cands: Sequence[OppCandidate]) -> List[OppCandidate]:
    tot = sum(c.weight for c in cands)
    if tot <= 0:
        w = 1.0 / max(1, len(cands))
        return [OppCandidate(c.token, w, c.kind, c.label, c.source) for c in cands]
    return [OppCandidate(c.token, c.weight / tot, c.kind, c.label, c.source) for c in cands]
