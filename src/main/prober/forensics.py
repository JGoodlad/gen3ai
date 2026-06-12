"""Decision-table forensic export over a run's ``eval_traces`` — the recurring behavioural-hypothesis
plumbing (run/step resolution, incoming-KO belief decode, softmax policy-confidence, critic ΔV) in ONE
place behind the prober, so ad-hoc scripts stop reinventing (and re-bugging) it.

:func:`build_decision_table` returns ONE row per captured decision with the forensic columns every
behavioural-hypothesis check needs — move-category, our/opp species+HP, policy ``conf``, ``reward``,
critic ``dV``, incoming-KO ``pko`` belief, faint flags, outcome. Exposed as
:meth:`~main.prober.session.ProbeSession.decision_table` and the ``query decision-table`` CLI;
``move_category`` / ``decision_table_digest`` are pure (unit-tested without traces).

(The run-level LUCK/MISTAKE bracket is the separate, canonical ``ProbeSession.falsify_scan`` /
``query falsify-scan`` in ``session.py`` — not duplicated here.)
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import json
import numpy as np

from agents.training.turn_delta import SELF_KO_MOVES
from main.prober.engine import decode_incoming_belief
from main.prober.model import ObsOffsets

# --------------------------------------------------------------------------- #
# Pure move taxonomy (unit-tested directly)
# --------------------------------------------------------------------------- #
_RECOVERY = frozenset({"recover", "softboiled", "milkdrink", "slackoff", "rest",
                       "morningsun", "synthesis", "moonlight", "wish"})
_SETUP = frozenset({"dragondance", "swordsdance", "calmmind", "curse", "bellydrum",
                    "agility", "bulkup", "amnesia", "nastyplot", "tailglow", "cosmicpower",
                    "acidarmor", "barrier", "irondefense", "howl", "meditate", "sharpen"})
_STALL = frozenset({"protect", "detect", "endure", "substitute"})
_STATUS = frozenset({"toxic", "willowisp", "thunderwave", "spore", "sleeppowder",
                     "stunspore", "glare", "hypnosis", "lovelykiss", "sing", "yawn",
                     "confuseray", "leechseed", "spikes", "reflect", "lightscreen", "haze",
                     "roar", "whirlwind", "encore", "taunt", "torment", "disable",
                     "aromatherapy", "healbell"})

CATEGORIES = ("selfko", "recovery", "setup", "stall", "status", "switch",
              "attack_or_other", "unknown")


def move_category(chosen: "str | None") -> str:
    """Coarse category of a chosen action label (lowercased move id, or ``switch:<species>``)."""
    if not chosen:
        return "unknown"
    c = chosen.lower()
    if c.startswith("switch"):
        return "switch"
    m = c.split(":")[0]
    if m in SELF_KO_MOVES:
        return "selfko"
    if m in _RECOVERY:
        return "recovery"
    if m in _SETUP:
        return "setup"
    if m in _STALL:
        return "stall"
    if m in _STATUS:
        return "status"
    return "attack_or_other"


def _hp_pct(x) -> Optional[int]:
    import re
    m = re.search(r"(\d+)", str(x) or "")
    return int(m.group(1)) if m else None


def _softmax_chosen(logits_row: np.ndarray, chosen_idx: int) -> float:
    z = np.asarray(logits_row, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return float(e[chosen_idx] / e.sum())


# --------------------------------------------------------------------------- #
# Decision table
# --------------------------------------------------------------------------- #
def build_decision_table(
    session,
    *,
    steps: Optional[Sequence[int]] = None,
    opponents: Optional[Sequence[str]] = None,
    outcomes: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    max_battles: Optional[int] = None,
) -> "list[dict]":
    """One forensic row per captured decision (``has_state==1``) across the matching battles.

    Reuses ``session.battles()`` for resolution (the discovery shard-prefix fix makes ``outcome``
    reliable), then loads summary+npz from each battle's path. ``dV`` is the raw critic value change
    ``V[i+1]-V[i]`` (NOT γ-discounted — the per-step critic move, the self-KO mechanism signal).
    ``conf`` is the softmax probability the policy put on the chosen action.
    """
    off = ObsOffsets.resolve()
    steps_s = set(steps) if steps else None
    opps_s = set(opponents) if opponents else None
    outs_s = set(outcomes) if outcomes else None
    cats_s = set(categories) if categories else None

    rows: "list[dict]" = []
    n_battles = 0
    for b in session.battles():
        if steps_s is not None and b["step"] not in steps_s:
            continue
        if opps_s is not None and b["opponent"] not in opps_s:
            continue
        if outs_s is not None and b["outcome"] not in outs_s:
            continue
        if max_battles is not None and n_battles >= max_battles:
            break
        n_battles += 1
        smf = b["id"]
        try:
            s = json.load(open(smf))
            d = np.load(smf.replace("_summary.json", "_states.npz"))
        except Exception:
            continue
        invs = s.get("invocations", [])
        V, HS, L, A, OB = d["values"], d["has_state"], d["logits"], d["actions"], d["obs"]
        for i, inv in enumerate(invs):
            if i >= len(L) or not HS[i]:
                continue
            chosen = (inv.get("chosen") or "").lower()
            cat = move_category(chosen)
            if cats_s is not None and cat not in cats_s:
                continue
            our = inv.get("our") or {}
            opp_ = inv.get("opp") or {}
            ev = " ".join(str(e) for e in (inv.get("outcome", {}).get("events") or [])).lower()
            try:
                view = decode_incoming_belief(OB[i].astype(np.float32), off)
                pko = getattr(view, "active_pko", None) if view is not None else None
            except Exception:
                pko = None
            rt = (inv.get("outcome", {}).get("reward") or {}).get("total")
            rows.append(dict(
                opp=b["opponent"], step=b["step"], outcome=b["outcome"],
                battle=b.get("short_id"), turn=inv.get("turn"), inv=i,
                chosen=chosen, cat=cat,
                our=our.get("species"), our_hp=_hp_pct(our.get("hp")),
                oppmon=opp_.get("species"), opp_hp=_hp_pct(opp_.get("hp")),
                conf=round(_softmax_chosen(L[i], int(A[i])), 3),
                reward=round(float(rt), 3) if rt is not None else None,
                dV=round(float(V[i + 1] - V[i]), 3) if i + 1 < len(V) else None,
                pko=round(float(pko), 3) if pko is not None else None,
                faint_us=("our:" in ev and "fainted" in ev),
                faint_opp=("opp:" in ev and "fainted" in ev),
            ))
    return rows


def decision_table_digest(rows: "list[dict]") -> dict:
    """A compact, human-readable summary of a decision table (counts, the per-category
    confidence/reward/dV medians — the forensic-template at a glance)."""
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 3) if xs else None
    by_cat = {}
    for c in CATEGORIES:
        sub = [r for r in rows if r["cat"] == c]
        if not sub:
            continue
        by_cat[c] = dict(
            n=len(sub), conf_med=med([r["conf"] for r in sub]),
            reward_med=med([r["reward"] for r in sub]), dV_med=med([r["dV"] for r in sub]),
        )
    return dict(
        n_decisions=len(rows),
        outcomes=dict(Counter(r["outcome"] for r in rows)),
        by_category=by_cat,
    )
