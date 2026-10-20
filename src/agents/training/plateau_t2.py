"""THE PLATEAU'S TIER 2 AND THE TWO-TIER SIGNATURE — the rule (``design_evaluation.md`` §8.2–§8.3; unit U9, offline
slice, 2026-10-19). Pure: no I/O. The reader is ``main.plateau status`` (via ``main.cycle_monitor``).

**Tier 2** at a check = the cycle monitor's verdict there (``agents.training.cycle_monitor``: NONE / SUSPECTED /
CONFIRMED / INCONCLUSIVE) plus the PANEL's.

**The panel** (§8.2): a FROZEN set of outside opponents (registered by name and sha256 in
``designs/research_state/plateau_panels.json``), each played by the newest node and by the W-back node at
:data:`PanelRule.pairs` mirrored pairs (:data:`PanelRule.candidate_pairs` at a CANDIDATE plateau — Tier 1 FLAT and
the monitor NONE — which makes P(PANEL FLAT | zero true gain) ≈ 0.95). Per opponent o, Δ_o = Elo(s_new,o) −
Elo(s_wback,o) by the logit (G4), its variance by the delta method from the pair-level SEs; Δ_panel = the equal-weight
mean over the panel. **PANEL RISING** iff its lower one-sided 95 % bound > 0; **PANEL FLAT** iff its upper one-sided
95 % bound < :data:`PanelRule.flat_bar_elo` (14 Elo: the plateau rate, 2 Elo per GPU-hour, over W ≈ 7 GPU-h); else
UNDECIDED. A bound within 1e-9 of its bar is not a crossing (rule 8). A cell short of its pairs is INCOMPLETE; a
cell whose aborted games exceed 25 % is INCONCLUSIVE.

**The signatures** (§8.3), per check, newest first:

| status | rule |
|---|---|
| ``TREADMILL`` | the monitor CONFIRMED at the newest decided check, whatever Tier 1 says |
| ``TRUE_PLATEAU`` | Tier 1 FLAT AND monitor NONE AND panel FLAT, at two consecutive checks (Δ apart) |
| ``PLATEAU_BUT_PANEL`` | Tier 1 FLAT AND monitor NONE at two consecutive checks, the panel NOT flat or not played (it names why) |
| ``TRANSITIVE_IMPROVEMENT`` | Tier 1 GAIN AND monitor NONE at the newest check |
| ``CONTINUE`` | anything else (UNDECIDED, SUSPECTED, INCONCLUSIVE, a check not yet read) |

STAGNATION (a TRUE_PLATEAU with ``main.policy_drift`` below its floor) is the operator's follow-up read, not computed
here. The G6 joint operating-characteristics simulation is NOT part of this slice (a FINDING).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from agents.training import cycle_monitor as CM
from agents.training import plateau_t1 as T1

RULE_NAME = "plateau_t2_panel"
RULE_VERSION = "gen3_plateau_t2_v1"
REQUEST_KIND = "panel"
PURPOSE = "plateau"

RISING, FLAT, UNDECIDED, INCOMPLETE, INCONCLUSIVE, NOT_DECLARED = (
    "RISING", "FLAT", "UNDECIDED", "INCOMPLETE", "INCONCLUSIVE", "NOT_DECLARED")
TREADMILL, TRUE_PLATEAU, PLATEAU_BUT_PANEL, TRANSITIVE_IMPROVEMENT, CONTINUE = (
    "TREADMILL", "TRUE_PLATEAU", "PLATEAU_BUT_PANEL", "TRANSITIVE_IMPROVEMENT", "CONTINUE")
BAND = 1e-9


@dataclass(frozen=True)
class PanelRule:
    pairs: int = 500
    candidate_pairs: int = 1000
    batch_pairs: int = 50
    flat_bar_elo: float = 14.0
    z: float = 1.6448536269514722            # one-sided 95 %
    abort_frac: float = 0.25

    def rule_string(self) -> str:
        return (f"{RULE_NAME}(pairs={self.pairs},candidate={self.candidate_pairs},batch={self.batch_pairs},"
                f"flat_bar_elo={self.flat_bar_elo},one_sided=0.95,mean_over_panel,logit_delta,band=1e-09)")

    def to_json(self) -> Dict[str, Any]:
        return {**asdict(self), "rule": self.rule_string(), "rule_version": RULE_VERSION}


REGISTERED = PanelRule()


def cell_elo(counts: Sequence[float]) -> Optional[Dict[str, float]]:
    """A cell's score, Elo (logit × 400/ln10) and Elo variance, pair level (``cycle_monitor.edge_stats``)."""
    import numpy as np

    c = np.asarray(counts, dtype=float)
    if c.sum() <= 0:
        return None
    n, m, var_m, y, w = CM.edge_stats(c[None, :])
    return {"pairs": int(n[0]), "score": float(m[0]), "elo": float(y[0]) * CM.ELO_PER_LOGIT,
            "var_elo": float(1.0 / w[0]) * CM.ELO_PER_LOGIT ** 2}


def panel_verdict(newest: Mapping[str, Mapping[str, Any]], wback: Mapping[str, Mapping[str, Any]],
                  members: Sequence[str], want_pairs: int, rule: PanelRule = REGISTERED) -> Dict[str, Any]:
    """Δ_panel between the newest and the W-back node. ``newest`` / ``wback``: member → ``{"counts": [5],
    "aborted_share": float}`` (absent = not played)."""
    if not members:
        return {"verdict": NOT_DECLARED}
    per: Dict[str, Any] = {}
    short = []
    for o in members:
        a, b = newest.get(o), wback.get(o)
        if a is None or b is None or sum(a["counts"]) < want_pairs or sum(b["counts"]) < want_pairs:
            short.append(o)
            continue
        if max(a.get("aborted_share", 0.0), b.get("aborted_share", 0.0)) > rule.abort_frac:
            return {"verdict": INCONCLUSIVE, "reason": f"{o}: aborted games exceed {rule.abort_frac:.0%}"}
        ea, eb = cell_elo(a["counts"]), cell_elo(b["counts"])
        assert ea is not None and eb is not None
        per[o] = {"newest": ea, "wback": eb, "delta_elo": ea["elo"] - eb["elo"], "var": ea["var_elo"] + eb["var_elo"]}
    if short:
        return {"verdict": INCOMPLETE, "short": short, "want_pairs": want_pairs}
    k = len(per)
    d = sum(v["delta_elo"] for v in per.values()) / k
    se = math.sqrt(sum(v["var"] for v in per.values())) / k
    lo, hi = d - rule.z * se, d + rule.z * se
    if abs(lo) <= BAND or abs(hi - rule.flat_bar_elo) <= BAND:
        v = UNDECIDED
    elif lo > 0:
        v = RISING
    elif hi < rule.flat_bar_elo:
        v = FLAT
    else:
        v = UNDECIDED
    return {"verdict": v, "delta_elo": round(d, 3), "se_elo": round(se, 3), "lower95": round(lo, 3),
            "upper95": round(hi, 3), "pairs": want_pairs,
            "per_member": {o: {"delta_elo": round(x["delta_elo"], 2), "newest_score": round(x["newest"]["score"], 4),
                               "wback_score": round(x["wback"]["score"], 4)} for o, x in per.items()}}


def signature(checks: Sequence[Mapping[str, Any]], delta_steps: int) -> Dict[str, Any]:
    """The run's two-tier status from per-check ``{"grid", "tier1", "monitor", "panel"}`` (module docstring)."""
    by = {int(c["grid"]): c for c in checks}
    decided = sorted(g for g, c in by.items() if c["tier1"] in T1.VERDICTS and c["monitor"] in CM.VERDICTS)
    if not decided:
        return {"status": CONTINUE, "as_of_check": None, "why": "no check has both a Tier-1 and a monitor verdict"}
    g = decided[-1]
    c = by[g]
    if c["monitor"] == CM.CONFIRMED:
        return {"status": TREADMILL, "as_of_check": g, "why": "the cycle monitor CONFIRMED cycling"}
    if c["tier1"] == T1.GAIN and c["monitor"] == CM.NONE:
        return {"status": TRANSITIVE_IMPROVEMENT, "as_of_check": g, "why": "Tier 1 GAIN, monitor NONE"}
    prev = by.get(g - delta_steps)
    pair = [prev, c] if prev is not None else None
    if pair and all(x["tier1"] == T1.FLAT and x["monitor"] == CM.NONE for x in pair):
        if all(x.get("panel") == FLAT for x in pair):
            return {"status": TRUE_PLATEAU, "as_of_check": g, "why": "Tier 1 FLAT, monitor NONE, panel FLAT at "
                    f"{g - delta_steps:,} and {g:,}"}
        return {"status": PLATEAU_BUT_PANEL, "as_of_check": g,
                "why": f"Tier 1 FLAT and monitor NONE at {g - delta_steps:,} and {g:,}; the panel reads "
                       f"{[x.get('panel') for x in pair]} — a TRUE PLATEAU also needs the panel FLAT at both"}
    return {"status": CONTINUE, "as_of_check": g,
            "why": f"Tier 1 {c['tier1']}, monitor {c['monitor']}, panel {c.get('panel')}"}
