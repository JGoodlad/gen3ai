"""CELLS and pair-level estimator helpers over a declared read.

A CELL is one (request, player, opponent): the unit a registered read judges complete or INCONCLUSIVE (§9.1;
X5 §7.4 registers it per cell — review M3(b)). A FAMILY is read across its LOOKS (its requests, in the order they
were opened), each look a set of cells.

**INCONCLUSIVE, never a verdict, per cell:** aborted games exceed 25 % of the cell's attempted games
(``4 x aborted > attempted``, integer arithmetic, so no input sits within a rounding error of the boundary —
standing rule 8), the cell attempted nothing, or (when the registration names one) it has fewer completed pairs
than its minimum. The across-request rules (two regimes in a request, two protocols in a pinned family) are the
reader's typed refusals (``MixedRegimeError``), which a decision maps to INCONCLUSIVE.

**Inference scope.** A pair-clustered interval is CONDITIONAL: meter noise only, on the run(s) as they are. A read
declared ``inference="across_runs"`` is refused here (:class:`InferenceScopeError`) — its error bar needs the run
term, i.e. a seed-level estimator over cells (X5 §7.4's cross), not a pooled pentanomial."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents.training import mirrored_pairs as MP
from agents.training.eval_ledger.reader import LedgerRead, MixedRegimeError

#: §9.1: a cell is INCONCLUSIVE when aborted games exceed 1/4 of its attempted games.
ABORT_LIMIT = (1, 4)


class InferenceScopeError(ValueError):
    """A conditional estimate asked of a read declared ``across_runs``."""


@dataclass(frozen=True)
class Cell:
    request_id: Optional[str]
    family: Optional[str]
    opened: Optional[str]
    player: str
    opponent: str
    player_id: str
    opponent_id: str
    rows: int
    w: int
    l: int
    d: int
    aborted: int
    pairs: Optional[Tuple[int, ...]]
    verdict: str
    reasons: Tuple[str, ...]

    @property
    def games(self) -> int:
        return self.w + self.l + self.d

    @property
    def attempted(self) -> int:
        return self.games + self.aborted

    @property
    def n_pairs(self) -> int:
        return sum(self.pairs) if self.pairs is not None else 0

    @property
    def score(self) -> Optional[float]:
        """The mean per-game score (a draw = 1/2): over PAIRS on a mirrored cell, over games otherwise."""
        if self.pairs is not None:
            return MP.pair_score(list(self.pairs))
        return (self.w + 0.5 * self.d) / self.games if self.games else None


def _side(p: Dict[str, Any]) -> str:
    return str(p["sha256"]) if p.get("sha256") else f"id:{p['id']}"


def cells(got: LedgerRead, *, min_pairs: Optional[int] = None) -> List[Cell]:
    """The read's cells, in (opened, request, player, opponent) order, each with its verdict."""
    groups: Dict[Tuple[Optional[str], str, str], List[Dict[str, Any]]] = {}
    for r in got.rows:
        req = r["request"] or {}
        groups.setdefault((req.get("id"), _side(r["player"]), _side(r["opponent"])), []).append(r)
    out = []
    for (rid, p, o), rs in groups.items():
        req = rs[0]["request"] or {}
        mirrored = all(r["pairs"] is not None for r in rs)
        if not mirrored and any(r["pairs"] is not None for r in rs):
            raise MixedRegimeError(f"cell {rid} {p[:12]} vs {o[:12]}: mirrored and unmirrored rows")
        w, l, d, ab = (sum(r["counts"][k] for r in rs) for k in ("w", "l", "d", "aborted"))
        pc = tuple(MP.add_counts(*[r["pairs"]["counts"] for r in rs]) or ()) if mirrored else None
        reasons = []
        attempted = w + l + d + ab
        if attempted == 0:
            reasons.append("no game attempted")
        elif ABORT_LIMIT[1] * ab > ABORT_LIMIT[0] * attempted:
            reasons.append(f"aborted {ab} of {attempted} attempted games > 25 %")
        if min_pairs is not None:
            n = sum(pc) if pc is not None else 0
            if pc is None:
                reasons.append("a pair minimum on an unmirrored cell")
            elif n < min_pairs:
                reasons.append(f"{n} completed pairs < the registered {min_pairs}")
        out.append(Cell(request_id=rid, family=req.get("family"), opened=req.get("opened"), player=p, opponent=o,
                        player_id=rs[0]["player"]["id"], opponent_id=rs[0]["opponent"]["id"], rows=len(rs), w=w, l=l,
                        d=d, aborted=ab, pairs=pc, verdict="INCONCLUSIVE" if reasons else "OK",
                        reasons=tuple(reasons)))
    return sorted(out, key=lambda c: (c.opened or "", c.request_id or "", c.player, c.opponent))


def looks(got: LedgerRead, *, min_pairs: Optional[int] = None) -> List[Tuple[str, List[Cell]]]:
    """A FAMILY read across its looks: ``[(request id, cells)]`` in the order the looks were opened."""
    if got.decl.requests != "family":
        raise ValueError(f"{got.decl.name}: looks() reads a family (requests='family')")
    by: Dict[str, List[Cell]] = {}
    for c in cells(got, min_pairs=min_pairs):
        by.setdefault(str(c.request_id), []).append(c)
    return sorted(by.items(), key=lambda kv: (kv[1][0].opened or "", kv[0]))


def pooled_pairs(got: LedgerRead) -> Dict[str, Any]:
    """ONE edge's pooled pentanomial, its mean score over pairs and the pair-clustered 95 % interval with the
    unbiased pair SE. CONDITIONAL inference only (module docstring)."""
    if got.decl.inference != "conditional":
        raise InferenceScopeError(f"{got.decl.name}: declared inference={got.decl.inference!r}; a pooled pentanomial "
                                  "interval is conditional on the run (meter noise only) — use a seed-level estimator")
    edges = {(_side(r["player"]), _side(r["opponent"])) for r in got.rows}
    if len(edges) != 1:
        raise MixedRegimeError(f"{got.decl.name}: pooled_pairs reads ONE edge; the read holds {len(edges)}")
    if any(r["pairs"] is None for r in got.rows):
        raise MixedRegimeError(f"{got.decl.name}: pooled_pairs reads mirrored rows only")
    pc = MP.add_counts(*[r["pairs"]["counts"] for r in got.rows]) or MP.empty_counts()
    n = MP.n_pairs(pc)
    ci = MP.pair_score_ci(pc)
    se = None
    if n >= 2:
        mu = MP.pair_score(pc)
        assert mu is not None
        se = math.sqrt(sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(pc)) / (n - 1) / n)
    return {"pair_counts": [int(c) for c in pc], "n_pairs": n, "score": ci[0] if ci else None,
            "score_ci95": [ci[1], ci[2]] if ci else None, "se": se, "regime_id": got.regime_id,
            "inference": "conditional", "rows": len(got.rows)}
