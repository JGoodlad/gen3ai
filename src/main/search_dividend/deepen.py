"""The search TREE and its backup — iterative deepening under the wall-clock budget.

The registration fixed depth at 1; the owner's amendment minutes later said "fixed depth" was
shorthand for CHEAP, not a constraint, and licensed **iterative deepening**: run the depth-1 sweep
first, then — while budget remains — expand the top-``m`` candidates one ply deeper. This module
owns the shape that makes that expressible, and nothing else: it has no sim, no session, no model,
so its arithmetic is unit-testable against a hand-built tree.

**The backup alternates, and it alternates the SAME way the depth-1 aggregation did.** At OUR
nodes the value is a MAX (we choose); at the opponent's it is an α-WEIGHTED AVERAGE (we
marginalize, per the three-axis variance measurement that put the behavior-weighted opponent share
at 59.7%). Depth-1's ``argmax_a Σ_c α(c)·V(s')`` is exactly this backup on a two-level tree — which
is the point: deepening must not quietly change the estimator it is deepening.

**The beam prunes OUR actions at the root, and only there.** An action dropped from the beam is
never compared against a deepened one: :func:`beam_actions` selects the top ``m`` by the depth-1
value, and the caller restricts the final argmax to that set (:func:`selectable`). The alternative
— deepening some actions and comparing their values against other actions' depth-1 values — is the
classic iterative-deepening inconsistency, and here it would have a direction: a deeper value
integrates more opponent replies, so it is systematically the more pessimistic of the two, and a
shallow action would win on depth rather than on merit.

**A ply is expanded WHOLE or not at all.** :func:`plan_beam` picks the largest ``m ≥ 2`` whose
entire next ply fits the remaining budget; if none fits, the deepening stops and the decision is
reported at the depth it actually reached. Expanding *part* of a ply would leave a MAX node whose
children sit at two different depths — the same inconsistency one level down, where it is harder
to see.

⚠️ **WIDTH IS SPENT FIRST, so depth only fires once the width CAPS bind — and that is the
registered order, not an oversight.** The allocator raises ``m_opp``, then ``k_worlds``, then
``r_dice`` as far as each will go before the next axis is touched, and deepening gets whatever is
left. With the default caps (6 / 8 / 8) width absorbs the whole clock at the swept budgets:
measured 2026-08-23 on a live 1 s oracle mirror, the first ply alone realized ``m_opp=6``,
``r_dice≈3.2`` and 114 scored arms in 0.78 s of a 1 s budget, leaving no ply affordable — every
decision reported depth 1. **The lever that buys depth is therefore ``--max-opp`` /
``--max-dice``, not a bigger budget**: lowering a cap frees the remainder for a ply, which is
exactly the width-versus-depth trade the sweep exists to price. Reporting the realized depth is
what makes that visible instead of leaving ``--max-depth 3`` reading like a promise.

🚨 **DEPTH ≥ 2 IS BUILT AND GATED BUT NOT YET TRUSTWORTHY — do not quote a depth-2 number.**
First live run, 2026-08-23 (oracle mirror, 3 s, ``--max-opp 2 --max-dice 1 --max-worlds 1``):
deepening engages exactly as designed — 18-20 of ~23 searched decisions deepen, realized depth
reaches 3, mean realized depth 1.83-2.11 — but the deeper successor REPLAY emits tens of thousands
of poke-env ``"message thinks p1: X is active, but it's not"`` warnings, and on a pool team
carrying a non-ASCII nickname (``data/teams/others/`` has them; the top level does not) it raises
``KeyError: 'ptãra'`` — the mojibake of "Ptéra" — inside ``get_pokemon``. **The depth-1 arm
produces exactly ZERO of either on the SAME game indices and the same pinned seeds**, so the
deeper replay is what exposes it. Two things are known and one is not:

* it FAILS SAFE — the exception is caught by :meth:`SearchEngine.choose` and recorded as a counted
  ``search_error`` fallback, so a degraded decision is visible in the histogram rather than silently
  scored, and the battle completes normally;
* it is NOT the forced-switch case — :func:`~main.search_dividend.search.branchable` now applies
  the root's own "clean move selection only" rule at every ply, and the warning volume did not
  move, so that hypothesis is CLOSED rather than merely untried;
* the double-encode point in the search-driver chunk transport is UNDIAGNOSED. Closing it is the
  first job before any depth-≥2 reading is published.

At the DEFAULT width caps this caveat is inert: width absorbs the whole budget at every swept
value, so the shipped sweep runs at depth 1 and is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# The smallest beam worth deepening. At m=1 there is nothing left to compare, so a ply spent there
# cannot change the decision — it can only spend budget and inflate the reported depth.
MIN_BEAM = 2


@dataclass
class TreeNode:
    """One state in the search tree — a successor the sim produced and the critic scored.

    ``children`` maps OUR action index to the list of ``(opponent weight, child)`` pairs that
    action leads to. Several pairs may share an action for two independent reasons: one per
    opponent candidate (the marginalization) and one per CRN dice draw (the average), and the
    backup treats them identically — a weighted mean over everything we do not control.
    """

    node_id: Optional[str]
    ended: bool
    value: Optional[float] = None
    #: ``{action_index: sim choice string}`` for OUR legal actions here, straight from the REAL
    #: mapper (``MaterializedTrace.action_choices``). Empty when the node was never materialized —
    #: which is exactly when it cannot be deepened, so the two facts stay together.
    our_tokens: Dict[int, str] = field(default_factory=dict)
    #: The child's open requests, as the search driver returned them. The opponent's marginalization
    #: set at the next ply is built from this and nothing else.
    requests: Optional[dict] = None
    #: OUR action indices from the branch decision to here — the ``Branch.actions`` a deeper
    #: materialization needs, accumulated rather than re-derived.
    path: Tuple[int, ...] = ()
    children: Dict[int, List[Tuple[float, "TreeNode"]]] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.path)

    def add_child(self, action: int, weight: float, child: "TreeNode") -> None:
        self.children.setdefault(int(action), []).append((float(weight), child))

    def expandable(self) -> bool:
        """Can a ply be grown from here? Needs a live node, an unfinished battle, and our own
        legal surface — a node missing any of the three is a leaf by construction, not by choice."""
        return bool(self.node_id) and not self.ended and bool(self.our_tokens)


def backup(node: TreeNode) -> Optional[float]:
    """The value of ``node``: MAX over our actions of the α-weighted mean over theirs.

    Returns the node's own leaf value when it has no children (or when no child backed up to
    anything) — a leaf that was scored is worth its score whether or not we ran out of budget
    before deepening it.
    """
    best: Optional[float] = None
    for kids in node.children.values():
        acc = 0.0
        wsum = 0.0
        for w, child in kids:
            v = backup(child)
            if v is None:
                continue
            acc += w * v
            wsum += w
        if wsum <= 0.0:
            continue
        val = acc / wsum
        if best is None or val > best:
            best = val
    return best if best is not None else node.value


def per_action_values(root: TreeNode) -> Dict[int, float]:
    """``{our action: value}`` at the ROOT — the α-weighted mean over the opponent's replies of
    each child's backed-up value. This is the quantity the decision maximizes, and at depth 1 it
    is byte-for-byte the expression the registration wrote down."""
    out: Dict[int, float] = {}
    for a, kids in root.children.items():
        acc = 0.0
        wsum = 0.0
        for w, child in kids:
            v = backup(child)
            if v is None:
                continue
            acc += w * v
            wsum += w
        out[int(a)] = (acc / wsum) if wsum > 0.0 else 0.0
    return out


def beam_actions(values: Dict[int, float], m: int) -> List[int]:
    """The top ``m`` actions by value, ties broken by the lower index so a beam is reproducible."""
    return [a for a, _v in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:max(0, int(m))]]


def leaves_under(root: TreeNode, actions: Sequence[int], depth: int) -> List[TreeNode]:
    """Every expandable leaf at exactly ``depth`` reachable through one of ``actions``."""
    keep = {int(a) for a in actions}
    out: List[TreeNode] = []

    def walk(node: TreeNode) -> None:
        if node.depth == depth:
            if node.expandable():
                out.append(node)
            return
        for kids in node.children.values():
            for _w, child in kids:
                walk(child)

    for a, kids in root.children.items():
        if int(a) not in keep:
            continue
        for _w, child in kids:
            walk(child)
    return out


def plan_beam(root: TreeNode, values: Dict[int, float], depth: int, *, m_opp: int,
              n_opp_at: Callable[[TreeNode], int], arm_cost_s: float, ply_overhead_s: float,
              remaining_s: float) -> Tuple[List[int], List[TreeNode], int]:
    """Pick the widest beam whose ENTIRE next ply fits ``remaining_s``.

    Returns ``(beam, leaves, n_arms)``; an empty beam means the ply does not fit and the deepening
    stops here. ``n_opp_at(leaf)`` returns how many opponent candidates that leaf would branch on
    — passed in rather than computed here so this module never has to know what a request is.

    Widest-that-fits rather than fixed: the budget is the only thing that should decide a width
    here, and a beam that is narrower than affordable throws away the comparison the ply was spent
    to make.
    """
    order = beam_actions(values, len(values))
    for m in range(len(order), MIN_BEAM - 1, -1):
        beam = order[:m]
        leaves = leaves_under(root, beam, depth)
        if not leaves:
            continue
        n_arms = sum(len(leaf.our_tokens) * max(1, min(int(m_opp), int(n_opp_at(leaf))))
                     for leaf in leaves)
        if n_arms <= 0:
            continue
        if n_arms * arm_cost_s + ply_overhead_s <= remaining_s:
            return beam, leaves, n_arms
    return [], [], 0
