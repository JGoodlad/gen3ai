"""EXTERNAL-ANCHOR reads — our checkpoint against a third-party gen3ou agent, at a matched and
VERIFIED regime, with the Wilson interval and the regime on every row.

`python -m main.anchors` is the entry point; `designs/ops/EXTERNAL_ANCHORS_SOP.md` is the
procedure (the three tiers, the greedy rule and why, the exact commands, the cost per read, the
known hazards and the standing numbers).
"""
from main.anchors.results import newcombe, wilson

__all__ = ["wilson", "newcombe"]
