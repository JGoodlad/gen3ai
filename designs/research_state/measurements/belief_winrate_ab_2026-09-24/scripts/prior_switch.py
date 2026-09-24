"""The PRIOR-ONLY inference switch (measurement-only; NOT production code).

Every prior-fused belief head in the pinned (`6eb9c776`) extractor computes

    posterior_logits = Smogon prior (a non-persistent buffer)  ⊕  Linear(read)   # the learned delta

and the delta Linear is zero-init, so a zero delta IS the prior. This switch registers a forward hook
on each delta Linear that, while ON, replaces the Linear's output with exact zeros. Everything
downstream of the delta is untouched: revealed moves are still pinned certain (`_REVEAL_LOGIT`, applied
AFTER the delta is added), the typed-HP composition still runs (on the prior HP-type posterior), and the
trained `reinject` projections still soft-embed the (now prior) posterior into the tokens.

Heads covered (`SCOPES`):
  * "all"  — MoveBelief.move_head (the reinjected move posterior; reaches pi/vf),
             HPTypeBelief.type_head (typed-HP composition + its own reinject; reaches pi/vf),
             SpreadBelief.nature_head + ev_head (the op + reinject; reaches pi/vf),
             ItemBelief.item_head (P(Choice Band) in the op; reaches pi/vf),
             BeliefHead.species_head (species SIDE READOUT; never enters pi/vf — covered for completeness).
  * "move" — MoveBelief.move_head only (the secondary attribution arm).
NOT covered (they are not prior⊕delta heads): BeliefHead.moves_head (a from-scratch side readout),
T0SpeciesPrior (parameter-free Smogon naive Bayes, already prior-only), the intent heads, the latent table.

While OFF the hooks return None, so the module output is the Linear's own output object, unchanged.
"""
from typing import Dict, List

import torch

# (attribute path on the features extractor, delta Linear attribute)
_ALL = [
    ("move_belief", "move_head"),
    ("hp_type_belief_head", "type_head"),
    ("spread_belief", "nature_head"),
    ("spread_belief", "ev_head"),
    ("item_belief_head", "item_head"),
    ("belief_head", "species_head"),
]
SCOPES: Dict[str, List[tuple]] = {"all": _ALL, "move": [("move_belief", "move_head")]}


class PriorOnlySwitch:
    """Hooks are registered once, at construction, and gated by `self.on`."""

    def __init__(self, model, scope: str = "all"):
        fe = model.policy.features_extractor
        self.scope = scope
        self.on = False
        self.calls = 0            # delta-Linear calls zeroed while ON (non-vacuity counter)
        self.covered = []
        self._handles = []
        for owner, attr in SCOPES[scope]:
            mod = getattr(fe, owner, None)
            if mod is None:
                raise RuntimeError(f"prior switch: extractor has no {owner!r} (is the belief stack on?)")
            lin = getattr(mod, attr, None)
            if not isinstance(lin, torch.nn.Linear):
                raise RuntimeError(f"prior switch: {owner}.{attr} is not a Linear delta head")
            self._handles.append(lin.register_forward_hook(self._hook))
            self.covered.append(f"{owner}.{attr}")

    def _hook(self, module, inputs, output):
        if not self.on:
            return None
        self.calls += 1
        return torch.zeros_like(output)

    def __enter__(self):
        self.on = True
        return self

    def __exit__(self, *exc):
        self.on = False
        return False

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
