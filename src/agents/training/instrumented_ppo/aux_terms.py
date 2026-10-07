"""The DELEGATES — terms whose bodies live in their own modules, bound onto the class here.

A family that moved out under the same pressure this split answers left a thin binding behind so every
call site and every `model._<name>` test resolves unchanged:

* `belief_bank` — the five supervised belief losses, as one declarative fold (one ROW per head).
"""
from agents.training import belief_bank as _belief_bank


class AuxTerms:
    """The belief-bank delegates."""

    # ALL FIVE supervised belief losses now live in `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _move_belief_loss = staticmethod(_belief_bank.move_belief_loss)
    # The three revealed-slot supervised losses MOVED to `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _spread_belief_loss = staticmethod(_belief_bank.spread_belief_loss)

    _nature_ev_belief_loss = staticmethod(_belief_bank.nature_ev_belief_loss)

    _hp_type_belief_loss = staticmethod(_belief_bank.hp_type_belief_loss)

    _move_belief_latent_loss = staticmethod(_belief_bank.move_belief_latent_loss)

