"""`QWinProbHead` — the per-ACTION win-probability readout (`gen3_q_winprob_head_v1`, E5 step 1).

WHY IT EXISTS, precisely. This network has a **V architecture, not a Q architecture**: every value
readout in the tree (`value_net`, `WinProbHead`, `ValueDistHead`, `ShadowValueHead`) evaluates a
STATE. So "what is my win probability if I click Rock Slide?" is not a read — it is a
*manufacturing* job: eleven successor states, eleven re-rolls through the simulator, and then
eleven forwards. A search teacher gets its per-action distribution for free; we pay the sim.

This head is the amortization of that cost. It scores each legal action from the token of the
entity that action selects — the SAME per-action tokens `PointerNativeActionHead` scores — so one
forward yields eleven win probabilities. Ledger 229e9f1 / 5edbd05: *the simulator distilled into a
forward pass, amortized one-ply search.* Downstream, a search leaf becomes free, "poor-man's
distillation" becomes teacher-cheap, and the **amortization residual** (this head vs a true
re-roll, per state class) is the value of one-ply search stated as a number.

🚨 **THE STARVATION TRAP, named up front, because it is what makes a naive version WORSE than
nothing.** On-policy data labels exactly ONE action per state — the one that was taken. The
measured preferred-alternative rate is p≈0.002, so a Q head trained on the on-policy stream alone
is untrained precisely on the moves the policy never tries: **confidently wrong exactly where a
per-action readout would be used.** The labels this head is designed for are therefore
COUNTERFACTUAL — per-action re-rolls from the R1 label factory — and the on-policy fallback is a
separately-weighted, default-off term whose bias is documented at its own coefficient.

WHAT IT IS STRUCTURALLY.

  * **ONE shared scorer over all eleven action slots.** Three input projections exist (moves,
    switches, struggle) only because the three families carry different WIDTHS — a move slot is
    `[E3 seat ⊕ its op move cells]`, a switch slot is `[our team token ⊕ its incoming/OAX cells]`,
    struggle has no entity at all. Everything after the projection is shared, so the readout is
    permutation-EQUIVARIANT within a family: permute our team and the six switch Q values permute
    with it. The pointer head's own lesson — a flat `Linear(ctx, 11)` learns "slot 0 is usually
    right" from an ordering that means nothing — applies identically here, and a Q head that
    memorised a slot index would be worthless as a search leaf.
  * **ZERO-INIT scorer (weight AND bias)** ⇒ every Q logit is exactly 0 at init ⇒ every
    P(win|s,a) = 0.5 ⇒ the untrained ranking is a uniform tie, which is the honest state of
    knowledge for a head that has seen no label. (`identity_init_test`'s guard set captures it by
    OBSERVATION, so SB3's ortho-init clobber is repaired for free — see
    `Gen3FeaturesExtractor.restore_identity_init`.)
  * **A SIDE readout.** The logits are stashed at `features_extractor.last_q_winprob_logits` and
    read only by the auxiliary loss, the offline probe and the prober. They are NEVER concatenated
    into `pi` or `vf`, and there is no `shaping` mode: `read_only` feeds every input STOP-GRAD, so
    the head trains its own parameters and cannot perturb the policy at any coefficient. That is
    deliberately the `CfEvidentialHead` contract rather than `WinProbHead`'s tri-state — a
    per-action readout carrying a counterfactual label is a strictly larger leak surface than a
    per-state one, and light trunk shaping is a *later* decision that owes its own gate (E5 step 1
    names it; nothing here presumes it).

THE CONTEXT VECTOR is `value_pooled` — the critic's whole-board "who's winning" summary — not the
pointer head's `latent_pi`. Two reasons, and the second is the binding one: (a) a win probability
is a value question, and `value_pooled` is the value-side aggregation; (b) `latent_pi` is produced
by `mlp_extractor` on the POLICY, downstream of the extractor, so a head living in the extractor
cannot see it. Q(s,a) needs both the board summary and the action's own entity token, and that is
exactly what this composes.
"""
from typing import Tuple

import torch

from agents.action.constants import (
    ACTION_SPACE_SIZE,
    MOVE_START,
    N_MOVE_SLOTS,
    N_SWITCH_SLOTS,
    STRUGGLE,
    SWITCH_START,
)
from agents.model.arch_constants import POINTER_HIDDEN

#: The legal values of `--q-winprob-mode`. `none` = the module is not built at all (the chain is
#: byte-for-byte the baseline); `read_only` = built, called on every forward, every input
#: stop-grad. There is deliberately NO `shaping` value — see the class docstring.
Q_WINPROB_MODES: Tuple[str, ...] = ("none", "read_only")


class QWinProbHead(torch.nn.Module):
    """Per-action P(win | s, a) logits over the pointer head's own action tokens.

    Output layout is the ACTION SPACE (`agents/action/constants.py`):
    ``[switch x6, move x4, struggle]``, so index `a` of the returned tensor is the win-probability
    logit of action `a` — the same index the policy's logits, the action mask and every label
    stream use. That identity is the whole interface: a Q head whose column order disagreed with
    the action space would be the order-mismatch bug class with no shape error to catch it.
    """

    def __init__(self, move_token_dim: int, d_model: int, ctx_dim: int,
                 move_cell_dim: int = 0, switch_cell_dim: int = 0,
                 hidden: int = POINTER_HIDDEN) -> None:
        super().__init__()
        self.move_cell_dim = int(move_cell_dim)
        self.switch_cell_dim = int(switch_cell_dim)
        self.hidden = int(hidden)
        self.ctx_proj = torch.nn.Linear(ctx_dim, hidden)
        self.move_proj = torch.nn.Linear(move_token_dim + self.move_cell_dim, hidden)
        self.switch_proj = torch.nn.Linear(d_model + self.switch_cell_dim, hidden)
        # ONE scorer for all eleven slots — the parameter sharing that makes the readout
        # equivariant. Zero-init (weight AND bias) => every logit is exactly 0 at init.
        self.q_score = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.q_score.weight)
        torch.nn.init.zeros_(self.q_score.bias)

    def forward(self, ctx_vec: torch.Tensor, move_tokens_req: torch.Tensor,
                move_valid: torch.Tensor, team_tokens: torch.Tensor,
                move_cells: torch.Tensor, switch_cells: torch.Tensor) -> torch.Tensor:
        """`ctx_vec` [B, ctx_dim] = `value_pooled`; the remaining five are `PointerInputs` verbatim.

        Returns [B, ACTION_SPACE_SIZE] win-probability LOGITS (``sigmoid`` ⇒ P(win | s, a)).

        An unresolved request slot (forced Struggle, or a mon with fewer than four moves) is
        forced to logit 0 ⇒ P = 0.5 ⇒ "no information", rather than a score computed from the
        zeroed placeholder token. It also cuts the gradient there, which is correct: such a slot
        selects no action and can carry no label.
        """
        c = self.ctx_proj(ctx_vec)                                            # [B,H]
        m_in = torch.cat([move_tokens_req, move_cells], dim=-1)               # [B,4,tok+cell]
        s_in = torch.cat([team_tokens, switch_cells], dim=-1)                 # [B,6,d_model+cell]
        m = torch.tanh(self.move_proj(m_in) + c[:, None, :])                  # [B,4,H]
        s = torch.tanh(self.switch_proj(s_in) + c[:, None, :])                # [B,6,H]
        move_q = self.q_score(m).squeeze(-1) * move_valid                     # [B,4]
        switch_q = self.q_score(s).squeeze(-1)                                # [B,6]
        struggle_q = self.q_score(torch.tanh(c))                              # [B,1]
        out = torch.cat([switch_q, move_q, struggle_q], dim=-1)               # [B,11]
        # A structural pin, not a defensive check: the concat order above IS the action space, and
        # the three constants below are the contract every consumer indexes by.
        assert out.shape[-1] == ACTION_SPACE_SIZE, (
            f"QWinProbHead emitted {out.shape[-1]} logits, action space is {ACTION_SPACE_SIZE} "
            f"(switch {SWITCH_START}..{N_SWITCH_SLOTS - 1}, move {MOVE_START}.."
            f"{MOVE_START + N_MOVE_SLOTS - 1}, struggle {STRUGGLE})")
        return out
