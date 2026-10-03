"""`QWinProbHead` — the per-ACTION readout over the pointer head's own action tokens.

**WHAT IS LEFT OF `gen3_q_winprob_head_v1` (E5 step 1) AFTER DELETION PASS L4.** The extractor-built
per-action P(win) head (`--q-winprob-mode read_only`, its two coefficients and the counterfactual-label
stream that trained it) was DELETED with the counterfactual training half. This CLASS survives because
the detached RIDE-ALONG **A head** (`ridealong_heads.AdvantageEnsemble`, `--ridealong-adv`) is built
from its shared-scorer structure — same per-action tokens, same zero-init scorer, same action-space
column order — and regresses it on the taken action's advantage instead of a win label. Nothing
else instantiates it.

WHY IT WAS BUILT, precisely. This network has a **V architecture, not a Q architecture**: every value
readout in the tree (`value_net`, `WinProbHead`) evaluates a STATE. So "what is my win probability if
I click Rock Slide?" is not a read — it is a *manufacturing* job: eleven successor states, eleven
re-rolls through the simulator, and then eleven forwards. This head scores each legal action from the
token of the entity that action selects — the SAME per-action tokens `PointerNativeActionHead`
scores — so one forward yields eleven values.

🚨 **THE STARVATION TRAP, kept because it is what makes a naive per-action readout WORSE than
nothing.** On-policy data labels exactly ONE action per state — the one that was taken. The
measured preferred-alternative rate is p≈0.002, so a head trained on the on-policy stream alone
is untrained precisely on the moves the policy never tries: **confidently wrong exactly where a
per-action readout would be used.** (The ride-along A head answers it with K bootstrapped members and
randomized priors, so a STARVED move keeps its members apart.)

WHAT IT IS STRUCTURALLY.

  * **ONE shared scorer over all eleven action slots.** Three input projections exist (moves,
    switches, struggle) only because the three families carry different WIDTHS — a move slot is
    `[E3 seat ⊕ its op move cells]`, a switch slot is `[our team token ⊕ its incoming/OAX cells]`,
    struggle has no entity at all. Everything after the projection is shared, so the readout is
    permutation-EQUIVARIANT within a family: permute our team and the six switch values permute
    with it. The pointer head's own lesson — a flat `Linear(ctx, 11)` learns "slot 0 is usually
    right" from an ordering that means nothing — applies identically here.
  * **ZERO-INIT scorer (weight AND bias)** ⇒ every logit is exactly 0 at init, which is the honest
    state of knowledge for a head that has seen no label.

THE CONTEXT VECTOR is `value_pooled` — the critic's whole-board summary — not the pointer head's
`latent_pi`: `latent_pi` is produced by `mlp_extractor` on the POLICY, downstream of the extractor,
so a readout living in the extractor cannot see it. A per-action value needs both the board summary
and the action's own entity token, and that is exactly what this composes.
"""
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
