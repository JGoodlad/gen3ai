"""The critic's entity pool: `UnifiedValueReadout`.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.

`MultiSeedValueReadout` — v61's k-seed window over the op's per-our-mon rows — lived here until
the critic-route deletion wave. It measured dV **0.0000, bit-exact, on two consecutive
end-of-run audits** (gen-13 and gen-14 at n=12,391), and it is deleted with the rest of the
post-assembler vf tail. Its own collapse contract (`seed_diagnostics.py`, `value_seeds/*`) went
with it. The seed line's finding is preserved in `designs/CHANGELOG.md`: a SHARED readout can
only constrain each seed's component along its own weight vector, so multiplicity was never the
missing axis — which is why both pressures applied to it (VICReg v62, quantile v63) were
themselves deleted at v78.
"""
import torch
from typing import Optional
from agents.model.arch_constants import (UVR_K, UVR_DIM, _UVR_N_SOURCES, _UVR_N_SOURCES_FULL,
    D_MODEL,
)


class UnifiedValueReadout(torch.nn.Module):
    """gen3_unified_value_readout_v1 (v80) — Stage-3 T3-DELIVER, the critic's ONE entity pool.

    `design_unified_belief.md` §3: the critic used to read magnitude through parallel bolt-on
    routes (the value CLS pool, the seed readout over op rows, the threat-inject, the hidden-opp
    concat) — two of which existed only because the others could not reach vf. This module is
    the designed successor CONTRACT, and the succession is now COMPLETE: the critic-route
    deletion wave retired the seed readout, the hidden-opp vf half and the nmr vf concat, and
    this pool carries **96–97% of the critic's whole route dependence** (gen-14, dV 5.490 of
    all_off 5.635). One attention pool over the critic's ENTITY-ROW SET — the 12
    post-transformer team tokens plus (when the op exists) its 6 per-our-mon incoming rows —
    each row projected to a shared `UVR_DIM` and tagged with a learned per-SOURCE type
    embedding, pooled by `UVR_K` learned queries. Permutation-invariant over entities by
    construction (softmax attention; a row's identity rides its content + source tag, never its
    position).

    Ships OPT-IN and ZERO-INIT: the output projection starts at exactly 0, so a cold-start
    forward contributes nothing (the M1 identity-at-init contract), and the flag OFF builds
    nothing (byte-identical baseline). vf-ONLY — the output is concatenated after the assembler
    into `value_pooled` (the v89 seam), so the policy is untouched at ANY weight, not merely at
    init. It did NOT replace the routes it succeeds by fiat — the critic_route_audit adjudicated
    those (`critic_route_audit.py`), and this pool won.

    Attention is EXPLICIT (softmax over ≤18 rows per query, the explicit-softmax pattern)
    rather than nn.MultiheadAttention: a degenerate all-masked row set (an all-fainted
    board) degrades to a uniform average instead of NaN, and the k×N pattern is
    stashed for diagnostics (`last_att`)."""

    def __init__(self, per_mon: int, full: bool = False):
        super().__init__()
        self.token_proj = torch.nn.Linear(D_MODEL, UVR_DIM)
        self.op_proj = torch.nn.Linear(per_mon, UVR_DIM) if per_mon > 0 else None
        # `full` (gen3_unified_value_readout_v2, v82): the COMPLETE row set — the refined
        # GLOBAL token (source 3) and the hidden-opp BELIEF queries (source 4) join the pool,
        # making it the one successor for every vf route the audit may condemn (nmr's direct
        # concat, the hidden-opp vf half, seed, threat). A SEPARATE flag/shape on purpose:
        # v80-shape checkpoints (gen-12 trains one) keep loading under the 3-row table.
        self.full = bool(full)
        n_sources = _UVR_N_SOURCES_FULL if full else _UVR_N_SOURCES
        self.source_emb = torch.nn.Parameter(torch.randn(n_sources, UVR_DIM) * 0.02)
        self.queries = torch.nn.Parameter(torch.randn(UVR_K, UVR_DIM) * (UVR_DIM ** -0.5))
        self.out_proj = torch.nn.Linear(UVR_K * UVR_DIM, D_MODEL)
        torch.nn.init.zeros_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)
        self.out_dim = D_MODEL
        self.last_att: Optional[torch.Tensor] = None    # [B, K, N] — the diagnostics read

    def forward(self, our_team_out: torch.Tensor, their_team_out: torch.Tensor,
                all_fainted: torch.Tensor,
                op_rows: Optional[torch.Tensor] = None,
                op_alive: Optional[torch.Tensor] = None,
                global_row: Optional[torch.Tensor] = None,
                belief_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """our/their_team_out [B,6,D_MODEL]; all_fainted [B,12] bool (True = masked);
        op_rows [B,6,per_mon] + op_alive [B,6] float when the op exists; under `full`:
        global_row [B,D_MODEL] (required — never masked) and belief_rows [B,K,D_MODEL]
        (optional — present only with a HiddenOppBeliefPool) → [B, D_MODEL], added into `value_pooled`."""
        rows = [self.token_proj(our_team_out) + self.source_emb[0],
                self.token_proj(their_team_out) + self.source_emb[1]]
        masks = [all_fainted]
        if self.op_proj is not None:
            if op_rows is None or op_alive is None:
                raise ValueError(
                    "value_entity_pool was built WITH the op's row source but the forward "
                    "supplied no op rows — a silent skip would shrink the row set and read "
                    "exactly like a working pool.")
            rows.append(self.op_proj(op_rows) + self.source_emb[2])
            masks.append(op_alive.clamp(max=1.0) < 0.5)
        if self.full:
            if global_row is None:
                raise ValueError(
                    "value_entity_pool_full is built but the forward supplied no global row — "
                    "a silent skip would shrink the row set and read like a working pool.")
            rows.append(self.token_proj(global_row)[:, None, :] + self.source_emb[3])
            masks.append(torch.zeros(global_row.shape[0], 1, dtype=torch.bool,
                                     device=global_row.device))
            if belief_rows is not None:
                rows.append(self.token_proj(belief_rows) + self.source_emb[4])
                masks.append(torch.zeros(belief_rows.shape[0], belief_rows.shape[1],
                                         dtype=torch.bool, device=belief_rows.device))
        kv = torch.cat(rows, dim=1)                                        # [B, N, dim]
        masked = torch.cat(masks, dim=1)                                   # [B, N] bool
        att = torch.einsum("kd,bnd->bkn", self.queries, kv) * (UVR_DIM ** -0.5)
        att = att.masked_fill(masked[:, None, :], -1e9)
        att = torch.softmax(att, dim=-1)                                   # [B, K, N]
        self.last_att = att
        out = torch.einsum("bkn,bnd->bkd", att, kv)                        # [B, K, dim]
        return self.out_proj(out.reshape(out.shape[0], UVR_K * UVR_DIM))  # type: ignore[no-any-return]
