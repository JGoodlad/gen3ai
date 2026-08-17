"""gen3_value_direct_routes_v1 — two direct CRITIC routes: the deadline clock, and α/β.

Both answer an audit finding, and both INJECT additively into `value_pooled`
(gen3_value_pooled_routes_v1, v89 — the old post-assembler vf-tail concat was structurally
bypassed by `--value-from-dist`, whose dist-head critic reads `value_pooled` only):

* **`ValueClockRoute`** — the v67 deadline clock (`gen3_deadline_clock_v1`: log-elapsed,
  remaining-linear, log-remaining) was built for the CRITIC — "a critic cannot price a deadline
  it has no resolution on", measured as a POSITIVE V on the final decision in 13 of 14 timeout
  losses. Its only direct head route since the ctx-dedup era is the `non_matchup_rest` concat,
  which the route audit read as dead — so the fix that was validated before the reroute gets its
  own explicit route: the 3 raw scalars through a zero-init projection, vf only.
* **`ValueIntentRoute`** — α/β have never reached the critic AS DISTRIBUTIONS: α enters vf only
  as a weighting inside `intent_value_reduce`'s physics cells, β not at all (it was structurally
  label-only until the v85 boom cell). The block was ORDERING — α/β are scored at T2, after the
  pools — which the post-pool injection point dissolves. The route feeds the PUBLISHED posteriors
  (stop-grad under `label_only`, so no PPO→α/β route opens) as probabilities: α over its K
  belief-sorted seats + SWITCH (a canonical order — the same axis every α consumer aligns to),
  β over the 6 team slots (the obs slot convention), with β's no-legal-candidate case gated to
  zero rather than a NaN softmax.

Zero-init ⇒ ON-at-init is bit-identical on the critic and the policy is untouched at any weight
(vf-only concat). Ledger M1: `restore_identity_init` captures both projections by observation.
"""
from __future__ import annotations

import torch

from agents.model.arch_constants import D_MODEL
from agents.observation.constants import CLOCK_DIM


class ValueClockRoute(torch.nn.Module):
    """`clock [B, CLOCK_DIM] → [B, D_MODEL]`, zero-init — added into `value_pooled`
    (gen3_value_pooled_routes_v1: the vf-tail concat never reached the dist-head critic)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(CLOCK_DIM, D_MODEL)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, clock: torch.Tensor) -> torch.Tensor:
        return self.proj(clock)  # type: ignore[no-any-return]


class ValueIntentRoute(torch.nn.Module):
    """`(published α logits [B,K+1], published β logits [B,6]) → [B, D_MODEL]`, zero-init —
    added into `value_pooled`. `n_seats` is the α head's K (entity_topk_seats), fixed per run."""

    def __init__(self, n_seats: int):
        super().__init__()
        self.n_seats = int(n_seats)
        self.proj = torch.nn.Linear(self.n_seats + 1 + 6, D_MODEL)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, alpha_logits: torch.Tensor, beta_logits: torch.Tensor) -> torch.Tensor:
        if alpha_logits.shape[-1] != self.n_seats + 1:
            raise ValueError(
                f"value intent route was built for {self.n_seats} seats + SWITCH but alpha has "
                f"{alpha_logits.shape[-1]} classes — the seat axis was reconfigured "
                "independently (the `op move-order` bug class).")
        alpha = torch.softmax(alpha_logits.float(), dim=-1).to(beta_logits.dtype)   # [B,K+1]
        has_cand = torch.isfinite(beta_logits).any(-1, keepdim=True).float()        # [B,1]
        beta = torch.softmax(beta_logits.float().clamp(min=-1e9), dim=-1
                             ).to(beta_logits.dtype) * has_cand                     # [B,6]
        return self.proj(torch.cat([alpha, beta], dim=-1))  # type: ignore[no-any-return]  # [B,out]
