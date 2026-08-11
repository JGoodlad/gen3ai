"""PAIR REDUCTION rungs — the typed, swappable reducers over the opponent's believed-move axis.

`design_pair_reduction.md` (ADOPTED §8.1, steps 3-4). The op's production reduction is the legacy
per-channel hard max (`DamageOperator._chan_max` + the eight inline `amax` calls) — R0, the
byte-identity anchor, deliberately kept OUTSIDE Contract W (its argmax is per-channel AND
per-defender; that incoherence is defect D2/D3, documented, not fixed in place). This module holds
the CONTRACT-obeying rungs that run BESIDE it (§9: add beside, never replace):

  Contract W (§3.1) — a weighted reducer emits ``α ∈ Δ^C``: ONE distribution over the believed-move
  axis, computed from the board and their moves ONLY — never per defender, never per channel. The
  reduced row is ``Σ_c α_c · cell[j, c, :]`` — a convex combination of HP fractions, which is an HP
  fraction (units survive). `α ≠ w`: presence belief in, usage belief out.

  Contract L (§3.2) — a latent reducer ``ρ(POOL_c φ(cell, w))``, φ before the pool (PointNet's
  lesson: max over LEARNED features, never the raw physical quantity), sum AND max pools (GIN:
  sum ≻ mean ≻ max; PNA: no single aggregator suffices). φ's input carries second-order channels
  (§4.1) so an α-weighted/pooled sum can express E[o] AND E[o²] ⇒ variance ⇒ hedging.

Rungs (the §4 ladder):
  R0  ``hard_max``     — no module (None); the legacy path in damage_op.py is untouched.
  R1  ``belief_mean``  — α = w/Σw, no params.
  R2W ``learned``      — α = softmax(log(w+eps) + g(board-level candidate stats)); g's head is
                         ZERO-INIT so α at init == normalize(w) EXACTLY (step 4).
  R2L ``deepsets_sum`` / ``deepsets_max`` — ρ(Σφ)/ρ(maxφ), ρ zero-init (identity-at-init: the
                         extractor's observation-captured guard re-zeros after SB3's ortho pass).
  R3  ``multi``        — [belief_mean ‖ deepsets_sum ‖ deepsets_max] concatenated.

The W-rungs emit ``[E_α[cell], E_α[cell²], Σ_c α_c·w_c]`` per defender — mean, second moment, and
the α-provenance (§3.1: what `provenance` was always trying to say). Everything here is
DELIVERY-AGNOSTIC: `DamageOperator` stashes the output on `last_reduced_extra`; wiring it into the
switch cell / prefuse / seed rows (and the config/versioning that entails) is the gen-6 boundary's
work, so the production forward stays byte-identical at any `how` until then.
"""
from __future__ import annotations

from typing import Optional

import torch

PAIR_REDUCE_EPS = 1e-8
PAIR_REDUCE_LATENT = 16          # ≥ set size 6 (Wagstaff 2019) — universality is free at 16
_W_HOWS = ("belief_mean", "learned")
_L_HOWS = ("deepsets_sum", "deepsets_max")
PAIR_REDUCE_HOWS = ("hard_max",) + _W_HOWS + _L_HOWS + ("multi",)


def alpha_belief_mean(w: torch.Tensor) -> torch.Tensor:
    """R1 — the honest marginal: α = w/Σw. [B,C] → [B,C]; all-zero belief rows stay all-zero
    (no threat ⇒ no usage mass ⇒ reduced row 0, matching the legacy no-threat gate)."""
    total = w.sum(dim=-1, keepdim=True)
    return torch.where(total > PAIR_REDUCE_EPS, w / total.clamp_min(PAIR_REDUCE_EPS),
                       torch.zeros_like(w))


def alpha_hard_max(w: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """R0 in CONTRACT form (§3.1 migration note): α = onehot(argmax_c w_c · ref_c) with `ref` a
    board-level reference roll [B,C]. Used by tests to show today's *modal* selection is inside
    Contract W; the production hard max stays the legacy per-channel path in damage_op.py."""
    idx = (w * ref).argmax(dim=-1, keepdim=True)
    return torch.zeros_like(w).scatter_(-1, idx, 1.0)


def reduce_with_alpha(alpha: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Contract W's one line: ``Σ_c α_c · cell`` — [B,C] × [B,J,C,F] → [B,J,F]. α is
    defender-independent by SIGNATURE (no J axis exists on it): D3 is a shape error here."""
    return torch.einsum("bc,bjcf->bjf", alpha, cells)


class AlphaLearned(torch.nn.Module):
    """R2W — α = softmax(log(w+eps) + g(candidate)), g reading BOARD-LEVEL candidate stats only
    (w_c + the defender-MEAN of each cell channel — no per-defender input can reach the logit, so
    Contract W's defender-independence is structural). Zero-init head ⇒ α(init) == w/Σw exactly."""

    def __init__(self, n_channels: int, hidden: int = PAIR_REDUCE_LATENT):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1 + n_channels, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, w: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        board_stats = cells.mean(dim=1)                                   # [B,C,F] (mean over J)
        feats = torch.cat([w[..., None], board_stats], dim=-1)            # [B,C,1+F]
        logits = self.net(feats).squeeze(-1)                              # [B,C]
        prior = torch.log(w.clamp_min(PAIR_REDUCE_EPS))
        alpha = torch.softmax(prior + logits, dim=-1)
        # Degenerate all-zero-belief rows: keep the R1 convention (row of zeros, not uniform).
        live = (w.sum(dim=-1, keepdim=True) > PAIR_REDUCE_EPS).float()
        return alpha * live


class DeepSetsReducer(torch.nn.Module):
    """R2L — ρ(POOL_c φ(cell, cell², w)). φ runs BEFORE the pool; pool is 'sum' or 'max' over the
    candidate axis; ρ's final Linear is zero-init (output ≡ 0 at construction — the extractor's
    identity-init guard protects it through SB3's ortho pass)."""

    def __init__(self, n_channels: int, pool: str, latent: int = PAIR_REDUCE_LATENT):
        super().__init__()
        assert pool in ("sum", "max")
        self.pool = pool
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(2 * n_channels + 1, latent), torch.nn.ReLU(),
            torch.nn.Linear(latent, latent),
        )
        self.rho = torch.nn.Linear(latent, latent)
        torch.nn.init.zeros_(self.rho.weight)
        torch.nn.init.zeros_(self.rho.bias)

    def forward(self, w: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        B, J, C, F = cells.shape
        msg_in = torch.cat([cells, cells.pow(2),
                            w[:, None, :, None].expand(B, J, C, 1)], dim=-1)   # [B,J,C,2F+1]
        msg = self.phi(msg_in)                                                 # [B,J,C,L]
        pooled = msg.sum(dim=2) if self.pool == "sum" else msg.amax(dim=2)     # [B,J,L]
        return self.rho(pooled)


class PairReducer(torch.nn.Module):
    """The facade `DamageOperator` builds: `how` selects the rung(s); `forward(w, cells)` returns
    the per-defender extra features [B,J,extra_dim]. `how='hard_max'` is represented by NOT
    building one (`build_pair_reducer` → None) — R0 has no contract-side module."""

    def __init__(self, how: str, n_channels: int):
        super().__init__()
        if how not in PAIR_REDUCE_HOWS or how == "hard_max":
            raise ValueError(f"PairReducer how={how!r} — one of {PAIR_REDUCE_HOWS[1:]}")
        self.how = how
        self.n_channels = n_channels
        w_hows = [h for h in _W_HOWS if how in (h, "multi")]
        if how == "multi":
            w_hows = ["belief_mean"]                       # §5 default bundle: R1 + both L pools
        self.w_hows = w_hows
        self.alpha_learned = AlphaLearned(n_channels) if "learned" in w_hows else None
        l_pools = [p for p in ("sum", "max")
                   if how == "multi" or how == f"deepsets_{p}"]
        self.deepsets = torch.nn.ModuleDict(
            {p: DeepSetsReducer(n_channels, p) for p in l_pools})
        # W-rung output: E[cell] + E[cell²] + α-provenance; L-rung: one latent per pool.
        # Delegated so `pair_reduce_extra_dim` (used before this module exists) cannot drift from it.
        self.extra_dim = pair_reduce_extra_dim(how, n_channels)

    def forward(self, w: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        parts = []
        for how in self.w_hows:
            alpha = (alpha_belief_mean(w) if how == "belief_mean"
                     else self.alpha_learned(w, cells))
            mean1 = reduce_with_alpha(alpha, cells)                            # E_α[cell]
            mean2 = reduce_with_alpha(alpha, cells.pow(2))                     # E_α[cell²] (§4.1)
            prov = (alpha * w).sum(dim=-1)[:, None, None].expand(
                cells.shape[0], cells.shape[1], 1)                             # Σ α·w (§3.1)
            parts += [mean1, mean2, prov]
        for pool in self.deepsets:
            parts.append(self.deepsets[pool](w, cells))
        return torch.cat(parts, dim=-1)


def pair_reduce_extra_dim(how: str, n_channels: int) -> int:
    """The width `PairReducer(how, n_channels)` emits per defender, WITHOUT building one.

    Exists because a consumer may need the width before the op is constructed (the extractor builds
    `CLSPool` earlier than `DamageOperator`, and module construction order is load-bearing — SB3
    restores optimizer state POSITIONALLY, so reordering to suit a new feature silently corrupts
    every resume). Kept as the single source of that arithmetic: `PairReducer.__init__` calls this
    too, so the two cannot drift.
    """
    if how == "hard_max":
        return 0
    if how not in PAIR_REDUCE_HOWS:
        raise ValueError(f"pair_reduce_extra_dim how={how!r} — one of {PAIR_REDUCE_HOWS}")
    w_hows = ["belief_mean"] if how == "multi" else [h for h in _W_HOWS if how == h]
    n_pools = len([p for p in ("sum", "max") if how == "multi" or how == f"deepsets_{p}"])
    return len(w_hows) * (2 * n_channels + 1) + n_pools * PAIR_REDUCE_LATENT


def build_pair_reducer(how: str, n_channels: int) -> Optional[PairReducer]:
    """None for 'hard_max' (the legacy path IS R0); a PairReducer for every other rung."""
    if how == "hard_max":
        return None
    return PairReducer(how, n_channels)
