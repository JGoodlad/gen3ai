"""The BOARD tokens and the per-mon OP CONTENT of `--token-encoding static` (stage 2, `gen3_static_board_v1`;
`designs/endstate/design_static_tokens.md` §4, from `design_entity_coverage_audit.md` §6.1 B1 + B2).

`legacy` never builds or calls anything here: its single GLOBAL token (seat 12) is untouched.

Under ``static`` the per-mon tokens carry NO board fact (stage 1), so the board gets a home of its own:

* **OUR SIDE** (seat 12) and **THEIR SIDE** (seat 13), one SHARED projection (`TeamTransformer.side_proj`)
  over SIDE-RELATIVE content — row 0 of `side_features` is our side's facts, row 1 theirs, in the SAME
  columns, so "this side's Spikes" is one weight whichever side it is, and the token TYPE tells them apart
  (the game's side symmetry becomes weight sharing; audit §4 A2);
* **FIELD** (seat 14): the weather, the clock, turns since progress.

The counts (alive / fainted / revealed) are explicit side columns because a fainted mon is a MASKED key and a
softmax over the alive keys is an average that cannot count (audit §4 A1). Everything here is read from the
CURRENT observation (no new obs fact; B5–B9 are a separate lever): the board block (`global_env` + the board
scalars of `reactive`) and, for the counts and Sleep Clause, the per-mon slots of the REAL context.

**OP CONTENT** (`OpContent`, audit B2, a requirement of the arm): an edge bias only reweights a softmax row, so
an AMOUNT reaches a token only as content (§4 A3). Each mon on BOTH sides gets a zero-init projection of its
Spikes entry cost and its end-of-turn ledger (the `x` / `g` cells, one shared projection both sides — the same
side relativity), and each of THEIR mons what our active does to it (the `d1` cells of our four request-order
moves). Our mons' incoming rows keep riding `prefuse_proj` (unchanged). Zero-init ⇒ the board-content
injection is exactly 0 at init (`restore_identity_init` protects it, by observation).
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

from agents.model.arch_constants import D_MODEL
from agents.model.extractor_ctx import ExtractorContext
from agents.observation.constants import (CONDITION_DIM, POKEMON_CONDITION_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET,
                                          POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE)

#: The condition one-hot is [None, BRN, PAR, SLP, FRZ, PSN, TOX] (`damage_op_layout._COND_SLP_IDX`).
_COND_SLP_IDX = 3
assert _COND_SLP_IDX < CONDITION_DIM

#: The SIDE token's columns, in order — each a fact about ONE side, the same column for both sides.
SIDE_FACTS: Tuple[str, ...] = (
    "spikes_layers",        # /3 (global_env.hazards: [ours, theirs])
    "reflect", "light_screen", "safeguard", "mist",     # presence (global_env.screens, interleaved ours/theirs)
    "wish_pending",         # reactive.wish_floating_{our,opp}
    "alive_count",          # /6, from the per-mon slots (ours: hp > 0; theirs: `opp_addressable` — unrevealed = alive)
    "fainted_count",        # /6 (reactive.fainted)
    "revealed_count",       # /6, Σ species_known over the side's slots
    "sleep_clause_used",    # 1 iff a mon of this side is asleep from a NON-Rest source (the op's Sleep Clause read)
)
SIDE_DIM = len(SIDE_FACTS)

#: The FIELD token's columns.
FIELD_FACTS: Tuple[str, ...] = (
    "weather_none", "weather_sun", "weather_rain", "weather_sand", "weather_hail",
    "weather_permanent", "weather_turns_left",
    "clock_log_elapsed", "clock_remaining", "clock_log_remaining",
    "turns_since_progress",
)
FIELD_DIM = len(FIELD_FACTS)

#: Board seats: (OUR SIDE, THEIR SIDE, FIELD). Legacy's three are all the single GLOBAL seat, which is what
#: makes `EdgeBias`'s legacy writes byte-identical through the same code path.
BOARD_SEATS_LEGACY: Tuple[int, int, int] = (2 * TEAM_SIZE, 2 * TEAM_SIZE, 2 * TEAM_SIZE)
BOARD_SEATS_STATIC: Tuple[int, int, int] = (2 * TEAM_SIZE, 2 * TEAM_SIZE + 1, 2 * TEAM_SIZE + 2)
N_BOARD_TOKENS_STATIC = 3

#: OP CONTENT input widths: the `x` cell [entry_chip, pursuit_p, pursuit_eff, grounded] ⊕ the `g` cell
#: [leftovers, weather_chip, status_tick, leech] (both sides); the `d1` cell of our 4 request-order moves on one
#: of their mons, 4 × [low, high, crit, pko, type_mult, revealed] (their side).
OPC_AMOUNT_DIM = 4 + 4
OPC_OUTGOING_DIM = 4 * 6


class BoardOffsets(NamedTuple):
    """Where the board scalars sit inside `ExtractorContext.non_matchup_rest` (= global_env ⊕ the 5 board
    scalars of `reactive`), read off the LAYOUT so a reshuffle of the reactive block cannot silently mis-slice."""
    tsp: int
    wish_our: int
    wish_opp: int


def board_offsets(layout: Dict[str, Any]) -> BoardOffsets:
    from agents.observation.constants import GLOBAL_ENV_DIM
    rl = layout['reactive_layout']
    for k in ("turns_since_progress", "wish_floating_our", "wish_floating_opp"):
        assert rl[k]['dim'] == 1, f"reactive.{k} is no longer one scalar — the board tokens' read must follow"
    return BoardOffsets(tsp=GLOBAL_ENV_DIM + rl['turns_since_progress']['offset'],
                        wish_our=GLOBAL_ENV_DIM + rl['wish_floating_our']['offset'],
                        wish_opp=GLOBAL_ENV_DIM + rl['wish_floating_opp']['offset'])


def side_features(ctx: ExtractorContext, off: BoardOffsets) -> torch.Tensor:
    """[B, 2, SIDE_DIM]: row 0 OUR side, row 1 THEIR side, the SAME columns (`SIDE_FACTS`). Reads the REAL
    context (under X5 fixed_mass a hypothesis row must not count as a revealed mon)."""
    pp = ctx.pokemon_part
    nmr = ctx.non_matchup_rest
    sc = ctx.screen_feature                                                   # [B,8] (ours, theirs) × 4
    hp = ctx.hp_and_active[:, :, 0]
    alive_ours = (hp[:, :TEAM_SIZE] > 0).to(pp.dtype).sum(1) / TEAM_SIZE      # [B]
    alive_opp = ctx.opp_addressable.to(pp.dtype).sum(1) / TEAM_SIZE
    known = pp[:, :, POKEMON_SPECIES_KNOWN_OFFSET]
    # Sleep Clause: a sleeper from a NON-Rest source on the side (the op's `nonrest_sleep`, both sides); the 0/1
    # observation bits make the clamp exact, with no comparison.
    nonrest = pp[:, :, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX] * (1.0 - pp[:, :, POKEMON_SLEEP_BELIEF_OFFSET])
    slp = torch.stack([nonrest[:, :TEAM_SIZE].sum(1), nonrest[:, TEAM_SIZE:].sum(1)], 1).clamp(max=1.0)
    ours = torch.stack([ctx.spikes_feature[:, 0], sc[:, 0], sc[:, 2], sc[:, 4], sc[:, 6],
                        nmr[:, off.wish_our], alive_ours, ctx.fainted_feature[:, 0],
                        known[:, :TEAM_SIZE].sum(1) / TEAM_SIZE, slp[:, 0]], dim=-1)
    theirs = torch.stack([ctx.spikes_feature[:, 1], sc[:, 1], sc[:, 3], sc[:, 5], sc[:, 7],
                          nmr[:, off.wish_opp], alive_opp, ctx.fainted_feature[:, 1],
                          known[:, TEAM_SIZE:].sum(1) / TEAM_SIZE, slp[:, 1]], dim=-1)
    return torch.stack([ours, theirs], dim=1)


def field_features(ctx: ExtractorContext, off: BoardOffsets) -> torch.Tensor:
    """[B, FIELD_DIM] (`FIELD_FACTS`): the weather block (one-hot + permanent + turns left), the 3 clock
    scalars, turns since progress."""
    return torch.cat([ctx.weather_feature, ctx.turn_feature, ctx.non_matchup_rest[:, off.tsp:off.tsp + 1]], dim=-1)


class OpContent(torch.nn.Module):
    """The per-mon OP CONTENT (audit B2) — module docstring. Two ZERO-INIT projections:

    * ``amount_proj`` (``OPC_AMOUNT_DIM`` → ``D_MODEL``), shared by BOTH sides: the mon's `x` cell (its Spikes
      chip on entry, the Pursuit exposure, grounded) ⊕ its `g` cell (its end-of-turn ledger);
    * ``outgoing_proj`` (``OPC_OUTGOING_DIM`` → ``D_MODEL``), THEIR mons only: the `d1` cells of our four
      request-order moves on that mon (what our active does to it). ``None`` when the op has no outgoing
      kernel (`--damage-outgoing` off).

    Our mons' incoming rows ride `prefuse_proj` (unchanged)."""

    def __init__(self, outgoing: bool):
        super().__init__()
        self.amount_proj = torch.nn.Linear(OPC_AMOUNT_DIM, D_MODEL)
        self.outgoing_proj: Optional[torch.nn.Linear] = (
            torch.nn.Linear(OPC_OUTGOING_DIM, D_MODEL) if outgoing else None)
        for lin in (self.amount_proj, self.outgoing_proj):
            if lin is not None:
                torch.nn.init.zeros_(lin.weight)
                torch.nn.init.zeros_(lin.bias)

    def forward(self, x_cells: Tuple[torch.Tensor, torch.Tensor], g_cells: Tuple[torch.Tensor, torch.Tensor],
                d1_cells: Optional[torch.Tensor]) -> torch.Tensor:
        """``x_cells`` / ``g_cells``: (ours [B,6,4], theirs [B,6,4]) as the `x` / `g` kernels return them;
        ``d1_cells`` [B,4,6,6] (our request-order move k, their mon d, cell) or None → [B, 12, D_MODEL]."""
        amounts = torch.cat([torch.cat([x_cells[0], g_cells[0]], dim=-1),
                             torch.cat([x_cells[1], g_cells[1]], dim=-1)], dim=1)       # [B,12,8]
        out: torch.Tensor = self.amount_proj(amounts)
        if self.outgoing_proj is not None:
            if d1_cells is None:
                raise ValueError("OpContent was built with the outgoing route but received no d1 cells — a "
                                 "silent skip would read exactly like a route that learned nothing.")
            B = d1_cells.shape[0]
            per_mon = d1_cells.permute(0, 2, 1, 3).reshape(B, TEAM_SIZE, OPC_OUTGOING_DIM)    # [B,6,24]
            out = torch.cat([out[:, :TEAM_SIZE], out[:, TEAM_SIZE:] + self.outgoing_proj(per_mon)], dim=1)
        return out
