"""X5's hypothesis TOKENS in the phase chain (`gen3_x5_belief_tokens_v1`, build unit U3;
`designs/endstate/design_x5_belief_tokens.md` §3.4 / §3.5). `--belief-tokens fixed_mass` only — the
`blob` arm never calls anything here, so its forward is byte-identical to the pre-X5 model.

WHAT THIS MODULE OWNS (pure functions over the U2 `HypothesisSet`; no parameters of its own — the
`hypothesis_marker` lives on `HypothesisBuilder`, built from X5's private seed):

* **The hypothesis CONTEXT** (`hypothesis_ctx`). The hidden opponent slots' per-mon observation rows
  are replaced by their hypothesis's DEX ROW (U1's table — produced by THE observation encoder), and
  the categorical ids are re-sliced from the substituted block by THE shared slicer
  (`slice_pokemon_categoricals`). Every MASK stays the REAL one (`opp_believed_mask`,
  `opp_addressable`, the fainted / key-padding masks, the active indices): a hypothesis row reads
  species-known 1 and HP 1.0, and nothing may re-derive "revealed" or "alive" from those cells
  (U1 / U2 hand-offs). The T0 belief heads (move / HP-type / item / spread) read their species from
  this context, so a hypothesis seat's posterior is SPECIES-SPECIFIC (§3.4) instead of a constant.
* **The hypothesis TOKENS** (`splice_hypothesis_tokens`). `PokemonEncoder` — the SAME encoder a
  revealed mon goes through — encodes the hypothesis context; the hidden opponent slots take that
  token + the learned `hypothesis_marker`. The encoder is per-mon (its only cross-mon input is the
  active-context scatter, and a hypothesis is never the active), so the non-hidden rows of that
  pass equal the real pass's; they are taken from the real pass anyway (`torch.where`).
* **The presence KEY BIAS** (`key_log_presence`, `float_key_mask`). Every EXPECTATION-class (class E)
  reduction over opponent tokens adds ``log π_j`` to every query's logit for key ``j`` (ToMe's
  proportional attention, Bolya et al. 2023): 0 on a revealed key, the hypothesis's log π on a hidden
  slot's key, OTHER's log-mass on the OTHER_species key, a masked key keeps its −1e9 / −inf. π is
  DETACHED (M10). Two invariances pin it (`hypothesis_tokens_test`): I1, a key at π = 0 gives the
  same output as that key masked; I2, a key at presence w gives the same output as two identical
  copies whose presences sum to w (the copies' total MASS equals the whole's).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, NamedTuple, Optional

import torch

from agents.model.extractor_ctx import ExtractorContext, slice_pokemon_categoricals
from agents.model.hypothesis_set import HypothesisSet
from agents.observation.constants import TEAM_SIZE


class OppPresence(NamedTuple):
    """What a class-E pool over opponent tokens needs (fixed_mass only; None under blob)."""
    slot_log_pi: torch.Tensor     # [B,6]  log π of each opp slot's key (0 revealed, MASKED_LOG_PRESENCE off)
    other_out: torch.Tensor       # [B,D]  OTHER_species' REFINED token (the trunk's output at its seat)
    other_log_mass: torch.Tensor  # [B]    log Σ_tail π (MASKED_LOG_PRESENCE when masked)
    other_live: torch.Tensor      # [B]    bool — OTHER is a real key (structural)


def hypothesis_ctx(ctx: ExtractorContext, hs: HypothesisSet, layout: Dict[str, Any]) -> ExtractorContext:
    """``ctx`` with every hidden opponent slot's per-mon row replaced by its hypothesis's dex row and
    the categorical ids re-sliced from the substituted block; every mask and index is the REAL one."""
    opp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE]
    rows = hs.slot_rows.to(opp.dtype)
    sub = torch.where(hs.slot_is_hypothesis.unsqueeze(-1), rows, opp)
    pp = torch.cat([ctx.pokemon_part[:, :TEAM_SIZE], sub], dim=1)
    ids = slice_pokemon_categoricals(pp, layout)
    return dataclasses.replace(
        ctx, pokemon_part=pp, species_ids=ids["species_ids"], all_move_ids=ids["all_move_ids"],
        all_move_type_ids=ids["all_move_type_ids"], item_ids=ids["item_ids"],
        ability1_ids=ids["ability1_ids"], ability2_ids=ids["ability2_ids"],
        type1_ids=ids["type1_ids"], type2_ids=ids["type2_ids"], hp_probs=ids["hp_probs"],
        hp_and_active=ids["hp_and_active"], last_move_ids=ids["last_move_ids"])


def splice_hypothesis_tokens(role_tokens: torch.Tensor, role_hyp: torch.Tensor,
                             hs: HypothesisSet, marker: torch.Tensor) -> torch.Tensor:
    """[B,12,D]: the hidden opponent slots take the hypothesis pass's token + ``marker``; every other
    row keeps the real pass's token."""
    opp = role_tokens[:, TEAM_SIZE:2 * TEAM_SIZE]
    hyp = role_hyp[:, TEAM_SIZE:2 * TEAM_SIZE] + marker
    opp = torch.where(hs.slot_is_hypothesis.unsqueeze(-1), hyp, opp)
    return torch.cat([role_tokens[:, :TEAM_SIZE], opp], dim=1)


def float_key_mask(masked: torch.Tensor, log_presence: torch.Tensor) -> torch.Tensor:
    """The FLOAT key-padding mask a class-E pool adds to its logits: ``−inf`` on a masked key (what
    `nn.MultiheadAttention` makes of a BOOL mask), ``log π`` otherwise (F-X5-14)."""
    neg = torch.full((), float("-inf"), dtype=log_presence.dtype, device=log_presence.device)
    return torch.where(masked, neg, log_presence)


def key_log_presence(n_tokens: int, hs: HypothesisSet, other_index: int,
                     e5_offset: Optional[int] = None, e4: Optional[torch.Tensor] = None,
                     e4_offset: Optional[int] = None, e5_active: Optional[torch.Tensor] = None,
                     opp_active_local: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B, n_tokens] the per-KEY log-presence the TeamTransformer adds to every query's logit (all
    heads). Layout (`TeamTransformer`): 6 our team · 6 their team · global · the extra seats (E3 [4],
    E4 [K], E5 [6], OTHER_species, the event seats). Zero everywhere except:

    * their team slot ``j``: ``hs.slot_log_pi[:, j]`` (0 revealed, the hypothesis's log π hidden);
    * E5 tail seat ``j`` (``e5_offset`` + j): the log π of its OWNER mon (§3.1 — without it a
      hypothesis's E5 seat would count as a whole token); the opponent ACTIVE's E5 seat is OTHER_move
      under fixed_mass and takes ``e5_active`` (its log-mass) when given;
    * the E4 seats (``e4_offset`` .. + K): ``e4`` [B,K] (log π_m; 0 for a revealed move) when given;
    * OTHER_species (``other_index``): ``hs.other_log_mass``.

    A MASKED key's entry is irrelevant (the caller zeroes it under the key-padding addend)."""
    B = hs.slot_log_pi.shape[0]
    dt = hs.slot_log_pi.dtype
    out = torch.zeros(B, n_tokens, dtype=dt, device=hs.slot_log_pi.device)
    out[:, TEAM_SIZE:2 * TEAM_SIZE] = hs.slot_log_pi
    if e5_offset is not None:
        e5 = hs.slot_log_pi
        if e5_active is not None:
            assert opp_active_local is not None
            onehot = torch.nn.functional.one_hot(opp_active_local, TEAM_SIZE).bool()
            e5 = torch.where(onehot, e5_active.to(dt).unsqueeze(-1), e5)
        out[:, e5_offset:e5_offset + TEAM_SIZE] = e5
    if e4 is not None:
        assert e4_offset is not None
        out[:, e4_offset:e4_offset + e4.shape[1]] = e4.to(dt)
    out[:, other_index] = hs.other_log_mass
    return out
