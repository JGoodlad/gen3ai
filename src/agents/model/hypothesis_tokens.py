"""X5's hypothesis TOKENS in the phase chain (`gen3_x5_belief_tokens_v1`, build unit U3;
`designs/endstate/design_x5_belief_tokens.md` §3.4 / §3.5). X5 only — the
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
  token + the learned `hypothesis_marker`; every other slot keeps the real pass's token
  (`torch.where`). AS BUILT (`gen3_x5_hyp_gather_v1`) the encoding is computed by
  `hypothesis_encode.gathered_hypothesis_tokens`: the encoder's species half once over the dex table,
  gathered, plus its row-level half (the clock / weather / fainted / hazards / screens context and the
  active-context scatter) — the per-row pass, split exactly at the two first Linears.
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
from agents.model.index_max import max_by_index
from agents.observation.constants import TEAM_SIZE


class OppPresence(NamedTuple):
    """What a class-E pool over opponent tokens needs (X5 only; None with the belief family off)."""
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


def splice_hypothesis_tokens(role_tokens: torch.Tensor, opp_hyp: torch.Tensor,
                             hs: HypothesisSet, marker: torch.Tensor) -> torch.Tensor:
    """[B,12,D]: the hidden opponent slots take the hypothesis pass's token ``opp_hyp`` [B,6,D] (the
    OPPONENT slots only — `hypothesis_encode.gathered_hypothesis_tokens`) + ``marker``; every other row
    keeps the real pass's token."""
    opp = role_tokens[:, TEAM_SIZE:2 * TEAM_SIZE]
    hyp = opp_hyp + marker
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


# ============================================================================ the move axis (U3 part 2)

#: The 16 typed Hidden Power nums (355 + t), the channels `compose_typed_hp` writes.
_TYPED_HP = tuple(range(355, 371))


@dataclasses.dataclass(frozen=True)
class FixedMassMoves:
    """The opponent ACTIVE's move axis under fixed_mass (§3.1 / §3.2 "Moves"): ONE ordering for every
    consumer — the E4 seats, the op's top-K (its pair cells, α's seats), the D3 / S3 edge cells and
    the intent operands — and the fixed-mass presence as the op's class-M candidate weights. Every
    tensor is DETACHED (M10).

    **The revealed Hidden Power seat** (U2 hand-off): a revealed HP occupies ONE seat as the typeless
    num 237 (its 16 typed channels are its type distribution, not further candidates). A num-keyed
    kernel cannot price 237 (BP 0, no type), so each seat-axis consumer runs on an EXTENDED axis
    ``idx_ext`` = the K seats ⊕ the 16 typed nums and contracts it with ``mix`` [B,K,K+16] — the
    identity for every other seat, and for the HP seat the typed weights ``P(t)`` (the composed
    posterior's typed channels renormalised: `compose_typed_hp` pins their sum to ≈ 1 on a reveal).
    The HP seat's physics is therefore E_t[f(HP_t)], exactly as the blob's typed candidates price it.
    """
    w_all: torch.Tensor        # [B,M] candidate weights: π_m unrevealed, 1 revealed (237 → its P(t) on 355..370), 0 off
    seat_nums: torch.Tensor    # [B,K] long — the seats in the one order (237 for a revealed HP; 0 off)
    seat_w: torch.Tensor       # [B,K] seat presence: 1 revealed, π_m unrevealed, 0 off
    seat_logp: torch.Tensor    # [B,K] its log (0 revealed; logsigmoid unrevealed; 0 off — masked by seat_on)
    seat_on: torch.Tensor      # [B,K] bool — a live seat that carries mass (structural)
    idx_ext: torch.Tensor      # [B,K+16] long — the seats ⊕ the 16 typed HP nums
    w_ext: torch.Tensor        # [B,K+16] — seat_w ⊕ ones (a revealed HP's type is uncertain, its presence 1)
    mix: torch.Tensor          # [B,K,K+16] — the extended axis → the K seats
    other_mass: torch.Tensor   # [B] OTHER_move: Σ π_m beyond the seats
    other_log_mass: torch.Tensor  # [B]
    other_live: torch.Tensor   # [B] bool
    beyond: torch.Tensor       # [B,M] bool — OTHER_move's members
    # X5 U4 (§3.7, the flat pointer): OTHER_move's PRICING weights — the renormalised tail
    # u_m = π_m·[m ∈ beyond] / Σ_beyond π (0 when OTHER_move is masked). Every per-candidate quantity
    # computed on the op's FULL candidate axis contracts with it into OTHER_move's seat-axis column:
    # E_tail[f(m)], the move-side twin of OTHER_species' `other_tail_probs` (M3 (c)).
    other_u: Optional[torch.Tensor] = None   # [B,M]

    def mix_seats(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Contract ``x``'s extended seat axis (size K+16 at ``dim``) onto the K seats."""
        xm = x.movedim(dim, -1)                                          # [B, ..., K+16]
        B = xm.shape[0]
        flat = xm.reshape(B, -1, xm.shape[-1])                           # [B, R, K+16]
        out = torch.einsum("bre,bke->brk", flat, self.mix.to(flat.dtype))
        return out.reshape(*xm.shape[:-1], self.mix.shape[1]).movedim(-1, dim)


def fixed_mass_moves(moves: Any, active_typed_logits: torch.Tensor) -> FixedMassMoves:
    """Build `FixedMassMoves` from U2's `MovePresence` and the active's COMPOSED move posterior
    ``active_typed_logits`` [B,M] (the typed-HP channels read for a revealed HP's type weights)."""
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM as HP
    pres = moves.presence
    pi = pres.pi                                                          # [B,M] detached, 0 off V
    B, M = pi.shape
    K = moves.seat_nums.shape[1]
    dev, dt = pi.device, pi.dtype
    rev_nums = torch.where(moves.seat_revealed, moves.seat_nums, torch.zeros_like(moves.seat_nums))
    revealed = torch.zeros(B, M, dtype=torch.bool, device=dev)
    revealed.scatter_(1, rev_nums, moves.seat_revealed)                  # index 0 only ever gets False
    revealed[:, 0] = False
    hp_seat = moves.seat_revealed & (moves.seat_nums == HP)              # [B,K]
    hp_rev = hp_seat.any(-1)                                             # [B]
    typed = torch.sigmoid(active_typed_logits.detach()[:, _TYPED_HP[0]:_TYPED_HP[-1] + 1].to(dt))
    p_t = typed / typed.sum(-1, keepdim=True).clamp(min=torch.finfo(dt).tiny)   # [B,16]
    p_t = torch.where(hp_rev.unsqueeze(-1), p_t, torch.zeros_like(p_t))
    one = torch.ones((), dtype=dt, device=dev)
    w_all = torch.where(revealed, one, pi)
    w_all = torch.cat([w_all[:, :_TYPED_HP[0]],
                       torch.where(hp_rev.unsqueeze(-1), p_t, w_all[:, _TYPED_HP[0]:_TYPED_HP[-1] + 1]),
                       w_all[:, _TYPED_HP[-1] + 1:]], dim=1)
    w_all = torch.cat([w_all[:, :HP], torch.zeros_like(w_all[:, HP:HP + 1]), w_all[:, HP + 1:]], dim=1)
    seat_w = moves.seat_pi
    k_live = (pres.k > 0).unsqueeze(-1)
    seat_on = moves.seat_live & (moves.seat_revealed | k_live)
    seat_logp = torch.where(moves.seat_revealed | ~seat_on, torch.zeros_like(seat_w),
                            pres.log_pi.gather(-1, moves.seat_nums))
    typed_nums = torch.arange(_TYPED_HP[0], _TYPED_HP[-1] + 1, device=dev).expand(B, -1)
    idx_ext = torch.cat([moves.seat_nums, typed_nums], dim=1)
    w_ext = torch.cat([seat_w, torch.ones(B, len(_TYPED_HP), dtype=dt, device=dev)], dim=1)
    eye = torch.eye(K, K + len(_TYPED_HP), dtype=dt, device=dev).expand(B, -1, -1)
    hp_row = torch.cat([torch.zeros(B, K, dtype=dt, device=dev), p_t], dim=1)    # [B,K+16]
    mix = torch.where(hp_seat.unsqueeze(-1), hp_row.unsqueeze(1), eye)
    # OTHER_move's pricing weights (U4): the tail's own presences, renormalised. A live OTHER_move has
    # Σ_beyond π > 0 (π never reaches 0 on a live candidate), so the division is exact; a masked one
    # reads 0 — a structural gate (`other_live`), never a float comparison.
    tw = torch.where(moves.beyond, w_all, torch.zeros_like(w_all))
    tiny = torch.finfo(dt).tiny
    other_u = torch.where(moves.other_live.unsqueeze(-1),
                          tw / tw.sum(-1, keepdim=True).clamp(min=tiny), torch.zeros_like(tw))
    return FixedMassMoves(w_all=w_all, seat_nums=moves.seat_nums, seat_w=seat_w, seat_logp=seat_logp,
                          seat_on=seat_on, idx_ext=idx_ext, w_ext=w_ext, mix=mix,
                          other_mass=moves.other_mass, other_log_mass=moves.other_log_mass,
                          other_live=moves.other_live, beyond=moves.beyond, other_u=other_u)


def other_move_cells(fm: FixedMassMoves, move_bp: torch.Tensor, move_acc: torch.Tensor,
                     move_phys: torch.Tensor) -> torch.Tensor:
    """[B,4] the opponent ACTIVE's E5 tail seat as OTHER_move (§3.1): ``[p_tail, worst_phys,
    worst_spec, revealed]`` with ``p_tail`` = OTHER_move's mass Σ_beyond π_m (no clamp — an expected
    count) and the worst-case features a PRESENCE-SCALED max over OTHER_move's members (class M,
    §9 M2 = C): max_m π_m · BP/150 · acc on each category channel. ``revealed`` = 1 (the active is)."""
    score = torch.where(fm.beyond, fm.w_all, torch.zeros_like(fm.w_all)) \
        * (move_bp / 150.0) * move_acc                                            # [B,M]
    worst_phys = max_by_index(score * move_phys)
    worst_spec = max_by_index(score * (1.0 - move_phys))
    return torch.stack([fm.other_mass.to(score.dtype), worst_phys, worst_spec,
                        torch.ones_like(worst_phys)], dim=-1)


# ============================================================================ the opponent-MON axis (U3 part 3)

@dataclasses.dataclass(frozen=True)
class OpRoster:
    """The damage op's opponent-MON axis under fixed_mass (§3.4 "Physics", §3.5; U3 part 3). Every
    tensor is DETACHED (M10: π weights no policy / critic route).

    A hidden opponent slot holds a CONCRETE hypothesis, so the op prices it like a mon of that species
    at its first appearance — pristine, full HP, no status, nothing revealed — by reading the HYPOTHESIS
    CONTEXT (`hypothesis_ctx`: species, types and the ability prior from U1's dex row). Three things
    change against the blob's gates, all structural:

    * **"alive" is `ctx.opp_addressable`** (alive-and-revealed OR hidden — exact: a mon cannot faint
      unrevealed), never the HP cell (F-X5-12). A hypothesis row reads HP 1.0, but no gate may depend
      on that.
    * **"species known"** (the gates that zeroed a slot because its types were unknown) holds on every
      slot: each is revealed or holds a hypothesis (`concrete`).
    * **the attacker's candidate moves** are its OWN fixed-mass move presence (`move_w`, §3.2: "a
      hypothesis seat's own move posterior … uses the same construction at k = 4"; a revealed mon at
      k = 4 − r, its revealed moves pinned at 1), selected by ONE per-mon order (`move_order`: revealed
      first, then by presence, ties to the lower num — no `torch.topk`, F-X5-13). The ACTIVE's row IS
      the move group's `FixedMassMoves.w_all` — the weights the op's incoming max already reads.

    Presence enters ONLY where a reduction runs over the mon axis: a per-(seat, mon) edge cell is
    "what this mon does IF present" and its presence is the trunk's log-π key bias (§3.5) — scaling the
    cell by π too would count it twice. The one class-M site over mons, `p_pur_vs_us`, weights each
    slot by `slot_pi` (§9 M2 = C). Beat Up's party sum (class E, linear) reads the hidden-team
    marginal `team_probs` = π / k over EVERY candidate (hypotheses + tail), so its expectation is
    exact.
    """
    alive: torch.Tensor          # [B,6] float — ctx.opp_addressable
    concrete: torch.Tensor       # [B,6] float — revealed, or holding a hypothesis (species known under it)
    hyp: torch.Tensor            # [B,6] bool — the slot holds a hypothesis
    slot_pi: torch.Tensor        # [B,6] presence at a max site over mons: 1 revealed, π hypothesis
    species_probs: torch.Tensor  # [B,6,S] one-hot(hypothesis species) on a hypothesis slot; 0 elsewhere
    team_probs: torch.Tensor     # [B,S] π / k: one hidden slot's species marginal (0 when k = 0)
    move_w: torch.Tensor         # [B,6,M] per-mon move presence (1 revealed; π_m; a revealed HP → P(t) on 355..370)
    move_order: torch.Tensor     # [B,6,M] long — per-mon ONE order over the selectable moves (rest last)
    move_rank: torch.Tensor      # [B,6,M] long — its inverse
    move_tie_gap: torch.Tensor   # [B] smallest gap at a per-mon selection boundary (alive mons; +inf: none)
    # --- OTHER_species (M3 (c), ORCHESTRATOR F4 (a) / (b)); None on an OTHER-mode roster itself
    other_live: Optional[torch.Tensor] = None     # [B] bool — OTHER is a real key (structural)
    other_any: Optional[torch.Tensor] = None      # [B] P(≥ 1 tail species present) = 1 − Π(1 − π): its max-site presence
    other_col: Optional[torch.Tensor] = None      # [B] long — a hidden slot: where an OTHER-mode pass's column is read
    other_pursuit: Optional[torch.Tensor] = None  # [B] OTHER's Pursuit presence (its averaged move presence)
    other: Optional["OpRoster"] = None            # the OTHER-MODE roster (`other_roster`)
    # --- set ONLY on an OTHER-mode roster: every hidden slot holds the tail-AVERAGED mon
    override: Optional[torch.Tensor] = None       # [B,6] bool — the slots holding OTHER in this pass
    att_base: Optional[torch.Tensor] = None       # [B,6,6] E_tail[base stats] (exact: the stat formulas are linear)
    has_type: Optional[torch.Tensor] = None       # [B,6,T] E_tail[1(type t is one of the species' types)]
    spe: Optional[torch.Tensor] = None            # [B,6] E_tail[speed] — the spread prior's mean (as the averaged bulk)
    spe_std: Optional[torch.Tensor] = None        # [B,6] E_tail[the spread prior's speed std]


def slot_move_presence(hb: Any, move_logits: torch.Tensor, species: torch.Tensor,
                       revealed_ids: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per opponent mon, its FIXED-MASS move presence (§3.2 "Moves", applied to every slot):
    ``(w [B,6,M], rev_move [B,6,M] bool, sel [B,6,M] bool)``.

    The construction is `HypothesisBuilder.move_group`'s, per slot: candidates = the slot species'
    legal moves minus its revealed ones (`move_candidates`), ``k_m = 4 − r``, π = the logistic
    fixed-size marginals of the composed posterior ``move_logits`` [B,6,M] (read DETACHED). ``w`` is
    `fixed_mass_moves`'s ``w_all`` rule per slot: 1 on a revealed move, π on an unrevealed candidate, a
    revealed Hidden Power as its typed weights ``P(t)`` on 355..370, the typeless 237 at 0. ``key`` is
    the order key (2 on a revealed non-HP move — structural, never a float compare with π — else
    ``w``); ``sel`` the selectable moves (the candidates of a LIVE group, the revealed moves but 237, and
    a revealed HP's typed channels; a mon whose four moves are all revealed has ``k_m = 0`` and no
    candidate — structural, so a π ≡ 0 row never reads as a tie). The key is `slot_order_key`."""
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM as HP
    from agents.model.hypothesis_set import MOVE_GROUP_MASS, fixed_mass_presence, move_candidates
    B, T, M = move_logits.shape
    S = hb.n_species
    sp = species.clamp(0, S - 1).long()                                              # [B,6]
    legal = hb.move_legal[sp].reshape(B * T, M)
    cand, revealed, r = move_candidates(legal, hb.move_valid, revealed_ids.reshape(B * T, -1))
    cand, revealed, r = cand.reshape(B, T, M), revealed.reshape(B, T, M), r.reshape(B, T)
    k_m = (MOVE_GROUP_MASS - r).clamp(min=0)
    pres = fixed_mass_presence(move_logits.detach(), cand, k_m)                      # [B,6,M]
    pi = pres.pi
    dt = pi.dtype
    hp_rev = revealed[..., HP]                                                       # [B,6]
    typed = torch.sigmoid(move_logits.detach()[..., _TYPED_HP[0]:_TYPED_HP[-1] + 1].to(dt))
    p_t = typed / typed.sum(-1, keepdim=True).clamp(min=torch.finfo(dt).tiny)       # [B,6,16]
    one = torch.ones((), dtype=dt, device=pi.device)
    w = torch.where(revealed, one, pi)
    w = torch.cat([w[..., :_TYPED_HP[0]],
                   torch.where(hp_rev.unsqueeze(-1), p_t, w[..., _TYPED_HP[0]:_TYPED_HP[-1] + 1]),
                   w[..., _TYPED_HP[-1] + 1:]], dim=-1)
    w = torch.cat([w[..., :HP], torch.zeros_like(w[..., HP:HP + 1]), w[..., HP + 1:]], dim=-1)
    num = torch.arange(M, device=pi.device)
    is_typed = (num >= _TYPED_HP[0]) & (num <= _TYPED_HP[-1])                        # [M]
    rev_move = revealed & (num != HP)                                                # [B,6,M]
    sel = (cand & pres.live.unsqueeze(-1)) | rev_move | (is_typed & hp_rev.unsqueeze(-1))
    return w, rev_move, sel


def slot_order_key(w: torch.Tensor, rev_move: torch.Tensor) -> torch.Tensor:
    """The per-mon order key: 2 on a revealed (non-HP) move — STRUCTURAL, pinned ahead of every
    presence, never a float compare with π — else the presence ``w`` (move_group's key rule)."""
    return torch.where(rev_move, torch.full_like(w, 2.0), w)


def build_op_roster(hb: Any, ctx: ExtractorContext, hctx: ExtractorContext, hs: HypothesisSet,
                    fm: Optional[FixedMassMoves], move_logits: torch.Tensor,
                    cuts: "tuple[int, ...]") -> "tuple[OpRoster, HypothesisSet]":
    """The op's opponent-mon roster (`OpRoster`) for one forward, and ``hs`` with the per-mon
    selection gap attached (`near_tie_rows` reads it). ``move_logits`` [B,6,M] is the composed move
    posterior (`last_move_belief_logits`); ``cuts`` the per-mon selection sizes the op and the E5 seats
    consume (`consequence_topk`, `entity_topk_seats`) — every one is a rule-8 boundary."""
    from agents.model.hypothesis_set import SetCuts, boundary_gap, ranks_of, stable_order
    B = ctx.batch_size
    opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
    believed = ctx.opp_believed_mask.bool()
    hyp = hs.slot_is_hypothesis
    dt = hs.species.pi.dtype
    alive = ctx.opp_addressable.to(dt)
    concrete = ((~believed) | hyp).to(dt)
    assert hs.slot_pi is not None
    S = hs.species.pi.shape[-1]
    species_probs = torch.nn.functional.one_hot(hs.slot_species.clamp(0, S - 1), S).to(dt) \
        * hyp.unsqueeze(-1).to(dt)                                                   # [B,6,S]
    k = hs.species.k
    team_probs = torch.where((k > 0).unsqueeze(-1),
                             hs.species.pi / k.clamp(min=1).unsqueeze(-1).to(dt),
                             torch.zeros_like(hs.species.pi))
    w, rev_move, sel = slot_move_presence(hb, move_logits, hctx.species_ids[:, opp], ctx.all_move_ids[:, opp, :])
    if fm is not None:      # the ACTIVE's row IS the move group's weights (the op's incoming max reads them)
        act = torch.nn.functional.one_hot(ctx.opp_active_local, TEAM_SIZE).bool().unsqueeze(-1)  # [B,6,1]
        w = torch.where(act, fm.w_all.to(w.dtype).unsqueeze(1), w)
    key = slot_order_key(w, rev_move)
    # The op reads each mon's first K candidates as a SET (gathers per candidate, then presence-scaled
    # max / sums over K — blob's `topk` feeds the same code) and the tail beyond K as a set: only the
    # pair straddling each cut is a boundary (gen3_behaviour_tie_consumed_v1; `boundary_gap` below).
    order = stable_order(key, sel, consumed=SetCuts(cuts))                           # [B,6,M]
    rank = ranks_of(order)
    sorted_key = key.gather(-1, order)
    n_sel = sel.sum(-1)                                                              # [B,6] structural count
    gap = torch.full((B, TEAM_SIZE), float("inf"), dtype=dt, device=w.device)
    for K in sorted(set(int(c) for c in cuts)):
        gap = torch.minimum(gap, boundary_gap(sorted_key, torch.full_like(n_sel, K), n_sel))
    gap = torch.where(ctx.opp_addressable, gap, torch.full_like(gap, float("inf"))).amin(-1)
    roster = OpRoster(alive=alive, concrete=concrete, hyp=hyp, slot_pi=hs.slot_pi.to(dt),
                      species_probs=species_probs, team_probs=team_probs, move_w=w,
                      move_order=order, move_rank=rank, move_tie_gap=gap)
    return roster, dataclasses.replace(hs, slot_moves_tie_gap=gap)


def bench_tail_cells(ro: OpRoster, K: int, move_bp: torch.Tensor, move_acc: torch.Tensor,
                     move_phys: torch.Tensor) -> torch.Tensor:
    """[B,6,4] every opponent mon's E5 tail seat under fixed_mass (§3.1: "the bench mons' E5 seats keep
    their features, from their own slot's move posterior"), made presence-aware: the tail is the mon's
    moves BEYOND rank ``K`` of its one order (the E4 cut), ``p_tail`` their summed presence (an expected
    count, unclamped — OTHER_move's rule), ``worst_*`` a presence-scaled max over them (class M, §9 M2 =
    C), ``revealed`` the slot's revealed bit is filled by the caller. (The active's row is OTHER_move,
    `other_move_cells`.)"""
    tail = ro.move_rank >= K                                                        # [B,6,M]
    tw = torch.where(tail, ro.move_w, torch.zeros_like(ro.move_w))
    score = tw * (move_bp / 150.0) * move_acc
    worst_phys = max_by_index(score * move_phys)
    worst_spec = max_by_index(score * (1.0 - move_phys))
    return torch.stack([tw.sum(-1), worst_phys, worst_spec], dim=-1)               # [B,6,3]


def other_roster(ro: OpRoster, hs: HypothesisSet, hb: Any, move_belief: Any, base_stats: torch.Tensor,
                 species_type: torch.Tensor, spread_prior: torch.Tensor, n_types: int,
                 spe_col: int, cuts: "Optional[tuple[int, ...]]" = None) -> OpRoster:
    """``ro`` with OTHER attached (M3 (c); ORCHESTRATOR F4 (a) / (b)) and its OTHER-MODE roster: a copy in
    which EVERY hidden slot holds OTHER — the renormalised tail ``P_tail`` (`HypothesisSet.other_tail_probs`)
    priced by the blob's own AVERAGED construction, ``P_tail @ tables``:

    * defender: ``species_probs`` = ``P_tail`` — `_outgoing_matrix`'s expected-latent read (E[def], E[spd],
      E[maxhp], E[type x ability multiplier]); ``concrete`` = 0 there, so P(KO) stays NULLED (the averaged
      defender's KO is a threshold of averaged stats — exactly the blob);
    * attacker: ``att_base`` = E[base stats] (atk / spa / max-HP are linear in them, so f(E[base]) = E[f]),
      ``has_type`` = E[STAB indicator] per move type, ``spe`` / ``spe_std`` = E over the spread prior;
    * moves: the PARAMETER-FREE E10 mixture over ``P_tail`` (`MoveBelief.hidden_slot_prior_logits`, the
      blob's hidden-slot prior) through the same k = 4 fixed-size construction and one order as every mon.

    OTHER is never immune by construction: its multiplier is an expectation over the tail, 0 only if every
    tail species is immune. At a max-type site OTHER enters with ``other_any`` (F4 (b)), never its mass."""
    from agents.model.hypothesis_set import SetCuts, fixed_mass_presence, move_candidates, ranks_of, stable_order
    pt = hs.other_tail_probs.to(base_stats.dtype)                                       # [B,S]
    B = pt.shape[0]
    hyp = ro.hyp
    dt = ro.move_w.dtype
    # ---- OTHER's averaged move presence (k = 4; nothing revealed; 237 is never a candidate)
    mix_logits = move_belief.hidden_slot_prior_logits(pt)                               # [B,M]
    M = mix_logits.shape[-1]
    legal = hb.move_valid.unsqueeze(0).expand(B, M)
    cand, _rev, _r = move_candidates(legal, hb.move_valid, torch.zeros(B, 4, dtype=torch.long, device=pt.device))
    # A DEAD OTHER (no hidden slot / an empty tail: ``P_tail`` = 0, so the mixture is UNIFORM) has no move
    # candidates. Its order is read only through ``where(hyp, …)`` (no hypothesis on such a row) and its
    # Pursuit presence only times ``other_any`` = 0 there, so every output is byte-identical; masking it
    # keeps the declared sort_head site from counting the uniform mixture's EXACT ties as rule-8 near-ties
    # (X5 U6: 29 of the K9 golden buffer's 64 rows — a 47 % K9(b) excluded share vs its 0.15 ceiling).
    cand = cand & hs.other_live.unsqueeze(-1)
    k4 = torch.full((B,), 4, dtype=torch.long, device=pt.device)
    pres = fixed_mass_presence(mix_logits.detach(), cand, k4)
    w_o = pres.pi.to(dt)                                                                # [B,M]
    # OTHER's order replaces each hypothesis slot's in the roster, so the op reads it exactly as
    # `build_op_roster`'s: a SET at each cut (``cuts``; None = every pair of the head, conservative)
    order_o = stable_order(w_o, cand, consumed=None if cuts is None else SetCuts(cuts))
    rank_o = ranks_of(order_o)
    h3 = hyp.unsqueeze(-1)
    move_w = torch.where(h3, w_o.unsqueeze(1), ro.move_w)
    move_order = torch.where(h3, order_o.unsqueeze(1), ro.move_order)
    move_rank = torch.where(h3, rank_o.unsqueeze(1), ro.move_rank)
    # ---- the tail's expected attacker / speed tables
    t = torch.arange(n_types, device=pt.device)
    st = species_type.long()
    has = ((t.unsqueeze(0) == st[:, 0:1]) | (t.unsqueeze(0) == st[:, 1:2])).to(pt.dtype)   # [S,T]
    e_base = pt @ base_stats                                                            # [B,6]
    e_has = pt @ has                                                                    # [B,T]
    e_spe = pt @ spread_prior[:, spe_col, 0]                                            # [B]
    e_std = pt @ spread_prior[:, spe_col, 1]
    sp_other = pt.unsqueeze(1).expand(-1, hyp.shape[1], -1) * hyp.unsqueeze(-1).to(pt.dtype)
    from agents.model.damage_tables import _pursuit_num
    other = dataclasses.replace(
        ro, concrete=(~hyp).to(ro.concrete.dtype), species_probs=sp_other.to(ro.species_probs.dtype),
        slot_pi=torch.where(hyp, hs.other_any.unsqueeze(-1).to(ro.slot_pi.dtype), ro.slot_pi),
        move_w=move_w, move_order=move_order, move_rank=move_rank, override=hyp,
        att_base=e_base.unsqueeze(1).expand(-1, hyp.shape[1], -1),
        has_type=e_has.unsqueeze(1).expand(-1, hyp.shape[1], -1),
        spe=e_spe.unsqueeze(-1).expand(-1, hyp.shape[1]), spe_std=e_std.unsqueeze(-1).expand(-1, hyp.shape[1]))
    col = torch.argmax(hyp.long(), dim=-1)                                              # the first hidden slot
    return dataclasses.replace(ro, other_live=hs.other_live, other_any=hs.other_any.to(dt), other_col=col,
                               other_pursuit=w_o[:, _pursuit_num()], other=other)


def other_column(cells: torch.Tensor, col: torch.Tensor, axis: int) -> torch.Tensor:
    """OTHER's cells out of an OTHER-mode pass: the opponent-mon axis ``axis`` of ``cells`` read at the
    hidden slot ``col`` [B] (every hidden slot holds OTHER there), kept as a size-1 axis."""
    shape = [1] * cells.dim()
    shape[0] = cells.shape[0]
    idx = col.view(shape).expand(*[cells.shape[i] if i != axis else 1 for i in range(cells.dim())])
    return cells.gather(axis, idx)
