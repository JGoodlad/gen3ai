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
    return FixedMassMoves(w_all=w_all, seat_nums=moves.seat_nums, seat_w=seat_w, seat_logp=seat_logp,
                          seat_on=seat_on, idx_ext=idx_ext, w_ext=w_ext, mix=mix,
                          other_mass=moves.other_mass, other_log_mass=moves.other_log_mass,
                          other_live=moves.other_live, beyond=moves.beyond)


def other_move_cells(fm: FixedMassMoves, move_bp: torch.Tensor, move_acc: torch.Tensor,
                     move_phys: torch.Tensor) -> torch.Tensor:
    """[B,4] the opponent ACTIVE's E5 tail seat as OTHER_move (§3.1): ``[p_tail, worst_phys,
    worst_spec, revealed]`` with ``p_tail`` = OTHER_move's mass Σ_beyond π_m (no clamp — an expected
    count) and the worst-case features a PRESENCE-SCALED max over OTHER_move's members (class M,
    §9 M2 = C): max_m π_m · BP/150 · acc on each category channel. ``revealed`` = 1 (the active is)."""
    score = torch.where(fm.beyond, fm.w_all, torch.zeros_like(fm.w_all)) \
        * (move_bp / 150.0) * move_acc                                            # [B,M]
    worst_phys = (score * move_phys).amax(-1)
    worst_spec = (score * (1.0 - move_phys)).amax(-1)
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
    from agents.model.hypothesis_set import boundary_gap, ranks_of, stable_order
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
    order = stable_order(key, sel)                                                   # [B,6,M]
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
    worst_phys = (score * move_phys).amax(-1)
    worst_spec = (score * (1.0 - move_phys)).amax(-1)
    return torch.stack([tw.sum(-1), worst_phys, worst_spec], dim=-1)               # [B,6,3]
