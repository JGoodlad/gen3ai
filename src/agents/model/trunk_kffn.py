"""PER-TOKEN-KIND FFN EXPERTS — `--trunk-ffn-by-kind {off,on}` (B-KFFN, `gen3_trunk_ffn_by_kind_v1`, config v161;
`designs/endstate/design_growth_builds_kffn_int.md` §1, the lever of X48).

WHAT. With ``on``, every trunk round applies, at its FFN position, ``ffn_k(h[:, segment])`` for each SEGMENT of its
tokens, where ``k`` is the segment's KIND, and concatenates the outputs back in position order. Nothing else in the
round changes (the attention, the norms, the residuals, the shared edge bias, the pre- / post-LN form). Five kinds:

    mon     the round's OWN ``linear1`` / ``linear2`` (state-dict keys unchanged): our 6 and their 6 team tokens, and
            OTHER_species (a Pokémon, typed THEIR_TEAM)
    board   ``kind_ffn.board.{linear1,linear2}``: OUR SIDE, THEIR SIDE, FIELD
    move    ``kind_ffn.move.*``: E3, our active's 4 move seats
    threat  ``kind_ffn.threat.*``: E4's K believed-move seats, then the E5 tail's OTHER_move seats
    event   ``kind_ffn.event.*``: the event seats

ROUTING BY SEAT ROLE, computed at BUILD time (:func:`ffn_kind_segments`, from the built seat modules; no literal), never
by token-type id. The table is a tuple of plain-int ``(start, end, kind_index)`` runs — six on the production layout:
OTHER_species sits AFTER the entity seats and BEFORE the event seats, so the Pokémon kind is two runs (F-GB-4). Every
round holds the same table (``round.ffn_segments``) and asserts at each forward that its input's token count equals the
table's last ``end`` (a layout change that forgets the table raises instead of mis-routing; a static-shape int compare,
free under compile).

WHERE THE EXPERTS LIVE. Inside the round classes (`TrunkRound`, the base of `team_transformer.BiasedEncoderLayer` and
`trunk_depth.IdentityInitRound`), as ``round.kind_ffn`` (an ``nn.ModuleDict`` over `EXPERT_KINDS`, or ``None`` when off),
and both round forwards call ONE method, ``round.ffn(h)``. That is what makes the bf16 trunk region cover the experts
with no change to `trunk_precision` (its class swap autocasts the whole round forward); experts built as SIBLING modules
called from `TeamTransformer.forward` would silently run fp32 outside the region. OFF builds nothing: ``kind_ffn`` and
``ffn_segments`` are ``None`` and ``ffn(h)`` is the literal shared expression, so the state dict, its bytes and the forward
are today's.

INIT. A FRESH ``on`` build constructs each round's experts as `IsolatedLinear`s (SB3's orthogonal re-init skips them)
inside ``torch.random.fork_rng`` from ``KFFN_INIT_SEED + round index`` (`build_kind_ffn`), so no other initial byte moves:
for a post-LN `BiasedEncoderLayer` the init SB3 gives the shared Linears (orthogonal, gain √2, bias 0); for a pre-LN
`IdentityInitRound` ``linear1`` likewise and ``linear2`` ZERO (weight and bias), so a fresh identity round is still the
identity. A GROWTH fork (``--grow trunk_ffn_by_kind=on``, `agents.model.growth`) builds the same modules and COPIES the
round's ``linear1`` / ``linear2`` into every expert (the parent's function, bit for bit at the gate's 64-row batch).
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch

from agents.model.arch_constants import D_MODEL, KFFN_INIT_SEED, TRANSFORMER_FFN_DIM

#: The flag's values: ``off`` (production, nothing built) and ``on``.
KFFN_MODES: Tuple[str, ...] = ("off", "on")
#: The five token kinds, by index (the segment table's third int). ``mon`` is the round's own FFN.
KINDS: Tuple[str, ...] = ("mon", "board", "move", "threat", "event")
KIND_MON, KIND_BOARD, KIND_MOVE, KIND_THREAT, KIND_EVENT = range(len(KINDS))
#: The kinds that get an expert of their own (``kind_ffn``'s keys, in this order).
EXPERT_KINDS: Tuple[str, ...] = KINDS[1:]

Segments = Tuple[Tuple[int, int, int], ...]


def check_mode(mode: str) -> str:
    if mode not in KFFN_MODES:
        raise ValueError(f"trunk_ffn_by_kind must be one of {KFFN_MODES}, got {mode!r}")
    return mode


class KindExpert(torch.nn.Module):
    """One kind's FFN: ``linear2(relu(linear1(h)))``, 128 → 256 → 128, out of `IsolatedLinear`s."""

    def __init__(self, d_model: int = D_MODEL, ffn_dim: int = TRANSFORMER_FFN_DIM, *, zero_out: bool = False) -> None:
        super().__init__()
        from agents.model.hypothesis_set import IsolatedLinear   # local: hypothesis_set is a heavy import
        self.linear1 = IsolatedLinear(d_model, ffn_dim)
        self.linear2 = IsolatedLinear(ffn_dim, d_model, zero=zero_out)
        lins = (self.linear1,) if zero_out else (self.linear1, self.linear2)
        for lin in lins:
            torch.nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
            assert lin.bias is not None
            torch.nn.init.zeros_(lin.bias)


def build_kind_ffn(round_index: int, *, identity: bool) -> torch.nn.ModuleDict:
    """Round ``round_index``'s experts (`EXPERT_KINDS` order), built from the PRIVATE seed ``KFFN_INIT_SEED +
    round_index`` (no global RNG draw). ``identity`` (a pre-LN `IdentityInitRound`): every expert's ``linear2`` zero."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(KFFN_INIT_SEED + int(round_index))
        return torch.nn.ModuleDict({k: KindExpert(zero_out=identity) for k in EXPERT_KINDS})


class TrunkRound(torch.nn.Module):
    """The base of every trunk round class: the FFN position, ``ffn(h)`` (module docstring)."""

    linear1: torch.nn.Module
    linear2: torch.nn.Module

    def __init__(self) -> None:
        super().__init__()
        self.kind_ffn: Optional[torch.nn.ModuleDict] = None
        self.ffn_segments: Optional[Segments] = None

    def ffn(self, h: torch.Tensor) -> torch.Tensor:
        """``h`` [B, n, d_model] → the FFN's output [B, n, d_model]: the shared FFN (off), or each segment through
        its kind's expert (on)."""
        if self.kind_ffn is None:
            out: torch.Tensor = self.linear2(torch.nn.functional.relu(self.linear1(h)))
            return out
        segs = self.ffn_segments
        assert segs is not None, "a round with kind_ffn experts holds no ffn_segments table"
        if h.shape[1] != segs[-1][1]:
            raise RuntimeError(
                f"trunk_ffn_by_kind: the round reads {h.shape[1]} tokens but its segment table covers {segs[-1][1]} — "
                "the seat layout changed without `trunk_kffn.ffn_kind_segments`")
        parts: List[torch.Tensor] = []
        for start, end, kind in segs:
            mod: Any = self if kind == KIND_MON else self.kind_ffn[KINDS[kind]]
            parts.append(mod.linear2(torch.nn.functional.relu(mod.linear1(h[:, start:end]))))
        return torch.cat(parts, dim=1)


def trunk_rounds(tt: Any) -> List[TrunkRound]:
    """The trunk's rounds in execution order: the post-LN layers, then the extra identity-init rounds."""
    return list(tt.transformer_layers) + (list(tt.extra_rounds) if tt.extra_rounds is not None else [])


def ffn_kind_segments(fe: Any) -> Segments:
    """THE segment table of a BUILT extractor (module docstring), from its seat modules: the team tokens
    (`team_transformer._their_token_slice`), the board (``_total_tokens``), the entity seats (`entity_seats`: 4 E3 +
    K E4 + the E5 tail), OTHER_species (built with X5's hypothesis builder) and the event seats (`history_events.n`).
    A missing kind is a typed `ValueError` naming it."""
    tt = fe.team_transformer
    es = getattr(fe, "entity_seats", None)
    ev = getattr(fe, "history_events", None)
    n_team = int(tt._their_token_slice.stop)
    base = int(tt._total_tokens)
    n_threat = int(es.n_seats) - 4 if es is not None else 0
    missing = [k for k, ok in (
        ("board (--token-encoding static: the three board tokens)", bool(getattr(tt, "static_board", False))),
        ("threat (entity_topk_seats > 0 and entity_tail_seats: the E4 / E5 seats)",
         es is not None and int(es.topk_seats) > 0 and bool(es.tail_seats)),
        ("event (history_events: the event seats)", ev is not None),
        ("mon's OTHER_species seat (opp_intent: X5's hypothesis set)", getattr(fe, "hypothesis_builder", None) is not None),
    ) if not ok]
    if missing:
        raise ValueError("trunk_ffn_by_kind='on' needs all five token kinds; this network lacks "
                         + "; ".join(missing))
    assert es is not None and ev is not None
    segs: List[Tuple[int, int, int]] = [
        (0, n_team, KIND_MON), (n_team, base, KIND_BOARD), (base, base + 4, KIND_MOVE),
        (base + 4, base + 4 + n_threat, KIND_THREAT)]
    pos = base + 4 + n_threat
    segs.append((pos, pos + 1, KIND_MON))                                                   # OTHER_species
    pos += 1
    segs.append((pos, pos + int(ev.n), KIND_EVENT))
    return tuple(segs)


def install_segments(fe: Any, segments: Segments) -> None:
    """Set the table on every round that carries experts (every round of an ``on`` network)."""
    for r in trunk_rounds(fe.team_transformer):
        if r.kind_ffn is None:
            raise RuntimeError("trunk_ffn_by_kind='on' but a trunk round carries no experts")
        r.ffn_segments = segments


def copy_shared_into_experts(r: TrunkRound) -> List[Tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    """Copy round ``r``'s shared ``linear1`` / ``linear2`` (weight and bias) into each of its experts, in place; returns
    ``[(expert parameter, the shared parameter it copies)]`` in module order."""
    assert r.kind_ffn is not None
    pairs: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
    with torch.no_grad():
        for k in EXPERT_KINDS:
            e = r.kind_ffn[k]
            for lin in ("linear1", "linear2"):
                for attr in ("weight", "bias"):
                    dst = getattr(getattr(e, lin), attr)
                    src = getattr(getattr(r, lin), attr)
                    dst.copy_(src)
                    pairs.append((dst, src))
    return pairs
