"""TRUNK FFN WIDTH — `--trunk-ffn-dim N` (B-FFNW, `gen3_trunk_ffn_width_v1`, config v164; T69, the N4 lever,
`designs/endstate/design_n4_capacity_screen.md` §2).

WHAT. Every trunk round's feed-forward block is ``TRANSFORMER_FFN_DIM`` (256) hidden units wide. ``N > 256`` gives each
round a WIDTH BRANCH ``round.ffn_extra`` beside its own ``linear1`` / ``linear2``, and the round's FFN position
(`trunk_kffn.TrunkRound.ffn`, the one place every round runs its FFN) computes

    ffn(h) = linear2(relu(linear1(h))) + ffn_extra.linear2(relu(ffn_extra.linear1(h)))

``ffn_extra.linear1`` is ``D_MODEL -> N - 256`` with a bias, ``ffn_extra.linear2`` is ``N - 256 -> D_MODEL`` with NO bias,
so the sum is exactly ONE FFN of width ``N`` over a split hidden layer (the base's output bias is the only one). The
round's own keys are unchanged, so a parent's state dict loads into a wider child by name.

WHY A BRANCH, NOT A RESIZED ``linear1`` / ``linear2``. A growth fork (``--grow trunk_ffn_dim=N``, `agents.model.growth`)
must start as its parent's function BIT FOR BIT (ruling Q-K2: no tolerance). A zero ``ffn_extra.linear2`` adds an exact
0.0 to the base FFN's output; a longer GEMM reduction over a resized hidden layer is not guaranteed to reproduce the
shorter one's bits (the B-KFFN finding F-GB-5).

INIT. The branch is built from `IsolatedLinear`s (SB3's orthogonal re-init skips them) inside
``torch.random.fork_rng`` from ``TRUNK_FFNW_INIT_SEED + round index`` (:func:`build_ffn_extra`), so building it draws
nothing from the global stream (every other parameter's initial bytes equal the 256-wide build's). ``linear1`` gets the
init SB3 gives the existing rounds' ``linear1`` (orthogonal, gain √2, bias 0); ``linear2`` is ZERO. A FRESH build at
``N`` and a GROWTH to ``N`` construct the same modules from the same seeds, so a grown state dict equals a fresh child's
(keys, order and shapes). ``N = 256`` builds nothing: the state dict, its bytes and the forward are unchanged.

WHERE IT LIVES. Inside the round (``round.ffn_extra``), called from ``round.ffn``, so the bf16 trunk region's class swap
covers it (a sibling module would silently run fp32: `agents/model/CLAUDE.md`).

REFUSED (typed): together with ``--trunk-ffn-by-kind on`` (the per-kind experts would have to decide whether to copy
the branch; neither N4 arm needs the pair).
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import torch

from agents.model.arch_constants import D_MODEL, TRANSFORMER_FFN_DIM, TRUNK_FFNW_INIT_SEED

#: The narrowest legal width: the base FFN's (nothing built).
FFN_DIM_BASE = TRANSFORMER_FFN_DIM


def check_ffn_dim(value: Any) -> int:
    """``value`` as the trunk FFN width (an int >= `FFN_DIM_BASE`), or `ValueError`."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"trunk_ffn_dim must be an integer, got {value!r}") from None
    if isinstance(value, float) and float(value) != v:
        raise ValueError(f"trunk_ffn_dim must be an integer, got {value!r}")
    if v < FFN_DIM_BASE:
        raise ValueError(f"trunk_ffn_dim must be >= {FFN_DIM_BASE} (the base FFN's width; a narrower FFN is not "
                         f"offered), got {v}")
    return v


class FfnWidthBranch(torch.nn.Module):
    """One round's width branch (module docstring): ``linear1`` ``D_MODEL -> extra`` (orthogonal, gain √2, bias 0),
    ``linear2`` ``extra -> D_MODEL``, ZERO, no bias. Its forward is inlined in `TrunkRound.ffn` (no ``forward`` here)."""

    linear1: torch.nn.Module
    linear2: torch.nn.Module

    def __init__(self, extra: int, d_model: int = D_MODEL) -> None:
        super().__init__()
        from agents.model.hypothesis_set import IsolatedLinear   # local: hypothesis_set is a heavy import
        if int(extra) <= 0:
            raise ValueError(f"a width branch needs a positive width, got {extra}")
        self.linear1 = IsolatedLinear(d_model, int(extra))
        self.linear2 = IsolatedLinear(int(extra), d_model, zero=True, bias=False)
        torch.nn.init.orthogonal_(self.linear1.weight, gain=math.sqrt(2))
        assert self.linear1.bias is not None
        torch.nn.init.zeros_(self.linear1.bias)


def build_ffn_extra(round_index: int, ffn_dim: int) -> Optional[FfnWidthBranch]:
    """Round ``round_index``'s width branch at trunk FFN width ``ffn_dim``, built from the PRIVATE seed
    ``TRUNK_FFNW_INIT_SEED + round_index`` (no global RNG draw); None at the base width (nothing built)."""
    extra = check_ffn_dim(ffn_dim) - FFN_DIM_BASE
    if extra == 0:
        return None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TRUNK_FFNW_INIT_SEED + int(round_index))
        return FfnWidthBranch(extra)


def ffn_dim_of_round(r: Any) -> int:
    """The FFN width round ``r`` is BUILT at: the base plus its branch's width."""
    br = getattr(r, "ffn_extra", None)
    return FFN_DIM_BASE + (int(br.linear1.weight.shape[0]) if br is not None else 0)


def install_branches(tt: Any, ffn_dim: int, *, only_missing: bool = False) -> List[torch.nn.Module]:
    """Give every round of trunk ``tt`` (`trunk_kffn.trunk_rounds` order) its width branch at ``ffn_dim``, on the
    trunk's device and dtype; returns the branches built. ``only_missing``: skip a round that already carries one (a
    depth growth on a wide parent: the parent's TRAINED branches stay). A round whose branch is at another width is a
    `ValueError` (a width is one value per network)."""
    from agents.model.trunk_kffn import trunk_rounds

    built: List[torch.nn.Module] = []
    for i, r in enumerate(trunk_rounds(tt)):
        have = getattr(r, "ffn_extra", None)
        if have is not None:
            if ffn_dim_of_round(r) != int(ffn_dim):
                raise ValueError(f"trunk round {i} is {ffn_dim_of_round(r)} wide, not {ffn_dim}")
            if only_missing:
                continue
            raise ValueError(f"trunk round {i} already carries a width branch")
        br = build_ffn_extra(i, ffn_dim)
        if br is None:
            continue
        ref: torch.Tensor = r.linear1.weight                                       # type: ignore[assignment]
        r.ffn_extra = br.to(device=ref.device, dtype=ref.dtype)
        built.append(r.ffn_extra)
    return built
