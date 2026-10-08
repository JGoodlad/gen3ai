"""`--obs-facts {off,v1}` (gen3_obs_facts_v1, the X5 version break's part 3) on a REAL SB3-built policy (the
construction path training uses — the identity_init_test rule), over REAL observation rows (the compile parity
fixture: 64 rows from reproducible battles, every one carrying a nonzero OBS-FACTS block).

Each test names what a revert breaks:

* `off` reads NONE of the block: perturbing every facts cell leaves the forward BIT-identical (fails if any
  module reads the block under `off`);
* `v1` at init IS the `off` network: every shared parameter byte-equal at the same seed (the injector draws no
  RNG and SB3's orthogonal re-init skips it), the injector all-zero, the outputs equal — and the gradient reaches
  every one of its four projections (fails on a non-zero init, an RNG-drawing init, a dead projection);
* `v1` READS the block: once the injector is nonzero, the facts move the output (fails if the route is dropped);
* every OBS-FACTS sub-block is CLASSIFIED for `--token-encoding static` (`FACTS_TOKEN_CLASS`), and under static
  the SIDE-class facts (screens) reach the side BOARD tokens and never a per-mon token, while the D-class facts
  (seen / choice / vol) reach the per-mon tokens and not the board (fails on an unclassified sub-block, a screen
  routed to a mon under static, or a side route that is not wired).
"""
from __future__ import annotations

from typing import Any, Tuple

import pytest
import torch

from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.identity_init_test import _build_real_policy
from agents.model.obs_facts_inject import FACTS_TOKEN_CLASS, ObsFactsInject
from agents.observation import constants as C

_TOGGLES = dict(move_belief_mode="both", t0_species_prior=True, damage_op=True,
                damage_outgoing=True)
_SCREENS = slice(C.OFFSET_OBS_FACTS + C.FACTS_SCREENS_OFFSET,
                 C.OFFSET_OBS_FACTS + C.FACTS_SCREENS_OFFSET + C.FACTS_SCREENS_DIM)
_SEEN = slice(C.OFFSET_OBS_FACTS + C.FACTS_SEEN_OFFSET, C.OFFSET_OBS_FACTS + C.FACTS_SEEN_OFFSET + C.FACTS_SEEN_DIM)


@pytest.fixture(scope="module")
def pair() -> Tuple[Any, Any, torch.Tensor]:
    off, enc = _build_real_policy(obs_facts="off", **_TOGGLES)
    v1, _ = _build_real_policy(obs_facts="v1", **_TOGGLES)
    obs, _mask = load_parity_rows(enc.dimension)
    x = torch.as_tensor(obs[:16])
    assert bool((x[:, C.OFFSET_OBS_FACTS:] != 0).any(dim=1).all()), "PRECONDITION: every row carries facts"
    return off.policy, v1.policy, x


@pytest.fixture(scope="module")
def static_v1() -> Tuple[Any, torch.Tensor]:
    m, enc = _build_real_policy(obs_facts="v1", token_encoding="static", **_TOGGLES)
    obs, _mask = load_parity_rows(enc.dimension)
    return m.policy, torch.as_tensor(obs[:16])


def _features(policy: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        pi, vf = policy.features_extractor({"observation": x})
    return pi, vf


def _perturbed(x: torch.Tensor, cols: slice = slice(C.OFFSET_OBS_FACTS, None)) -> torch.Tensor:
    y = x.clone()
    y[:, cols] = y[:, cols] + 0.5
    y[:, C.OFFSET_OBS_FACTS + C.FACTS_CHOICE_OFFSET + 2] = 85.0     # a valid move id (Thunderbolt)
    return y


def _plant(inj: ObsFactsInject, seed: int = 5) -> dict:
    saved = {k: v.clone() for k, v in inj.state_dict().items()}
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in inj.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return saved


def test_the_observation_carries_the_block_at_its_declared_offset(pair):
    _, v1, x = pair
    assert x.shape[1] == C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM == 2845
    sl = v1.features_extractor.unpack._slices["obs_facts"]
    assert (sl.start, sl.stop) == (C.OFFSET_OBS_FACTS, C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM)


def test_off_builds_nothing_and_reads_none_of_the_block(pair):
    off, _, x = pair
    assert off.features_extractor.obs_facts_inject is None
    a, b = _features(off, x), _features(off, _perturbed(x))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_v1_at_init_is_the_off_network(pair):
    off, v1, x = pair
    so, s1 = off.state_dict(), v1.state_dict()
    extra = sorted(set(s1) - set(so))
    assert extra and all(".obs_facts_inject." in k for k in extra), extra
    assert set(so) <= set(s1)
    moved = [k for k in so if not torch.equal(so[k], s1[k])]
    assert not moved, f"a v1 build shifted shared initial weights: {moved[:5]}"
    assert all(torch.count_nonzero(s1[k]) == 0 for k in extra), "the injector is zero-init"
    a, b = _features(off, x), _features(v1, x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_v1_the_gradient_reaches_every_projection(pair):
    _, v1, x = pair
    inj = v1.features_extractor.obs_facts_inject
    v1.zero_grad()
    pi, vf = v1.features_extractor({"observation": x})
    (pi.square().sum() + vf.square().sum()).backward()
    for name in ("seen_proj", "vol_proj", "choice_proj", "screens_proj"):
        g = getattr(inj, name).bias.grad
        assert g is not None and torch.count_nonzero(g) > 0, name
    v1.zero_grad()


def test_v1_reads_the_block_once_the_injector_is_nonzero(pair):
    _, v1, x = pair
    inj = v1.features_extractor.obs_facts_inject
    saved = _plant(inj)
    try:
        a, b = _features(v1, x), _features(v1, _perturbed(x))
        assert not torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    finally:
        inj.load_state_dict(saved)


# ------------------------------------------------------------------ the static encoder's classification
def test_every_sub_block_is_classified_and_an_unclassified_one_is_refused(pair):
    _, v1, _ = pair
    layout = v1.features_extractor.layout
    assert set(layout["obs_facts"]) == set(FACTS_TOKEN_CLASS)
    assert FACTS_TOKEN_CLASS == {"seen": "D", "choice": "D", "vol": "D", "screens": "SIDE"}
    bad = {**layout, "obs_facts": {**layout["obs_facts"], "new_fact": {"offset": 84, "dim": 1}}}
    with pytest.raises(ValueError, match="no FACTS_TOKEN_CLASS"):
        ObsFactsInject(bad)


def test_static_routes_screens_to_the_side_tokens_and_never_a_mon(pair, static_v1):
    policy, x = static_v1
    fe = policy.features_extractor
    inj = fe.obs_facts_inject
    assert inj.side_to_board and fe.team_transformer.static_board
    saved = _plant(inj)
    try:
        with torch.no_grad():
            ctx = fe.unpack({"observation": x})
            tokens = fe.pokemon_encoder(ctx, fe.embeddings)
            for cols, mon_moves, side_moves in ((_SCREENS, False, True), (_SEEN, True, False)):
                y = x.clone()
                y[:, cols] = y[:, cols] + 0.5
                cy = fe.unpack({"observation": y})
                d_mon = (inj(ctx.obs_facts, tokens, ctx.our_active_idx, ctx.opp_active_local, fe.embeddings)
                         - inj(cy.obs_facts, tokens, cy.our_active_idx, cy.opp_active_local, fe.embeddings))
                d_side = inj.side_rows(ctx.obs_facts) - inj.side_rows(cy.obs_facts)
                assert bool(d_mon.abs().max() > 0) is mon_moves, (cols, "per-mon")
                assert bool(d_side.abs().max() > 0) is side_moves, (cols, "side")
            # the side route is WIRED: the screens move the forward through the board tokens alone
            y = x.clone()
            y[:, _SCREENS] = y[:, _SCREENS] + 0.5
            a, b = _features(policy, x), _features(policy, y)
            assert not torch.equal(a[0], b[0])
    finally:
        inj.load_state_dict(saved)


def test_legacy_routes_screens_to_the_mons(pair):
    _, v1, x = pair
    fe = v1.features_extractor
    inj = fe.obs_facts_inject
    assert not inj.side_to_board
    saved = _plant(inj)
    try:
        with torch.no_grad():
            ctx = fe.unpack({"observation": x})
            tokens = fe.pokemon_encoder(ctx, fe.embeddings)
            y = x.clone()
            y[:, _SCREENS] = y[:, _SCREENS] + 0.5
            cy = fe.unpack({"observation": y})
            d = (inj(ctx.obs_facts, tokens, ctx.our_active_idx, ctx.opp_active_local, fe.embeddings)
                 - inj(cy.obs_facts, tokens, cy.our_active_idx, cy.opp_active_local, fe.embeddings))
            assert bool((d.abs().amax(-1) > 0).all()), "every token of both sides reads its side's screens"
        with pytest.raises(ValueError, match="legacy has no side token"):
            fe.team_transformer(tokens, ctx, fe.embeddings, board_side_extra=inj.side_rows(ctx.obs_facts))
    finally:
        inj.load_state_dict(saved)
