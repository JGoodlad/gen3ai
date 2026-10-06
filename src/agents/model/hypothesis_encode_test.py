"""`gen3_x5_hyp_gather_v1` — X5's hypothesis tokens are the per-row `PokemonEncoder` pass, split EXACTLY at
its two first Linears (`hypothesis_encode`; `designs/research_state/measurements/x5_hyp_gather_2026-10-05/`).

* the fixed_mass forward takes the gathered path ONCE and runs `PokemonEncoder` ONCE (the real pass) —
  the per-row hypothesis pass is gone (fails on revert: 0 gathered calls, 2 encoder passes);
* blob never reaches it (its arithmetic is the X5 A/B's blob seeds', §7.5);
* the VALUES are the per-row pass's: at fp64 within 1e-12 (exact up to reassociation — measured ~6e-15),
  at fp32 within the declared 2e-5 (measured ~3e-6 on token values up to ~6), on every hypothesis slot of
  the K9 golden's real rows;
* the split is exact on EVERY opponent slot, the ACTIVE one included (its active-context scatter), and in
  its GRADIENTS (every encoder / embedding parameter, fp64).

The per-row pass (`PokemonEncoder.forward` on `hypothesis_ctx`) stays the DEFINITION: a change to the
encoder that `hypothesis_encode` does not mirror fails here.
"""
from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

import agents.model.extractor_forward as EF
from agents.model.hypothesis_encode import gathered_hypothesis_tokens
from agents.model.hypothesis_tokens import hypothesis_ctx
from agents.observation.constants import TEAM_SIZE

#: The declared fp32 bound on |gathered − per-row| (token values reach ~6; reassociation of two fp32 sums).
FP32_ATOL = 2e-5
#: The fp64 bound: exactness up to reassociation (measured ~6e-15).
FP64_ATOL = 1e-12


def _rows(model: Any, n: int = 0) -> Dict[str, torch.Tensor]:
    """The K9 golden buffer's real rows; ``n`` > 0 takes ``n`` of them, evenly spaced over the buffer (so
    early-game and late-game rows both appear), cycling when the buffer is shorter."""
    from stable_baselines3.common.utils import obs_as_tensor
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    if n:
        N = next(iter(obs.values())).shape[0]
        idx = (torch.arange(n) * max(1, N // n) + torch.arange(n) // N) % N
        obs = {k: v[idx] for k, v in obs.items()}
    return obs


@pytest.fixture(scope="module")
def fixed_mass() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_arm_learner("fixed_mass")
    LG.load_buffer_into(m, LG.arm_buffer("fixed_mass"))
    assert m.policy.features_extractor.hypothesis_builder is not None        # PRECONDITION: the X5 arm
    return m


@pytest.fixture(scope="module")
def blob() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    assert m.policy.features_extractor.hypothesis_builder is None            # PRECONDITION: blob
    return m


def _counting(monkeypatch: Any) -> List[int]:
    """Count the forward's calls of the gathered path. ``raising=False`` so a REVERT (the extractor no longer
    naming the function) fails on the COUNT assertion below, not on a missing attribute here."""
    calls = [0]
    real = gathered_hypothesis_tokens

    def counted(*a: Any, **k: Any) -> torch.Tensor:
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(EF, "gathered_hypothesis_tokens", counted, raising=False)
    return calls


def _encoder_passes(fe: Any) -> Tuple[List[int], Any]:
    n = [0]

    def hook(*_: Any) -> None:
        n[0] += 1
    return n, fe.pokemon_encoder.register_forward_hook(hook)


def test_the_fixed_mass_forward_gathers_once_and_runs_the_encoder_once(fixed_mass, monkeypatch):
    fe = fixed_mass.policy.features_extractor
    calls = _counting(monkeypatch)
    n, h = _encoder_passes(fe)
    try:
        with torch.no_grad():
            fe(_rows(fixed_mass))
    finally:
        h.remove()
    assert n[0] == 1, f"PokemonEncoder ran {n[0]}x — the per-row hypothesis pass is back"
    assert calls[0] == 1, f"the gathered hypothesis path ran {calls[0]}x (expected once per forward)"


def test_the_blob_forward_never_reaches_the_gathered_path(blob, monkeypatch):
    fe = blob.policy.features_extractor
    calls = _counting(monkeypatch)
    n, h = _encoder_passes(fe)
    try:
        with torch.no_grad():
            fe(_rows(blob))
    finally:
        h.remove()
    assert calls[0] == 0 and n[0] == 1


def _both(fe: Any, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    ctx = fe.unpack(obs)
    hs = fe._build_hypothesis_species(ctx, fe.pokemon_encoder(ctx, fe.embeddings))
    ref = fe.pokemon_encoder(hypothesis_ctx(ctx, hs, fe.layout), fe.embeddings)[:, TEAM_SIZE:2 * TEAM_SIZE]
    new = gathered_hypothesis_tokens(fe.pokemon_encoder, fe.embeddings, ctx, hs.slot_species,
                                     fe.hypothesis_builder.dex_rows)
    return ref, new, hs


def _fp64(model: Any, n: int = 0) -> Tuple[Any, Dict[str, torch.Tensor]]:
    pol = copy.deepcopy(model.policy).double()
    obs = {k: (v.double() if v.is_floating_point() else v) for k, v in _rows(model, n).items()}
    return pol.features_extractor, obs


#: Both static branches: B·6 < 400 encodes the slots' own dex rows (T2's small buckets); B·6 ≥ 400 the table.
BRANCHES = [pytest.param(8, id="per_slot_B8"), pytest.param(128, id="table_B128")]


@pytest.mark.parametrize("n", BRANCHES)
def test_the_gathered_tokens_are_the_per_row_pass_on_every_hypothesis_slot(fixed_mass, n):
    fe = fixed_mass.policy.features_extractor
    with torch.no_grad():
        ref, new, hs = _both(fe, _rows(fixed_mass, n))
    hyp = hs.slot_is_hypothesis
    assert int(hyp.sum()) >= 4, "PRECONDITION: the rows hold hypothesis slots"
    assert torch.isfinite(new).all()
    d32 = (ref - new)[hyp].abs().max().item()
    assert d32 <= FP32_ATOL, f"fp32 |gathered - per-row| = {d32:.3g} > {FP32_ATOL}"
    fe64, obs64 = _fp64(fixed_mass, n)
    with torch.no_grad():
        ref64, new64, hs64 = _both(fe64, obs64)
    d64 = (ref64 - new64)[hs64.slot_is_hypothesis].abs().max().item()
    assert d64 <= FP64_ATOL, f"fp64 |gathered - per-row| = {d64:.3g} > {FP64_ATOL}: not an exact split"


@pytest.mark.parametrize("n", BRANCHES)
def test_the_split_is_exact_on_every_slot_the_active_one_included_values_and_gradients(fixed_mass, n):
    """Every opponent slot holds a dex row (the ACTIVE too, so its active-context scatter is exercised), and
    both paths' outputs AND parameter gradients agree at fp64, on both static branches."""
    fe, obs = _fp64(fixed_mass, n)
    g = torch.Generator().manual_seed(7)
    ctx = fe.unpack(obs)
    # the real rows rarely carry an opponent active context (2 of the golden's 64), so PLANT one on every
    # row — both paths read the same ``ctx.opp_ctx_raw``
    ctx = dataclasses.replace(ctx, opp_ctx_raw=torch.randn(ctx.opp_ctx_raw.shape, generator=g, dtype=torch.float64))
    S = fe.hypothesis_builder.dex_rows.shape[0]
    species = torch.randint(1, S, (ctx.batch_size, TEAM_SIZE), generator=g)
    rows = fe.hypothesis_builder.dex_rows[species].to(ctx.pokemon_part.dtype)
    fake = SimpleNamespace(slot_rows=rows, slot_is_hypothesis=torch.ones_like(species, dtype=torch.bool))
    hctx = hypothesis_ctx(ctx, fake, fe.layout)          # type: ignore[arg-type]
    params = [p for p in list(fe.pokemon_encoder.parameters()) + list(fe.embeddings.parameters())
              if p.requires_grad]
    w = torch.randn(ctx.batch_size, TEAM_SIZE, 128, generator=g, dtype=torch.float64)

    def grads(tok: torch.Tensor) -> List[torch.Tensor]:
        gs = torch.autograd.grad((tok * w).sum(), params, allow_unused=True)
        return [torch.zeros_like(p) if x is None else x for p, x in zip(params, gs)]
    ref = fe.pokemon_encoder(hctx, fe.embeddings)[:, TEAM_SIZE:2 * TEAM_SIZE]
    g_ref = grads(ref)
    new = gathered_hypothesis_tokens(fe.pokemon_encoder, fe.embeddings, ctx, species,
                                     fe.hypothesis_builder.dex_rows)
    g_new = grads(new)
    assert (ref - new).abs().max().item() <= FP64_ATOL
    for p, a, b in zip(params, g_ref, g_new):
        scale = max(1.0, a.abs().max().item())
        assert (a - b).abs().max().item() <= FP64_ATOL * scale, f"gradient differs on a {tuple(p.shape)} parameter"
