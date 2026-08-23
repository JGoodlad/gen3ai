"""Unit tests for the unified two-part move belief (`move_prior_fusion`).

Pins: the move-prior-logits buffer (sigmoid recovers the Smogon usage prob; floor for unseen; Hidden
Power num-237 sums typed usage), the fusion math in `MoveBelief` (posterior = prior ⊕ learned-delta;
zero head ⇒ belief == prior; REVEALED moves pinned certain), gradient still flowing to the head,
off-path byte-identity (no buffer, state_dict unchanged), and the extractor dependency guard.
"""
import math

import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor, MoveBelief, _REVEAL_LOGIT
from agents.model import damage_tables as dt
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents import gen3_data

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_N_MOVES = _layout["max_moves"]
_N_SPECIES = _layout["max_species"]
_EMB = _layout["move_embedding_dim"]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _make_model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mappings, **kw)


# --------------------------------------------------------------------------- buffer
def test_prior_logits_recover_usage_probability():
    P = dt.build_move_prior_logits(_N_SPECIES, _N_MOVES)
    assert P.shape == (_N_SPECIES, _N_MOVES) and P.dtype == torch.float32
    assert torch.isfinite(P).all()
    sk = gen3_data.species.get("skarmory").num
    prior = gen3_data.priors.moves("skarmory")
    for mid in ("spikes", "roar", "drillpeck"):
        num = gen3_data.moves.get(mid).num
        assert abs(_sigmoid(P[sk, num].item()) - prior[mid]) < 0.02


def test_prior_logits_floor_for_unseen_and_unknown_species():
    """Two DIFFERENT reasons a cell can be low, which the unconditional legality gate
    (gen3_unconditional_move_legality_v1) must keep apart:

      * "not known to be ILLEGAL"  → the liftable floor  (unseen move / unknown species)
      * "known to be illegal"      → ~0                  (species has a learnset, move isn't in it)

    This test used to read `skarmory`/`fireblast` as the "unseen move" case and assert it sat at the
    floor. That was the LEGACY flat-prior behaviour and it is now wrong on its own premise: Skarmory
    genuinely CANNOT learn Fire Blast (asserted below), so floor-level mass there is exactly the
    phantom threat the gate exists to delete. The unseen-move case is re-derived from a move Skarmory
    CAN learn but the usage data never records."""
    P = dt.build_move_prior_logits(_N_SPECIES, _N_MOVES)
    sk = gen3_data.species.get("skarmory").num
    legal = gen3_data.learnset.get_legal_moves("skarmory")

    # ILLEGAL → ~0, never the floor.
    fb = gen3_data.moves.get("fireblast").num
    assert "fireblast" not in legal, "FIXTURE STALE: Skarmory can learn Fire Blast"
    assert _sigmoid(P[sk, fb].item()) < 1e-5

    # LEGAL but with no recorded usage → the liftable floor. Membership is tested on `_belief_num`,
    # not `num`: the builder writes at the BELIEF num, which collapses every typed Hidden Power onto
    # 237, so a raw-`num` comparison would happily pick a move whose cell a usage entry already wrote.
    usage_nums = {dt._belief_num(m, gen3_data.moves.get(m))
                  for m in gen3_data.priors.moves("skarmory") if gen3_data.moves.get(m) is not None}
    unseen = next(bn for bn in (dt._belief_num(m, gen3_data.moves.get(m)) for m in sorted(legal)
                                if gen3_data.moves.get(m) is not None)
                  if bn not in usage_nums and 0 <= bn < _N_MOVES)     # e.g. Double Team / Swagger
    assert abs(_sigmoid(P[sk, unseen].item()) - dt._PRIOR_FLOOR) < 1e-3

    # species num 0 = unknown sentinel → floor everywhere (no species-specific prior).
    # NOTHING is known about its movepool, so every move stays POSSIBLE: "not known to be illegal"
    # must never collapse into "known to be illegal".
    assert torch.allclose(torch.sigmoid(P[0]), torch.full((_N_MOVES,), dt._PRIOR_FLOOR), atol=1e-4)


def test_prior_logits_hidden_power_sums_typed_usage():
    """The typed HP usages fold into the 237 PRESENCE channel; P(has HP) = Σ typed usage
    (a mon runs ≤1 HP type).

    🚨 This test SKIPPED ON EVERY TREE, FOREVER, until 2026-08-23. It selected the typed variants
    by ``moves.get(mid).num == 237`` — true only before `gen3_typed_hidden_power_ids_v1` gave the
    16 typed HPs their own dex nums 355-370. After that the filter matched NOTHING, `typed_sum`
    was 0.0 for every species, and the test skipped with a message blaming the DATA ("no HP-running
    species in the sample") rather than itself. The production code was right the whole time — it
    keys the fold on the move ID, via `_belief_num` — so the fix is to ask the SAME seam the
    production path asks instead of re-deriving the rule from a literal.
    """
    P = dt.build_move_prior_logits(_N_SPECIES, _N_MOVES)
    n_ran = 0
    for sid in ("zapdos", "celebi", "salamence"):
        snum = gen3_data.species.get(sid).num
        prior = gen3_data.priors.moves(sid)
        # `_belief_num` IS the production fold (id-keyed, not num-keyed). Routing through it means
        # a future re-keying moves this test with the code rather than silently past it.
        typed_sum = sum(p for mid, p in prior.items()
                        if (md := gen3_data.moves.get(mid)) is not None
                        and dt._belief_num(mid, md) == dt.HIDDEN_POWER_NUM)
        if typed_sum <= dt._PRIOR_FLOOR:
            continue
        got = _sigmoid(P[snum, dt.HIDDEN_POWER_NUM].item())
        assert abs(got - min(typed_sum, 1.0 - 1e-6)) < 0.02, (sid, got, typed_sum)
        n_ran += 1
    # ASSERT, never skip: all three are famous HP carriers in the committed Smogon priors, so a
    # zero here means the selection broke again — which is the exact failure this test just had.
    assert n_ran == 3, (
        f"only {n_ran}/3 sampled species had typed-HP usage above the prior floor — the typed-HP "
        f"selection is broken again (it was num-keyed and matched nothing for over a generation), "
        f"or the committed priors changed. Do not turn this back into a skip.")


# --------------------------------------------------------------------------- fusion math
def _mb(prior_fusion=True):
    return MoveBelief(_N_MOVES, _EMB, prior_fusion=prior_fusion, n_species=_N_SPECIES)


def test_zero_head_posterior_equals_prior():
    """With the learned delta zeroed, the posterior over UNREVEALED moves == the Smogon prior."""
    mb = _mb()
    torch.nn.init.zeros_(mb.move_head.weight)
    torch.nn.init.zeros_(mb.move_head.bias)
    emb = torch.nn.Embedding(_N_MOVES, _EMB)
    sk = gen3_data.species.get("skarmory").num
    species = torch.zeros(1, 6, dtype=torch.long); species[0, 0] = sk
    _, logits = mb(torch.zeros(1, 6, 128), torch.ones(1, 6, dtype=torch.bool), emb,
                   opp_species_ids=species, opp_move_ids=torch.zeros(1, 6, 4, dtype=torch.long))
    spikes = gen3_data.moves.get("spikes").num
    assert abs(torch.sigmoid(logits[0, 0, spikes]).item() - gen3_data.priors.moves("skarmory")["spikes"]) < 0.02


def test_revealed_move_pinned_certain():
    """A move seen this battle (opp move id > 0) is pinned to ~P 1 regardless of the prior."""
    mb = _mb()
    torch.nn.init.zeros_(mb.move_head.weight); torch.nn.init.zeros_(mb.move_head.bias)
    emb = torch.nn.Embedding(_N_MOVES, _EMB)
    sk = gen3_data.species.get("skarmory").num
    fb = gen3_data.moves.get("fireblast").num            # off-prior move
    species = torch.zeros(1, 6, dtype=torch.long); species[0, 0] = sk
    move_ids = torch.zeros(1, 6, 4, dtype=torch.long); move_ids[0, 0, 0] = fb
    _, logits = mb(torch.zeros(1, 6, 128), torch.ones(1, 6, dtype=torch.bool), emb,
                   opp_species_ids=species, opp_move_ids=move_ids)
    assert torch.sigmoid(logits[0, 0, fb]).item() > 0.999
    assert logits[0, 0, fb].item() == pytest.approx(_REVEAL_LOGIT, abs=1e-4)


def test_fusion_gradient_reaches_move_head():
    mb = _mb()
    emb = torch.nn.Embedding(_N_MOVES, _EMB)
    _, logits = mb(torch.randn(2, 6, 128), torch.ones(2, 6, dtype=torch.bool), emb,
                   opp_species_ids=torch.zeros(2, 6, dtype=torch.long),
                   opp_move_ids=torch.zeros(2, 6, 4, dtype=torch.long))
    logits.sum().backward()
    assert mb.move_head.weight.grad is not None and mb.move_head.weight.grad.abs().sum() > 0


def test_off_is_byte_identical():
    """prior_fusion=False builds no buffer and reproduces the from-scratch head exactly."""
    off = _mb(prior_fusion=False)
    assert off.prior_fusion is False and not hasattr(off, "move_prior_logits")
    emb = torch.nn.Embedding(_N_MOVES, _EMB)
    opp = torch.randn(2, 6, 128)
    _, logits = off(opp, torch.ones(2, 6, dtype=torch.bool), emb,
                    opp_species_ids=torch.zeros(2, 6, dtype=torch.long))   # ignored when off
    assert torch.allclose(logits, off.move_head(opp))   # raw head output, no fusion


# --------------------------------------------------------------------------- extractor wiring
def test_buffer_is_non_persistent_and_state_dict_unchanged():
    """The prior buffer is recomputable physics → not in the state_dict; on/off keys identical."""
    on, _ = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True), None
    off = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=False)
    assert set(on.state_dict()) == set(off.state_dict())
    assert not any("move_prior_logits" in k for k in on.state_dict())


def test_extractor_guard_requires_move_belief_on():
    with pytest.raises(ValueError, match="move_prior_fusion"):
        _make_model(attend_unrevealed_opponents=True, move_belief_mode="off", move_prior_fusion=True)
    # valid with a head present.
    _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True)


def test_full_forward_finite_with_fusion():
    m = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                    damage_op=True, move_prior_fusion=True)
    m.eval()
    with torch.no_grad():
        pi, vf = m.forward({"observation": torch.rand(3, _layout["total_dim"])})
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
