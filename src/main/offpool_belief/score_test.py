"""The scorer on a REAL ``MaskablePPO``-built production-surface learner (``learner_golden.build_learner``: every
learned delta non-zero) over the golden buffer's real rows, with labels drawn from each row's own candidate sets.

The pins: the prior column IS the deltas-zeroed model (a model with zeroed deltas scores post == prior exactly, on
every metric and team); a live model's posterior differs; the closed-form λ blend of the species leg is a third
forward at that λ (and the move / item blends are NOT, which is pinned too); the observation is sliced where the constants say; and a bank of another observation
width is refused.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch as th

from main.offpool_belief import OffPoolBeliefError
from main.offpool_belief import metrics as M
from main.offpool_belief import score as SC

N_TEAMS = 4


@pytest.fixture(scope="module")
def learner() -> Tuple[Any, Dict[str, np.ndarray]]:
    from agents.training import learner_golden as LG

    th.manual_seed(0)
    model = LG.build_learner(trunk_precision="fp32")
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = rb.observations["observation"].reshape((-1,) + rb.observations["observation"].shape[2:])
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    mask = rb.action_masks.reshape(-1, rb.action_masks.shape[-1]) > 0.5
    model.policy.set_training_mode(False)
    return model, _labelled(model, obs, mask)


def _labelled(model: Any, obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Bank-shaped rows: per row, up to two hidden species drawn from the row's own candidate set, up to three
    legal non-Hidden-Power moves per revealed opponent mon, and a real item per revealed mon."""
    fe = model.policy.features_extractor
    ix = SC.obs_index(fe.pokemon_encoder.layout)
    fw = SC._forward(model, obs, mask, 0.0)
    legal = (fe.move_belief.move_prior_probs > 2.0 * SC._illegal_prob()).numpy()
    n_items = fw["item_logp"].shape[-1]
    opp = obs[:, ix.opp_start:ix.opp_start + ix.n_slots * ix.slot_dim].reshape(len(obs), ix.n_slots, ix.slot_dim)
    species = np.rint(opp[:, :, ix.species]).astype(np.int64)
    rng = np.random.default_rng(7)
    n = len(obs)
    bs = -np.ones((n, 6), np.int32)
    km = -np.ones((n, 6, 4), np.int32)
    il = -np.ones((n, 6), np.int32)
    im = np.zeros((n, 6), np.float32)
    for r in range(n):
        cands = np.flatnonzero(fw["species_cand"][r])
        if cands.size:
            pick = rng.choice(cands, size=min(2, cands.size), replace=False)
            bs[r, :len(pick)] = pick
        for i in range(6):
            if species[r, i] <= 0:
                continue
            lm = np.flatnonzero(legal[species[r, i]])
            lm = lm[(lm != 237) & ((lm < 355) | (lm > 370))]
            pick = rng.choice(lm, size=min(3, lm.size), replace=False)
            km[r, i, :len(pick)] = pick
            il[r, i], im[r, i] = 1 + int(rng.integers(0, n_items - 1)), 1.0
    # A seeded shuffle, not ``row % N_TEAMS``: the buffer is step-major over 4 envs, so that would make a team of
    # ONE env's rows, and one env of the golden buffer shows a hidden candidate on only a few of its rows.
    team = (np.random.default_rng(11).permutation(n) % N_TEAMS).astype(np.int32)
    return {"obs": obs, "mask": mask, "team": team, "belief_species": bs,
            "known_moves": km, "item_label": il, "item_mask": im}


def _zeroed(model: Any) -> Any:
    m = copy.deepcopy(model)
    fe = m.policy.features_extractor
    with th.no_grad():
        for lin in (fe.hypothesis_builder.delta_out, fe.move_belief.move_head, fe.item_belief_head.item_head):
            lin.weight.zero_()
            lin.bias.zero_()
    return m


def test_the_observation_is_sliced_where_the_constants_say(learner: Any) -> None:
    from agents.observation import constants as K

    ix = SC.obs_index(learner[0].policy.features_extractor.pokemon_encoder.layout)
    assert (ix.opp_start, ix.slot_dim, ix.n_slots) == (K.OFFSET_OPP_TEAM, K.POKEMON_FULL_DIM, 6)
    assert ix.species == K.POKEMON_SPECIES_OFFSET
    assert ix.move_ids == tuple(K.POKEMON_MOVES_OFFSET + K.MOVE_SLOT_DIM * j for j in range(4))
    assert ix.move_known == tuple(K.POKEMON_MOVES_OFFSET + K.MOVE_SLOT_DIM * j + 6 for j in range(4))
    assert ix.item_known == K.POKEMON_ITEMS_OFFSET + 1
    assert ix.total_dim == learner[1]["obs"].shape[1]


def test_every_family_has_units_on_the_fixture(learner: Any) -> None:
    model, rows = learner
    ts = SC.score_rows(model, rows, N_TEAMS, chunk=16)
    for f in M.FAMILIES:
        assert (ts.units[f] > 0).all(), f"{f}: a team has no unit, the equalities below would be vacuous"


def test_zeroed_deltas_score_exactly_the_prior(learner: Any) -> None:
    model, rows = learner
    ts = SC.score_rows(_zeroed(model), rows, N_TEAMS, mixes=(0.25,), chunk=16)
    for metric in M.METRICS:
        assert np.array_equal(ts.sums[metric][M.POST], ts.sums[metric][M.PRIOR]), metric
    summ = SC.summarize(ts)
    for metric in ("species_logp", "moves_logp", "items_logp"):
        d = summ[metric]["post_minus_prior"]
        assert d["mean"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0, (metric, d)
        assert abs(summ[metric]["mix0.25_minus_prior"]["mean"]) < 1e-9          # a blend of a thing with itself


def test_a_live_model_differs_from_its_prior_and_the_prior_column_is_the_zeroed_model(learner: Any) -> None:
    model, rows = learner
    live = SC.score_rows(model, rows, N_TEAMS, chunk=16)
    zero = SC.score_rows(_zeroed(model), rows, N_TEAMS, chunk=16)
    for metric in ("species_logp", "moves_logp", "items_logp"):
        assert not np.array_equal(live.sums[metric][M.POST], live.sums[metric][M.PRIOR]), metric
        # the prior the live model is compared with is exactly what its heads return with no learned delta
        assert np.array_equal(live.sums[metric][M.PRIOR], zero.sums[metric][M.POST]), metric
    for f in M.FAMILIES:
        assert np.array_equal(live.units[f], zero.units[f])
    assert SC.summarize(live)["species_logp"]["post_minus_prior"]["mean"] != 0.0


def test_scoring_leaves_the_model_unmixed_and_the_chunk_size_is_not_a_knob(learner: Any) -> None:
    from agents.model import belief_prior_mix as BPM

    model, rows = learner
    a = SC.score_rows(model, rows, N_TEAMS, chunk=16)
    assert BPM.belief_prior_mix_of(model) == 0.0
    b = SC.score_rows(model, rows, N_TEAMS, chunk=48)
    for f in M.FAMILIES:
        assert np.array_equal(a.units[f], b.units[f])
    for metric in M.METRICS:
        np.testing.assert_allclose(a.sums[metric][M.POST], b.sums[metric][M.POST], rtol=0, atol=1e-4)


def test_the_species_blend_is_the_model_run_at_that_mix_and_the_other_legs_are_not(learner: Any) -> None:
    """The species presence is the first belief of the forward, so its closed-form blend IS a third forward at
    λ = 0.25. The move and item heads read tokens DOWNSTREAM of the mixed presence, so the model run at λ gives them
    a different delta: their blend column is the head's own output blended with its prior, nothing more, and this
    pins that the two are not the same number (a doc that calls the column "the model at λ" would be wrong)."""
    model, rows = learner
    obs, mask = rows["obs"], rows["mask"]
    post, prior, mid = (SC._forward(model, obs, mask, lam) for lam in (0.0, 1.0, 0.25))
    cand = post["species_cand"]
    assert cand.any()
    np.testing.assert_allclose(M.mix_log_prob(post["species_log_pi"], prior["species_log_pi"], 0.25)[cand],
                               mid["species_log_pi"][cand], rtol=0, atol=2e-5)
    blend = M.mix_log_prob(post["item_logp"], prior["item_logp"], 0.25)
    assert np.abs(blend - mid["item_logp"]).max() > 1e-3


def test_a_bank_of_another_width_and_a_bare_run_directory_are_refused(learner: Any, tmp_path: Any) -> None:
    model, rows = learner
    narrow = dict(rows, obs=rows["obs"][:, :-1])
    with pytest.raises(OffPoolBeliefError, match="wide"):
        SC.score_rows(model, narrow, N_TEAMS)
    with pytest.raises(OffPoolBeliefError, match="name the checkpoint .zip"):
        SC.load_checkpoint(str(tmp_path))


def test_the_semantics_gap_names_the_rows_between() -> None:
    from agents.model.model_version import constants as MV

    head = int(MV.OBS_SEMANTICS_VERSION)
    assert SC.obs_gap(head, head)["rows_between"] == []
    gap = SC.obs_gap(161, 168)["rows_between"]
    assert gap == [162, 163, 165, 166, 167, 168]
    assert SC.obs_gap(None, head)["rows_between"] is None and "unknown" in SC.obs_gap(None, head)["note"]
