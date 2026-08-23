"""Unit tests for the capacity battery's PURE math (`agents.model.capacity_probes`).

Everything here runs without a checkpoint, without torch on the hot path, and in milliseconds —
the estimators are separable from the model on purpose, because a battery whose numbers can only
be checked by running it on a 2M-parameter network is a battery nobody checks.

The properties that actually matter for a cross-generation instrument are pinned explicitly:

  * `random_targets` depends ONLY on (obs row, seed) — never on which other rows were sampled.
    A batch-statistic target would be silently re-defined by every generation's own eval traces,
    which makes the difference between two artifacts unreadable while looking fine.
  * `ridge_oof` is a real out-of-fold fit (it cannot see the test rows) and picks a penalty per
    target.
  * `jsonable` produces STRICT JSON — the artifact is meant to be read by something other than
    Python.
"""
import json

import numpy as np
import pytest

from agents.model.capacity_probes import (
    CAPACITY_BATTERY_VERSION, FEATURE_TAPS, L2_GRID, auc_score, ground_truth_facts, jsonable,
    kfold_indices, parameter_utilization, r2_columns, random_targets, rank_summary, ridge_oof,
)


# --------------------------------------------------------------------------- kfold

def test_kfold_partitions_exactly_once():
    folds = kfold_indices(37, 5, seed=3)
    assert len(folds) == 5
    joined = np.sort(np.concatenate(folds))
    assert np.array_equal(joined, np.arange(37))
    assert all(np.array_equal(f, np.sort(f)) for f in folds)      # each fold is sorted


def test_kfold_is_seed_deterministic_and_seed_sensitive():
    a = np.concatenate(kfold_indices(50, 5, seed=1))
    b = np.concatenate(kfold_indices(50, 5, seed=1))
    c = np.concatenate(kfold_indices(50, 5, seed=2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("n,folds", [(4, 5), (10, 1), (10, 0)])
def test_kfold_refuses_impossible_splits(n, folds):
    with pytest.raises(ValueError):
        kfold_indices(n, folds, seed=0)


# --------------------------------------------------------------------------- random targets

def test_random_targets_shape_range_and_determinism():
    obs = np.random.default_rng(0).standard_normal((40, 25)) * 30.0
    t1 = random_targets(obs, 8, seed=7)
    t2 = random_targets(obs, 8, seed=7)
    assert t1.shape == (40, 8)
    assert np.array_equal(t1, t2)
    assert np.all(np.abs(t1) < 1.0)                               # tanh range
    assert not np.array_equal(t1, random_targets(obs, 8, seed=8))


def test_random_targets_use_no_batch_statistics():
    """THE cross-generation property: t_k(x) must not depend on the other sampled rows.

    Two generations sample different states from their own eval traces. If the target family were
    z-scored (or otherwise normalized against the batch), each run would be fitting a DIFFERENT
    function while the artifact reported the same `seed` — and the whole point of the metric is
    that two artifacts are differenceable.
    """
    obs = np.random.default_rng(1).standard_normal((60, 12)) * 100.0
    full = random_targets(obs, 4, seed=0)
    subset = random_targets(obs[[3, 17, 41]], 4, seed=0)
    assert np.allclose(full[[3, 17, 41]], subset, atol=1e-12)


def test_random_targets_are_not_saturated_by_wide_scale_columns():
    """The signed-log squash is load-bearing: raw obs mixes ~600-valued IDs with 0-1 fractions,
    and an unsquashed projection saturates tanh into a near-binary target no head fits gradually."""
    rng = np.random.default_rng(2)
    obs = np.column_stack([rng.integers(0, 600, size=(200, 6)).astype(float),
                           rng.random((200, 6))])
    t = random_targets(obs, 8, seed=0)
    saturated = np.mean(np.abs(t) > 0.99)
    assert saturated < 0.5, f"{saturated:.2%} of target values are saturated"
    assert t.std() > 0.05


def test_random_targets_rejects_non_matrix():
    with pytest.raises(ValueError):
        random_targets(np.zeros(10), 4, seed=0)


# --------------------------------------------------------------------------- ridge

def test_ridge_oof_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8))
    W = rng.standard_normal((8, 3))
    Y = X @ W + 0.01 * rng.standard_normal((300, 3))
    oof, l2 = ridge_oof(X, Y, folds=5, seed=0)
    assert oof.shape == (300, 3)
    assert l2.shape == (3,)
    assert np.all(r2_columns(Y, oof) > 0.98)
    assert set(l2.tolist()) <= set(L2_GRID)


def test_ridge_oof_finds_no_signal_in_pure_noise():
    """The honest half: an OOF fit to an unrelated target must NOT come out positive. A probe that
    scores well on noise would read every generation's representation as informative."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 30))
    Y = rng.standard_normal((200, 2))
    oof, _ = ridge_oof(X, Y, folds=5, seed=0)
    assert np.all(r2_columns(Y, oof) < 0.15)


def test_ridge_oof_predictions_are_out_of_fold():
    """A row's prediction must come from a model that never saw it — checked by making the target
    a per-row lookup only memorisation could fit."""
    rng = np.random.default_rng(9)
    X = np.eye(40) + 0.0 * rng.standard_normal((40, 40))    # each row its own one-hot
    Y = rng.standard_normal((40, 1))                        # unlearnable across folds
    oof, _ = ridge_oof(X, Y, folds=4, seed=0)
    assert r2_columns(Y, oof)[0] < 0.2


def test_ridge_oof_survives_constant_columns():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal((120, 4)), np.full(120, 2.5), np.zeros(120)])
    Y = (X[:, :4] @ np.array([1.0, -2.0, 0.5, 3.0]))[:, None]
    oof, _ = ridge_oof(X, Y, folds=4, seed=0)
    assert np.isfinite(oof).all()
    assert r2_columns(Y, oof)[0] > 0.95


def test_ridge_oof_accepts_a_1d_target():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] * 2.0
    oof, l2 = ridge_oof(X, y, folds=5, seed=0)
    assert oof.shape == (100, 1) and l2.shape == (1,)


def test_ridge_oof_chooses_a_penalty_per_target():
    """One easy target and one pure-noise target must not be forced onto a single compromise λ."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, 40))
    Y = np.column_stack([X[:, 0] * 5.0, rng.standard_normal(300)])
    _, l2 = ridge_oof(X, Y, folds=5, seed=0)
    assert l2[1] >= l2[0], "the noise target should take at least as much regularisation"


# --------------------------------------------------------------------------- r2 / auc

def test_r2_columns_known_values():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert r2_columns(y, y)[0] == pytest.approx(1.0)
    assert r2_columns(y, np.full_like(y, y.mean()))[0] == pytest.approx(0.0)


def test_r2_columns_may_be_negative_and_is_not_clipped():
    y = np.array([[1.0], [2.0], [3.0]])
    bad = np.array([[10.0], [-10.0], [10.0]])
    assert r2_columns(y, bad)[0] < -1.0


def test_r2_columns_constant_target_is_zero_not_nan():
    y = np.full((10, 1), 3.0)
    out = r2_columns(y, np.zeros((10, 1)))
    assert out[0] == 0.0 and np.isfinite(out[0])


def test_auc_score_perfect_reversed_and_chance():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_score_known_partial_value():
    # 2 positives, 2 negatives; one positive outranks both negatives, one outranks neither.
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert auc_score(y, np.array([0.2, 0.9, 0.5, 0.1])) == pytest.approx(0.5)


def test_auc_score_is_nan_with_one_class():
    assert np.isnan(auc_score(np.ones(5), np.arange(5.0)))
    assert np.isnan(auc_score(np.zeros(5), np.arange(5.0)))


# --------------------------------------------------------------------------- rank

def test_rank_summary_on_rank_one_data():
    v = np.random.default_rng(0).standard_normal(20)
    Z = np.outer(np.linspace(-1, 1, 200), v)
    r = rank_summary(Z)
    assert r["pr"] == pytest.approx(1.0, abs=1e-6)
    assert r["srank99"] == 1
    assert r["dim"] == 20 and r["n_rows"] == 200
    assert r["pr_frac"] == pytest.approx(1.0 / 20)


def test_rank_summary_on_isotropic_data():
    Z = np.random.default_rng(1).standard_normal((6000, 10))
    r = rank_summary(Z)
    assert 8.0 < r["pr"] <= 10.0
    assert r["srank99"] >= 9


def test_rank_summary_is_translation_invariant():
    """Centered by construction — an offset must not change the reading, or a tap whose mean
    drifts across generations would look like a rank change."""
    Z = np.random.default_rng(2).standard_normal((500, 12))
    a, b = rank_summary(Z), rank_summary(Z + 1000.0)
    assert a["pr"] == pytest.approx(b["pr"], rel=1e-6)
    assert a["srank99"] == b["srank99"]


# --------------------------------------------------------------------------- facts

def _real_layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def test_ground_truth_facts_read_the_planted_values():
    """Built on the REAL layout — the facts must be read through it, never from a copied index."""
    from agents.observation.constants import POKEMON_ACTIVE_OFFSET

    layout = _real_layout()
    parts, pk = layout["parts"], layout["pokemon"]
    hp_off = pk["hp"]["offset"]
    obs = np.zeros((2, layout["total_dim"]), dtype=np.float32)

    for side, slot, hp in (("our_team", 2, 0.4), ("opp_team", 5, 0.75)):
        p = parts[side]
        block = obs[:, p["start"]:p["end"]].reshape(2, *p["reshape"])
        block[:, slot, POKEMON_ACTIVE_OFFSET] = 1.0
        block[:, slot, hp_off] = hp
        block[:, 0, hp_off] = 0.9                       # a second live mon on each side
        obs[:, p["start"]:p["end"]] = block.reshape(2, -1)

    g0 = parts["global"]["start"] + layout["global_layout"]["hazards"]["offset"]
    obs[0, g0] = 1.0 / 3.0                              # our side: 1 layer
    obs[1, g0 + 1] = 1.0                                # their side: 3 layers

    facts = ground_truth_facts(obs, layout)
    assert facts["our_active_hp"]["values"] == pytest.approx([0.4, 0.4])
    assert facts["opp_active_hp"]["values"] == pytest.approx([0.75, 0.75])
    assert facts["our_alive"]["values"] == pytest.approx([2.0, 2.0])
    assert facts["opp_alive"]["values"] == pytest.approx([2.0, 2.0])
    assert facts["our_spikes"]["values"] == pytest.approx([1.0, 0.0])
    assert facts["opp_spikes"]["values"] == pytest.approx([0.0, 1.0])


def test_ground_truth_facts_declare_task_and_note():
    layout = _real_layout()
    facts = ground_truth_facts(np.zeros((3, layout["total_dim"]), dtype=np.float32), layout)
    assert facts
    for name, spec in facts.items():
        assert spec["task"] in ("regression", "classification"), name
        assert spec["note"] and isinstance(spec["note"], str), name
        assert len(spec["values"]) == 3, name


# --------------------------------------------------------------------------- params

def test_parameter_utilization_groups_by_phase():
    import torch

    net = torch.nn.Module()
    net.alpha = torch.nn.Linear(4, 4, bias=False)
    net.beta = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.alpha.weight.fill_(0.0)
        net.beta.weight.fill_(1.0)
    out = parameter_utilization(net)
    assert out["n_params_total"] == 20
    assert out["phases"]["alpha"]["n_params"] == 16
    assert out["phases"]["alpha"]["zero_frac"] == 1.0
    assert out["phases"]["beta"]["rms"] == pytest.approx(1.0)
    assert out["phases"]["beta"]["l2_norm"] == pytest.approx(2.0)
    assert sum(p["param_share"] for p in out["phases"].values()) == pytest.approx(1.0)
    assert list(out["phases"]) == ["alpha", "beta"]         # descending param count


# --------------------------------------------------------------------------- json

def test_jsonable_round_trips_and_is_strict():
    body = {"a": np.float32(1.5), "b": [np.int64(3), float("nan"), float("inf")],
            "c": {"d": np.bool_(True), "e": None}, "f": (1, 2)}
    clean = jsonable(body)
    text = json.dumps(clean, allow_nan=False)               # would raise on NaN/Infinity
    assert json.loads(text) == clean
    assert clean["a"] == 1.5
    assert clean["b"] == [3, None, None]
    assert clean["c"] == {"d": True, "e": None}
    assert clean["f"] == [1, 2]


def test_jsonable_stringifies_the_unexpected():
    class Weird:
        def __repr__(self):
            return "<weird>"
    assert jsonable({"x": Weird()}) == {"x": "<weird>"}


def test_battery_constants_are_stable_surface():
    assert CAPACITY_BATTERY_VERSION >= 1
    assert FEATURE_TAPS == ("role_tokens", "team_tokens", "value_pooled",
                            "pi_features", "vf_features")
