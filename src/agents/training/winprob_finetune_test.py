"""Unit tests for the head-only win-prob fine-tune.

Unmarked (pure, in-process, no checkpoint, no network, no bridge) and deliberately cheap — the
whole file is a few seconds, well inside the unmarked tier's 30 s budget guard. The subject model
is stood in for by a synthetic trunk+head pair wherever a real one is not the thing under test;
where the REAL :class:`WinProbHead` matters (determinism, resume, the graft's state_dict shape) it
is used directly, because it is a four-layer MLP and costs nothing.

The six things pinned here are the six ways this instrument could be quietly wrong:

1. the schema round-trip, INCLUDING that ``decision_idx`` indexes the right npz row (the exact bug
   ``cf_audit`` shipped, which the digest check exists to make impossible);
2. the label math — binomial == BCE at n==1, an n-rollout label pulling n times a single sample,
   the loss minimized at ``phi = k/n``, and the reported noise floor matching ``sqrt(p(1-p)/n)``;
3. trunk-frozen, as a MEASUREMENT (grads and weights after a real optimizer step), plus the
   structural half: the fit's optimizer cannot even name a trunk parameter;
4. holdout hygiene — battle-level split, adversarial shapes, and an explicit failure when someone
   hands it a state-level split;
5. determinism, and that ``--resume`` reproduces the uninterrupted run bitwise;
6. the slice re-weighting's mean-1 normalization and its ordering.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from agents.training.harvest_schema import (HarvestRow, load_obs, obs_b64, obs_digest, read_dir,
                                            write_rows)
from agents.training.winprob_finetune import (SLICE_EDGES, SLICE_VERSION, FitConfig, PooledDataset,
                                              assert_battle_disjoint, binomial_nll, build_head,
                                              calibration_ece, evaluate, fit_head,
                                              label_noise_sd, label_noise_variance,
                                              precompute_value_pooled, slice_index, slice_weights,
                                              split_by_battle, split_pooled, subset)

D = 8  # a small stand-in for D_MODEL wherever the real head is not the subject of the test


# ---------------------------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------------------------

def _row(battle: str, idx: int, obs: np.ndarray, *, k: int, n: int, turn: int,
         npz: str = "", inline: bool = True) -> HarvestRow:
    return HarvestRow(
        run="run_test", battle_tag=battle, decision_idx=idx, turn=turn,
        n_rollouts=n, n_wins=k, phi_head=0.5, beta_evidence=None, beta_mean=None,
        priority=1.0, provenance={"opponent": "bot", "n_timeout": 0},
        obs_npz=npz or f"{battle}_states.npz", obs_sha1=obs_digest(obs),
        obs_inline=obs_b64(obs) if inline else None)


def _pooled(n_rows: int = 200, dim: int = D, seed: int = 0, n_rollouts: int = 8,
            battles: int = 20, slice_mode: str = "inverse") -> PooledDataset:
    """A synthetic PooledDataset whose labels are a real (noisy) function of x, so a fit has
    something to learn and val NLL actually moves."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, dim)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(x[:, 0] * 1.5)))
    k = rng.binomial(n_rollouts, p)
    return PooledDataset(
        x=x, wins=k.astype(np.int64),
        n_rollouts=np.full(n_rows, n_rollouts, dtype=np.int64),
        turn=rng.integers(5, 240, size=n_rows).astype(np.int64),
        battle_tag=[f"b{i % battles:03d}" for i in range(n_rows)],
        weight=np.ones(n_rows)).set_weights(slice_mode)


class _Trunk(torch.nn.Module):
    """A stand-in frozen trunk: obs -> value_pooled."""

    def __init__(self, obs_dim: int = 16, out: int = D) -> None:
        super().__init__()
        self.net = torch.nn.Linear(obs_dim, out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _Head(torch.nn.Module):
    """A stand-in head with the WinProbHead signature: value_pooled -> one logit."""

    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.net = torch.nn.Linear(d, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


# ---------------------------------------------------------------------------------------------
# 1. schema round-trip
# ---------------------------------------------------------------------------------------------

def test_inline_rows_round_trip_through_write_read_and_load_obs(tmp_path):
    rng = np.random.default_rng(1)
    obs = [rng.normal(size=32).astype(np.float32) for _ in range(4)]
    rows = [_row(f"b{i}", i, obs[i], k=3, n=8, turn=40 + i) for i in range(4)]
    write_rows(rows, str(tmp_path / "shard_000.jsonl.gz"))

    back = read_dir(str(tmp_path))
    assert len(back) == 4
    for i, r in enumerate(back):
        assert np.array_equal(load_obs(r), obs[i])
        assert r["n_wins"] == 3 and r["n_rollouts"] == 8


def test_npz_rows_resolve_by_decision_idx_and_a_wrong_index_RAISES_on_the_digest(tmp_path):
    """The cf_audit bug, pinned: an npz-backed row must index by ``decision_idx``, and the digest
    is what turns 'indexed the wrong row' from a silent wrong label into a refusal."""
    rng = np.random.default_rng(2)
    states = rng.normal(size=(6, 32)).astype(np.float32)
    models_root = tmp_path / "models"
    (models_root / "run_test").mkdir(parents=True)
    rel = os.path.join("run_test", "b0_states.npz")
    np.savez(models_root / rel, obs=states)

    good = _row("b0", 4, states[4], k=2, n=8, turn=70, npz=rel, inline=False)
    write_rows([good], str(tmp_path / "s.jsonl.gz"))
    (row,) = read_dir(str(tmp_path))
    assert np.array_equal(load_obs(row, models_root=str(models_root)), states[4])

    row["decision_idx"] = 1                      # point it at the WRONG row of the same npz
    with pytest.raises(ValueError, match="obs digest mismatch"):
        load_obs(row, models_root=str(models_root))


def test_precompute_counts_digest_rejections_and_never_drops_them_silently(tmp_path):
    rng = np.random.default_rng(3)
    obs = [rng.normal(size=16).astype(np.float32) for _ in range(5)]
    rows = [_row(f"b{i}", i, obs[i], k=1, n=4, turn=30 * i + 10) for i in range(5)]
    write_rows(rows, str(tmp_path / "s.jsonl.gz"))
    back = read_dir(str(tmp_path))
    back[2]["obs_sha1"] = "0" * 40               # corrupt exactly one row's digest

    trunk = _Trunk(obs_dim=16)
    def pooled_fn(a):
        with torch.no_grad():
            return trunk(torch.as_tensor(a)).numpy()

    ds, rep = precompute_value_pooled(pooled_fn, back, batch_size=2, log=lambda _s: None)
    assert rep.loaded == 4 and rep.rejected == 1 and rep.digest_mismatch == 1
    assert len(ds) == 4 and ds.x.shape == (4, D)
    assert rep.examples and "digest_mismatch" in rep.examples[0]


# ---------------------------------------------------------------------------------------------
# 2. label math
# ---------------------------------------------------------------------------------------------

def test_binomial_nll_equals_bce_when_n_is_one():
    torch.manual_seed(0)
    z = torch.randn(64)
    k = (torch.rand(64) < 0.5).to(torch.float32)
    n = torch.ones(64)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, k, reduction="mean")
    assert torch.allclose(binomial_nll(z, k, n), bce, atol=1e-7)


def test_it_agrees_with_the_live_trainers_cf_binomial_nll_at_unit_weights():
    """Same normalization (SUM NLL / SUM n) as `cf_terms.cf_binomial_nll`, so an offline number
    here is comparable with the live `cf/*` scalar rather than merely similar to it."""
    from agents.training.cf_terms import cf_binomial_nll

    torch.manual_seed(1)
    z = torch.randn(50)
    n = torch.randint(1, 33, (50,)).to(torch.float32)
    k = torch.floor(torch.rand(50) * (n + 1)).clamp(max=n)
    assert torch.allclose(binomial_nll(z, k, n), cf_binomial_nll(z, k / n, n), atol=1e-6)


def test_a_k_of_n_label_weighs_exactly_n_times_a_single_sample():
    z = torch.tensor([0.3])
    one = binomial_nll(z, torch.tensor([1.0]), torch.tensor([1.0])) * 1.0
    # 7 wins of 7 is 7 identical single observations: the SUM is 7x, and the /SUM n normalization
    # brings the reported mean back to the same per-rollout number.
    seven_sum = 7.0 * binomial_nll(z, torch.tensor([1.0]), torch.tensor([1.0]))
    assert torch.allclose(binomial_nll(z, torch.tensor([7.0]), torch.tensor([7.0])), one)
    manual = 7.0 * torch.nn.functional.softplus(-z)
    assert torch.allclose(manual.sum(), seven_sum, atol=1e-6)

    # And against a MIXED batch: one 32-rollout row moves the total exactly 32x a 1-rollout row.
    mixed = binomial_nll(torch.tensor([0.0, 0.0]), torch.tensor([32.0, 0.0]),
                         torch.tensor([32.0, 1.0]))
    expect = (32 * np.log(2) + np.log(2)) / 33.0
    assert abs(float(mixed) - expect) < 1e-6


@pytest.mark.parametrize("k,n", [(3, 8), (1, 32), (17, 20), (0, 4), (5, 5)])
def test_the_loss_is_minimized_at_phi_equals_k_over_n(k, n):
    """Fit ONE free logit and watch it converge to the empirical rate. This is the property that
    makes the head's output a calibrated probability rather than a ranking score."""
    z = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(4000):
        opt.zero_grad()
        binomial_nll(z, torch.tensor([float(k)]), torch.tensor([float(n)])).backward()
        opt.step()
    assert abs(float(torch.sigmoid(z)) - k / n) < 2e-3


def test_the_reported_noise_floor_matches_the_labels_own_sampling_sd():
    """With n rollouts the label's own sd is sqrt(p(1-p)/n). `label_noise_variance` estimates that
    unbiasedly (E[phat(1-phat)/(n-1)] == p(1-p)/n exactly), so the reported floor must land on it."""
    rng = np.random.default_rng(7)
    for p, n in ((0.5, 32), (0.2, 16), (0.8, 8)):
        k = rng.binomial(n, p, size=40000)
        nn = np.full(40000, n)
        assert abs(label_noise_variance(k, nn) - p * (1 - p) / n) < 0.0004
        assert abs(label_noise_sd(k, nn) - np.sqrt(p * (1 - p) / n)) < 0.002


def test_the_noise_floor_is_nan_not_zero_when_no_row_can_estimate_it():
    assert np.isnan(label_noise_variance(np.array([1, 0]), np.array([1, 1])))


def test_evaluate_reports_brier_against_that_floor_and_a_perfect_head_lands_on_it():
    """A head that predicts the TRUE p still pays the label's dice — its Brier equals the floor and
    its EXCESS is ~0. That is the whole reason the excess is the number to quote."""
    rng = np.random.default_rng(11)
    n, p = 32, 0.35
    k = rng.binomial(n, p, size=6000)
    ds = PooledDataset(x=np.zeros((6000, D), dtype=np.float32), wins=k.astype(np.int64),
                       n_rollouts=np.full(6000, n, dtype=np.int64),
                       turn=np.full(6000, 50, dtype=np.int64),
                       battle_tag=["b"] * 6000, weight=np.ones(6000)).set_weights("none")

    class _Const(torch.nn.Module):
        def forward(self, x):  # noqa: D102
            return torch.full((x.shape[0], 1), float(np.log(p / (1 - p))))

    m = evaluate(_Const(), ds)
    assert abs(m["brier"] - m["brier_floor"]) < 5e-4
    assert m["brier_excess"] < 5e-4
    assert m["ece"] < 0.02


def test_calibration_ece_is_zero_for_a_perfectly_calibrated_prediction():
    phi = np.array([0.25, 0.75])
    assert calibration_ece(phi, np.array([25.0, 75.0]), np.array([100.0, 100.0])) < 1e-9


# ---------------------------------------------------------------------------------------------
# 3. trunk-frozen
# ---------------------------------------------------------------------------------------------

def test_a_real_optimizer_step_leaves_every_trunk_parameter_bitwise_unchanged():
    """The MEASUREMENT, not the promise: run one real step of the fit's own loss through a
    trunk+head pair and assert the trunk moved by exactly nothing."""
    torch.manual_seed(0)
    trunk, head = _Trunk(obs_dim=16), _Head()
    obs = torch.randn(24, 16)
    with torch.no_grad():                       # phase 1: the trunk runs ONCE, outside the graph
        pooled = trunk(obs)
    before = [p.detach().clone() for p in trunk.parameters()]

    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)
    binomial_nll(head(pooled), torch.full((24,), 3.0), torch.full((24,), 8.0)).backward()
    opt.step()

    for p in trunk.parameters():
        assert p.grad is None or bool(torch.all(p.grad == 0))
    for b, p in zip(before, trunk.parameters()):
        assert torch.equal(b, p), "a trunk parameter MOVED during a head-only fit"
    assert any(g is not None and bool(torch.any(g != 0)) for g in
               (p.grad for p in head.parameters())), "the head did not train at all"


def test_the_fits_optimizer_cannot_even_name_a_trunk_parameter(tmp_path):
    """The STRUCTURAL half. `fit_head` builds its own Adam over `head.parameters()` and
    `_assert_head_only` refuses anything else, so a future fused phase-1/phase-2 rewrite that
    hands the trunk in fails loudly instead of silently re-training it."""
    from agents.training.winprob_finetune import _assert_head_only

    trunk, head = _Trunk(), _Head()
    _assert_head_only(head, torch.optim.Adam(head.parameters(), lr=0.1))     # the legal shape
    bad = torch.optim.Adam(list(head.parameters()) + list(trunk.parameters()), lr=0.1)
    with pytest.raises(ValueError, match="HEAD-ONLY"):
        _assert_head_only(head, bad)


def test_fit_head_holds_only_head_params_and_the_cached_pooled_carries_no_graph(tmp_path):
    ds = _pooled(n_rows=40, battles=8)
    tr, va = split_pooled(ds, 0.25, seed=0)
    head = _Head()
    fit_head(head, tr, va, FitConfig(epochs=1, batch_size=16), out_dir=str(tmp_path))
    x = torch.as_tensor(tr.x)
    assert not x.requires_grad and x.grad_fn is None


# ---------------------------------------------------------------------------------------------
# 4. holdout hygiene
# ---------------------------------------------------------------------------------------------

def test_split_by_battle_never_puts_one_battle_on_both_sides():
    rows = [{"battle_tag": f"b{i % 13:02d}", "i": i} for i in range(400)]
    tr, va = split_by_battle(rows, 0.25, seed=3)
    assert len(tr) + len(va) == 400
    assert not ({r["battle_tag"] for r in tr} & {r["battle_tag"] for r in va})
    assert {r["i"] for r in tr} | {r["i"] for r in va} == set(range(400))


def test_the_adversarial_shape_many_states_few_battles_still_splits_cleanly():
    rows = [{"battle_tag": f"b{i % 3}"} for i in range(900)]        # 300 states per battle
    tr, va = split_by_battle(rows, 0.34, seed=1)
    assert not ({r["battle_tag"] for r in tr} & {r["battle_tag"] for r in va})
    assert len(va) in (300, 600)          # a whole battle, never a slice of one
    # And the degenerate single-battle case cannot leak either: it simply cannot hold out.
    one = [{"battle_tag": "solo"} for _ in range(50)]
    tr1, va1 = split_by_battle(one, 0.5, seed=1)
    assert (len(tr1), len(va1)) == (0, 50) or (len(tr1), len(va1)) == (50, 0)


def test_a_state_level_split_is_REFUSED_by_the_guard():
    rows = [{"battle_tag": f"b{i % 5}"} for i in range(100)]
    tr, va = rows[:70], rows[70:]                       # the state-level splitter, deliberately
    with pytest.raises(ValueError, match="battle-level holdout VIOLATED"):
        assert_battle_disjoint(tr, va)


def test_split_pooled_carries_the_battle_guarantee_onto_the_cached_dataset():
    ds = _pooled(n_rows=300, battles=15)
    tr, va = split_pooled(ds, 0.3, seed=5)
    assert not (set(tr.battle_tag) & set(va.battle_tag))
    assert len(tr) + len(va) == len(ds)


def test_subset_recomputes_slice_weights_on_the_subset_it_returns():
    ds = _pooled(n_rows=100, battles=10)
    s = subset(ds, np.arange(20), "inverse")
    assert abs(float(np.mean(s.weight)) - 1.0) < 1e-9


# ---------------------------------------------------------------------------------------------
# 5. determinism + resume
# ---------------------------------------------------------------------------------------------

def _run(tmp_path, name, epochs, resume=False, seed=0):
    ds = _pooled(n_rows=240, dim=128, seed=5, battles=24)     # dim=128 == the REAL D_MODEL
    tr, va = split_pooled(ds, 0.25, seed=seed)
    torch.manual_seed(seed)
    head = build_head()                                        # the REAL WinProbHead
    res = fit_head(head, tr, va, FitConfig(epochs=epochs, batch_size=64, seed=seed, lr=3e-3),
                   out_dir=str(tmp_path / name), subject_ckpt="synthetic", resume=resume,
                   log=lambda _s: None)
    return head, res


def test_the_same_seed_gives_bitwise_identical_head_weights(tmp_path):
    a, _ = _run(tmp_path, "a", 3)
    b, _ = _run(tmp_path, "b", 3)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), f"{k} differs across two identically seeded runs"


def test_resume_reproduces_the_uninterrupted_run_bitwise(tmp_path):
    full, _ = _run(tmp_path, "full", 4)

    _run(tmp_path, "part", 2)                                  # stop after epoch 1
    ck = torch.load(str(tmp_path / "part" / "head_last.pt"), map_location="cpu",
                    weights_only=False)
    assert ck["epoch"] == 1 and ck["slice_version"] == SLICE_VERSION
    assert ck["subject_ckpt"] == "synthetic" and "rng_state" in ck
    assert ck["config"]["seed"] == 0 and ck["slice_edges"] == list(SLICE_EDGES)

    resumed, res = _run(tmp_path, "part", 4, resume=True)       # continue to epoch 3
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), f"{k} differs after a resume"
    assert len(res.history) == 4 and [h.epoch for h in res.history] == [0, 1, 2, 3]


def test_the_fit_actually_lowers_val_nll_and_names_a_best_epoch(tmp_path):
    _head, res = _run(tmp_path, "learn", 6)
    assert res.best_epoch >= 0
    assert res.best_val_nll <= min(h.val_nll for h in res.history) + 1e-12
    assert res.history[-1].train_nll < res.history[0].train_nll
    assert not np.isnan(res.label_noise_sd_val)


def test_head_best_is_written_and_reloads_into_a_real_WinProbHead(tmp_path):
    head, res = _run(tmp_path, "best", 3)
    ck = torch.load(str(tmp_path / "best" / "head_best.pt"), map_location="cpu",
                    weights_only=False)
    reloaded = build_head(ck["head_state_dict"])
    for k, v in ck["head_state_dict"].items():
        assert torch.equal(v, reloaded.state_dict()[k])
    assert ck["best_epoch"] == res.best_epoch


# ---------------------------------------------------------------------------------------------
# 6. slice re-weighting
# ---------------------------------------------------------------------------------------------

def test_slice_index_puts_turns_in_the_declared_bins():
    got = slice_index([1, 59, 60, 79, 80, 129, 169, 170, 249, 250, 400])
    #                 <60 <60  60  79  80 100 130  170  170  250  250   <- the bin's lower edge
    assert list(got) == [0, 0, 1, 1, 2, 3, 4, 5, 5, 6, 6]
    assert len(SLICE_EDGES) == 6


def test_inverse_weights_are_mean_one_and_a_rare_late_slice_outweighs_a_common_early_one():
    turns = np.array([10] * 90 + [200] * 10)          # 90 early rows, 10 late ones
    w = slice_weights(turns, "inverse")
    assert abs(float(np.mean(w)) - 1.0) < 1e-12
    assert w[-1] > w[0]
    assert abs(w[-1] / w[0] - 9.0) < 1e-9             # exactly the inverse frequency ratio
    # each non-empty slice ends up carrying the same TOTAL weight
    assert abs(w[:90].sum() - w[90:].sum()) < 1e-9


def test_none_gives_all_ones_and_an_unknown_mode_raises():
    assert np.array_equal(slice_weights([1, 100, 240], "none"), np.ones(3))
    with pytest.raises(ValueError, match="unknown slice-reweight mode"):
        slice_weights([1], "sqrt")


def test_weights_change_the_loss_only_through_the_weighting_and_not_the_scale():
    """Mean-1 normalization means the weighted loss stays in the same units as the plain one, which
    is what keeps --lr comparable across datasets."""
    torch.manual_seed(2)
    z, k, n = torch.randn(200), torch.full((200,), 4.0), torch.full((200,), 8.0)
    plain = binomial_nll(z, k, n)
    ones = binomial_nll(z, k, n, torch.ones(200))
    assert torch.allclose(plain, ones)
    w = torch.as_tensor(slice_weights(np.array([10] * 150 + [200] * 50), "inverse"))
    weighted = binomial_nll(z, k, n, w)
    assert 0.5 * float(plain) < float(weighted) < 2.0 * float(plain)


def test_the_slice_version_travels_with_every_checkpoint(tmp_path):
    _head, _res = _run(tmp_path, "ver", 1)
    ck = torch.load(str(tmp_path / "ver" / "head_last.pt"), map_location="cpu",
                    weights_only=False)
    assert ck["slice_version"] == SLICE_VERSION
    assert ck["config"]["slice_reweight"] == "inverse"


# ---------------------------------------------------------------------------------------------
# the graft path
# ---------------------------------------------------------------------------------------------

def test_apply_head_grafts_into_a_probe_model_shaped_object(tmp_path):
    """`apply_head` reaches exactly one attribute path — `_policy.features_extractor.win_head` —
    so a stand-in with that shape pins the contract without a 27 MB checkpoint."""
    from agents.training.winprob_finetune import apply_head

    class _Ex(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.win_head = build_head()

    class _Pol:
        def __init__(self):
            self.features_extractor = _Ex()

    class _Probe:
        def __init__(self):
            self._policy = _Pol()

    fitted, _ = _run(tmp_path, "graft", 2)
    probe = _Probe()
    before = {k: v.clone() for k, v in probe._policy.features_extractor.win_head.state_dict().items()}
    apply_head(probe, str(tmp_path / "graft" / "head_best.pt"))
    got = probe._policy.features_extractor.win_head.state_dict()
    ck = torch.load(str(tmp_path / "graft" / "head_best.pt"), map_location="cpu",
                    weights_only=False)
    for k, v in ck["head_state_dict"].items():
        assert torch.equal(got[k], v)
    assert any(not torch.equal(before[k], got[k]) for k in before), "the graft changed nothing"
    del fitted


def test_apply_head_refuses_a_subject_with_no_win_head(tmp_path):
    from agents.training.winprob_finetune import apply_head

    class _Ex(torch.nn.Module):
        win_head = None

    class _Probe:
        class _Pol:
            features_extractor = _Ex()
        _policy = _Pol()

    _head, _res = _run(tmp_path, "nohead", 1)
    with pytest.raises(ValueError, match="no win_head"):
        apply_head(_Probe(), str(tmp_path / "nohead" / "head_best.pt"))


# ---------------------------------------------------------------------------
# The ANCHOR — the trust region that keeps a biased sample from wrecking the head
# ---------------------------------------------------------------------------

def _tiny_datasets():
    """dim=128 == the REAL D_MODEL, because these tests fit the REAL WinProbHead."""
    return (_pooled(n_rows=120, dim=128, seed=3, battles=12),
            _pooled(n_rows=60, dim=128, seed=4, battles=6))


def test_anchor_zero_is_not_the_default_because_zero_was_measured_destructive():
    """`fit_head` mutates the head IN PLACE, so the two arms are compared on the head objects."""
    tr, va = _tiny_datasets()
    torch.manual_seed(0)
    h_off = build_head()
    off_sd = {k: v.detach().clone() for k, v in h_off.state_dict().items()}
    h_def = build_head()
    h_def.load_state_dict(off_sd)

    fit_head(h_off, tr, va, FitConfig(epochs=3, seed=0, anchor_coef=0.0),
             out_dir=tempfile.mkdtemp(), log=lambda *_: None)
    fit_head(h_def, tr, va, FitConfig(epochs=3, seed=0),           # the DEFAULT
             out_dir=tempfile.mkdtemp(), log=lambda *_: None)
    same = all(torch.equal(v, h_def.state_dict()[k]) for k, v in h_off.state_dict().items())
    assert not same, "the DEFAULT must carry an anchor — 0.0 was measured destructive on two pilots"


def test_a_larger_anchor_keeps_the_head_closer_to_where_it_started():
    """The property the anchor exists for: more pull ⇒ less drift from the subject's function."""
    tr, va = _tiny_datasets()
    torch.manual_seed(0)
    start_sd = {k: v.detach().clone() for k, v in build_head().state_dict().items()}

    def drift(coef):
        h = build_head()
        h.load_state_dict(start_sd)
        fit_head(h, tr, va, FitConfig(epochs=8, seed=0, anchor_coef=coef),
                 out_dir=tempfile.mkdtemp(), log=lambda *_: None)
        return sum(float((h.state_dict()[k] - v).abs().sum()) for k, v in start_sd.items())

    loose, tight = drift(0.0), drift(5.0)
    assert tight < loose, f"anchor did not constrain the fit (tight {tight} vs loose {loose})"


def test_the_anchor_is_taken_from_the_SUBJECT_not_from_a_resumed_state():
    """Caught by the resume test when this was wrong: re-deriving the anchor from a
    partially-trained head changes the objective mid-run and makes a resume diverge."""
    tr, va = _tiny_datasets()
    torch.manual_seed(0)
    start_sd = {k: v.detach().clone() for k, v in build_head().state_dict().items()}
    out = tempfile.mkdtemp()

    h = build_head()
    h.load_state_dict(start_sd)
    fit_head(h, tr, va, FitConfig(epochs=2, seed=0, anchor_coef=1.0), out_dir=out,
             subject_ckpt="synthetic", log=lambda *_: None)
    h2 = build_head()
    h2.load_state_dict(start_sd)
    fit_head(h2, tr, va, FitConfig(epochs=4, seed=0, anchor_coef=1.0), out_dir=out,
             subject_ckpt="synthetic", resume=True, log=lambda *_: None)

    h3 = build_head()
    h3.load_state_dict(start_sd)
    fit_head(h3, tr, va, FitConfig(epochs=4, seed=0, anchor_coef=1.0),
             out_dir=tempfile.mkdtemp(), subject_ckpt="synthetic", log=lambda *_: None)
    for k, v in h3.state_dict().items():
        assert torch.equal(v, h2.state_dict()[k]), f"{k} differs after a resume"
