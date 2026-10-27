"""PIKL(λ), L1's anchored acting rule, ported into ``main.value_search.rules`` (S-U build B2,
``designs/endstate/design_search_u_headroom.md`` §6).

Two halves:
* COMMITTED (always runs): the rule against L1's REGISTERED formula, written out here as the PREREG states it (§3:
  argmax Q(a) + λ·log π(a) over the policy-ordered legal actions, a tie within 1e-6 to the earlier one), on
  synthetic states with ties and near-ties; λ = 0 is ARGMAX; the search's acting rule plays it.
* ARCHIVE (the owner's box): L1's banked rows (``~/gen3ai_archive/search_l1_l5/cache.pkl``) re-scored with this rule at
  the fitted λ = 0.015, TRUNC-4-K8: the held-out pooled IMP and deviation share of E2 and X26 equal
  ``result_l1.json``'s to its rounding, and the Tier-1 fit curve's λ = 0.015 point equals ``fit_l1.json``'s. It skips,
  naming why, when the archive is absent.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from main.value_search import rules
from utils.paths import repo_path

L1_DIR = repo_path("designs/research_state/measurements/search_l1_l5_2026-10-20")
CACHE = Path(os.environ.get("GEN3AI_READS_ARCHIVE", str(Path.home() / "gen3ai_archive"))) / "search_l1_l5" / "cache.pkl"
TIE = 1e-6


def _l1_pick(lam: float, q: np.ndarray, logit: np.ndarray) -> int:
    """The PREREG's formula over POLICY-ORDERED arrays (index 0 = the policy's top choice)."""
    z = logit - logit.max()
    lsm = z - math.log(float(np.exp(z).sum()))
    s = q + lam * lsm
    m = float(np.max(s))
    return int(np.argmax(s >= m - TIE))


def _ours(lam: float, q: np.ndarray, logit: np.ndarray) -> int:
    """This module's rule on the same state, the actions named by their policy-order index."""
    lg = np.asarray(logit, dtype=np.float64)
    legal = list(range(len(lg)))
    order = rules.policy_order(lg, legal)
    assert order == legal, "the fixture's logits must already be in policy order"
    return rules.pick(rules.pikl_scores(lg, {a: float(q[a]) for a in legal}, lam, order), order)


def test_pikl_equals_the_registered_formula_on_synthetic_states():
    rng = np.random.default_rng(20261027)
    n_dev = 0
    for i in range(4000):
        k = int(rng.integers(2, 10))
        logit = np.sort(rng.normal(0.0, 2.0, size=k))[::-1].copy()
        if i % 7 == 0:
            logit[1] = logit[0] - 1e-3                     # a near-tie in the prior
        q = rng.choice(np.arange(17) / 16.0, size=k)       # LEAF-K8-like values: ties are common
        if i % 5 == 0:
            q = rng.uniform(0.0, 1.0, size=k)
        for lam in (0.0, 0.001, 0.015, 0.1, 1.0):
            a, b = _ours(lam, q, logit), _l1_pick(lam, q, logit)
            assert a == b, (i, lam, q.tolist(), logit.tolist())
            n_dev += int(a != 0)
    assert n_dev > 1000          # the fixture exercises deviations, not only the policy's top choice


def test_pikl_lambda_zero_is_argmax_and_a_big_lambda_is_the_policy():
    lg = np.array([2.0, 1.0, 0.0])
    q = {0: 0.3, 1: 0.7, 2: 0.5}
    order = [0, 1, 2]
    assert rules.pick(rules.pikl_scores(lg, q, 0.0, order), order) == 1
    assert rules.pick(rules.pikl_scores(lg, q, 100.0, order), order) == 0
    with pytest.raises(ValueError):
        rules.pikl_scores(lg, q, math.inf, order)
    with pytest.raises(ValueError):
        rules.pikl_scores(lg, q, -0.1, order)


def test_pikl_ties_go_to_the_policy_order():
    lg = np.array([1.0, 1.0 - 1e-12, 0.0])       # 0 and 1 tie in score within TIE
    q = {0: 0.5, 1: 0.5, 2: 0.4}
    assert rules.pick(rules.pikl_scores(lg, q, 0.015, [0, 1, 2]), [0, 1, 2]) == 0


def test_the_registered_lambda_is_l1s_fitted_value():
    fit = json.loads((L1_DIR / "fit_l1.json").read_text())
    assert fit["PIKL"]["TRUNC-4-K8"]["fitted"] == rules.PIKL_LAMBDA_L1 == 0.015


@pytest.fixture(scope="module")
def l1_recs():
    if not CACHE.exists():
        pytest.skip(f"L1's banked cache {CACHE} is absent (the archive half runs on the owner's box)")
    return pickle.loads(CACHE.read_bytes())["recs"]


def _bank_stats(recs, bank: str, lam: float):
    imps, devs = [], []
    for r in recs:
        if r["bank"] != bank or r.get("F") is None:
            continue
        q = np.asarray(r["F"]["X4"], dtype=np.float64).mean(1)
        logit = np.asarray(r["logit"], dtype=np.float64)
        a = _ours(lam, q, logit)
        assert a == _l1_pick(lam, q, logit)
        imps.append(r["Tp"][a] - r["Tp"][0])
        devs.append(float(a != 0))
    return float(np.mean(imps)), float(np.mean(devs)), len(imps)


def test_archive_pikl_rescoring_reproduces_l1s_recorded_cells(l1_recs):
    res = json.loads((L1_DIR / "result_l1.json").read_text())["primary"]["PIKL"]
    for bank in ("e2", "x26"):
        cell = res[bank]["TRUNC-4-K8"]
        assert cell["x"] == rules.PIKL_LAMBDA_L1
        mean, dev, n = _bank_stats(l1_recs, bank, rules.PIKL_LAMBDA_L1)
        assert n == cell["IMP"]["n"]
        assert round(mean, 6) == cell["IMP"]["mean"], (bank, mean)
        assert round(dev, 4) == cell["deviates_from_top1_share"], (bank, dev)


def test_archive_tier1_fit_point_reproduces(l1_recs):
    fit = json.loads((L1_DIR / "fit_l1.json").read_text())["PIKL"]["TRUNC-4-K8"]["tier1_curve"]
    want = next(p for p in fit if p["x"] == rules.PIKL_LAMBDA_L1)
    mean, _dev, n = _bank_stats(l1_recs, "tier1", rules.PIKL_LAMBDA_L1)
    assert n == want["n"]
    assert round(mean, 6) == want["mean"]
