"""`belief_bank_static` == the legacy `belief_bank` losses (`gen3_static_belief_bank_v1`, K8).

The legacy functions ARE the oracle (they stay in the tree as the reference implementation). For every
row: the term, its presence (legacy ``None`` <=> ``present`` False and a 0.0 term), every metric the
legacy function returned (and no other present one), and the gradient w.r.t. every graph input —
float64 to 1e-12 (the SAME arithmetic, summed in a different order), float32 within float rounding.
Inputs: REAL stashes and labels (the K9 learner golden's production-surface learner on its committed
real buffer), plus synthetic edge cases (nothing scored, everything scored, every believed-slot count
k = 0..6, PAD labels, a NaN in a scored vs an unscored slot). A trace test pins `fullgraph=True` on
torch 2.8."""
from __future__ import annotations

import math

import pytest
import torch as th

from agents.training import belief_bank as BB
from agents.training import belief_bank_static as BS

_LEGACY = {r.name: r.loss_fn for r in BB.ROWS}


def _real_args():
    """{row name: argv} from one real forward of the production-surface learner on its golden buffer."""
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
    fe = model.policy.features_extractor
    model.policy.set_training_mode(True)
    model.policy.extract_features(obs)
    params = {"moves_weight": float(model.opp_belief_moves_weight)}
    out = {}
    for row in BB.ROWS:
        out[row.name] = [BB._resolve_arg(src, k, fe, obs, params) for src, k in row.args]
    return out


def _leafify(argv, dtype):
    """Detached leaf copies (float tensors -> ``dtype`` with grad; ints/strings/dicts as given)."""
    leaves = []

    def conv(a):
        if th.is_tensor(a):
            if a.is_floating_point():
                t = a.detach().to(dtype).clone().requires_grad_(True)
                leaves.append(t)
                return t
            return a.detach().clone()
        if isinstance(a, dict):
            return {k: conv(v) for k, v in a.items()}
        return a
    return [conv(a) for a in argv], leaves


def _compare(name, argv, dtype, tol):
    # the legacy functions build some tensors with the DEFAULT dtype (`th.zeros(...)`), so a float64
    # comparison runs under a float64 default — restored on the way out (the torch state guard).
    prev = th.get_default_dtype()
    th.set_default_dtype(dtype)
    try:
        _compare_at(name, argv, dtype, tol)
    finally:
        th.set_default_dtype(prev)


def _compare_at(name, argv, dtype, tol):
    la, l_leaves = _leafify(argv, dtype)
    sa, s_leaves = _leafify(argv, dtype)
    legacy = _LEGACY[name](*la)
    static = BS._STATIC_FNS[name](*sa)
    if legacy is None:
        assert not bool(static.present), f"{name}: legacy None but static present"
        assert float(static.loss) == 0.0
        return
    raw, lm = legacy
    assert bool(static.present), f"{name}: legacy scored but static absent"
    assert math.isclose(float(static.loss), float(raw), rel_tol=tol, abs_tol=tol), \
        (name, float(static.loss), float(raw))
    for k, v in lm.items():
        assert k in static.metrics, (name, k)
        sv, sw = static.metrics[k]
        assert float(sw) == 1.0, (name, k)
        assert math.isclose(float(sv), float(v), rel_tol=max(tol, 1e-6), abs_tol=max(tol, 1e-6)), \
            (name, k, float(sv), float(v))
    extra = {k for k, (_, w) in static.metrics.items() if float(w) == 1.0} - set(lm)
    assert not extra, (name, extra)
    gl = th.autograd.grad(raw, l_leaves, allow_unused=True)
    gs = th.autograd.grad(static.loss, s_leaves, allow_unused=True)
    for a, b in zip(gl, gs):
        if a is None and b is None:
            continue
        a = th.zeros_like(b) if a is None else a
        b = th.zeros_like(a) if b is None else b
        scale = max(1.0, float(a.abs().max()))
        assert float((a - b).abs().max()) <= tol * scale * 10, (name, float((a - b).abs().max()))


@pytest.fixture(scope="module")
def real():
    return _real_args()


@pytest.mark.parametrize("name", [r.name for r in BB.ROWS])
def test_static_equals_legacy_on_REAL_stashes_float64(real, name):
    _compare(name, real[name], th.float64, 1e-12)


@pytest.mark.parametrize("name", [r.name for r in BB.ROWS])
def test_static_equals_legacy_on_REAL_stashes_float32(real, name):
    _compare(name, real[name], th.float32, 2e-6)


# ------------------------------------------------------------------------------- synthetic edges
def _synthetic_hidden(B=96, S=50, M=40, seed=0, scored="mixed"):
    g = th.Generator().manual_seed(seed)
    sp = th.randint(1, S, (B, 6), generator=g)
    mv = th.randint(1, M, (B, 6, 4), generator=g)
    if scored == "none":
        sp[:] = -1
    elif scored == "mixed":                       # every k from 0 to 6, rows cycling
        for b in range(B):
            k = b % 7
            perm = th.randperm(6, generator=g)
            sp[b, perm[k:]] = -1
    mv[sp < 0] = -1
    mv[:, :, 3] = th.where(th.rand(B, 6, generator=g) < 0.4, th.full((B, 6), -1), mv[:, :, 3])
    bl = {"species": th.randn(B, 6, S, generator=g), "moves": th.randn(B, 6, M, generator=g)}
    return [bl, sp, mv, 1.0]


@pytest.mark.parametrize("scored", ["none", "mixed", "all"])
@pytest.mark.parametrize("dtype,tol", [(th.float64, 1e-12), (th.float32, 2e-6)])
def test_hidden_team_every_k_and_nothing_scored(scored, dtype, tol):
    _compare("hidden_team", _synthetic_hidden(scored=scored), dtype, tol)


@pytest.mark.parametrize("mode", ["revealed", "unrevealed", "both"])
@pytest.mark.parametrize("dtype,tol", [(th.float64, 1e-12), (th.float32, 2e-6)])
def test_move_belief_every_mode_and_k(mode, dtype, tol):
    g = th.Generator().manual_seed(1)
    B, M = 70, 40
    ml = th.randn(B, 6, M, generator=g)
    known = th.randint(0, M, (B, 6, 4), generator=g)
    believed = th.randint(0, M, (B, 6, 4), generator=g)
    for b in range(B):
        k = b % 7
        hid = th.randperm(6, generator=g)[:k]
        rev = th.ones(6, dtype=th.bool)
        rev[hid] = False
        known[b, ~rev] = -1
        believed[b, rev] = -1
    known[::5, :, 2:] = -1
    _compare("move_belief", [ml, known, believed, mode], dtype, tol)


@pytest.mark.parametrize("dtype,tol", [(th.float64, 1e-12), (th.float32, 2e-6)])
def test_revealed_heads_with_nothing_and_everything_scored(dtype, tol):
    g = th.Generator().manual_seed(2)
    B = 40
    for frac in (0.0, 0.5, 1.0):
        mask = (th.rand(B, 6, generator=g) < frac).float()
        _compare("spread", [th.randn(B, 6, 5, generator=g) * 50 + 200,
                            th.randn(B, 6, 5, generator=g) * 50 + 200, mask], dtype, tol)
        _compare("hp_type", [th.randn(B, 6, 16, generator=g),
                             th.randint(-1, 16, (B, 6), generator=g), mask], dtype, tol)
        _compare("item", [th.randn(B, 6, 30, generator=g),
                          th.randint(-1, 30, (B, 6), generator=g), mask], dtype, tol)
        evm = (th.rand(B, 6, generator=g) < 0.7).float()
        _compare("nature_ev", [th.randn(B, 6, 25, generator=g), th.rand(B, 6, 5, generator=g) * 252,
                               th.randint(0, 25, (B, 6), generator=g), mask,
                               th.rand(B, 6, 5, generator=g) * 252, evm], dtype, tol)
        km = th.randint(-1, 40, (B, 6, 4), generator=g)
        km[mask < 0.5] = -1
        _compare("move_latent", [th.randn(B, 6, 40, generator=g), th.randn(40, 8, generator=g), km],
                 dtype, tol)


def test_a_NaN_in_a_SCORED_slot_reaches_the_term_and_an_UNSCORED_one_does_not():
    """K9(c) fail-closed: masking must exclude unscored slots BEFORE the reduction (Inf*0 = NaN),
    and must not hide a NaN in a supervised slot."""
    g = th.Generator().manual_seed(3)
    B = 8
    mask = th.zeros(B, 6)
    mask[:, 0] = 1.0
    logits = th.randn(B, 6, 16, generator=g)
    label = th.randint(0, 16, (B, 6), generator=g)
    bad_unscored = logits.clone()
    bad_unscored[:, 3] = float("nan")
    assert math.isfinite(float(BS.hp_type_terms(bad_unscored, label, mask).loss))
    bad_scored = logits.clone()
    bad_scored[2, 0, 5] = float("nan")
    assert math.isnan(float(BS.hp_type_terms(bad_scored, label, mask).loss))


def test_compute_static_folds_in_registry_order_with_coefficients_and_prefixed_metrics(real):
    class _Fe:
        def __init__(self, args):
            self._a = args
    # one row's worth through the real registry walk, against the legacy walk's own term
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
    fe = model.policy.features_extractor
    model.policy.set_training_mode(True)
    with th.no_grad():
        model.policy.extract_features(obs)
    coefs = {"spread_belief_coef": 0.05, "hp_type_belief_coef": 0.05, "item_belief_coef": 0.05}
    gates = {"spread": True, "hp_type": True, "item": True}
    leg = BB.compute(fe, obs, coefs=coefs, gates=gates, site="revealed")
    sta = BS.compute_static(fe, obs, coefs=coefs, gates=gates, site="revealed")
    assert [r.name for r, _, _ in leg] == [t.row.name for t in sta if bool(t.present)]
    for (row, term, mets), st in zip(leg, [t for t in sta if bool(t.present)]):
        assert math.isclose(float(term), float(st.term), rel_tol=2e-6)
        for k, v in mets.items():
            assert row.prefix + k in st.metrics


def test_check_label_vocab_refuses_a_corrupt_buffer():
    import numpy as np
    BS.check_label_vocab({"belief_species": np.array([[1, 2, -1]])}, n_species=400, n_moves=400)
    with pytest.raises(ValueError, match="species max 400"):
        BS.check_label_vocab({"belief_species": np.array([[1, 400]])}, n_species=400, n_moves=400)
    with pytest.raises(ValueError, match="known_moves max 512"):
        BS.check_label_vocab({"known_moves": np.array([[512]])}, n_species=400, n_moves=400)


@pytest.mark.skipif(not th.__version__.startswith("2.8"), reason="fullgraph regions target torch 2.8")
@pytest.mark.parametrize("name", [r.name for r in BB.ROWS])
def test_every_static_row_traces_as_ONE_fullgraph_region(real, name):
    argv = real[name]
    tens = [a for a in argv if th.is_tensor(a)]
    if not tens and not any(isinstance(a, dict) for a in argv):
        pytest.skip(f"{name}: inputs absent on the production surface")
    fn = BS._STATIC_FNS[name]

    def region(*a):
        r = fn(*a)
        return r.loss, r.present, {k: v for k, v in r.metrics.items()}
    th._dynamo.reset()
    try:
        compiled = th.compile(region, fullgraph=True, backend="aot_eager")
        la, leaves = _leafify(argv, th.float32)
        out = compiled(*la)
        ref = region(*_leafify(argv, th.float32)[0])
        assert math.isclose(float(out[0]), float(ref[0]), rel_tol=1e-6, abs_tol=1e-7)
        if leaves and out[0].requires_grad:
            out[0].backward()
    finally:
        th._dynamo.reset()
