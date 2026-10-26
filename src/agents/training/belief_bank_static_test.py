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
        # gen3_label_vocab_raise_v1: a SCORED slot holds a real class, PAD -1 only where unscored (a scored PAD used
        # to be clamped onto class 0 by both implementations and now raises in both)
        hp_lab = th.randint(0, 16, (B, 6), generator=g)
        hp_lab[mask < 0.5] = -1
        it_lab = th.randint(0, 30, (B, 6), generator=g)
        it_lab[mask < 0.5] = -1
        _compare("hp_type", [th.randn(B, 6, 16, generator=g), hp_lab, mask], dtype, tol)
        _compare("item", [th.randn(B, 6, 30, generator=g), it_lab, mask], dtype, tol)
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


# ---- gen3_label_vocab_raise_v1: the item / Hidden Power label loss RAISES, never clamps -------------------------
def _item_inputs(label, mask, n_cls=600):
    g = th.Generator().manual_seed(0)
    logits = th.randn(2, 6, n_cls, generator=g, dtype=th.float64)
    return logits, th.tensor(label, dtype=th.int64), th.tensor(mask, dtype=th.float32)


@pytest.mark.parametrize("fn,n_cls", [(BS.item_terms, 600), (BS.hp_type_terms, 16)])
def test_the_static_twin_raises_on_a_scored_label_outside_its_classes(fn, n_cls):
    pad = [[-1] * 6, [-1] * 6]
    mask = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    ok = [[5, -1, -1, -1, -1, -1], [-1] * 6]
    logits, lab, m = _item_inputs(ok, mask, n_cls)
    t = fn(logits, lab, m)                                   # NON-VACUITY: a valid label scores
    assert bool(t.present) and bool(th.isfinite(t.loss))
    for bad in (n_cls, n_cls + 7, -1, -5):                   # one past the classes (the old max-clamp), a scored PAD (the old min-clamp)
        lab_bad = [[bad, -1, -1, -1, -1, -1], [-1] * 6]
        logits, lab, m = _item_inputs(lab_bad, mask, n_cls)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            fn(logits, lab, m)
    # an UNSCORED slot may hold anything: the mask zeroes it before the CE
    stale = [[5, 10_000, -1, -77, -1, -1], [-1] * 6]
    logits, lab, m = _item_inputs(stale, mask, n_cls)
    assert bool(fn(logits, lab, m).present)
    del pad


@pytest.mark.parametrize("fn,legacy,n_cls", [(BS.item_terms, BB.item_belief_loss, 600),
                                             (BS.hp_type_terms, BB.hp_type_belief_loss, 16)])
def test_the_legacy_and_the_static_loss_agree_on_valid_labels_and_both_raise_on_invalid_ones(fn, legacy, n_cls):
    mask = [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]]
    lab = [[3, n_cls - 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, 2]]
    logits, label, m = _item_inputs(lab, mask, n_cls)
    leg_loss, _ = legacy(logits, label, m)
    assert math.isclose(float(leg_loss), float(fn(logits, label, m).loss), rel_tol=1e-12)
    for bad in (n_cls, -1):
        lab_bad = [[bad, n_cls - 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, 2]]
        logits, label, m = _item_inputs(lab_bad, mask, n_cls)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            legacy(logits, label, m)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            fn(logits, label, m)


def test_check_label_vocab_covers_the_item_and_hidden_power_labels_at_their_scored_slots():
    import numpy as np
    obs = {"item_label": np.array([[5, -1, -1]]), "item_mask": np.array([[1.0, 0.0, 0.0]]),
           "hp_type_label": np.array([[15, -1, -1]]), "hp_type_mask": np.array([[1.0, 0.0, 0.0]])}
    BS.check_label_vocab(obs, n_species=400, n_moves=400, n_items=600, n_hp_types=16)
    # a SCORED item id at the table size, a SCORED PAD, a SCORED hp type of 16: each refused, each naming its family
    for key, mask_key, bad, n, match in (("item_label", "item_mask", 600, 600, "item belief label"),
                                         ("item_label", "item_mask", -1, 600, "item belief label"),
                                         ("hp_type_label", "hp_type_mask", 16, 16, "hidden-power type belief label"),
                                         ("hp_type_label", "hp_type_mask", -1, 16, "hidden-power type belief label")):
        broken = {k: v.copy() for k, v in obs.items()}
        broken[key][0, 0] = bad
        with pytest.raises(ValueError, match=match):
            BS.check_label_vocab(broken, n_species=400, n_moves=400, n_items=600, n_hp_types=16)
    # an UNSCORED slot holding garbage is not read
    stale = {k: v.copy() for k, v in obs.items()}
    stale["item_label"][0, 1] = 10_000
    BS.check_label_vocab(stale, n_species=400, n_moves=400, n_items=600, n_hp_types=16)
    # species / moves below the PAD are corrupt too
    with pytest.raises(ValueError, match="species min -5"):
        BS.check_label_vocab({"belief_species": np.array([[1, -5]])}, n_species=400, n_moves=400)


def test_a_scored_label_outside_its_classes_poisons_the_term_so_a_program_that_does_not_raise_still_fails_closed():
    """Inductor on CPU does NOT raise on a scored `-1` (measured; it does for `>= n_cls`), so the twin also turns the
    slot's CE into NaN: K9(c)'s `check_loss_finite` then names the term. Valid labels are untouched, bit for bit."""
    per = th.arange(6, dtype=th.float64).reshape(2, 3) + 0.5
    lab = th.tensor([[3, -1, 600], [0, 5, -1]])
    mask = th.tensor([[True, True, False], [True, True, False]])        # (0,1) is a SCORED PAD; (0,2) and (1,2) hold garbage but are unscored
    out = BS.scored_label_nan(per, lab, mask, 600)
    assert th.isnan(out[0, 1]) and not th.isnan(out[0, 0]) and not th.isnan(out[1, 0]) and not th.isnan(out[1, 1])
    assert not th.isnan(out[0, 2]) and not th.isnan(out[1, 2]), "an UNSCORED slot's PAD / stale label is never read"
    ok = th.tensor([[3, 7, 599], [0, 5, 1]])
    assert th.equal(BS.scored_label_nan(per, ok, th.ones_like(mask), 600), per), "valid labels are bit-identical"
