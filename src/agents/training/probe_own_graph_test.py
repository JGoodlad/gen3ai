"""The gradient probes on their OWN graphs (gen3_probe_own_graph_v1; `instrumented_ppo/probe_forward.py`).

THE GUARANTEE, on the K9 learner golden: the golden update with every probe ON (the first update of a
process always is) and the same update with them all SKIPPED both reproduce the COMMITTED golden —
recorded while the probes still read the training step's retained graph — and leave identical parameter
and group hashes, optimizer state, pinned losses and torch / numpy / python RNG streams. The ON arm must
really have re-run R1 (`ProbeForward.calls`) and emitted the probes' tags, so the equality is not two arms
that never differed.

THE VALUES: on one thread on the CPU, each probe's own-graph reading equals the reading off the retained
graph to the bit — the grad-balance metrics and the per-term noise sampler's squared norms and accumulators.
The CUDA cell asks the same on the card to a tolerance (two forwards there need not be bit-equal), with the
retained read taken twice as its control.

THE RULES: a re-run that registers different terms is a `ProbeGraphError`; a re-run draws from no RNG
stream; a probe writes no `.grad`; and `train()` takes every probe gradient after the micro-batch's
`backward()` and before the optimizer step, with nothing retained.
"""
from __future__ import annotations

import inspect
import random
import re

import numpy as np
import pytest
import torch as th

from agents.model.extractor_compiles_test import _cuda_skip_reason
from agents.training import grad_balance as GB
from agents.training.instrumented_ppo import diagnostics_cadence as DC
from agents.training.instrumented_ppo import noise_scale_terms as NS
from agents.training.instrumented_ppo import probe_forward as PF

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")

PROBE_PREFIX = "grad/"


def _learner():
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _rng_state():
    return th.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _same_rng(a, b) -> bool:
    return (th.equal(a[0], b[0]) and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and a[2] == b[2])


def _opt_tensors(model):
    state = model.policy.optimizer.state_dict()["state"]
    return {(i, k): v.detach().clone() for i, st in state.items() for k, v in st.items() if th.is_tensor(v)}


def _golden_arm(monkeypatch, *, skip: bool):
    """One golden update (`learner_golden.check`: the pinned buffer, the pinned seeding, one `train()`),
    with the probes on (a fresh process's first update) or all skipped."""
    from agents.training import learner_golden as LG
    calls = {"n": 0}
    real = PF.ProbeForward.__call__

    def counted(self):
        calls["n"] += 1
        return real(self)

    model = LG.build_learner()
    if skip:
        model.diagnostics_every = 2 ** 31 - 1
        model._diagnostics_ran_in_process = True
        model.rank_probe_every_update = False
        model.noise_terms_every_update = False
    random.seed(LG.UPDATE_SEED)                      # `check` seeds numpy and torch itself
    with monkeypatch.context() as mp:
        mp.setattr(PF.ProbeForward, "__call__", counted)
        if skip:                                     # a fresh learner sits at rollout 0, which every cadence divides
            mp.setattr(DC, "rollout_index", lambda _model: 1)
        diffs, fingerprint = LG.check(model)
    return {"diffs": diffs, "fingerprint": fingerprint, "opt": _opt_tensors(model), "rng": _rng_state(),
            "logged": dict(model.logger.name_to_value), "reruns": calls["n"],
            "accum": max(1, int(getattr(model, "grad_accum_steps", 1)))}


@pytest.mark.slow  # two golden updates (build + train) on one thread: ~40 s on CPU
def test_the_golden_update_is_bit_identical_with_the_probes_on_their_own_graphs_and_with_them_skipped(
        monkeypatch, restore_torch_globals):
    on = _golden_arm(monkeypatch, skip=False)
    off = _golden_arm(monkeypatch, skip=True)
    # the ON arm is not vacuous: the probes ran, through their own forwards
    assert any(k.startswith(PROBE_PREFIX) for k in on["logged"]), "no grad/* tag: the grad-balance probe did not run"
    assert on["reruns"] >= 2, on["reruns"]
    if on["accum"] >= 2:
        assert on["reruns"] > 2 + on["accum"], "the per-term noise probe re-ran nothing"
        assert "train/noise_per_term_ms" in on["logged"]
    assert off["reruns"] == 0 and not any(k.startswith(PROBE_PREFIX) for k in off["logged"])
    # both arms ARE the committed golden (parameters, per-group bytes, every pinned loss) ...
    assert on["diffs"] == [], on["diffs"]
    assert off["diffs"] == [], off["diffs"]
    assert on["fingerprint"] == off["fingerprint"]
    # ... and the optimizer state and the three RNG streams are the same with the probes on and skipped
    assert on["opt"].keys() == off["opt"].keys() and on["opt"]
    for k in on["opt"]:
        assert th.equal(on["opt"][k], off["opt"][k]), f"a probe changed optimizer state {k}"
    assert _same_rng(on["rng"], off["rng"]), "a probe drew from an RNG stream"


# ── the values: own graph == retained graph ───────────────────────────────────────────────────
def _micro(model, rows: int = 64):
    """R1's arguments on the golden's rows, its eager body, the real forward and its presence read."""
    from agents.model import compile_regions as cr
    from agents.training.instrumented_ppo.micro_step import micro_step
    model.policy.set_training_mode(True)
    args = cr._r1_args(model, cr.r1_batch(model, rows, 0))
    st = args[8]
    coefs = {"ent_coef": st.ent_coef, "vf_coef": st.vf_coef, "critic_winprob": st.critic_winprob}
    mo = micro_step(*args)
    present = {k: bool(v) for k, v in mo.present.items()}
    return micro_step, args, coefs, mo, present


def _retained(terms, shared):
    policy_term, value_term, aux = terms
    return GB.grad_balance_metrics(policy_term, value_term, shared, aux)


def _balance_pair(model):
    step, args, coefs, mo, present = _micro(model)
    shared = GB.shared_trunk_parameters(model.policy.features_extractor)
    assert shared
    retained = _retained(PF.balance_terms(mo, present, **coefs), shared)
    again = _retained(PF.balance_terms(step(*args), present, **coefs), shared)
    pf = PF.ProbeForward(step, args, present, **coefs)
    fresh = GB.grad_balance_metrics_fresh(pf.balance_terms, shared)
    return retained, again, fresh, pf


def test_the_grad_balance_probe_reads_the_retained_graphs_values_from_its_own_forwards(restore_torch_globals):
    th.set_num_threads(1)
    retained, again, fresh, pf = _balance_pair(_learner())
    assert retained == again, "the control: two retained reads differ, so nothing below means anything"
    assert "grad/policy_value_cosine" in retained and len(retained) >= 6, sorted(retained)
    n_aux = sum(1 for k in retained if k.endswith("_policy_cosine"))
    assert n_aux >= 1, f"the golden learner registers no auxiliary term: {sorted(retained)}"
    assert all(np.isfinite(v) for v in retained.values())
    assert any(v != 0.0 for v in retained.values())
    assert fresh == retained                                       # every key, every float, to the bit
    assert pf.calls == 2 + n_aux                                   # one forward per gradient, no more


def _sampler_state(s):
    return ({g: float(v) for g, v in s.small_sq.items() if v is not None},
            {g: [None if t is None else t.clone() for t in buf] for g, buf in s._accum.items()})


def test_the_per_term_noise_probe_reads_the_retained_graphs_values_and_writes_no_grad(restore_torch_globals):
    th.set_num_threads(1)
    model = _learner()
    step, args, coefs, mo, present = _micro(model)
    params = list(model.policy.parameters())
    # the reference: each group's gradient off the forward's own retained graph, before any backward
    ref = NS.PerTermNoiseSampler(params)
    NS.tag_terms(ref, mo.terms, mo.term_groups, present)
    ref.flush_micro()
    ref_small, ref_accum = _sampler_state(ref)
    assert set(ref_small) >= {"policy", "entropy"} and all(v > 0 for v in ref_small.values()), ref_small
    # the learner's order: tag, the real backward (its graph is gone), then the probe on its own forwards
    for p in params:
        p.grad = None
    own = NS.PerTermNoiseSampler(params)
    mo2 = step(*args)
    NS.tag_terms(own, mo2.terms, mo2.term_groups, present)
    mo2.loss.backward()
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    pf = PF.ProbeForward(step, args, present, **coefs)
    own.flush_micro(pf.grouped_terms)
    own_small, own_accum = _sampler_state(own)
    assert pf.calls == len(ref_small) and own.micros == 1 and own.probe_seconds > 0
    assert own_small == ref_small
    assert own_accum.keys() == ref_accum.keys()
    for g in ref_accum:
        for a, b in zip(ref_accum[g], own_accum[g]):
            assert (a is None and b is None) or th.equal(a, b), g
    for p, g in zip(params, grads):                                # the probe wrote no `.grad`
        assert (p.grad is None and g is None) or th.equal(p.grad, g)
    for p in params:
        p.grad = None


@_skip_cuda
@pytest.mark.slow
def test_on_the_card_the_probes_own_graph_readings_match_the_retained_graphs(restore_torch_globals):
    """CUDA: the same comparison to a tolerance far from either side's noise (two forwards on a card need not
    be bit-equal). Keys whose retained value is within 1e-6 of zero are excluded, not tolerated."""
    model = _learner()
    dev = th.device("cuda")
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    retained, again, fresh, pf = _balance_pair(model)
    keys = [k for k, v in retained.items() if abs(v) > 1e-6]
    assert len(keys) >= 6 and fresh.keys() == retained.keys()

    def worst(a, b):
        return max(abs(a[k] - b[k]) / abs(b[k]) for k in keys)

    control, got = worst(again, retained), worst(fresh, retained)
    print(f"[probe_own_graph] cuda grad-balance: {len(keys)} keys, retained-vs-retained max rel {control:.3g}, "
          f"own-vs-retained max rel {got:.3g}, bit-equal {fresh == retained}")
    assert control < 1e-3, f"the CONTROL failed: two retained reads differ by {control:.3g} on this card"
    assert got < 1e-3, got


# ── the rules ─────────────────────────────────────────────────────────────────────────────────
def _toy():
    w = th.nn.Parameter(th.tensor([1.0, -2.0, 0.5]))
    x = th.tensor([0.3, 0.7, -1.1])

    def terms():
        h = (w * x).tanh()
        return h.sum(), (h ** 2).sum(), {"belief": (h * x).sum(), "win_prob": h.exp().sum()}
    return w, terms


def test_a_re_run_that_registers_different_terms_is_a_ProbeGraphError():
    w, terms = _toy()
    calls = {"n": 0}

    def shrinking():
        calls["n"] += 1
        p, v, aux = terms()
        return p, v, (aux if calls["n"] == 1 else {"belief": aux["belief"]})
    with pytest.raises(GB.ProbeGraphError, match="win_prob"):
        GB.grad_balance_metrics_fresh(shrinking, [w])
    assert GB.grad_balance_metrics_fresh(terms, [w]) == _retained(terms(), [w])
    sampler = NS.PerTermNoiseSampler([w])
    sampler.add("policy", terms()[0])
    with pytest.raises(GB.ProbeGraphError, match="policy"):
        sampler.flush_micro(lambda: {})
    assert not hasattr(sampler, "failed") and sampler.micros == 0


def test_a_re_run_draws_from_no_rng_stream():
    w, _ = _toy()

    def region(*_args):
        return th.rand(3) * w                                      # a forward that DOES draw

    pf = PF.ProbeForward(region, (th.nn.Linear(1, 1),), {}, ent_coef=0.0, vf_coef=0.5, critic_winprob=False)
    th.manual_seed(11)
    before = th.get_rng_state().clone()
    first, second = pf(), pf()
    assert th.equal(th.get_rng_state(), before), "the probe's forward moved the global torch stream"
    assert th.equal(first, second) and pf.calls == 2


def test_the_terms_rule_is_one_rule_for_the_real_forward_and_every_re_run():
    terms = {"policy": th.ones((), requires_grad=True), "entropy": th.ones((), requires_grad=True),
             "belief": th.ones((), requires_grad=True), "win_prob": th.ones(())}
    groups = {"policy": "policy", "entropy": "entropy", "belief": "aux", "win_prob": "aux"}
    got = NS.grouped_terms(terms, groups, {"belief": False, "win_prob": True})
    assert {g: len(v) for g, v in got.items()} == {"policy": 1, "entropy": 1}     # unscored / not on a graph
    sampler = NS.PerTermNoiseSampler([th.nn.Parameter(th.ones(1))])
    NS.tag_terms(sampler, terms, groups, {"belief": False, "win_prob": True})
    assert {g: len(v) for g, v in sampler._pending.items()} == {g: len(v) for g, v in got.items()}


def test_train_takes_every_probe_gradient_after_the_backward_and_before_the_step_with_nothing_retained():
    from agents.training.instrumented_ppo import ppo
    src = inspect.getsource(ppo.InstrumentedMaskablePPO.train)
    order = [src.index(s) for s in (
        "_mo = self._micro_region(_retains)(*_r1_args)",
        "(loss / accum).backward()",
        "_ProbeForward(self._micro_region(True), _r1_args, _mpres,",
        "grad_balance = grad_balance_metrics_fresh(_probe_fw.balance_terms, shared_trunk)",
        "_ntg.flush_micro(_probe_fw.grouped_terms)",
        "if micro_in_group == accum:")]
    assert order == sorted(order), order
    assert "retain_graph" not in "".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"(?<![\w])grad_balance_metrics\(", src) and "flush_micro()" not in src
    assert inspect.signature(GB.grad_balance_metrics_fresh).parameters["retain_graph"].default is False
    assert "except" not in inspect.getsource(NS.PerTermNoiseSampler.flush_micro)
