"""The fp32 reference arm's bounded saved activations (`saved_offload`, gen3_reference_offload_v1).

THE GUARANTEE: the reference arm's numbers do not move. On the K9 golden's production-surface learner,
`compile_regions._r1_reference` with every saved tensor forced through the host round trip returns the loss,
the log-probs and the flat gradient of the plain arm, bit for bit (one thread; a second plain arm is the
reproducibility control, and the ledger proves the tensors really left and came back). The CUDA cell asks
the same on the card, and that the bounded arm's peak is lower. CUDA's eager backward is NOT run-to-run
bit-stable on these rows (measured 2026-11-11: two plain arms agree on the loss and the log-probs and differ
in the gradient), so the cell runs its compared arms under `torch.use_deterministic_algorithms` and judges
the bounded arm against the MEASURED spread of four plain arms (`judge_offload`): zero spread demands bit
identity; a non-zero spread, a distance inside 4x of it, with 4x-16x INCONCLUSIVE and never a pass.

Then the rule, on toys: what stays where it is (a CPU tensor, a small one, a parameter, an expanded view),
the budget counted per STORAGE in forward order, one copy per tensor and version, a non-dense view refused,
and the scope's ledger.

THE LEDGER'S INVARIANT is per handle (one host copy): `offloaded == returned + never_unpacked + host_live`,
and no host copy outlives the graph (`SO.unsettled`). The card's two readings that broke the old
"returned >= offloaded" (451 out / 407 back; 99 out / 108 back) are reproduced here on toys and on the real
learner: a saved tensor whose node the loss never reaches is never unpacked, and one tensor saved by several
nodes is unpacked by each.
"""
from __future__ import annotations

import contextlib
import itertools
import warnings

import pytest
import torch

from agents.model import saved_offload as SO
from agents.model.extractor_compiles_test import _cuda_skip_reason

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")

MiB = SO.MiB


def _forced(**kw):
    return SO.SavedOffload(force=True, **{"budget_bytes": 0, "min_bytes": 0, **kw})


def _hooks(off):
    return torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack)


def _toy(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 32, generator=g)
    w1 = torch.randn(32, 48, generator=g, requires_grad=True)
    w2 = torch.randn(48, 1, generator=g, requires_grad=True)
    return x, w1, w2


def _toy_grads(off=None):
    x, w1, w2 = _toy()
    ctx = _hooks(off) if off is not None else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
    with ctx:
        h = torch.tanh(x @ w1)
        loss = ((h * h.sigmoid()) @ w2).pow(2).mean() + (h[:, ::2] * h[:, 1::2]).mean()
        return torch.autograd.grad(loss, [w1, w2])


# ── the rule, on toys ─────────────────────────────────────────────────────────────────────────
def test_a_forced_round_trip_leaves_the_gradient_bit_identical_and_really_happened():
    plain = _toy_grads()
    off = _forced()
    got = _toy_grads(off)
    assert all(torch.equal(a, b) for a, b in zip(plain, got))
    assert off.offloaded > 0 and off.returned == off.offloaded and off.kept_bytes == 0
    assert off.unpacks > off.returned, "`h` is saved by tanh, the product and the sigmoid: unpacked by each"
    assert SO.unsettled(off.ledger()) is None
    assert off.refused_layout == 2, "the strided slices `h[:, ::2]`, `h[:, 1::2]` are refused, never re-laid out"


def test_a_tensor_that_is_not_on_a_card_is_never_touched_unless_forced():
    off = SO.SavedOffload(budget_bytes=0, min_bytes=0)             # the production object on a CPU learner
    got = _toy_grads(off)
    assert all(torch.equal(a, b) for a, b in zip(_toy_grads(), got))
    assert (off.offloaded, off.returned, off.kept_bytes, off.refused_layout) == (0, 0, 0, 0)


def test_small_tensors_parameters_and_expanded_views_stay_where_they_are():
    off = _forced(min_bytes=1024)
    small = torch.zeros(8)                                         # 32 bytes < min_bytes
    assert isinstance(off.pack(small), torch.Tensor)
    param = torch.zeros(4096, requires_grad=True)
    assert isinstance(off.pack(param), torch.Tensor)
    expanded = torch.zeros(1, 512).expand(64, 512)                 # 128 KiB of elements on 2 KiB of storage
    assert isinstance(off.pack(expanded), torch.Tensor)
    assert off.offloaded == 0 and off.kept_bytes == 0
    plain = torch.zeros(4096)                                      # a non-parameter leaf of 16 KiB: offloaded
    held = off.pack(plain)
    assert isinstance(held, SO._Hosted) and off.offloaded == 1
    assert torch.equal(off.unpack(held), plain) and off.returned == 1
    assert torch.equal(off.unpack(held), plain) and (off.returned, off.unpacks) == (1, 2)


def _handle_states(off):
    return (off.offloaded, off.returned, off.unpacks, off.never_unpacked, off.host_live)


def test_a_saved_tensor_whose_node_never_runs_is_never_unpacked_and_freed_with_the_graph():
    """The card's first reading (451 out, 407 back): a tensor read only through a `.detach()` was saved by
    a node the loss never reaches. Its host copy waits for the graph and is booked `never_unpacked`."""
    import weakref
    off = _forced()
    x, w1, _ = _toy()
    with _hooks(off):
        h = x @ w1                                                 # MmBackward saves x
        gate = torch.tanh(h)                                       # TanhBackward saves its result ...
        loss = (gate.detach() * h).sum()                           # ... and is not on the loss's graph
        host = weakref.ref(off._seen[id(gate)][2]().host)          # x, gate, and the product's `gate.detach()`
        assert _handle_states(off) == (3, 0, 0, 0, 3)
        loss.backward()
        # the two nodes that ran released their copies; the unreached node's copy still waits
        assert _handle_states(off) == (3, 2, 2, 0, 1) and off.host_live_bytes == gate.numel() * 4
        assert off.unpacks < off.offloaded, "the old `returned >= offloaded` is false here, as on the card"
        assert "outlive the graph" in SO.unsettled(off.ledger())
        assert SO.unsettled(off.ledger(), graph_freed=False) is None
        assert host() is not None
        del gate, loss, h                                          # the graph dies
    assert _handle_states(off) == (3, 2, 2, 1, 0) and off.host_live_bytes == 0
    assert off.never_unpacked_bytes == 64 * 48 * 4
    assert host() is None, "the host copy outlived the graph"
    assert SO.unsettled(off.ledger()) is None
    assert torch.equal(w1.grad, x.t() @ torch.tanh(x @ w1).detach())


def test_a_tensor_saved_by_several_nodes_is_one_handle_unpacked_by_each():
    """The card's second reading (99 out, 108 back): one copy out, one copy back per node that runs."""
    off = _forced()
    x, w1, _ = _toy()
    with _hooks(off):
        h = torch.tanh(x @ w1)
        loss = (h * h.sigmoid() + h.exp()).sum()                   # h: saved by tanh and twice by the product
        assert off.offloaded == 4 and off.host_live == 4           # x, h, sigmoid(h), exp(h)
        loss.backward(retain_graph=True)
        assert (off.returned, off.unpacks, off.host_live) == (4, 6, 4), "h: one handle, three unpacks"
        loss.backward()
        assert (off.returned, off.unpacks, off.never_unpacked) == (4, 12, 0)
        del h, loss
    assert off.unpacks > off.offloaded and off.host_live == 0 and SO.unsettled(off.ledger()) is None


def test_unsettled_names_each_broken_part_of_the_invariant():
    led = {**_forced().ledger(), "offloaded": 5.0, "returned": 3.0, "unpacks": 4.0, "never_unpacked": 2.0}
    assert SO.unsettled(led) is None
    assert "!= returned" in SO.unsettled({**led, "never_unpacked": 1.0})          # a handle unaccounted for
    assert "unpacks for" in SO.unsettled({**led, "unpacks": 2.0})
    live = {**led, "never_unpacked": 1.0, "host_live": 1.0, "host_live_mib": 0.5}
    assert "outlive the graph" in SO.unsettled(live) and SO.unsettled(live, graph_freed=False) is None


def test_the_budget_is_counted_per_storage_in_forward_order():
    a, b, c = torch.ones(1024), torch.ones(1024) * 2, torch.ones(1024) * 3      # 4 KiB each
    off = _forced(budget_bytes=2 * 4096)
    held = [off.pack(t) for t in (a, b, c)]
    assert [isinstance(h, SO._Hosted) for h in held] == [False, False, True]
    assert off.kept_bytes == 2 * 4096 and off.offloaded_bytes == 4096
    # a view of a storage that stays, stays, and is not counted again; a view of one that went, goes
    assert isinstance(off.pack(a.view(32, 32)), torch.Tensor)
    assert isinstance(off.pack(c.view(32, 32)), SO._Hosted)
    assert off.kept_bytes == 2 * 4096
    # a kept tensor is held detached: the graph never owns a tensor that owns the graph
    y = (torch.ones(1024, requires_grad=True) * 2)
    kept = _forced(budget_bytes=1 << 30).pack(y)
    assert kept.grad_fn is None and kept.untyped_storage()._cdata == y.untyped_storage()._cdata


def test_one_copy_per_tensor_and_per_version():
    off = _forced()
    t = torch.arange(1024, dtype=torch.float32)
    first = off.pack(t)
    assert off.pack(t) is first and off.offloaded == 1            # saved by a second node: the same copy
    t.add_(1)                                                      # a new version is a different value
    second = off.pack(t)
    assert second is not first and off.offloaded == 2
    assert torch.equal(off.unpack(first) + 1, off.unpack(second))
    # the graph alone owns a handle: when every node that saved the tensor is gone, so is the host copy,
    # and a later save of the same tensor is a new one
    assert off.host_live == 2
    del first, second
    assert (off.host_live, off.host_live_bytes, off.never_unpacked) == (0, 0, 0)
    third = off.pack(t)
    assert isinstance(third, SO._Hosted) and (off.offloaded, off.host_live) == (3, 1)


def test_a_copy_whose_layout_differs_is_refused_and_the_original_kept():
    off = _forced()
    base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    strided = base[:, ::2]                                         # not dense in its storage
    held = off.pack(strided)
    assert isinstance(held, torch.Tensor) and held.stride() == strided.stride()
    assert (off.refused_layout, off.refused_bytes, off.offloaded) == (1, 128, 0)
    transposed = base.t()                                          # dense: the copy keeps its strides
    held = off.pack(transposed)
    assert isinstance(held, SO._Hosted) and held.host.stride() == transposed.stride()


def test_the_scope_is_fp32_scope_plus_the_hooks_and_publishes_its_ledger():
    from agents.model import trunk_precision as TP
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    before = [type(m) for m in model.policy.modules()]
    assert TP.trunk_precision_of(model.policy) == "bf16"
    with SO.fp32_reference_scope(model.policy, _forced()) as off:
        assert TP.trunk_precision_of(model.policy) == "fp32"
        x = torch.randn(256, 8)
        w = torch.randn(8, 8, requires_grad=True)
        (torch.tanh(x @ w) ** 2).sum().backward()
        assert off.offloaded > 0 and off._seen
    assert [type(m) for m in model.policy.modules()] == before
    assert not off._seen, "the scope still holds the arm's tensors"
    assert SO.LAST == off.ledger() and SO.LAST["offloaded"] > 0 and SO.LAST["returned"] > 0
    assert SO.unsettled(SO.LAST) is None, "a toy graph is gone at scope exit: nothing may wait on the host"
    assert set(SO.LAST) == {"kept_mib", "offloaded_mib", "offloaded", "returned", "unpacks", "never_unpacked",
                            "never_unpacked_mib", "host_live", "host_live_mib", "refused_layout",
                            "refused_mib", "seconds", "budget_mib"}


def test_the_reference_arm_is_the_one_the_gate_and_the_canary_call_through_the_scope():
    import inspect

    from agents.model import compile_regions as cr
    src = inspect.getsource(cr._r1_reference)
    assert "with fp32_reference_scope(model.policy):" in src and "fp32_scope(" not in src
    assert SO.DEVICE_BUDGET_BYTES == 2048 * MiB and SO.OFFLOAD_MIN_BYTES == MiB


# ── the guarantee, on the real learner ────────────────────────────────────────────────────────
def _reference(model, args, monkeypatch, made=None, **forced):
    """One reference arm -> (its numbers, its ledger AT SCOPE EXIT). ``made`` collects the arm's
    `SavedOffload`, whose ledger can be read again once the model's next forward dropped the last of the
    graph (`settled_after_the_next_forward`)."""
    from agents.model import compile_regions as cr
    real = SO.SavedOffload

    def build():
        off = real(**forced)
        if made is not None:
            made.append(off)
        return off

    with monkeypatch.context() as mp:
        mp.setattr(SO, "SavedOffload", build)
        out = cr._r1_reference(model, args)
    return out, dict(SO.LAST)


def settled_after_the_next_forward(offs):
    """Each arm's ledger now that a LATER arm's forward has replaced the extractor's stashes — so the
    caller runs one more arm after the last one judged. Returns {index: what is unsettled}; {} is the pass."""
    return {i: bad for i, bad in ((i, SO.unsettled(off.ledger())) for i, off in enumerate(offs)) if bad}


def _same(a, b):
    return {k: bool(torch.equal(a[k], b[k])) for k in ("loss", "log_prob", "grad", "grad_sizes")}


def test_on_the_real_learner_the_offloaded_reference_arm_is_bit_identical(monkeypatch, restore_torch_globals):
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    torch.set_num_threads(1)                       # a threaded reduction is not run-to-run bit-stable
    torch.set_float32_matmul_precision("highest")
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model.policy.set_training_mode(True)
    args = cr._r1_args(model, cr.r1_batch(model, 64, 0))
    plain, led0 = _reference(model, args, monkeypatch)
    again, _ = _reference(model, args, monkeypatch)
    assert all(_same(plain, again).values()), "the control: two plain arms differ, so nothing below means anything"
    assert led0["offloaded"] == 0 and led0["returned"] == 0, "a CPU learner's reference arm must be untouched"
    assert float(plain["grad"].abs().sum()) > 0
    offs: list = []
    full, led = _reference(model, args, monkeypatch, offs, force=True, budget_bytes=0, min_bytes=0)
    assert _same(plain, full) == {"loss": True, "log_prob": True, "grad": True, "grad_sizes": True}
    assert led["offloaded"] > 100 and led["returned"] > 100 and led["kept_mib"] == 0
    # both of the card's readings, on the real learner: handles never unpacked (nodes the loss does not
    # reach), and more copies back than handles returned (a tensor saved by several nodes)
    assert led["never_unpacked"] > 0 and led["returned"] < led["offloaded"]
    assert led["unpacks"] > led["returned"]
    assert SO.unsettled(led, graph_freed=False) is None
    # at scope exit the extractor's stashes still hold a little of the graph: its copies are live, counted
    assert 0 < led["host_live"] <= 8 and led["host_live_mib"] < 0.01 * led["offloaded_mib"]
    # a budget in the middle: some storages stay, the rest go, and the numbers are still the plain arm's
    half_bytes = int(led["offloaded_mib"] * MiB / 2)
    half, led2 = _reference(model, args, monkeypatch, offs, force=True, budget_bytes=half_bytes, min_bytes=0)
    assert all(_same(plain, half).values())
    assert 0 < led2["kept_mib"] * MiB <= half_bytes and 0 < led2["offloaded"] < led["offloaded"]
    # the production thresholds, forced: only tensors of >= 1 MiB are candidates
    prod, led3 = _reference(model, args, monkeypatch, offs, force=True, budget_bytes=0)
    assert all(_same(plain, prod).values()) and led3["offloaded"] < led["offloaded"]
    assert all(SO.unsettled(x, graph_freed=False) is None for x in (led2, led3))
    # NOTHING ACCUMULATES: an earlier arm's last host copies went when the next forward replaced the
    # stashes, and the last arm's go with one more forward
    assert [off.host_live for off in offs[:2]] == [0, 0]
    _reference(model, args, monkeypatch)
    assert len(offs) == 3 and settled_after_the_next_forward(offs) == {}
    assert sum(off.host_live_bytes for off in offs) == 0


# ── the card's cell: judged against the measured control spread ───────────────────────────────
KEYS = ("loss", "log_prob", "grad", "grad_sizes")
#: The spread rule's two bars, in multiples of the control spread (the largest pairwise distance among the
#: plain arms). A byte copy adds nothing, so the bounded arm is one more draw from the plain arms' own
#: distribution: its distance to a plain arm is a relative L2 norm over every gradient element, which
#: concentrates (measured spread between pairs: a few percent), so 4x is never reached by a healthy arm,
#: and a wrong operand (a stale version, a re-laid-out view) moves the gradient by orders of magnitude
#: more than 16x a rounding-order spread. Between the bars the cell refuses to decide (root rule 8).
PASS_WITHIN, FAIL_BEYOND = 4.0, 16.0
BIT, WITHIN, DIFFERS, INCONCLUSIVE = "bit-identical", "within the control spread", "DIFFERS", "INCONCLUSIVE"


def distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """0.0 iff the two tensors are equal element for element; else ||a - b|| / ||a|| in float64, and
    infinite where that is not a positive number (a NaN, a zero norm), so "not equal" is never 0."""
    if torch.equal(a, b):
        return 0.0
    a64, b64 = a.detach().double().cpu(), b.detach().double().cpu()
    d = float((a64 - b64).norm() / a64.norm())
    return d if d > 0.0 else float("inf")


def spread_verdict(spread: float, d: float) -> str:
    """One field's verdict from the control spread and the judged arm's distance. Deterministic at the
    declared bars: zero spread (the plain arms are bit-stable) demands bit identity."""
    if d == 0.0:
        return BIT
    if spread == 0.0 or d >= FAIL_BEYOND * spread:
        return DIFFERS
    return WITHIN if d <= PASS_WITHIN * spread else INCONCLUSIVE


def judge_offload(arm, judged, n_control: int = 4):
    """``arm(**forced) -> (out, ledger, peak)``. Runs ``n_control`` plain (unbounded) arms, then each of
    ``judged`` (name -> the SavedOffload kwargs); returns the plain arms' ledgers, the per-field control
    spread, and per judged arm its (verdicts by field, distances by field, ledger, peak)."""
    plains = [arm(**UNBOUNDED) for _ in range(n_control)]
    spread = {k: max(distance(a[0][k], b[0][k]) for a, b in itertools.combinations(plains, 2)) for k in KEYS}
    ref = plains[0][0]
    out = {}
    for name, kw in judged.items():
        got, led, peak = arm(**kw)
        dist = {k: distance(ref[k], got[k]) for k in KEYS}
        out[name] = ({k: spread_verdict(spread[k], dist[k]) for k in KEYS}, dist, led, peak)
    return [p[1] for p in plains], spread, out


UNBOUNDED = dict(budget_bytes=1 << 60)


def ledgers_settle(bounded_leds, offs, one_more_arm):
    """The cell's ledger rule -> {what: why it fails}; {} is the pass. Each bounded arm's ledger at scope
    exit really offloaded and returned, and accounts for every handle (`graph_freed=False`: the extractor's
    stashes may still hold a few); then, after ``one_more_arm()`` has replaced those stashes, every arm's
    `SavedOffload` in ``offs`` is fully settled: no host copy outlives its graph."""
    bad = {}
    for i, led in enumerate(bounded_leds):
        why = SO.unsettled(led, graph_freed=False)
        if not (led["offloaded"] > 0 and led["returned"] > 0):
            why = f"nothing offloaded or nothing returned: {led}"
        if why:
            bad[f"bounded ledger {i} at scope exit"] = why
    one_more_arm()
    bad.update({f"arm {i} after the next forward": why
                for i, why in settled_after_the_next_forward(offs[:-1]).items()})
    return bad


@contextlib.contextmanager
def deterministic_algorithms(monkeypatch):
    """`torch.use_deterministic_algorithms(True, warn_only=True)` with the cuBLAS workspace it needs, both
    restored on exit. Yields the list of ops that announced they have NO deterministic implementation
    (warn-only: they still run, and the control spread — not this list — decides the rule)."""
    was = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    undetermined: list = []
    with monkeypatch.context() as mp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mp.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield undetermined
        finally:
            torch.use_deterministic_algorithms(was, warn_only=was_warn)
            undetermined.extend(sorted({str(w.message).split(" does not have a deterministic")[0]
                                        for w in caught if "deterministic implementation" in str(w.message)}))


def test_the_spread_rule_is_deterministic_at_its_declared_bars():
    assert spread_verdict(0.0, 0.0) == BIT and spread_verdict(1e-7, 0.0) == BIT
    assert spread_verdict(0.0, 1e-12) == DIFFERS, "bit-stable plain arms demand bit identity"
    s = 1e-7
    assert [spread_verdict(s, m * s) for m in (0.5, 1.0, 4.0)] == [WITHIN] * 3
    assert [spread_verdict(s, m * s) for m in (4.5, 8.0, 15.0)] == [INCONCLUSIVE] * 3
    assert [spread_verdict(s, m * s) for m in (16.0, 1e3, float("inf"))] == [DIFFERS] * 3
    a = torch.tensor([3.0, 4.0])
    assert distance(a, a.clone()) == 0.0
    assert distance(a, torch.tensor([3.0, 4.5])) == pytest.approx(0.1, rel=1e-12)       # 0.5 / |(3, 4)|
    assert distance(a, torch.tensor([3.0, float("nan")])) == float("inf")
    assert distance(torch.zeros(2), torch.tensor([0.0, 1e-30])) == float("inf")


def test_the_deterministic_scope_restores_the_mode_and_the_environment(monkeypatch):
    import os
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert not torch.are_deterministic_algorithms_enabled()
    with deterministic_algorithms(monkeypatch) as undetermined:
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        torch.zeros(4).put_(torch.tensor([0, 1]), torch.ones(2))      # no deterministic implementation: warns
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert len(undetermined) == 1 and "put_" in undetermined[0], undetermined


def test_the_cells_judgement_runs_on_the_real_learner_and_catches_a_moved_number(monkeypatch,
                                                                                 restore_torch_globals):
    """The CUDA cell's whole flow on the CPU (the offload forced, one thread, so the control spread is 0
    and the rule is bit identity): the bounded arms pass, under deterministic algorithms as the card runs
    them — and an arm whose gradient moved by one part in 1e6 is DIFFERS, so the cell can fail."""
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model.policy.set_training_mode(True)
    args = cr._r1_args(model, cr.r1_batch(model, 64, 0))
    tamper = {"on": False}

    offs: list = []

    def arm(**forced):
        out, led = _reference(model, args, monkeypatch, offs, force=True, **forced)
        if tamper["on"] and forced != UNBOUNDED:
            out = {**out, "grad": out["grad"] * (1.0 + 1e-6)}
        return out, led, None

    judged = {"nothing kept": dict(budget_bytes=0, min_bytes=0), "default budget": {}}
    with deterministic_algorithms(monkeypatch):
        plain_leds, spread, got = judge_offload(arm, judged)
        tamper["on"] = True
        _, _, moved = judge_offload(arm, {"nothing kept": dict(budget_bytes=0, min_bytes=0)}, n_control=2)
    assert all(led["offloaded"] == 0 for led in plain_leds)
    assert spread == dict.fromkeys(KEYS, 0.0), spread
    for name in judged:
        assert got[name][0] == dict.fromkeys(KEYS, BIT), (name, got[name][:2])
    assert got["nothing kept"][2]["offloaded"] > 100
    # 64 rows save less than the default budget, so only the arm with nothing kept has a ledger to judge
    assert got["default budget"][2]["offloaded"] == 0
    assert ledgers_settle([got["nothing kept"][2]], offs, lambda: arm(**UNBOUNDED)) == {}
    assert sum(off.offloaded for off in offs) > 200 and sum(off.host_live for off in offs[:-1]) == 0
    assert moved["nothing kept"][0]["grad"] == DIFFERS and moved["nothing kept"][0]["loss"] == BIT


@_skip_cuda
@pytest.mark.slow
def test_on_the_card_the_bounded_reference_arm_matches_the_unbounded_one_and_peaks_lower(monkeypatch,
                                                                                         restore_torch_globals):
    """CUDA: the production object (no forcing). Under deterministic algorithms, four plain arms are the
    control and give each field's spread; the arm with nothing kept on the card and the arm at the
    default budget are judged against it (`judge_offload`: zero spread demands bit identity, else within
    4x of it; 4x-16x is INCONCLUSIVE and fails as such). Then the same three arms on the PRODUCTION
    kernels (deterministic mode off) for the peaks and the ledger the run will see."""
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    model.policy.set_training_mode(True)
    args = cr._r1_args(model, cr.r1_batch(model, 2048, 0))

    offs: list = []

    def arm(**forced):
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        out, led = _reference(model, args, monkeypatch, offs, **forced)
        torch.cuda.synchronize(dev)
        return out, led, torch.cuda.max_memory_allocated(dev) / MiB

    judged = {"nothing kept": dict(budget_bytes=0), "default budget": {}}
    with deterministic_algorithms(monkeypatch) as undetermined:
        plain_leds, spread, got = judge_offload(arm, judged)
    _, _, peak_p = arm(**UNBOUNDED)
    _, led_b, peak_b = arm(budget_bytes=0)
    _, led_d, peak_d = arm()
    print(f"[saved_offload] B=2048 control spread over 4 plain arms (deterministic algorithms; ops with no "
          f"deterministic implementation: {undetermined or 'none'}): {spread}")
    for name, (verdicts, dist, led, _peak) in got.items():
        print(f"[saved_offload] {name}: {verdicts} distances {dist} ledger {led}")
    print(f"[saved_offload] B=2048 peak MiB on the production kernels: unbounded {peak_p:.0f}, default budget "
          f"{peak_d:.0f} (ledger {led_d}), nothing kept {peak_b:.0f} (ledger {led_b})")
    assert all(led["offloaded"] == 0 for led in plain_leds)
    for name, (verdicts, dist, _led, _peak) in got.items():
        bad = {k: (v, dist[k], spread[k]) for k, v in verdicts.items() if v not in (BIT, WITHIN)}
        assert not bad, f"{name}: field -> (verdict, distance, control spread) {bad}"
    # the ledger, per handle: every host copy returned, never unpacked or (at scope exit only) still held by
    # the extractor's stashes; none left once one more arm's forward has replaced them
    bounded = [got["nothing kept"][2], got["default budget"][2], led_b, led_d]
    assert ledgers_settle(bounded, offs, lambda: arm(**UNBOUNDED)) == {}
    assert peak_b < peak_p - 0.25 * led_b["offloaded_mib"], (peak_b, peak_p, led_b)
