"""The fp32 reference arm's bounded saved activations (`saved_offload`, gen3_reference_offload_v1).

THE GUARANTEE: the reference arm's numbers do not move. On the K9 golden's production-surface learner,
`compile_regions._r1_reference` with every saved tensor forced through the host round trip returns the loss,
the log-probs and the flat gradient of the plain arm, bit for bit (one thread; a second plain arm is the
reproducibility control, and the ledger proves the tensors really left and came back). The CUDA cell asks
the same on the card, and that the bounded arm's peak is lower.

Then the rule, on toys: what stays where it is (a CPU tensor, a small one, a parameter, an expanded view),
the budget counted per STORAGE in forward order, one copy per tensor and version, a non-dense view refused,
and the scope's ledger.
"""
from __future__ import annotations

import functools

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
    assert off.offloaded > 0 and off.returned >= off.offloaded and off.kept_bytes == 0
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
    assert set(SO.LAST) == {"kept_mib", "offloaded_mib", "offloaded", "returned", "refused_layout",
                            "refused_mib", "seconds", "budget_mib"}


def test_the_reference_arm_is_the_one_the_gate_and_the_canary_call_through_the_scope():
    import inspect

    from agents.model import compile_regions as cr
    src = inspect.getsource(cr._r1_reference)
    assert "with fp32_reference_scope(model.policy):" in src and "fp32_scope(" not in src
    assert SO.DEVICE_BUDGET_BYTES == 2048 * MiB and SO.OFFLOAD_MIN_BYTES == MiB


# ── the guarantee, on the real learner ────────────────────────────────────────────────────────
def _reference(model, args, monkeypatch, **forced):
    from agents.model import compile_regions as cr
    with monkeypatch.context() as mp:
        if forced:
            mp.setattr(SO, "SavedOffload", functools.partial(SO.SavedOffload, **forced))
        out = cr._r1_reference(model, args)
    return out, dict(SO.LAST)


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
    full, led = _reference(model, args, monkeypatch, force=True, budget_bytes=0, min_bytes=0)
    assert _same(plain, full) == {"loss": True, "log_prob": True, "grad": True, "grad_sizes": True}
    assert led["offloaded"] > 100 and led["returned"] >= led["offloaded"] and led["kept_mib"] == 0
    # a budget in the middle: some storages stay, the rest go, and the numbers are still the plain arm's
    half_bytes = int(led["offloaded_mib"] * MiB / 2)
    half, led2 = _reference(model, args, monkeypatch, force=True, budget_bytes=half_bytes, min_bytes=0)
    assert all(_same(plain, half).values())
    assert 0 < led2["kept_mib"] * MiB <= half_bytes and 0 < led2["offloaded"] < led["offloaded"]
    # the production thresholds, forced: only tensors of >= 1 MiB are candidates
    prod, led3 = _reference(model, args, monkeypatch, force=True, budget_bytes=0)
    assert all(_same(plain, prod).values()) and led3["offloaded"] < led["offloaded"]


@_skip_cuda
@pytest.mark.slow
def test_on_the_card_the_bounded_reference_arm_is_bit_identical_and_peaks_lower(monkeypatch,
                                                                                restore_torch_globals):
    """CUDA: the production object (no forcing). The plain arm twice is the control (CUDA's eager kernels
    are run-to-run bit-stable on these rows; if the control fails, this cell says so and decides nothing
    about the offload); then the arm with nothing kept on the card against it, and its peak below."""
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

    def arm(**forced):
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        out, led = _reference(model, args, monkeypatch, **forced)
        torch.cuda.synchronize(dev)
        return out, led, torch.cuda.max_memory_allocated(dev) / MiB

    unbounded = dict(budget_bytes=1 << 60)
    plain, led_p, peak_p = arm(**unbounded)
    again, _, _ = arm(**unbounded)
    assert led_p["offloaded"] == 0
    assert all(_same(plain, again).values()), "the CONTROL failed: two unbounded arms differ on this card"
    bounded, led_b, peak_b = arm(budget_bytes=0)
    assert all(_same(plain, bounded).values()), _same(plain, bounded)
    assert led_b["offloaded"] > 0 and led_b["returned"] >= led_b["offloaded"]
    assert peak_b < peak_p - 0.25 * led_b["offloaded_mib"], (peak_b, peak_p, led_b)
    default, led_d, peak_d = arm()
    assert all(_same(plain, default).values())
    print(f"[saved_offload] B=2048 peak MiB: unbounded {peak_p:.0f}, default budget {peak_d:.0f} "
          f"(ledger {led_d}), nothing kept {peak_b:.0f} (ledger {led_b})")
