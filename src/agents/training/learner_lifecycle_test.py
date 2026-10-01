"""K6 freeze guard (`gen3_learner_freeze_v1`): every acquisition kind after the freeze is a typed FATAL
naming the object and its construction site; the declared optimizer state is bit-identical to torch's
lazy init; a healthy production-surface update acquires NOTHING (and leaves the K9 golden unchanged).

Each "lazy X" test FAILS if the guard's rule for X is reverted (the check stops reporting it)."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from agents.training import learner_lifecycle as LL
from main.exit_codes import TrainExitCode, exit_code_for


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 8)
        self.head = nn.Linear(8, 2)
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.body(x))) * self.scale


class _Model:
    """The attributes `learner_objects` reads: a policy with an optimizer, like SB3's."""

    def __init__(self, opt_cls=torch.optim.AdamW, **kw) -> None:
        torch.manual_seed(0)
        self.policy = _Policy()
        self.policy.optimizer = opt_cls(self.policy.parameters(), lr=1e-2, **kw)

    def step(self, x: torch.Tensor) -> None:
        self.policy.optimizer.zero_grad()
        self.policy(x).square().mean().backward()
        self.policy.optimizer.step()


@pytest.fixture
def frozen():
    made = []

    def make(model):
        fz = LL.LearnerFreeze(model)
        fz.freeze("test")
        made.append(fz)
        return fz
    yield make
    for fz in made:
        fz.release()


X = torch.randn(16, 4, generator=torch.Generator().manual_seed(1))


# ------------------------------------------------------------------------- declared optimizer state
@pytest.mark.parametrize("opt_cls,kw", [(torch.optim.AdamW, {"weight_decay": 1e-2, "eps": 1e-5}),
                                        (torch.optim.Adam, {}),
                                        (torch.optim.Adam, {"weight_decay": 1e-3, "amsgrad": True})])
def test_declared_state_is_bit_identical_to_torch_lazy_init(opt_cls, kw):
    lazy, decl = _Model(opt_cls, **kw), _Model(opt_cls, **kw)
    assert LL.declare_optimizer_state(decl.policy.optimizer) == 4      # 2 Linears x (weight, bias)
    # the declaration moved no weight (AdamW's decoupled decay acts even on a zero gradient)
    for a, b in zip(lazy.policy.parameters(), decl.policy.parameters()):
        assert torch.equal(a, b)
        assert b.grad is None
    for _ in range(3):
        lazy.step(X)
        decl.step(X)
    for a, b in zip(lazy.policy.parameters(), decl.policy.parameters()):
        assert torch.equal(a, b)
    for pa, pb in zip(lazy.policy.parameters(), decl.policy.parameters()):
        sa, sb = lazy.policy.optimizer.state[pa], decl.policy.optimizer.state[pb]
        assert sorted(sa) == sorted(sb)
        for k in sa:
            assert torch.equal(torch.as_tensor(sa[k]), torch.as_tensor(sb[k])), k


def test_declared_state_skips_params_that_already_have_state():
    m = _Model()
    m.step(X)
    before = copy.deepcopy(m.policy.optimizer.state_dict())
    assert LL.declare_optimizer_state(m.policy.optimizer) == 0
    after = m.policy.optimizer.state_dict()
    for k, st in before["state"].items():
        for f, v in st.items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(after["state"][k][f]))


def test_an_undeclarable_optimizer_is_refused_typed():
    m = _Model(torch.optim.SGD, momentum=0.9)
    with pytest.raises(LL.LazyAcquisitionError, match="SGD"):
        LL.declare_optimizer_state(m.policy.optimizer)


# ------------------------------------------------------------------------- every acquisition kind
def test_healthy_steps_after_the_freeze_acquire_nothing(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    for _ in range(3):
        m.step(X)
        fz.check("update end")
    assert fz.violations_now() == []


def test_a_lazy_optimizer_is_fatal_and_names_its_construction_site(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.lazy_opt = torch.optim.Adam(m.policy.head.parameters(), lr=1e-3)   # the 09-30 ride-along shape
    with pytest.raises(LL.LazyAcquisitionError) as ei:
        fz.check("update end")
    msg = str(ei.value)
    assert "NEW OPTIMIZER model.lazy_opt" in msg and "Adam" in msg
    assert "test_a_lazy_optimizer_is_fatal_and_names_its_construction_site" in msg   # the site
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_CONFIG)


def test_a_lazy_optimizer_that_STEPS_is_refused_before_it_moves_a_weight(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    side = torch.optim.SGD(m.policy.head.parameters(), lr=1.0)   # held nowhere the walk can see
    w0 = m.policy.head.weight.detach().clone()
    m.policy(X).sum().backward()
    with pytest.raises(LL.LazyAcquisitionError, match="STEPPED after the freeze"):
        side.step()
    assert torch.equal(m.policy.head.weight, w0)
    # sticky: a caller that swallowed the raise cannot hide it from the next check
    with pytest.raises(LL.LazyAcquisitionError, match="STEPPED"):
        fz.check("update end")


def test_a_lazy_parameter_is_fatal_and_names_its_registration_site(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.policy.head.extra = nn.Parameter(torch.zeros(3))
    with pytest.raises(LL.LazyAcquisitionError) as ei:
        fz.check("update end")
    msg = str(ei.value)
    assert "NEW PARAMETER policy.head.extra" in msg and "(3,)" in msg
    assert "registered as parameter 'extra'" in msg
    assert "test_a_lazy_parameter_is_fatal" in msg


def test_a_replaced_parameter_is_fatal(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.policy.head.bias = nn.Parameter(m.policy.head.bias.detach().clone())   # same name, new object
    with pytest.raises(LL.LazyAcquisitionError, match="REPLACED PARAMETER policy.head.bias"):
        fz.check("update end")


def test_a_lazy_module_is_fatal(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.policy.probe = nn.Identity()
    with pytest.raises(LL.LazyAcquisitionError, match="NEW MODULE policy.probe = Identity"):
        fz.check("update end")


def test_a_lazy_buffer_is_fatal(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.policy.head.register_buffer("running", torch.zeros(2))
    with pytest.raises(LL.LazyAcquisitionError, match="NEW BUFFER policy.head.running"):
        fz.check("update end")


def test_optimizer_state_born_after_the_freeze_is_fatal(frozen):
    m = _Model()                                   # NOT declared: the first step creates the state
    fz = frozen(m)
    m.step(X)
    with pytest.raises(LL.LazyAcquisitionError, match="OPTIMIZER STATE created after the freeze"):
        fz.check("update end")


def test_a_param_group_added_after_the_freeze_is_fatal(frozen):
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    m.policy.optimizer.add_param_group({"params": [nn.Parameter(torch.zeros(2))]})
    with pytest.raises(LL.LazyAcquisitionError, match="PARAM GROUPS CHANGED"):
        fz.check("update end")


def test_an_object_built_OUTSIDE_the_learner_is_recorded_but_not_judged(frozen):
    """An in-process opponent (`--debug`) builds modules after the freeze: not the learner's."""
    m = _Model()
    LL.declare_optimizer_state(m.policy.optimizer)
    fz = frozen(m)
    other = _Policy()                              # never attached to the model
    fz.check("update end")
    assert fz.recorder.site_of(other.body) is not None


def test_release_restores_torch(frozen):
    orig_init = torch.optim.Optimizer.__init__
    m = _Model()
    fz = frozen(m)
    assert torch.optim.Optimizer.__init__ is not orig_init
    fz.release()
    assert torch.optim.Optimizer.__init__ is orig_init
    import torch.nn.modules.module as mm
    import torch.optim.optimizer as oo
    assert not mm._global_parameter_registration_hooks
    assert not mm._global_module_registration_hooks
    assert not mm._global_buffer_registration_hooks
    assert not oo._global_optimizer_pre_hooks
    torch.optim.SGD(m.policy.parameters(), lr=1.0).step()       # nothing judges after release


# ------------------------------------------------------------------------- attach
class _Trainer(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.lazy = False
        self.calls = []

    def collect_rollouts(self):
        self.calls.append("collect")
        return True

    def train(self):
        self.calls.append("train")
        if self.lazy:
            self.late = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.step(X)

    def learn(self):
        for _ in range(3):
            self.collect_rollouts()
            self.train()


def test_attach_freezes_at_the_first_rollout_checks_every_update_and_releases():
    t = _Trainer()
    LL.declare_learner_startup(t)
    fz = LL.attach(t)
    assert fz.frozen is None                       # startup is not over until the first rollout
    t.learn()
    assert t.calls == ["collect", "train"] * 3
    assert fz.checks == 6 and fz.frozen is None    # released when learn() returned


def test_attach_turns_a_lazy_acquisition_inside_train_into_the_typed_fatal():
    t = _Trainer()
    LL.declare_learner_startup(t)
    LL.attach(t)
    t.lazy = True
    with pytest.raises(LL.LazyAcquisitionError, match="NEW OPTIMIZER model.late"):
        t.learn()
    assert torch.optim.Optimizer.__init__.__name__ == "__init__"    # released on the way out


# ------------------------------------------------------------------------- the production surface
def test_a_production_surface_update_acquires_nothing_and_leaves_the_golden_unchanged():
    """The real thing: the K9 learner golden's production-surface learner, DECLARED (optimizer state
    created at startup) and FROZEN, runs one real `train()` on the committed real buffer. The guard
    finds nothing, AND the update's fingerprint equals the recorded golden — so the declaration is
    bit-identical to torch's lazy init on the production optimizer (AdamW, weight decay)."""
    from agents.training import learner_golden as LG

    model = LG.build_learner()
    LL.declare_learner_startup(model)
    fz = LL.LearnerFreeze(model)
    fz.freeze("test")
    try:
        got = LG.compute(model)
        fz.check("update end")
    finally:
        fz.release()
    rec = LG.load_golden()["entries"].get(LG.torch_key())
    if rec is None:
        pytest.fail(f"no learner golden recorded for torch {LG.torch_key()}")
    assert LG.diff(rec, got) == []


# ------------------------------------------------------------------------------ the memory half
def _mem_sample(update, phase, allocated_mib, reserved_mib=6000, free_mib=6000):
    from agents.training import cuda_memory_trend as cmt
    MiB = cmt.MiB
    stats = {"allocated_bytes.all.current": allocated_mib * MiB,
             "reserved_bytes.all.current": reserved_mib * MiB,
             "reserved_bytes.all.peak": reserved_mib * MiB,
             "allocated_bytes.all.peak": allocated_mib * MiB,
             "active_bytes.all.current": allocated_mib * MiB,
             "inactive_split_bytes.all.current": 0, "segment.all.current": 10,
             "num_alloc_retries": 0, "num_ooms": 0}
    return cmt.sample_from_stats(stats, free_mib * MiB, 12000 * MiB, update=update, phase=phase)


class _MemModel:
    """The three methods `attach` wraps, plus a TB logger — no torch model needed for the memory half."""
    device = "cpu"

    def __init__(self):
        self.rec = {}
        self._logger = self
        self.policy = None

    def record(self, k, v):
        self.rec[k] = v

    def collect_rollouts(self, *a, **k):
        return True

    def train(self):
        return None

    def learn(self):
        while self.collect_rollouts():
            self.train()


def _watch(model, leak_mib_per_update, lines):
    """A CudaMemoryWatch on a fake CUDA device whose quiescent allocated floor grows by
    ``leak_mib_per_update`` (0 = healthy) against a card with ~1 GiB headroom."""
    state = {"n": 0}

    def sampler(device, *, update, phase):
        state["n"] += 1
        return _mem_sample(update, phase, 3000 + leak_mib_per_update * update, free_mib=1000)
    return LL.CudaMemoryWatch(model, emit=lines.append, say=lines.append, device="cuda:0",
                              sampler=sampler)


def _drive(model, watch, updates):
    watch.start()
    for _ in range(updates):
        watch.observe("post_rollout")
        model.train()
        watch.observe("post_update")


def test_the_memory_half_STOPS_a_sustained_leak_with_a_typed_FATAL_and_logs_every_window():
    from main.exit_codes import TrainExitCode, exit_code_for
    m, lines = _MemModel(), []
    w = _watch(m, 40, lines)
    with pytest.raises(LL.CudaMemoryLeakError) as ei:
        _drive(m, w, 200)
    # its OWN code — not FATAL_CONFIG: a leak is no configuration error, and the launcher restarts it
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_CUDA_LEAK)
    assert "CUDA MEMORY LEAK" in str(ei.value) and LL.STOP_TAG in str(ei.value)
    assert sum("[CudaMemTrend]" in x for x in lines) >= 5          # the projection, every window
    assert "lifecycle/cuda_floor_mib" in m.rec and "lifecycle/cuda_updates_to_ceiling" in m.rec


def test_a_healthy_or_a_one_off_step_never_stops():
    m, lines = _MemModel(), []
    _drive(m, _watch(m, 0, lines), 120)                               # healthy: flat floor
    assert not any("STOP" in x for x in lines)
    m2, lines2 = _MemModel(), []

    def step(device, *, update, phase):                               # one +200 MiB step, then flat
        return _mem_sample(update, phase, 3000 + (200 if update > 30 else 0), free_mib=1000)
    _drive(m2, LL.CudaMemoryWatch(m2, emit=lines2.append, say=lines2.append, device="cuda:0",
                                  sampler=step), 120)
    assert not any("STOP" in x for x in lines2)


def test_attach_wires_the_memory_half_around_rollout_and_update_and_is_inert_off_cuda():
    m, lines = _MemModel(), []
    calls = []

    def sampler(device, *, update, phase):
        calls.append((update, phase))
        return _mem_sample(update, phase, 3000)
    watch = LL.CudaMemoryWatch(m, emit=lines.append, say=lines.append, device="cuda:0",
                               sampler=sampler)
    orig = m.collect_rollouts
    n = {"i": 0}

    def three(*a, **k):
        n["i"] += 1
        return n["i"] <= 3
    m.collect_rollouts = three
    freeze = LL.attach(m, emit=lines.append, memory=watch)
    freeze.freeze = lambda where: None                              # the object-graph half is tested above
    freeze.check = lambda where: None
    m.learn()
    assert calls[:4] == [(0, "post_rollout"), (1, "post_update"), (1, "post_rollout"), (2, "post_update")]
    assert freeze.memory is watch
    cpu = LL.CudaMemoryWatch(_MemModel())                            # device "cpu": no trend, no sample
    cpu.start()
    cpu.observe("post_update")
    assert cpu.trend is None and orig() is True
