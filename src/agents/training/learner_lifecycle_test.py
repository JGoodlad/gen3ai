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
