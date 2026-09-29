"""Tests for COMPILE CONTROL — the torch._dynamo adapter + sentinel (`agents.model.compile_control`,
gen3_compile_sentinel_v1).

The sentinel exists because both of the failures it guards are SILENT in torch 2.5.1: a cache-limit
hit logs one warning and runs the frame eager forever; a late recompile costs a stall and spends the
cache. Each test here fails on a revert of the mechanism it names:

  (a) >8 guard signatures on one code object trip the cache-limit DETECTOR -> typed FATAL
      (revert: remove the logging handler, and `check()` passes);
  (b) after the lock a new shape raises the typed FATAL at the call site
      (revert: drop `error_on_recompile` AND the start callback, and the call just recompiles);
  (c) the healthy production alternation (rollout batch n_envs / train batch batch_size, eval/no-grad
      vs train/grad) converges, LOCKS, and then runs clean;
  (d) the sentinel changes NO numerics (outputs and gradients bit-identical on vs off).

They run on BOTH supported torch versions (2.5.1: `error_on_recompile`; 2.8: the
`fail_on_recompile` stance — Lane K1) and assert each version's own semantics where they differ
(`_MODE` below). They run on CPU with dynamo's `eager` backend: the cache, its guards, its limit and its recompile
path are dynamo's, independent of the backend, so these are the real mechanism at unit-test cost. The
production-graph count on the real extractor at fp32 and TF32 is the CUDA test at the bottom
(skips while a trainer holds the card; run by hand on an idle card).
"""
import pytest
import torch

from agents.model import compile_control as cc
from agents.model.compile_control import (CompileControl, CompileSentinelError, TrainMsWatch,
                                          cache_entries_by_code, find_recompile_error)
from agents.model.compile_trainer import CompileTrainerError, eager_extractor


def is_recompile_error(exc):
    return find_recompile_error(exc) is not None


# The installed torch's lock mechanism, and the name its cache-limit warning uses (torch 2.8 renamed
# `cache_size_limit` to `recompile_limit`, keeping the old name as a config ALIAS).
_MODE = cc.lock_mode()
_LIMIT_NAME = "cache_size_limit" if _MODE == "error_on_recompile" else "recompile_limit"


def _stance() -> str:
    """The dynamo stance in force ("default" on a torch without stances)."""
    from torch._dynamo import eval_frame
    st = getattr(eval_frame, "_stance", None)
    return getattr(st, "stance", "default")


@pytest.fixture
def sentinel():
    torch._dynamo.reset()
    prev = (torch._dynamo.config.error_on_recompile, torch._dynamo.config.suppress_errors)
    s = CompileControl().install()
    try:
        yield s
    finally:
        s.uninstall()
        torch._dynamo.config.error_on_recompile, torch._dynamo.config.suppress_errors = prev
        torch._dynamo.reset()


class _Net(torch.nn.Module):
    """A stand-in extractor: a train/eval-sensitive module with a (pi, vf) pair, like the real one."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(16, 8)
        self.drop = torch.nn.Dropout(0.0)

    def forward(self, obs):
        h = self.drop(torch.tanh(self.lin(obs["observation"])))
        return h, h * 2.0


def _obs(b):
    g = torch.Generator().manual_seed(b)
    return {"observation": torch.rand(b, 16, generator=g)}


def _compile_bound(net, ctl=None):
    """What `compile_trainer_extractor` does: patch the BOUND forward on the instance (through the
    control's `wrap_compiled` when one is given — production always passes it)."""
    compiled = torch.compile(net.forward, backend="eager")
    net.forward = ctl.wrap_compiled(compiled) if ctl is not None else compiled
    return net


def _one_iteration(net, n_envs=6, batch=24, rollout_steps=3, minibatches=2):
    """One production iteration's call pattern: rollout (eval, no-grad, batch n_envs) then the
    update (train, grad, batch batch_size, forward + backward)."""
    net.eval()
    with torch.no_grad():
        for _ in range(rollout_steps):
            net(_obs(n_envs))
    net.train()
    for _ in range(minibatches):
        pi, vf = net(_obs(batch))
        (pi.square().mean() + vf.square().mean()).backward()
        net.zero_grad(set_to_none=True)


def test_the_typed_fatal_is_a_CompileTrainerError_so_the_launcher_treats_it_as_config_fatal():
    assert issubclass(CompileSentinelError, CompileTrainerError)


# --------------------------------------------------------------------------- (a) the detector
def test_a_more_than_cache_limit_distinct_shapes_trips_the_detector_and_the_check_is_FATAL(sentinel):
    """Nine RANKS on one code object: automatic-dynamic cannot merge a rank change, so each is a new
    entry and the ninth hits `cache_size_limit` (8) — dynamo then logs one warning and runs the
    frame EAGER. The detector must turn that into the typed FATAL at the next check."""
    f = torch.compile(lambda x: (x * 2).sum(), backend="eager")
    for r in range(1, 11):
        f(torch.ones([2] * r))                  # still returns the right answer — eager, silently
    assert sentinel.limit_hits, "the cache-limit warning was not recognised"
    assert sentinel.limit_hits[0][0] == _LIMIT_NAME
    with pytest.raises(CompileSentinelError, match=_LIMIT_NAME):
        sentinel.check("rollout end")


def test_a_cache_limit_hit_is_FATAL_even_before_the_lock(sentinel):
    """A hit during warm-up (e.g. startup graphs eating the headroom — the T32b risk) is not a
    'warm-up recompile', it is the fallback itself."""
    assert not sentinel.locked
    f = torch.compile(lambda x: x.sum(), backend="eager")
    for r in range(1, 11):
        f(torch.ones([1] * r))
    with pytest.raises(CompileSentinelError):
        sentinel.check("update end")


def test_an_unrelated_dynamo_warning_is_not_a_hit(sentinel):
    import logging
    logging.getLogger("torch._dynamo.convert_frame").warning("some other dynamo warning %s", 1)
    sentinel.check("anywhere")                  # must not raise


# --------------------------------------------------------------------------- (b) the lock
def test_b_after_the_lock_a_new_shape_raises_the_typed_FATAL_at_the_call_site(sentinel):
    net = _compile_bound(_Net(), sentinel)
    _one_iteration(net)
    sentinel.lock("iteration 1")
    with pytest.raises(CompileSentinelError, match="RECOMPILED after the compile lock"):
        with sentinel.guard("rollout"):  # noqa: SIM117
            net.eval()
            with torch.no_grad():
                net({"observation": torch.rand(4, 2, 16)})   # a rank change: a new guard set
    assert sentinel.compiles_after_lock + sentinel.rejected_after_lock >= 1


def test_b_a_swallowed_RecompileError_is_still_caught_by_the_sticky_check(sentinel):
    """A caller that catches everything (a diagnostic's `except Exception: pass`) must not hide a
    late recompile: `wrap_compiled` (and on 2.5.1 also the compile-START callback) counted it, so the next
    check is FATAL."""
    net = _compile_bound(_Net(), sentinel)
    _one_iteration(net)
    sentinel.lock("iteration 1")
    try:
        with torch.no_grad():
            net({"observation": torch.rand(4, 2, 16)})
    except Exception as exc:                    # the swallow
        assert is_recompile_error(exc)
    with pytest.raises(CompileSentinelError, match="AFTER the compile lock"):
        sentinel.check("update end")


def test_b_a_first_compile_of_a_NEW_code_object_after_the_lock_is_FATAL(sentinel):
    """`error_on_recompile` only fires on a RE-compile; a never-seen frame compiling late is caught
    by the start callback (2.5.1). The 2.8 stance REJECTS it at the call site instead (any cache
    miss on a compiled callable, a first compile included), so it never compiles at all."""
    net = _compile_bound(_Net(), sentinel)
    _one_iteration(net)
    sentinel.lock("iteration 1")
    late = torch.compile(lambda x: x + 1, backend="eager")
    if _MODE == "stance":
        with pytest.raises(RuntimeError) as ei:
            late(torch.ones(3))
        assert is_recompile_error(ei.value)
        return
    late(torch.ones(3))
    with pytest.raises(CompileSentinelError):
        sentinel.check("update end")


# --------------------------------------------------------------------------- (c) the healthy case
def test_c_the_healthy_two_shape_alternation_converges_LOCKS_and_runs_clean(sentinel):
    net = _compile_bound(_Net(), sentinel)
    _one_iteration(net)
    line = sentinel.lock("iteration 1")
    assert "COMPILE LOCK" in line
    # the HEALTHY lock line must not trip scripts/ops/watch_run.sh's failure grep (2026-09-28:
    # the old "... is FATAL." wording made every watcher on a sentinel pin exit after update 1)
    import re
    assert not re.search(r"OutOfMemory|CUDA out of memory|FATAL|Traceback", line, re.I), line
    ent = cache_entries_by_code()
    assert ent and max(ent.values()) < torch._dynamo.config.cache_size_limit
    for _ in range(5):                          # steady state: the same shapes forever
        with sentinel.guard("iteration"):
            _one_iteration(net)
    # automatic-dynamic made dim 0 symbolic, so a DIFFERENT size (e.g. a diagnostic probe) is
    # served by the same graph without a recompile — only a new guard signature would be fatal.
    with sentinel.guard("probe"):
        net.train()
        net(_obs(17))[0].sum().backward()
    sentinel.check("steady state")
    assert sentinel.compiles_after_lock == 0 and sentinel.rejected_after_lock == 0
    s = sentinel.stats()
    assert s["compile/recompiles_after_lock"] == 0 and s["compile/cache_limit_hits"] == 0
    assert s["compile/locked"] == 1.0 and s["compile/graphs_total"] >= 1


# --------------------------------------------------------------------------- (d) no numerics
def _run(with_sentinel):
    torch._dynamo.reset()
    s = CompileControl().install() if with_sentinel else None
    try:
        net = _compile_bound(_Net())
        outs, grads = [], []
        for it in range(3):
            if s is not None and it == 1:
                s.lock("iteration 1")
            net.eval()
            with torch.no_grad():
                outs.append(net(_obs(6))[0].clone())
            net.train()
            pi, vf = net(_obs(24))
            (pi.square().mean() + vf.square().mean()).backward()
            grads.append(net.lin.weight.grad.clone())
            outs.append(vf.detach().clone())
            net.zero_grad(set_to_none=True)
        return outs, grads
    finally:
        if s is not None:
            s.uninstall()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def test_d_the_sentinel_changes_NO_numerics():
    a_out, a_grad = _run(False)
    b_out, b_grad = _run(True)
    for x, y in zip(a_out + a_grad, b_out + b_grad):
        assert torch.equal(x, y)


def test_uninstall_restores_error_on_recompile(sentinel):
    net = _compile_bound(_Net(), sentinel)
    _one_iteration(net)
    sentinel.lock("iteration 1")
    if _MODE == "stance":
        assert _stance() == "fail_on_recompile"
    else:
        assert torch._dynamo.config.error_on_recompile is True
    sentinel.uninstall()
    assert torch._dynamo.config.error_on_recompile is False and _stance() == "default"


# --------------------------------------------------------------------------- the backstop
def test_train_ms_watch_warns_after_three_consecutive_slow_updates_and_never_raises():
    w = TrainMsWatch()
    for v in (100, 102, 98, 101, 99):
        assert w.observe(v) is None
    assert w.baseline == 100
    assert w.observe(150) is None and w.observe(150) is None
    line = w.observe(150)
    assert line is not None and "COMPILE REGRESSION?" in line
    assert w.scalars()["compile/regression_flag"] == 1.0
    assert w.observe(151) is None               # said once per episode
    assert w.observe(100) is None
    assert w.scalars()["compile/regression_flag"] == 0.0


def test_train_ms_watch_resets_the_streak_on_a_normal_update():
    w = TrainMsWatch()
    for v in (100,) * 5:
        w.observe(v)
    for v in (150, 150, 100, 150, 150):
        assert w.observe(v) is None


# --------------------------------------------------------------------------- the phases
def test_gate_then_reset_drops_every_gate_graph_before_production(sentinel):
    """Phase 2. The gate's graphs live on the SAME code objects production uses (the cache is per
    code object, not per compiled wrapper), so without the reset they spend production's slots."""
    net = _Net()
    with sentinel.gate():
        g = torch.compile(net.forward, backend="eager")
        net.train()
        g(_obs(64))
        with torch.no_grad():
            g(_obs(64))
    assert sum(sentinel.entries_after_gate.values()) >= 2
    # a SEPARATE compiled wrapper of the same bound method shares those entries — the reason a
    # "separate callable for the gate" would not have isolated them:
    assert max(cache_entries_by_code().values()) >= 2
    sentinel.reset()
    assert cache_entries_by_code() == {}


def test_prewarm_is_rng_neutral_and_its_signatures_survive_the_lock(sentinel):
    net = _compile_bound(_Net())
    torch.manual_seed(123)
    expect = torch.rand(3)
    torch.manual_seed(123)

    def _rollout():
        torch.rand(5)                            # a prewarm that consumes RNG must not leak it
        net.eval()
        with torch.no_grad():
            net(_obs(6))
    sentinel.prewarm([("rollout", _rollout)])
    assert torch.equal(torch.rand(3), expect)
    assert sentinel.prewarmed == ["rollout"]
    sentinel.lock("prewarm")
    with sentinel.guard("rollout"):
        net.eval()
        with torch.no_grad():
            net(_obs(6))


def test_attach_locks_after_the_first_update_and_a_late_recompile_exits_FATAL_CONFIG(sentinel,
                                                                                      monkeypatch):
    """The trainer wiring: the lock lands after the first `train()`, the compile/* scalars are
    recorded, and a violation inside `train()` leaves by `os._exit(FATAL_CONFIG)` — an exception
    would reach `model_build`'s generic `except` and become a RESTARTABLE crash."""
    from main.exit_codes import TrainExitCode
    net = _compile_bound(_Net())
    rec = {}

    class _Logger:
        name_to_value = {"train/train_ms": 100.0}

        def record(self, k, v):
            rec[k] = v

    class _Model:
        logger = _Logger()

        def collect_rollouts(self):
            net.eval()
            with torch.no_grad():
                net(_obs(6))
            return True

        def train(self):
            net.train()
            net(_obs(24))[0].sum().backward()

        def learn(self):
            return None

    exits = []

    def _fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(cc.os, "_exit", _fake_exit)
    m = _Model()
    sentinel.attach(m)
    assert m.collect_rollouts() is True
    m.train()
    assert sentinel.locked and rec["compile/locked"] == 1.0
    assert rec["compile/recompiles_after_lock"] == 0.0
    m.collect_rollouts()
    m.train()
    assert not exits

    def _bad_train():
        net({"observation": torch.rand(2, 3, 16)})   # a new guard signature after the lock
    m.train = _bad_train
    sentinel.attach(m)                           # re-wrap the replaced train
    with pytest.raises(SystemExit):
        m.train()
    assert exits == [int(TrainExitCode.FATAL_CONFIG)]


def test_eager_extractor_routes_around_the_compiled_forward_and_restores_it(sentinel):
    net = _compile_bound(_Net())
    _one_iteration(net)
    sentinel.lock("iteration 1")
    compiled = vars(net)["forward"]
    with sentinel.guard("late caller"), eager_extractor(net):
        assert "forward" not in vars(net)
        net({"observation": torch.rand(1, 16)})          # batch 1: would recompile if compiled
        net({"observation": torch.rand(2, 3, 16)})       # a new rank: likewise
    assert vars(net)["forward"] is compiled
    assert sentinel.compiles_after_lock == 0


def test_eager_extractor_is_a_noop_on_an_uncompiled_module():
    net = _Net()
    with eager_extractor(net):
        net(_obs(3))
    assert "forward" not in vars(net)


def test_the_launcher_classifies_the_sentinel_FATAL_as_non_restartable():
    from main.launcher.run import _fatal_config_reason
    assert _fatal_config_reason(1, ["x", f"{cc.FATAL_TAG} at update end: ..."]) is not None
    assert _fatal_config_reason(1, ["[CompileTrainer] FATAL: parity"]) is not None


# --------------------------------------------------------------------------- the version guard
def test_an_unknown_torch_version_is_REFUSED():
    with pytest.raises(CompileSentinelError, match="does not support torch"):
        cc.lock_mode("2.8.0+cu128")
    with pytest.raises(CompileSentinelError, match="no recorded torch-internals row"):
        cc.verify_torch_internals("2.8.0+cu128")
    assert cc.lock_mode("2.5.1+cu121") == "error_on_recompile"
    assert cc.lock_mode("2.8.0+cu126") == "stance"
    assert set(cc._SUPPORTED) == set(cc._SOURCE_HASHES)    # a supported torch always has a row


def test_the_source_hash_tripwire_matches_the_installed_torch():
    """HASH = the code changed. Fails on a torch whose internals differ from the recorded row; the
    message names the drifted qualnames with both hashes (the SB3 `_verify_upstream_unchanged`
    pattern)."""
    cc.verify_torch_internals()                  # the installed torch: must be clean


def test_the_source_hash_tripwire_NAMES_a_drifted_internal(monkeypatch):
    row = dict(cc._SOURCE_HASHES[torch.__version__])
    key = next(k for k in row if ".exceeds_" in k)   # renamed between 2.5.1 and 2.8
    row[key] = "0" * 64
    monkeypatch.setitem(cc._SOURCE_HASHES, torch.__version__, row)
    with pytest.raises(CompileSentinelError) as ei:
        cc.verify_torch_internals()
    msg = str(ei.value)
    assert key in msg and "0" * 64 in msg and "re-record" in msg
    assert "compute_cache_size" not in msg       # only the drifted one is named


def test_every_hashed_internal_resolves():
    got = cc.source_hashes(sorted(cc._SOURCE_HASHES[torch.__version__]))
    assert not [q for q, h in got.items() if h.startswith("<missing")]


# --------------------------------------------------------------------------- functorch config row
# gen3_donated_buffer_off_v1 (Lane K1b). The learner's read-only probes put a `retain_graph=True`
# backward through the compiled graph after the prewarm's plain one; with AOTAutograd's donated
# buffers ON (the torch >= 2.6 default) that raises. Revert `apply_compile_config` in `install()`
# and the first two tests fail; the third pins that the pin changes no number; the fourth pins
# the torch-2.8 hazard that made the cache-key tag necessary (the Inductor cache key is blind to
# donation, so the tag must reach the key).
def _donating_step(fn, lin, x):
    """The production order: a plain backward FIRST (the prewarm), then a retain-graph probe and
    the real backward on a second forward (grad-balance, then `loss.backward()`)."""
    fn(x).square().sum().backward()
    plain = lin.weight.grad.clone()
    lin.zero_grad(set_to_none=True)
    out = fn(x).square().sum()
    probe = torch.autograd.grad(out, [lin.weight], retain_graph=True)[0]
    out.backward()
    return plain, probe, lin.weight.grad.clone()


def _saved_intermediate_net():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8)
    # sin's backward saves its INPUT, an intermediate that is not a graph output: a donated buffer.
    return lin, torch.compile(lambda x: torch.sin(lin(x)) * 2.0, backend="aot_eager")


def test_install_pins_donated_buffer_OFF_and_uninstall_restores_it():
    import torch._functorch.config as fcfg
    prev = fcfg.donated_buffer
    try:
        fcfg.donated_buffer = True                       # the torch >= 2.6 default
        ctl = CompileControl().install()
        assert fcfg.donated_buffer is False
        if _MODE == "stance":                            # torch 2.8: the Inductor cache-key tag
            import torch.compiler.config as ccfg
            assert ccfg.cache_key_tag == "gen3_donated_buffer_off_v1"
        ctl.uninstall()
        assert fcfg.donated_buffer is True
        if _MODE == "stance":
            assert ccfg.cache_key_tag == ""
        assert set(cc._COMPILE_CONFIG) == set(cc._SUPPORTED)     # every supported torch has a row
    finally:
        fcfg.donated_buffer = prev


def test_contract_a_donated_backward_REFUSES_a_retain_graph_probe_and_the_pinned_row_accepts_it():
    import torch._functorch.config as fcfg
    prev = fcfg.donated_buffer
    x = torch.rand(4, 16)
    try:
        torch._dynamo.reset()
        fcfg.donated_buffer = True
        lin, fn = _saved_intermediate_net()
        with pytest.raises(RuntimeError, match="donated buffers"):
            _donating_step(fn, lin, x)
        torch._dynamo.reset()
        ctl = CompileControl().install()
        try:
            lin, fn = _saved_intermediate_net()
            _, probe, full = _donating_step(fn, lin, x)
            assert torch.equal(probe, full)              # the probe read the same gradient
        finally:
            ctl.uninstall()
    finally:
        fcfg.donated_buffer = prev
        torch._dynamo.reset()


def test_the_donated_buffer_pin_changes_no_gradient():
    import torch._functorch.config as fcfg
    prev = fcfg.donated_buffer
    x = torch.rand(4, 16)
    grads = {}
    try:
        for donated in (True, False):
            torch._dynamo.reset()
            fcfg.donated_buffer = donated
            lin, fn = _saved_intermediate_net()
            fn(x).square().sum().backward()
            grads[donated] = (lin.weight.grad.clone(), lin.bias.grad.clone())
    finally:
        fcfg.donated_buffer = prev
        torch._dynamo.reset()
    assert all(torch.equal(a, b) for a, b in zip(grads[True], grads[False]))


@pytest.mark.skipif(_MODE != "stance", reason="torch.compiler.config.cache_key_tag exists from 2.6")
def test_contract_the_cache_key_tag_reaches_the_Inductor_cache_key_and_donation_does_not():
    """Why the tag is in the row: `FxGraphHashDetails` (the Inductor FX-graph cache key) carries
    `cache_key_tag` but NOT the backward's donated indices — so, without the tag, a backward compiled
    with donation by another process is served to our donation-off compile (measured 2026-09-29)."""
    import inspect

    import torch.compiler.config as ccfg
    from torch._inductor.codecache import FxGraphHashDetails
    src = inspect.getsource(FxGraphHashDetails.__init__)
    assert "cache_key_tag" in src and "donated" not in src
    prev = ccfg.cache_key_tag
    try:
        ccfg.cache_key_tag = "gen3_donated_buffer_off_v1"
        d = FxGraphHashDetails(None, [], {}, [])
        assert d.cache_key_tag == "gen3_donated_buffer_off_v1"
    finally:
        ccfg.cache_key_tag = prev


# --------------------------------------------------------------------------- CONTRACT tests
# BEHAVIOUR = the semantics changed. Each internal the adapter relies on, exercised on the
# installed torch (the hash row says the code is the same; these say it still DOES the same).
def test_contract_cache_limit_warning_text_and_logger():
    """`convert_frame` emits `torch._dynamo hit config.<limit>` on the `torch._dynamo.convert_frame`
    logger, and the call still RETURNS (the silent eager fallback the detector exists for)."""
    import logging
    torch._dynamo.reset()
    seen = []

    class _H(logging.Handler):
        def emit(self, r):
            seen.append((r.name, r.getMessage()))
    h = _H(level=logging.WARNING)
    logging.getLogger("torch._dynamo.convert_frame").addHandler(h)
    try:
        f = torch.compile(lambda x: x.sum(), backend="eager")
        for r in range(1, 11):
            assert float(f(torch.ones([1] * r))) == 1.0
    finally:
        logging.getLogger("torch._dynamo.convert_frame").removeHandler(h)
        torch._dynamo.reset()
    hits = [m for n, m in seen if "hit config." in m]
    assert hits and hits[0].startswith(f"torch._dynamo hit config.{_LIMIT_NAME} (8)")


def test_contract_error_on_recompile_raises_RecompileError_raw_at_the_call_site():
    from torch._dynamo.exc import RecompileError
    torch._dynamo.reset()
    f = torch.compile(lambda x: x * 2, backend="eager")
    f(torch.ones(3))
    try:
        with torch._dynamo.config.patch(error_on_recompile=True):
            with pytest.raises(RecompileError):
                f(torch.ones(3, 3))
            f(torch.ones(3))                     # a cache HIT never raises
    finally:
        torch._dynamo.reset()


def test_contract_start_callback_fires_once_per_frame_compile_and_never_on_a_hit():
    from torch._dynamo.callback import callback_handler
    torch._dynamo.reset()
    n = []
    # 2.5.1 calls it with no argument, 2.8 with a `CallbackArgs`
    cb = callback_handler.register_start_callback(lambda *a: n.append(1))
    try:
        f = torch.compile(lambda x: x + 1, backend="eager")
        f(torch.ones(2))
        assert len(n) == 1
        f(torch.ones(2))
        assert len(n) == 1
    finally:
        callback_handler.remove_start_callback(cb)
        torch._dynamo.reset()


@pytest.mark.skipif(_MODE != "stance", reason="the fail_on_recompile stance exists from torch 2.6")
def test_contract_stance_rejects_a_recompile_AND_a_new_frame_at_the_call_site_and_releases():
    """torch 2.8: `set_stance("fail_on_recompile")` raises (a RuntimeError carrying
    `_STANCE_REJECT_TEXT`, found by `find_recompile_error`) on a cache MISS — a new guard set on a
    compiled callable AND the first call of a never-compiled one — never on a HIT, compiles NOTHING
    (no start callback, no new graph), and `set_stance("default")` releases it. ALSO pinned: the
    start callback does NOT fire for a recompile `error_on_recompile` rejects (it now runs after the
    recompile check) — why `wrap_compiled` exists."""
    from torch._dynamo.callback import callback_handler
    from torch._dynamo.exc import RecompileError
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    n = []
    cb = callback_handler.register_start_callback(lambda *a: n.append(1))
    try:
        f = torch.compile(lambda x: x * 3, backend="eager")
        f(torch.ones(3))
        g0, n0 = int(counters["stats"]["unique_graphs"]), len(n)
        torch.compiler.set_stance("fail_on_recompile")
        try:
            f(torch.ones(3))                                  # a HIT: fine
            with pytest.raises(RuntimeError) as e1:
                f(torch.ones(3, 3))                           # a rank change: a new guard set
            assert find_recompile_error(e1.value) is e1.value
            with pytest.raises(RuntimeError) as e2:
                torch.compile(lambda x: x - 3, backend="eager")(torch.ones(3))   # a new frame
            assert find_recompile_error(e2.value) is not None
        finally:
            torch.compiler.set_stance("default")
        assert int(counters["stats"]["unique_graphs"]) == g0 and len(n) == n0
        with torch._dynamo.config.patch(error_on_recompile=True):
            with pytest.raises(RecompileError):
                f(torch.ones(2, 2, 2))
        assert len(n) == n0, "2.8 moved the start callback AFTER the recompile check"
        f(torch.ones(3, 3))                                   # released: compiles again
        assert len(n) == n0 + 1
    finally:
        callback_handler.remove_start_callback(cb)
        torch._dynamo.reset()


def test_contract_graph_counter_entry_list_and_reset():
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    before = int(counters["stats"]["unique_graphs"])
    f = torch.compile(lambda x: x - 1, backend="eager")
    f(torch.ones(2))
    f(torch.ones(2, 2))
    assert int(counters["stats"]["unique_graphs"]) - before == 2
    ent = cache_entries_by_code()
    assert ent and max(ent.values()) == 2
    torch._dynamo.reset()
    assert cache_entries_by_code() == {}


def test_contract_cache_is_per_code_object_shared_across_compiled_wrappers():
    """The fact the post-gate reset rests on: two `torch.compile` wrappers of the same function
    fill ONE code object's cache (torch 2.5.1)."""
    torch._dynamo.reset()

    def g(x):
        return x.cos()
    torch.compile(g, backend="eager")(torch.ones(2))
    torch.compile(g, backend="eager")(torch.ones(2, 2))
    try:
        from torch._dynamo import eval_frame
        assert len(eval_frame._debug_get_cache_entry_list(g.__code__)) == 2
    finally:
        torch._dynamo.reset()


# --------------------------------------------------------------------------- the real extractor
from agents.model.extractor_compiles_test import _cuda_skip_reason  # noqa: E402

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")


@_skip_cuda
@pytest.mark.slow
@pytest.mark.parametrize("precision", ["highest", "high"])
def test_the_production_extractor_locks_with_headroom_at_fp32_and_tf32(precision):
    """Item 6 on the card: the real production extractor through the real gate, the reset and the
    production prewarm (rollout batch 48, train batch 256 as the dynamic-train proxy) LOCKS with
    every code object under `cache_size_limit`, and then runs a steady iteration clean."""
    from agents.model.compile_trainer import compile_trainer_extractor, production_prewarm_calls
    from agents.model.extractor_compiles_test import _build_production_extractor
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        fe, layout = _build_production_extractor()
        fe = fe.cuda()
        fe.obs_dim = layout["total_dim"]
        pol = torch.nn.Module()
        pol.features_extractor = fe
        m = torch.nn.Module()
        m.policy = pol
        compile_trainer_extractor(m, True, batch=16)
        ctl = cc.control()
        ctl.reset()
        import gymnasium as gym
        import numpy as np
        pol.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(
            0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)})
        pol.set_training_mode = lambda mode: pol.train(mode)
        pol.extract_features = lambda obs: fe(obs)
        ctl.prewarm(production_prewarm_calls(m, n_envs=48, batch_size=256, batch1=False))
        ctl.lock("prewarm")
        ent = cc.cache_entries_by_code()
        assert ent and max(ent.values()) < cc.cache_size_limit(), ent
        with ctl.guard("steady iteration"):
            for label, fn in production_prewarm_calls(m, n_envs=48, batch_size=256, batch1=False):
                fn()
        assert ctl.compiles_after_lock == 0
    finally:
        torch.set_float32_matmul_precision(prev)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()
    assert _stance() == "default"


def test_debug_with_in_process_opponent_compiles_is_refused_at_startup():
    """--debug's DummyVecEnv compiles opponents in the learner process, on the learner's code
    objects, after the lock — refused at startup (FATAL_CONFIG), not discovered at the first
    snapshot load."""
    import argparse
    from main.exit_codes import TrainExitCode
    from main.train.lifecycle import _arm_compile_sentinel
    args = argparse.Namespace(compile_trainer=True, debug=True, compile_opponents=True)
    with pytest.raises(SystemExit) as ei:
        _arm_compile_sentinel(None, args)
    assert ei.value.code == TrainExitCode.FATAL_CONFIG


def test_arming_is_a_noop_when_the_learner_is_not_compiled():
    import argparse
    from main.train.lifecycle import _arm_compile_sentinel
    _arm_compile_sentinel(None, argparse.Namespace(compile_trainer=False))


def test_the_attached_wrappers_are_excluded_from_every_checkpoint():
    """`attach` sets INSTANCE attributes (`collect_rollouts`, `train`, `_compile_control`) on the
    model; SB3's `save` pickles `__dict__` minus `_excluded_save_params`, so without the exclusion
    every save after the first update would try to cloudpickle a live sentinel (logging handler,
    dynamo callbacks) — the `_correction_buffer` lock hazard again."""
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    excluded = InstrumentedMaskablePPO._excluded_save_params(
        InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO))

    class _M:
        logger = None

        def collect_rollouts(self):
            return True

        def train(self):
            return None

        def learn(self):
            return None
    m = _M()
    CompileControl().attach(m)
    assert set(vars(m)) <= set(excluded), set(vars(m)) - set(excluded)


def test_the_lock_is_RELEASED_when_learn_returns_so_the_in_process_final_eval_can_compile(sentinel):
    """MEASURED on the real trainer (2026-09-28 end-to-end smoke): after `learn()` the trainer runs
    its FINAL EVALUATION in-process on the same compiled forward at batch 1 / no-grad, and the
    still-locked sentinel raised RecompileError there. The lock's scope is training."""
    net = _compile_bound(_Net())

    class _Model:
        logger = None

        def collect_rollouts(self):
            net.eval()
            with torch.no_grad():
                net(_obs(6))
            return True

        def train(self):
            net.train()
            net(_obs(24))[0].sum().backward()

        def learn(self):
            for _ in range(3):
                self.collect_rollouts()
                self.train()

    m = _Model()
    sentinel.attach(m)
    m.learn()
    assert not sentinel.locked and torch._dynamo.config.error_on_recompile is False
    assert _stance() == "default"
    net.eval()
    with torch.no_grad():
        net({"observation": torch.rand(1, 16)})       # the final eval's batch-1 call: compiles
    assert sentinel.compiles_after_lock == 0


def test_the_lock_is_released_even_when_learn_raises(sentinel):
    class _Model:
        logger = None

        def collect_rollouts(self):
            return True

        def train(self):
            return None

        def learn(self):
            self.collect_rollouts()
            self.train()
            raise RuntimeError("boom")

    m = _Model()
    sentinel.attach(m)
    with pytest.raises(RuntimeError):
        m.learn()
    assert not sentinel.locked and torch._dynamo.config.error_on_recompile is False
    assert _stance() == "default"


@pytest.mark.parametrize("module, needle, count", [
    ("agents.training.instrumented_ppo.ppo", "with _eager_fe(", 3),     # search-teacher x2, OPD
    ("agents.training.instrumented_ppo.distill_grad_project", "with eager_extractor(", 1),
    ("agents.training.fork_callback", "eager_extractor(", 1),
    ("agents.training.fork_driver", "eager_extractor(", 1),
])
def test_the_late_signature_callers_stay_routed_to_eager(module, needle, count):
    """The learner-process callers whose signature can FIRST appear after the lock (a different obs
    key set, a batch that may be 1) run the EAGER forward. Unwrap one and the lock kills the run the
    first time it fires — so the routing is pinned here, not left to a comment."""
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module(module))
    assert src.count(needle) >= count, f"{module}: expected {count}x {needle!r}"
