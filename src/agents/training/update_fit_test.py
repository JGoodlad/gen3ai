"""gen3_update_fit_v1 — the dry first update at startup (`agents/training/update_fit.py`):

* the learner after a dry update is EXACTLY the learner before it: the next real `train()` lands on
  bit-identical parameters and logs identical losses (the learner golden's surface, CPU);
* the verdict is deterministic — a headroom one MiB short of the declared 1,024 is REFUSED, an OOM
  inside the dry update is REFUSED, both FATAL_CONFIG; at the bar it passes;
* the trainer runs the check on both startup paths, where the superseded lower-bound check was;
* **the fill's inputs** (`gen3_update_fit_fill_inputs_v1`): with `--magnet-kl-coef` on, the dry update trains WITH
  the magnet — its fixture's column is served by T2's magnet slot, as the collector's fill does — and restores
  exactly; with no slot to serve it, it refuses by name before `train()`; every write of `RustCollector._fill` is
  declared to the dry update (the class: CE-3d's first launch died here at step 0, on CUDA only).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import agents.training.update_fit as UF


def _learner():
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _update(model, seed=True):
    from agents.training import learner_golden as LG
    if seed:
        np.random.seed(LG.UPDATE_SEED)
        torch.manual_seed(LG.UPDATE_SEED)
    model.train()
    losses = {k: v for k, v in model.logger.name_to_value.items()
              if LG._is_pinned_key(k) and isinstance(v, (int, float, np.floating))}
    return LG.params_sha256(model), losses


@pytest.mark.slow  # ship gate (2026-10-19): 20 s beside a live run
def test_a_dry_update_leaves_the_learner_bit_identical():
    from agents.training import learner_golden as LG
    ref = _update(_learner())
    m = _learner()
    before = (LG.params_sha256(m), sorted(vars(m)), m._n_updates,
              {k: v.clone() for k, v in m.policy.optimizer.state_dict()["state"].get(0, {}).items()
               if torch.is_tensor(v)})
    np.random.seed(LG.UPDATE_SEED)                       # the dry update must not consume the streams
    torch.manual_seed(LG.UPDATE_SEED)
    r = UF.dry_update(m)
    assert r.oom is None and r.rows == 64 and "golden" in r.source
    assert (LG.params_sha256(m), sorted(vars(m)), m._n_updates) == before[:3]
    for k, v in before[3].items():
        assert torch.equal(m.policy.optimizer.state_dict()["state"][0][k], v), k
    assert not m.logger.name_to_value                   # its records do not reach the real update's
    got = _update(m, seed=False)                        # the streams continue from BEFORE the dry update
    assert got[0] == ref[0]                             # the next real update: bit-identical weights
    assert got[1] == ref[1]                             # ... and the same logged losses


def test_a_dry_update_restores_the_loggers_producer_sites_with_its_values():
    """`gen3_metric_producer_guard_v1`: the snapshot carries `name_to_site` beside the values, or the dry update's own
    producers would stay behind as a stale site for every key it wrote (a false duplicate on the next real record)."""
    m = _learner()
    m.logger.record("zz/pending", 1.0)
    sites = dict(m.logger.name_to_site)
    assert set(sites) == {"zz/pending"}
    UF.dry_update(m)
    assert dict(m.logger.name_to_site) == sites and set(m.logger.name_to_value) == {"zz/pending"}


def test_the_dry_update_really_trains_on_its_fixture():
    """Teeth for the test above: inside the dry update the weights DO move (it is a real update)."""
    from agents.training import learner_golden as LG
    m = _learner()
    h0 = LG.params_sha256(m)
    seen = {}
    real_train = type(m).train

    def spy(self):
        real_train(self)
        seen["after"] = LG.params_sha256(self)
    m.train = spy.__get__(m)
    UF.dry_update(m)
    assert seen["after"] != h0 and LG.params_sha256(m) == h0


def _reading(headroom, oom=None):
    card = 10_000.0
    return UF.FitReading(demand_mib=card - headroom, peak_alloc_mib=8_000.0, reserved_mib=9_000.0,
                         device_free_mib=1_000.0, card_mib=card, headroom_mib=headroom,
                         floor_alloc_mib=1_900.0, seconds=4.0, rows=98_304, source="fixture", oom=oom)


def test_the_verdict_is_deterministic_at_the_declared_bar(tmp_path):
    from main.exit_codes import TrainExitCode, exit_code_for
    assert UF.FIT_HEADROOM_MIB == 1024.0
    line = UF.verdict(_reading(1024.0), "staged", str(tmp_path))
    assert "headroom 1,024 MiB" in line and (tmp_path / "update_fit.json").exists()
    with pytest.raises(UF.UpdateWontFit, match="does not fit") as ei:
        UF.verdict(_reading(1023.0), "resident")
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_CONFIG)
    with pytest.raises(UF.UpdateWontFit, match="OUT OF MEMORY"):
        UF.verdict(_reading(5_000.0, oom="CUDA out of memory. Tried to allocate 62.00 MiB"), "resident")


def test_a_learner_with_no_logger_yet_runs_the_dry_update_and_still_has_none():
    """At startup sb3's `_logger` does not exist (`_setup_learn` makes it) — measured: the first real
    launch died there (AttributeError). The dry update brings its own and leaves none behind."""
    from agents.training import learner_golden as LG
    m = _learner()
    del m._logger
    h0 = LG.params_sha256(m)
    r = UF.dry_update(m)
    assert r.oom is None and "_logger" not in vars(m) and LG.params_sha256(m) == h0


def test_a_dry_update_runs_with_the_overlap_runner_attached():
    """X14 (`--overlap-collect`): the collector's overlap runner is attached BEFORE the startup check, so the dry
    update's `train()` reaches the decoupled-PPO rewrite, which needs the proximal log-probs only a real FILL
    computes. Found by the 2026-10-20 core review (A1): it raised `OverlapError` at every CUDA launch with the
    flag on. The dry update hands it an identity rewrite (w = 1) and leaves nothing behind."""
    from types import SimpleNamespace

    from agents.training import learner_golden as LG
    m = _learner()
    m._rust_collector = SimpleNamespace(overlap=SimpleNamespace(
        warm=False, active=False, snapshots=SimpleNamespace(has=lambda v: False)))
    h0 = LG.params_sha256(m)
    r = UF.dry_update(m)
    assert r.oom is None and "_overlap_prox" not in vars(m) and LG.params_sha256(m) == h0


def test_the_check_is_inert_off_cuda():
    assert UF.check_update_fits(_learner()) is None


def test_the_dry_update_hands_every_recorded_key_to_the_startup_metric_key_check(monkeypatch):
    """gen3_metric_key_collision_v1: the reading carries every key the dry update recorded, and on CUDA
    `check_update_fits` refuses a colliding set (typed, FATAL_CONFIG) BEFORE the verdict — the CUDA branch
    driven here by a stand-in model and reading (no card needed)."""
    import types

    from agents.training.train_logger import MetricKeyCollisionError
    r = UF.dry_update(_learner())
    assert "train/loss" in r.metric_keys and "train/approx_kl" in r.metric_keys
    planted = UF.FitReading(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 64, "stub",
                            metric_keys=("note/text", "note/text/text_summary"))
    monkeypatch.setattr(UF, "dry_update", lambda m, **_k: planted)
    cuda_model = types.SimpleNamespace(policy=types.SimpleNamespace(
        parameters=lambda: iter([types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))])))
    with pytest.raises(MetricKeyCollisionError, match="dry first update"):
        UF.check_update_fits(cuda_model)


def test_the_trainer_runs_the_check_on_both_startup_paths():
    from main.train import model_build
    src = inspect.getsource(model_build)
    assert src.count("_update_fit.check_update_fits(model, model_dir)") == 2
    assert "check_device_batch_fits" not in src


def test_the_dry_update_runs_the_behaviour_probe_as_the_real_update_does(monkeypatch):
    """Under the Rust core the first real update runs K9(b)'s pre-loop PROBE (an eager grad-mode
    forward); the dry update runs it too — on log-probs it computed itself, so a fatal-mode check
    passes — and restores the check, the row versions and the weights."""
    from agents.training import learner_golden as LG
    from agents.training.rust_rollout import consistency as C
    calls = []
    real = C.behaviour_probe

    def spy(model):
        calls.append(str(model.behaviour_check))
        out = real(model)
        assert out is not None and out["behaviour/max_abs_dlogp_current"] < 1e-4
        return out
    monkeypatch.setattr(C, "behaviour_probe", spy)
    m = _learner()
    m.behaviour_check = "fatal"
    h0 = LG.params_sha256(m)
    UF.dry_update(m)
    assert calls == ["warn"]
    assert m.behaviour_check == "fatal" and getattr(m, "_rust_row_versions", None) is None
    assert LG.params_sha256(m) == h0


def test_the_probe_releases_its_graph():
    """gen3_probe_releases_graph_v1: after the probe no policy stash holds an autograd graph."""
    import numpy as np
    from agents.training.rust_rollout import consistency as C
    m = _learner()
    m.behaviour_check = "warn"
    m._rust_row_versions = np.zeros((int(m.rollout_buffer.buffer_size), int(m.rollout_buffer.n_envs)), np.int64)
    m._rust_version = 0
    C.behaviour_probe(m)
    held = [(type(mod).__name__, k) for mod in m.policy.modules() for k, v in vars(mod).items()
            if isinstance(v, torch.Tensor) and v.grad_fn is not None]
    assert not held and getattr(m.policy, "_last_pi_distribution", None) is None


def test_the_probe_s_forward_is_bit_identical_without_its_saved_tensors():
    """gen3_probe_releases_graph_v1: dropping the saved tensors changes no value — the probe's log-probs
    (and V) are bit-identical to a plain grad-mode forward's."""
    from agents.training.rust_rollout import consistency as C
    from stable_baselines3.common.utils import obs_as_tensor
    m = _learner()
    buf = m.rollout_buffer
    obs = obs_as_tensor({k: v.reshape(-1, *v.shape[2:])[:16] for k, v in buf.observations.items()}, m.device)
    acts = torch.as_tensor(buf.actions.reshape(-1)[:16]).long()
    masks = torch.as_tensor(buf.action_masks.reshape(-1, buf.action_masks.shape[-1])[:16])
    m.policy.set_training_mode(True)
    with torch.enable_grad():
        v0, l0, e0 = m.policy.evaluate_actions(obs, acts, action_masks=masks)
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(C._drop_saved, C._never_unpacked):
        v1, l1, e1 = m.policy.evaluate_actions(obs, acts, action_masks=masks)
    assert torch.equal(l0, l1) and torch.equal(v0, v1) and torch.equal(e0, e1)
    with pytest.raises(RuntimeError, match="never backpropagated"):
        l1.sum().backward()


# ------------------------------------------------------------- gen3_update_fit_v2: the demand is what TRAINING needs
def test_the_demand_is_the_peak_reserved_over_the_update_AND_the_mid_run_canary():
    """Fails on revert: the canary's segment is part of the demand (v1 measured the update alone), the update's own
    demand is reported beside it, and the fragmentation is reserved minus allocated over the whole sequence."""
    peaks = {"forward": [6_000.0, 7_000.0], "probes": [6_500.0, 7_200.0], "update_tail": [1_800.0, 7_200.0],
             "canary": [7_900.0, 8_400.0]}
    d = UF.fit_demand(peaks, ["forward", "probes", "update_tail"], (0.0, 0.0))
    assert (d.demand_mib, d.peak_alloc_mib, d.update_demand_mib) == (8_400.0, 7_900.0, 7_200.0)
    assert (d.canary_demand_mib, d.canary_peak_alloc_mib) == (8_400.0, 7_900.0)
    r = UF.FitReading(d.demand_mib, d.peak_alloc_mib, 8_400.0, 2_000.0, 10_400.0, 2_000.0, 1_200.0, 1.0, 98_304,
                      "fixture", update_demand_mib=d.update_demand_mib, canary_demand_mib=d.canary_demand_mib,
                      canary_peak_alloc_mib=d.canary_peak_alloc_mib)
    assert r.fragmentation_mib == 500.0 and r.as_dict()["fragmentation_mib"] == 500.0
    line = UF.verdict(r, "staged")
    assert "the update 7,200" in line and "the canary's check 8,400 reserved" in line and "emptied cache" in line
    # no compiled learner: no canary segment, the update alone
    d0 = UF.fit_demand({k: v for k, v in peaks.items() if k != "canary"}, ["forward", "probes", "update_tail"],
                       (0.0, 0.0))
    assert d0.demand_mib == 7_200.0 and d0.canary_demand_mib is None


class _SpyCanary:
    def __init__(self, model, seen, note=None):
        self.model, self.seen, self.note = model, seen, note

    def measure(self):
        from agents.training import learner_golden as LG
        self.seen.append(("canary", LG.params_sha256(self.model), self.model.rollout_buffer is self.seen[0][1]))
        return self.note


def test_the_dry_update_measures_the_canary_AFTER_the_update_on_the_RESTORED_learner():
    """gen3_update_fit_v2 step 2: the canary's check runs once, after the dry update, on the learner put back
    exactly (the weights the startup gate passed, the live rollout buffer) — the run's own order at update 10; its
    disagreement text rides on the reading as a NOTE, never a verdict. Fails on revert (no canary call, or one inside
    the dry update)."""
    from agents.training import learner_golden as LG
    m = _learner()
    h0 = LG.params_sha256(m)
    seen = [("buffer", m.rollout_buffer, None)]
    real_train = type(m).train

    def spy(self):
        real_train(self)
        seen.append(("train", LG.params_sha256(self), None))
    m.train = spy.__get__(m)
    r = UF.dry_update(m, canary=_SpyCanary(m, seen, note="a planted disagreement"))
    assert [s[0] for s in seen] == ["buffer", "train", "canary"]
    assert seen[1][1] != h0 and seen[2][1] == h0 and seen[2][2] is True
    assert r.canary_note == "a planted disagreement" and LG.params_sha256(m) == h0
    assert "the dry canary's comparison did not pass" in UF.verdict(
        UF.FitReading(1.0, 1.0, 1.0, 5_000.0, 5_001.0, 5_000.0, 1.0, 1.0, 64, "f", canary_note=r.canary_note))


def test_the_check_hands_the_dry_update_THIS_models_armed_canary(monkeypatch):
    """`check_update_fits` measures with the canary `arm_compile_sentinel` armed for this model, and never with a
    canary armed for another model (a stale control in the process)."""
    import types

    import agents.training.metric_keys as MK
    from agents.model import compile_control as cc
    got = {}
    planted = UF.FitReading(1.0, 1.0, 1.0, 5_000.0, 5_001.0, 5_000.0, 1.0, 1.0, 64, "stub")

    def fake(m, **k):
        got.update(k)
        return planted
    monkeypatch.setattr(UF, "dry_update", fake)
    monkeypatch.setattr(UF, "verdict", lambda *a, **k: "ok")
    monkeypatch.setattr(MK, "check_startup_metric_keys", lambda *a, **k: None)
    cuda_model = types.SimpleNamespace(policy=types.SimpleNamespace(
        parameters=lambda: iter([types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))])))
    saved = cc._CONTROL                                  # restored by hand: the torch-state guard reads it at teardown
    try:
        mine = types.SimpleNamespace(model=cuda_model)
        cc._CONTROL = types.SimpleNamespace(canary=mine)  # type: ignore[assignment]
        UF.check_update_fits(cuda_model)
        assert got["canary"] is mine
        cc._CONTROL = types.SimpleNamespace(canary=types.SimpleNamespace(model=object()))  # type: ignore[assignment]
        UF.check_update_fits(cuda_model)
        assert got["canary"] is None
    finally:
        cc._CONTROL = saved


def test_the_reading_starts_from_an_EMPTIED_cache(monkeypatch):
    """gen3_update_fit_v2 step 1: the cached blocks are handed back BEFORE the floor is read — a startup transient
    held in the allocator's cache is never read as the update's demand (the `rb_xs_s2701` false refusal). Driven
    with a stand-in `torch.cuda` (no card needed); fails on revert."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    UF.release_cached_blocks(torch.device("cuda"))
    assert calls == ["sync", "empty_cache"]
    src = inspect.getsource(UF.dry_update)
    assert src.index("release_cached_blocks(dev)") < src.index("floor = torch.cuda.memory_allocated(dev)")


# ------------------------------------------------ gen3_update_fit_fill_inputs_v1: what only the collector's fill writes
BETA = 0.05


class _MagnetSvc:
    """A stand-in for T2 with a magnet slot (the real one: `rust_rollout/magnet_integration_test.py`): serves the
    masked uniform over each row's legal actions, -inf on the illegal ones (as T2 does), in bounded flushes."""

    def __init__(self, cap: int) -> None:
        from types import SimpleNamespace
        self.spec = SimpleNamespace(max_rows_per_flush=cap)
        self.calls, self.flushes, self.pending = [], 0, 0

    def submit(self, slot, obs, mask, _priority):
        from types import SimpleNamespace

        from agents.training import magnet as M
        assert obs.shape[0] == mask.shape[0] <= self.spec.max_rows_per_flush
        self.calls.append((int(slot), int(obs.shape[0])))
        self.pending += 1
        legal = torch.as_tensor(np.asarray(mask)).to(torch.bool)
        lp = torch.where(legal, M.masked_uniform_logp(legal), torch.full(legal.shape, -float("inf"))).numpy()

        def host():
            assert self.pending == 0, "read before the flush"
            return lp.astype(np.float32), np.zeros(len(lp), np.float32), np.zeros(len(lp), np.int64)
        return SimpleNamespace(host=host)

    def flush(self):
        self.flushes += 1
        self.pending = 0


def _magnet_learner(cap=24, slot=7, collector=True):
    """The golden learner with the trailing magnet ON, as a launch builds it: `magnet.attach` (the magnet = the
    loaded weights), then a collector that declares T2's magnet slot."""
    from types import SimpleNamespace

    from agents.training import magnet as M
    m = _learner()
    m.magnet_kl_coef = BETA
    M.attach(m, model_path=None, same_run=False, saved_coef=0.0, say=lambda _m: None)
    svc = _MagnetSvc(cap)
    if collector:
        m._rust_collector = SimpleNamespace(svc=svc, magnet_slot=slot, overlap=None)
    return m, svc


def _spy_train(m, seen):
    from agents.training import learner_golden as LG
    real_train = type(m).train

    def spy(self):
        col = self.rollout_buffer.magnet_logp
        seen["column"] = None if col is None else np.array(col, copy=True)
        seen["masks"] = np.array(self.rollout_buffer.action_masks, copy=True)
        real_train(self)
        seen["after"] = LG.params_sha256(self)
    m.train = spy.__get__(m)


def test_the_dry_update_with_the_magnet_ON_trains_on_the_column_T2s_magnet_slot_serves_and_restores_exactly():
    """THE INCIDENT (CE-3d fork 1, `rb_x35_ce3dswap_s7321`, 2026-11-11): `--magnet-kl-coef 0.05` died at step 0 in
    the startup fit check — `MagnetError: the rollout buffer has no magnet column` — because the fixture rollout had
    none and only the collector's fill writes it. The dry update now fills its fixture's column the way the fill
    does and trains WITH the term. Fails on revert (the MagnetError)."""
    from agents.training import learner_golden as LG
    from agents.training import magnet as M
    m, svc = _magnet_learner()
    live = m.rollout_buffer
    before = (LG.params_sha256(m), sorted(vars(m)), m._n_updates, m._magnet, m._magnet.step)
    rng = (torch.get_rng_state().clone(), np.random.get_state()[1].copy())
    seen = {}
    _spy_train(m, seen)
    keys = sorted(vars(m))                                # (with the spy bound)
    r = UF.dry_update(m)
    assert r.oom is None and r.rows == 64
    # train() ran, with the magnet's column over the fixture's rows: T2's magnet slot, the declared flush bound
    assert svc.calls == [(7, 24), (7, 24), (7, 16)] and svc.flushes == 3
    col, legal = seen["column"], seen["masks"].astype(bool)
    assert col is not None and col.shape == legal.shape and col.dtype == np.float32
    assert (~legal).any() and (col[~legal] == 0.0).all()                              # 0.0 on every illegal entry
    want = np.broadcast_to(-np.log(legal.sum(-1, keepdims=True).astype(np.float64)), legal.shape)
    np.testing.assert_allclose(col[legal], want[legal], atol=1e-6)                    # what the slot served
    assert M.TAG_KL in r.metric_keys and seen["after"] != before[0]                   # the TERM was in the update
    assert not [k for k in r.metric_keys if k.startswith("magnet/")]                  # no fill meter from a dry pass
    # ... and the learner is exactly the learner before it
    assert (LG.params_sha256(m), m._n_updates, m._magnet, m._magnet.step) == (before[0], *before[2:])
    assert sorted(vars(m)) == keys and m.magnet_kl_coef == BETA
    assert m.rollout_buffer is live and live.magnet_logp is None                      # the live buffer: untouched
    assert not m.logger.name_to_value
    assert torch.equal(torch.get_rng_state(), rng[0]) and np.array_equal(np.random.get_state()[1], rng[1])


def test_the_dry_update_with_the_magnet_ON_is_the_update_a_filled_rollout_takes():
    """Teeth: the dry update's KL is the KL of a real `train()` on the same rows with the same column (the masked
    uniform: > 0 on a near-uniform fresh policy), not a term skipped or zeroed."""
    from agents.training import magnet as M
    m, _svc = _magnet_learner()
    got = {}
    real_train = type(m).train

    def spy(self):
        real_train(self)
        got["kl"] = float(self.logger.name_to_value[M.TAG_KL])
    m.train = spy.__get__(m)
    UF.dry_update(m)
    assert np.isfinite(got["kl"]) and got["kl"] > 5e-3, got


@pytest.mark.parametrize("case, why", [
    ("no collector", "no Rust collector"),
    ("no slot", "declares no T2 magnet slot"),
    ("no magnet", "holds no magnet"),
])
def test_the_dry_update_with_the_magnet_ON_and_nothing_to_serve_its_column_REFUSES_before_train(case, why):
    from types import SimpleNamespace

    from agents.training import learner_golden as LG
    from main.exit_codes import TrainExitCode, exit_code_for
    m, svc = _magnet_learner(collector=(case != "no collector"))
    if case == "no slot":
        m._rust_collector = SimpleNamespace(svc=svc, magnet_slot=None, overlap=None)
    if case == "no magnet":
        m._magnet = None
    live, h0 = m.rollout_buffer, LG.params_sha256(m)
    seen = {}
    _spy_train(m, seen)
    with pytest.raises(UF.FillInputMissing, match="--magnet-kl-coef 0.05 > 0") as e:
        UF.dry_update(m)
    assert why in str(e.value) and "never trains without the run's regulariser" in str(e.value)
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)
    assert not seen and not svc.calls                                                 # BEFORE train(), nothing served
    assert m.rollout_buffer is live and LG.params_sha256(m) == h0 and "_overlap_prox" not in vars(m)


def test_a_lever_that_is_on_and_whose_fill_input_is_not_provided_is_refused_BY_NAME_before_train(monkeypatch):
    """The class, at run time: an entry of `FILL_INPUTS` whose `provide` leaves nothing where `train()` reads it."""
    m = _learner()
    seen = {}
    _spy_train(m, seen)
    monkeypatch.setattr(UF, "FILL_INPUTS", UF.FILL_INPUTS + (UF.FillInput(
        writes="rollout_buffer.zz_column", lever="--zz-lever", on=lambda model: True,
        provide=lambda model, buf: None, present=lambda model, buf: getattr(buf, "zz_column", None) is not None,
        how="a planted entry"),))
    with pytest.raises(UF.FillInputMissing, match=r"--zz-lever is on.*rollout_buffer\.zz_column"):
        UF.dry_update(m)
    assert not seen


def test_a_typed_refusal_out_of_the_dry_updates_train_says_it_came_from_the_FIXTURE_rollout(monkeypatch):
    """The incident's own shape (a lever's `train()` meets a rollout no collector filled — here by removing the
    magnet's entry): the lever's error keeps its type and exit code, and its traceback says where it was raised."""
    from agents.training import magnet as M
    m, _svc = _magnet_learner()
    monkeypatch.setattr(UF, "FILL_INPUTS", tuple(i for i in UF.FILL_INPUTS if "magnet" not in i.writes))
    with pytest.raises(M.MagnetError, match="no magnet column") as e:
        UF.dry_update(m)
    assert any("DRY update" in n and "FILL_INPUTS" in n for n in e.value.__notes__)


def test_the_fixture_carries_no_optional_column_of_the_live_rows():
    m = _learner()
    rb = m.rollout_buffer
    rb.magnet_logp = np.full(rb.action_masks.shape, -2.0, dtype=np.float32)
    rb.is_weights = np.ones(rb.log_probs.shape, dtype=np.float32)
    buf, _src = UF.fixture_buffer(m)
    assert buf.magnet_logp is None and buf.is_weights is None and rb.magnet_logp is not None


def test_the_check_runs_the_dry_update_with_the_magnet_ON(monkeypatch):
    """`check_update_fits`' CUDA branch, driven on the CPU learner (the device test and the card's verdict stubbed,
    everything between them real): the dry update with the lever on, then the startup metric-key check over the keys
    it recorded — `train/magnet_kl` among them. Before the fix no CPU test reached this path with the magnet on."""
    from agents.training import magnet as M
    m, svc = _magnet_learner()
    got = {}
    monkeypatch.setattr(UF, "_on_cuda", lambda model: True)
    monkeypatch.setattr(UF, "verdict", lambda r, mode="?", run_dir=None: got.setdefault("reading", r) and "ok")
    assert UF.check_update_fits(m) == "ok"
    assert M.TAG_KL in got["reading"].metric_keys and got["reading"].oom is None and svc.flushes == 3


def _fill_sources():
    from agents.training import magnet as M
    from agents.training.rust_rollout.collector import RustCollector
    return {"RustCollector._fill": RustCollector._fill, "magnet.fill": M.fill}


def _model_writes_and_callees(fn):
    """What ``fn``'s source ASSIGNS on ``model`` (attribute paths), and every callee it hands ``model`` (or an
    attribute of it) to."""
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    writes, callees = set(), set()
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for t in targets:
            for leaf in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                s = ast.unparse(leaf)
                if s.startswith("model."):
                    writes.add(s[len("model."):])
        if isinstance(node, ast.Call) and any(ast.unparse(a).split(".")[0] == "model" for a in node.args):
            callees.add(ast.unparse(node.func))
    return writes, callees


#: Every callee `RustCollector._fill` hands the learner (or its buffer) to, and what it may write there. A NEW one
#: may write an input `train()` reads: read it, declare its writes in `update_fit` (`FILL_INPUTS`, or
#: `FILL_WRITES_TRAIN_DOES_NOT_REQUIRE` with the reason), then add it here.
_FILL_CALLEES = {
    "self._ensure_buffer": "the buffer itself (its declared shape): the fixture is a copy of it",
    "S.fill_complete": "the buffer's own arrays: the fixture's rows stand in for them",
    "self.overlap.proximal_logp": "reads the buffer; its result is `_overlap_prox`, written by `_fill`",
    "M.fill": "`rollout_buffer.magnet_logp` (parsed below) and the logger's `magnet/*` records",
    "self._record": "the logger only",
    "getattr": "a read",
}


def test_every_write_of_the_collectors_fill_is_declared_to_the_dry_update():
    """THE CLASS behind the incident: an input of `train()` that ONLY the collector's fill writes is absent from a
    fixture rollout, and the dry update runs on CUDA only, so nothing on the CPU sees it. This holds the fill's
    source to `update_fit`'s two tables: a new write on the learner — or a new callee handed the learner — fails
    here until someone has decided what the dry update hands `train()` in its place."""
    src = _fill_sources()
    writes, callees = set(), set()
    for name, fn in src.items():
        w, c = _model_writes_and_callees(fn)
        writes |= w
        callees |= c if name == "RustCollector._fill" else set()
    assert writes >= {"_rust_fill", "_overlap_prox", "rollout_buffer.magnet_logp"}     # the parser sees them
    provided = {i.writes for i in UF.FILL_INPUTS}
    declared = provided | set(UF.FILL_WRITES_TRAIN_DOES_NOT_REQUIRE)
    assert not provided & set(UF.FILL_WRITES_TRAIN_DOES_NOT_REQUIRE)
    assert writes == declared, (
        f"the collector's fill writes {sorted(writes - declared)} that `update_fit` does not declare, and "
        f"`update_fit` declares {sorted(declared - writes)} that the fill no longer writes: an input `train()` reads "
        f"goes in `FILL_INPUTS` (the dry update provides it), any other in `FILL_WRITES_TRAIN_DOES_NOT_REQUIRE`")
    assert callees == set(_FILL_CALLEES), (sorted(callees - set(_FILL_CALLEES)), sorted(set(_FILL_CALLEES) - callees))
    assert all(v.strip() for v in UF.FILL_WRITES_TRAIN_DOES_NOT_REQUIRE.values())
    # the magnet's entry is the fill's OWN pass, never a stand-in, and it is on exactly when the coefficient is
    mag = next(i for i in UF.FILL_INPUTS if i.writes == "rollout_buffer.magnet_logp")
    assert "fill_column" in inspect.getsource(mag.provide) and "masked_uniform" not in inspect.getsource(mag.provide)
    m = _learner()
    assert mag.on(m) is False
    m.magnet_kl_coef = BETA
    assert mag.on(m) is True


def test_the_magnets_pass_is_booked_into_the_demand_as_its_own_segment():
    """On the card the fill's T2 pass is measured, not assumed free: its segment is closed by a mark BEFORE the
    behaviour probe's setup and `train()`'s own `start` mark (which books nothing), so it reaches `fit_demand`."""
    src = inspect.getsource(UF.dry_update)
    assert (src.index("provide_fill_inputs(model, buf)") < src.index("phases.mark(FILL_PHASE)")
            < src.index("_behaviour_probe_setup(model, buf, dump_dir)") < src.index("model.train()"))
    assert src.index("release_cached_blocks(dev)") < src.index("provide_fill_inputs(model, buf)")
    assert UF.FILL_PHASE != "start"
    d = UF.fit_demand({UF.FILL_PHASE: [2_000.0, 9_000.0], "forward": [6_000.0, 7_000.0]},
                      [UF.FILL_PHASE, "forward"], (0.0, 0.0))
    assert d.demand_mib == d.update_demand_mib == 9_000.0
