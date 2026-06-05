"""Unit tests for the idempotent reconcile loop (DistilledOpponentManager).

A tiny in-memory ``Sim`` plays the role of the disk + the distill subprocesses, so the pure
reconcile *logic* (backfill, steady-state no-op, new promotion, eviction, capacity escalation,
exhaustion, concurrency, min-pool fallback) is tested without GPU/bridge.
"""
from agents.training.distill.manager import DistilledOpponentManager, DEFAULT_LADDER


class Sim:
    def __init__(self):
        self.ready = set()          # steps with a gate-PASSED distilled.pt
        self.artifacts = set()      # steps with ANY artifact
        self._pending = {}          # handle(step) -> None (running) | result dict
        self.removed = []

    # ── manager hooks ──
    def ready_steps(self):
        return set(self.ready)

    def list_artifacts(self):
        return set(self.artifacts)

    def run(self, step, size):
        self._pending[step] = None  # spawned, running
        return step                 # handle == step

    def poll(self, handle):
        return self._pending.get(handle)

    def remove(self, step):
        self.removed.append(step)
        self.ready.discard(step)
        self.artifacts.discard(step)

    # ── test control: complete an in-flight job ──
    def finish(self, step, passed=True, speedup=4.0):
        self._pending[step] = {"passed": passed, "speedup": speedup}
        if passed and speedup >= 2.0:
            self.ready.add(step)
            self.artifacts.add(step)


def _mgr(sim, **kw):
    return DistilledOpponentManager(
        ready_steps_fn=sim.ready_steps, list_artifacts_fn=sim.list_artifacts,
        run_distill_fn=sim.run, poll_fn=sim.poll, remove_fn=sim.remove, **kw)


def _drain(mgr, sim, active, ticks=10):
    """Reconcile + finish all in-flight jobs (passing) until steady, simulating async completion."""
    last = None
    for _ in range(ticks):
        last = mgr.reconcile(active)
        for step in list(mgr._jobs):
            sim.finish(step)        # all pass
    return last


def test_empty_pool_is_noop():
    sim = Sim(); mgr = _mgr(sim)
    r = mgr.reconcile([])
    assert r.spawned == [] and r.n_active == 0 and not r.all_distilled and r.use_full


def test_backfill_then_steady_then_noop():
    sim = Sim(); mgr = _mgr(sim, max_concurrent=4)
    active = {1, 2, 3, 4, 5}
    r = mgr.reconcile(active)                 # backfill: 5 missing, cap 4
    assert len(r.spawned) == 4 and not r.all_distilled and r.use_full
    r = _drain(mgr, sim, active)              # finish all, including the 5th
    assert r.all_distilled and not r.use_full
    assert r.frac_distilled == 1.0 and r.sampleable == active
    # IDEMPOTENT: nothing to do -> no-op
    r2 = mgr.reconcile(active)
    assert r2.spawned == [] and r2.all_distilled and r2.n_running == 0


def test_new_promotion_dips_then_recovers():
    sim = Sim(); mgr = _mgr(sim)
    active = {1, 2, 3}
    _drain(mgr, sim, active)
    assert mgr.reconcile(active).all_distilled
    active2 = {1, 2, 3, 4}                     # a promotion
    r = mgr.reconcile(active2)
    assert r.spawned == [4] and not r.all_distilled   # one missing -> not 100%
    r = _drain(mgr, sim, active2)
    assert r.all_distilled                    # back to steady once 4 is distilled


def test_eviction_cleans_up():
    sim = Sim(); mgr = _mgr(sim)
    active = {1, 2, 3}
    _drain(mgr, sim, active)
    mgr.reconcile({2, 3})                      # 1 slid out of the window
    assert 1 in sim.removed and 1 not in sim.ready


def test_capacity_escalation_climbs_ladder():
    sim = Sim(); mgr = _mgr(sim, min_pool=1)
    active = {1}
    mgr.reconcile(active)                      # spawn rung 0
    assert mgr._job_rung[1] == 0
    sim.finish(1, passed=False)               # gate fail
    r = mgr.reconcile(active)                  # harvest fail -> re-spawn rung 1
    assert mgr._job_rung[1] == 1 and r.spawned == [1] and not r.all_distilled
    sim.finish(1, passed=True)                # passes at the bigger size
    r = mgr.reconcile(active)
    assert r.all_distilled


def test_speedup_floor_triggers_escalation():
    sim = Sim(); mgr = _mgr(sim, min_speedup=2.0)
    mgr.reconcile({1})
    sim.finish(1, passed=True, speedup=1.5)   # faithful but too slow -> not deployable
    r = mgr.reconcile({1})
    assert mgr._job_rung[1] == 1 and 1 not in sim.ready   # escalated, not deployed


def test_exhaustion_drops_snapshot_but_keeps_rest_distilled():
    sim = Sim(); mgr = _mgr(sim, min_pool=1)
    active = {1, 2}
    # distill 2 successfully
    mgr.reconcile(active); sim.finish(2, passed=True)
    # fail 1 at every ladder rung -> exhausted
    for _ in range(len(DEFAULT_LADDER)):
        mgr.reconcile(active)
        if 1 in mgr._jobs:
            sim.finish(1, passed=False)
    r = mgr.reconcile(active)
    assert 1 in mgr._exhausted
    assert r.all_distilled                     # the rest (just {2}) is fully distilled
    assert r.sampleable == {2} and 1 not in r.sampleable   # exhausted snapshot not sampled


def test_too_few_deployable_falls_back_to_full():
    sim = Sim(); mgr = _mgr(sim, min_pool=2)
    active = {1, 2}
    mgr.reconcile(active); sim.finish(2, passed=True)
    for _ in range(len(DEFAULT_LADDER)):
        mgr.reconcile(active)
        if 1 in mgr._jobs:
            sim.finish(1, passed=False)
    r = mgr.reconcile(active)
    # only {2} deployable but min_pool=2 -> not enough -> full pool (samples everything)
    assert not r.all_distilled and r.use_full and r.sampleable == active


def test_concurrency_cap_respected():
    sim = Sim(); mgr = _mgr(sim, max_concurrent=2)
    r = mgr.reconcile({1, 2, 3, 4, 5})
    assert len(r.spawned) == 2 and r.n_running == 2


def test_harvested_reports_deploy_and_is_empty_when_idle():
    # The Events panel feeds off ReconcileResult.harvested — one entry per job that finished
    # THIS tick, empty otherwise (so no spurious event lines on a steady no-op).
    sim = Sim(); mgr = _mgr(sim, min_pool=1)
    active = {1}
    r = mgr.reconcile(active)                       # spawn rung 0; nothing finished yet
    assert r.harvested == []
    sim.finish(1, passed=True, speedup=4.0)
    r = mgr.reconcile(active)                       # harvest -> deployed
    assert [h["action"] for h in r.harvested] == ["deployed"]
    assert r.harvested[0]["step"] == 1 and r.harvested[0]["passed"] and r.harvested[0]["speedup"] == 4.0
    assert mgr.reconcile(active).harvested == []    # steady-state: no new events


def test_harvested_reports_escalation_then_exhaustion():
    sim = Sim(); mgr = _mgr(sim, min_pool=1)
    active = {1}
    actions: list = []
    for _ in range(len(DEFAULT_LADDER) + 1):
        r = mgr.reconcile(active)
        actions += [h["action"] for h in r.harvested]
        if 1 in mgr._jobs:
            sim.finish(1, passed=False)            # fail at the current rung
    assert actions.count("escalated") == len(DEFAULT_LADDER) - 1
    assert actions.count("exhausted") == 1


def test_harvested_escalation_carries_next_rung():
    sim = Sim(); mgr = _mgr(sim, min_pool=1)
    mgr.reconcile({1})                              # spawn rung 0
    sim.finish(1, passed=False)
    r = mgr.reconcile({1})                          # harvest fail -> escalate to rung 1
    h = r.harvested[0]
    assert h["action"] == "escalated" and h["next_rung"] == 1 and h["rung"] == 0 and not h["passed"]


def test_recover_escalation_from_failed_manifests():
    # restart-safe: a manager rebuilt with failed manifests resumes the ladder instead of rung 0
    sim = Sim()
    failed = {1: {"config": DEFAULT_LADDER[0]},     # failed rung 0 -> retry rung 1
              2: {"config": DEFAULT_LADDER[-1]}}    # failed last rung -> exhausted
    mgr = _mgr(sim, min_pool=1, recover_fn=lambda: failed)
    assert mgr._next_rung.get(1) == 1 and 2 in mgr._exhausted
    r = mgr.reconcile({1, 2})
    assert mgr._job_rung.get(1) == 1               # 1 re-spawns at the recovered rung
    assert 2 not in mgr._jobs and r.n_exhausted == 1  # 2 isn't retried
