"""``gen3_t2_reserved_after_freeze_v1`` downstream: the rollout collector's and the eval executor's
``check_lifecycle`` read T2's reserved-byte growth (the one acquisition visible under
``expandable_segments:True``) through the SAME helper as T2's own guard
(``agents.inference.service.reserved_growth``): counted, warned at 1, 2, 4, 8, ..., never fatal while
``RESERVED_GROWTH_POISONS`` is off. CPU only — the T2 counters are a plain dict, the core a stub."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import pytest

from agents.inference.service import reserved_growth as RG
from agents.training.rust_eval.executor import RustEvalCore
from agents.training.rust_rollout.collector import LifecycleViolation, RustCollector

MiB = 2**20
ZERO_T2 = {"compiles_after_freeze": 0, "captures_after_freeze": 0, "cuda_segments_after_freeze": 0,
           RG.BYTES_KEY: 0, RG.GROWTHS_KEY: 0}


class _Core:
    def __init__(self):
        self.cols, self.n = {}, 2
        self.counters: Dict[str, int] = {"THREADS_SPAWNED_AFTER_FREEZE": 0, "PROC_SPAWNS_AFTER_FREEZE": 0}

    def after_freeze(self) -> Dict[str, int]:
        return dict(self.counters)


def _collector(svc_counters):
    core = _Core()
    # the collector's __init__ needs a full core / opponents / stager; check_lifecycle reads only these
    col = RustCollector.__new__(RustCollector)
    col.core, col.svc, col.cfg = core, SimpleNamespace(counters=svc_counters), SimpleNamespace(respawn_budget=0)
    col._t2_reserved = RG.ReservedGrowthWatch("the rollout collector's lifecycle check: T2")
    return col


def _eval_core(svc_counters):
    return RustEvalCore(core=_Core(), svc=SimpleNamespace(counters=svc_counters), lib=None, table=None,
                        team_table=None, turn_limit=100, trainee_builder=None, opp_builder=None,
                        fixed_builders={}, names=())


BUILDERS = {"collector": _collector, "eval": _eval_core}


# ---- the shared helper ---------------------------------------------------------------------------------
def test_note_growth_counts_every_growth_and_warns_on_the_doubling_schedule(capsys):
    c: Dict[str, int] = {}
    assert RG.note_growth(c, "w", 0) is None and RG.note_growth(c, "w", -5) is None
    assert c == {}                                                  # no growth: nothing counted
    for i in range(9):
        assert RG.note_growth(c, "w", MiB) is None
    assert c == {RG.BYTES_KEY: 9 * MiB, RG.GROWTHS_KEY: 9}
    err = capsys.readouterr().err
    assert [f"growth #{n}," in err for n in range(1, 10)] == [n in (1, 2, 4, 8) for n in range(1, 10)]


def test_note_growth_returns_the_violation_once_armed(monkeypatch, capsys):
    monkeypatch.setattr(RG, "RESERVED_GROWTH_POISONS", True)
    c: Dict[str, int] = {}
    assert RG.note_growth(c, "w", 3 * MiB) == (f"w grew the CUDA reservation by {3 * MiB} bytes after "
                                               "the freeze (the declared lifecycle allows none)")
    assert c[RG.GROWTHS_KEY] == 1 and capsys.readouterr().err == ""


def test_the_watch_sees_only_the_bytes_since_its_last_check():
    w = RG.ReservedGrowthWatch("w")
    for now in (0, 4 * MiB, 4 * MiB, 6 * MiB):
        assert w.check({RG.BYTES_KEY: now}) is None
    assert w.counters == {RG.BYTES_KEY: 6 * MiB, RG.GROWTHS_KEY: 2}  # the 3rd check saw nothing new


# ---- the two downstream lifecycle checks ---------------------------------------------------------------
@pytest.mark.parametrize("site", sorted(BUILDERS))
def test_check_lifecycle_counts_and_warns_on_t2_reserved_growth_and_never_raises(site, capsys):
    """The expandable_segments shape: T2's segment counter stays 0 while its reserved counter grows.
    Every check passes, returns the reserved pair, and warns on checks 1, 2 and 4 that saw growth."""
    t2 = dict(ZERO_T2)
    obj = BUILDERS[site](t2)
    assert obj.check_lifecycle()[RG.BYTES_KEY] == 0
    for i in range(1, 6):
        t2[RG.BYTES_KEY] += i * MiB
        t2[RG.GROWTHS_KEY] += 1
        life = obj.check_lifecycle()
    assert life[RG.BYTES_KEY] == 15 * MiB and life[RG.GROWTHS_KEY] == 5
    assert life["cuda_segments_after_freeze"] == 0
    err = capsys.readouterr().err
    assert err.count("lifecycle check: T2 grew the CUDA reservation") == 3
    assert "growth #1," in err and "growth #2," in err and "growth #4," in err and "growth #3," not in err


@pytest.mark.parametrize("site", sorted(BUILDERS))
def test_check_lifecycle_raises_on_reserved_growth_once_armed(site, monkeypatch):
    monkeypatch.setattr(RG, "RESERVED_GROWTH_POISONS", True)
    t2 = dict(ZERO_T2)
    obj = BUILDERS[site](t2)
    obj.check_lifecycle()                                           # no growth: passes armed too
    t2[RG.BYTES_KEY] = MiB
    with pytest.raises(LifecycleViolation, match=f"grew the CUDA reservation by {MiB} bytes"):
        obj.check_lifecycle()


@pytest.mark.parametrize("site", sorted(BUILDERS))
def test_a_t2_segment_still_raises_unchanged(site):
    t2 = {**ZERO_T2, "cuda_segments_after_freeze": 1, RG.BYTES_KEY: 2 * MiB}
    with pytest.raises(LifecycleViolation, match="cuda_segments_after_freeze"):
        BUILDERS[site](t2).check_lifecycle()
