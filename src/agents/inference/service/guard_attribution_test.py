"""gen3_t2_guard_attribution_v1 — T2's frozen guard books a dynamo compile / a matmul-precision change to T2 only when
T2's OWN thread made it.

The defect (core review 2026-10-20, X14 finding 6): the guard diffed two PROCESS-WIDE values around a flush — dynamo's
graph counter and ``torch.get_float32_matmul_precision()`` — so with X14's overlap on (T2 flushing on the collector's
worker thread while the learner runs on the main thread) a learner recompile or precision change inside a T2 window was a
FATAL ``LifecycleViolation`` on a healthy run: the same false-blame class as C-1, which ``stream_attribution_test.py``
closed for memory.

These tests drive the REAL guard on CPU with REAL threads: "the learner" is another ``threading.Thread`` that compiles a
fresh dynamo graph (``backend="eager"``) or calls ``torch.set_float32_matmul_precision`` inside the window; "T2" is the
thread that entered the guard. Each service-level test FAILS when the guard is reverted to the process-wide reads.
No GPU is involved: the attribution is by thread, and the card's proof is the overlap probe in ``service_cuda_test.py``.
"""
from __future__ import annotations

import contextlib
import itertools
import threading
from typing import Any, Dict

import pytest
import torch

from agents.inference.service import InferenceService, LifecycleViolation, ServiceSpec, SlotGroupSpec
from agents.inference.service import guard_attribution as GA
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.parity import fixture_rows
from agents.model.compile_control import GraphMarks
from agents.model.compile_graph_ledger_test import compile_one_graph, on_thread
from agents.model.inductor_compile import compile_on


@pytest.fixture(scope="module")
def policy():
    return perturbed_fresh_policy(0)


def _svc(policy):
    return InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, policy),), device="cpu",
                                        backend="eager", buckets=(2,))).startup()


@pytest.fixture
def precision():
    """The precision the process had; a test that changes it gets it back (the root conftest fails a leak)."""
    prev = torch.get_float32_matmul_precision()
    assert prev == "highest", "precondition: the suite runs at fp32 'highest'"
    try:
        yield prev
    finally:
        torch.set_float32_matmul_precision(prev)


def _other(value: str) -> None:
    on_thread(lambda: torch.set_float32_matmul_precision(value))


# ---------------------------------------------------------------- dynamo graphs (service level; FAIL on revert)
def test_a_learner_compile_on_another_thread_inside_the_window_is_foreign_not_fatal(policy):
    """The X14 shape: the learner recompiles on its own thread while T2's worker flushes. Before: the process-wide graph
    counter moved inside T2's window -> ``LifecycleViolation`` on a healthy run. Now T2 stays FROZEN and serves."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        on_thread(compile_one_graph)
    st = svc.stats()
    assert st["state"] == "FROZEN" and st["compiles_after_freeze"] == 0
    assert st[GA.FOREIGN_COMPILES_KEY] == 1 and st[GA.COMPILE_FALLBACKS_KEY] == 0


def test_t2s_own_compile_still_poisons_beside_a_foreign_one(policy):
    """(a) the guard is not blinded: in ONE window the learner compiles 2 graphs and T2's own thread compiles 1. T2 is
    charged exactly its own 1 (the message names 1, not 3) and poisons; the learner's 2 are foreign."""
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match=r"1 dynamo graphs, 0 CUDA segments"):
        with svc._frozen_guard("flush"):
            on_thread(compile_one_graph)
            compile_one_graph()
            on_thread(compile_one_graph)
    st = svc.stats()
    assert st["state"] == "POISONED" and st["compiles_after_freeze"] == 1
    assert st[GA.FOREIGN_COMPILES_KEY] == 2 and st[GA.COMPILE_FALLBACKS_KEY] == 0


def test_a_serial_run_is_unchanged_one_thread_means_every_compile_is_t2s(policy):
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match=r"2 dynamo graphs"):
        with svc._frozen_guard("flush"):
            compile_one_graph()
            compile_one_graph()
    assert svc.counters["compiles_after_freeze"] == 2 and svc.counters[GA.FOREIGN_COMPILES_KEY] == 0


def test_a_compile_between_windows_is_in_no_window(policy):
    """A learner compile BETWEEN two T2 windows is neither: the baseline is each window's own start."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass
    on_thread(compile_one_graph)
    compile_one_graph()                                      # even T2's thread, outside any window: not a window's
    with svc._frozen_guard("flush"):
        pass
    assert svc.counters["compiles_after_freeze"] == 0 and svc.counters[GA.FOREIGN_COMPILES_KEY] == 0


def test_an_unavailable_ledger_books_the_window_process_wide_and_counts_it(policy, monkeypatch):
    """The fallback is the old rule (conservative): the whole window is T2's, and it is counted."""
    from agents.inference.service import service as svc_mod

    svc = _svc(policy)
    monkeypatch.setattr(svc_mod, "_graph_marks", lambda: None)
    with pytest.raises(LifecycleViolation, match=r"1 dynamo graphs"):
        with svc._frozen_guard("flush"):
            on_thread(compile_one_graph)
    assert svc.counters[GA.COMPILE_FALLBACKS_KEY] == 1 and svc.counters["compiles_after_freeze"] == 1


def test_a_graph_the_ledger_never_saw_is_booked_to_t2_and_counted(policy):
    """The counter moved without the tally (a C-level write): unplaced, so T2's (conservative) and a fallback."""
    from torch._dynamo.utils import counters

    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match=r"1 dynamo graphs"):
        with svc._frozen_guard("flush"):
            stats = counters["stats"]
            dict.__setitem__(stats, "unique_graphs", int(dict.get(stats, "unique_graphs", 0)) + 1)
    assert svc.counters[GA.COMPILE_FALLBACKS_KEY] == 1


# ---------------------------------------------------------------- the fp32 matmul precision (service level; FAIL on revert)
def test_a_foreign_precision_change_inside_the_window_is_not_fatal(policy, precision, capsys):
    """The learner thread sets the process precision while T2's worker flushes, and leaves it set (the strongest case:
    the exit read still sees the change). The old guard read the global at exit and poisoned T2."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        _other("high")
    st = svc.stats()
    assert st["state"] == "FROZEN" and st[GA.FOREIGN_PRECISION_KEY] == 1 and st[GA.PRECISION_FALLBACKS_KEY] == 0
    assert "[T2] WARN flush: fp32 matmul precision changed after the freeze ('highest' -> 'high')" in capsys.readouterr().err


def test_a_standing_foreign_precision_is_foreign_in_every_later_window_too(policy, precision):
    """The learner's change persists into later windows with nobody setting anything in them: still not T2's."""
    svc = _svc(policy)
    _other("high")
    for _ in range(3):
        with svc._frozen_guard("flush"):
            pass
    assert svc.state == "FROZEN" and svc.counters[GA.FOREIGN_PRECISION_KEY] == 3


def test_a_foreign_change_that_is_set_back_inside_the_window_is_foreign(policy, precision):
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        _other("medium")
        _other("highest")
    assert svc.state == "FROZEN" and svc.counters[GA.FOREIGN_PRECISION_KEY] == 1


def test_t2s_own_precision_change_is_still_fatal(policy, precision):
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match=r"precision changed after the freeze \('highest' -> 'high'\), by T2's own thread"):
        with svc._frozen_guard("flush"):
            torch.set_float32_matmul_precision("high")
    assert svc.state == "POISONED" and svc.counters[GA.FOREIGN_PRECISION_KEY] == 0


def test_t2s_own_precision_change_set_back_inside_the_window_is_still_fatal(policy, precision):
    """The flush ran under another precision in between, which a read at the window's end cannot see."""
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match="by T2's own thread"):
        with svc._frozen_guard("flush"):
            torch.set_float32_matmul_precision("high")
            torch.set_float32_matmul_precision("highest")


def test_t2s_own_precision_change_between_windows_is_fatal_at_the_next_one(policy, precision):
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass
    torch.set_float32_matmul_precision("high")               # T2's thread, outside any window
    with pytest.raises(LifecycleViolation, match="by T2's own thread"):
        with svc._frozen_guard("flush"):
            pass


def test_t2s_own_change_is_seen_beside_a_foreign_one(policy, precision):
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match="by T2's own thread"):
        with svc._frozen_guard("flush"):
            _other("high")
            torch.set_float32_matmul_precision("medium")
    assert svc.state == "POISONED"


def test_a_change_the_ledger_never_saw_is_fatal_like_the_old_rule_and_counted(policy, precision):
    """``torch.backends.cuda.matmul.allow_tf32`` and C++ setters do not pass through the wrapped setter: the setter is
    unknown, so the guard keeps the old rule (fatal) and counts the fallback."""
    svc = _svc(policy)
    raw = torch.set_float32_matmul_precision.__wrapped__      # type: ignore[attr-defined]  # the unwrapped setter
    with pytest.raises(LifecycleViolation, match="by a setter the ledger did not see"):
        with svc._frozen_guard("flush"):
            raw("high")
    assert svc.counters[GA.PRECISION_FALLBACKS_KEY] == 1 and svc.state == "POISONED"


def test_a_healthy_flush_with_the_precision_untouched_reads_clean(policy, precision):
    svc = _svc(policy)
    for _ in range(3):
        with svc._frozen_guard("flush"):
            pass
    st = svc.stats()
    assert st["state"] == "FROZEN" and all(st[k] == 0 for k in GA.GUARD_ATTRIBUTION_KEYS + ("compiles_after_freeze",))


# ---------------------------------------------------------------- the pure pieces
def _marks(gen: int, total: int, **by_thread: int) -> GraphMarks:
    return GraphMarks(gen, total, {int(k[1:]): v for k, v in by_thread.items()})


def test_attribute_compiles_splits_a_window_by_thread():
    a, b = _marks(1, 10, t7=4, t9=6), _marks(1, 15, t7=5, t9=10)       # T2 = thread 7 compiled 1, thread 9 compiled 4
    assert GA.attribute_compiles(a, b, 7, 5) == GA.CompileShare(1, 4, False)
    assert GA.attribute_compiles(a, b, 9, 5) == GA.CompileShare(4, 1, False)
    assert GA.attribute_compiles(a, b, 3, 5) == GA.CompileShare(0, 5, False)    # a thread that compiled nothing
    assert GA.attribute_compiles(a, a, 7, 0) == GA.CompileShare(0, 0, False)


def test_attribute_compiles_books_what_it_cannot_split_process_wide():
    a = _marks(1, 10, t7=4, t9=6)
    assert GA.attribute_compiles(None, a, 7, 3) == GA.CompileShare(3, 0, True)              # no instrument
    assert GA.attribute_compiles(a, None, 7, 3) == GA.CompileShare(3, 0, True)
    assert GA.attribute_compiles(a, _marks(2, 13, t7=4, t9=9), 7, 3) == GA.CompileShare(3, 0, True)   # rebuilt mid-window
    assert GA.attribute_compiles(None, None, 7, 0) == GA.CompileShare(0, 0, False)                    # nothing to split
    assert GA.attribute_compiles(None, None, 7, -1) == GA.CompileShare(0, 0, False)                   # unreadable counter
    # the counter moved by 3, the tally by 1 (thread 9's): 2 unplaced graphs are T2's, conservatively
    assert GA.attribute_compiles(a, _marks(1, 13, t7=4, t9=7), 7, 3) == GA.CompileShare(2, 1, True)


R = GA.PrecisionRead


def test_judge_precision_names_who_is_answerable():
    base = "highest"
    ev = GA.PrecisionEvent
    clean = R("highest", 3, 7)
    assert GA.judge_precision(base, clean, R("highest", 3, 7), [], 7).kind == GA.OK
    # set by thread 9, left set: foreign for T2 = thread 7
    assert GA.judge_precision(base, clean, R("high", 4, 9), [ev(4, 9, "high")], 7).kind == GA.FOREIGN
    # the same change by T2's own thread: own
    assert GA.judge_precision(base, clean, R("high", 4, 7), [ev(4, 7, "high")], 7).kind == GA.OWN
    # set and set back: the final read is clean, the event is not
    assert GA.judge_precision(base, clean, R("highest", 5, 7), [ev(4, 7, "high"), ev(5, 7, "highest")], 7).kind == GA.OWN
    assert GA.judge_precision(base, clean, R("highest", 5, 7), [ev(4, 9, "high"), ev(5, 9, "highest")], 7).kind == GA.FOREIGN
    # a set TO the baseline is not a departure
    assert GA.judge_precision(base, clean, R("highest", 4, 9), [ev(4, 9, "highest")], 7).kind == GA.OK
    # carried in from before the window
    assert GA.judge_precision(base, R("high", 4, 9), R("high", 4, 9), [], 7).kind == GA.FOREIGN
    assert GA.judge_precision(base, R("high", 4, 7), R("high", 4, 7), [], 7).kind == GA.OWN
    # a standing foreign change that T2's own thread then sets to ANOTHER non-baseline value is T2's
    assert GA.judge_precision(base, R("high", 4, 9), R("medium", 5, 7), [ev(5, 7, "medium")], 7).kind == GA.OWN
    # unseen setter (owner None) or a dropped log: the old rule
    assert GA.judge_precision(base, clean, R("high", 3, None), [], 7).kind == GA.UNKNOWN
    assert GA.judge_precision(base, clean, R("highest", 9, 7), None, 7).kind == GA.UNKNOWN
    # own beats unknown beats foreign
    assert GA.judge_precision(base, R("high", 4, 9), R("medium", 5, 7), [ev(5, 7, "medium")], 7).kind == GA.OWN
    assert GA.judge_precision(base, R("high", 4, 9), R("medium", 5, None), [], 7).kind == GA.UNKNOWN


def test_the_ledger_records_the_thread_that_changed_the_value_and_no_no_op(precision):
    GA.PRECISION.install()
    r0 = GA.PRECISION.read()
    other = on_thread(lambda: torch.set_float32_matmul_precision("high"))
    r1 = GA.PRECISION.read()
    assert (r1.value, r1.owner, r1.seq) == ("high", other, r0.seq + 1)
    with pytest.warns(UserWarning):                            # torch WARNS and keeps the value for a bad name
        torch.set_float32_matmul_precision("bogus")
    torch.set_float32_matmul_precision("high")                 # a set to the value it already has
    assert GA.PRECISION.read() == r1                           # neither is an event, and the owner is unchanged
    torch.set_float32_matmul_precision("highest")
    r2 = GA.PRECISION.read()
    assert (r2.value, r2.owner) == ("highest", threading.get_ident())
    assert [(e.tid, e.value) for e in GA.PRECISION.events_after(r0.seq) or []] == [
        (other, "high"), (threading.get_ident(), "highest")]


def test_the_wrap_is_installed_once_and_returns_the_raw_setter_through_wrapped(precision):
    assert GA.PRECISION.install() and GA.PRECISION.install()
    wrapped = torch.set_float32_matmul_precision
    assert getattr(wrapped, GA._HOOK) is GA.PRECISION
    assert not hasattr(wrapped.__wrapped__, GA._HOOK)         # one layer, however many installs
    assert GA.PrecisionLedger().install() is False            # a second ledger never stacks on the first's wrap


def test_the_event_log_is_bounded_and_says_so_when_it_dropped_sets():
    led = GA.PrecisionLedger(maxlen=2)
    for i in range(1, 5):
        led._events.append(GA.PrecisionEvent(i, 1, "high"))
        led._seq = i
    assert led.events_after(2) is not None and len(led.events_after(2) or []) == 2
    assert led.events_after(0) is None                         # events 1 and 2 fell off the end


def test_note_foreign_counts_every_event_and_warns_on_powers_of_two(capsys):
    counters: dict = {}
    for _ in range(5):
        GA.note_foreign(counters, GA.FOREIGN_PRECISION_KEY, "w")
    assert counters[GA.FOREIGN_PRECISION_KEY] == 5
    assert capsys.readouterr().err.count("[T2] WARN w (foreign #") == 3    # 1, 2, 4


def test_the_service_counters_carry_the_attribution_keys(policy):
    st = _svc(policy).stats()
    assert all(st[k] == 0 for k in GA.GUARD_ATTRIBUTION_KEYS)


# ---------------------------------------------------------------- the X14 overlap PROBE (one body, two hosts)
# T2 flushes on a WORKER thread while "the learner" (the thread that calls `run_overlap_probe`) compiles a dynamo graph
# and changes the fp32 matmul precision beside it. Run on CPU by `test_the_overlap_probe_judges_every_leg_on_cpu` below
# (eager backend: the THREAD logic) and, in a fresh process on the card (graph backend, the worker on its own CUDA stream,
# Inductor compiles), by `service_cuda_test.test_overlap_compile_and_precision_are_foreign_not_fatal` — both judged by
# `check_overlap_probe`, so a card verdict can differ from the CPU one only by the hardware, not by the checking.
#
# Legs (`run_overlap_probe` returns each one's counter deltas and the service state after it):
#   healthy            15 steady-state flushes on the worker: nothing booked anywhere.
#   foreign_compile    inside a guard window on the worker the learner compiles a fresh graph: FOREIGN, T2 FROZEN.
#   foreign_precision  inside a window the learner sets the precision and leaves it set: FOREIGN, T2 FROZEN; restored after.
#   healthy_again      3 flushes once restored: nothing booked (a standing foreign change is gone with the change).
#   concurrent         the worker flushes in a loop while the learner compiles 3 graphs and flips the precision and back:
#                      T2 stays FROZEN with 0 own compiles (the foreign counters are timing-dependent: recorded, not judged).
#   own_compile        the WORKER compiles a graph inside its own window: T2's - fatal.


PROBE_KEYS = ("compiles_after_freeze", GA.FOREIGN_COMPILES_KEY, GA.COMPILE_FALLBACKS_KEY,
        GA.FOREIGN_PRECISION_KEY, GA.PRECISION_FALLBACKS_KEY)
PROBE_WAIT_S = 900.0                       # a card compile is minutes; a hung thread must not hang the probe for ever


def run_overlap_probe(device: str = "cpu", backend: str = "eager") -> Dict[str, Any]:
    cuda = device == "cuda"
    rows = 8 if cuda else 2
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, perturbed_fresh_policy(0)),), device=device,
                                       backend=backend, buckets=(rows,))).startup()
    obs, mask = fixture_rows(svc.obs_dim, rows)
    worker_stream = torch.cuda.Stream() if cuda else None       # X14: the worker's OWN stream; the learner keeps the default
    baseline = torch.get_float32_matmul_precision()
    serial = itertools.count(7000)

    def fresh_graph(inductor: bool = False) -> None:
        """Compile a NEW dynamo graph on the calling thread (a fresh code object: never a cache hit). ``inductor`` (the
        learner's compile, on the card only) lowers it through Inductor; else dynamo traces it and stops (``eager``)."""
        fn = eval(f"lambda x: x * {next(serial)} + 1")
        compiled = compile_on(fn, device=device, backend=None if inductor and cuda else "eager")
        compiled(torch.ones(8, device=device))

    def learner_compile() -> None:
        fresh_graph(inductor=True)

    def snap() -> Dict[str, int]:
        st = svc.stats()
        return {k: int(st[k]) for k in PROBE_KEYS}

    def leg(before: Dict[str, int], error: Any) -> Dict[str, Any]:
        after = snap()
        return {"delta": {k: after[k] - before[k] for k in PROBE_KEYS}, "error": error, "state": svc.state}

    def start_worker(body: Any) -> Any:
        box: Dict[str, Any] = {}

        def target() -> None:
            try:
                with torch.cuda.stream(worker_stream) if cuda else contextlib.nullcontext():
                    body()
                if cuda:
                    worker_stream.synchronize()
            except BaseException as exc:                          # noqa: BLE001 — reported in the leg
                box["error"] = f"{type(exc).__name__}: {exc}"

        t = threading.Thread(target=target)
        t.start()
        return t, box

    def window(learner: Any, own: Any = None) -> Any:
        """One guard window on the worker thread; the learner acts INSIDE it (events order it deterministically)."""
        go, done = threading.Event(), threading.Event()

        def body() -> None:
            with svc._frozen_guard("probe"):
                go.set()
                done.wait(PROBE_WAIT_S)
                if own is not None:
                    own()

        t, box = start_worker(body)
        try:
            if go.wait(PROBE_WAIT_S):
                learner()
        finally:
            done.set()
            t.join(PROBE_WAIT_S)
        return box.get("error")

    out: Dict[str, Any] = {"device": device, "backend": backend}
    try:
        b = snap()
        t, box = start_worker(lambda: [svc.score(0, obs, mask) for _ in range(15)])
        t.join(PROBE_WAIT_S)
        out["healthy"] = leg(b, box.get("error"))

        b = snap()
        out["foreign_compile"] = leg(b, window(learner_compile))

        b = snap()
        out["foreign_precision"] = leg(b, window(lambda: torch.set_float32_matmul_precision("high")))
        torch.set_float32_matmul_precision(baseline)

        b = snap()
        t, box = start_worker(lambda: [svc.score(0, obs, mask) for _ in range(3)])
        t.join(PROBE_WAIT_S)
        out["healthy_again"] = leg(b, box.get("error"))

        b, stop = snap(), threading.Event()

        def flush_loop() -> None:
            while not stop.is_set():
                svc.score(0, obs, mask)

        t, box = start_worker(flush_loop)
        try:
            for _ in range(3):
                learner_compile()
            torch.set_float32_matmul_precision("high")
            torch.set_float32_matmul_precision(baseline)
        finally:
            stop.set()
            t.join(PROBE_WAIT_S)
        out["concurrent"] = leg(b, box.get("error"))

        b = snap()
        out["own_compile"] = leg(b, window(lambda: None, own=fresh_graph))
    finally:
        torch.set_float32_matmul_precision(baseline)
    return out


def check_overlap_probe(r: Dict[str, Any]) -> None:
    """Judge every leg of `run_overlap_probe`'s result (raises AssertionError carrying the whole record). Counts are ``>= 1``, not
    ``== 1``, where the learner's compile is an Inductor one on the card: the exact CPU counts are pinned by
    ``guard_attribution_test``."""
    def zero(leg: Dict[str, Any], *keys: str) -> None:
        assert all(leg["delta"][k] == 0 for k in keys), r

    for name in ("healthy", "foreign_compile", "foreign_precision", "healthy_again", "concurrent"):
        leg = r[name]
        assert leg["error"] is None and leg["state"] == "FROZEN", (name, r)
        zero(leg, "compiles_after_freeze", GA.COMPILE_FALLBACKS_KEY, GA.PRECISION_FALLBACKS_KEY)
    for name in ("healthy", "healthy_again"):
        zero(r[name], GA.FOREIGN_COMPILES_KEY, GA.FOREIGN_PRECISION_KEY)
    fc = r["foreign_compile"]["delta"]                                                     # (b) the learner's compile
    assert fc[GA.FOREIGN_COMPILES_KEY] >= 1 and fc[GA.FOREIGN_PRECISION_KEY] == 0, r
    fp = r["foreign_precision"]["delta"]                                                   # (b) the learner's change
    assert fp[GA.FOREIGN_PRECISION_KEY] == 1 and fp[GA.FOREIGN_COMPILES_KEY] == 0, r
    own = r["own_compile"]                                                                 # (a) T2's own compile
    n = own["delta"]["compiles_after_freeze"]
    assert own["state"] == "POISONED" and n >= 1, r
    assert f"{n} dynamo graphs" in (own["error"] or ""), r
    assert own["delta"][GA.FOREIGN_COMPILES_KEY] == 0, r


# ---------------------------------------------------------------- the probe, on CPU
def test_the_overlap_probe_judges_every_leg_on_cpu(precision):
    """The card's probe (`run_overlap_probe`, the one `service_cuda_test` runs on the graph backend) on CPU: a real worker
    thread flushing and flowing through real guard windows while the calling thread compiles and flips the precision.
    Keeps the unrun GPU test's body and its judge honest; the exact CPU counts are asserted here."""
    r = run_overlap_probe("cpu", "eager")
    check_overlap_probe(r)
    assert r["foreign_compile"]["delta"][GA.FOREIGN_COMPILES_KEY] == 1
    assert r["own_compile"]["delta"]["compiles_after_freeze"] == 1 and "1 dynamo graphs" in r["own_compile"]["error"]
    assert torch.get_float32_matmul_precision() == precision
