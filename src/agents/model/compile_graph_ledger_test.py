"""`compile_control.thread_graph_marks` — the per-thread ledger of dynamo graph compiles (gen3_t2_guard_attribution_v1).

CONTRACT tests on the installed torch: they drive REAL dynamo compiles (``backend="eager"``: dynamo traces and
counts a graph, nothing is lowered) on real threads, so a torch that moved ``counters["stats"]["unique_graphs"]`` off
the compiling thread fails here, not in a run. T2's frozen guard reads this ledger
(``agents.inference.service.guard_attribution.attribute_compiles``; its tests build on these facts).
"""
from __future__ import annotations

import collections
import itertools
import threading

import torch

from agents.model import compile_control as CC

_N = itertools.count(5000)


def compile_one_graph() -> None:
    """Compile exactly one fresh dynamo graph on the calling thread. ``eval`` builds a NEW code object each call, so
    dynamo's per-code cache can never turn a second call (or a re-run in the same process) into a hit."""
    fn = eval(f"lambda x: x * {next(_N)} + 1")
    torch.compile(fn, backend="eager")(torch.ones(3))


def on_thread(fn) -> int:
    """Run ``fn`` on a fresh thread, join it, and return that thread's ident (an exception propagates)."""
    box: dict = {}

    def run() -> None:
        box["tid"] = threading.get_ident()
        try:
            fn()
        except BaseException as exc:                         # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=300)
    assert not t.is_alive()
    if "error" in box:
        raise box["error"]
    return int(box["tid"])


def test_a_compile_is_tallied_to_the_thread_that_made_it_and_to_no_other():
    m0 = CC.thread_graph_marks()
    assert m0 is not None
    mine = threading.get_ident()
    other = on_thread(compile_one_graph)
    m1 = CC.thread_graph_marks()
    assert m1 is not None and m1.generation == m0.generation
    assert m1.total - m0.total == 1
    assert m1.by_thread.get(other, 0) - m0.by_thread.get(other, 0) == 1
    assert m1.by_thread.get(mine, 0) == m0.by_thread.get(mine, 0)
    compile_one_graph()                                      # and this thread's own
    m2 = CC.thread_graph_marks()
    assert m2 is not None
    assert m2.total - m1.total == 1 and m2.by_thread[mine] - m1.by_thread.get(mine, 0) == 1
    assert m2.by_thread[other] == m1.by_thread[other]


def test_the_tally_always_adds_up_to_the_real_counter():
    m0 = CC.thread_graph_marks()
    assert m0 is not None
    compile_one_graph()
    on_thread(compile_one_graph)
    on_thread(compile_one_graph)
    m1 = CC.thread_graph_marks()
    assert m1 is not None
    assert m1.total - m0.total == 3 == sum(m1.by_thread.values()) - sum(m0.by_thread.values())
    assert m1.total == CC.dynamo_graphs_total()                # the ledger reads the same counter the guard always did


def test_a_counter_rebuilt_under_the_ledger_is_instrumented_again_and_the_window_is_marked():
    """torch's own test harness (and `main.compile_inventory.capture`) clears ``counters``: the next read re-instruments
    it and reports a NEW generation, so a window that straddled the rebuild is never split on a half-seen tally."""
    from torch._dynamo.utils import counters

    m0 = CC.thread_graph_marks()
    assert m0 is not None
    counters["stats"] = collections.Counter(counters["stats"])     # what a clear-and-refill leaves behind
    m1 = CC.thread_graph_marks()
    assert m1 is not None and m1.generation == m0.generation + 1 and m1.total == m0.total
    other = on_thread(compile_one_graph)                           # and it tallies again
    m2 = CC.thread_graph_marks()
    assert m2 is not None and m2.generation == m1.generation
    assert m2.total - m1.total == 1 and m2.by_thread[other] - m1.by_thread.get(other, 0) == 1


def test_an_increment_that_bypasses_the_tally_shows_as_total_beyond_the_tally():
    """``dict.update`` on an empty Counter (and any C-level write) skips ``__setitem__``: the real counter moves, the tally
    does not, which is what `attribute_compiles` books to T2 (conservatively) as an unplaced graph."""
    from torch._dynamo.utils import counters

    m0 = CC.thread_graph_marks()
    assert m0 is not None
    stats = counters["stats"]
    dict.__setitem__(stats, "unique_graphs", int(dict.get(stats, "unique_graphs", 0)) + 1)
    m1 = CC.thread_graph_marks()
    assert m1 is not None and m1.generation == m0.generation
    assert m1.total - m0.total == 1
    assert sum(m1.by_thread.values()) == sum(m0.by_thread.values())


def test_other_counters_of_the_stats_bucket_are_untouched():
    from torch._dynamo.utils import counters

    CC.thread_graph_marks()
    before = int(counters["stats"]["calls_captured"])
    counters["stats"]["calls_captured"] += 2
    assert counters["stats"]["calls_captured"] == before + 2
    assert type(counters["stats"]).__name__ == "_ThreadGraphStats"
