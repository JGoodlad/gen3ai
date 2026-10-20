"""gen3_t2_stream_attribution_v1 — T2's frozen guard books to T2 only what grew on T2's OWN streams.

The defect (core review 2026-10-20 C-1, X14 finding 6): the guard read PROCESS-WIDE allocator counters, so with X14's
overlap on (T2 flushing on the collector's worker stream while the learner runs ``train()`` on the default stream)
a learner SEGMENT allocated inside a T2 window was a FATAL ``cuda_segments_after_freeze``, and a learner reservation
growth was booked as T2's (fatal once ``RESERVED_GROWTH_POISONS`` is armed). These tests drive the REAL guard on CPU
with a fake allocator (marks + per-stream segments, the two reads ``torch.cuda`` gives the guard); the service-level
ones FAIL on revert to the process-wide reads. The card's proof is ``service_cuda_test.py``'s overlap-attribution test.
"""
from __future__ import annotations

import pytest
import torch

from agents.inference.service import InferenceService, LifecycleViolation, ServiceSpec, SlotGroupSpec
from agents.inference.service import reserved_growth as RG
from agents.inference.service.fixtures import perturbed_fresh_policy

MiB = 2**20
LEARNER = 0          # the legacy default stream: the learner's train() under overlap
T2 = 7               # the overlap worker's own stream (T2's calling stream)
LANE = 11            # a T2 lane stream


class FakeAllocator:
    """A caching allocator's two views: cumulative process-wide marks and per-stream segments."""

    def __init__(self):
        self.segs = {}                  # address -> dict
        self.marks = RG.Marks(0, 0, 0)
        self._next = 0x1000
        self.snapshot_reads = 0
        self.fail_snapshot = False

    def _addr(self):
        self._next += 0x1000
        return self._next

    def expandable(self, stream, nbytes):
        """Map ``nbytes`` more on ``stream``'s expandable segment (segment counter unchanged)."""
        seg = next((g for g in self.segs.values() if g["stream"] == stream and g["is_expandable"]), None)
        if seg is None:
            a = self._addr()
            seg = self.segs[a] = {"stream": stream, "address": a, "total_size": 0, "is_expandable": True}
        seg["total_size"] += nbytes
        m = self.marks
        self.marks = RG.Marks(m.segments, m.reserved + nbytes, m.released)

    def segment(self, stream, nbytes):
        """A new ``cudaMalloc`` segment on ``stream`` (the default allocator, no expandable segments)."""
        a = self._addr()
        self.segs[a] = {"stream": stream, "address": a, "total_size": nbytes, "is_expandable": False}
        m = self.marks
        self.marks = RG.Marks(m.segments + 1, m.reserved + nbytes, m.released)
        return a

    def release(self, stream, nbytes):
        """Unmap ``nbytes`` of ``stream``'s expandable segment (``empty_cache`` / an OOM retry's cache release)."""
        seg = next(g for g in self.segs.values() if g["stream"] == stream and g["is_expandable"])
        seg["total_size"] -= nbytes
        m = self.marks
        self.marks = RG.Marks(m.segments, m.reserved, m.released + nbytes)

    def snapshot(self):
        self.snapshot_reads += 1
        if self.fail_snapshot:
            raise RuntimeError("snapshot unavailable")
        return [dict(g) for g in self.segs.values()]


@pytest.fixture
def fake(monkeypatch):
    from agents.inference.service import service as svc_mod

    fa = FakeAllocator()
    fa.expandable(LEARNER, 512 * MiB)            # the learner's and T2's footprints at the freeze
    fa.expandable(T2, 16 * MiB)
    fa.expandable(LANE, 8 * MiB)
    monkeypatch.setattr(svc_mod, "_alloc_marks", lambda device: fa.marks)
    monkeypatch.setattr(svc_mod, "_alloc_segments", lambda device: fa.snapshot())
    monkeypatch.setattr(svc_mod, "_t2_streams", lambda engine, device: frozenset({T2, LANE}))
    return fa


@pytest.fixture(scope="module")
def policy():
    return perturbed_fresh_policy(0)


def _svc(policy):
    return InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 1, policy),), device="cpu",
                                        backend="eager", buckets=(2,))).startup()


# ---------------------------------------------------------------- the guard (service level; FAIL on revert)
def test_a_learner_segment_inside_a_worker_flush_is_foreign_not_a_t2_fatal(policy, fake):
    """C-1: without ``expandable_segments`` the learner's ``cudaMalloc`` during an overlapped flush was booked to T2
    as a new segment -> LifecycleViolation on a healthy run. Now it is FOREIGN; T2 stays FROZEN and serves."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        fake.segment(LEARNER, 20 * MiB)
    st = svc.stats()
    assert st["state"] == "FROZEN"
    assert st["cuda_segments_after_freeze"] == 0 and st["cuda_reserved_bytes_after_freeze"] == 0
    assert st[RG.FOREIGN_SEGMENTS_KEY] == 1 and st[RG.FOREIGN_BYTES_KEY] == 20 * MiB


def test_a_learner_reservation_growth_is_not_booked_to_t2_even_when_armed(policy, fake, monkeypatch):
    """X14 finding 6 (the ``expandable_segments`` shape): the learner maps 64 MiB on the default stream while the
    worker flushes. With the switch ARMED the old guard poisoned T2; now T2's counter reads 0 and nothing fails."""
    monkeypatch.setattr(RG, "RESERVED_GROWTH_POISONS", True)
    svc = _svc(policy)
    for _ in range(3):
        with svc._frozen_guard("flush"):
            fake.expandable(LEARNER, 64 * MiB)
    st = svc.stats()
    assert st["state"] == "FROZEN"
    assert st[RG.BYTES_KEY] == 0 and st[RG.GROWTHS_KEY] == 0
    assert st[RG.FOREIGN_BYTES_KEY] == 3 * 64 * MiB


def test_t2s_own_growth_is_still_seen_beside_a_learner_growth(policy, fake, capsys):
    """(a) the guard is not blinded: in ONE window the learner maps 64 MiB and T2 maps 2 MiB on its worker stream
    and 1 MiB on a lane. T2 is booked exactly its 3 MiB (warned), the learner's 64 MiB is foreign."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        fake.expandable(LEARNER, 64 * MiB)
        fake.expandable(T2, 2 * MiB)
        fake.expandable(LANE, 1 * MiB)
    st = svc.stats()
    assert st[RG.BYTES_KEY] == 3 * MiB and st[RG.GROWTHS_KEY] == 1
    assert st[RG.FOREIGN_BYTES_KEY] == 64 * MiB
    assert "[T2] WARN flush grew the CUDA reservation by 3.0 MiB" in capsys.readouterr().err


def test_t2s_own_new_segment_still_poisons(policy, fake):
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match="1 CUDA segments"):
        with svc._frozen_guard("flush"):
            fake.segment(LEARNER, 4 * MiB)                  # foreign: not the cause
            fake.segment(LANE, 2 * MiB)                     # T2's lane: fatal
    assert svc.state == "POISONED"
    assert svc.counters["cuda_segments_after_freeze"] == 1 and svc.counters[RG.FOREIGN_SEGMENTS_KEY] == 1


def test_t2s_own_growth_poisons_once_armed(policy, fake, monkeypatch):
    monkeypatch.setattr(RG, "RESERVED_GROWTH_POISONS", True)
    svc = _svc(policy)
    with pytest.raises(LifecycleViolation, match=f"by {2 * MiB} bytes"):
        with svc._frozen_guard("flush"):
            fake.expandable(LEARNER, 64 * MiB)
            fake.expandable(T2, 2 * MiB)
    assert svc.state == "POISONED"


def test_a_steady_state_takes_no_snapshot(policy, fake):
    """The cost bar: when nothing is mapped anywhere, the windows read the marks only."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass                                                # the first window reads its baseline
    reads = fake.snapshot_reads
    for _ in range(50):
        with svc._frozen_guard("flush"):
            pass
    assert fake.snapshot_reads == reads
    assert svc.counters[RG.SNAPSHOTS_KEY] == reads


def test_a_release_between_windows_re_reads_the_baseline(policy, fake):
    """An ``empty_cache`` between two windows unmaps 8 MiB of T2's stream; T2's next flush maps 8 MiB back. That
    IS an acquisition after the freeze (the process-wide rule counts it too), and it is seen only because ``enter``
    re-reads the baseline when the marks moved — a stale baseline would hide it."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass
    fake.release(T2, 8 * MiB)
    with svc._frozen_guard("flush"):
        fake.expandable(T2, 8 * MiB)
    assert svc.counters[RG.BYTES_KEY] == 8 * MiB


def test_growth_between_windows_is_not_booked_to_the_next_window(policy, fake):
    """Something maps 4 MiB on T2's worker stream BETWEEN two T2 windows (the worker's own non-T2 code), and the
    learner maps 64 MiB inside the next window (so it is read). T2 is charged nothing: the baseline was re-read at
    the window's start. A stale baseline would book the 4 MiB to T2."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass
    fake.expandable(T2, 4 * MiB)
    with svc._frozen_guard("flush"):
        fake.expandable(LEARNER, 64 * MiB)
    assert svc.counters[RG.BYTES_KEY] == 0 and svc.counters[RG.FOREIGN_BYTES_KEY] == 64 * MiB


def test_an_unreadable_snapshot_books_the_window_process_wide(policy, fake):
    """The fallback is today's rule (conservative): the whole window is T2's, and it is counted."""
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        pass
    fake.fail_snapshot = True
    with pytest.raises(LifecycleViolation, match="1 CUDA segments"):
        with svc._frozen_guard("flush"):
            fake.segment(LEARNER, 4 * MiB)
    assert svc.counters[RG.FALLBACKS_KEY] == 1


def test_a_window_that_grew_and_released_is_flagged(policy, fake):
    svc = _svc(policy)
    with svc._frozen_guard("flush"):
        fake.release(LEARNER, 32 * MiB)
        fake.expandable(LEARNER, 64 * MiB)
    assert svc.counters[RG.RELEASED_KEY] == 1 and svc.counters[RG.BYTES_KEY] == 0


def test_t2s_page_rounded_growth_lies_in_the_card_tests_band(policy, fake):
    """The card test's (a) band (``service_cuda_test.t2_own_growth_band``) must admit a CORRECT guard's reading. torch
    2.8 maps an ``expandable_segments`` large-pool segment in whole 20 MiB pages (``ExpandableSegment::map``), so T2's
    128 MiB on its worker stream (no free mapped slack: a graph flush allocates nothing there) maps 7 pages = 140 MiB,
    and the learner's 256 MiB maps 13 = 260 MiB. The old bar (``<= 128 MiB``) failed this healthy window on the card."""
    from agents.inference.service.service_cuda_test import t2_own_growth_band

    svc = _svc(policy)
    with svc._frozen_guard("probe"):
        fake.expandable(LEARNER, 13 * 20 * MiB)      # the learner's 256 MiB request
        fake.expandable(T2, 7 * 20 * MiB)            # T2's 128 MiB request
    own = svc.counters[RG.BYTES_KEY]
    assert own == 140 * MiB and svc.counters[RG.FOREIGN_BYTES_KEY] == 260 * MiB
    lo, hi = t2_own_growth_band(128 * MiB, 16 * MiB)
    assert lo <= own <= hi
    assert hi < 13 * 20 * MiB                        # still excludes the learner's bytes: the band discriminates


# ---------------------------------------------------------------- the pure pieces
def test_stream_ledger_sums_mapped_ranges_and_names_only_malloc_segments():
    led = RG.stream_ledger([
        {"stream": 0, "address": 1, "total_size": 10, "is_expandable": True},
        {"stream": 0, "address": 2, "total_size": 5, "is_expandable": True},     # a second mapped range
        {"stream": 0, "address": 3, "total_size": 7, "is_expandable": False},
        {"stream": 9, "address": 4, "total_size": 2}])                           # key absent: not expandable
    assert led == {0: (22, frozenset({3})), 9: (2, frozenset({4}))}


def test_a_stream_born_inside_the_window_is_all_growth():
    fa = FakeAllocator()
    counters = {}
    attr = RG.StreamAttribution(fa.snapshot, counters)
    m0 = fa.marks
    attr.enter(m0)
    fa.expandable(T2, 6 * MiB)                  # the worker stream's FIRST allocation
    out = attr.exit(m0, fa.marks, frozenset({T2}))
    assert out == RG.Attributed(6 * MiB, 0, 0, 0)


def test_t2_streams_are_the_lanes_and_the_callers_current_stream(monkeypatch):
    from types import SimpleNamespace

    from agents.inference.service import service as svc_mod

    monkeypatch.setattr(svc_mod.torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=T2))
    eng = SimpleNamespace(streams=[SimpleNamespace(cuda_stream=LANE), SimpleNamespace(cuda_stream=13)])
    assert svc_mod._t2_streams(eng, torch.device("cuda", 0)) == frozenset({T2, LANE, 13})
    assert svc_mod._t2_streams(eng, torch.device("cpu")) == frozenset()


def test_alloc_segments_keeps_only_the_services_device(monkeypatch):
    from agents.inference.service import service as svc_mod

    segs = [{"device": 0, "stream": 0, "address": 1, "total_size": 1},
            {"device": 1, "stream": 0, "address": 2, "total_size": 1}]
    monkeypatch.setattr(svc_mod.torch.cuda, "memory_snapshot", lambda *a, **k: segs)
    assert svc_mod._alloc_segments(torch.device("cuda", 1)) == [segs[1]]
    assert svc_mod._alloc_segments(torch.device("cpu")) == []


def test_new_mapping_ceiling_rounds_a_request_up_to_whole_large_pages():
    assert RG.EXPANDABLE_LARGE_PAGE == 20 * MiB                                  # torch 2.8's kLargeBuffer
    assert [RG.new_mapping_ceiling(n * MiB) for n in (128, 120, 256, 21)] == [140 * MiB, 120 * MiB, 260 * MiB, 40 * MiB]
    assert RG.new_mapping_ceiling(1) == 20 * MiB
