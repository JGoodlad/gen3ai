"""``utils.heavy_admission``: the memory-aware admission rule of the heavy slots, as arithmetic on fakes (the slots and the
queue against REAL processes are ``heavy_slots_admission_test.py``).

Load-bearing pins, each FAILS on revert of what it guards:
* a capped job is admitted while the caps fit, on the HIGHEST free slot; one byte over the budget waits (exact integers);
* the hard maximum on the count; a job alone always runs;
* FIFO: a job waits behind a queued job whose cap does not fit, and needs room for every job ahead of it;
* EVERY unknown is the fixed count of 2: no declared cap, no budget, a holder with no known cap, a job without a cap ahead;
* an old-code waiter ahead holds a capped job back only while a base slot is free;
* a holder is charged the larger of its declared cap and its scopes; a scope nobody holds a slot for is charged too;
* the box read: the budget is min(slice cap, MemTotal - reserve), an empty scope is not counted, a scope with no limit
  or an unreadable MemTotal makes the budget unknown;
* the switch back: the file, the environment over it, anything but ``memory`` reads as ``count``.
"""
from __future__ import annotations

import pytest

from utils import heavy_admission as HA

G = HA.GiB


@pytest.fixture(autouse=True)
def _no_ambient(monkeypatch):
    for k in (HA.MODE_ENV, HA.MAX_SLOTS_ENV, HA.RESERVE_ENV, HA.MEMINFO_ENV, HA.CGROUP_ENV, HA.SLICE_ENV, HA.SLICE_GB_ENV):
        monkeypatch.delenv(k, raising=False)


def _decide(cap, *, free, held=(), other=0, budget=62 * G, ahead=()):
    return HA.decide(None if cap is None else cap * G, free=free, held=[None if c is None else c * G for c in held],
                     other=other * G, budget=budget, ahead=[None if a is None else a * G for a in ahead])


# ── the memory rule ──────────────────────────────────────────────────────────────────────────────────────────

def test_a_capped_job_runs_beside_two_others_while_the_caps_fit_and_takes_the_highest_free_slot():
    d = _decide(12, free=[0, 1], held=[12, 24])
    assert (d.slot, d.rule) == (1, HA.RULE_MEMORY), "the third job waited although 48 of 62 GiB fit (the fixed count of 2)"
    assert _decide(12, free=[0, 1, 2, 3]).slot == 3, "a capped job must leave the base slots to the old code"
    assert _decide(12, free=[1, 3], held=[12, 12]).slot == 3


def test_a_cap_that_does_not_fit_waits_and_the_boundary_is_exact():
    assert _decide(24, free=[0, 1], held=[24, 24]).slot is None                      # 72 > 62
    assert HA.decide(14 * G, free=[0], held=[24 * G, 24 * G], other=0, budget=62 * G, ahead=[]).slot == 0   # = 62
    over = HA.decide(14 * G + 1, free=[0], held=[24 * G, 24 * G], other=0, budget=62 * G, ahead=[])
    assert over.slot is None and over.rule == HA.RULE_MEMORY and "does not fit" in over.why


def test_the_count_has_a_hard_maximum_however_much_memory_is_left():
    d = _decide(1, free=[], held=[1, 1, 1, 1])
    assert d.slot is None and d.rule == HA.RULE_MEMORY
    assert HA.MAX_SLOTS == 4 and HA.BASE_SLOTS == 2 and HA.max_slots() == 4


def test_a_job_alone_always_runs_even_past_the_budget():
    assert _decide(80, free=[0, 1, 2, 3]).slot == 3, "a job bigger than the budget would starve"
    assert _decide(12, free=[0, 1, 2, 3], other=60).slot == 3, "capped jobs holding no slot must not freeze the queue"
    assert _decide(12, free=[0, 1, 2], held=[12], other=60).slot is None             # not alone: 84 > 62


def test_a_scope_that_holds_no_slot_is_committed_memory():
    assert _decide(12, free=[0, 1, 2], held=[12], other=40).slot is None             # 64 > 62
    assert _decide(12, free=[0, 1, 2], held=[12], other=38).slot == 2                # = 62


# ── FIFO ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_job_waits_behind_a_queued_job_whose_cap_does_not_fit():
    d = _decide(12, free=[0, 1], held=[24, 24], ahead=[24])
    assert d.slot is None and "queued behind" in d.why, "a small job passed a big one: big jobs would starve"
    assert _decide(12, free=[0, 1], held=[24, 24]).slot == 1                         # control: nobody ahead, 60 fits


def test_a_job_needs_room_for_every_job_ahead_of_it():
    assert _decide(12, free=[0, 1], held=[12, 12], ahead=[12]).slot == 0             # 48 fits; the head gets slot 1
    assert _decide(12, free=[0], held=[12, 12, 12], ahead=[12]).slot is None         # one slot, and it is the head's
    assert _decide(24, free=[0, 1], held=[12, 12], ahead=[24]).slot is None          # 72 > 62 once the head is in
    assert _decide(12, free=[0, 1, 2, 3], ahead=[80]).slot is None, "the head runs alone first; this job does not fit beside it"


# ── fail-safe: every unknown is the fixed count of 2 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {"cap": None, "held": [12, 12]},                    # this job declares no cap
    {"cap": 12, "held": [12, 12], "budget": None},      # MemTotal / the cgroup tree could not be read
    {"cap": 12, "held": [12, None]},                    # a holder with neither a declared cap nor a scope
    {"cap": 12, "held": [12, 12], "ahead": [0]},        # a job that declares no cap is queued ahead
])
def test_an_unknown_falls_back_to_the_fixed_count_of_two(kw):
    kw = dict(kw)
    cap = kw.pop("cap")
    d = _decide(cap, free=[0, 1], **kw)
    assert d.slot is None and d.rule == HA.RULE_COUNT and "fixed count of 2" in d.why, d
    kw["held"] = kw["held"][-1:]                        # one holder (the unknown one, where there is one)
    if not kw.get("ahead"):
        assert _decide(cap, free=[0, 1, 2], **kw).slot == 0, "the fixed count admits the second job, on the lowest slot"
    else:
        assert _decide(cap, free=[0, 1, 2], **kw).slot is None, "one place left, and it is the uncapped job's (FIFO)"


def test_an_old_code_waiter_ahead_holds_a_capped_job_back_only_while_a_base_slot_is_free():
    assert _decide(12, free=[2, 3], held=[12, 12], ahead=[None]).slot == 3, \
        "an old-code waiter sees slot 0 and 1 only: with both held it must not block the slots above"
    d = _decide(12, free=[1, 3], held=[12, 12], ahead=[None])
    assert d.slot is None and d.rule == HA.RULE_COUNT, "the free base slot is the old-code waiter's (it was first)"


# ── charges ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_holder_is_charged_its_declared_cap_or_its_scopes_and_orphan_scopes_are_charged_too():
    anc = {500: [400, 1], 600: [1], 700: [100, 1]}
    holders = [HA.Holder(3, (100,), 24 * G),           # the gate: declared 24, its mem_cap child 700 runs a 24 G scope
               HA.Holder(2, (200,), None),             # an old-code job: no declared cap, its own scope
               HA.Holder(1, (300,), 12 * G),           # admitted a moment ago: declared, no scope yet
               HA.Holder(0, (400,), None)]             # an old-code gate: its child 500's scope
    scopes = [HA.Scope("gen3ai-capped-pytest-700.scope", 700, 24 * G),
              HA.Scope("gen3ai-capped-x-200.scope", 200, 12 * G),
              HA.Scope("gen3ai-capped-y-500.scope", 500, 16 * G),
              HA.Scope("gen3ai-capped-gpu-600.scope", 600, 40 * G),      # a BLOCKING / GPU job: no slot
              HA.Scope("gen3ai-capped-odd.scope", 0, 2 * G)]             # a name with no pid: nobody's
    held, other = HA.charges(holders, scopes, ancestors=lambda p: anc.get(p, []))
    assert held == [24 * G, 12 * G, 12 * G, 16 * G] and other == 42 * G
    held, other = HA.charges([HA.Holder(0, (900,), None)], [], ancestors=lambda p: [])
    assert held == [None] and other == 0, "a holder with no declared cap and no scope must read as UNKNOWN"
    # a nested capped job is a second cgroup: both caps count
    held, _ = HA.charges([HA.Holder(0, (100,), 48 * G)],
                         [HA.Scope("gen3ai-capped-a-100.scope", 100, 48 * G), HA.Scope("gen3ai-capped-b-700.scope", 700, 12 * G)],
                         ancestors=lambda p: anc.get(p, []))
    assert held == [60 * G]


# ── the box read ─────────────────────────────────────────────────────────────────────────────────────────────

def _scope(root, name, cap, populated=1):
    p = root / name
    p.mkdir(parents=True)
    (p / "memory.max").write_text(f"{cap}\n")
    (p / "cgroup.events").write_text(f"populated {populated}\nfrozen 0\n")


@pytest.fixture
def box(tmp_path, monkeypatch):
    mi = tmp_path / "meminfo"
    mi.write_text(f"MemTotal:       {90 * G // 1024} kB\nMemFree:  1 kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    monkeypatch.setenv(HA.MEMINFO_ENV, str(mi))
    monkeypatch.setenv(HA.CGROUP_ENV, str(cg))
    return cg


def test_the_budget_is_memtotal_less_the_reserve_and_never_more_than_the_slice(box, monkeypatch):
    b = HA.read_box()
    assert b.budget == (90 - 28) * G and b.scopes == () and b.why == "", "an empty / absent slice is known-empty"
    (box / "memory.max").write_text(f"{48 * G}\n")
    assert HA.read_box().budget == 48 * G, "the slice's own MemoryMax bounds what its scopes can use together"
    (box / "memory.max").write_text("max\n")
    assert HA.read_box().budget == 62 * G                       # not yet set: the declared 64 GiB stands in
    monkeypatch.setenv(HA.RESERVE_ENV, "40")
    assert HA.read_box().budget == 50 * G
    monkeypatch.setenv(HA.RESERVE_ENV, "95")
    b = HA.read_box()
    assert b.budget is None and "reserve" in b.why


def test_live_scopes_are_read_and_an_empty_one_is_not_counted(box):
    _scope(box, "gen3ai-capped-x35kakuna-4242.scope", 12 * G)
    _scope(box, "gen3ai-capped-gone-4243.scope", 24 * G, populated=0)
    (box / "some-other.scope").mkdir()
    b = HA.read_box()
    assert [(s.unit, s.wrapper, s.cap) for s in b.scopes] == [("gen3ai-capped-x35kakuna-4242.scope", 4242, 12 * G)]


def test_a_scope_with_no_limit_or_an_unreadable_memtotal_makes_the_budget_unknown(box, monkeypatch, tmp_path):
    _scope(box, "gen3ai-capped-nolimit-77.scope", "max")
    b = HA.read_box()
    assert b.budget is None and "memory.max" in b.why
    monkeypatch.setenv(HA.MEMINFO_ENV, str(tmp_path / "absent"))
    b = HA.read_box()
    assert b.budget is None and "MemTotal" in b.why


def test_the_real_slice_directory_is_the_one_mem_cap_creates():
    sd = HA.slice_dir()
    assert sd is None or str(sd).endswith(".service/gen3ai.slice/gen3ai-heavy.slice"), sd


# ── the switch back ──────────────────────────────────────────────────────────────────────────────────────────

def test_the_mode_file_switches_the_box_back_and_the_environment_wins(tmp_path, monkeypatch):
    d = tmp_path / "slots"
    assert HA.mode(d) == (HA.MODE_MEMORY, "default")
    assert "count" in HA.set_mode(d, "count")
    assert HA.mode(d)[0] == HA.MODE_COUNT
    monkeypatch.setenv(HA.MODE_ENV, "memory")
    assert HA.mode(d)[0] == HA.MODE_MEMORY, "the environment must win over the file"
    monkeypatch.setenv(HA.MODE_ENV, "COUNT")
    assert HA.mode(d)[0] == HA.MODE_COUNT
    monkeypatch.delenv(HA.MODE_ENV)
    HA.set_mode(d, "memory")
    assert HA.mode(d) == (HA.MODE_MEMORY, "default") and not (d / HA.MODE_FILE).exists()
    HA.set_mode(d, "memory")                                    # idempotent
    with pytest.raises(ValueError):
        HA.set_mode(d, "fast")


@pytest.mark.parametrize("text", ["", "off", "memory please", "2"])
def test_anything_that_is_not_exactly_memory_reads_as_count(tmp_path, monkeypatch, text):
    (tmp_path / HA.MODE_FILE).write_text(text)
    assert HA.mode(tmp_path)[0] == HA.MODE_COUNT
    monkeypatch.setenv(HA.MODE_ENV, text or "x")
    assert HA.mode(tmp_path)[0] == HA.MODE_COUNT


# ── the measured constants ───────────────────────────────────────────────────────────────────────────────────

def test_the_reserve_covers_the_measured_training_peak_and_the_budget_fits_the_jobs_it_was_sized_for():
    """Measured 2026-11-09 (module docs): the training tree's worst summed RSS 18.4 GiB, 7.5 GiB of everything else.
    On the box's 89.66 GiB the budget must hold four 12 GiB play units, and a 24 GiB routine gate beside three of them."""
    assert HA.RESERVE_GB >= 18.4 + 7.5
    budget = min(HA.DEFAULT_SLICE_GB, 89.66 - HA.RESERVE_GB)
    assert 4 * 12 <= budget and 24 + 3 * 12 <= budget
    assert budget <= HA.DEFAULT_SLICE_GB, "the heavy jobs' caps may never sum past the slice's own cap"


def test_the_hard_maximum_may_be_raised_or_lowered_but_never_below_the_base(monkeypatch):
    monkeypatch.setenv(HA.MAX_SLOTS_ENV, "3")
    assert HA.max_slots() == 3
    monkeypatch.setenv(HA.MAX_SLOTS_ENV, "1")
    with pytest.raises(ValueError):
        HA.max_slots()
