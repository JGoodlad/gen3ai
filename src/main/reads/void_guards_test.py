"""The VOID gate's OPT-IN run-dir guards (``main.reads.rules.void_gate``; N1-P PREREG §9 guards 3 and 4, AM-N1P-1
FINDING 11: the ``d1`` rule's gate read the K9(b) record only, and the pin and the pool seed were operator checks). A spec
that registers ``params.void.pin`` / ``params.void.pool_seed`` has every gated run's own ``metadata.json`` and
``snapshots/pool_seed.json`` checked before ``decide``; a spec that names neither is gated exactly as before. Each guard
fails on revert. Run dirs are synthetic; one test reads the committed N1-P s3501 run when this box has it."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from main.reads import api, layout as LY
from main.reads import rules as RU
from main.reads import runner as RN
from main.reads.pipeline_test import CODE, FakeSlot, _refpop, _spec, _void
from main.reads.spec import SpecError, load, parse

PIN = "5f8f6cfa37de8e283ba553c0d415680cf51bff18"
FORK = 280000170


def _run_dir(root: Path, name: str, *, git_hash: str = PIN, spans: Optional[List[str]] = None,
             pool: Optional[Dict[str, Any]] = None, metadata: bool = True) -> Path:
    """A run dir with the artefacts the guards read: ``metadata.json`` (``git_hash`` + one ``pin_history`` entry per
    span, default the registered commit alone) and ``snapshots/pool_seed.json`` (the T54 record)."""
    rd = root / name
    (rd / "snapshots").mkdir(parents=True, exist_ok=True)
    if metadata:
        (rd / "metadata.json").write_text(json.dumps({
            "git_hash": git_hash,
            "pin_history": [{"git_hash": h, "pin_source": "pin_commit"} for h in (spans if spans is not None else [PIN])]}))
    if pool is not None:
        (rd / "snapshots" / "pool_seed.json").write_text(json.dumps(pool))
    return rd


def _pool(**over: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"schema": 2, "fork_step": FORK, "snapshots": ["snapshot_000164000130.zip",
                                                                        "snapshot_000268000171.zip"],
                         "dropped_steps": [286000124, 292000193]}
    d.update(over)
    return d


# ------------------------------------------------------------------------------------------------ the pin
def test_the_pin_guard_passes_a_run_that_recorded_only_the_registered_commit(tmp_path):
    rd = _run_dir(tmp_path, "r")
    g = RU.void_pin_guard(rd, PIN[:8])                          # a prefix of at least 7 digits names the commit
    assert g["pass"] and g["why"] == "" and g["recorded"] == [PIN] and g["want"] == PIN[:8]
    assert RU.void_pin_guard(rd, PIN)["pass"]


@pytest.mark.parametrize("kw,needle", [
    ({"git_hash": "a" * 40}, "are not the registered"),                       # the sidecar's git_hash is another commit
    ({"spans": [PIN, "b" * 40]}, "are not the registered"),                   # a second pin_history span (a re-pin mid-run)
    ({"metadata": False}, "cannot show its pin"),
])
def test_the_pin_guard_fails_a_run_whose_recorded_code_is_not_the_registered_commit(tmp_path, kw, needle):
    g = RU.void_pin_guard(_run_dir(tmp_path, "r", **kw), PIN)
    assert not g["pass"] and needle in g["why"], g


def test_the_pin_guard_needs_a_pin_history_span_and_a_readable_sidecar(tmp_path):
    rd = _run_dir(tmp_path, "r")
    (rd / "metadata.json").write_text(json.dumps({"git_hash": PIN, "pin_history": []}))
    g = RU.void_pin_guard(rd, PIN)
    assert not g["pass"] and "no pin_history span" in g["why"]
    (rd / "metadata.json").write_text("{not json")
    g = RU.void_pin_guard(rd, PIN)
    assert not g["pass"] and "unreadable" in g["why"]


# ------------------------------------------------------------------------------------------------ the pool seed
def test_the_pool_seed_guard_passes_the_t54_record_of_the_registered_fork(tmp_path):
    g = RU.void_pool_seed_guard(_run_dir(tmp_path, "r", pool=_pool()), FORK)
    assert g["pass"] and g["why"] == "" and (g["got"], g["snapshots"], g["dropped"]) == (FORK, 2, 2)


@pytest.mark.parametrize("over,needle", [
    ({"fork_step": FORK + 1}, "!= the registered"),
    ({"fork_step": None}, "fork_step is null"),
    ({"schema": 1}, "schema 1 != 2"),
    ({"snapshots": ["snapshot_000300000001.zip"]}, "later than fork_step"),          # a snapshot past the fork leaked in
    ({"snapshots": ["not_a_snapshot.zip"]}, "later than fork_step"),                  # (or unnamed)
    ({"dropped_steps": [FORK - 5]}, "not later than fork_step"),                      # the filter dropped an earlier one
])
def test_the_pool_seed_guard_fails_each_way_the_t54_record_is_wrong(tmp_path, over, needle):
    g = RU.void_pool_seed_guard(_run_dir(tmp_path, "r", pool=_pool(**over)), FORK)
    assert not g["pass"] and needle in g["why"], g


def test_the_pool_seed_guard_fails_a_run_with_no_record(tmp_path):
    g = RU.void_pool_seed_guard(_run_dir(tmp_path, "r"), FORK)
    assert not g["pass"] and "no readable snapshots/pool_seed.json" in g["why"]


# ------------------------------------------------------------------------------------------------ the gate and the spec
def _gated(tmp_path, *, pin: Optional[str] = None, pool_seed: Optional[Dict[str, Any]] = None, bad: Optional[str] = None):
    """SPRT's spec (``pipeline_test``) whose three ON seeds' run dirs carry the guards' artefacts; ``bad`` names the
    seed (3502) whose artefact is wrong: ``pin`` | ``pool``."""
    refpop, rdoc = _refpop(tmp_path)
    void = _void(tmp_path)
    for lab, run in void["runs"].items():
        s3502 = lab.endswith("3502")
        _run_dir(Path(run).parent, Path(run).name, git_hash="a" * 40 if s3502 and bad == "pin" else PIN,
                 pool=_pool(fork_step=FORK + 7) if s3502 and bad == "pool" else _pool())
    if pin is not None:
        void["pin"] = pin
    if pool_seed is not None:
        void["pool_seed"] = pool_seed
    return _spec(tmp_path, refpop, rdoc["content_sha256"], void=void), rdoc


def test_a_spec_that_registers_neither_guard_is_gated_exactly_as_before(tmp_path):
    p, _ = _gated(tmp_path, bad="pin")                 # the pin IS wrong on 3502, and nobody asked
    gate = RU.void_gate(load(p))
    assert all(set(g) == {"run", "k9b", "k9b_file", "batch2", "pass"} for g in gate.values()), "the row keeps its shape"
    assert RU.void_failures(gate) == {}


def test_the_gate_carries_each_registered_guard_and_names_the_seed_that_fails(tmp_path):
    p, _ = _gated(tmp_path, pin=PIN[:12], pool_seed={"fork_step": FORK}, bad="pin")
    gate = RU.void_gate(load(p))
    assert {lab: g["pass"] for lab, g in gate.items()} == {"sprt_s3501": True, "sprt_s3502": False, "sprt_s3503": True}
    assert gate["sprt_s3501"]["pin"]["pass"] and gate["sprt_s3501"]["pool_seed"]["pass"]
    fail = RU.void_failures(gate)
    assert set(fail) == {"sprt_s3502"} and "pin: recorded commit(s)" in fail["sprt_s3502"]
    assert "pool seed" not in fail["sprt_s3502"], "only the failing guard is named"
    (tmp_path / "again").mkdir()
    p, _ = _gated(tmp_path / "again", pin=PIN[:12], pool_seed={"fork_step": FORK}, bad="pool")
    fail = RU.void_failures(RU.void_gate(load(p)))
    assert set(fail) == {"sprt_s3502"} and "pool seed: fork_step" in fail["sprt_s3502"] and "pin:" not in fail["sprt_s3502"]


def test_each_guard_gates_alone_and_k9b_still_decides(tmp_path):
    p, _ = _gated(tmp_path, pin=PIN, bad="pool")        # the pool seed is wrong on 3502, but only the pin is registered
    gate = RU.void_gate(load(p))
    assert RU.void_failures(gate) == {} and "pool_seed" not in gate["sprt_s3502"] and gate["sprt_s3502"]["pin"]["pass"]
    # a failing K9(b) still reads first and keeps its text; the guards' text follows it
    doc = json.loads(p.read_text())
    kf = Path(doc["decision"]["params"]["void"]["k9b_dir"]) / "rb_x35_sprt_s3502.json"
    kf.write_text(json.dumps({"verdict": "FAIL"}))
    assert RU.void_failures(RU.void_gate(load(p))) == {"sprt_s3502": "K9(b) FAIL, SPRT V-S1/V-S2 PASS"}, \
        "the text a K9(b) failure always had"


@pytest.mark.parametrize("key,val,needle", [
    ("pin", "xyz", "void.pin: a commit sha"),
    ("pin", "abc12", "void.pin: a commit sha"),                    # shorter than 7 digits
    ("pin", ["5f8f6cfa"], "void.pin: a commit sha"),
    ("pool_seed", {"fork_step": 0}, "void.pool_seed"),
    ("pool_seed", {"fork_step": "280000170"}, "void.pool_seed"),
    ("pool_seed", {"fork_step": True}, "void.pool_seed"),
    ("pool_seed", {"fork_step": FORK, "extra": 1}, "void.pool_seed"),
    ("pool_seed", FORK, "void.pool_seed"),
    ("banner", "x", r"void: \{k9b_dir"),                           # a key the gate does not know stays refused
])
def test_a_malformed_guard_registration_is_refused_at_load(tmp_path, key, val, needle):
    p, _ = _gated(tmp_path)
    doc = json.loads(p.read_text())
    doc["decision"]["params"]["void"][key] = val
    with pytest.raises(SpecError, match=needle):
        parse(doc)


def test_the_registered_guards_refuse_the_record_before_decide_and_the_gate_file_says_which(tmp_path, no_git):
    p, rdoc = _gated(tmp_path, pin=PIN[:12], pool_seed={"fork_step": FORK}, bad="pin")
    state = tmp_path / "state"
    RN.plan(p, state, code=CODE, step_of=lambda _p: 300_000_000)
    assert RN.run(state, "s", time.time() + 7200, run_s=FakeSlot(state, rdoc), sleep=lambda s: None) == 0
    with pytest.raises(api.RuleInputError, match=r"VOID gate fails.*sprt_s3502.*pin: recorded commit"):
        api.record(state)
    assert not LY.record_file(state).exists() and not LY.decision_file(state).exists()
    gate = json.loads(LY.void_file(state).read_text())
    assert [lab for lab, g in gate.items() if not g["pass"]] == ["sprt_s3502"]
    assert gate["sprt_s3502"]["pin"]["pass"] is False and gate["sprt_s3502"]["pool_seed"]["pass"] is True


def test_a_passing_gate_with_both_guards_is_carried_in_the_record(tmp_path, no_git):
    p, rdoc = _gated(tmp_path, pin=PIN[:12], pool_seed={"fork_step": FORK})
    state = tmp_path / "state"
    RN.plan(p, state, code=CODE, step_of=lambda _p: 300_000_000)
    assert RN.run(state, "s", time.time() + 7200, run_s=FakeSlot(state, rdoc), sleep=lambda s: None) == 0
    rec = api.record(state)
    assert all(g["pass"] and g["pin"]["pass"] and g["pool_seed"]["pass"] for g in rec["void"].values())
    assert api.load_record(LY.record_file(state))["void"] == rec["void"]


def test_a_record_whose_gate_holds_a_failed_guard_is_refused_by_the_record_check():
    from main.reads import record as REC
    from main.reads.record_test import VOID_PASS, _rec

    r = _rec(rule="x35_r1", verdict="NULL", decision={"arm": "SPRT", "look": 1, "verdict": "NULL"}, arm="SPRT", look=1,
             void=copy.deepcopy(VOID_PASS))
    assert REC.check(r) == []
    r["void"]["sprt_s3501"] = {"run": "r", "k9b": "PASS", "batch2": None, "pass": False,
                               "pin": {"pass": False, "why": "recorded commit(s) ['aaaaaaaaaaaa'] are not the registered x"}}
    assert any("VOID gate fails" in p and "pin:" in p for p in REC.check(r)), REC.check(r)


def test_the_committed_n1p_s3501_run_passes_the_guards_n1ps_prereg_names():
    """UNVERIFIED off the owner's box: the real run dir is read when it exists (metadata + T54 record)."""
    from utils.paths import main_models_dir

    rd = main_models_dir() / "rb_x35_n1p_s3501"
    if not ((rd / "metadata.json").is_file() and (rd / "snapshots" / "pool_seed.json").is_file()):
        pytest.skip("rb_x35_n1p_s3501's metadata.json / snapshots/pool_seed.json are not on this box")
    assert RU.void_pin_guard(rd, "5f8f6cfa")["pass"]
    assert RU.void_pool_seed_guard(rd, 280000170)["pass"]
    assert not RU.void_pin_guard(rd, "5f8f6cfb")["pass"] and not RU.void_pool_seed_guard(rd, 280000171)["pass"]
