"""The M5 gate harness's REGISTRY and VERDICT logic (routine, no battles): every lane the program doc
marks BUILT has a BUILT row; every declared test exists; a NOT BUILT lane never reads as a pass; a
skip-only or empty row is not a pass; a timeout is INCONCLUSIVE; a GPU part not run reads NOT RUN;
the M5 verdict is NOT MET while any component is missing."""
import json

from main.rust_core_m5 import __main__ as M
from main.rust_core_m5 import depth3 as D3
from main.rust_core_m5 import gates as G
from main.rust_core_m5 import lanes as L

_DOC = """### M5 — test
| lane | owns | gate | depends | size |
|---|---|---|---|---|
| **0 — core (FIRST)** | x | y | — | H |
| A — FFI front end | x | y | 0 | M |
| D — episode (**BUILT 2026-09-29**) | x | y | 0 | M |
| G — training | x | y | A | H |
| **K — learner** | x | y | — | M |

**Lane A BUILT (2026-09-29).** text
**Lane K1 BUILT (2026-09-28):** text
**Lane 0 (BUILDING from 2026-09-28).** text
### M6 — next
"""


def test_the_registry_agrees_with_the_program_doc_and_the_tree():
    assert L.registry_problems(L.program_doc_text()) == []


def test_doc_parsing_reads_table_markers_paragraphs_and_sub_lanes():
    assert L.doc_lanes(_DOC) == {"0": False, "A": True, "D": True, "G": False, "K": True}


def test_a_built_lane_without_a_row_fails_the_registry(monkeypatch):
    doc = L.program_doc_text()
    rows = tuple(r for r in L.LANES if r.lane != "D")
    monkeypatch.setattr(L, "LANES", rows)
    probs = L.registry_problems(doc)
    assert any("lane D is in the program doc's lane table and has no row" in p for p in probs), probs


def test_a_built_lane_whose_row_says_not_built_fails(monkeypatch):
    import dataclasses

    rows = tuple(dataclasses.replace(r, built=False, tests=(), gpu_tests=(), pending="later") if r.lane == "A" else r
                 for r in L.LANES)
    monkeypatch.setattr(L, "LANES", rows)
    probs = L.registry_problems(L.program_doc_text())
    assert any("lane A is BUILT per the program doc and its row says NOT BUILT" in p for p in probs), probs


def test_a_renamed_test_node_fails_the_registry(monkeypatch):
    import dataclasses

    rows = tuple(dataclasses.replace(r, tests=(*r.tests, "src/utils/rust_env/ffi_test.py::test_no_such_gate"))
                 if r.lane == "A" else r for r in L.LANES)
    monkeypatch.setattr(L, "LANES", rows)
    assert any("test_no_such_gate names no test function" in p for p in L.registry_problems(L.program_doc_text()))


def _row(lane):
    return L.by_lane()[lane]


def _with_h_not_built(monkeypatch):
    """Every M5 lane is BUILT since Lane H (2026-09-30): the NOT BUILT paths are exercised on a registry
    whose H row is put back to NOT BUILT (read at call time: ``gates`` and ``__main__`` use ``L.LANES``)."""
    import dataclasses

    rows = tuple(dataclasses.replace(r, built=False, tests=(), gpu_tests=(), pending="the gate")
                 if r.lane == "H" else r for r in L.LANES)
    monkeypatch.setattr(L, "LANES", rows)
    return rows


def test_a_not_built_lane_never_reads_as_a_pass(monkeypatch):
    _with_h_not_built(monkeypatch)
    v = G.fold(_row("H"), {"src/agents/training/anything_test.py::test_x": "pass"}, source="run:commit", gpu_ran=True)
    assert v.verdict == G.NOT_BUILT and not v.counts


def test_the_fold_pass_fail_skip_timeout_and_empty():
    f = "src/utils/rust_env/ffi_test.py"
    a = _row("A")
    assert G.fold(a, {f"{f}::t1": "pass", f"{f}::t2": "skip"}, source="", gpu_ran=False).verdict == G.PASS
    assert G.fold(a, {f"{f}::t1": "pass", f"{f}::t2": "fail"}, source="", gpu_ran=False).verdict == G.FAIL
    assert G.fold(a, {f"{f}::t1": "skip"}, source="", gpu_ran=False).verdict == G.INCONCLUSIVE
    assert G.fold(a, {}, source="", gpu_ran=False).verdict == G.INCONCLUSIVE
    assert G.fold(a, {f"{f}::t1": "pass", f"{f}::t2": "inconclusive"}, source="", gpu_ran=False).verdict == G.INCONCLUSIVE
    # another lane's node never counts for this row
    assert G.fold(a, {"src/utils/rust_env/proc_test.py::t": "pass"}, source="", gpu_ran=False).verdict == G.INCONCLUSIVE


def test_a_gpu_part_not_run_reads_not_run_and_a_gpu_failure_fails_the_row():
    e = _row("E")
    cpu = "src/agents/training/rust_env_opponents_test.py::test_x"
    gpu = e.gpu_tests[0] + "[greedy]"
    v = G.fold(e, {cpu: "pass", gpu: "skip"}, source="", gpu_ran=False)
    assert (v.verdict, v.gpu) == (G.PASS, G.NOT_RUN)
    v = G.fold(e, {cpu: "pass", gpu: "fail"}, source="", gpu_ran=True)
    assert (v.verdict, v.gpu) == (G.FAIL, G.FAIL)


def test_a_shared_node_counts_for_every_row_that_declares_it():
    node = "src/utils/rust_env/core_cargo_test.py::test_the_rust_env_core_suite_passes"
    for lane in ("0", "D", "E", "I"):
        assert G._belongs(node, _row(lane).tests), lane
    assert not G._belongs(node, _row("A").tests)


def test_recorded_reads_the_slow_tier_bank_and_marks_stale(monkeypatch):
    _with_h_not_built(monkeypatch)
    st = {"tests": {"src/agents/training/rust_env_episode_parity_test.py::test_milestone_episodes_equal_gen3env[pool]":
                    {"status": "pass", "commit": "aaaaaaaa11", "at": "2026-09-29T00:00:00Z"},
                    "src/utils/rust_env/bots_gate_test.py::test_milestone_every_decision_is_equal_at_scale":
                    {"status": "fail", "commit": "bbbbbbbb22", "at": "2026-09-29T00:00:00Z"}}}
    by = {v.lane: v for v in G.recorded(st, head="aaaaaaaa11")}
    assert by["D"].verdict == G.PASS and "STALE" not in by["D"].detail
    assert by["F"].verdict == G.FAIL and "STALE" in by["F"].detail
    assert by["A"].verdict == G.UNRECORDED
    assert by["H"].verdict == G.NOT_BUILT


def test_the_m5_verdict_is_not_met_while_a_component_is_missing(tmp_path, monkeypatch):
    _with_h_not_built(monkeypatch)
    res = M.compose(tmp_path)
    assert res["m5_gate"] == "NOT MET"
    assert any("slice N" in k for k in res["missing"])
    rows = [{"lane": r.lane, "verdict": G.PASS if r.built else G.NOT_BUILT, "gpu": ""} for r in L.LANES]
    (tmp_path / "gates_milestone.json").write_text(json.dumps({"tier": "milestone", "rows": rows}))
    for comp in ("slice_n", "depth3"):
        (tmp_path / f"{comp}_milestone.json").write_text(json.dumps({"ok": True, "commit": "c" * 40, "refused_both": 0}))
    (tmp_path / "throughput_n48.json").write_text(json.dumps({"regime": {"n_envs": 48}}))
    res = M.compose(tmp_path)
    # everything measured and green — still NOT MET: lane H is NOT BUILT
    assert res["m5_gate"] == "NOT MET"
    assert sorted(k.split(" — ")[0] for k in res["missing"]) == ["lane H"], res["missing"]


def test_a_commit_tier_slice_is_not_the_milestone_gate(tmp_path):
    (tmp_path / "slice_n_commit.json").write_text(json.dumps({"ok": True, "commit": "c" * 40}))
    items = {k: v for k, v, _ in M.compose(tmp_path)["items"]}
    assert items["slice N at the env level"] == G.INCONCLUSIVE


def test_the_depth3_slice_gates_searchs_default_depth():
    assert D3.default_depth() == 3


def test_gates_from_status_runs_nothing_and_writes_its_table(tmp_path, capsys, monkeypatch):
    _with_h_not_built(monkeypatch)
    rc = M.main(["--results", str(tmp_path), "gates", "--from-status"])
    out = capsys.readouterr().out
    assert "| H |" in out and "NOT BUILT" in out
    doc = json.loads((tmp_path / "gates_recorded.json").read_text())
    assert {r["lane"] for r in doc["rows"]} == {r.lane for r in L.LANES}
    assert rc in (0, 1)


def test_a_skip_never_replaces_a_banked_verdict(tmp_path):
    """F-LJ-5: a GPU test skips without GEN3AI_TEST_ALLOW_GPU; banking that skip would erase its PASS."""
    target, scratch = tmp_path / "status.json", tmp_path / "scratch.json"
    row = lambda s, c: {"status": s, "commit": c, "at": "t", "duration_s": 1.0, "contention": 1.0, "detail": ""}  # noqa: E731
    target.write_text(json.dumps({"tests": {"gpu": row("pass", "old"), "cpu": row("pass", "old")}}))
    scratch.write_text(json.dumps({"tests": {"gpu": row("skip", "new"), "cpu": row("fail", "new"),
                                             "fresh": row("skip", "new")}}))
    assert G.bank_without_skip_clobber(scratch, target) == ["cpu", "fresh"]
    got = json.loads(target.read_text())["tests"]
    assert got["gpu"]["status"] == "pass" and got["cpu"]["status"] == "fail" and got["fresh"]["status"] == "skip"


def test_a_subset_run_overlays_its_lanes_without_erasing_the_table(tmp_path):
    rows = [{"lane": r.lane, "verdict": G.PASS if r.built else G.NOT_BUILT, "gpu": ""} for r in L.LANES]
    rows = [dict(r, verdict=G.FAIL) if r["lane"] == "E" else r for r in rows]
    (tmp_path / "gates_milestone.json").write_text(json.dumps({"tier": "milestone", "rows": rows}))
    sub = [{"lane": "E", "verdict": G.PASS, "gpu": G.PASS}, {"lane": "T2", "verdict": G.PASS, "gpu": G.PASS}]
    (tmp_path / "gates_milestone_lanes_E_T2.json").write_text(json.dumps({"tier": "milestone", "rows": sub}))
    items = {k.split(" — ")[0]: (v, e) for k, v, e in M.compose(tmp_path)["items"]}
    assert items["lane E"][0] == G.PASS and "GPU part PASS" in items["lane E"][1] and "lanes_E_T2" in items["lane E"][1]
    assert items["lane A"] == (G.PASS, "milestone")
