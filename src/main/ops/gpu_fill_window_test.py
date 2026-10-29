"""`main.ops.gpu_fill_window`: the census set, the CUDA-id selection (and its collection plugin, in a real pytest
subprocess), the per-node bound, the chain the two windows parse into (no step wraps itself; the rows land outside the
worktree), the preflights' marker, the registered stage-2 rule on synthetic reads, and `merge-rows`' refusals."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from main.ops import gpu_fill_window as G

GPU_ID = "src/x_test.py::test_on_cuda"


def test_the_census_is_w0s_census_in_its_order():
    from main.ops import w0_window as W
    want = [(run, at) for run, ats in W.census_lists().items() for at in ats]
    assert [(r, a) for r, a, _ in G.CENSUS] == want and len(G.CENSUS) == 20
    assert all(z.startswith("checkpoint_") and z.endswith("_steps.zip") for _, _, z in G.CENSUS)


def test_the_stage2_shift_is_the_cap_1_2():
    from agents.model import compile_regions as cr
    assert abs(cr.R1_BAR_K * G.STAGE2_SHIFT_MAX - G.STAGE2_MARGIN) < 1e-12
    assert cr.resolve_shift_max(G.STAGE2_SHIFT_MAX) == G.STAGE2_SHIFT_MAX


def test_select_cuda_ids_takes_cuda_skips_and_the_declared_body_skips_only():
    rows = [{"id": "a::t1", "skip": "the root conftest.py hides the GPU from the whole suite"},
            {"id": "a::t2", "skip": None}, {"id": "a::t3", "skip": "needs CUDA"},
            {"id": "a::t4", "skip": "the committed window-D reads are absent"}, {"id": "b::t5", "skip": None}]
    assert G.select_cuda_ids(rows, runtime=("b::t5",)) == ["a::t1", "a::t3", "b::t5"]
    with pytest.raises(G.FillError, match="no longer collected"):
        G.select_cuda_ids(rows, runtime=("gone::t",))


def test_the_collection_plugin_reads_skip_marks_in_a_real_pytest(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x_test.py").write_text(
        "import pytest\npytestmark = pytest.mark.slow\n"
        "@pytest.mark.skipif(True, reason='needs CUDA')\ndef test_on_cuda(): pass\n"
        "def test_on_cpu(): pass\n"
        "@pytest.mark.parametrize('f', [False, True])\ndef test_body(f): pass\n")
    ids = G.cuda_slow_ids(str(tmp_path), sys.executable, str(tmp_path), runtime=("src/x_test.py::test_body[True]",))
    assert ids == [GPU_ID, "src/x_test.py::test_body[True]"]


@pytest.mark.parametrize("dur,bound", [(None, 1800), (0.2, 600), (199.9, 600), (404.65, 1260), (835.77, 2520)])
def test_node_timeout(dur, bound):
    assert G.node_timeout(dur) == bound


def _base(tmp_path: Path) -> dict:
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    return {"schema": "arm_chain_v1", "name": "q", "job_dir": str(job), "repo": str(tmp_path / "repo"),
            "python": "/py", "status_file": str(job / "status.txt"), "models_dir": str(tmp_path / "models"),
            "backend": {"kind": "x35", "launch_script": "l.sh", "stage": "s", "lease_owner": "Training Oct",
                        "lease_token_file": str(tmp_path / "tok")},
            "records": {"stage": "s", "submodules": []}, "items": [{"kind": "arm"}]}


def _items(tmp_path: Path, ids=(GPU_ID, "src/y_test.py::test_b")):
    return G.window_items(worktree=str(tmp_path / "wt"), require_commit="abcdef1", python="/py",
                          out_dir=str(tmp_path / "out"), models_dir=str(tmp_path / "models"),
                          job_dir=str(tmp_path / "job"), ids=list(ids), durations={GPU_ID: 404.65},
                          u1000_run_dir=str(tmp_path / "u1000"))


def test_the_two_windows_parse_as_a_chain_and_no_step_wraps_itself(tmp_path):
    from main.ops.arm_chain import spec
    doc = G.chain_doc(_base(tmp_path), "x35-gpufill", _items(tmp_path))
    c = spec.parse(doc)
    assert [w.name for w in c.items] == ["canary-stage2", "slow-gpu"]
    marker = str(tmp_path / "job" / G.PREEMPT_MARKER)
    for w in c.items:
        assert w.on_fail == "hold" and w.preempt_marker == marker and w.worktree == str(tmp_path / "wt")
        assert not w.steps[0].gpu and "preflight" in w.steps[0].argv[3] and marker in w.steps[0].argv
        for st in w.steps:
            assert not any(a.endswith(("gpu_lock.sh", "mem_cap.sh")) for a in st.argv)
    canary = c.items[0].steps
    assert sum(st.gpu for st in canary) == 20 and canary[-1].argv[3] == "canary-verdict"
    for st in canary[1:-1]:
        a = list(st.argv)
        assert a[a.index("--canary-shift-max") + 1] == "0.3" and a[a.index("--batch") + 1] == "2048"
        assert a[a.index("--out") + 1].startswith(str(tmp_path / "out" / "canary_stage2"))
    assert any(str(tmp_path / "u1000") in a for a in canary[-2].argv)
    slow = c.items[1].steps[1:]
    status = f"GEN3AI_SLOW_STATUS_FILE={tmp_path / 'out' / 'slow_gpu' / 'slow_tier_status.json'}"
    assert [list(st.argv[:2]) for st in slow] == [["env", status]] * 2
    assert [st.argv[5] for st in slow] == [GPU_ID, "src/y_test.py::test_b"]     # banked (shortest) first, unknown last
    assert [st.timeout_s for st in slow] == [1260, 1800]


def test_an_out_dir_inside_the_worktree_or_models_is_refused(tmp_path):
    for bad in (tmp_path / "wt" / "o", tmp_path / "models" / "o"):
        with pytest.raises(G.FillError, match="inside"):
            G.window_items(worktree=str(tmp_path / "wt"), require_commit="abcdef1", python="/py", out_dir=str(bad),
                           models_dir=str(tmp_path / "models"), job_dir=str(tmp_path), ids=[GPU_ID], durations={})


def test_preflight_canary_writes_the_marker_on_a_missing_checkpoint_and_passes_when_present(tmp_path):
    models, u1000, marker = tmp_path / "models", tmp_path / "u1000", tmp_path / "NO_GO"
    assert G.preflight_canary(str(tmp_path / "out"), str(models), str(u1000), str(marker)) == 3
    assert "missing" in marker.read_text()
    marker.unlink()
    for _, rd, z in G.census_anchors(str(models), str(u1000)):
        Path(z).parent.mkdir(parents=True, exist_ok=True)
        Path(z).write_bytes(b"z")
        for f in ("model_config.json", "metadata.json"):
            (Path(rd) / f).write_text("{}")
    assert G.preflight_canary(str(tmp_path / "out"), str(models), str(u1000), str(marker)) == 0
    assert not marker.exists()


def test_preflight_slow_refuses_a_missing_test_file(tmp_path):
    marker = tmp_path / "NO_GO"
    (tmp_path / "wt" / "src").mkdir(parents=True)
    assert G.preflight_slow(str(tmp_path / "out"), str(tmp_path / "wt"), str(marker), [GPU_ID]) == 3
    (tmp_path / "wt" / "src" / "x_test.py").write_text("")
    marker.unlink()
    assert G.preflight_slow(str(tmp_path / "out"), str(tmp_path / "wt"), str(marker), [GPU_ID]) == 0


# ------------------------------------------------------------------------------------------------ the stage-2 rule
def _reads(refused=(), boundary=(), repeats=3):
    return {"status": "read", "reads": [{"repeat": r, "slice": s, "refused": (r, s) in refused,
                                         "boundary": (r, s) in boundary, "margin": 1.2, "max_ratio": 2.0, "text": ""}
                                        for r in range(repeats) for s in (0, 2, 1)]}


def _census(**over):
    return {f"{r}_{a}": over.get(f"{r}_{a}", _reads()) for r, a, _ in G.CENSUS}


def test_stage2_rule_adopt_reject_inconclusive():
    lbl = "rb_x26_s2601_100M"
    assert G.stage2_verdict(_census(), 3, 20)["verdict"] == "ADOPT"
    v = G.stage2_verdict(_census(**{lbl: _reads(refused={(0, 0), (2, 0)})}), 3, 20)
    assert v["verdict"] == "REJECT" and v["rejecting"] == [lbl]
    for one in (_reads(refused={(0, 0)}), _reads(refused={(1, 1), (2, 1)}), _reads(boundary={(0, 2)}),
                {"status": "missing", "why": "no summary"}, {"status": "invalid", "why": "margin"}):
        assert G.stage2_verdict(_census(**{lbl: one}), 3, 20)["verdict"] == "INCONCLUSIVE"
    short = _census()
    short.pop(lbl)
    assert G.stage2_verdict(short, 3, 20)["verdict"] == "INCONCLUSIVE"


def _write_ckpt(d: Path, margin=1.2, refused=False, ratio=2.0, repeats=1, cpu=False):
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"complete": True, "repeats": repeats, "validity": {"valid": True}}))
    can = {"refused": refused, "text": "R1 refused" if refused else "ok", "watch": {"clip_margin": {"margin": margin}},
           "all_params": {"p": {"ratio": ratio, "compiled": 1, "eager": 1}}} if not cpu else None
    for r in range(repeats):
        (d / f"repeat_{r:02d}.json").write_text(json.dumps({"slices": {s: {"canary": can} for s in ("0", "2", "1")}}))


def test_canary_verdict_reads_the_replay_files_and_checks_the_margin(tmp_path):
    root = tmp_path / "canary_stage2"
    for r, a, _ in G.CENSUS:
        _write_ckpt(root / f"{r}_{a}")
    assert G.canary_verdict(str(tmp_path), repeats=1)["verdict"] == "ADOPT"
    assert json.loads((root / "verdict.json").read_text())["verdict"] == "ADOPT"
    import shutil
    lbl = "rb_x26c_s2601_480M"
    for kw, want in ((dict(margin=1.4), "INCONCLUSIVE"), (dict(cpu=True), "INCONCLUSIVE"), (dict(ratio=12.0 * 1.0005), "INCONCLUSIVE"),
                     (dict(refused=True), "REJECT")):
        shutil.rmtree(root / lbl)
        _write_ckpt(root / lbl, **kw)
        v = G.canary_verdict(str(tmp_path), repeats=1)
        assert v["verdict"] == want, (kw, v)


# ------------------------------------------------------------------------------------------------ merge-rows
def _status(p: Path, tests: dict) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": "gen3_slow_tier_status_v1", "tests": tests}))
    return p


def test_merge_rows_folds_by_the_writers_rule_and_never_into_main(tmp_path):
    row = {"status": "pass", "commit": "c1", "at": "2026-10-25T00:00:00Z", "duration_s": 1.0, "contention": 1.0,
           "detail": ""}
    wt = _status(tmp_path / "wt" / "designs" / "ops" / "slow_tier_status.json", {"a::t": row, "b::t": row})
    rows = _status(tmp_path / "out" / "rows.json", {
        "a::t": {**row, "status": "fail", "commit": "c2", "detail": "boom"},
        "b::t": {**row, "status": "skip", "commit": "c2"}, "c::t": {**row, "commit": "c2"}})
    assert G.merge_rows(rows, wt, main_root=str(tmp_path / "main")) == 3
    got = json.loads(wt.read_text())["tests"]
    assert got["a::t"]["status"] == "fail" and got["c::t"]["commit"] == "c2"
    assert got["b::t"]["status"] == "pass" and got["b::t"]["last_skipped_commit"] == "c2"   # a skip never replaces
    main = _status(tmp_path / "main" / "designs" / "ops" / "slow_tier_status.json", {})
    with pytest.raises(G.FillError, match="MAIN"):
        G.merge_rows(rows, main, main_root=str(tmp_path / "main"))
    with pytest.raises(G.FillError, match="not an existing"):
        G.merge_rows(rows, tmp_path / "nope.json", main_root=str(tmp_path / "main"))
    assert os.path.isfile(main)
