"""``main.cycle_monitor`` through a REAL ledger (no engine): ``tick`` plays every check's row as one family request,
decides each check ONCE when every row up to it is complete, tops up a SUSPECTED check and decides CONFIRMED, resumes
without replaying, refuses a changed node file, ``verify`` re-derives each ``cycle_flag`` decision, ``status`` /
``main.plateau status`` print the Tier-2 columns and the signature, and the ladder read orients the in-run ladder's
mirrored edges. The games are synthetic rows written under claims as ``main.h2h`` writes them (an Elo model per
node). The engine path is ``main/h2h/cycle_monitor_integration_test.py`` (slow: a real tick on the h2h engine), and
was run by hand on X34 (``research_state/measurements/cycle_monitor_retro_2026-10-19/``). Also: the size bar's
fields on every decided check, the panel's cells through ``tick --panel``, and the registered panel's shape."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from agents.training import cycle_monitor as CM
from agents.training import eval_ledger as L
from agents.training.eval_ledger import audit as A
from agents.training import plateau_t2 as T2
from agents.training.eval_ledger import testkit as K
from main import cycle_monitor as M

D = 10_000_000
FAST = CM.MonitorRule(n_boot=200, posterior_draws=10, pairs_per_edge=100, batch_pairs=50, topup_pairs=100)
CLOCK = K.Clock()


class Ref:
    def __init__(self, path: str):
        p = Path(path)
        self.zip_path, self.num_timesteps = path, int(p.name.split("_")[1])
        self.sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        self.id = f"{p.parent.parent.name}@{self.num_timesteps}"

    def block(self) -> Dict[str, Any]:
        return {"id": self.id, "sha256": self.sha256, "path": self.zip_path, "run": "r", "step": self.num_timesteps,
                "rung": "explicit_zip"}


def make_run(tmp: Path, n: int, name: str = "rb_fake") -> Path:
    d = tmp / name / "checkpoints"
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        (d / f"checkpoint_{k * D + 11}_steps.zip").write_bytes(f"{name}-{k}".encode())
    return d.parent


class FakePlay:
    """Writes each cell's missing batches as synthetic rows from ``elo`` (node index = step // D), with a planted
    hole on (newer, older) index pairs; records the cells it played."""

    def __init__(self, elo, planted=None, seed=0):
        self.elo, self.planted, self.rng = elo, planted or {}, np.random.default_rng(seed)
        self.played: list = []

    def __call__(self, job, root, writer, compute, run, emit, inflight, batch_pairs):
        reg = K.regime()
        if job.family is not None:
            writer.register_family(job.family, decision_kind=CM.DECISION_KIND, rule=CM.REGISTERED.rule_string(),
                                   protocol=K.PROTO)
        req = writer.open_request(job.rid, kind=job.kind, purpose=job.purpose, family=job.family,
                                  protocol=None if job.family else K.PROTO, spec={"consumer": job.spec})
        have = {(r["player"]["sha256"], r["opponent"]["sha256"], int(r["request"]["batch"]))
                for r in (L.read(M.MATRIX_ROWS, root=root, family=job.family).rows if job.family else
                          L.read(M.PANEL_ROWS, root=root, request_id=job.rid).rows)
                if r["request"]["id"] == job.rid}
        for p, o, pairs in job.cells:
            i, j = p.num_timesteps // D, o.num_timesteps // D
            dl = self.elo[i] - self.elo[j] + self.planted.get((i, j), 0.0)
            prob = 1 / (1 + 10 ** (-dl / 400))
            for b in range(pairs // batch_pairs):
                if (p.sha256, o.sha256, b) in have:
                    continue
                wins = (self.rng.random((batch_pairs, 2)) < prob).sum(1)
                pc = [0, 0, 0, 0, 0]
                for v in wins:
                    pc[2 * int(v)] += 1
                writer.clock.advance(1)
                c = writer.claim(job.rid, batch=b, player=p.sha256, opponent=o.sha256, regime_id=reg["regime_id"],
                                 expected_wall_s=60.0)
                key = M.schedule_key(job.key_kind, run, p.sha256, o.sha256, job.key_salt)
                row = K.make_row(writer.next_row_id(), request=req, batch=b, pc=pc, p=p.sha256, o=o.sha256,
                                 purpose=job.purpose, ts=writer.clock.iso(), schedule_key=key,
                                 cycle_seed=int(hashlib.sha256(f"{key}:{b}".encode()).hexdigest()[:12], 16))
                row["player"].update(p.block(), kind="checkpoint")
                row["opponent"].update(o.block(), kind="checkpoint")
                writer.append_row(row, c)
                self.played.append((job.kind, i, j, b))


CLIMB = [400 * math.log(1 + k) for k in range(12)]


def run_tick(run_dir, root, play, **kw):
    return M.tick(run_dir, [], str(root), None, rule=FAST, play=play, resolve=Ref, emit=lambda _m: None, **kw)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "REGISTERED", FAST)          # the status reader and verify use the registered rule
    monkeypatch.setattr(L.LedgerWriter, "__init__", _clocked(L.LedgerWriter.__init__))
    return make_run(tmp_path, 7), tmp_path / "ledger"


def _clocked(init):
    def f(self, root, producer="h2h", clock=None, **kw):
        init(self, root, producer=producer, clock=clock or CLOCK, **kw)
    return f


def decisions(root):
    return L.read_decisions(root=root, kind=CM.DECISION_KIND)


def test_tick_plays_rows_decides_once_and_resumes(setup):
    run_dir, root = setup
    fake = FakePlay(CLIMB)
    run_tick(run_dir, root, fake)
    ds = decisions(root)
    by = {d["request_id"]: d for d in ds}
    rids = [CM.row_request_id("rb_fake", g * D) for g in range(3, 8)]      # the 2D check has one edge: N/A
    assert sorted(by) == sorted(rids)
    # a per-check false-alarm rate of ~6–10 % is the design's (§2.5): the driver test asserts each check is DECIDED
    # (a SUSPECTED one after its top-up), never which way a transitive check's noise falls
    assert all(d["verdict"] in CM.VERDICTS for d in ds), [(d["request_id"], d["verdict"]) for d in ds]
    assert all(d["family"] == M.family_id("rb_fake") for d in ds)
    # every row is a 100-pair edge of its check's request; the 70M check's row holds lags 1..6
    assert sum(1 for k, i, _j, _b in fake.played if k == CM.REQUEST_KIND_ROW and i == 7) == 6 * 2
    n = len(fake.played)
    run_tick(run_dir, root, fake)                                            # resume: nothing to play or decide
    assert len(fake.played) == n and len(decisions(root)) == len(ds)
    for d in ds:                                                            # verify re-derives every verdict
        rep = A.verify(Path(root), d["decision_id"])
        assert rep.ok and rep.verdict_rederived == d["verdict"], (rep.problems, rep.note)


def test_a_planted_hole_tops_up_and_confirms(setup):
    run_dir, root = setup
    fake = FakePlay(CLIMB, planted={(7, 3): -250.0})
    run_tick(run_dir, root, fake)
    by = {d["request_id"]: d["verdict"] for d in decisions(root)}
    assert by[CM.row_request_id("rb_fake", 7 * D)] == CM.CONFIRMED
    assert any(k == CM.REQUEST_KIND_TOPUP and (i, j) == (7, 3) for k, i, j, _b in fake.played)
    d7 = next(d for d in decisions(root) if d["request_id"] == CM.row_request_id("rb_fake", 7 * D))
    assert A.verify(Path(root), d7["decision_id"]).verdict_rederived == CM.CONFIRMED


def test_a_check_waits_for_every_earlier_row(setup):
    run_dir, root = setup
    fake = FakePlay(CLIMB)
    run_tick(run_dir, root, fake, grids=[7 * D])
    assert decisions(root) == []                               # rows of 30M..60M are missing: nothing is decided
    st = {s.row.grid: s.state for s in M.check_states("rb_fake", M.node_series(run_dir), Path(root), rule=FAST)}
    assert st[7 * D] == M.WAITING               # its own row is complete; it waits for the earlier rows
    assert st[3 * D] == M.NOT_STARTED


def test_a_changed_node_file_is_refused(setup):
    run_dir, root = setup
    run_tick(run_dir, root, FakePlay(CLIMB), grids=[3 * D])
    (run_dir / "checkpoints" / f"checkpoint_{3 * D + 11}_steps.zip").write_bytes(b"replaced")
    with pytest.raises(M.MonitorDriverError, match="FROZEN"):
        M.tick(run_dir, [], str(root), None, rule=FAST, play=_reopen_only, resolve=Ref, emit=lambda _m: None,
               grids=[3 * D])


def _reopen_only(job, root, writer, *a):
    """The real driver's re-open: a request whose frozen spec changed is a refusal."""
    try:
        writer.open_request(job.rid, kind=job.kind, purpose=job.purpose, family=job.family, protocol=None,
                            spec={"consumer": job.spec})
    except L.RequestSpecError as e:
        raise M.MonitorDriverError(f"{job.rid}: its FROZEN inputs changed — {e}") from None


def test_status_and_the_plateau_reader_print_tier2(setup, capsys):
    run_dir, root = setup
    run_tick(run_dir, root, FakePlay(CLIMB))
    rep = M.tier2_by_check("rb_fake", M.node_series(run_dir), Path(root), {})
    assert all(c["monitor"] in CM.VERDICTS for c in rep["checks"][1:])
    assert rep["checks"][0]["monitor"] == CM.NOT_APPLICABLE
    assert rep["signature"]["status"] == "CONTINUE"                     # no Tier-1 verdicts in this ledger
    out = "\n".join(M.format_tier2(rep))
    assert "TWO-TIER STATUS: CONTINUE" in out and "NOT DECLARED" in out
    # the size bar: every decided check carries its windowed size, its state and the COUNTED verdict
    for c in rep["checks"][1:]:
        assert c["size"] in (T2.MATERIAL, T2.NEGLIGIBLE, T2.SIZE_INCONCLUSIVE)
        assert c["size_detail"]["min_step"] == c["grid"] - 8 * D
        assert c["counted"] == T2.counted_monitor(c["monitor"], c["size"])
    assert rep["size_rule"]["bar_pp"] == 1.7 and "size = the recent window" in out
    # the size is a pure function of the rows: a second read gives the same numbers
    again = M.tier2_by_check("rb_fake", M.node_series(run_dir), Path(root), {})
    assert [c["size_detail"] for c in again["checks"]] == [c["size_detail"] for c in rep["checks"]]


def test_the_ladder_edges_orient_both_seat_halves(tmp_path):
    d = tmp_path / "rb_l" / "snapshot_ladder"
    d.mkdir(parents=True)
    lines = [{"a": 4, "b": 2, "mirrored": True, "regime_id": "r", "pair_counts": {"a_p1": [5, 0, 23, 0, 22],
                                                                                    "b_p1": [23, 0, 22, 0, 5]}},
             {"a": 2, "b": 1, "mirrored": False, "regime_id": "r"},
             {"a": 3, "b": 1, "mirrored": True, "regime_id": "q", "pair_counts": {"a_p1": [1, 0, 0, 0, 0],
                                                                                    "b_p1": [1, 0, 0, 0, 0]}}]
    (d / "games.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    edges, meta = M.ladder_edges(d.parent)
    assert edges == [("snap:4", 4, "snap:2", 2, [10.0, 0.0, 45.0, 0.0, 45.0])]
    assert meta["dropped"] == {"no_pair_counts": 1, "other_regime": 1}


def test_the_ladder_read_runs_per_check_and_windows(tmp_path):
    d = tmp_path / "rb_l" / "snapshot_ladder"
    d.mkdir(parents=True)
    rng = np.random.default_rng(4)
    steps = [k * D // 2 for k in range(1, 13)]                                  # a node every 5M
    lines = []
    for i, a in enumerate(steps):
        for b in steps[max(0, i - 4):i]:
            p = 1 / (1 + 10 ** (-(CLIMB[a // D] - CLIMB[b // D] + 30) / 400))
            half = [int(x) for x in np.bincount(2 * (rng.random((50, 2)) < p).sum(1), minlength=5)]
            lines.append({"a": a, "b": b, "mirrored": True, "regime_id": "r",
                          "pair_counts": {"a_p1": half, "b_p1": half[::-1]}})
    (d / "games.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    rep = M.ladder_read(d.parent, rule=FAST, emit=lambda _m: None)
    assert [c["grid"] for c in rep["checks"]] == [D * k for k in range(1, 7)]
    assert all(c["verdict"] in CM.VERDICTS + (CM.NOT_APPLICABLE,) for c in rep["checks"])
    win = M.ladder_read(d.parent, rule=FAST, window_steps=2 * D, emit=lambda _m: None)
    assert win["window_steps"] == 2 * D
    assert win["checks"][-1]["summary"]["nodes"] < rep["checks"][-1]["summary"]["nodes"]
    # the size reads the last 80M of grid whatever --window-steps says; with Tier 1, a DESCRIPTIVE signature
    last = rep["checks"][-1]
    assert last["size_detail"]["min_step"] == last["grid"] - 8 * D and win["checks"][-1]["size_detail"] == \
        last["size_detail"]
    t1 = M.ladder_read(d.parent, rule=FAST, t1_states={5 * D: "FLAT", 6 * D: "FLAT"}, emit=lambda _m: None)
    assert t1["signature"]["status"] in (T2.CONTINUE, T2.PLATEAU_BUT_PANEL) and "DESCRIPTIVE" in t1["signature"]["note"]
    assert "signature" not in rep


def _panel_file(tmp: Path, member: Path, sha: str) -> Path:
    p = tmp / "plateau_panels.json"
    p.write_text(json.dumps({"schema": M.PANELS_SCHEMA, "panels": {"p1": {"members": [
        {"id": "outside", "path": str(member), "sha256": sha}]}}}))
    return p


def test_tick_plays_the_panel_and_status_reads_it(setup, tmp_path):
    run_dir, root = setup
    other = make_run(tmp_path, 3, name="rb_outside")                    # another lineage: a frozen outside member
    member = other / "checkpoints" / f"checkpoint_{3 * D + 11}_steps.zip"
    sha = hashlib.sha256(member.read_bytes()).hexdigest()
    reg = _panel_file(tmp_path, member, sha)
    fake = FakePlay(CLIMB)
    run_tick(run_dir, root, fake, panel="p1", panel_path=reg, t1_root=Path(root))
    panel_cells = sorted({(i, j) for k, i, j, _b in fake.played if k == "panel"})
    assert panel_cells == [(1, 3), (2, 3), (6, 3), (7, 3)]               # 60M / 70M and their W-back 10M / 20M
    assert all(sum(1 for k, i, j, _b in fake.played if k == "panel" and (i, j) == c) == 500 // 50
               for c in panel_cells)                                     # 500 pairs: no Tier-1 FLAT, no candidate
    rep = M.tier2_by_check("rb_fake", M.node_series(run_dir), Path(root), {}, panel="p1", panel_path=reg)
    by = {c["grid"]: c for c in rep["checks"]}
    assert by[7 * D]["panel"] in (T2.RISING, T2.FLAT, T2.UNDECIDED) and by[7 * D]["panel_detail"]["pairs"] == 500
    assert by[4 * D]["panel"] == T2.INCOMPLETE                            # no W-back node 50M before 40M
    # a member whose file no longer hashes to the registry is refused (a moved panel is a different panel)
    bad = _panel_file(tmp_path, member, "0" * 64)
    with pytest.raises(M.MonitorDriverError, match="registered 000000000000"):
        run_tick(run_dir, root, fake, panel="p1", panel_path=bad, t1_root=Path(root))


def test_the_registered_outside_panel_names_frozen_baselines():
    """``designs/research_state/plateau_panels.json``: every member is a named baseline's EXPLICIT checkpoint, its
    sha256 is the baselines registry's, and it says what "outside" means for it (no file is read here)."""
    from agents.training import baselines as B

    panels = M.load_panels()["panels"]
    assert "outside_v1" in panels
    for name, p in panels.items():
        assert p["members"] and p.get("cost_per_check") and p.get("rule"), name
        for m in p["members"]:
            assert {"id", "path", "sha256", "outside", "in_band"} <= set(m), m
            if "baseline" in m:
                b = B.get(m["baseline"])
                assert m["sha256"] == b.sha256 and m["path"] == f"{b.run}/{b.checkpoint}", m
                assert not b.era_checkout_only, m                         # playable at HEAD's architecture


def test_outside_v2_is_six_distinct_in_band_end_state_lineages_and_v1_is_untouched():
    """``outside_v2`` (2026-10-20, F-ED-29) is the design's SIX members: distinct runs, each a named baseline
    and a MEASURED in-band score vs X26's P* and X34's T*; ``outside_v1`` keeps its one member (its rows are
    keyed by the name, so it is never edited in place)."""
    panels = M.load_panels()["panels"]
    assert [m["id"] for m in panels["outside_v1"]["members"]] == ["production"]
    mem = panels["outside_v2"]["members"]
    assert len(mem) == 6
    assert len({m["path"].split("/")[0] for m in mem}) == 6 and len({m["sha256"] for m in mem}) == 6
    assert all(m["baseline"] == m["id"] and m["path"].startswith("rb_es_end2_s") for m in mem)
    assert "production" in {m["id"] for m in mem}
    assert all(m["in_band"].startswith("MEASURED") and "INSIDE" in m["in_band"] for m in mem)


def test_a_member_file_resolves_absolute_or_run_relative(tmp_path):
    f = tmp_path / "x.zip"
    f.write_bytes(b"x")
    assert M.member_file({"id": "a", "path": str(f)}) == str(f)
    with pytest.raises(M.MonitorDriverError, match="does not resolve"):
        M.member_file({"id": "b", "path": "rb_no_such_run_ever/final_model.zip"})
