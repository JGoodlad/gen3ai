"""``main.value_search.live_identity``: the live S′ against the banked offline S′, matched on the decision's identity
(the key AND the frame and honest-log digests). Synthetic files: an identical match passes; a pick or a q-only
difference fails; a log-digest difference is ``key_only`` and never compared; a key twice in ``selected`` refuses;
too few matched fails; every attempt dir is read; the CLI's exit codes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from main.anchors.forced_branch import log_sha256
from main.value_search import live_identity as LI

KEY = "su_live_v1|abcdef0123456789abcd|t12|n13"
KEY2 = "su_live_v1|abcdef0123456789abcd|t13|n14"
LOG = {"format_id": "gen3ou", "seed": "1,2,3,4", "names": ["a", "b"], "teams": ["x", "y"], "cmds": ["CHOOSE p1 move a"]}
FRAME = "f" * 64
SEARCH = {"searched": True, "played": 6, "q": {"3": 0.5, "6": 0.61}, "log_pi": {"3": -0.1, "6": -2.3},
          "root_v": 0.55, "cpu_s": 9.0, "deviation": True}


def _w(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _bank(tmp: Path, *, sel_extra: List[Dict[str, Any]] = (), search=None) -> Dict[str, Path]:
    sel = [{"key": KEY, "log": LOG, "frame_sha256": FRAME, "cell": "pstar__forum72__200"},
           {"key": KEY2, "log": dict(LOG, cmds=[]), "frame_sha256": "e" * 64, "cell": "pstar__forum72__200"},
           *sel_extra]
    sc = [{"key": KEY, "panel": "forum72", "pi_argmax": 3, "search": dict(search or SEARCH)},
          {"key": KEY2, "panel": "forum72", "pi_argmax": 3,
           "search": {"searched": False, "why": "excluded", "played": 3, "cpu_s": 1.0}}]
    return {"selected": _w(tmp / "bank" / "selected.jsonl", sel), "scored": _w(tmp / "bank" / "scored.jsonl", sc)}


def _live(**over: Any) -> Dict[str, Any]:
    r = {"schema": "gen3_live_search_decision_v1", "turn": 12, "n": 13, "key": KEY, "pi_argmax": 3,
         "log_sha256": log_sha256(LOG), "frame_sha256": FRAME, **SEARCH, "cpu_s": 31.0, "wall_s": 40.0}
    r.update(over)
    return r


def _cells(tmp: Path, recs: List[Dict[str, Any]], name: str = "pstar_s__forum72__200__g00") -> Path:
    cells = tmp / "cells"
    _w(cells / name / "search_decisions.jsonl", [{"turn": 3, "n": 3, "searched": False, "why": "before_turn"},
                                                 *recs])
    return cells


def _check(tmp: Path, recs: List[Dict[str, Any]], min_matched: int = 1, **bank_kw: Any) -> Dict[str, Any]:
    b = _bank(tmp, **bank_kw)
    return LI.check(_cells(tmp, recs), b["selected"], b["scored"], min_matched=min_matched)


def test_an_identical_match_passes_and_the_cpu_fields_are_not_compared(tmp_path):
    doc = _check(tmp_path, [_live()])
    assert doc["verdict"] == LI.PASS and doc["reason"] is None
    assert (doc["n_records"], doc["n_matched"], doc["n_key_only"], doc["n_unbanked"]) == (1, 1, 0, 0)
    assert doc["by_panel"] == {"forum72": {"n_matched": 1, "n_key_only": 0}} and doc["mismatches"] == []
    assert doc["schema"] == LI.SCHEMA and len(doc["inputs"]["scored"]["sha256"]) == 64


def test_an_excluded_state_matches_an_offline_exclusion(tmp_path):
    ex = {"key": KEY2, "pi_argmax": 3, "log_sha256": log_sha256(dict(LOG, cmds=[])), "frame_sha256": "e" * 64,
          "searched": False, "why": "excluded", "played": 3, "cpu_s": 2.0}
    assert _check(tmp_path, [ex])["verdict"] == LI.PASS
    doc = _check(tmp_path / "b", [dict(ex, searched=True, why=None, played=6)])
    assert doc["verdict"] == LI.FAIL and {m["field"] for m in doc["mismatches"]} == {"searched", "why", "played"}


def test_a_pick_mismatch_fails(tmp_path):
    doc = _check(tmp_path, [_live(played=3)])
    assert doc["verdict"] == LI.FAIL and doc["mismatches"] == [
        {"key": KEY, "cell": "pstar_s__forum72__200__g00", "field": "played", "live": 3, "offline": 6}]


def test_a_q_only_mismatch_fails(tmp_path):
    doc = _check(tmp_path, [_live(q={"3": 0.5, "6": 0.610001})])
    assert doc["verdict"] == LI.FAIL and [m["field"] for m in doc["mismatches"]] == ["q"]
    assert "mismatch" in doc["reason"]


def test_a_log_digest_mismatch_is_key_only_and_never_compared(tmp_path):
    doc = _check(tmp_path, [_live(log_sha256="0" * 64, played=3)], min_matched=1)
    assert (doc["n_matched"], doc["n_key_only"], doc["mismatches"]) == (0, 1, [])
    assert doc["verdict"] == LI.FAIL and "fewer than min_matched" in doc["reason"]
    doc = _check(tmp_path / "f", [_live(frame_sha256="1" * 64, played=3), _live()], min_matched=1)
    assert (doc["n_matched"], doc["n_key_only"], doc["verdict"]) == (1, 1, LI.PASS)


def test_a_duplicate_key_in_selected_refuses(tmp_path):
    with pytest.raises(LI.IdentityInputError, match="appears twice"):
        _check(tmp_path, [_live()], sel_extra=[{"key": KEY, "log": LOG, "frame_sha256": FRAME}])


def test_too_few_matched_fails(tmp_path):
    doc = _check(tmp_path, [_live()], min_matched=2)
    assert doc["verdict"] == LI.FAIL and doc["reason"] == "1 matched state(s), fewer than min_matched 2"


def test_an_unbanked_key_and_a_record_without_identity(tmp_path):
    doc = _check(tmp_path, [_live(), _live(key="su_live_v1|zz|t20|n22")])
    assert (doc["n_unbanked"], doc["verdict"]) == (1, LI.PASS)
    rec = _live()
    del rec["log_sha256"]
    doc = _check(tmp_path / "x", [rec, _live()])
    assert doc["n_no_identity"] == 1 and doc["verdict"] == LI.FAIL and "carry no log_sha256" in doc["reason"]


def test_every_attempt_dir_is_read_and_reported(tmp_path):
    b = _bank(tmp_path)
    cells = _cells(tmp_path, [_live()])
    _w(cells / "pstar_s__forum72__200__g00.partial.1" / "search_decisions.jsonl", [_live(played=3)])
    (cells / "pstar_s__forum72__200__g01").mkdir()                       # no decision log: not an attempt dir
    doc = LI.check(cells, b["selected"], b["scored"], min_matched=1)
    assert sorted(doc["by_dir"]) == ["pstar_s__forum72__200__g00", "pstar_s__forum72__200__g00.partial.1"]
    assert doc["by_dir"]["pstar_s__forum72__200__g00.partial.1"]["mismatches"] == 1 and doc["verdict"] == LI.FAIL


def test_the_cli_exit_codes(tmp_path, capsys):
    b = _bank(tmp_path)
    cells = _cells(tmp_path, [_live()])
    argv = ["check", "--cells", str(cells), "--selected", str(b["selected"]), "--scored", str(b["scored"])]
    out = tmp_path / "doc.json"
    assert LI.main(argv + ["--min-matched", "1", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == LI.PASS
    assert LI.main(argv + ["--min-matched", "5"]) == 1
    assert LI.main(argv[:-2] + ["--scored", str(tmp_path / "nope.jsonl"), "--min-matched", "1"]) == 2
    printed = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert printed[0]["n_matched"] == 1 and "q" not in json.dumps(printed[0])


# ------------------------------------------------------------------------------------------------ the re-score
def _skey(i: int) -> str:
    """The i-th key (from 0) that IS in the hashed re-score sample."""
    from main.anchors.search_side import rescore_sampled

    found = [k for k in (f"su_live_v1|d{j}|t20|n21" for j in range(5000)) if rescore_sampled(k)]
    return found[i]


def _unkey(i: int) -> str:
    from main.anchors.search_side import rescore_sampled

    return [k for k in (f"su_live_v1|u{j}|t20|n21" for j in range(200)) if not rescore_sampled(k)][i]


def _plan_state(tmp: Path, units: List[tuple]) -> Path:
    """A kakuna state with ``plan.json`` of per-game units ``(name, teamset, unit_seed, game)``; returns its cells."""
    st = tmp / "anchor" / "g" / "pstar_s"
    rows = [{"name": n, "label": "pstar_s", "arm": "pstar_s", "seed": 1, "teamset": ts, "unit_seed": s, "model": "/m",
             "model_sha256": "a" * 64, "games": 1, "game": g, "search": "r0s4_rolls@10"} for n, ts, s, g in units]
    st.mkdir(parents=True)
    (st / "plan.json").write_text(json.dumps({"units": rows}))
    return st / "cells"


def _srec(key: str, **over: Any) -> Dict[str, Any]:
    r = {"key": key, "search": "r0s4_rolls@10", "rescore_sample": True, "log": LOG, "at": 1, "side": "p1",
         "turn": 20, "n": 21, "tokens": {"3": "move a", "6": "move b"}, "mask": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
         "pi_argmax": 3, "frame_sha256": FRAME, "log_sha256": log_sha256(LOG), **SEARCH}
    r.update(over)
    return r


class _Stub:
    """A GateScorer stand-in: returns the live record's own search fields unless told otherwise."""

    def __init__(self, change=None):
        self.change, self.calls = change or {}, []
        self.s = self

    def root(self, rec):
        self.calls.append(rec["key"])
        if rec["key"] in self.change.get("refuse", ()):
            raise RuntimeError("the kept log's replayed row is not the live row we played on")
        return None, {"frame": {}, "rec": rec}

    def _search(self, log, at, side, row, mask, d, key, argmax):
        out = {k: d["rec"].get(k) for k in LI.RESCORE_FIELDS}
        out.update(self.change.get(key, {}))
        return dict(out, cpu_s=2.0)

    def close(self):
        pass


@pytest.fixture
def stub_rows(monkeypatch):
    import numpy as np

    from agents.battle import core_obs

    monkeypatch.setattr(core_obs, "wrap_row", lambda frame, dim=None: np.zeros(4, dtype=np.float32))


def _two_panel_cells(tmp: Path) -> Path:
    cells = _plan_state(tmp, [("pstar_s__p72__201__g00", "p72", 201, 0), ("pstar_s__forum72__200__g01", "forum72", 200, 1),
                              ("pstar_s__forum72__200__g00", "forum72", 200, 0)])
    _w(cells / "pstar_s__p72__201__g00" / "search_decisions.jsonl", [_srec(_skey(0))])
    _w(cells / "pstar_s__forum72__200__g01" / "search_decisions.jsonl", [_srec(_skey(1)), _live(key=_unkey(0))])
    _w(cells / "pstar_s__forum72__200__g00" / "search_decisions.jsonl", [_srec(_skey(2)), _srec(_skey(3))])
    _w(cells / "pstar_s__forum72__200__g00.partial.1" / "search_decisions.jsonl", [_srec(_skey(2))])
    return cells


def test_the_registered_order_is_the_key_hash_first_then_panel_seed_game_and_dir(tmp_path):
    import hashlib

    cells = _two_panel_cells(tmp_path)
    got = LI.registered_order(cells, ["forum72", "p72"], 10)
    cell_of = {_skey(0): ["pstar_s__p72__201__g00"], _skey(1): ["pstar_s__forum72__200__g01"],
               _skey(2): ["pstar_s__forum72__200__g00", "pstar_s__forum72__200__g00.partial.1"],
               _skey(3): ["pstar_s__forum72__200__g00"]}
    want = [{"cell": c, "key": k} for k in sorted(cell_of, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
            for c in cell_of[k]]                       # one key in two attempts: the dir breaks the tie
    assert got == want
    with pytest.raises(LI.IdentityInputError, match="not one of"):
        LI.registered_order(cells, ["forum72"], 10)


def test_the_first_n_of_the_registered_order_spans_panels_and_seeds(tmp_path):
    """M1: on a multi-panel fixture (3 panels x 10 seeds, one sampled record per cell) the first 9 are a hash draw,
    not the first panel's first seeds (a panel-first order would give 9 heldout72_hpfix cells of seeds 300-308)."""
    panels = ["heldout72_hpfix", "forum72", "p72"]
    units = [(f"pstar_s__{ts}__{s}__g00", ts, s, 0) for ts in panels for s in range(300, 310)]
    cells = _plan_state(tmp_path, units)
    for i, (name, _ts, _s, _g) in enumerate(units):
        _w(cells / name / "search_decisions.jsonl", [_srec(_skey(i))])
    first = LI.sampled(cells, panels)[:9]
    assert len({e["panel"] for e in first}) == 3
    assert max(e["unit_seed"] for e in first) - min(e["unit_seed"] for e in first) >= 5


def test_a_sample_flag_that_is_not_the_keys_hash_refuses(tmp_path):
    cells = _plan_state(tmp_path, [("pstar_s__p72__201__g00", "p72", 201, 0)])
    _w(cells / "pstar_s__p72__201__g00" / "search_decisions.jsonl", [_live(key=_unkey(1), rescore_sample=True)])
    with pytest.raises(LI.IdentityInputError, match="not the key's hash"):
        LI.sampled(cells, ["p72"])
    _w(cells / "pstar_s__p72__201__g00" / "search_decisions.jsonl", [_live(key=_skey(4))])     # sampled, not flagged
    with pytest.raises(LI.IdentityInputError, match="not the key's hash"):
        LI.sampled(cells, ["p72"])


def _rescore(tmp: Path, cells: Path, stub: Any, **kw: Any) -> Dict[str, Any]:
    return LI.rescore(cells, ["forum72", "p72"], out=tmp / "rescore.json", model="/m", model_sha256="a" * 64,
                      engine="E", scorer=stub, **kw)


def test_an_identical_rescore_passes_and_covers_the_first_n(tmp_path, stub_rows):
    cells = _two_panel_cells(tmp_path)
    doc = _rescore(tmp_path, cells, _Stub(), max_n=4, min_n=4)
    assert doc["verdict"] == LI.PASS and doc["n_sampled"] == 5 and doc["n_rescored"] == 4
    assert doc["rescored"] == LI.registered_order(cells, ["forum72", "p72"], 4)
    panels = [("p72" if "p72" in e["cell"] else "forum72") for e in doc["rescored"]]
    assert doc["by_panel"] == {p: panels.count(p) for p in set(panels)} and sum(doc["by_panel"].values()) == 4
    assert json.loads((tmp_path / "rescore.json").read_text()) == doc


def test_a_pick_or_q_or_root_guard_difference_fails_and_too_few_fails(tmp_path, stub_rows):
    cells = _two_panel_cells(tmp_path)
    doc = _rescore(tmp_path / "a", cells, _Stub({_skey(1): {"played": 3}, _skey(0): {"q": {"3": 0.5, "6": 0.6}}}))
    assert doc["verdict"] == LI.FAIL and sorted(m["field"] for m in doc["mismatches"]) == ["played", "q"]
    doc = _rescore(tmp_path / "b", cells, _Stub({"refuse": [_skey(0)]}))
    assert [m["field"] for m in doc["mismatches"]] == ["root_guard"] and doc["verdict"] == LI.FAIL
    doc = _rescore(tmp_path / "c", cells, _Stub(), min_n=6)
    assert doc["verdict"] == LI.FAIL and "fewer than --min 6" in doc["reason"]


def test_the_rescore_resumes_without_rescoring_a_done_state(tmp_path, stub_rows):
    cells = _two_panel_cells(tmp_path)
    first = _Stub()
    _rescore(tmp_path, cells, first, max_n=2, min_n=1)
    again = _Stub()
    doc = _rescore(tmp_path, cells, again, max_n=5, min_n=1)
    assert len(first.calls) == 2 and len(again.calls) == 3 and doc["n_rescored"] == 5


# ------------------------------------------------------------------------------------------------ the bank tooth
def _bkey(prefix: str, n: int, sampled: bool = True) -> List[str]:
    return [k for k in (f"su_live_v1|{prefix}{j}|t15|n16" for j in range(20000)) if LI.bank_sampled(k) == sampled][:n]


def _bank_files(tmp: Path, keys: List[str], unscored: List[str] = (), others: List[str] = ()) -> Dict[str, Path]:
    sel = [_srec(k, panel="forum72") for k in [*keys, *unscored, *others]]
    blk = {k: v for k, v in SEARCH.items()} | {"config": {"name": "r0s4_rolls"}, "spent": 16, "wall_s": 3.0,
                                               "world_cpu_s": 2.0}
    sc = [{"key": k, "panel": "forum72", "pi_argmax": 3, "search": dict(blk)} for k in [*keys, *others]]
    return {"selected": _w(tmp / "b" / "selected.jsonl", sel), "scored": _w(tmp / "b" / "scored.jsonl", sc)}


class _BankStub(_Stub):
    """Reproduces the bank row's block (timings different) unless told otherwise."""

    def __init__(self, bank: Dict[str, Any], change=None):
        super().__init__(change)
        self.bank = bank

    def _search(self, log, at, side, row, mask, d, key, argmax):
        out = dict(self.bank[key], cpu_s=99.0, wall_s=99.0, world_cpu_s=1.0)
        out.update(self.change.get(key, {}))
        return out


def _tooth(tmp: Path, files: Dict[str, Path], stub: Any, **kw: Any) -> Dict[str, Any]:
    return LI.bank_tooth(files["selected"], files["scored"], out=tmp / "tooth.json", search="r0s4_rolls@10",
                         model="/m", model_sha256="a" * 64, engine="E", scorer=stub, **kw)


def _bank_of(files: Dict[str, Path]) -> Dict[str, Any]:
    return {json.loads(x)["key"]: json.loads(x)["search"] for x in files["scored"].read_text().splitlines()}


def test_the_bank_tooth_samples_by_hash_skips_unscored_and_caps(tmp_path, stub_rows):
    import hashlib

    keys, unscored, others = _bkey("k", 5), _bkey("u", 2), _bkey("o", 3, sampled=False)
    files = _bank_files(tmp_path, keys, unscored, others)
    stub = _BankStub(_bank_of(files))
    doc = _tooth(tmp_path, files, stub, max_n=4)
    want = [k for k in sorted(keys + unscored, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
            if k in keys][:4]
    assert doc["compared"] == want and stub.calls == want
    assert (doc["n_sampled"], doc["n_unscored_skipped"], doc["n_compared"], doc["verdict"]) == (7, 2, 4, LI.PASS)
    assert not set(others) & set(stub.calls), "an unsampled key is never compared"


def test_the_bank_tooth_fails_on_any_non_timing_difference_and_ignores_timings(tmp_path, stub_rows):
    keys = _bkey("k", 3)
    files = _bank_files(tmp_path, keys)
    bank = _bank_of(files)
    assert _tooth(tmp_path / "a", files, _BankStub(bank))["verdict"] == LI.PASS          # cpu / wall differ: ignored
    doc = _tooth(tmp_path / "b", files, _BankStub(bank, {keys[0]: {"spent": 17}, keys[1]: {"played": 3}}))
    assert doc["verdict"] == LI.FAIL and sorted(m["field"] for m in doc["mismatches"]) == ["played", "spent"]
    doc = _tooth(tmp_path / "c", files, _BankStub(bank, {"refuse": [keys[2]]}))
    assert [m["field"] for m in doc["mismatches"]] == ["root_guard"]


def test_one_thread_sets_the_blas_env_inside_the_function(monkeypatch):
    import os

    import torch

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(k, "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads()
    try:
        LI.one_thread()
        assert all(os.environ[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
