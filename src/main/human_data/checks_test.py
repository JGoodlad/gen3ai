"""``checks reencode`` (PREREG Amendment 5 §A5.7, M3): the obs-slice naming, the byte comparison, and the rid matching
against a registered rowset, on fixtures (the per-battle extractor is a stub; the real one needs the live reader)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from main.human_data import checks as C
from main.human_data import rowset as RS


@pytest.fixture(scope="module")
def encoder():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings())


@pytest.fixture(scope="module")
def layout(encoder):
    return encoder.get_layout()


def _col(layout, team: str, slot: int, blk: str) -> int:
    p = layout["parts"][team]
    return p["start"] + slot * p["reshape"][1] + layout["pokemon"][blk]["offset"]


def test_obs_slices_name_every_column_from_the_layout(layout, encoder):
    names, col = C.obs_slices(layout)
    assert col.shape == (layout["total_dim"],) and (col >= 0).all()
    assert len(names) == len(set(names))
    assert names[col[_col(layout, "opp_team", 3, "hp")]] == "opp_team.hp"
    f0 = layout["obs_facts_offset"] + layout["obs_facts"]["vol"]["offset"]
    assert names[col[f0]] == "obs_facts.vol"
    assert names[col[layout["event_window_offset"]]] == "event_window"
    assert names[col[layout["parts"]["context"]["start"]]] == "context"
    names, col = C.obs_slices(layout, encoder.active_context_encoder.get_layout())
    assert (col >= 0).all() and "context" not in names
    ctx = encoder.active_context_encoder.get_layout()
    side1 = layout["parts"]["context"]["start"] + layout["parts"]["context"]["reshape"][1]
    assert names[col[side1 + ctx["volatiles"]["offset"]]] == "context.volatiles"
    assert names[col[layout["parts"]["context"]["start"] + ctx["boosts"]["offset"]]] == "context.boosts"


def _rows(n: int, dim: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"rows": rng.random((n, dim), dtype=np.float32), "mask": np.ones((n, RS.N_ACT), np.int8),
            "act": np.zeros(n, np.int8), "hidden": np.zeros(n, bool), "cons": np.ones((n, RS.N_ACT), np.int8),
            "moveset": np.zeros((n, RS.N_ACT), np.int8), "rid": np.arange(n, dtype=np.int64) + 100,
            "bkey": np.zeros(n, np.int64), "acct": np.zeros(n, np.int64), "seat": np.zeros(n, np.int8),
            "k": np.arange(n, dtype=np.int16), "turn": np.arange(n, dtype=np.int16), "forced": np.zeros(n, bool),
            "rating": np.full(n, 1400, np.int16), "quit_dist": np.full(n, -1, np.int16),
            "end": np.zeros(n, np.int8)}


def test_compare_rows_counts_bytes_fields_and_slices(layout):
    names, col = C.obs_slices(layout)
    ref = _rows(8, layout["total_dim"])
    ours = {f: v.copy() for f, v in ref.items()}
    ours["rows"][0, _col(layout, "our_team", 2, "hp")] += 0.5
    ours["rows"][1, layout["obs_facts_offset"] + layout["obs_facts"]["vol"]["offset"]] += 1.0
    ref["rows"][2, layout["event_window_offset"]] = 0.0
    ours["rows"][2, layout["event_window_offset"]] = -0.0          # equal as floats, different BYTES
    ours["mask"][3, 4] = 0
    ours["act"][4] = 3
    ours["turn"][5] = 99                                           # metadata only
    ours["cons"][6, 1] = 0
    c = C.compare_rows(ours, ref, names, col)
    assert c["rows_compared"] == 8
    assert c["rows_differing_obs"] == 3
    assert c["rows_differing_by_obs_slice"] == {"event_window": 1, "obs_facts.vol": 1, "our_team.hp": 1}
    assert c["obs_columns_differing"] == 3 and {d["slice"] for d in c["obs_columns"]} == set(
        c["rows_differing_by_obs_slice"])
    assert (c["rows_differing_mask"], c["rows_differing_consistent_set"], c["rows_differing_label"]) == (1, 1, 1)
    assert c["rows_differing_label_by_field"] == {"act": 1, "hidden": 0, "moveset": 0}
    assert c["rows_differing_meta_only"] == 1
    assert c["rows_differing"] == 6 and c["share_differing"] == pytest.approx(6 / 8)
    with pytest.raises(ValueError):
        C.compare_rows({**ours, "rid": ours["rid"][::-1].copy()}, ref, names, col)


def _write_registered(d: Path, arr: Dict[str, np.ndarray]) -> None:
    d.mkdir(parents=True)
    tr = RS._write_shard(d, "train_0000", {f: v[:3] for f, v in arr.items()})
    va = RS._write_shard(d, "val", {f: v[3:] for f, v in arr.items()})
    (d / "rows_manifest.json").write_text(json.dumps({
        "rowset": "fixture.r2", "train_shards": [tr], "val": va, "extraction_commit": ["old"],
        "source_sha256": "old", "live_reader_sha256": "old", "layout_hash": "old"}))


def test_reencode_matches_by_rid_and_counts_skips(tmp_path, monkeypatch, layout):
    battles = [f"2026-05-0{i}/battle-gen3ou-{i}.log" for i in range(1, 5)]
    prefix = "/corpus/"
    parts = tmp_path / "r1" / "parts"
    parts.mkdir(parents=True)
    (parts / "part_0000.json").write_text(json.dumps({"battles": battles}))
    hom = tmp_path / "hom"
    hom.mkdir()
    (hom / "humans_0000.json").write_text(json.dumps({"part_size": 400, "battles": [prefix + b for b in battles]}))
    keep = [{"rel": b, "sha": "x", "n": 1, "seats": ["p1"]} for b in battles]
    with gzip.open(parts.parent / "keep.jsonl.gz", "wt") as fh:
        fh.write("".join(json.dumps(k) + "\n" for k in keep))
    (parts.parent / "scan.json").write_text(json.dumps(
        {"snapshot": "snapX", "keep_sha256": RS.S_sha(parts.parent / "keep.jsonl.gz")}))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"snapshots": {"snapX": {"population": "pop"}}}))

    sample = C.vo8_sample(parts, hom, 2, prefix)
    assert len(sample.chosen) == 2 and sample.chosen == sorted(sample.chosen)
    dim = layout["total_dim"]
    today: Dict[str, Dict[str, np.ndarray]] = {}
    for i, rel in enumerate(sample.chosen):
        a = _rows(3, dim, seed=i)
        a["rid"] = np.asarray([RS.key64(f"{rel}|p1|{k}") for k in range(3)], np.int64)
        a["bkey"][:] = RS.key64(rel)
        today[rel] = a
    first, second = sample.chosen
    today[second]["quit_dist"][2] = 0                       # the derived rowset dropped it (the quitter's tail)
    reg_rows: List[Dict[str, np.ndarray]] = []
    for rel in sample.chosen:
        reg_rows.append({f: v.copy() for f, v in today[rel].items()})
    cat = RS._concat(reg_rows)
    keep_reg = cat["rid"] != today[second]["rid"][2]
    reg = {f: v[keep_reg] for f, v in cat.items()}
    reg["rows"][1, _col(layout, "opp_team", 0, "status_counters")] += 1.0   # one row's obs moved since
    extra = _rows(1, dim, seed=9)
    extra["rid"][:] = RS.key64(f"{first}|p1|7")             # a registered row today's extraction no longer makes
    extra["bkey"][:] = RS.key64(first)
    reg = RS._concat([reg, extra])
    _write_registered(tmp_path / "r2" / "rows", reg)

    monkeypatch.setattr(RS, "_code_identity", lambda: {
        "commit": "new", "source_sha256": "new", "sources_dirty_vs_commit": False, "live_reader_sha256": "new",
        "encoder_identity": {}, "layout_hash": "new"})
    monkeypatch.setattr(C, "_reader_bin", lambda: "/stub/live_reader")
    seen = []

    def stub(args):
        snap, population, corpus, meta = args
        seen.append((snap, population, meta["rel"]))
        return {"counts": {"seats_read": 1}, "parts": [today[meta["rel"]]]}

    res = C.reencode(tmp_path / "r2" / "rows", parts, hom, 2, prefix, tmp_path, registry, Path("/corpus"),
                     extract=stub)
    assert seen == [("snapX", "pop", rel) for rel in sample.chosen]
    assert res["rows_today"] == 6
    assert (res["rows_skipped_not_in_registered"], res["rows_skipped_quitter_tail"]) == (1, 1)
    assert res["registered_rows_not_made_today"] == 1
    assert res["rows_compared"] == 5
    assert res["rows_differing"] == 1 and res["rows_differing_obs"] == 1
    assert res["rows_differing_by_obs_slice"] == {"opp_team.status_counters": 1}
    assert res["over_limit"] is True and res["pass"] is True
    assert res["extract_counts"] == {"seats_read": 2}
    assert res["today"]["live_reader_bin"] == "/stub/live_reader" and res["registered"]["source_sha256"] == "old"
