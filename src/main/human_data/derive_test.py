"""The derived rowset (``derive.py``, X59-S Amendment 2): the quitter-tail and flagged-account rules, the carried S-P
labels, the unrestricted S-P0 labels, and that the trainer reads what it writes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from main.human_data import derive as DV
from main.human_data import rowset as RS
from main.human_data import student as ST
from main.human_data import teacher as TE


def _parent(d: Path, n: int = 40000, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    rows = d / "rows"
    lab = d / "labels_pstar"
    rows.mkdir(parents=True)
    lab.mkdir()

    def arr(k):
        mask = np.zeros((k, 11), np.int8)
        mask[:, [0, 1, 6, 7, 8]] = 1
        cons = mask.copy()
        cons[:, 1] = 0
        ms = np.zeros((k, 11), np.int8)
        ms[:, [6, 7, 8]] = 1
        hidden = rng.random(k) < 0.1
        act = np.where(hidden, -1, rng.choice([0, 6, 7, 8], k)).astype(np.int8)
        q = np.where(rng.random(k) < 0.4, rng.integers(2, 30, k), -1).astype(np.int16)
        return {"rows": rng.normal(size=(k, 4)).astype(np.float32), "mask": mask, "act": act, "hidden": hidden,
                "cons": cons, "moveset": ms, "rid": rng.integers(0, 2 ** 62, k), "bkey": rng.integers(0, 50, k),
                "acct": rng.integers(0, 20, k).astype(np.int64), "seat": np.zeros(k, np.int8),
                "k": np.zeros(k, np.int16), "turn": np.zeros(k, np.int16), "forced": np.zeros(k, bool),
                "rating": np.full(k, 1400, np.int16), "quit_dist": q, "end": np.zeros(k, np.int8)}

    recs = []
    for name, k in (("train_0000", RS.SHARD), ("train_0001", RS.SHARD), ("train_0002", n - 2 * RS.SHARD), ("val", 500)):
        z = arr(k)
        rec = RS._write_shard(rows, name, z)
        lg = rng.normal(size=(k, 11)).astype(np.float32)
        u = TE.matched_uniform("PARENT", z["rid"])
        ap = TE.inverse_cdf(TE.masked_probs(lg, z["cons"]), u)
        ap[z["hidden"]] = -1
        np.savez_compressed(lab / f"{name}.npz", act_p=ap, rid=z["rid"], logits=lg, out_mass=np.zeros(k, np.float32))
        (lab / f"{name}.json").write_text(json.dumps({"name": name, "rows_sha256": rec["sha256"],
                                                      "sha256": RS.S_sha(lab / f"{name}.npz"), "rows": k}))
        recs.append(rec)
    man = {"rowset": "t.r1", "train_shards": recs[:3], "val": recs[3], "train_rows": n, "val_rows": 500,
           "train_hidden": 0, "filter_spec": {}}
    (rows / "rows_manifest.json").write_text(json.dumps(man))
    (lab / "labels_manifest.json").write_text(json.dumps({"rows_manifest_sha256": RS.S_sha(rows / "rows_manifest.json"),
                                                          "rule": "r", "pstar": "p", "pstar_sha256": "s"}))


def _all(d: Path, fields, sub="rows"):
    man = json.loads((d / "rows" / "rows_manifest.json").read_text())
    recs = man["train_shards"]
    return {f: np.concatenate([np.load(d / sub / f"{r['name']}.npz")[f] for r in recs]) for f in fields}


def test_derive_drops_the_tail_and_the_flagged_accounts_and_carries_the_labels(tmp_path):
    par, out = tmp_path / "r1", tmp_path / "r2"
    _parent(par)
    drop = tmp_path / "drop.json"
    drop.write_text(json.dumps({"drop_accounts": [3, 7]}))
    m = DV.derive(par, out, "t.r2", drop)
    P = _all(par, ("rid", "quit_dist", "acct"))
    keep = ~(((P["quit_dist"] >= 0) & (P["quit_dist"] <= DV.QUIT_TAIL)) | np.isin(P["acct"], [3, 7]))
    D = _all(out, ("rid", "quit_dist", "acct", "mask", "hidden"))
    assert np.array_equal(D["rid"], P["rid"][keep])                     # order kept, exactly the survivors
    assert m["train_rows"] == int(keep.sum()) and all(s["rows"] == RS.SHARD for s in m["train_shards"][:-1])
    assert not ((D["quit_dist"] >= 0) & (D["quit_dist"] <= 9)).any() and not np.isin(D["acct"], [3, 7]).any()
    assert ((D["quit_dist"] == -1) | (D["quit_dist"] >= 10)).all()
    PL = _all(par, ("act_p",), "labels_pstar")
    L1 = _all(out, ("act_p",), "labels_pstar")
    assert np.array_equal(L1["act_p"], PL["act_p"][keep])               # S-P's label never moves
    L0 = _all(out, ("act_p",), "labels_pstar0")
    lab = ~D["hidden"]
    assert (L0["act_p"][~lab] == -1).all() and (D["mask"][np.nonzero(lab)[0], L0["act_p"][lab]] == 1).all()
    assert (L0["act_p"][lab] == 1).any()                                # unrestricted: the closed-to-human action
    assert m["derived"]["train_counts"]["drop:quitter_tail"] == int(((P["quit_dist"] >= 0)
                                                                     & (P["quit_dist"] <= 9)).sum())


def test_the_trainer_reads_the_derived_columns(tmp_path):
    par, out = tmp_path / "r1", tmp_path / "r2"
    _parent(par)
    DV.derive(par, out, "t.r2", None)
    man, _ = RS.load_manifest(out / "rows")
    for arm, labels in (("pstar", out / "labels_pstar"), ("pstar0", out / "labels_pstar0")):
        z = ST._shard(out / "rows", man["train_shards"][0], arm, labels)
        assert z["target"].sum(1).min() >= 1
