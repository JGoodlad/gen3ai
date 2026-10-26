r"""A DERIVED rowset: a registered rowset's rows, filtered by row rules, with its P\* label columns carried along
(X59-S PREREG Amendment 2: REVIEW E M1 and B2, before any student trains).

    python -m main.human_data.derive --parent <r1 dir> --out <r2 dir> --id hr_spec_v1.r2 [--drop-accounts <json>]

Writes ``<out>/rows`` (``rowset``'s shard format, ``rows_manifest.json``), ``<out>/labels_pstar`` (S-P) and
``<out>/labels_pstar0`` (S-P0). The rules, each counted:

* **The quitter's tail (M1):** a row of the seat that forfeited or timed out with ``0 <= quit_dist <=``
  :data:`QUIT_TAIL` is dropped: the quitter's last ``QUIT_TAIL + 1`` decisions leave (the parent already dropped
  the last ``rowset.QUIT_DROP``). REVIEW E measured P\*'s agreement with the quitter FALLING over its last ~8 kept
  decisions while it RISES for the forfeit winner and both seats of KO games (``results/review_e_checks.json``).
* **Flagged accounts (B2):** every row (TRAIN and VALIDATION) of an account on the Kakuna-family audit's drop list,
  when the audit's registered rule fires (``main.human_data.audit``; an absent list drops none).

The surviving rows keep their order and are re-cut into shards of ``rowset.SHARD``. Their labels are the parent's:

* ``labels_pstar`` (S-P): each surviving row's parent label, unchanged (its matched uniform stays keyed by the
  PARENT rows manifest, so no row's label moves);
* ``labels_pstar0`` (S-P0, REVIEW E M2): one action of P\* at T = 1 over the LEGAL actions (unrestricted), drawn
  with the SAME matched uniform from the parent label column's stored logits; a hidden row keeps the set label (-1).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from main.human_data import rowset as RS
from main.human_data import teacher as TE

QUIT_TAIL = 9
LABEL_FIELDS = ("act_p", "rid", "logits", "out_mass")


def load_drop_list(path: Optional[Path]) -> Set[int]:
    """The audit's drop list: JSON ``{"drop_accounts": [acct keys]}`` (empty when the rule did not fire)."""
    if path is None:
        return set()
    body = json.loads(Path(path).read_text())
    return {int(x) for x in body["drop_accounts"]}


def keep_mask(z: Dict[str, np.ndarray], drop: Set[int], counts: collections.Counter) -> np.ndarray:
    """The rows that survive the derived rules (module docstring); counts every dropped row by its first reason."""
    q = z["quit_dist"].astype(np.int64)
    tail = (q >= 0) & (q <= QUIT_TAIL)
    acct = np.isin(z["acct"], np.fromiter(drop, np.int64, len(drop))) if drop else np.zeros(len(q), bool)
    counts["drop:quitter_tail"] += int(tail.sum())
    counts["drop:flagged_account"] += int((acct & ~tail).sum())
    return ~(tail | acct)


def _labels(parent_labels: Path, rec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    js = json.loads((parent_labels / f"{rec['name']}.json").read_text())
    if js["rows_sha256"] != rec["sha256"]:
        raise RS.RowsetError(f"{rec['name']}: the parent label column was made on other rows")
    p = parent_labels / f"{rec['name']}.npz"
    if RS.S_sha(p) != js["sha256"]:
        raise RS.RowsetError(f"{p}: sha256 differs from its label record")
    z = np.load(p)
    return {f: z[f] for f in LABEL_FIELDS}


def pstar0(logits: np.ndarray, mask: np.ndarray, hidden: np.ndarray, rid: np.ndarray, parent_sha: str) -> np.ndarray:
    r"""S-P0's label (module docstring): P\* at T = 1 over the legal actions, the parent's matched uniform."""
    a = TE.inverse_cdf(TE.masked_probs(logits, mask), TE.matched_uniform(parent_sha, rid))
    a[hidden.astype(bool)] = -1
    return a


def _write_labels(d: Path, name: str, rows_rec: Dict[str, Any], lab: Dict[str, np.ndarray]) -> Dict[str, Any]:
    import os

    p = d / f"{name}.npz"
    tmp = d / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **lab)  # type: ignore[arg-type]
    os.replace(tmp, p)
    r = {"name": name, "rows_sha256": rows_rec["sha256"], "sha256": RS.S_sha(p), "rows": int(len(lab["act_p"]))}
    (d / f"{name}.json").write_text(json.dumps(r, indent=1))
    return r


def derive(parent: Path, out: Path, rowset_id: str, drop_path: Optional[Path]) -> Dict[str, Any]:
    """The derived rowset and its two label columns (module docstring)."""
    from utils.git import get_git_hash

    prow, plab = parent / "rows", parent / "labels_pstar"
    pman, psha = RS.load_manifest(prow)
    plman = json.loads((plab / "labels_manifest.json").read_text())
    if plman["rows_manifest_sha256"] != psha:
        raise RS.RowsetError("the parent's S-P labels were made on other rows")
    drop = load_drop_list(drop_path)
    rdir, l1, l0 = out / "rows", out / "labels_pstar", out / "labels_pstar0"
    for d in (rdir, l1, l0):
        d.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter = collections.Counter()
    shards: List[Dict[str, Any]] = []
    lrec1: List[Dict[str, Any]] = []
    lrec0: List[Dict[str, Any]] = []

    def emit(name: str, rows: Dict[str, np.ndarray], lab: Dict[str, np.ndarray]) -> Dict[str, Any]:
        rec = RS._write_shard(rdir, name, rows)
        lrec1.append(_write_labels(l1, name, rec, lab))
        a0 = pstar0(lab["logits"], rows["mask"], rows["hidden"], rows["rid"], psha)
        lrec0.append(_write_labels(l0, name, rec, {**lab, "act_p": a0}))
        return rec

    carry_r: List[Dict[str, np.ndarray]] = []
    carry_l: List[Dict[str, np.ndarray]] = []
    n_carry = 0
    for rec in pman["train_shards"]:
        z = RS.load_shard(prow, rec)
        lab = _labels(plab, rec)
        if not np.array_equal(lab["rid"], z["rid"]):
            raise RS.RowsetError(f"{rec['name']}: the parent label column's row ids differ from the rows'")
        k = keep_mask(z, drop, counts)
        carry_r.append({f: z[f][k] for f in RS.FIELDS})
        carry_l.append({f: lab[f][k] for f in LABEL_FIELDS})
        n_carry += int(k.sum())
        while n_carry >= RS.SHARD:
            cr = {f: np.concatenate([t[f] for t in carry_r]) for f in RS.FIELDS}
            cl = {f: np.concatenate([t[f] for t in carry_l]) for f in LABEL_FIELDS}
            shards.append(emit(f"train_{len(shards):04d}", {f: cr[f][:RS.SHARD] for f in RS.FIELDS},
                               {f: cl[f][:RS.SHARD] for f in LABEL_FIELDS}))
            carry_r = [{f: cr[f][RS.SHARD:] for f in RS.FIELDS}]
            carry_l = [{f: cl[f][RS.SHARD:] for f in LABEL_FIELDS}]
            n_carry -= RS.SHARD
    if n_carry:
        shards.append(emit(f"train_{len(shards):04d}", {f: np.concatenate([t[f] for t in carry_r]) for f in RS.FIELDS},
                           {f: np.concatenate([t[f] for t in carry_l]) for f in LABEL_FIELDS}))
    vc: collections.Counter = collections.Counter()
    z = RS.load_shard(prow, pman["val"])
    lab = _labels(plab, pman["val"])
    k = keep_mask(z, drop, vc)
    vrec = emit("val", {f: z[f][k] for f in RS.FIELDS}, {f: lab[f][k] for f in LABEL_FIELDS})
    man = {k_: v for k_, v in pman.items() if k_ not in ("train_shards", "val", "train_rows", "val_rows",
                                                          "train_hidden", "row_profile", "rowset")}
    man.update({"rowset": rowset_id, "parent_rowset": pman["rowset"], "parent_rows_manifest_sha256": psha,
                "derived": {"quitter_tail": f"0 <= quit_dist <= {QUIT_TAIL} dropped (the quitter's last "
                                            f"{QUIT_TAIL + 1} decisions)",
                            "flagged_accounts": len(drop),
                            "drop_list_sha256": RS.S_sha(drop_path) if drop_path is not None else None,
                            "train_counts": dict(counts), "val_counts": dict(vc)},
                "train_shards": shards, "val": vrec, "train_rows": int(sum(s["rows"] for s in shards)),
                "val_rows": vrec["rows"], "train_hidden": int(sum(s["hidden"] for s in shards)),
                "derive_commit": get_git_hash()})
    man["filter_spec"] = {**pman["filter_spec"], "quitter_tail_dropped": QUIT_TAIL + 1}
    man["row_profile"] = RS._row_profile(rdir, shards, vrec)
    (rdir / "rows_manifest.json").write_text(json.dumps(man, indent=1))
    sha = RS.S_sha(rdir / "rows_manifest.json")
    for d, recs, schema, rule in (
            (l1, lrec1, "gen3_x59s_pstar_labels_v1", plman["rule"] + "; carried unchanged from the parent rowset's "
             "column (its uniform keyed by the PARENT rows manifest sha256)"),
            (l0, lrec0, "gen3_x59s_pstar0_labels_v1", "one action from P* at T = 1 over the LEGAL actions "
             "(unrestricted, S-P0), uniform = sha256('x59s_pstar_label|' + PARENT rows manifest sha256 + '|' + "
             "row id), from the parent column's stored logits; hidden rows: -1 (the set label)")):
        (d / "labels_manifest.json").write_text(json.dumps(
            {"schema": schema, "rows_manifest_sha256": sha, "rowset": rowset_id,
             "parent_rows_manifest_sha256": psha, "parent_labels_manifest_sha256": RS.S_sha(plab / "labels_manifest.json"),
             "pstar": plman["pstar"], "pstar_sha256": plman["pstar_sha256"], "rule": rule, "shards": recs}, indent=1))
    return {"rows_manifest_sha256": sha, **man}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.derive", description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent", required=True, help="the parent rowset dir (holding rows/ and labels_pstar/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--drop-accounts", default=None, help="the audit's drop list (JSON, drop_accounts)")
    a = ap.parse_args(argv)
    try:
        RS._guard_out(Path(a.out))
        m = derive(Path(a.parent), Path(a.out), a.id, Path(a.drop_accounts) if a.drop_accounts else None)
    except RS.RowsetError as exc:
        print(f"[derive] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({k: m[k] for k in ("rowset", "rows_manifest_sha256", "train_rows", "val_rows", "train_hidden",
                                        "derived")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
