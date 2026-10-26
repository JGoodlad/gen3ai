"""X59-S's pre-training checks (PREREG §2.4, §7): the input registration and the extraction reproduction.

    python -m main.human_data.checks hom-repro --rows <rowset>/rows --parts <rowset>/parts \
        --hom ~/gen3ai_archive/x50_r1b/train --n 200 --out <dir>/results/hom_repro.json
    python -m main.human_data.checks register --rows <rowset>/rows --labels <rowset>/labels_pstar \
        --labels0 <rowset>/labels_pstar0 --selfplay <dir>/ss_rows --out <dir>/results/registration.json

* ``hom-repro`` (VOID 8): a seeded sample of kept battles that the rung-1b HOM extraction also read; for every
  decision both extractions hold, the completed row must be BYTE-equal and the own-action label equal. The rung-1b
  parts keep a decision when either label is known, so a hidden-move row is compared where that extraction kept it.
* ``register``: the sha256 of the rows manifest, the S-P labels manifest and the S-S rows manifest, their row counts
  and the cross-checks the trainer relies on (S-S has the rowset's train / validation counts; the labels were made on
  this rowset; Amendment 2: the S-P0 labels too, and S-S's team draw faced the human panel rule). Written BEFORE any
  student trains; the trainer refuses any other input (``student.check_registration``).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main.human_data import rowset as RS

SEED = 20261025


def hom_repro(rows: Path, parts: Path, hom: Path, n: int, corpus_prefix: str) -> Dict[str, Any]:
    """VOID 8 (module docstring)."""
    rel_of: Dict[int, str] = {}
    part_of: Dict[str, str] = {}
    for js in sorted(parts.glob("part_*.json")):
        for rel in json.loads(js.read_text())["battles"]:
            rel_of[RS.key64(rel)] = rel
            part_of[rel] = js.stem
    hom_index: Dict[str, tuple] = {}
    for js in sorted(hom.glob("humans_*.json")):
        m = json.loads(js.read_text())
        first = int(js.stem.rsplit("_", 1)[1]) * int(m["part_size"])
        for i, b in enumerate(m["battles"]):
            if b.startswith(corpus_prefix):
                hom_index[b[len(corpus_prefix):]] = (js.stem, first + i)
    both = sorted(r for r in rel_of.values() if r in hom_index)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(both), size=min(n, len(both)), replace=False).tolist())
    chosen = [both[i] for i in pick]
    by_hom: Dict[str, List[str]] = collections.defaultdict(list)
    by_ours: Dict[str, List[str]] = collections.defaultdict(list)
    for rel in chosen:
        by_hom[hom_index[rel][0]].append(rel)
        by_ours[part_of[rel]].append(rel)
    ours: Dict[tuple, tuple] = {}
    for part, rels in by_ours.items():
        zf = np.load(parts / f"{part}.npz")
        z = {f: zf[f] for f in ("bkey", "seat", "k", "act")}      # an NpzFile decompresses on EVERY access
        keys = {RS.key64(r): r for r in rels}
        sel = np.isin(z["bkey"], list(keys))
        idx = np.nonzero(sel)[0]
        rows = zf["rows"][idx] if len(idx) else None
        for n_, j in enumerate(idx):
            ours[(keys[int(z["bkey"][j])], int(z["seat"][j]), int(z["k"][j]))] = (rows[n_], int(z["act"][j]))
        del zf
    c = collections.Counter()
    for part, rels in by_hom.items():
        zf = np.load(hom / f"{part}.npz")
        z = {f: zf[f] for f in ("battle", "mode", "seat", "k", "own_action", "rows")}
        del zf
        for rel in rels:
            bi = hom_index[rel][1]
            sel = np.nonzero((z["battle"] == bi) & (z["mode"] == 2))[0]
            for j in sel:
                key = (rel, int(z["seat"][j]), int(z["k"][j]))
                if key not in ours:
                    continue
                row, act = ours[key]
                c["compared"] += 1
                c["row_equal"] += int(np.array_equal(row.view(np.uint32), z["rows"][j].astype(np.float32).view(np.uint32)))
                c["label_equal"] += int(act == int(z["own_action"][j]))
    ok = c["compared"] > 0 and c["row_equal"] == c["compared"] and c["label_equal"] == c["compared"]
    return {"seed": SEED, "battles_in_both": len(both), "battles_sampled": len(chosen), **dict(c), "pass": ok}


def register(rows: Path, labels: Path, selfplay: Path, labels0: Optional[Path] = None,
             require_panel_rule: bool = False) -> Dict[str, Any]:
    man, sha = RS.load_manifest(rows)
    lman = json.loads((labels / "labels_manifest.json").read_text())
    lsha = RS.S_sha(labels / "labels_manifest.json")
    sman, ssha = RS.load_manifest(selfplay)
    problems = []
    if lman["rows_manifest_sha256"] != sha:
        problems.append("the S-P labels were made on other rows")
    if len(lman["shards"]) != len(man["train_shards"]) + 1:
        problems.append("the S-P labels do not cover every shard")
    if sman["like_rows_manifest_sha256"] != sha:
        problems.append("the S-S rows were sized on other rows")
    if (sman["train_rows"], sman["val_rows"]) != (man["train_rows"], man["val_rows"]):
        problems.append("the S-S row counts differ from the rowset's")
    if [s["rows"] for s in sman["train_shards"]] != [s["rows"] for s in man["train_shards"]]:
        problems.append("the S-S shard sizes differ from the rowset's (the step budgets would differ)")
    l0sha = None
    if labels0 is not None:
        l0man = json.loads((labels0 / "labels_manifest.json").read_text())
        l0sha = RS.S_sha(labels0 / "labels_manifest.json")
        if l0man["rows_manifest_sha256"] != sha or len(l0man["shards"]) != len(man["train_shards"]) + 1:
            problems.append("the S-P0 labels were made on other rows, or do not cover every shard")
        if l0man.get("pstar_sha256") != lman["pstar_sha256"]:
            problems.append("the S-P0 and S-P labels name different P* checkpoints")
    if require_panel_rule and not (sman.get("panel_rule") or {}).get("profile_sha256"):
        problems.append("the S-S team draw did not face the human panel rule (REVIEW E B1)")
    return {"rowset": man["rowset"], "rows_manifest_sha256": sha, "train_rows": man["train_rows"],
            "val_rows": man["val_rows"], "train_hidden": man["train_hidden"],
            "labels_manifest_sha256": lsha, "pstar_sha256": lman["pstar_sha256"],
            "selfplay_manifest_sha256": ssha, "selfplay_rows": [sman["train_rows"], sman["val_rows"]],
            "labels0_manifest_sha256": l0sha,
            "selfplay_panel_rule_profile_sha256": (sman.get("panel_rule") or {}).get("profile_sha256"),
            "problems": problems, "pass": not problems}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.checks", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hom-repro")
    h.add_argument("--rows", required=True)
    h.add_argument("--parts", required=True)
    h.add_argument("--hom", required=True)
    h.add_argument("--n", type=int, default=200)
    h.add_argument("--corpus-prefix", default=str(RS.S.default_corpus()) + "/")
    r = sub.add_parser("register")
    r.add_argument("--rows", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--selfplay", required=True)
    r.add_argument("--labels0", default=None, help="the S-P0 labels (Amendment 2)")
    r.add_argument("--require-panel-rule", action="store_true", help="Amendment 2: S-S must face the panel rule")
    for p in (h, r):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "hom-repro":
        res = hom_repro(Path(a.rows), Path(a.parts), Path(a.hom), a.n, a.corpus_prefix)
    else:
        res = register(Path(a.rows), Path(a.labels), Path(a.selfplay), Path(a.labels0) if a.labels0 else None,
                       a.require_panel_rule)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
