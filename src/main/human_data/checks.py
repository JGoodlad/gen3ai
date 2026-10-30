"""X59-S's pre-training checks (PREREG §2.4, §7): the input registration and the extraction reproduction.

    python -m main.human_data.checks hom-repro --rows <rowset>/rows --parts <rowset>/parts \
        --hom ~/gen3ai_archive/x50_r1b/train --n 200 --out <dir>/results/hom_repro.json
    python -m main.human_data.checks reencode --rows <derived rowset>/rows --parts <rowset>/parts \
        --hom ~/gen3ai_archive/x50_r1b/train --n 200 [--within <name>=<sub-rowset rows dir> ...] \
        --out <dir>/results/m3_reencode.json
    python -m main.human_data.checks register --rows <rowset>/rows --labels <rowset>/labels_pstar \
        --labels0 <rowset>/labels_pstar0 --selfplay <dir>/ss_rows --out <dir>/results/registration.json

* ``hom-repro`` (VOID 8): a seeded sample of kept battles that the rung-1b HOM extraction also read; for every
  decision both extractions hold, the completed row must be BYTE-equal and the own-action label equal. The rung-1b
  parts keep a decision when either label is known, so a hidden-move row is compared where that extraction kept it.
* ``reencode`` (PREREG Amendment 5 §A5.7, M3): VOID 8's SAME seeded battles (:func:`vo8_sample`) re-extracted by
  TODAY's rowset path (``rowset._extract_job``: the live reader ``$POKESIM_LIVE_READER_BIN`` names, stamp-checked
  against this tree's Rust sources, and today's encoder), every row compared BYTE for byte with the same row (by rid)
  of a registered rowset: the share of rows differing in the observation, the mask, the consistent set or the label,
  and where in the observation (slices named by ``Gen3ObservationEncoder.get_layout()``). Rows the registered rowset
  dropped are skipped and counted. Descriptive: the share is a registered limit (> 1.0 % routes the NLL twin read).
  ``--within`` repeats the comparison on the compared rows a SUB-rowset of ``--rows`` trains on (its TRAIN rids; the
  ≥ 1500 follow-up's arms, Amendment 7, inherit A5.7 through it).
* ``register``: the sha256 of the rows manifest, the S-P labels manifest and the S-S rows manifest, their row counts
  and the cross-checks the trainer relies on (S-S has the rowset's train / validation counts; the labels were made on
  this rowset; Amendment 2: the S-P0 labels too, and S-S's team draw faced the human panel rule). Written BEFORE any
  student trains; the trainer refuses any other input (``student.check_registration``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from main.human_data import rowset as RS

SEED = 20261025


class Vo8Sample(NamedTuple):
    """VOID 8's seeded battle sample: the kept battles both extractions read, and the ``n`` drawn from them."""
    both: List[str]
    chosen: List[str]
    hom_index: Dict[str, Tuple[str, int]]      # rel -> (HOM part, battle index in it)
    part_of: Dict[str, str]                    # rel -> rowset part


def vo8_sample(parts: Path, hom: Path, n: int, corpus_prefix: str) -> Vo8Sample:
    """The ONE sampler of VOID 8's battles (``hom-repro`` and ``reencode`` both draw it): :data:`SEED`, ``n`` of the
    kept battles (``parts``) the rung-1b HOM extraction (``hom``) also read."""
    part_of: Dict[str, str] = {}
    for js in sorted(parts.glob("part_*.json")):
        for rel in json.loads(js.read_text())["battles"]:
            part_of[rel] = js.stem
    hom_index: Dict[str, Tuple[str, int]] = {}
    for js in sorted(hom.glob("humans_*.json")):
        m = json.loads(js.read_text())
        first = int(js.stem.rsplit("_", 1)[1]) * int(m["part_size"])
        for i, b in enumerate(m["battles"]):
            if b.startswith(corpus_prefix):
                hom_index[b[len(corpus_prefix):]] = (js.stem, first + i)
    both = sorted(r for r in part_of if r in hom_index)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(both), size=min(n, len(both)), replace=False).tolist())
    return Vo8Sample(both, [both[i] for i in pick], hom_index, part_of)


def hom_repro(rows: Path, parts: Path, hom: Path, n: int, corpus_prefix: str) -> Dict[str, Any]:
    """VOID 8 (module docstring)."""
    both, chosen, hom_index, part_of = vo8_sample(parts, hom, n, corpus_prefix)
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


#: the per-row fields ``reencode`` compares, by what they are (the obs bytes are ``rows``)
LABEL_FIELDS = ("act", "hidden", "moveset")
META_FIELDS = ("bkey", "acct", "seat", "k", "turn", "forced", "rating", "quit_dist", "end")
LIMIT_SHARE = 0.01          # PREREG A5.7: more than 1.0 % of rows differing triggers the NLL twin measurement


def obs_slices(layout: Dict[str, Any], context_layout: Optional[Dict[str, Any]] = None) -> Tuple[List[str], np.ndarray]:
    """Names for the observation's columns from ``Gen3ObservationEncoder.get_layout()`` (never hardcoded indices):
    (the slice names, a per-column index into them). Team columns are named by the per-mon block (both teams'
    six slots pooled: ``our_team.hp``), the active context by ``context_layout``'s blocks (the active-context
    encoder's ``get_layout()``; both sides pooled), the reactive part by its sub-blocks (pair history, event window,
    the OBS-FACTS sub-blocks); a column no named block covers takes its part's ``.other``."""
    dim = int(layout["total_dim"])
    names: List[str] = []
    col = np.full(dim, -1, np.int64)

    def put(name: str, start: int, end: int) -> None:
        if name not in names:
            names.append(name)
        col[start:end] = names.index(name)

    parts = layout["parts"]
    for part in ("our_team", "opp_team", "context", "global", "reactive"):
        put(part if part == "context" else f"{part}.other", parts[part]["start"], parts[part]["end"])
    for team in ("our_team", "opp_team"):
        n_mon, mon_dim = parts[team]["reshape"]
        for slot in range(n_mon):
            base = parts[team]["start"] + slot * mon_dim
            for blk, spec in layout["pokemon"].items():
                if isinstance(spec, dict) and "offset" in spec:
                    put(f"{team}.{blk}", base + spec["offset"], base + spec["offset"] + spec["dim"])
    if context_layout:
        n_side, side_dim = parts["context"]["reshape"]
        for side in range(n_side):
            base = parts["context"]["start"] + side * side_dim
            for blk, spec in context_layout.items():
                put(f"context.{blk}", base + spec["offset"], base + spec["offset"] + spec["dim"])
    g0 = parts["global"]["start"]
    for blk, spec in layout["global_layout"].items():
        put(f"global.{blk}", g0 + spec["offset"], g0 + spec["offset"] + spec["dim"])
    r0 = parts["reactive"]["start"]
    for blk, spec in layout["reactive_layout"].items():
        put(f"reactive.{blk}", r0 + spec["offset"], r0 + spec["offset"] + spec["dim"])
    put("pair_history", layout["pair_history_offset"], layout["pair_history_offset"] + layout["pair_history_dim"])
    put("event_window", layout["event_window_offset"], layout["event_window_offset"] + layout["event_window_dim"])
    f0 = layout["obs_facts_offset"]
    put("obs_facts.other", f0, f0 + layout["obs_facts_dim"])
    for blk, spec in layout["obs_facts"].items():
        put(f"obs_facts.{blk}", f0 + spec["offset"], f0 + spec["offset"] + spec["dim"])
    if (col < 0).any():
        raise ValueError(f"{int((col < 0).sum())} observation columns are outside every layout part")
    used = sorted(set(col.tolist()))                    # a part-level name every sub-block overwrote names nothing
    remap = np.full(len(names), -1, np.int64)
    remap[used] = np.arange(len(used))
    return [names[i] for i in used], remap[col]


def compare_rows(ours: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], names: Sequence[str],
                 col: np.ndarray) -> Dict[str, Any]:
    """Row-aligned ``ours`` vs ``ref`` (same rids, same order): how many rows differ in the observation BYTES, the
    mask, the consistent set, the label (act / hidden / moveset) and the row metadata, and per observation slice how
    many rows differ in it."""
    if not np.array_equal(ours["rid"], ref["rid"]):
        raise ValueError("compare_rows needs row-aligned inputs (the rids differ)")
    n = len(ours["rid"])
    a = np.ascontiguousarray(ours["rows"], np.float32).view(np.uint32)
    b = np.ascontiguousarray(ref["rows"], np.float32).view(np.uint32)
    if a.shape != b.shape:
        raise ValueError(f"observation widths differ: {a.shape} vs {b.shape} (not byte-comparable)")
    cell = a != b
    obs = cell.any(axis=1)

    def rows_differ(f: str) -> np.ndarray:
        x, y = np.asarray(ours[f]), np.asarray(ref[f])
        return (x != y).reshape(n, -1).any(axis=1) if n else np.zeros(0, bool)

    mask, cons = rows_differ("mask"), rows_differ("cons")
    label = np.zeros(n, bool)
    per_label = {}
    for f in LABEL_FIELDS:
        d = rows_differ(f)
        per_label[f] = int(d.sum())
        label |= d
    meta = np.zeros(n, bool)
    for f in META_FIELDS:
        meta |= rows_differ(f)
    anyd = obs | mask | cons | label
    per_slice: Dict[str, int] = {}
    cols_differ = cell.any(axis=0)
    for i, name in enumerate(names):
        sel = col == i
        hit = int(cell[:, sel].any(axis=1).sum())
        if hit:
            per_slice[name] = hit
    return {"rows_compared": n, "rows_differing": int(anyd.sum()),
            "share_differing": (int(anyd.sum()) / n) if n else None,
            "rows_differing_obs": int(obs.sum()), "rows_differing_mask": int(mask.sum()),
            "rows_differing_consistent_set": int(cons.sum()), "rows_differing_label": int(label.sum()),
            "rows_differing_label_by_field": per_label, "rows_differing_meta_only": int((meta & ~anyd).sum()),
            "obs_columns_differing": int(cols_differ.sum()),
            "obs_columns": [{"col": int(c), "slice": names[col[c]], "rows": int(cell[:, c].sum())}
                            for c in np.nonzero(cols_differ)[0]],
            "rows_differing_by_obs_slice": dict(sorted(per_slice.items(), key=lambda kv: (-kv[1], kv[0])))}


def _r2_rows(rows: Path, rids: np.ndarray, bkeys: np.ndarray) -> Tuple[Dict[str, np.ndarray], int]:
    """The registered rowset's rows whose rid is in ``rids`` (every shard sha-checked), and how many of ITS rows of
    the sampled battles (``bkeys``) are not in ``rids``."""
    man, _ = RS.load_manifest(rows)
    got: List[Dict[str, np.ndarray]] = []
    missing = 0
    for rec in [*man["train_shards"], man["val"]]:
        z = RS.load_shard(rows, rec, ("rid", "bkey"))
        in_b = np.isin(z["bkey"], bkeys)
        hit = np.isin(z["rid"], rids)
        missing += int((in_b & ~hit).sum())
        idx = np.nonzero(hit)[0]
        if len(idx):
            zf = np.load(Path(rows) / f"{rec['name']}.npz")
            got.append({f: zf[f][idx] for f in RS.FIELDS})
            del zf
    return RS._concat(got), missing


def _within(name: str, sub: Path, ours: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], names: Sequence[str],
            col: np.ndarray) -> Dict[str, Any]:
    """The comparison restricted to the rows the sub-rowset ``sub`` TRAINS on (by rid; every shard sha-checked)."""
    man, sha = RS.load_manifest(sub)
    rids = np.concatenate([RS.load_shard(sub, rec, ("rid",))["rid"] for rec in man["train_shards"]])
    sel = np.isin(ours["rid"], rids)
    c = compare_rows({f: v[sel] for f, v in ours.items()}, {f: v[sel] for f, v in ref.items()}, names, col)
    return {"name": name, "rowset": man.get("rowset"), "rows_manifest_sha256": sha,
            **{k: v for k, v in c.items() if k != "obs_columns"},
            "over_limit": c["rows_compared"] > 0 and c["rows_differing"] * 100 > c["rows_compared"]}


def reencode(rows: Path, parts: Path, hom: Path, n: int, corpus_prefix: str, store: Path, registry: Path,
             corpus: Path, extract=None, within: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    """PREREG A5.7 (M3): VOID 8's battles re-extracted by TODAY's rowset path, each row compared byte for byte with
    the SAME row (by rid) of the registered rowset ``rows``. A row the registered rowset dropped (the derived rules)
    is skipped and counted; a registered row today's extraction no longer makes is counted. ``extract`` is the
    per-battle extractor (``rowset._extract_job``; a test passes a stub)."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.human_data.derive import QUIT_TAIL

    extract = extract or RS._extract_job
    sample = vo8_sample(parts, hom, n, corpus_prefix)
    want = set(sample.chosen)
    rdir = Path(parts).parent
    scan_res = json.loads((rdir / "scan.json").read_text())
    if RS.S_sha(rdir / "keep.jsonl.gz") != scan_res["keep_sha256"]:
        raise RS.RowsetError("keep.jsonl.gz differs from the scan's record")
    snap = scan_res["snapshot"]
    population = RS.S.registry_row(snap, registry)["population"]
    with gzip.open(rdir / "keep.jsonl.gz", "rt") as fh:
        metas = {k["rel"]: k for k in (json.loads(x) for x in fh) if k["rel"] in want}
    if set(metas) != want:
        raise RS.RowsetError(f"{len(want - set(metas))} sampled battles are not in keep.jsonl.gz")
    counts: collections.Counter = collections.Counter()
    got: List[Dict[str, np.ndarray]] = []
    for rel in sample.chosen:
        r = extract((snap, population, str(corpus), metas[rel]))
        counts.update(r["counts"])
        got.extend(r["parts"])
    ours = RS._concat(got)
    if not ours:
        raise RS.RowsetError("today's extraction made no rows for the sample")
    ref, r2_missing = _r2_rows(rows, ours["rid"], np.asarray([RS.key64(r) for r in sample.chosen], np.int64))
    pos = {int(r): i for i, r in enumerate(ref["rid"])} if ref else {}
    if len(pos) != len(ref.get("rid", ())):
        raise RS.RowsetError("the registered rowset repeats a rid")
    in_ref = np.asarray([int(r) in pos for r in ours["rid"]], bool)
    skipped = ~in_ref
    q = ours["quit_dist"].astype(np.int64)
    tail = skipped & (q >= 0) & (q <= QUIT_TAIL)
    order = np.asarray([pos[int(r)] for r in ours["rid"][in_ref]], np.int64)
    enc = Gen3ObservationEncoder(load_mappings())
    names, col = obs_slices(enc.get_layout(), enc.active_context_encoder.get_layout())
    ours_c, ref_c = {f: ours[f][in_ref] for f in RS.FIELDS}, {f: ref[f][order] for f in RS.FIELDS}
    cmp_ = compare_rows(ours_c, ref_c, names, col)
    strata = {k: _within(k, Path(v), ours_c, ref_c, names, col) for k, v in (within or {}).items()}
    man, man_sha = RS.load_manifest(rows)
    today = RS._code_identity()
    share = cmp_["share_differing"]
    return {"schema": "x59s_m3_reencode_v1", "check": "PREREG Amendment 5 A5.7 (M3)", "seed": SEED,
            "battles_in_both": len(sample.both), "battles_sampled": len(sample.chosen),
            "battles_sha256": hashlib.sha256("\n".join(sample.chosen).encode()).hexdigest(),
            "rowset": man["rowset"], "rows_manifest_sha256": man_sha,
            "registered": {k: man[k] for k in ("extraction_commit", "source_sha256", "live_reader_sha256",
                                                "layout_hash")},
            "today": {"commit": today["commit"], "source_sha256": today["source_sha256"],
                      "sources_dirty_vs_commit": today["sources_dirty_vs_commit"],
                      "live_reader_bin": _reader_bin(), "live_reader_sha256": today["live_reader_sha256"],
                      "encoder_identity": today["encoder_identity"], "layout_hash": today["layout_hash"]},
            "extract_counts": dict(sorted(counts.items())),
            "rows_today": int(len(ours["rid"])),
            "rows_skipped_not_in_registered": int(skipped.sum()),
            "rows_skipped_quitter_tail": int(tail.sum()),
            "rows_skipped_other": int((skipped & ~tail).sum()),
            "registered_rows_not_made_today": r2_missing,
            **cmp_,
            "limit_share": LIMIT_SHARE,
            "over_limit": share is not None and cmp_["rows_differing"] * 100 > cmp_["rows_compared"],
            **({"within": strata} if within else {}),
            "pass": cmp_["rows_compared"] > 0}


def _reader_bin() -> str:
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin

    return str(resolve_live_reader_bin())


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
    e = sub.add_parser("reencode", help="PREREG A5.7 (M3): VOID 8's battles re-extracted today vs the registered rows")
    e.add_argument("--rows", required=True, help="the registered rowset's rows dir (hr_spec_v1.r2/rows)")
    e.add_argument("--parts", required=True, help="the parent rowset's parts dir (its scan.json and keep sit beside)")
    e.add_argument("--hom", required=True)
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--corpus-prefix", default=str(RS.S.default_corpus()) + "/")
    e.add_argument("--store", default=None)
    e.add_argument("--registry", default=None)
    e.add_argument("--corpus", default=None)
    e.add_argument("--within", action="append", default=[], metavar="NAME=ROWS",
                   help="also compare on the rows this sub-rowset of --rows trains on (repeatable)")
    r = sub.add_parser("register")
    r.add_argument("--rows", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--selfplay", required=True)
    r.add_argument("--labels0", default=None, help="the S-P0 labels (Amendment 2)")
    r.add_argument("--require-panel-rule", action="store_true", help="Amendment 2: S-S must face the panel rule")
    for p in (h, e, r):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "hom-repro":
        res = hom_repro(Path(a.rows), Path(a.parts), Path(a.hom), a.n, a.corpus_prefix)
    elif a.cmd == "reencode":
        res = reencode(Path(a.rows), Path(a.parts), Path(a.hom), a.n, a.corpus_prefix,
                       Path(a.store) if a.store else RS.S.default_store(),
                       Path(a.registry) if a.registry else RS.S.default_registry(),
                       Path(a.corpus) if a.corpus else RS.S.default_corpus(),
                       within=dict(w.split("=", 1) for w in a.within) or None)
    else:
        res = register(Path(a.rows), Path(a.labels), Path(a.selfplay), Path(a.labels0) if a.labels0 else None,
                       a.require_panel_rule)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
