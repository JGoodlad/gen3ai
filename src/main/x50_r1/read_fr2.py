"""FR-2, the HOM's TRACE TWIN (``design_human_replay_lever.md`` §2.2, registered in
``design_x50_rung1b_human_opponent_model.md`` §8.5): does a model trained on C-GAME rows learn the completion artefact?

    python -m main.x50_r1.read_fr2 --traces <archive>/traces_oppseat --hom <archive>/hom/model/hom_x50r1b.zip \
        --pstar <main>/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --out <dir>

* **Rows:** a trace extraction (``main.x50_r1.run traces --complete --seat opponent``): the COMPLETED rows (mode 2,
  C-GAME's team) and the TRUE-team rows of the same decisions, each labelled with the viewer's own action ((b)'s
  label, ``own_action``). The true rows are ``--true-rows full`` (mode 1: the true team with the seat's REAL
  requests, the registration's) or ``own_true`` (mode 3: the true team with SYNTHESISED requests; descriptive). A
  decision is the key ``(battle, seat, k)`` (``k`` the decision's index in the seat's read; every mode reads the same
  decision points). It is read iff BOTH rows exist, both carry a label and both have >= 2 legal actions; every other
  key is counted under its drop reason, and a duplicated key is a defect.
* **The exclusion** (X59-S PREREG Amendment 4c): ``--exclude <list.json>`` (``main.x50_r1.fr2_exclude``) drops every
  row of the battles it names, both modes, whichever seat the extraction holds, BEFORE the pairing; ``fr2.json``
  records the list's path, sha256 and count and the battles / rows it dropped, and the forward cache's tag carries
  the list's sha (a read with another list never reuses a chunk).
* **The viewer:** the registration reads the OPPONENT's seat (the sentinel or bot as viewer and actor). Every read
  row's viewer is checked against its trace summary's ``trainee_side`` (and the part's ``seat`` stamp, when it has
  one); ``--viewer`` declares the seat the extraction holds and any row of the other seat refuses the read. Only
  ``--viewer opponent --true-rows full`` is the registered FR-2; anything else writes the same numbers with
  ``registered: false`` and no ``class``.
* **NLL:** both models through the masked 11-way policy softmax in float64 with the network in eval mode
  (:func:`main.x50_r1.read_rb1.forward_nll`: ``opp_likelihood.log_probs``, the function the HOM's own validation reads).
* **Statistic:** twin gap = mean NLL(completed) − mean NLL(true) per model, decision-weighted over the paired
  decisions; D = gap(HOM) − gap(P*). Interval: battle clusters, 2,000 resamples, seed 50 (R0's), ONE draw of battles
  per resample shared by both models and both modes (:func:`main.x50_r1.read_r0.joint_boot`); percentile 95 %.
* **Class** (:data:`main.human_data.x59s_read.FR2_CLASSES`): ARTEFACT iff LB > 0.05 + 1e-9; NO-ARTEFACT iff
  UB < 0.05 − 1e-9; else UNCLEAR (a bound within 1e-9 of the bar takes UNCLEAR: root rule 8).

Refuses (exit 2) unless the HOM zip's sha256 equals its sidecar's and P*'s equals :data:`PSTAR_SHA256` and the
sidecar's init. Forwards are cached per (model sha, part, chunk of :data:`CHUNK` rows) under ``--cache`` (default
``<traces>/../fr2_cache``, never the repo): a killed read resumes, and a chunk whose cached keys differ refuses.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from main.human_data.x59s_read import FR2_CLASSES
from main.x50_r1 import fr2_exclude
from main.x50_r1 import read as R1
from main.x50_r1.hom import sha256_file
from main.x50_r1.read_r0 import MODES, _ci, joint_boot

BAR = 0.05                      # nats
TIE = 1e-9                      # root rule 8
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"   # X26 at 280,000,170 steps
MODE_C = MODES["completed"]
#: the true rows: the true team with the seat's REAL requests (registered) or with synthesised ones (descriptive)
TRUE_ROWS = {"full": MODES["full"], "own_true": MODES["own_true"]}
REGISTERED_TRUE_ROWS = "full"
#: X59-S PREREG Amendment 4c: the registered battle exclusion (d8d674b99's silent volatile ends), by its sha256
REGISTERED_EXCLUDE_SHA256 = "1c6f1ef8a14f27d08dfbbdc3df355f8d12d5114d48f31a11d1fc721e45dd79f7"
CHUNK = 4096
VIEWERS = ("opponent", "trainee")
REGISTERED_VIEWER = "opponent"
MODEL_NAMES = ("hom", "pstar")
THREADS = 2                     # 6 threads ran 60x slower on the shared agent cores

#: (rows [n, d] float32, mask [n, 11], action [n]) -> per-row NLL [n] float64
Forward = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class Fr2Error(RuntimeError):
    """An input that is not what the registration names (a sha, a seat, a pairing, a cache): no read."""


def is_registered(viewer: str, true_rows: str, exclusion: Optional[Dict[str, Any]]) -> bool:
    """The registered FR-2: the opponent's seat, real-request true rows and Amendment 4c's exclusion list (by sha256)."""
    return (viewer == REGISTERED_VIEWER and true_rows == REGISTERED_TRUE_ROWS and exclusion is not None
            and exclusion.get("sha256") == REGISTERED_EXCLUDE_SHA256)


def fr2_class(lo: float, hi: float) -> str:
    """The registered class of D's 95 % CI ``[lo, hi]`` against +0.05 nats, rule 8's 1e-9 band taking UNCLEAR."""
    if lo > BAR + TIE:
        return FR2_CLASSES[0]
    if hi < BAR - TIE:
        return FR2_CLASSES[1]
    return FR2_CLASSES[2]


def verify_inputs(hom: Path, sidecar: Path, pstar: Path) -> Dict[str, str]:
    """The HOM is the frozen one its sidecar names, and P* is the registered checkpoint and the HOM's init."""
    side = json.loads(sidecar.read_text())
    shas = {"hom": sha256_file(hom), "pstar": sha256_file(pstar), "sidecar": sha256_file(sidecar)}
    if shas["hom"] != side.get("hom_sha256"):
        raise Fr2Error(f"{hom}: sha256 {shas['hom']} is not the sidecar's {side.get('hom_sha256')} (not the frozen HOM)")
    if shas["pstar"] != PSTAR_SHA256:
        raise Fr2Error(f"{pstar}: sha256 {shas['pstar']} is not P*'s {PSTAR_SHA256}")
    if side.get("init_sha256") != PSTAR_SHA256:
        raise Fr2Error(f"{sidecar}: the HOM's init {side.get('init_sha256')} is not P* {PSTAR_SHA256}")
    return shas


def part_battles(js: Path, meta: Dict[str, Any]) -> Dict[int, Path]:
    """A trace part's global battle ids (the part's index x its part size + the place in the part) -> summary path."""
    first = int(js.stem.rsplit("_", 1)[1]) * int(meta["part_size"])
    return {first + i: Path(p) for i, p in enumerate(meta["battles"])}


def trainee_side(summary_path: Path) -> int:
    """The trainee's seat of one trace (0 = p1, 1 = p2: the extraction's ``seat`` code), from the stored meta."""
    from main.prober.core_trace import load_summary_meta

    side = (load_summary_meta(str(summary_path)).get("trace_source") or {}).get("trainee_side")
    if side not in ("p1", "p2"):
        raise Fr2Error(f"{summary_path}: trainee_side {side!r}")
    return 0 if side == "p1" else 1


def viewers(z: Dict[str, np.ndarray], sides: Dict[int, int]) -> np.ndarray:
    """Per row, its viewer: ``trainee`` iff the row's seat is the trace's trainee side, else ``opponent``."""
    ts = np.array([sides[int(b)] for b in z["battle"]], np.int8)
    return np.where(z["seat"] == ts, "trainee", "opponent")


def pair_part(z: Dict[str, np.ndarray], true_mode: int) -> Tuple[np.ndarray, np.ndarray, collections.Counter]:
    """Indices ``(ic, it)`` into ``z`` of the paired decisions (aligned: ``ic[j]`` and ``it[j]`` are one decision's
    completed (mode 2) and true (``true_mode``) rows, in key order) and every other key's drop reason."""
    if true_mode == MODE_C:
        raise Fr2Error("the true rows cannot be the completed rows")
    MODE_T = true_mode
    rows: Dict[int, Dict[Tuple[int, int, int], int]] = {MODE_C: {}, MODE_T: {}}
    for m, d in rows.items():
        for i in np.nonzero(z["mode"] == m)[0]:
            key = (int(z["battle"][i]), int(z["seat"][i]), int(z["k"][i]))
            if key in d:
                raise Fr2Error(f"mode {m}: decision {key} appears twice (no 1:1 pairing)")
            d[key] = int(i)
    drops: collections.Counter = collections.Counter()
    ic: List[int] = []
    it: List[int] = []
    oa, nl = z["own_action"], z["n_legal"]
    for key in sorted(set(rows[MODE_C]) | set(rows[MODE_T])):
        c, t = rows[MODE_C].get(key), rows[MODE_T].get(key)
        if c is None or t is None:
            drops["only_true" if c is None else "only_completed"] += 1
        elif oa[c] < 0 or oa[t] < 0:
            drops["no_own_action_label"] += 1
        elif nl[c] < 2 or nl[t] < 2:
            drops["fewer_than_2_legal"] += 1
        else:
            ic.append(c)
            it.append(t)
    return np.asarray(ic, np.int64), np.asarray(it, np.int64), drops


def cached_nll(fwd: Forward, tag: str, part: str, z: Dict[str, np.ndarray], idx: np.ndarray,
               cache: Path) -> np.ndarray:
    """``fwd``'s NLL of rows ``idx``, chunk by chunk; a chunk already in ``cache`` with the same keys is reused."""
    out: List[np.ndarray] = []
    for c0 in range(0, len(idx), CHUNK):
        ii = idx[c0:c0 + CHUNK]
        keys = np.stack([z[f][ii].astype(np.int64) for f in ("battle", "seat", "k", "mode")], 1)
        cf = cache / f"{tag}_{part}_{c0 // CHUNK:04d}.npz"
        if cf.exists():
            c = np.load(cf)
            if not np.array_equal(c["keys"], keys):
                raise Fr2Error(f"{cf}: the cached rows are not this read's (a stale cache): remove it")
            out.append(c["nll"])
            continue
        nll = np.asarray(fwd(z["rows"][ii], z["mask"][ii], z["own_action"][ii]), np.float64)
        tmp = cf.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, nll=nll)
        tmp.replace(cf)
        out.append(nll)
    return np.concatenate(out) if out else np.zeros(0, np.float64)


def twin(nll: Dict[str, np.ndarray], battle: np.ndarray, seed: int = R1.BOOT_SEED) -> Dict[str, Any]:
    """The gaps and D from per-decision NLLs ``nll['<model>_c' | '<model>_t']`` (aligned with ``battle``), every
    resample ONE draw of battles shared by both models and both modes."""
    pops = {k: (v.astype(np.float64), battle) for k, v in nll.items()}
    bt = joint_boot(pops, np.random.default_rng(seed))
    mean = {k: float(v.mean()) for k, v in nll.items()}
    gap_b = {m: bt[f"{m}_c"] - bt[f"{m}_t"] for m in MODEL_NAMES}
    d_b = gap_b["hom"] - gap_b["pstar"]
    gaps = {m: mean[f"{m}_c"] - mean[f"{m}_t"] for m in MODEL_NAMES}
    return {"nll": {m: {"completed": mean[f"{m}_c"], "true": mean[f"{m}_t"]} for m in MODEL_NAMES},
            "gap": {m: {"mean": gaps[m], "ci95": _ci(gap_b[m])} for m in MODEL_NAMES},
            "D": {"mean": gaps["hom"] - gaps["pstar"], "ci95": _ci(d_b)}}


def read(traces: Path, forwards: Dict[str, Forward], tags: Dict[str, str], cache: Path,
         viewer: str, true_rows: str = REGISTERED_TRUE_ROWS,
         exclude: Optional[FrozenSet[str]] = None) -> Dict[str, Any]:
    """Every part of the trace extraction -> the paired NLLs, the counts and the twin statistics. ``exclude``: the
    :func:`main.x50_r1.fr2_exclude.norm` keys of battles whose rows are dropped before the pairing."""
    if viewer not in VIEWERS:
        raise Fr2Error(f"--viewer {viewer!r}: one of {VIEWERS}")
    if true_rows not in TRUE_ROWS:
        raise Fr2Error(f"--true-rows {true_rows!r}: one of {tuple(TRUE_ROWS)}")
    MODE_T = TRUE_ROWS[true_rows]
    cache.mkdir(parents=True, exist_ok=True)
    acc: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    drops: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    excluded: collections.Counter = collections.Counter()
    parts: List[Dict[str, Any]] = []
    for js in R1.load_parts(traces):
        meta = json.loads(js.read_text())
        npz = js.with_suffix(".npz")
        parts.append({"part": js.stem, "json_sha256": sha256_file(js), "npz_sha256": sha256_file(npz),
                      "commit": meta.get("commit"), "complete": meta.get("complete")})
        if not meta.get("complete"):
            raise Fr2Error(f"{js}: not a --complete extraction (no mode 2 / 3 rows)")
        if meta.get("seat", viewer) != viewer:
            raise Fr2Error(f"{js}: the part is stamped seat {meta.get('seat')!r}, not the {viewer}'s")
        z = dict(np.load(npz))
        if not z:
            continue
        sel = np.isin(z["mode"], (MODE_C, MODE_T))
        z = {k: v[sel] for k, v in z.items()}
        paths = part_battles(js, meta)
        if exclude:
            gone = {b for b in np.unique(z["battle"]).tolist() if fr2_exclude.norm(str(paths[b])) in exclude}
            drop = np.isin(z["battle"], list(gone))
            excluded["battles"] += len(gone)
            excluded["rows"] += int(drop.sum())
            z = {k: v[~drop] for k, v in z.items()}
            if not len(z["battle"]):
                continue
        vw = viewers(z, {b: trainee_side(paths[b]) for b in np.unique(z["battle"]).tolist()})
        for v, n in zip(*np.unique(vw, return_counts=True)):
            seen[str(v)] += int(n)
        if (vw != viewer).any():
            raise Fr2Error(f"{js}: {int((vw != viewer).sum())} of {len(vw)} read rows are not the "
                           f"{viewer}'s seat (viewers {dict(seen)}); the extraction holds another seat")
        ic, it, dr = pair_part(z, MODE_T)
        drops.update(dr)
        idx = np.concatenate([ic, it])
        for name, fwd in forwards.items():
            nll = cached_nll(fwd, tags[name], js.stem, z, idx, cache)
            acc[f"{name}_c"].append(nll[:len(ic)])
            acc[f"{name}_t"].append(nll[len(ic):])
        acc["battle"].append(z["battle"][ic].astype(np.int64))
        acc["opp"].append(np.array([paths[int(b)].parent.name for b in z["battle"][ic]], dtype=object))
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    if not len(arr.get("battle", ())):
        raise Fr2Error(f"{traces}: no paired decision")
    opp = arr.pop("opp").astype(str)
    battle = arr.pop("battle")
    sent = np.char.startswith(opp, "sentinel_")
    counts = {"parts": len(parts), "battles": int(len(np.unique(battle))), "decisions_paired": int(len(battle)),
              "decisions_paired_vs_sentinel": int(sent.sum()),
              "decisions_paired_vs_bot": int(np.isin(opp, R1.BOTS).sum()),
              "dropped": dict(sorted(drops.items())), "excluded": dict(sorted(excluded.items())), "read_rows_by_viewer": dict(seen),
              "true_rows": f"{true_rows} (mode {MODE_T})"}
    return {"counts": counts, "parts": parts, **twin(arr, battle)}


def _model_forward(path: Path) -> Forward:
    """A lazy forward: the checkpoint loads on its first uncached chunk (a resumed read loads nothing it skips)."""
    holder: Dict[str, Any] = {}

    def fwd(rows: np.ndarray, mask: np.ndarray, act: np.ndarray) -> np.ndarray:
        from main.policy_spectrum.reader import load_checkpoint
        from main.x50_r1.read_rb1 import forward_nll

        if "m" not in holder:
            holder["m"] = load_checkpoint(path)
        return forward_nll(holder["m"], {"rows": rows, "mask": mask, "own_action": act})["nll"]

    return fwd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.read_fr2", description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True)
    ap.add_argument("--hom", required=True)
    ap.add_argument("--sidecar", default=None, help="default: hom_sidecar.json beside --hom")
    ap.add_argument("--pstar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", choices=VIEWERS, default=REGISTERED_VIEWER,
                    help="the seat the extraction holds; only 'opponent' is the registered FR-2")
    ap.add_argument("--true-rows", choices=tuple(TRUE_ROWS), default=REGISTERED_TRUE_ROWS,
                    help="full: the true team with REAL requests (mode 1, registered); own_true: synthesised (mode 3)")
    ap.add_argument("--cache", default=None, help="default: <traces>/../fr2_cache")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--exclude", default=None,
                    help="a main.x50_r1.fr2_exclude list (Amendment 4c): its battles' rows drop from both modes")
    a = ap.parse_args(argv)
    t0 = time.time()
    hom, pstar = Path(a.hom), Path(a.pstar)
    sidecar = Path(a.sidecar) if a.sidecar else hom.parent / "hom_sidecar.json"
    traces = Path(a.traces)
    cache = Path(a.cache) if a.cache else traces.resolve().parent / "fr2_cache"
    exclusion: Optional[Dict[str, Any]] = None
    keys: Optional[FrozenSet[str]] = None
    try:
        if a.exclude:
            try:
                keys, xsha = fr2_exclude.load(Path(a.exclude))
            except (OSError, ValueError) as exc:
                raise Fr2Error(f"--exclude {a.exclude}: {exc}") from exc
            exclusion = {"path": a.exclude, "sha256": xsha, "count": len(keys)}
        xtag = f"_x{exclusion['sha256'][:12]}" if exclusion else ""
        shas = verify_inputs(hom, sidecar, pstar)
        from main.policy_spectrum.reader import inference_globals

        with inference_globals(a.threads):
            res = read(traces, {"hom": _model_forward(hom), "pstar": _model_forward(pstar)},
                       {m: f"{m}_{shas[m][:12]}_{a.true_rows}{xtag}" for m in MODEL_NAMES}, cache, a.viewer,
                       a.true_rows, keys)
    except Fr2Error as exc:
        print(f"[fr2] REFUSED: {exc}", file=sys.stderr)
        return 2
    from utils.git import get_git_hash

    registered = is_registered(a.viewer, a.true_rows, exclusion)
    cls = fr2_class(*res["D"]["ci95"])
    out = {"schema": "x59s_fr2_trace_twin_v1",
           "registration": "designs/endstate/design_human_replay_lever.md §2.2; "
                           "designs/endstate/design_x50_rung1b_human_opponent_model.md §8.5",
           "viewer": a.viewer, "true_rows": a.true_rows, "registered": registered,
           "exclude": None if exclusion is None else {**exclusion, "dropped": res["counts"]["excluded"]},
           ("class" if registered else "class_unregistered"): cls,
           "rule": f"D = gap(HOM) - gap(P*), gap = mean NLL(completed) - mean NLL(true); ARTEFACT iff D's CI lower > "
                   f"{BAR} + {TIE}; NO-ARTEFACT iff its upper < {BAR} - {TIE}; else UNCLEAR",
           **res,
           "bootstrap": {"n": R1.BOOT_N, "seed": R1.BOOT_SEED, "cluster": "battle",
                         "resampling": "joint: one battle draw for both models and both modes", "ci": "percentile 95 %"},
           "inputs": {"traces": str(traces), "hom": str(hom), "hom_sha256": shas["hom"], "sidecar": str(sidecar),
                      "sidecar_sha256": shas["sidecar"], "pstar": str(pstar), "pstar_sha256": shas["pstar"]},
           "commit": get_git_hash(), "threads": a.threads, "seconds": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "fr2.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"viewer": a.viewer, "registered": registered, "class": cls, "D": res["D"],
                      "counts": res["counts"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
