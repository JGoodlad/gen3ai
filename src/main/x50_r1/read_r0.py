"""X50 rung 1b — the REGISTERED read R0 (``design_x50_rung1b_human_opponent_model.md`` §8.1): is F-R1-1 fixed?

    python -m main.x50_r1.read_r0 --humans <archive>/humans --traces <archive>/traces \
        --checkpoint <main>/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip --out <dir>

The extraction (``main.x50_r1.run --complete``) holds, per seat, R1's spectator rows (mode 0), the trainee's
training rows (mode 1, traces), the C-GAME completed rows (mode 2) and the OWN-TRUE rows (mode 3, traces: the
trainee's true team with synthesised decisions). X26's head reads each row once (R1's :func:`read.forward_part`),
and the per-part numbers are CACHED under ``<out>/cache`` (a chunked, resumable read).

* PRIMARY: P_c = NLL(S_c) − NLL(S-full). FIXED iff P_c's CI upper < +0.08 nats; REDUCED iff not FIXED and the
  paired reduction NLL(S) − NLL(S_c) has CI lower > +0.05; NOT-FIXED otherwise. Rule 8: a CI end within 0.002
  nats of a bar takes the more conservative label.
* Re-read beside it with R1's rule (±0.05): Δ NLL(H_c − S_c), and its move and switch strata.
* Descriptive: S-own-true, splitting the rest of the penalty into completion error (S_c − S-own-true) and the
  synthesised decision (S-own-true − S-full).

Bootstrap: battle clusters, 2,000 resamples, seed 50. Contrasts within the traces (same battles) resample the
battles JOINTLY; H against S resamples each population on its own (R1's).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main.x50_r1 import read as R1

FIX_BAR = 0.08         # nats
REDUCE_BAR = 0.05
BAND = 0.002           # rule 8
MODES = {"spectator": 0, "full": 1, "completed": 2, "own_true": 3}


def fixed_rule(pc_hi: float, red_lo: float) -> str:
    """The registered R0 rule (§8.1), with rule 8's band taking the conservative side of each bar."""
    if pc_hi < FIX_BAR - BAND:
        return "FIXED"
    if red_lo > REDUCE_BAR + BAND:
        return "REDUCED"
    return "NOT-FIXED"


def gap_rule(lo: float, hi: float) -> str:
    """R1's rule on Δ NLL(H − S) (±0.05), with rule 8's band (a CI end within 0.002 of ±0.05: INCONCLUSIVE)."""
    m = R1.MARGIN
    if lo > m + BAND:
        return "GAP"
    if lo > -m + BAND and hi < m - BAND:
        return "NO GAP"
    return "INCONCLUSIVE"


def collect(model: Any, d: Path, kind: str, cache: Path) -> Dict[str, np.ndarray]:
    """Every part of one extraction directory → per-row arrays (rows with an opponent-action label only; humans:
    the labelled player rated ≥ 1500, R0's band), with the per-part forward numbers cached
    (``cache/<kind>_<part>.npz``)."""
    keep = ("label", "hp", "forced", "placeholder", "turn", "k", "battle", "seat", "mode", "rating_labelled",
            "rating_viewer",
            "own_action", "n_legal")
    out: Dict[str, List[np.ndarray]] = {}
    opp_of: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    cache.mkdir(parents=True, exist_ok=True)
    for js in R1.load_parts(d):
        meta = json.loads(js.read_text())
        for k, v in meta["counts"].items():
            counts[k] = counts.get(k, 0) + int(v)
        if kind == "traces":
            first = int(js.stem.rsplit("_", 1)[1]) * int(meta["part_size"])
            for i, p in enumerate(meta["battles"]):
                opp_of[first + i] = Path(p).parent.name
        npz = dict(np.load(js.with_suffix(".npz")))
        if not npz:
            continue
        sel = npz["label"][:, 0] != 2
        if kind == "humans":
            sel &= npz["rating_labelled"] >= R1.PRIMARY_RATING     # R0 reads the primary band only (§8.1)
        npz = {k: v[sel] for k, v in npz.items()}
        cf = cache / f"{kind}_{js.stem}.npz"
        if cf.exists():
            fw = dict(np.load(cf))
        else:
            fw = R1.forward_part(model, npz)
            tmp = cf.with_suffix(".tmp.npz")
            np.savez(tmp, **fw)
            tmp.replace(cf)
        for k in keep:
            out.setdefault(k, []).append(npz[k])
        for k, v in fw.items():
            out.setdefault(k, []).append(v)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    if kind == "traces":
        arr["opp"] = np.array([opp_of[int(b)] for b in arr["battle"]])
    arr["_counts"] = counts  # type: ignore[assignment]
    return arr


def joint_boot(pops: Dict[str, tuple], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """BOOT_N JOINT battle-cluster bootstrap means of each population ``name -> (values, battle ids)``: one draw of
    battles (from the union) per resample, every population's mean over its rows in the drawn battles."""
    uniq = np.unique(np.concatenate([b for _, b in pops.values()]))
    idx = {n: np.searchsorted(uniq, b) for n, (_, b) in pops.items()}
    sums = {n: np.bincount(idx[n], weights=x, minlength=len(uniq)) for n, (x, _) in pops.items()}
    cnts = {n: np.bincount(idx[n], minlength=len(uniq)).astype(float) for n in pops}
    out = {n: np.empty(R1.BOOT_N) for n in pops}
    for b in range(R1.BOOT_N):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        for n in pops:
            den = (cnts[n] * w).sum()
            out[n][b] = (sums[n] * w).sum() / den if den > 0 else np.nan   # (a draw holding none of its battles)
    return out


def _ci(x: np.ndarray) -> List[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.read_r0", description=__doc__.split("\n\n")[0])
    ap.add_argument("--humans", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="default: <humans>/../read_cache")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    t0 = time.time()
    cache = Path(a.cache) if a.cache else Path(a.humans).resolve().parent / "read_cache"
    with inference_globals(a.threads):
        model = load_checkpoint(Path(a.checkpoint))
        H = collect(model, Path(a.humans), "humans", cache)
        T = collect(model, Path(a.traces), "traces", cache)
    rng = np.random.default_rng(R1.BOOT_SEED)
    sup_h = (~H["hp"]) & (H["tgt"] >= 0)
    sup_t = (~T["hp"]) & (T["tgt"] >= 0)
    sent = np.char.startswith(T["opp"].astype(str), "sentinel_")
    bot = np.isin(T["opp"], R1.BOTS)
    prim = H["rating_labelled"] >= R1.PRIMARY_RATING
    sel: Dict[str, tuple] = {}
    for name, mode in MODES.items():
        hm = H["mode"] == mode
        tm = T["mode"] == mode
        tag = {"spectator": "", "full": "_full", "completed": "_c", "own_true": "_own_true"}[name]
        if hm.any():
            sel[f"H{tag}"] = (H, sup_h & hm & prim)
            for kname, kv in (("move", 0), ("switch", 1)):
                sel[f"H{tag}_{kname}"] = (H, sup_h & hm & prim & (H["label"][:, 0] == kv))
        sel[f"S{tag}"] = (T, sup_t & tm & sent)
        sel[f"B{tag}"] = (T, sup_t & tm & bot)
        for kname, kv in (("move", 0), ("switch", 1)):
            sel[f"S{tag}_{kname}"] = (T, sup_t & tm & sent & (T["label"][:, 0] == kv))
    pops = {k: R1.summarize(arr, m, rng) for k, (arr, m) in sel.items()}

    # JOINT bootstrap within the traces (the same battles), per contrast family
    def joint(names: List[str]) -> Dict[str, np.ndarray]:
        return joint_boot({n: (sel[n][0]["nll"][sel[n][1]].astype(float), sel[n][0]["battle"][sel[n][1]])
                           for n in names}, np.random.default_rng(R1.BOOT_SEED))

    js = joint(["S", "S_full", "S_c", "S_own_true"])
    jb = joint(["B", "B_full", "B_c", "B_own_true"])

    def jd(boot: Dict[str, np.ndarray], x: str, y: str) -> Dict[str, Any]:
        return {"mean": pops[x]["nll"]["mean"] - pops[y]["nll"]["mean"], "ci95": _ci(boot[x] - boot[y]),
                "bootstrap": "joint (battle clusters)"}

    def ind(x: str, y: str) -> Dict[str, Any]:
        d = pops[x]["nll"]["_boot"] - pops[y]["nll"]["_boot"]
        return {"mean": pops[x]["nll"]["mean"] - pops[y]["nll"]["mean"], "ci95": _ci(d),
                "bootstrap": "independent (battle clusters, R1's)"}

    deltas = {
        "P_c = S_c - S_full": jd(js, "S_c", "S_full"),
        "reduction = S - S_c": jd(js, "S", "S_c"),
        "R1 penalty S - S_full (re-extracted)": jd(js, "S", "S_full"),
        "completion error S_c - S_own_true": jd(js, "S_c", "S_own_true"),
        "synthesised decision S_own_true - S_full": jd(js, "S_own_true", "S_full"),
        "B_c - B_full": jd(jb, "B_c", "B_full"),
        "B - B_c": jd(jb, "B", "B_c"),
        "B_c - B_own_true": jd(jb, "B_c", "B_own_true"),
        "B_own_true - B_full": jd(jb, "B_own_true", "B_full"),
        "H_c - S_c": ind("H_c", "S_c"),
        "H_c - S_c move": ind("H_c_move", "S_c_move"),
        "H_c - S_c switch": ind("H_c_switch", "S_c_switch"),
        "H - S (R1, re-extracted)": ind("H", "S"),
        "H_c - B_c": ind("H_c", "B_c"),
    }
    pc, red = deltas["P_c = S_c - S_full"]["ci95"], deltas["reduction = S - S_c"]["ci95"]
    verdict = fixed_rule(pc[1], red[0])
    gap = gap_rule(*deltas["H_c - S_c"]["ci95"])
    # (b)'s labels, DESCRIPTIVE ONLY (R-b1 reads them; nothing here does)
    own = {}
    for nm, arr in (("humans", H), ("traces", T)):
        m = arr["mode"] == 2
        own[nm] = {"completed_rows_with_a_label": int(m.sum()),
                   "own_action_labelled": int((m & (arr["own_action"] >= 0)).sum())}
    out = {"schema": "x50_r1b_r0_read_v1",
           "registration": "designs/endstate/design_x50_rung1b_human_opponent_model.md §8.1",
           "checkpoint": a.checkpoint, "verdict": verdict,
           "rule": f"P_c CI upper < +{FIX_BAR} FIXED; else reduction CI lower > +{REDUCE_BAR} REDUCED; else "
                   f"NOT-FIXED (rule 8 band {BAND})",
           "r1_gap_reread": {"verdict": gap, "rule": f"Δ NLL(H_c − S_c) CI vs ±{R1.MARGIN} (rule 8 band {BAND})"},
           "deltas": deltas, "populations": R1.strip(pops),
           "extraction_counts": {"humans": H["_counts"], "traces": T["_counts"]},
           "own_action_rows_in_read_parts": own,
           "bootstrap": {"n": R1.BOOT_N, "seed": R1.BOOT_SEED, "cluster": "battle"},
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "r0_read.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdict": verdict, "gap": gap,
                      "deltas": {k: v for k, v in deltas.items()}}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
