"""X50 rung 1b — the REGISTERED read R-b1 (``design_x50_rung1b_human_opponent_model.md`` §8.2): does the HOM
out-predict P* on held-out humans? A GATE for R-b2.

    python -m main.x50_r1.read_rb1 --heldout <archive>/heldout --hom <archive>/hom/model/hom_x50r1b.zip \
        --sidecar <archive>/hom/model/hom_sidecar.json --pstar <P* zip> --out <dir>

* Rows: the held-out dates (09-21..23, the extraction at the Mimic fix), completed (mode 2), (b)'s label (the acting
  human's own action), >= 2 legal actions; PRIMARY: the acting human rated >= 1500.
* Both models through the masked 11-way softmax at T = 1 in float64 (``opp_likelihood.log_probs``, the judge's).
* PRIMARY: Δ = NLL(HOM) − NLL(P*), paired per row. LEARNED iff Δ's CI upper < −0.05 nats; rule 8: an upper within
  0.002 of the bar takes the conservative label (NOT LEARNED).
* Registered secondary (decides nothing): Δ on VOLUNTARY switches (a switch at a non-forced decision), on moves, and
  on forced replacements apart. Also: top-1 for both, the all-ratings band, rating bands.

The HOM must be FROZEN and HASHED before this runs: its zip's sha256 must equal the sidecar's (a committed record),
else the read refuses. Bootstrap: battle clusters, 2,000 resamples, seed 50 (R1's). Forwards are cached per part
and per model sha256 under ``<out>/../rb1_cache`` (resumable).
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
from main.x50_r1.hom import sha256_file
from main.x50_r1.read_r0 import _ci, joint_boot

BAR = -0.05
BAND = 0.002
SWITCH_MAX = 5          # action indices 0..5 are switches, 6..9 moves (10: Struggle / Recharge)
BANDS = ((0, 1300), (1300, 1500), (1500, 1600), (1600, 1700), (1700, 10000))


def learned_rule(hi: float) -> str:
    return "LEARNED" if hi < BAR - BAND else "NOT LEARNED"


def forward_nll(model: Any, z: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    from main.policy_spectrum.qhat import logits_and_values
    from main.value_search.opp_likelihood import log_probs

    m = z["mask"].astype(np.uint8)
    out = logits_and_values(model, z["rows"].astype(np.float32), m, batch=1024)
    lp, _ = log_probs(out, m, z["own_action"].astype(np.int64))
    lg = np.where(m.astype(bool), out[:, :11].astype(np.float64), -np.inf)
    return {"nll": -lp, "top1": (lg.argmax(axis=1) == z["own_action"]).astype(np.int8)}


def collect(models: Dict[str, Any], shas: Dict[str, str], d: Path, cache: Path) -> Dict[str, np.ndarray]:
    keep = ("forced", "battle", "rating_viewer", "own_action", "n_legal")
    out: Dict[str, List[np.ndarray]] = {}
    cache.mkdir(parents=True, exist_ok=True)
    for js in R1.load_parts(d):
        z = dict(np.load(js.with_suffix(".npz")))
        if not z:
            continue
        sel = (z["mode"] == 2) & (z["own_action"] >= 0) & (z["n_legal"] >= 2)
        z = {k: v[sel] for k, v in z.items()}
        for k in keep:
            out.setdefault(k, []).append(z[k])
        for name, model in models.items():
            cf = cache / f"{name}_{shas[name][:12]}_{js.stem}.npz"
            if cf.exists():
                fw = dict(np.load(cf))
            else:
                fw = forward_nll(model, z)
                tmp = cf.with_suffix(".tmp.npz")
                np.savez(tmp, **fw)
                tmp.replace(cf)
            for k, v in fw.items():
                out.setdefault(f"{name}_{k}", []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def contrast(A: Dict[str, np.ndarray], sel: np.ndarray) -> Dict[str, Any]:
    """Δ NLL(HOM − P*) on ``sel``, with both NLLs and top-1s, battle-clustered (one joint draw for all four)."""
    b = A["battle"][sel]
    pops = {"d": (A["hom_nll"][sel] - A["pstar_nll"][sel], b), "hom": (A["hom_nll"][sel], b),
            "pstar": (A["pstar_nll"][sel], b), "hom_top1": (A["hom_top1"][sel].astype(float), b),
            "pstar_top1": (A["pstar_top1"][sel].astype(float), b)}
    bt = joint_boot(pops, np.random.default_rng(R1.BOOT_SEED))
    return {"rows": int(sel.sum()), "battles": int(len(np.unique(b))),
            **{k: {"mean": float(np.mean(v[0])), "ci95": _ci(bt[k])} for k, v in pops.items()}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.read_rb1", description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--hom", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--pstar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    t0 = time.time()
    side = json.loads(Path(a.sidecar).read_text())
    shas = {"hom": sha256_file(Path(a.hom)), "pstar": sha256_file(Path(a.pstar))}
    if shas["hom"] != side["hom_sha256"]:
        print("[rb1] the HOM's sha256 differs from its sidecar: not the frozen model; no read", file=sys.stderr)
        return 2
    if shas["pstar"] != side["init_sha256"]:
        print("[rb1] P*'s sha256 differs from the HOM's init: no read", file=sys.stderr)
        return 2
    with inference_globals(a.threads):
        models = {"hom": load_checkpoint(Path(a.hom)), "pstar": load_checkpoint(Path(a.pstar))}
        A = collect(models, shas, Path(a.heldout), Path(a.out).resolve().parent / "rb1_cache")
    prim = A["rating_viewer"] >= R1.PRIMARY_RATING
    sw = A["own_action"] <= SWITCH_MAX
    forced = A["forced"].astype(bool)
    strata = {
        "PRIMARY (>= 1500)": prim,
        "voluntary switches (>= 1500)": prim & sw & ~forced,
        "moves (>= 1500)": prim & ~sw,
        "forced replacements (>= 1500)": prim & forced,
        "all ratings": np.ones_like(prim),
        **{f"rating {lo}-{hi - 1}": (A["rating_viewer"] >= lo) & (A["rating_viewer"] < hi) for lo, hi in BANDS},
    }
    res = {k: contrast(A, s) for k, s in strata.items() if s.any()}
    hi = res["PRIMARY (>= 1500)"]["d"]["ci95"][1]
    verdict = learned_rule(hi)
    out = {"schema": "x50_r1b_rb1_read_v1",
           "registration": "designs/endstate/design_x50_rung1b_human_opponent_model.md §8.2",
           "verdict": verdict, "rule": f"Δ NLL(HOM − P*) CI upper < {BAR} LEARNED (rule 8 band {BAND})",
           "hom": a.hom, "hom_sha256": shas["hom"], "pstar": a.pstar, "pstar_sha256": shas["pstar"],
           "strata": res, "bootstrap": {"n": R1.BOOT_N, "seed": R1.BOOT_SEED, "cluster": "battle"},
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rb1_read.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdict": verdict, "primary": res["PRIMARY (>= 1500)"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
