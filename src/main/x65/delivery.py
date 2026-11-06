r"""B4's DELIVERY METER (PREREG §5): did each student learn S′'s choices on the panel it never trained on? Read on the
banked decisions alone (zero Kakuna CPU), before any game; the read rule ``main.reads.rules_distill_play`` reads its
result file and classes NOT-DELIVERED when it says so.

    python -m main.x65.delivery score --bank <bank> --bank-sha256 <registered> --fold <read panel> \
        --export <dir>/s_h/model/export.json --init <P* zip> --out <dir>/delivery/<read panel>.npz
    python -m main.x65.delivery gate --scores <dir>/delivery --bank-sha256 <registered> \
        --out ~/gen3ai_archive/x65_exit0_2026-11-05/delivery.json      # the path the read specs name

``score``: on EVERY label row of the read panel (the searched arm's verified searched decisions; inner validation does
not apply to a read panel), a(student) = 1[argmax of the exported student's masked logits = ``played``] and
a(P\*) = 1[``pi_argmax`` = ``played``] (the recorded greedy choice); plus, reported only, the per-decision leak
counters (``attribution_miss`` >= 1, ``rewritten_moves`` >= 1, neither; the leak block's ``residual`` is the preset's
static list, the same on every record, so it splits nothing),
whether the decision was a DEPARTURE (``played`` ≠ ``pi_argmax``), the student's and P\*'s V (the drift), and on the read
panel's searched-arm decisions before turn 10 the student's agreement with P\*'s recorded choice. The student is read
from the EXPORTED zip (its sha256 checked against ``export.json``), through this checkout's loader and the live
searcher's forward (``qhat.logits_and_values``).

``gate``: Δagree_X = mean over panel X's label rows of a(student) − a(P\*); the pooled Δ is the mean of the three
panels' Δagree; its interval bootstraps DEALS (one game a deal) with replacement within each panel, B = 4,000,
``default_rng(20261105)``, panels drawn in :data:`bank.U_PANELS` order each replicate; LB = the 2.5 % percentile
(``numpy.percentile``, linear). DELIVERED iff LB > 0, and a LB within :data:`RULE8_EPS` of 0 is never a crossing
(root rule 8: not delivered, flagged ``boundary``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from main.x65 import bank as BK
from main.x65 import student as ST

SCHEMA = "gen3_x65_delivery_v1"
SCORE_SCHEMA = "gen3_x65_delivery_scores_v1"
B_BOOT = 4000
SEED = 20261105
RULE8_EPS = 0.0005
FIELDS = ("obs", "mask", "label", "played", "pi_argmax", "deal", "turn", "arm", "leak_attr", "leak_rewr")


class DeliveryError(RuntimeError):
    pass


def _forward(zp: Path, obs: np.ndarray, mask: np.ndarray, threads: int = 2) -> np.ndarray:
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    with inference_globals(threads):
        return logits_and_values(load_checkpoint(zp), obs, mask.astype(np.uint8), batch=1024)


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(mask.astype(bool), logits[:, : BK.N_ACT].astype(np.float64), -np.inf), axis=1)


def score(bank: Path, bank_sha256: str, fold: str, export: Path, init: Path, out: Path) -> dict[str, Any]:
    ex = json.loads(export.read_text())
    zp = Path(ex["zip"])
    if BK.sha256_file(zp) != ex["zip_sha256"] or not ex.get("bit_identical"):
        raise DeliveryError(f"{zp}: not the bit-identical export {ex['zip_sha256'][:12]}")
    if ex["student"] != ST.FOLDS[fold][0]:
        raise DeliveryError(f"{ex['student']} is not the student read on {fold}")
    if BK.sha256_file(init) != ST.PSTAR_SHA256:
        raise DeliveryError(f"{init}: not P*'s bytes")
    bk = BK.check(bank, bank_sha256)
    z = BK.load_rows(bank, [fold], FIELDS, labels=[BK.LABELS["searched"]])
    lab = z["label"]
    pre = (z["turn"] < BK.FROM_TURN)
    keep = lab | pre
    obs, mask = z["obs"][keep], z["mask"][keep]
    s_out = _forward(zp, obs, mask)
    p_out = _forward(init, obs, mask)
    s_arg = masked_argmax(s_out, mask)
    lab_k, pre_k = lab[keep], pre[keep]
    played, pia = z["played"][keep].astype(int), z["pi_argmax"][keep].astype(int)
    doc = {"schema": SCORE_SCHEMA, "fold": fold, "student": ex["student"], "zip_sha256": ex["zip_sha256"],
           "bank_manifest_sha256": bk["manifest_sha256"],
           "deal": z["deal"][keep][lab_k], "a_student": (s_arg[lab_k] == played[lab_k]).astype(np.int8),
           "a_pstar": (pia[lab_k] == played[lab_k]).astype(np.int8),
           "departure": (pia[lab_k] != played[lab_k]),
           "leak_attr": z["leak_attr"][keep][lab_k] >= 1, "leak_rewr": z["leak_rewr"][keep][lab_k] >= 1,
           "v_student": s_out[lab_k, BK.N_ACT], "v_pstar": p_out[lab_k, BK.N_ACT],
           "pre_agree_pstar": (s_arg[pre_k] == pia[pre_k]).astype(np.int8)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in doc.items()})  # type: ignore[arg-type]
    return {"fold": fold, "rows": int(lab_k.sum()), "pre_rows": int(pre_k.sum())}


def _deal_sums(deal: np.ndarray, diff: np.ndarray):
    keys, inv = np.unique(deal, return_inverse=True)
    return np.bincount(inv, weights=diff.astype(np.float64), minlength=len(keys)), \
        np.bincount(inv, minlength=len(keys)).astype(np.float64)


def bootstrap_lb(per_panel: dict[str, dict[str, np.ndarray]], B: int = B_BOOT, seed: int = SEED) -> dict[str, Any]:
    """The pooled Δagree and its deal-clustered percentile interval (module docstring). ``per_panel[p]`` holds
    ``deal`` and ``diff`` (a(student) − a(P*)) per label row."""
    panels = [p for p in BK.U_PANELS if p in per_panel]
    if panels != list(BK.U_PANELS):
        raise DeliveryError(f"the gate needs all three panels, got {sorted(per_panel)}")
    sums = {p: _deal_sums(per_panel[p]["deal"], per_panel[p]["diff"]) for p in panels}
    est = {p: float(s.sum() / c.sum()) for p, (s, c) in sums.items()}
    pooled = float(np.mean([est[p] for p in panels]))
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        vals = []
        for p in panels:
            s, c = sums[p]
            ix = rng.integers(0, len(s), len(s))
            vals.append(s[ix].sum() / c[ix].sum())
        reps[b] = np.mean(vals)
    from main.reads.rules_search_headroom import cmp

    lb = float(np.percentile(reps, 2.5))
    delivered, boundary = cmp(lb, 0, RULE8_EPS, above=True)       # exact fractions: the read rule's own test
    return {"per_panel": est, "pooled": pooled, "lb": lb, "ub": float(np.percentile(reps, 97.5)),
            "delivered": bool(delivered), "boundary": bool(boundary)}


def gate(scores: Path, bank_sha256: str) -> dict[str, Any]:
    per: dict[str, dict[str, np.ndarray]] = {}
    docs: dict[str, dict[str, Any]] = {}
    for p in BK.U_PANELS:
        with np.load(scores / f"{p}.npz") as z:
            d = {k: z[k] for k in z.files}
        if str(d["schema"]) != SCORE_SCHEMA or str(d["fold"]) != p or str(d["bank_manifest_sha256"]) != bank_sha256:
            raise DeliveryError(f"{p}.npz: not a {SCORE_SCHEMA} score of fold {p} on the registered bank")
        docs[p] = d
        per[p] = {"deal": d["deal"], "diff": d["a_student"].astype(np.int64) - d["a_pstar"].astype(np.int64)}
    g = bootstrap_lb(per)
    rep: dict[str, Any] = {}
    for p, d in docs.items():
        diff = per[p]["diff"]
        # the leak block's ``residual`` / ``partial`` are the PRESET's static leak names, the same on every searched
        # record (``worlds``: ``SEARCH.residual_of(spec)``), so only the per-decision counters split (REVIEW R-m4)
        flags = {"attribution_miss": d["leak_attr"], "rewritten_moves": d["leak_rewr"]}
        flags["neither"] = ~(d["leak_attr"] | d["leak_rewr"])
        rep[p] = {"by_exposure": {k: {"rows": int(m.sum()), "delta": float(diff[m].mean()) if m.any() else None}
                                  for k, m in flags.items()},
                  "departures": {"rows": int(d["departure"].sum()),
                                 "agree_student": float(d["a_student"][d["departure"]].mean())
                                 if d["departure"].any() else None},
                  "pre_turn10_agree_with_pstar": float(d["pre_agree_pstar"].mean())
                  if len(d["pre_agree_pstar"]) else None,
                  "v_drift_mean_abs": float(np.mean(np.abs(d["v_student"].astype(np.float64)
                                                           - d["v_pstar"].astype(np.float64))))}
    rep["students"] = {p: str(docs[p]["student"]) for p in BK.U_PANELS}
    rep["ub"], rep["rule8_eps"] = g["ub"], RULE8_EPS
    # exactly the read rule's key set (``rules_distill_play.DELIVERY_KEYS``): everything else lives under ``reported``
    doc = {"schema": SCHEMA, "bank_manifest_sha256": bank_sha256,
           "students": {p: {"zip_sha256": str(docs[p]["zip_sha256"])} for p in BK.U_PANELS},
           "per_panel": {p: {"n_rows": len(docs[p]["deal"]), "n_deals": len(np.unique(docs[p]["deal"])),
                             "agree_student": float(docs[p]["a_student"].mean()),
                             "agree_pstar": float(docs[p]["a_pstar"].mean()), "delta": g["per_panel"][p]}
                         for p in BK.U_PANELS},
           "pooled": g["pooled"], "lb": g["lb"], "B": B_BOOT, "seed": SEED,
           "delivered": g["delivered"], "boundary": g["boundary"], "reported": rep}
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x65.delivery", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--bank", required=True)
    s.add_argument("--fold", required=True, choices=sorted(ST.FOLDS))
    s.add_argument("--export", required=True)
    s.add_argument("--init", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--scores", required=True)
    for p in (s, g):
        p.add_argument("--bank-sha256", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "score":
            print(json.dumps(score(Path(a.bank), a.bank_sha256, a.fold, Path(a.export), Path(a.init),
                                   Path(a.out))), flush=True)
            return 0
        doc = gate(Path(a.scores), a.bank_sha256)
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print(json.dumps({k: doc[k] for k in ("pooled", "lb", "delivered", "boundary")}), flush=True)
        return 0
    except (DeliveryError, BK.BankError) as exc:
        print(f"[x65 delivery] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
