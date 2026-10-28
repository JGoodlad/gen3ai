"""THE CANARY RECALIBRATION BACKTEST (2026-10-28). It backs the DRAFT proposal
`designs/training/canary_recalibration_2026-10-28.md`. It runs on CPU in seconds, reads JSON only, and is READ-ONLY on
`models/` and the archives.

It applies the CURRENT canary rules and the PROPOSED ones to three sources:

  1. THE W0 CUDA CENSUS (`../canary_row_null_design_2026-10-25/gpu_census.sh`: 20 X26-lineage checkpoints x 64 fixture
     rows, a 32-seed fp32 dither null per row and seed, the compiled bf16 arm (Cbf) and eager bf16 (Ebf) at the canary's
     B = 2048 on CUDA). It derives the per-row margin's k at several floors and seed counts, then counts its refusals
     three ways: in-sample, leave-one-checkpoint-out and cross-lineage (X26 -> X26c and back). It also counts the
     confirmation outcomes: both halves refusing in the same update, or two consecutive census checkpoints refusing.
  2. EVERY BANKED CANARY RECORD (`<run>/canary_verdicts.jsonl` under `models/`, the X35 family plus the P* lineage
     rb_x26 / rb_x26c). For each check (the scheduled one and each confirmation half) it computes every judged
     quantity's fraction of its bar, as `agents.model.compile_canary.bar_fractions` does. It reads them under the
     current bars and the proposed ones. It applies the `main.ops.canary_decide` family ratio (compiled / eager
     max |d log pi|) as if each record were a FATAL. It applies the per-row margin to the record's worst row against a
     PROXY null (the census null of that fixture row at the nearest census checkpoint, and pessimistically the
     smallest over all 20). A record carries only its worst row, so the per-row result is a LOWER bound on refusals.
  3. THE X-CHECK ANCHORS (`main.ops.canary_dump_replay --anchor`, 5 bit-identical repeats each; repeat 0 is read).
     The same quantities on their three slices.

    python backtest.py --census ~/.claude/jobs/x35/w0_2026-10-28/census \\
        --xcheck ~/.claude/jobs/x35/w0_2026-10-28/xcheck --models <main checkout>/models \\
        --json results/backtest.json > results/backtest.txt

The current bars are read from the source text of `src/agents/model/compile_regions.py` and
`src/main/ops/canary_decide.py` with no import, so no torch is loaded. A drift from the constants below aborts the
run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---- the CURRENT rules (verified against the source text in `check_constants`) ---------------------------------------
M = 0.72                                   # R1_CLIP_MARGIN = R1_BAR_K 4 x R1_LOGPROB_SHIFT_MAX 0.18
R1_BAR_K = 4.0
DRIFT_BARS = {"loss": 0.014, "grad_rel_l2": 0.68}
ENVELOPE_K = {"loss": 5.4, "grad_rel_l2": 6.4, "param": 12.0}
LOSS_FLOOR = max(1e-4, 2.0 ** -9)          # BF16_LOSS_FLOOR
GRAD_FLOOR = math.sqrt(2.0 * (1.0 - 0.9999))   # BF16_GRAD_FLOOR
FAMILY_BARS = {"c0": 2.64, "x47": 2.40}
HALVES = {"confirm_slice2": range(32), "confirm_slice1": range(32, 64)}   # compile_trainer.FIXTURE_HALVES

# ---- the PROPOSAL (the doc's defaults; every one can be overridden on the command line) ----------------------------
PROPOSED = {
    "row_k": None,         # None = DERIVED: 4 x the healthy max of the B = 2048 bf16 arms (Cbf + Ebf) at the
                           # production null (row_seeds, row_floor), rounded UP to 2 significant digits (R1's convention)
    "row_floor": 0.02,     # canary_row_null.ROW_NULL_FLOOR
    "row_seeds": 16,       # canary_row_null.ROW_NULL_SEEDS
    "row_cap": 1.2,        # the per-row bar's ceiling = the scalar relocation margin (the ruled --canary-shift-max 0.3)
}

ROOT = Path(__file__).resolve().parents[4]


def check_constants(root: Path) -> list[str]:
    """Compare this script's CURRENT constants with the source text. An empty list means they agree."""
    bad: list[str] = []
    cr = (root / "src/agents/model/compile_regions.py").read_text()
    cd = (root / "src/main/ops/canary_decide.py").read_text()
    want = {
        r"^BF16_DRIFT_BARS[^=]*=\s*(\{[^}]*\})": DRIFT_BARS,
        r"^BF16_ENVELOPE_K[^=]*=\s*(\{[^}]*\})": ENVELOPE_K,
    }
    for pat, val in want.items():
        m = re.search(pat, cr, re.MULTILINE)
        got = json.loads(m.group(1).replace("'", '"')) if m else None
        if got != val:
            bad.append(f"{pat}: source {got} != {val}")
    for name, val in (("R1_LOGPROB_SHIFT_MAX", 0.18), ("R1_BAR_K", 4.0)):
        m = re.search(rf"^{name}\s*=\s*([0-9.]+)", cr, re.MULTILINE)
        if not m or float(m.group(1)) != val:
            bad.append(f"{name}: source {m.group(1) if m else None} != {val}")
    m = re.search(r"^FAMILY_BARS[^=]*=\s*(\{[^}]*\})", cd, re.MULTILINE)
    got = json.loads(m.group(1).replace("'", '"')) if m else None
    if got != FAMILY_BARS:
        bad.append(f"FAMILY_BARS: source {got} != {FAMILY_BARS}")
    return bad


def round_up_2sig(x: float) -> float:
    """R1's convention for a bar: rounded UP to 2 significant digits."""
    if x <= 0:
        return x
    e = math.floor(math.log10(x)) - 1
    return float(f"{math.ceil(x / 10 ** e - 1e-9) * 10 ** e:.10g}")


# =====================================================================================================================
# 1. THE CENSUS
# =====================================================================================================================
def load_census(d: Path) -> list[dict[str, Any]]:
    """One entry per checkpoint: label, step, the [64, S] per-seed null, and each arm's per-row shift."""
    out = []
    for f in sorted(glob.glob(str(d / "read_*.json"))):
        r = json.load(open(f))
        rows = r["dither"]["rows"]
        per_seed = np.array([rows[str(i)]["E32~"] for i in range(64)], dtype=float)
        full = np.asarray(r["dither"]["E32~_per_row_max"], dtype=float)
        if not np.allclose(per_seed.max(1), full, atol=1e-7):
            raise SystemExit(f"{f}: the per-seed null does not reproduce E32~_per_row_max")
        c = r["compiled"]
        if c["batch"] != 2048 or c["device"] != "cuda":
            raise SystemExit(f"{f}: compiled read at B {c['batch']} on {c['device']}, not the canary's B 2048 on cuda")
        step = int(re.search(r"checkpoint_(\d+)_steps", r["zip"]).group(1))
        out.append({"label": r["label"], "step": step, "per_seed": per_seed,
                    "arms": {"Cbf": np.asarray(c["per_row"]["Cbf_vs_E32"], dtype=float),
                             "Ebf": np.asarray(c["per_row"]["Ebf_vs_E32"], dtype=float),
                             "Ebf_B64": np.asarray(r["per_row"]["abs_dlogp_Ebf"], dtype=float)}})
    out.sort(key=lambda e: e["step"])
    return out


def null_k(e: dict[str, Any], seeds: int) -> np.ndarray:
    return e["per_seed"][:, :seeds].max(1)


def ratios(e: dict[str, Any], arm: str, seeds: int, floor: float) -> np.ndarray:
    return e["arms"][arm] / np.maximum(null_k(e, seeds), floor)


def bar_per_row(e: dict[str, Any], seeds: int, floor: float, k: float, cap: float | None) -> np.ndarray:
    b = k * np.maximum(null_k(e, seeds), floor)
    return np.minimum(b, cap) if cap else b


def judge(e: dict[str, Any], arm: str, seeds: int, floor: float, k: float, cap: float | None) -> dict[str, Any]:
    """The per-row rule on one checkpoint: failing rows, the scheduled verdict, and each half's verdict."""
    fail = np.where(e["arms"][arm] > bar_per_row(e, seeds, floor, k, cap))[0]
    halves = {h: bool(any(i in rs for i in fail)) for h, rs in HALVES.items()}
    return {"fail_rows": [int(i) for i in fail], "refuses": bool(len(fail)), "both_halves": all(halves.values())}


def derive_row_k(cs: list[dict[str, Any]], seeds: int, floor: float) -> tuple[float, float, Any]:
    """(k, the healthy max, where): R1's convention on the canary's own arms at B = 2048 (Cbf, Ebf)."""
    mx, at = -1.0, None
    for e in cs:
        for a in ("Cbf", "Ebf"):
            r = ratios(e, a, seeds, floor)
            j = int(r.argmax())
            if r[j] > mx:
                mx, at = float(r[j]), (e["label"], a, j, float(e["arms"][a][j]), float(null_k(e, seeds)[j]))
    return round_up_2sig(R1_BAR_K * mx), mx, at


def census_section(cs: list[dict[str, Any]], prop: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"checkpoints": [e["label"] for e in cs]}
    # (a) k by arm set x floor x seeds
    table = {}
    for seeds in (8, 16, 32):
        for floor in (0.01, 0.02, 0.05):
            for arms in (("Cbf",), ("Ebf",), ("Cbf", "Ebf"), ("Cbf", "Ebf", "Ebf_B64")):
                mx, at = -1.0, None
                for e in cs:
                    for a in arms:
                        r = ratios(e, a, seeds, floor)
                        j = int(r.argmax())
                        if r[j] > mx:
                            mx, at = float(r[j]), (e["label"], a, j, float(e["arms"][a][j]), float(null_k(e, seeds)[j]))
                table[f"K{seeds}_f{floor}_{'+'.join(arms)}"] = {"max_ratio": mx, "k_4x": round_up_2sig(R1_BAR_K * mx),
                                                                "argmax": at}
    out["k_table"] = table
    # (b) the proposal in-sample: refusals, both-halves, consecutive pairs, the teeth
    k, f, s, cap = prop["row_k"], prop["row_floor"], int(prop["row_seeds"]), prop["row_cap"]
    per = []
    for e in cs:
        j = judge(e, "Cbf", s, f, k, cap)
        ev = judge(e, "Ebf", s, f, k, cap)
        per.append({"label": e["label"], "Cbf": j, "Ebf_events": ev["fail_rows"],
                    "Cbf_max_shift": float(e["arms"]["Cbf"].max()), "Cbf_row_of_max": int(e["arms"]["Cbf"].argmax())})
    consec = sum(1 for a, b in zip(per, per[1:]) if a["Cbf"]["refuses"] and b["Cbf"]["refuses"])
    bars = np.concatenate([bar_per_row(e, s, f, k, cap) for e in cs])
    out["proposal_in_sample"] = {
        "rule": f"Cbf row i refuses iff shift_i > min({k} x max(null_K{s}_i, {f}), {cap})",
        "checkpoints_refusing": sum(p["Cbf"]["refuses"] for p in per),
        "checkpoints_both_halves": sum(p["Cbf"]["both_halves"] for p in per),
        "consecutive_pairs_refusing": consec,
        "ebf_yardstick_rows_over_bar": sum(len(p["Ebf_events"]) for p in per),
        "row_reads": int(bars.size),
        "bar_median": float(np.median(bars)), "bar_p10": float(np.quantile(bars, 0.1)),
        "bar_p90": float(np.quantile(bars, 0.9)), "bar_max": float(bars.max()),
        "rows_tighter_than_M": float((bars < M).mean()), "rows_at_cap": float((bars >= cap - 1e-12).mean()),
        "share_of_row_reads_catching_a_planted_shift_of": {str(x): float((bars < x).mean())
                                                           for x in (0.2, 0.3, 0.5, 0.72, 1.0)},
        "old_rule_refusals": int(sum((e["arms"]["Cbf"] > M).sum() for e in cs)),
        "per_checkpoint": per,
    }
    # (c) leave-one-checkpoint-out: k from the other 19 (Cbf + Ebf pooled), judged on the held-out one
    loo = []
    for i, e in enumerate(cs):
        mx = max(float(ratios(o, a, s, f).max()) for j, o in enumerate(cs) if j != i for a in ("Cbf", "Ebf"))
        kk = round_up_2sig(R1_BAR_K * mx)
        jj = judge(e, "Cbf", s, f, kk, cap)
        loo.append({"held_out": e["label"], "k": kk, "refuses": jj["refuses"], "fail_rows": jj["fail_rows"],
                    "held_out_max_ratio": float(ratios(e, "Cbf", s, f).max())})
    out["leave_one_out"] = {"refusing": sum(x["refuses"] for x in loo), "of": len(loo),
                            "k_range": [min(x["k"] for x in loo), max(x["k"] for x in loo)], "rows": loo}
    # (d) cross-lineage
    xl = {}
    for name, src, dst in (("x26->x26c", "rb_x26_s", "rb_x26c_"), ("x26c->x26", "rb_x26c_", "rb_x26_s")):
        a = [e for e in cs if e["label"].startswith(src) or (src == "rb_x26c_" and e["label"].startswith("x26c_"))]
        b = [e for e in cs if e["label"].startswith(dst) or (dst == "rb_x26c_" and e["label"].startswith("x26c_"))]
        mx = max(float(ratios(o, arm, s, f).max()) for o in a for arm in ("Cbf", "Ebf"))
        kk = round_up_2sig(R1_BAR_K * mx)
        xl[name] = {"k": kk, "source_n": len(a), "target_n": len(b),
                    "target_refusing": sum(judge(e, "Cbf", s, f, kk, cap)["refuses"] for e in b)}
    out["cross_lineage"] = xl
    # (e) seeds: the bar's tightness and the in-sample refusals at the proposal's k, for each seed count
    out["seed_sensitivity_at_proposed_k"] = {
        f"K{ss}": {"checkpoints_refusing": sum(judge(e, "Cbf", ss, f, k, cap)["refuses"] for e in cs),
                   "bar_median": float(np.median(np.concatenate([bar_per_row(e, ss, f, k, cap) for e in cs])))}
        for ss in (8, 16, 32)}
    return out


# =====================================================================================================================
# 2. THE BANKED RECORDS (and 3. the X-CHECK anchors, read through the same function)
# =====================================================================================================================
def fractions(w: dict[str, Any], *, loss_k: float, param_k: float) -> dict[str, float]:
    """Every judged quantity of one check as the fraction of its bar (compile_canary.bar_fractions' arithmetic)."""
    out: dict[str, float] = {}
    cm = w.get("clip_margin") or {}
    sh = cm.get("max_abs_dlogp") or {}
    if sh and cm.get("margin"):
        out["clip_margin"] = max(float(v) for v in sh.values()) / float(cm["margin"])
    g = w.get("global") or {}
    if w.get("rule") == "bf16":
        if "drift_loss" in g:
            out["drift_loss"] = g["drift_loss"] / DRIFT_BARS["loss"]
            if "envelope_loss" in g:
                out["envelope_loss"] = g["envelope_loss"] / (loss_k * max(g["drift_loss"], LOSS_FLOOR))
        if "drift_grad" in g:
            out["drift_grad"] = g["drift_grad"] / DRIFT_BARS["grad_rel_l2"]
            if "envelope_grad" in g:
                out["envelope_grad"] = g["envelope_grad"] / (ENVELOPE_K["grad_rel_l2"] * max(g["drift_grad"], GRAD_FLOOR))
        wr = (w.get("worst") or {}).get("ratio")
        if wr is not None:
            out["param_ratio"] = float(wr) / param_k
    return out


def loss_envelope_ratio(w: dict[str, Any]) -> float | None:
    """envelope_loss / max(drift_loss, the floor): the statistic BF16_ENVELOPE_K['loss'] bars."""
    g = w.get("global") or {}
    if "envelope_loss" not in g or "drift_loss" not in g:
        return None
    return g["envelope_loss"] / max(g["drift_loss"], LOSS_FLOOR)


def family_ratio(w: dict[str, Any]) -> float | None:
    sh = (w.get("clip_margin") or {}).get("max_abs_dlogp") or {}
    c, e = sh.get("compiled"), sh.get("eager")
    return (float(c) / float(e)) if (c is not None and e) else None


def group_of(run: str) -> str:
    if run.startswith("rb_x35t_"):
        return "toy"
    if run.startswith("rb_x35_"):
        return "x35"
    if run.startswith(("rb_x26_", "rb_x26c_")):
        return "pstar_lineage"
    return "other"


def load_records(models: Path) -> list[dict[str, Any]]:
    recs = []
    for f in sorted(glob.glob(str(models / "rb_x35*" / "canary_verdicts.jsonl"))
                    + glob.glob(str(models / "rb_x26_s2601" / "canary_verdicts.jsonl"))
                    + glob.glob(str(models / "rb_x26c_s2601" / "canary_verdicts.jsonl"))):
        run = Path(f).parent.name
        for i, ln in enumerate(open(f)):
            if ln.strip():
                r = json.loads(ln)
                r["_run"], r["_line"], r["_group"] = run, i, group_of(run)
                recs.append(r)
    return recs


def archive_copies(archive: Path, recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Every archived canary_verdicts.jsonl line must be a copy of a models/ record (matched on step, update, verdict)."""
    have = {(r.get("num_timesteps"), r.get("update"), r.get("verdict")) for r in recs}
    out = {}
    for f in sorted(glob.glob(str(archive / "*" / "canary_verdicts.jsonl"))):
        lines = [json.loads(x) for x in open(f) if x.strip()]
        out[f] = {"lines": len(lines),
                  "not_in_models": sum((r.get("num_timesteps"), r.get("update"), r.get("verdict")) not in have
                                       for r in lines)}
    return out


def proxy_null(cs: list[dict[str, Any]], step: int, seeds: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Per fixture row: (label, the NEAREST census checkpoint's null, the smaller of the two census checkpoints that
    BRACKET the step, the smallest over all 20). An X35 arm forks P* (rb_x26_s2601 @ 280M), so its lineage checkpoint
    is read by its step; the toy arms get no proxy (other weights)."""
    lineage = sorted((e for e in cs if e["label"].startswith(("rb_x26", "x26c_"))), key=lambda e: e["step"])
    near = min(lineage, key=lambda e: abs(e["step"] - step))
    lo = [e for e in lineage if e["step"] <= step]
    hi = [e for e in lineage if e["step"] >= step]
    br = [x for x in ((lo[-1] if lo else None), (hi[0] if hi else None)) if x is not None]
    return (near["label"], null_k(near, seeds), np.min([null_k(e, seeds) for e in br], axis=0),
            np.min([null_k(e, seeds) for e in cs], axis=0))


def records_section(recs: list[dict[str, Any]], cs: list[dict[str, Any]], prop: dict[str, float],
                    loss_k_new: float) -> dict[str, Any]:
    k, f, s, cap = prop["row_k"], prop["row_floor"], int(prop["row_seeds"]), prop["row_cap"]
    rows = []
    for r in recs:
        for chk, w in (r.get("watch") or {}).items():
            if not isinstance(w, dict) or w.get("rule") != "bf16":
                continue
            cur = fractions(w, loss_k=ENVELOPE_K["loss"], param_k=ENVELOPE_K["param"])
            new = fractions(w, loss_k=loss_k_new, param_k=ENVELOPE_K["param"])
            cm = w.get("clip_margin") or {}
            row = {"run": r["_run"], "group": r["_group"], "update": r.get("update"), "step": r.get("num_timesteps"),
                   "verdict": r.get("verdict"), "check": chk, "margin": cm.get("margin"),
                   "cur": cur, "new": new, "loss_env_ratio": loss_envelope_ratio(w), "family_ratio": family_ratio(w),
                   "worst_param": (w.get("worst") or {}).get("param")}
            if r["_group"] in ("x35", "pstar_lineage") and cm.get("worst_fixture_row"):
                lab, nn, nbr, nmin = proxy_null(cs, int(r.get("num_timesteps") or 0), s)
                pr = {}
                for arm in ("compiled", "eager"):
                    fr = (cm.get("worst_fixture_row") or {}).get(arm)
                    sh = (cm.get("max_abs_dlogp") or {}).get(arm)
                    if fr is None or sh is None:
                        continue
                    b_near = min(k * max(float(nn[fr]), f), cap)
                    b_br = min(k * max(float(nbr[fr]), f), cap)
                    b_min = min(k * max(float(nmin[fr]), f), cap)
                    pr[arm] = {"row": int(fr), "shift": float(sh), "bar_nearest": b_near, "bar_bracket": b_br,
                               "bar_pessimistic": b_min, "over_nearest": float(sh) > b_near,
                               "over_bracket": float(sh) > b_br, "over_pessimistic": float(sh) > b_min}
                row["per_row_proxy"] = {"census": lab, **pr}
            rows.append(row)
    return {"rows": rows}


def summarise_records(sec: dict[str, Any], prop: dict[str, float], loss_k_new: float) -> dict[str, Any]:
    rows = sec["rows"]
    out: dict[str, Any] = {}
    for grp in ("x35", "pstar_lineage", "toy"):
        g = [x for x in rows if x["group"] == grp]
        sched = [x for x in g if x["check"] == "scheduled"]
        verdicts: dict[str, int] = {}
        for x in sched:
            verdicts[x["verdict"]] = verdicts.get(x["verdict"], 0) + 1
        def over(xs, key, q, thr=1.0):
            return sum(1 for x in xs if x[key].get(q, 0.0) > thr)
        qs = ("clip_margin", "drift_loss", "drift_grad", "envelope_loss", "envelope_grad", "param_ratio")
        fam = [x["family_ratio"] for x in sched if x["family_ratio"] is not None and x["verdict"] == "pass"]
        env = [x["loss_env_ratio"] for x in g if x["loss_env_ratio"] is not None]
        env_pass = [x["loss_env_ratio"] for x in sched if x["loss_env_ratio"] is not None and x["verdict"] == "pass"]
        env_halves = [x["loss_env_ratio"] for x in g if x["loss_env_ratio"] is not None and x["check"] in HALVES]
        pw = [x["cur"].get("param_ratio", 0) * ENVELOPE_K["param"] for x in sched if x["verdict"] == "pass"]
        s = {"runs": len({x["run"] for x in g}), "scheduled_records": len(sched), "checks": len(g),
             "recorded_verdicts": verdicts,
             "scheduled_over_bar_current": {q: over(sched, "cur", q) for q in qs},
             "all_checks_over_bar_current": {q: over(g, "cur", q) for q in qs},
             "scheduled_near_margin_current": {q: over(sched, "cur", q, 0.9) for q in qs},
             "scheduled_over_loss_bar_proposed": over(sched, "new", "envelope_loss"),
             "all_checks_over_loss_bar_proposed": over(g, "new", "envelope_loss"),
             "loss_env_ratio_max_all_checks": max(env) if env else None,
             "loss_env_ratio_max_scheduled_pass": max(env_pass) if env_pass else None,
             "loss_env_ratio_max_on_halves": max(env_halves) if env_halves else None,
             "loss_env_ratio_halves_n": len(env_halves),
             "param_ratio_max_scheduled_pass": max(pw) if pw else None,
             "param_ratio_pass_over_10": sorted(round(v, 2) for v in pw if v > 10),
             "edge_bias_worst": {"scheduled_records_where_worst": sum(
                                     1 for x in sched if str(x["worst_param"]).startswith("features_extractor.edge_bias.")),
                                 "ratios_over_8": sorted(
                                     (round(x["cur"]["param_ratio"] * ENVELOPE_K["param"], 2), x["run"], x["update"],
                                      x["check"], x["worst_param"].split(".", 2)[-1])
                                     for x in g if str(x["worst_param"]).startswith("features_extractor.edge_bias.")
                                     and x["cur"].get("param_ratio", 0) * ENVELOPE_K["param"] > 8)},
             "family_ratio_max_pass": max(fam) if fam else None,
             "family_ratio_pass_over_c0_bar": sum(1 for v in fam if v > FAMILY_BARS["c0"]),
             "family_ratio_pass_n": len(fam)}
        pp = [x for x in sched if "per_row_proxy" in x]
        if pp:
            s["per_row_proxy_scheduled"] = {
                "n": len(pp),
                "compiled_over_nearest": sum(1 for x in pp if x["per_row_proxy"].get("compiled", {}).get("over_nearest")),
                "compiled_over_bracket": sum(1 for x in pp if x["per_row_proxy"].get("compiled", {}).get("over_bracket")),
                "compiled_over_pessimistic": sum(1 for x in pp
                                                 if x["per_row_proxy"].get("compiled", {}).get("over_pessimistic")),
                "eager_over_nearest (not judged in the proposal)": sum(
                    1 for x in pp if x["per_row_proxy"].get("eager", {}).get("over_nearest")),
                "which_compiled_over_bracket": [
                    (x["run"], x["update"], x["per_row_proxy"]["compiled"]["row"],
                     round(x["per_row_proxy"]["compiled"]["shift"], 3),
                     round(x["per_row_proxy"]["compiled"]["bar_bracket"], 3))
                    for x in pp if x["per_row_proxy"].get("compiled", {}).get("over_bracket")]}
        out[grp] = s
    # the near-margin records (>= 0.9 of any current bar on the scheduled check), x35 + P* lineage
    out["near_margin_records"] = [
        {"run": x["run"], "update": x["update"], "verdict": x["verdict"],
         "quantities": {q: round(v, 3) for q, v in x["cur"].items() if v >= 0.9}}
        for x in rows if x["group"] in ("x35", "pstar_lineage") and x["check"] == "scheduled"
        and any(v >= 0.9 for v in x["cur"].values())]
    # the FATAL arithmetic under the current rules, x35 + P* lineage: same-update both halves, or consecutive scheduled
    fat = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for x in rows:
        if x["group"] in ("x35", "pstar_lineage"):
            by_run.setdefault(x["run"], []).append(x)
    for run, xs in by_run.items():
        sched = sorted([x for x in xs if x["check"] == "scheduled"], key=lambda x: x["step"] or 0)
        refused = [any(v > 1 for v in x["cur"].values()) for x in sched]
        for a, b, xa, xb in zip(refused, refused[1:], sched, sched[1:]):
            if a and b:
                fat.append({"run": run, "kind": "consecutive", "updates": [xa["update"], xb["update"]]})
        for x in sched:
            halves = [y for y in xs if y["step"] == x["step"] and y["check"] in HALVES]
            if len(halves) == 2 and all(any(v > 1 for v in y["cur"].values()) for y in halves):
                fat.append({"run": run, "kind": "both_halves", "updates": [x["update"]]})
    out["fatal_paths_current"] = fat
    return out


def anchors_section(xdir: Path, loss_k_new: float) -> dict[str, Any]:
    out = {}
    for d in sorted(p for p in xdir.iterdir() if p.is_dir() and (p / "repeat_00.json").exists()):
        rep = json.load(open(d / "repeat_00.json"))
        sl = {}
        for s, v in rep["slices"].items():
            w = v["canary"]["watch"]
            sl[s] = {"class": v["canary"]["class"],
                     "cur": {q: round(x, 3) for q, x in fractions(w, loss_k=ENVELOPE_K["loss"],
                                                                   param_k=ENVELOPE_K["param"]).items()},
                     "envelope_loss_proposed": round(fractions(w, loss_k=loss_k_new,
                                                               param_k=ENVELOPE_K["param"]).get("envelope_loss", 0), 3),
                     "loss_env_ratio": loss_envelope_ratio(w), "family_ratio": family_ratio(w),
                     "worst": (w.get("worst") or {}).get("param")}
        out[d.name] = sl
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--xcheck", required=True, type=Path)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--archive", type=Path, default=Path.home() / "gen3ai_archive")
    ap.add_argument("--row-k", type=float, default=PROPOSED["row_k"])
    ap.add_argument("--row-floor", type=float, default=PROPOSED["row_floor"])
    ap.add_argument("--row-seeds", type=int, default=PROPOSED["row_seeds"])
    ap.add_argument("--row-cap", type=float, default=PROPOSED["row_cap"])
    ap.add_argument("--loss-k", type=float, default=None,
                    help="the proposed BF16_ENVELOPE_K['loss'] (default: derived = 4 x the production healthy max)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    bad = check_constants(ROOT)
    if bad:
        print("REFUSING: the current rules drifted from this script:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    cs = load_census(a.census)
    k_derived, k_max, k_at = derive_row_k(cs, a.row_seeds, a.row_floor)
    prop = {"row_k": a.row_k if a.row_k is not None else k_derived, "row_floor": a.row_floor,
            "row_seeds": a.row_seeds, "row_cap": a.row_cap,
            "row_k_basis": f"4 x max {k_max:.3f} (Cbf + Ebf at B 2048, K {a.row_seeds}, floor {a.row_floor}) at {k_at}"}
    recs = load_records(a.models)
    # the proposed loss K: 4 x the maximum of the loss-envelope ratio over EVERY check of every production record (x35 +
    # P* lineage, the scheduled check and each confirmation half; none refused on the loss), rounded up (R1's
    # convention). The X-CHECK anchors are kept OUT of the basis: they are the out-of-sample test.
    prod_env = []
    for r in recs:
        if r["_group"] not in ("x35", "pstar_lineage"):
            continue
        for chk, w in (r.get("watch") or {}).items():
            if isinstance(w, dict) and w.get("rule") == "bf16":
                v = loss_envelope_ratio(w)
                if v is not None:
                    prod_env.append(v)
    loss_k_new = a.loss_k if a.loss_k is not None else round_up_2sig(R1_BAR_K * max(prod_env))
    rec = records_section(recs, cs, prop, loss_k_new)
    res = {"proposal": {**prop, "loss_k": loss_k_new, "loss_k_basis": f"4 x max {max(prod_env):.3f} over "
                        f"{len(prod_env)} production checks (every slice)"},
           "current": {"M": M, "DRIFT_BARS": DRIFT_BARS, "ENVELOPE_K": ENVELOPE_K, "LOSS_FLOOR": LOSS_FLOOR,
                       "GRAD_FLOOR": GRAD_FLOOR, "FAMILY_BARS": FAMILY_BARS},
           "census": census_section(cs, prop),
           "records": summarise_records(rec, prop, loss_k_new),
           "anchors": anchors_section(a.xcheck, loss_k_new),
           "archive_copies": archive_copies(a.archive, recs) if a.archive.is_dir() else {}}
    print(json.dumps({k: v for k, v in res.items() if k != "census"}, indent=1, default=str))
    c = dict(res["census"])
    c["proposal_in_sample"] = {k: v for k, v in c["proposal_in_sample"].items() if k != "per_checkpoint"}
    c["leave_one_out"] = {k: v for k, v in c["leave_one_out"].items() if k != "rows"}
    c["k_table"] = {k: v for k, v in c["k_table"].items() if "_f0.02_" in k}
    print(json.dumps({"census": c}, indent=1, default=str))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
