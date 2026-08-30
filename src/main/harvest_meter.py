"""harvest_meter — probe O's battery, re-run PAIRED on held-out stall tails, pre vs post fine-tune.

    python -m main.harvest_meter <harvest_dir> [--head <finetune_out>/head_best.pt] [--out DIR]

This is the **reducibility measurement** the harvest exists to enable. Probe O
(``designs/research_state/measurements/stall_tail_head_reading_2026-08-29.md``) established that a
trained win-prob head reads hopeful on 34.8% of the tails of games it is losing by construction.
The registered follow-up was whether that tail is REDUCIBLE — and the discriminating test is
whether purpose-harvested labels move it. This module scores that.

What is measured, and against what
----------------------------------
For every **held-out** battle (the split ``main.harvest`` wrote to ``holdout.json`` before it
bought a single label), the last ``K = 5`` recorded decisions are re-scored through the win-prob
head twice: once with the SUBJECT's head, once with the fine-tuned head. Probe O's metrics,
verbatim:

===============  =========================================================================
``detect_le05``  ``phi_T <= 0.5`` — probe O's substantive detection criterion
``detect``       ``phi_T <= 0.5 OR phi_T < phi_{T-4}`` — the criterion AS REGISTERED, which
                 probe O found saturates (its "declining" half fires 83-89% in every class,
                 including regular losses). Reported because it was registered, never as the
                 headline. A criterion is not retro-fitted after the fact; it is reported
                 twice.
``miss``         ``phi_T >= 0.5`` on a game that is a loss by construction — probe O's 34.8%
``overconf``     ``max(phi over last 5) >= 0.70``
``c3band``       ``mean(phi over last 5) in [0.70, 0.98]`` — C3's over-confidence band
===============  =========================================================================

Three properties of this design matter more than the numbers it produces.

**1. It is PAIRED, and probe O was not.** Probe O compared populations across runs with a
run-clustered bootstrap because it had no alternative — it was reading recorded values from models
that no longer existed. Here the same states are scored by both arms, so the comparison is a
within-state difference and the bootstrap is over BATTLES (the cluster), not over states. That is
strictly more powerful at the same n, and at these n it is the difference between a readable result
and noise.

**2. "Pre" is the SUBJECT's reading, not the trace's.** The recorded ``win_probs`` column came from
whichever checkpoint played the battle. Re-scoring both arms with the same trunk isolates the head
change and nothing else. This makes the meter's "pre" numbers legitimately different from probe O's
published ones, and they should not be compared to them directly — probe O measured a corpus of
many runs' own readings; this measures one subject's reading of a subset of that corpus.

**3. The LONG-WIN CONTROL is not optional, and it is the whole falsification.** A head that has
learned "late means lost" scores PERFECTLY on every metric above, and it is worse than the head it
replaced — probe O measured ``LONG_WIN`` at phi = 0.986 with 96.3% at or above 0.7, so the head's
length reading is currently RIGHT and a repair that breaks it is a regression wearing a success's
clothes. The meter therefore scores an untouched long-win control set (battles neither harvested
nor held out) and reports it beside the detection rate. **A detection gain with a long-win
collapse is a FAILED run**, and the summary says so in those words rather than leaving a reader to
notice.

n is small and is stated everywhere
-----------------------------------
The current-arch corpus holds 48 cap battles in total. A 35% holdout is ~17. Every table carries
its n, the bootstrap is over battles, and the verdict line refuses to call a difference whose CI
spans zero. This is a smoke of the pipeline first and a measurement second; the pilot's job is to
prove the meter can read, not to settle reducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The tail window probe O reads. Not a knob: changing it makes the numbers incomparable with the
#: probe this meter exists to re-run.
K_TAIL = 5

#: C3's over-confidence band.
C3_LO, C3_HI = 0.70, 0.98


def _load_tail(prefix: str, k: int = K_TAIL) -> "Optional[Tuple[np.ndarray, dict]]":
    """The last ``k`` recorded decision states of one battle, as ``(obs[k, D], meta)``.

    ``has_state`` gates the rows exactly as probe O did — a decision with no recorded state is not
    a decision the head was asked about.
    """
    try:
        with open(prefix + "_summary.json") as fh:
            summ = json.load(fh)
        with np.load(prefix + "_states.npz") as z:
            if "obs" not in z.files or "has_state" not in z.files:
                return None
            obs = np.asarray(z["obs"], dtype=np.float32)
            has = np.asarray(z["has_state"], dtype=int)
            rec = np.asarray(z["win_probs"], dtype=float) if "win_probs" in z.files else None
    except Exception:                                                   # noqa: BLE001
        return None
    idx = [i for i in range(min(len(has), len(obs))) if has[i]]
    if len(idx) < k:
        return None
    idx = idx[-k:]
    meta = summ.get("meta") or {}
    return obs[idx], {
        "result": str(meta.get("result") or "").lower(),
        "turns": int(meta.get("turns") or 0),
        "recorded_phi_T": float(rec[idx[-1]]) if rec is not None and idx[-1] < len(rec) else None,
    }


def metrics_from_phi(phi: Sequence[float]) -> dict:
    """Probe O's per-battle metrics over one tail of ``K_TAIL`` phi values."""
    p = np.asarray(phi, dtype=float)
    pT = float(p[-1])
    declining = bool(pT < float(p[0]))
    return {
        "phi_T": pT,
        "phi_mean": float(p.mean()),
        "phi_max": float(p.max()),
        "detect_le05": float(pT <= 0.5),
        "detect": float(pT <= 0.5 or declining),
        "miss": float(pT >= 0.5),
        "miss_090": float(pT >= 0.90),
        "miss_098": float(pT >= 0.98),
        "overconf": float(p.max() >= C3_LO),
        "c3band": float(C3_LO <= p.mean() <= C3_HI),
    }


def score_battles(pm, prefixes: Sequence[str]) -> "Tuple[List[dict], List[str]]":
    """Score every battle's tail through ``pm``'s win-prob head. Returns ``(rows, dropped)``."""
    from main.harvest import _win_prob_batch

    rows, dropped = [], []
    for pref in prefixes:
        got = _load_tail(pref)
        if got is None:
            dropped.append(pref)
            continue
        obs, meta = got
        phi = _win_prob_batch(pm, obs)
        if phi is None:
            dropped.append(pref)
            continue
        row = {"battle_tag": pref, **meta, **metrics_from_phi(phi),
               "phi_traj": [round(float(x), 4) for x in phi]}
        rows.append(row)
    return rows, dropped


def paired_diff_ci(pre: Sequence[float], post: Sequence[float], *, draws: int = 4000,
                   seed: int = 11) -> dict:
    """Bootstrap CI on the PAIRED post-minus-pre difference, resampling BATTLES.

    Battles are the independent unit; the five decisions inside one tail are not. Resampling states
    would produce an interval roughly ``sqrt(5)`` too narrow and would be the Simpson-adjacent
    error the ledger keeps re-learning, so the cluster is the battle and the statistic is the mean
    of within-battle differences.
    """
    d = np.asarray(post, dtype=float) - np.asarray(pre, dtype=float)
    n = len(d)
    if n == 0:
        return {"n": 0, "diff": None, "ci": None, "significant": False}
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(draws)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"n": n, "diff": float(d.mean()), "ci": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def levels(rows: Sequence[dict], metrics: Sequence[str]) -> Dict[str, dict]:
    """Mean and battle-bootstrapped CI of each metric on ONE arm.

    A PRE-only run is not a degenerate pre/post with the post missing — it is a **baseline read of
    the battery on this holdout**, and it is worth having on its own (it says what the subject
    currently does on the states the fine-tune will be judged on). So the levels are always
    computed and always printed, whether or not a second arm exists.
    """
    out: Dict[str, dict] = {}
    rng = np.random.default_rng(5)
    for m in metrics:
        v = np.asarray([r[m] for r in rows], dtype=float)
        if not len(v):
            out[m] = {"n": 0, "mean": None, "ci": None}
            continue
        boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)])
        out[m] = {"n": int(len(v)), "mean": float(v.mean()),
                  "ci": [round(float(np.percentile(boots, 2.5)), 4),
                         round(float(np.percentile(boots, 97.5)), 4)]}
    return out


def compare(pre_rows: Sequence[dict], post_rows: Sequence[dict],
            metrics: Sequence[str]) -> Dict[str, dict]:
    """Pair the two arms on ``battle_tag`` and diff every metric."""
    post_by = {r["battle_tag"]: r for r in post_rows}
    pairs = [(a, post_by[a["battle_tag"]]) for a in pre_rows if a["battle_tag"] in post_by]
    out: Dict[str, dict] = {}
    for m in metrics:
        pre = [a[m] for a, _ in pairs]
        post = [b[m] for _, b in pairs]
        res = paired_diff_ci(pre, post)
        res.update({"pre": float(np.mean(pre)) if pre else None,
                    "post": float(np.mean(post)) if post else None})
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def control_battles(models_root: str, holdout: dict, *, n: int = 40,
                    min_turns: int = 100, seed: int = 0) -> List[str]:
    """LONG_WIN control battles — long games the model WON, touched by neither arm of the pipeline.

    Excludes every harvested battle and every held-out battle, so this set is uncontaminated in
    both directions: the fine-tune never saw it, and it is not the population the fine-tune was
    aimed at. It exists to answer one question — did the repair break the reading that was already
    right? — and it can only answer it if nothing selected it.
    """
    import glob
    import random

    from main.harvest import current_arch_runs

    used = set(holdout.get("holdout_battles", [])) | set(holdout.get("harvested_battles", []))
    found: List[str] = []
    for run in current_arch_runs(models_root):
        for sp in glob.glob(os.path.join(models_root, run, "eval_traces", "*", "*",
                                         "*_summary.json")):
            base = sp[: -len("_summary.json")]
            tag = os.path.relpath(base, models_root)
            if tag in used:
                continue
            try:
                with open(sp) as fh:
                    meta = (json.load(fh).get("meta") or {})
            except Exception:                                           # noqa: BLE001
                continue
            if str(meta.get("result") or "").lower() == "win" \
                    and int(meta.get("turns") or 0) >= min_turns:
                found.append(tag)
    rng = random.Random(seed)
    found = sorted(found)
    rng.shuffle(found)
    return sorted(found[:n])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_HEADLINE = ("detect_le05", "miss", "miss_098", "overconf", "c3band", "detect", "phi_T")


def render(report: dict) -> str:
    L: List[str] = []
    a = L.append
    a("# Harvest meter — probe O's battery, paired, pre vs post win-prob head fine-tune")
    a("")
    a(f"**Date** {report['finished_iso']} · **subject** `{report['subject_ckpt']}` · "
      f"**head** `{report['head']}` · **harvest** `{report['harvest_dir']}`")
    a("")
    a("Held-out battles were split at the BATTLE level by `main.harvest` before any label was "
      "bought, and no held-out battle contributed a single training state. Both arms score the "
      "SAME states through the SAME trunk, so every difference below is the head's.")
    a("")
    titles = {
        "cap": "Held-out CAP endings (250-turn forfeits — probe O's headline class)",
        "long_loss": "Held-out LONG LOSSES (turns >= 100)",
        "doomed": "Held-out doomed tails",
        "control": "LONG-WIN control (untouched by both arms) — the falsification",
    }
    for key in [k for k in ("cap", "long_loss", "doomed") if k in report["populations"]] + \
            ([("control")] if "control" in report["populations"] else []):
        blk = report["populations"][key]
        name = titles.get(key, key)
        if not blk["n_battles"]:
            a(f"## {name}")
            a("")
            a("_No battles in this population._")
            a("")
            continue
        a(f"## {name} — n = {blk['n_battles']} battles")
        a("")
        rec = [r.get("recorded_phi_T") for r in blk.get("pre_rows", [])
               if r.get("recorded_phi_T") is not None]
        if rec:
            a(f"_Mean **recorded** phi_T at decision time (each producing run's OWN head, "
              f"on-policy) = **{float(np.mean(rec)):.4f}** over {len(rec)} battles. The subject's "
              f"re-scored `phi_T` below is a DIFFERENT quantity — a later checkpoint reading the "
              f"same states offline — and the two are worth comparing before any pre/post is._")
            a("")
        if blk["compare"]:
            a("| metric | pre | post | paired diff | CI95 | |")
            a("|---|---:|---:|---:|---|---|")
            for m in _HEADLINE:
                r = blk["compare"].get(m)
                if not r:
                    continue
                ci = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" if r["ci"] else "—"
                flag = "**SIG**" if r["significant"] else "n.s."
                a(f"| `{m}` | {r['pre']:.4f} | {r['post']:.4f} | {r['diff']:+.4f} | {ci} | "
                  f"{flag} |")
        else:
            a("| metric | value | CI95 (battle bootstrap) |")
            a("|---|---:|---|")
            for m in _HEADLINE:
                r = blk["levels"].get(m)
                if not r or r["mean"] is None:
                    continue
                a(f"| `{m}` | {r['mean']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] |")
        a("")
    a("## Verdict")
    a("")
    for line in report["verdict"]:
        a(f"- {line}")
    a("")
    a("## Caveats")
    a("")
    for c in report["caveats"]:
        a(f"- {c}")
    a("")
    return "\n".join(L)


def verdict_lines(report: dict) -> List[str]:
    """The read, stated in words, including the failure mode that looks like a success."""
    out: List[str] = []
    pops = report["populations"]
    # Prefer the headline class when it has members; fall back to the wider doomed population.
    stall_key = next((k for k in ("cap", "long_loss", "doomed")
                      if (pops.get(k) or {}).get("compare")), None)
    stall = pops.get(stall_key) or {}
    ctrl = pops.get("control") or {}
    sc, cc = stall.get("compare") or {}, ctrl.get("compare") or {}
    if stall_key:
        out.append(f"Doomed-tail population scored: **{stall_key}** "
                   f"(n = {stall.get('n_battles')} battles).")

    det = sc.get("detect_le05")
    miss = sc.get("miss")
    if det and det["ci"]:
        moved = "IMPROVED" if det["diff"] > 0 and det["significant"] else (
            "DEGRADED" if det["diff"] < 0 and det["significant"] else "did not move")
        out.append(
            f"Detection on held-out stall tails **{moved}**: `detect_le05` "
            f"{det['pre']:.3f} -> {det['post']:.3f} ({det['diff']:+.4f}, CI "
            f"[{det['ci'][0]:+.4f}, {det['ci'][1]:+.4f}], n={det['n']} battles).")
    if miss and miss["ci"]:
        out.append(
            f"Outright misses (`phi_T >= 0.5` on a game lost by construction): "
            f"{miss['pre']:.3f} -> {miss['post']:.3f} ({miss['diff']:+.4f}, CI "
            f"[{miss['ci'][0]:+.4f}, {miss['ci'][1]:+.4f}]).")

    cphi = cc.get("phi_T")
    if cphi and cphi["ci"]:
        broke = cphi["significant"] and cphi["diff"] < -0.05
        out.append(
            f"LONG-WIN control mean `phi_T` {cphi['pre']:.3f} -> {cphi['post']:.3f} "
            f"({cphi['diff']:+.4f}, CI [{cphi['ci'][0]:+.4f}, {cphi['ci'][1]:+.4f}]) — "
            + ("**CONTROL BROKEN: the head has moved toward reading long games as lost. A "
               "detection gain bought this way is a REGRESSION, not a repair.**" if broke else
               "control holds; the detection change is not a blanket late-game pessimism."))
        if det and det["significant"] and det["diff"] > 0 and broke:
            out.append(
                "**FAILED RUN.** Detection improved AND the long-win control collapsed — that is "
                "the degenerate solution ('late means lost'), which scores perfectly on every "
                "stall-tail metric while being strictly worse than the head it replaced.")
    if not any(r and r.get("significant") for r in sc.values()):
        out.append(
            "No stall-tail metric moved significantly. At this n that is **uninformative rather "
            "than negative** — the pilot sizes the pipeline, not the effect.")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.harvest_meter",
        description="Re-run probe O's battery on held-out stall tails, paired pre/post fine-tune.")
    p.add_argument("harvest_dir", help="a harvest output directory (must carry holdout.json)")
    p.add_argument("--head", default=None,
                   help="the fine-tuned win-prob head checkpoint (.pt from winprob_finetune). "
                        "Omit to score the PRE arm only — a baseline read of the battery.")
    p.add_argument("--subject", default=None,
                   help="override the subject checkpoint (default: the one in holdout.json)")
    p.add_argument("--models-root", default=None)
    p.add_argument("--control-n", type=int, default=40)
    p.add_argument("--out", default=None, help="default: <harvest_dir>/meter")
    p.add_argument("--seed", type=int, default=0)
    return p


def graft_head(pm, head_ckpt: str) -> dict:
    """Graft a fine-tuned ``WinProbHead`` into an already-loaded ``ProbeModel``, and PROVE it moved.

    The graft itself is ``winprob_finetune.apply_head`` — the producer of the artifact owns the
    format, and a second loader here would be a second belief about the same file that drifts the
    first time the checkpoint dict gains a key.

    What this adds is the refusal: the head's tensors are snapshotted before and compared after, and
    a graft that changed NOTHING raises. That is not paranoia — a silent no-op makes the post arm a
    bitwise duplicate of the pre arm, and every metric in this meter would then read a perfect,
    perfectly confident null. A measurement instrument must not be able to report "no effect"
    because it failed to apply the treatment.
    """
    import torch

    from agents.training.winprob_finetune import apply_head

    ex = getattr(pm._policy, "features_extractor", None)
    head = getattr(ex, "win_head", None) if ex is not None else None
    if head is None:
        raise RuntimeError(
            "this checkpoint carries no win-prob head (`--win-prob-mode none`) — there is nothing "
            "to graft onto and nothing for the meter to read")
    before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    apply_head(pm, head_ckpt)
    after = getattr(ex, "win_head").state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    if not changed:
        raise RuntimeError(
            f"grafting {head_ckpt} changed NOTHING — the fine-tuned head is bitwise identical to "
            "the subject's. The post arm would be the pre arm and every difference would read as "
            "exactly zero. Refusing to report that as a measurement.")
    return {"tensors_changed": changed, "n_tensors": len(before),
            "max_abs_delta": max(float((after[k] - before[k]).abs().max()) for k in changed)}


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = build_parser().parse_args(argv)
    import torch
    torch.set_num_threads(1)

    from main.prober.model import ProbeModel
    from utils.paths import main_models_dir

    hpath = os.path.join(args.harvest_dir, "holdout.json")
    if not os.path.exists(hpath):
        print(f"harvest_meter: no holdout.json under {args.harvest_dir} — the meter reads the "
              f"split the harvest committed to, it does not invent one", file=sys.stderr)
        return 2
    with open(hpath) as fh:
        holdout = json.load(fh)

    models_root = (args.models_root or holdout.get("models_root")
                   or (str(main_models_dir()) if main_models_dir() else None))
    if not models_root:
        print("harvest_meter: no models root", file=sys.stderr)
        return 2
    subject = args.subject or os.path.join(models_root, holdout["subject_ckpt"])
    out_dir = args.out or os.path.join(args.harvest_dir, "meter")
    os.makedirs(out_dir, exist_ok=True)

    # The held-out doomed tails, SPLIT BY CLASS. They are never pooled: the head reads a cap
    # ending and an ordinary long loss very differently (probe O: detect_le05 0.652 vs 0.94-0.95),
    # so one averaged number would hide the cell the whole exercise is about.
    held = set(holdout.get("holdout_battles", []))
    classes = holdout.get("meter_population") or {}
    groups: Dict[str, List[str]] = {}
    for cls, tags in sorted(classes.items()):
        members = [t for t in tags if t in held]
        if members:
            groups[cls] = [os.path.join(models_root, t) for t in members]
    if not groups:                      # a holdout written before the class map existed
        groups["doomed"] = [os.path.join(models_root, t) for t in sorted(held)]
    groups["control"] = [os.path.join(models_root, t)
                         for t in control_battles(models_root, holdout, n=args.control_n,
                                                  seed=args.seed)]
    print("harvest_meter: " + ", ".join(f"{len(v)} {k}" for k, v in groups.items()), flush=True)

    pm = ProbeModel.load(subject, "cpu")
    pre = {k: score_battles(pm, v)[0] for k, v in groups.items()}
    print("  pre: " + ", ".join(f"{k}={len(v)}" for k, v in pre.items()), flush=True)

    graft: Optional[dict] = None
    post: Dict[str, list] = {k: [] for k in groups}
    if args.head:
        graft = graft_head(pm, args.head)
        print(f"  grafted head: {len(graft['tensors_changed'])}/{graft['n_tensors']} tensors "
              f"changed, max |delta| {graft['max_abs_delta']:.4g}", flush=True)
        post = {k: score_battles(pm, v)[0] for k, v in groups.items()}

    report = {
        "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harvest_dir": os.path.abspath(args.harvest_dir),
        "subject_ckpt": holdout.get("subject_ckpt", subject),
        "head": args.head or "(none — PRE arm only)",
        "k_tail": K_TAIL,
        "populations": {
            k: {"n_battles": len(pre[k]), "pre_rows": pre[k], "post_rows": post[k],
                "levels": levels(pre[k], _HEADLINE),
                "compare": compare(pre[k], post[k], _HEADLINE) if post[k] else {}}
            for k in groups
        },
        "graft": graft,
        "caveats": [
            "The current-arch corpus holds 48 cap battles in TOTAL. Every n here is small and is "
            "printed; a difference whose CI spans zero is reported as not moving, never as a null.",
            "'Pre' is the SUBJECT's re-scored reading, not the trace's recorded win_probs (those "
            "came from whichever checkpoint played the battle). These numbers are therefore NOT "
            "directly comparable to probe O's published levels.",
            "The bootstrap resamples BATTLES. The five decisions inside one tail are not "
            "independent and resampling them would narrow every interval by about sqrt(5).",
            "`detect` is reported as REGISTERED even though probe O showed its 'declining' half "
            "saturates in every class; `detect_le05` is the substantive criterion.",
        ],
    }
    report["verdict"] = verdict_lines(report) if any(post.values()) else [
        "PRE arm only — no fine-tuned head was supplied, so this is a baseline read of the "
        "battery on the held-out split, not a pre/post comparison."]

    with open(os.path.join(out_dir, "meter.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(out_dir, "meter.md"), "w") as fh:
        fh.write(render(report))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
