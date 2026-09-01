"""M1 — THE AXIS SPLIT: is fold quality on TAUGHT teams a different function of fleet SHAPE
than the externality on UNTAUGHT teams?

Pure assembly + statistics over meters that already exist plus the three arms measured by
`axis_split_untaught_arm.py`. No models are loaded, no battles are played here.

THE THREE METERS, all on ONE instrument family (fixed target = rev-1 @24M snapshot, opponent
draws from the validated 719-pool, stochastic both sides, rust bridge):

  TAUGHT-9   `pilot_<arm>_n300.json`  — 9 meter teams x 300 games. Taught by EVERY fold here.
  TAUGHT-COV `cov_<arm>.json`         — 3 coverage teams x 300. Taught by rev-3/COMPFOLD/rev-4;
                                        UNTAUGHT by rev-2 — so rev-2's row on this cut is an
                                        externality reading, not a fold-quality one.
  UNTAUGHT-8 `untaught_<arm>.json`    — 8 teams x 200 that NO fold in the table ever pinned.

Shape is DERIVED from recorded run metadata (fold -> `distill_teacher` -> each teacher's
`trainee_teams`), never hand-copied, and the untaught set's untaughtness is ASSERTED against
every derived taught union rather than assumed.

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):
  python axis_split_taught_untaught.py --out <stem> [--probe-dir DIR] [--models-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

BOOT = 20000
BOOT_SEED = 20260831

# ---------------------------------------------------------------- shape derivation


def _orig_flag(meta: dict, flag: str) -> str | None:
    cmd = meta.get("original_command") or ""
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
    return m.group(1).strip("'\"") if m else None


def _run_meta(models: Path, run: str) -> dict:
    return json.loads((models / run / "metadata.json").read_text())


def _run_steps(models: Path, run: str) -> int | None:
    s = _orig_flag(_run_meta(models, run), "--steps")
    return int(s) if s else None


def _teams_of(models: Path, run: str) -> list[str]:
    """The teacher's recorded --trainee-teams, as bare team ids."""
    ca = _run_meta(models, run).get("cli_args", {})
    tt = ca.get("trainee_teams")
    if isinstance(tt, str):
        tt = tt.split(",")
    return [os.path.basename(t).replace(".txt", "") for t in (tt or []) if t]


def derive_shape(models: Path, fold_run: str) -> dict:
    """(teachers, distilled teams, trained teams, budget/team) for one FOLD run."""
    meta = _run_meta(models, fold_run)
    spec = _orig_flag(meta, "--distill-teacher") or meta.get("cli_args", {}).get("distill_teacher") or ""
    spec = spec.strip("'\"")
    teachers: dict[str, list[str]] = {}
    trained: dict[str, list[str]] = {}
    for chunk in [c for c in spec.split(";") if c.strip()]:
        path, _, sel = chunk.partition(":")
        run = Path(path.strip()).parts[1] if path.strip().startswith("models/") else Path(path.strip()).parts[0]
        tr = _teams_of(models, run)
        trained[run] = tr
        if sel.strip() == "*" or not sel.strip():
            teachers[run] = tr
        else:
            teachers[run] = [os.path.basename(t).replace(".txt", "") for t in sel.split(",") if t]
    distilled = sorted({t for v in teachers.values() for t in v})
    trained_all = sorted({t for v in trained.values() for t in v})
    # per-team budget: teacher fork length / teams the TEACHER trained on.
    # CUMULATIVE, because a teacher may itself be a fork of an earlier exploiter on the same
    # teams (REFOLD1's REVIVE teachers continue rev-4's R4S3 teachers) — charging only the last
    # leg would report 0.5M/team for a teacher that has seen 1.75M/team.
    budgets, budgets_last = [], []
    for run in teachers:
        legs, cur = [], run
        for _ in range(8):
            tm = _run_meta(models, cur)
            if not _teams_of(models, cur):  # not an exploiter run — the chain ends here
                break
            steps = _orig_flag(tm, "--steps")
            parent = _orig_flag(tm, "--model")
            if not steps or not parent:
                break
            prun = Path(parent).parts[-2] if "/" in parent else None
            psteps = _run_steps(models, prun) if prun else None
            if not psteps:
                break
            legs.append((int(steps) - psteps) / max(1, len(_teams_of(models, cur))))
            cur = prun
        if legs:
            budgets.append(sum(legs))
            budgets_last.append(legs[0])
    return {
        "fold_run": fold_run,
        "teachers": sorted(teachers),
        "n_teachers": len(teachers),
        "distilled_teams": distilled,
        "n_distinct_taught": len(distilled),
        "teams_per_teacher_distilled": round(sum(len(v) for v in teachers.values()) / max(1, len(teachers)), 3),
        "teams_per_teacher_trained": round(sum(len(v) for v in trained.values()) / max(1, len(trained)), 3),
        "trained_union": trained_all,
        "budget_per_trained_team_M": round(sum(budgets) / len(budgets) / 1e6, 3) if budgets else None,
        "budget_last_leg_per_team_M": round(sum(budgets_last) / len(budgets_last) / 1e6, 3) if budgets_last else None,
        "fork_parent": _orig_flag(meta, "--model"),
        "target": _orig_flag(meta, "--exploiter"),
    }


# ---------------------------------------------------------------- statistics


def _rows(d: dict) -> dict[str, tuple[int, int]]:
    return {k: (v["wins"], v["games"]) for k, v in d.items() if isinstance(v, dict) and "wins" in v and k != "POOLED"}


def contrast(arm: dict, base: dict) -> dict:
    """Equal-weight per-team mean difference, cluster-bootstrapped over TEAMS.

    The two arms are NOT battle-paired (both sides act stochastically and the sim dice are
    free); they share only the per-team opponent-team draw SEQUENCE. So the per-team
    difference is an unpaired two-proportion contrast and the pooled z sums both variances.
    """
    a, b = _rows(arm), _rows(base)
    teams = sorted(set(a) & set(b))
    per = {}
    diffs, var = [], 0.0
    for t in teams:
        wa, na = a[t]
        wb, nb = b[t]
        pa, pb = wa / na, wb / nb
        per[t] = {
            "arm_wr": round(pa, 4),
            "base_wr": round(pb, 4),
            "delta_pp": round(100 * (pa - pb), 2),
            "n": [na, nb],
        }
        diffs.append(pa - pb)
        var += pa * (1 - pa) / na + pb * (1 - pb) / nb
    k = len(teams)
    mean = sum(diffs) / k
    se = math.sqrt(var) / k
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        s = [diffs[rng.randrange(k)] for _ in range(k)]
        boots.append(sum(s) / k)
    boots.sort()
    return {
        "teams": k,
        "delta_pp": round(100 * mean, 3),
        "ci_pp_cluster": [round(100 * boots[int(0.025 * BOOT)], 3), round(100 * boots[int(0.975 * BOOT)], 3)],
        "z_battle": round(mean / se, 3) if se else None,
        "n_negative": sum(1 for d in diffs if d < 0),
        "arm_wr": round(sum(a[t][0] / a[t][1] for t in teams) / k, 4),
        "base_wr": round(sum(b[t][0] / b[t][1] for t in teams) / k, 4),
        "per_team": per,
    }


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# ---------------------------------------------------------------- main


FOLDS = {
    # label            fold run                       parent arm tag
    "rev-2": ("ai_v9_59_R2ACTION_0827", "rev1fin", "R2ACTION"),
    "rev-3": ("ai_v9_70_R3ACTION_0828", "R2ACTION", "R3ACTION"),
    "COMPFOLD": ("ai_v9_91_COMPFOLD_0831", "R2ACTION", "COMPFOLD"),
    "rev-4": ("ai_v9_76_R4ACTION_0830", "R2ACTION", "R4ACTION"),
    "REFOLD1": ("ai_v9_82_REFOLD1_0830", "R2ACTION", "REFOLD1"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", default=str(here / "axis_split_taught_untaught_2026-08-31"))
    ap.add_argument("--probe-dir", default=str(here / "axis_split_inputs"))
    ap.add_argument("--models-dir", default="/home/goodlad/dev/gen3ai/models")
    args = ap.parse_args()
    P, M = Path(args.probe_dir), Path(args.models_dir)

    def load(name: str) -> dict | None:
        p = P / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    out: dict = {"probe": "M1 axis split — taught vs untaught fold externality", "date": "2026-08-31"}

    # ---- shape, derived
    shapes = {lbl: derive_shape(M, run) for lbl, (run, _, _) in FOLDS.items()}
    out["shape"] = shapes

    # ---- the untaught set really is untaught
    untaught_ids = json.loads((P / "untaught_teams.json").read_text())
    leaks = {
        lbl: sorted(set(untaught_ids) & set(s["trained_union"]) | set(untaught_ids) & set(s["distilled_teams"]))
        for lbl, s in shapes.items()
    }
    assert not any(leaks.values()), f"untaught set is contaminated: {leaks}"
    out["untaught_set"] = {"teams": untaught_ids, "leak_check": leaks}

    # ---- what each CUT is, structurally, for each fold — asserted, not assumed.
    # The 9-slice meter is rev-2's own taught set; the 3 coverage teams are what rev-3's fleet
    # ADDED. Both facts are derived above, so they are checked rather than believed.
    meter9 = set(shapes["rev-2"]["distilled_teams"])
    cov3 = set(shapes["rev-3"]["distilled_teams"]) - meter9
    assert len(meter9) == 9, f"9-slice meter is not 9 teams: {sorted(meter9)}"
    assert len(cov3) == 3, f"coverage cut is not 3 teams: {sorted(cov3)}"
    membership = {}
    for lbl, s in shapes.items():
        taught = set(s["distilled_teams"])
        membership[lbl] = {
            "meter9_taught": sorted(meter9 & taught) == sorted(meter9),
            "cov3_taught": sorted(cov3 & taught) == sorted(cov3),
            "untaught8_taught": bool(set(untaught_ids) & taught),
        }
    assert all(m["meter9_taught"] for m in membership.values()), membership
    assert not membership["rev-2"]["cov3_taught"], "rev-2 was expected NOT to teach the coverage cut"
    assert all(membership[k]["cov3_taught"] for k in ("rev-3", "COMPFOLD", "rev-4")), membership
    out["cut_membership"] = {
        "meter9": sorted(meter9),
        "cov3": sorted(cov3),
        "per_fold": membership,
        "note": "cov3 is TAUGHT for rev-3/COMPFOLD/rev-4 and UNTAUGHT for rev-2 — rev-2's cov row "
        "is an externality reading (the original -5.9pp treadmill), not a fold-quality one.",
    }

    # ---- the three cuts
    cuts = {"taught_9slice": "pilot_%s_n300", "taught_cov3": "cov_%s", "untaught_8": "untaught_%s"}
    results: dict[str, dict] = {}
    for cut, pat in cuts.items():
        results[cut] = {}
        for lbl, (_, parent, tag) in FOLDS.items():
            arm = load(pat % tag) or load(pat.replace("_n300", "") % tag)
            base = load(pat % parent) or load(pat.replace("_n300", "") % parent)
            if arm is None or base is None:
                results[cut][lbl] = {"status": "MISSING", "arm": arm is not None, "base": base is not None}
                continue
            results[cut][lbl] = contrast(arm, base)
    out["cuts"] = results

    # ---- v8, on a DIFFERENT instrument (probe P). Carried, flagged, never pooled.
    v8p = here / "v8_redistribution_pfsp_2026-08-30.json"
    if v8p.exists():
        v8 = json.loads(v8p.read_text())
        out["v8_cross_instrument"] = {
            "untaught": v8["untaught"],
            "taught": v8["taught"],
            "shape": {
                "n_teachers": 3,
                "n_distinct_taught": 22,
                "teams_per_teacher_distilled": round(22 / 3, 3),
                "note": "probe P archaeology: 3 teachers, 23 team paths (2 identical) = 22 distinct",
            },
            "instrument": "probe P — parent-vs-fold, fixed ancestor opponent, CRN-paired, greedy. "
            "NOT the fixed-rev-1-target meter used for rev-2/3/4/COMPFOLD.",
        }

    # ---- the SHAPE contrasts, arm vs arm on each cut.
    # These are the numbers the fleet-shape decision actually rests on, and only the first is
    # a single-variable contrast: COMPFOLD reuses rev-4's own teacher checkpoints, so teachers,
    # teacher training and per-team budget are held EXACTLY fixed and only the distilled team
    # list moves. rev-3 vs COMPFOLD moves three things at once.
    shape_contrasts = {
        "team_count_12_vs_24 (COMPFOLD - rev-4) [CLEAN: same teachers, same budget]": ("COMPFOLD", "R4ACTION"),
        "teacher_count_6_vs_3 (rev-3 - COMPFOLD) [CONFOUNDED: teachers x breadth x budget]": ("R3ACTION", "COMPFOLD"),
        "budget_1.76_vs_1.26 (REFOLD1 - rev-4) [CLEAN: identical 3x8 shape]": ("REFOLD1", "R4ACTION"),
    }
    out["shape_contrasts"] = {}
    for name, (a_tag, b_tag) in shape_contrasts.items():
        row = {}
        for cut, pat in cuts.items():
            a = load(pat % a_tag) or load(pat.replace("_n300", "") % a_tag)
            b = load(pat % b_tag) or load(pat.replace("_n300", "") % b_tag)
            if a is None or b is None:
                row[cut] = "MISSING"
                continue
            c = contrast(a, b)
            row[cut] = {k: c[k] for k in ("teams", "delta_pp", "ci_pp_cluster", "z_battle", "n_negative")}
        out["shape_contrasts"][name] = row

    # ---- what the RUNNING 40-team fleet can and cannot be scored on
    r5p = P / "r5_fleet_teams.json"
    if r5p.exists():
        r5 = json.loads(r5p.read_text())
        r5_teams = sorted({t for k, v in r5.items() if k != "_meta" for t in v})
        r5_cuts = {
            "meter9": sorted(set(r5_teams) & meter9),
            "cov3": sorted(set(r5_teams) & cov3),
            "untaught8": sorted(set(r5_teams) & set(untaught_ids)),
            "rev4_extra12": sorted(set(r5_teams) & (set(shapes["rev-4"]["distilled_teams"]) - meter9 - cov3)),
        }
        out["rev5_scoring"] = {
            "n_arms": len([k for k in r5 if k != "_meta"]),
            "n_distinct_teams": len(r5_teams),
            "overlap_with_existing_cuts": r5_cuts,
            "verdict": {
                "meter9_is_taught_for_rev5": bool(r5_cuts["meter9"]),
                "cov3_is_taught_for_rev5": bool(r5_cuts["cov3"]),
                "untaught8_still_untaught": not r5_cuts["untaught8"],
            },
        }

    # ---- the two orderings
    rows = []
    for lbl in FOLDS:
        s = shapes[lbl]
        t9 = results["taught_9slice"].get(lbl, {})
        tc = results["taught_cov3"].get(lbl, {})
        u8 = results["untaught_8"].get(lbl, {})
        rows.append(
            {
                "fold": lbl,
                "teachers": s["n_teachers"],
                "teams_per_teacher_distilled": s["teams_per_teacher_distilled"],
                "teams_per_teacher_trained": s["teams_per_teacher_trained"],
                "n_distinct_taught": s["n_distinct_taught"],
                "budget_per_trained_team_M": s["budget_per_trained_team_M"],
                "budget_last_leg_per_team_M": s["budget_last_leg_per_team_M"],
                "taught_9slice_delta_pp": t9.get("delta_pp"),
                "taught_9slice_ci": t9.get("ci_pp_cluster"),
                "taught_cov3_delta_pp": tc.get("delta_pp"),
                "taught_cov3_ci": tc.get("ci_pp_cluster"),
                "untaught_delta_pp": u8.get("delta_pp"),
                "untaught_ci": u8.get("ci_pp_cluster"),
                "untaught_z": u8.get("z_battle"),
            }
        )
    out["rows"] = rows

    # rank correlations among the folds that have BOTH axes on THIS instrument
    both = [r for r in rows if r["taught_9slice_delta_pp"] is not None and r["untaught_delta_pp"] is not None]
    out["ordering"] = {
        "n_points": len(both),
        "taught_9slice_order": [r["fold"] for r in sorted(both, key=lambda r: -r["taught_9slice_delta_pp"])],
        "taught_cov3_order": [
            r["fold"] for r in sorted([b for b in both if b["taught_cov3_delta_pp"] is not None], key=lambda r: -r["taught_cov3_delta_pp"])
        ],
        "untaught_order": [r["fold"] for r in sorted(both, key=lambda r: -r["untaught_delta_pp"])],
    }
    if len(both) >= 3:
        out["ordering"]["spearman_taught9_vs_untaught"] = round(
            spearman([r["taught_9slice_delta_pp"] for r in both], [r["untaught_delta_pp"] for r in both]), 4
        )
        out["ordering"]["spearman_cov3_vs_untaught"] = round(
            spearman(
                [r["taught_cov3_delta_pp"] for r in both if r["taught_cov3_delta_pp"] is not None],
                [r["untaught_delta_pp"] for r in both if r["taught_cov3_delta_pp"] is not None],
            ),
            4,
        )
        for var in ("teachers", "teams_per_teacher_distilled", "n_distinct_taught", "budget_per_trained_team_M"):
            xs = [r[var] for r in both]
            if any(x is None for x in xs):
                continue
            out["ordering"][f"spearman_{var}_vs_untaught"] = round(
                spearman(xs, [r["untaught_delta_pp"] for r in both]), 4
            )
            out["ordering"][f"spearman_{var}_vs_taught9"] = round(
                spearman(xs, [r["taught_9slice_delta_pp"] for r in both]), 4
            )

    Path(args.out + ".json").write_text(json.dumps(out, indent=1))

    # ---- the tables, generated (never hand-transcribed into the report)
    def fmt(v, ci=None, z=None):
        if v is None:
            return "*(not measured)*"
        s = f"**{v:+.2f}**"
        if ci:
            s += f" [{ci[0]:+.2f}, {ci[1]:+.2f}]"
        if z is not None:
            s += f" z={z:+.2f}"
        return s

    lines = ["| fold | teachers | teams/teacher (distilled·trained) | distinct taught | budget/team (M) | TAUGHT-9 Δpp | TAUGHT-COV3 Δpp | UNTAUGHT-8 Δpp |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['teachers']} | {r['teams_per_teacher_distilled']:g}·{r['teams_per_teacher_trained']:g} "
            f"| {r['n_distinct_taught']} | {r['budget_per_trained_team_M']} "
            f"| {fmt(r['taught_9slice_delta_pp'], r['taught_9slice_ci'])} "
            f"| {fmt(r['taught_cov3_delta_pp'], r['taught_cov3_ci'])} "
            f"| {fmt(r['untaught_delta_pp'], r['untaught_ci'], r['untaught_z'])} |"
        )
    tables = "\n".join(lines)
    per_team = []
    for cut in cuts:
        for lbl in FOLDS:
            c = results[cut].get(lbl, {})
            if "per_team" not in c:
                continue
            per_team.append(f"\n**{cut} · {lbl}** (arm {c['arm_wr']:.4f} vs base {c['base_wr']:.4f}, "
                            f"{c['n_negative']}/{c['teams']} teams negative)")
            per_team.append("| team | arm | base | Δpp |")
            per_team.append("|---|---|---|---|")
            for t, v in c["per_team"].items():
                per_team.append(f"| `{t}` | {v['arm_wr']:.3f} | {v['base_wr']:.3f} | {v['delta_pp']:+.1f} |")
    Path(args.out + "_tables.md").write_text(tables + "\n" + "\n".join(per_team) + "\n")
    print(tables)
    print("\n" + json.dumps(out["ordering"], indent=1))
    print(f"\nwrote {args.out}.json and {args.out}_tables.md")


if __name__ == "__main__":
    main()
