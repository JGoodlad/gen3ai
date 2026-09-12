#!/usr/bin/env python3
"""The critic ladder's fifteen 10M arms, read on STRENGTH as a family.

Registered in ``PREDICTION.md`` (committed before this script produced a number).

WHAT THIS IS
------------
Every arm of the win-prob critic ladder (`designs/research_state/winprob_critic_ladder_2026-09-08.md`)
was read on CRITIC rows. None was ever read on STRENGTH as a family. This produces that table.

THE INSTRUMENT — and why it is not `main.elo`
---------------------------------------------
* Headline = `<run>/snapshot_ladder/ladder.json`, the dense ±10 ladder, never `eval/elo`
  (UNDERSTANDING.md §3.2 rule 1).
* `python -m main.elo` has **no matched-count cross-run mode**: one positional `run_dir`, and it
  WRITES `<run_dir>/elo/`. Not used.
* `python -m main.critic_gate --at-snapshots N` *does* implement the rule, but PAIRWISE only
  (`run` vs `--parent`). This script calls the library function both of them sit on —
  `agents.training.snapshot_ladder.fit_ladder(run_dir, first_n=N, steps=…, write=False)` — whose
  own docstring is the definition of "matched SNAPSHOT COUNT". `first_n` forces `write=False`, so
  NOTHING under `models/` is written.

MATCHED COUNT = 4, over the COMMON step set
-------------------------------------------
Twelve arms rate exactly {4000032, 6000000, 8000016, 10000032}. Three (`lambda09`, `lambda095`,
`rollout`) rate a fifth, EARLIER node at 2000016 — they promoted at 2M, the self-play crossing
being a per-arm lottery (standing rule 15, 2026-09-10 amendment). A naive `first_n=4` on those
three keeps {2M,4M,6M,8M} and would read their **8M** node against everyone else's **10M**:
matched count, UN-matched steps. So all fifteen are refit on the COMMON four steps. Every arm is
then the newest node of a four-node fit over the identical step set.

NEWEST-NODE INFLATION (§3.2 rule 2)
-----------------------------------
All fifteen runs are FINISHED, and every arm's 10M node occupies the identical structural position
(newest of a four-node fit), so the inflation is common-mode and cancels in a pairwise Δ to first
order. The registered cross-check is the **8M node of the same fit** (second-newest, refit once):
if the ordering and the floor verdict agree, inflation is not driving the read.
"""
from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations

sys.path.insert(0, "src")

from agents.training import lineage as lineage_mod          # noqa: E402
from agents.training import snapshot_ladder as sl           # noqa: E402
from utils.paths import main_models_dir                     # noqa: E402

ARMS = [
    ("vf15",        "ai_v12_10_ladder_vf15",        "--vf-coef 1.5 (capacity: triple the critic's share of the shared trunk)"),
    ("ctrl10M",     "ai_v12_11_ladder_ctrl10M",     "CONTROL — arm A's command at 10M, seed 42"),
    ("cflabels",    "ai_v12_12_ladder_cflabels",    "--cf-records --cf-winprob-coef 0.5 (R-rollout MC win-fraction labels)"),
    ("tdaux",       "ai_v12_13_ladder_tdaux",       "--td-aux-coef 1.0 (Bellman residual in probability space)"),
    ("truevalue",   "ai_v12_14_ladder_truevalue",   "--value-true-team (PRIVILEGED critic — ceiling probe)"),
    ("ctrl10M_b",   "ai_v12_15_ladder_ctrl10M_b",   "CONTROL replicate — seed 1001"),
    ("ctrl10M_c",   "ai_v12_16_ladder_ctrl10M_c",   "CONTROL replicate — seed 1002"),
    ("strata",      "ai_v12_17_ladder_strata",      "--win-prob-strata-weight 1.0 (class-balanced win-prob BCE)"),
    ("lambda09",    "ai_v12_19_ladder_lambda09",    "--win-prob-lambda 0.9 (lambda-return win-prob targets)"),
    ("denseaux",    "ai_v12_20_ladder_denseaux",    "--win-prob-dense-aux 1.0 (dense within-game auxiliary targets)"),
    ("lambda09_b",  "ai_v12_21_ladder_lambda09_b",  "lambda-0.9 RUN-LEVEL replicate — seed 1002"),
    ("lambda095",   "ai_v12_22_ladder_lambda095",   "--win-prob-lambda 0.95 (the middle dose rung)"),
    ("rollout",     "ai_v12_23_ladder_rollout",     "--win-prob-rollout-target --win-prob-rollout-weight 64 (anchored rollout labels)"),
    ("strata_b",    "ai_v12_24_ladder_strata_b",    "strata RUN-LEVEL replicate — seed 1002"),
    ("vf15_b",      "ai_v12_25_ladder_vf15_b",      "vf-1.5 RUN-LEVEL replicate — seed 1001"),
]
CONTROLS = ["ctrl10M", "ctrl10M_b", "ctrl10M_c"]
EXCLUDED = [
    ("ctrl10M_shaped", "ai_v12_26_ladder_ctrl10M_shaped",
     "LIVE on the GPU at read time; no snapshot_ladder/ directory exists. Excluded, not read."),
]
RANDOM_BOT = "random"   # eval-only floor; EXCLUDED from win_rate_vs_bots (eval_callback.py:363)
Z = 1.959963984540054


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def descriptors(run_dir: str) -> dict:
    """The NON-read columns: eval regime (never assumed), bot win rate at 10M, pin, seed."""
    d: dict = {}
    mc = _read_json(os.path.join(run_dir, "model_config.json"))
    # §3.2 rule 5 — the eval regime is RECORDED and INHERITED; read it, never assume it.
    d["eval_sentinel_greedy"] = mc.get("eval_sentinel_greedy", "ABSENT (pre-config-v112)")

    md = _read_json(os.path.join(run_dir, "metadata.json"))
    pins = md.get("pin_history") or []
    d["pin"] = [{"git_hash": p.get("git_hash"), "pin_source": p.get("pin_source"),
                 "first_step": p.get("first_step"), "last_step": p.get("last_step")} for p in pins]
    d["pin_short"] = ",".join((p.get("git_hash") or "?")[:8] for p in pins) or "NONE RECORDED"
    # the SEED comes from the run's own recorded original_command (the immutable block), read
    # through the documented reader — never re-derived
    cmd = lineage_mod.read_original_command(run_dir) or ""
    d["seed_in_command"] = lineage_mod._flag_value(cmd.split(), ("--seed",))
    d["seed_cli_args"] = (md.get("cli_args") or {}).get("seed")
    # a flagless launch takes the parser DEFAULT, which the resolved `cli_args` records; the
    # command is the provenance, `cli_args` is the value the run actually ran at
    d["seed"] = d["seed_in_command"] or d["seed_cli_args"]
    d["seed_source"] = "--seed in original_command" if d["seed_in_command"] else "cli_args (flagless: parser default)"
    d["num_timesteps"] = md.get("num_timesteps")

    # eval/win_rate_vs_bots at the 10M cycle, from the run's own eval_results.jsonl row
    rows = []
    p = os.path.join(run_dir, "eval_results.jsonl")
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    row = max((r for r in rows if r.get("step") is not None), key=lambda r: r["step"])
    bots = row.get("bots") or {}
    counts = row.get("counts") or {}
    named = [(k, v) for k, v in bots.items() if k != RANDOM_BOT]
    wr = sum(v for _, v in named) / len(named) if named else float("nan")
    w = sum(counts[k][0] for k, _ in named if k in counts)
    n = sum(counts[k][1] for k, _ in named if k in counts)
    half = Z * math.sqrt(wr * (1 - wr) / n) if n else float("nan")
    d["eval_row_step"] = row["step"]
    d["win_rate_vs_bots"] = round(wr, 5)
    d["win_rate_vs_bots_n_games"] = n
    d["win_rate_vs_bots_n_bots"] = len(named)
    d["win_rate_vs_bots_ci95"] = [round(wr - half, 4), round(wr + half, 4)]
    d["win_rate_random_excluded"] = bots.get(RANDOM_BOT)
    d["sentinel_regime_row"] = row.get("sentinel_regime")
    d["bots_at_10M"] = bots
    assert w == round(wr * n), "bot win count and mean disagree"
    return d


def main() -> int:
    models = main_models_dir()
    if models is None:
        print("REFUSED: no models/ archive on this box (main_models_dir() is None).")
        return 2
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # ── 0. roster + completeness ───────────────────────────────────────────────────────────
    arms, missing = [], []
    for label, name, lever in ARMS:
        rd = os.path.join(str(models), name)
        lj = os.path.join(rd, "snapshot_ladder", "ladder.json")
        if not os.path.exists(lj):
            missing.append((label, name, "no snapshot_ladder/ladder.json"))
            continue
        arms.append({"label": label, "run": name, "run_dir": rd, "lever": lever,
                     "committed": _read_json(lj)})
    if missing:
        print("REFUSED — an arm in the registered roster has no ladder:", missing)
        return 2
    for label, name, why in EXCLUDED:
        rd = os.path.join(str(models), name)
        present = os.path.exists(os.path.join(rd, "snapshot_ladder", "ladder.json"))
        print(f"[excluded] {label}: {why}"
              + ("  🚨 A LADDER NOW EXISTS — re-read." if present else "  (confirmed absent)"))

    # ── 1. the matched count and the common step set ───────────────────────────────────────
    step_sets = {a["label"]: sorted(int(k) for k in a["committed"]["ratings"]) for a in arms}
    common = sorted(set.intersection(*(set(v) for v in step_sets.values())))
    matched_n = min(len(v) for v in step_sets.values())
    if len(common) != matched_n:
        print(f"REFUSED: matched count {matched_n} but the common step set has {len(common)} "
              f"steps ({common}). A matched-count fit over a non-common step set would compare "
              f"different amounts of training; resolve by hand before reading.")
        return 2
    print(f"[matched] count={matched_n}  common steps={common}")

    # ── 2. the matched-count REFIT — both sides through THIS tree's fit_ladder ─────────────
    for a in arms:
        doc = sl.fit_ladder(a["run_dir"], first_n=matched_n, steps=list(common), write=False)
        assert doc["first_n"] == matched_n and not doc.get("wrote")
        if not doc["converged"]:
            print(f"REFUSED: the matched-count fit of {a['label']} did not converge.")
            return 2
        if len(doc["ratings"]) != matched_n:
            print(f"REFUSED: {a['label']}'s first-{matched_n} prefix rates "
                  f"{len(doc['ratings'])} node(s) — the prefix has no measured edge inside itself.")
            return 2
        a["refit"] = doc
        a["n_nodes_committed"] = len(a["committed"]["ratings"])
        a["dropped_nodes"] = sorted(set(step_sets[a["label"]]) - set(common))
        for row, step in (("headline", common[-1]), ("crosscheck", common[-2])):
            a[row] = {"step": step,
                      "elo": doc["ratings"][str(step)],
                      "se": doc["se"][str(step)],
                      "committed_elo": a["committed"]["ratings"].get(str(step))}
        a["fit_quality"] = doc["fit_quality"]
        a["n_frozen_pairs"] = doc["n_frozen_pairs_measured"]
        a["n_pairs_possible"] = doc["n_pairs_possible"]
        a["eval_sentinel_edges_dropped"] = doc["eval_sentinel_edges_dropped"]
        a["desc"] = descriptors(a["run_dir"])

    by = {a["label"]: a for a in arms}

    # ── 3. THE FLOOR — max pairwise |Δ| over the three controls (standing rule 3) ──────────
    def floor_of(row):
        pairs = []
        for x, y in combinations(CONTROLS, 2):
            dx, dy = by[x][row], by[y][row]
            d = dx["elo"] - dy["elo"]
            se = math.sqrt(dx["se"] ** 2 + dy["se"] ** 2)
            pairs.append({"pair": f"{x} - {y}", "delta": round(d, 1), "abs": round(abs(d), 1),
                          "se": round(se, 1),
                          "ci95": [round(d - Z * se, 1), round(d + Z * se, 1)]})
        return {"pairs": pairs, "floor": max(p["abs"] for p in pairs),
                "mean_abs": round(sum(p["abs"] for p in pairs) / len(pairs), 1)}

    floors = {row: floor_of(row) for row in ("headline", "crosscheck")}

    # ── 4. the deltas + the registered decision rule ───────────────────────────────────────
    def verdict(label, row, floor):
        a = by[label][row]
        ds = []
        for c in CONTROLS:
            cc = by[c][row]
            d = a["elo"] - cc["elo"]
            se = math.sqrt(a["se"] ** 2 + cc["se"] ** 2)
            lo, hi = d - Z * se, d + Z * se
            ds.append({"control": c, "delta": round(d, 1), "se": round(se, 1),
                       "ci95": [round(lo, 1), round(hi, 1)],
                       "abs_gt_floor": abs(d) > floor,
                       # clause 3: the DELTA's own CI must exclude the floor, same direction
                       "ci_clears_floor": (lo > floor) if d > 0 else (hi < -floor)})
        if label in CONTROLS:                    # a control vs itself is a 0 by construction
            ds = [d for d in ds if d["control"] != label]
        clause1 = all(d["abs_gt_floor"] for d in ds)
        clause2 = len({d["delta"] > 0 for d in ds}) == 1
        clause3 = all(d["ci_clears_floor"] for d in ds)
        if clause1 and clause2 and clause3:
            v = "OUTSIDE THE FLOOR (detection)"
        elif clause1 and clause2:
            v = "CANDIDATE (point clears all three, no delta CI does)"
        else:
            v = "WITHIN FLOOR — NOT DETECTED"
        return {"deltas": ds, "clause1_all_three_points": clause1,
                "clause2_same_sign": clause2, "clause3_delta_ci": clause3, "verdict": v}

    for a in arms:
        for row in ("headline", "crosscheck"):
            a[f"verdict_{row}"] = verdict(a["label"], row, floors[row]["floor"])

    # ── 5. the ordering cross-check (Spearman, 10M row vs 8M row) ──────────────────────────
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    h = [a["headline"]["elo"] for a in arms]
    c = [a["crosscheck"]["elo"] for a in arms]
    rh, rc = rank(h), rank(c)
    mh, mc_ = sum(rh) / len(rh), sum(rc) / len(rc)
    num = sum((x - mh) * (y - mc_) for x, y in zip(rh, rc))
    den = math.sqrt(sum((x - mh) ** 2 for x in rh) * sum((y - mc_) ** 2 for y in rc))
    spearman = round(num / den, 4) if den else float("nan")

    doc = {
        "read": "ladder_strength_table_2026-09-12",
        "instrument": "agents.training.snapshot_ladder.fit_ladder(first_n=N, steps=COMMON, write=False)",
        "why_not_main_elo": "main.elo takes one positional run_dir, has no matched-count cross-run "
                            "mode, and writes <run_dir>/elo/. main.critic_gate --at-snapshots "
                            "implements the rule but only pairwise (run vs --parent).",
        "matched_count": matched_n,
        "common_steps": common,
        "headline_step": common[-1],
        "crosscheck_step": common[-2],
        "n_arms": len(arms),
        "excluded": [{"label": l, "run": n, "why": w} for l, n, w in EXCLUDED],
        "floors": floors,
        "spearman_headline_vs_crosscheck": spearman,
        "elo_range_headline": round(max(h) - min(h), 1),
        "elo_range_crosscheck": round(max(c) - min(c), 1),
        "arms": [{k: v for k, v in a.items() if k != "committed"} for a in arms],
        "committed_ladders": {a["label"]: a["committed"] for a in arms},
    }
    with open(os.path.join(out_dir, "ladder_strength.json"), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)

    # ── 6. the human table ─────────────────────────────────────────────────────────────────
    lines = []
    fl = floors["headline"]["floor"]
    lines.append(f"MATCHED COUNT {matched_n} over the COMMON step set {common}")
    lines.append(f"headline node = {common[-1]}   cross-check node = {common[-2]}")
    lines.append(f"FLOOR (headline) = {fl} Elo   |   FLOOR (cross-check) = {floors['crosscheck']['floor']} Elo")
    for p in floors["headline"]["pairs"]:
        lines.append(f"   control pair {p['pair']:<26} Δ {p['delta']:+7.1f}  se {p['se']:.1f}  CI {p['ci95']}")
    lines.append("")
    hdr = (f"{'arm':<12} {'Elo@10M':>8} {'se':>5} {'committed':>9} | "
           f"{'Δ ctrl':>8} {'Δ ctrl_b':>9} {'Δ ctrl_c':>9} {'se(Δ)':>6} {'CI95 halfwidth':>14} | "
           f"{'Elo@8M':>7} | {'wr_bots':>7} {'greedy':>6} {'pin':>9} {'seed':>5}  verdict")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for a in sorted(arms, key=lambda x: -x["headline"]["elo"]):
        v = a["verdict_headline"]
        dd = {d["control"]: d for d in v["deltas"]}
        def f(c):
            return f"{dd[c]['delta']:+.1f}" if c in dd else "  —  "
        se_d = next(iter(dd.values()))["se"]
        de = a["desc"]
        lines.append(
            f"{a['label']:<12} {a['headline']['elo']:>8.1f} {a['headline']['se']:>5.1f} "
            f"{a['headline']['committed_elo']:>9.1f} | {f('ctrl10M'):>8} {f('ctrl10M_b'):>9} "
            f"{f('ctrl10M_c'):>9} {se_d:>6.1f} {'+-%.0f' % (Z*se_d):>14} | "
            f"{a['crosscheck']['elo']:>7.1f} | {de['win_rate_vs_bots']:>7.3f} "
            f"{str(de['eval_sentinel_greedy']):>6} {de['pin_short'][:8]:>9} {str(de['seed']):>5}  "
            f"{v['verdict']}")
    lines.append("")
    lines.append(f"CROSS-CHECK ROW — the {common[-2]} node of the SAME four-node fit "
                 f"(second-newest; floor {floors['crosscheck']['floor']} Elo)")
    hdr2 = (f"{'arm':<12} {'Elo@8M':>8} {'se':>5} | {'Δ ctrl':>8} {'Δ ctrl_b':>9} "
            f"{'Δ ctrl_c':>9} {'se(Δ)':>6} |  verdict")
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for a in sorted(arms, key=lambda x: -x["crosscheck"]["elo"]):
        v = a["verdict_crosscheck"]
        dd = {d["control"]: d for d in v["deltas"]}
        def g(c):
            return f"{dd[c]['delta']:+.1f}" if c in dd else "  —  "
        se_d = next(iter(dd.values()))["se"]
        lines.append(f"{a['label']:<12} {a['crosscheck']['elo']:>8.1f} {a['crosscheck']['se']:>5.1f} | "
                     f"{g('ctrl10M'):>8} {g('ctrl10M_b'):>9} {g('ctrl10M_c'):>9} {se_d:>6.1f} |  "
                     f"{v['verdict']}")
    lines.append("")
    lines.append(f"Elo range across the fifteen arms: {doc['elo_range_headline']} (10M) / "
                 f"{doc['elo_range_crosscheck']} (8M)")
    lines.append(f"Spearman rank corr (10M row vs 8M row, n=15): {spearman}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out_dir, "ladder_strength_table.txt"), "w") as fh:
        fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
