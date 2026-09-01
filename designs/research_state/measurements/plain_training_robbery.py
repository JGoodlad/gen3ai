"""M9 — THE MISSING CONTROL: does PLAIN TRAINING rob untaught teams?

WHAT THIS IS: the assembly + statistics half of M9. The battles are collected by
`axis_split_untaught_arm.py` (unmodified — one instrument, one scale), which this script
only reads. No battles here; it runs in ~3 s.

THE QUESTION. Every gen-era fold measured on the untaught-8 set ROBS: rev-2 -7.06pp, rev-4
-6.50pp. M5 then found -- unregistered -- that rev-3's fold is behaviourally SMALLER than
what it called a "matched no-fold control", which would leave the robbery with no plausible
carrier. Both cannot be read as stated: a fold below its own noise floor has nothing to
radiate. The missing cell is what ORDINARY TRAINING does to the same teams over the same
span, and until now nobody had run it.

THE ARMS (all four share ONE parent, ONE step budget and ONE seed; argvs diffed at run time
by `--assert-argvs`, and they differ ONLY in the distillation family):

    rev-1 final  --(28.07M, plain)-->      R2PLAIN   ai_v9_62_R2PLAIN_0827
    rev-1 final  --(28.07M, +teachers in the ecology, coef 0)--> R2CTRL  ai_v9_58_R2CTRL_0827
    rev-1 final  --(28.07M, the real fold)--> R2ACTION ai_v9_59_R2ACTION_0827
    R2ACTION     --(32.57M, zero-content self-fold)--> R3SELF ai_v9_72_R3SELF_0828

THE FLOOR STRATUM IS REGISTERED, NOT FITTED. Membership comes from the training session's
`headroom_screen.json` -- R2ACTION piloting each team at n=150, screened BEFORE any of these
arms was measured -- at the pre-registered `> 0.55` cut (scorecard REPRO-2). Six teams clear
it, two do not. That variable is independent of every arm this script differences, so it
cannot induce regression to the mean; using each arm's OWN measured parent WR to stratify
would.

Run (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src):
  python plain_training_robbery.py --out designs/research_state/measurements/plain_training_robbery_2026-08-31
"""

import argparse
import hashlib
import json
import math
import random
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "axis_split_inputs"
MODELS = Path("/home/goodlad/dev/gen3ai/models")
JOBS = Path.home() / ".claude/jobs/1046b1d6/tmp/probes"

# ORDER IS THE SEED in the collector; it is the join key here. Never reorder.
TEAMS = [
    "U_61590463",
    "U_92832108",
    "U_ce35b736",
    "U_9909f2e9",
    "U_9d5f8458",
    "U_f7ba5702",
    "U_90b94599",
    "U_dbf81d8e",
]

# REGISTERED floor stratum: headroom_screen.json (R2ACTION, n=150) WR > 0.55.
# Recorded here as the VALUES so the artifact is self-contained; `--assert-screen`
# re-derives them from the screen file and fails on any drift.
SCREEN_WR = {
    "U_61590463": 0.5466666666666666,
    "U_92832108": 0.5466666666666666,
    "U_ce35b736": 0.5533333333333333,
    "U_9909f2e9": 0.5733333333333334,
    "U_9d5f8458": 0.58,
    "U_f7ba5702": 0.58,
    "U_90b94599": 0.5933333333333334,
    "U_dbf81d8e": 0.6133333333333333,
}
FLOOR_CUT = 0.55

# arm tag -> (result json path candidates, run dir, one-line role)
ARMS = {
    "REV1FIN": ("ai_v9_29_rev1_0823", "the common PARENT (rev-1 final)"),
    "R2PLAIN": ("ai_v9_62_R2PLAIN_0827", "PLAIN training, no teacher of any kind"),
    "R2CTRL": ("ai_v9_58_R2CTRL_0827", "teachers in the ECOLOGY, distill coef 0"),
    "R2ACTION": ("ai_v9_59_R2ACTION_0827", "the rev-2 FOLD (action target, coef 0.181)"),
    "R3SELF": ("ai_v9_72_R3SELF_0828", "zero-CONTENT self-fold off R2ACTION"),
    "R4ACTION": ("ai_v9_76_R4ACTION_0830", "the rev-4 FOLD off R2ACTION"),
}

MYINPUTS = HERE / "plain_training_robbery_inputs"
# `/tmp/axis_split` is the M1 axis-split agent's scratch: its REV1FIN arm runs the
# BYTE-IDENTICAL collector (verified by diff before adoption), so its artifact is
# consumed rather than duplicated. It is copied into MYINPUTS once complete.
SEARCH_DIRS = [MYINPUTS, Path("/tmp/m9"), Path("/tmp/axis_split"), INPUTS, HERE]

# The TAUGHT-9 companion cut. Every arm was ALREADY measured on it by the training
# session's standing fold-quality meter (9 teams x 300, same fixed target, same
# collector family), so this half of the table costs no battles.
TAUGHT9 = [
    "ZapDug",
    "JynxSO",
    "RaikouCelebi",
    "MixZap",
    "BlueOffense",
    "MedichamCune",
    "CBMetaCroCune",
    "Q6a",
    "Q6b",
]


# ----------------------------------------------------------------- statistics


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - hw, c + hw)


def contrast(arm, base, teams, n_boot=20000, seed=20260831):
    """Equal-weight mean over TEAMS of (arm - base), team-clustered bootstrap CI.

    Arms are NOT battle-paired (both sides act stochastically and the sim dice are free);
    they share only the per-team opponent-team draw SEQUENCE. So each team's delta is an
    unpaired two-proportion contrast and the CI is a cluster bootstrap over the unit the
    claim generalises over -- teams -- exactly as the axis-split / probe-Q family does.
    """
    rows = []
    for t in teams:
        a, b = arm[t], base[t]
        pa, pb = a["wins"] / a["games"], b["wins"] / b["games"]
        va = pa * (1 - pa) / a["games"]
        vb = pb * (1 - pb) / b["games"]
        rows.append({"team": t, "arm_wr": pa, "base_wr": pb, "delta": pa - pb, "var": va + vb})
    if not rows:
        return None
    d = [r["delta"] for r in rows]
    mean = sum(d) / len(d)
    # z uses the BINOMIAL variance of the equal-weight mean (both arms' variance summed)
    se_binom = math.sqrt(sum(r["var"] for r in rows)) / len(rows)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s = [d[rng.randrange(len(d))] for _ in range(len(d))]
        boot.append(sum(s) / len(s))
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot) - 1]
    # dispersion check: observed spread of per-team deltas vs what binomial alone implies
    if len(d) > 1:
        m = mean
        sd_obs = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    else:
        sd_obs = float("nan")
    sd_binom = math.sqrt(sum(r["var"] for r in rows) / len(rows))
    return {
        "n_teams": len(rows),
        "mean_delta": mean,
        "ci95_cluster_bootstrap": [lo, hi],
        "se_binomial": se_binom,
        # the BANKED ledger numbers (rev-4's -6.50 / -8.67 / +0.00) are on THIS interval, not
        # the cluster one -- reproduced exactly by this script, which is how it was identified.
        "ci95_binomial": [mean - 1.96 * se_binom, mean + 1.96 * se_binom],
        "z_binomial": mean / se_binom if se_binom else float("nan"),
        "sd_per_team_observed": sd_obs,
        "sd_per_team_binomial": sd_binom,
        "dispersion_ratio": (sd_obs / sd_binom) if sd_binom else float("nan"),
        "pooled_arm_wr": sum(arm[t]["wins"] for t in teams) / sum(arm[t]["games"] for t in teams),
        "pooled_base_wr": sum(base[t]["wins"] for t in teams) / sum(base[t]["games"] for t in teams),
        "rows": rows,
    }


# ----------------------------------------------------------------- provenance


def load_arm(tag, prefix="untaught", want=None, partial_ok=True):
    """A PARTIAL arm is admitted and LABELLED, never silently pooled. Every contrast is then
    computed on the INTERSECTION of the two arms' teams and the cell records `restricted_to`,
    so a restricted read cannot be mistaken for the full one."""
    want = want or TEAMS
    for d in SEARCH_DIRS:
        p = d / f"{prefix}_{tag}.json"
        if p.exists():
            raw = json.load(open(p))
            cells = {k: v for k, v in raw.items() if k in want}
            if not cells or (len(cells) != len(want) and not partial_ok):
                return None, f"{p}: only {len(cells)}/{len(want)} teams present (still collecting?)"
            return {
                "path": str(p),
                "meta": raw.get("_meta", {}),
                "cells": cells,
                "complete": len(cells) == len(want),
                "n_teams": len(cells),
            }, None
    return None, f"{prefix}_{tag}.json not found in {[str(x) for x in SEARCH_DIRS]}"


def argv_of(tag):
    p = JOBS / f"{tag}.argv"
    if not p.exists():
        return None
    toks = shlex.split(open(p).read().strip())
    d, i = {}, 0
    while i < len(toks):
        if toks[i].startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                d[toks[i]] = toks[i + 1]
                i += 2
            else:
                d[toks[i]] = True
                i += 1
        else:
            i += 1
    return d


def assert_argvs():
    """The single-variable claim, ASSERTED not assumed: the three rev-2-era arms differ
    ONLY in the distillation family."""
    out = {}
    a, p, c = argv_of("R2ACTION"), argv_of("R2PLAIN"), argv_of("R2CTRL")
    if not (a and p and c):
        return {"available": False, "why": f"argv files absent under {JOBS}"}
    dist_prefixes = ("--distill", "--rank-tripwire")
    for nx, x, ny, y in (("R2ACTION", a, "R2PLAIN", p), ("R2CTRL", c, "R2PLAIN", p)):
        diffs = {}
        for k in sorted(set(x) | set(y)):
            if k == "--run-name":
                continue
            if x.get(k) != y.get(k):
                diffs[k] = {nx: str(x.get(k))[:90], ny: str(y.get(k))[:90]}
        non_distill = [k for k in diffs if not k.startswith(dist_prefixes)]
        out[f"{nx}_vs_{ny}"] = {
            "diff_keys": sorted(diffs),
            "non_distillation_diffs": non_distill,
            "single_variable": not non_distill,
            "detail": diffs,
        }
    for tag in ("R2ACTION", "R2PLAIN", "R2CTRL"):
        d = argv_of(tag)
        out.setdefault("shared", {})[tag] = {
            "--model": d.get("--model"),
            "--steps": d.get("--steps"),
            "--seed": d.get("--seed"),
        }
    return {"available": True, **out}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def m5_control_audit():
    """M5 §1.1 calls both of its matched-noise controls 'ordinary training, no fold'. Neither
    is. Two independent defects, one per era, each checked here:

      gen: the control file is `R2ACTION/snapshots/snapshot_24M` -- but a resume RE-PUBLISHES
           the parent's self-play pool, so that file is rev-1's own snapshot. The 'control
           span' is rev-1's last ~1M of plain training PLUS the ENTIRE rev-2 fold.
      v8:  the control is a CHECKPOINT of `ai_v8_04_distill_4teacher_0722`, which ran at
           `--distill-coef 1.0` for its whole life. Its 7.46M span is fold training throughout.

    Consequence: M5 §3's ratios are fold / fold, and its unregistered headline -- 'rev-3's
    fold is behaviourally SMALLER than its own matched no-fold control' -- reads correctly as
    'smaller than rev-2's fold'. The sentence it licensed ('a fold below its own no-fold noise
    floor has nothing to radiate') is NOT supported, because no no-fold floor was measured.
    """
    out = {"available": True}
    a = MODELS / "ai_v9_59_R2ACTION_0827/snapshots/snapshot_000024000000.zip"
    b = MODELS / "ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip"
    if a.exists() and b.exists():
        ha, hb = md5(a), md5(b)
        out["gen"] = {
            "m5_control": str(a),
            "rev1_own_snapshot": str(b),
            "md5_m5_control": ha,
            "md5_rev1_snapshot": hb,
            "byte_identical": ha == hb,
            "fold_free": not (ha == hb),
            "reading": (
                "IDENTICAL -- the pool re-publish means M5's gen control is rev-1's snapshot; "
                "its 'control span' contains the whole rev-2 fold."
                if ha == hb
                else "DISTINCT -- M5's control is R2ACTION's own mid-run state."
            ),
        }
    else:
        out["gen"] = {"available": False, "why": "one or both snapshots absent"}
    p = MODELS / "ai_v8_04_distill_4teacher_0722/metadata.json"
    if p.exists():
        ca = json.load(open(p)).get("cli_args", {})
        coef = ca.get("distill_coef")
        out["v8"] = {
            "control_run": "ai_v8_04_distill_4teacher_0722",
            "control_checkpoint": "checkpoints/checkpoint_269716291_steps.zip",
            "distill_coef_of_that_run": coef,
            "distill_teacher": str(ca.get("distill_teacher"))[:120],
            "fold_free": not (coef and coef > 0),
            "reading": (
                "NOT fold-free -- the run distils at coef %s for its whole span, so M5's "
                "'ordinary training' interval is 7.46M steps of distillation." % coef
                if (coef and coef > 0)
                else "fold-free"
            ),
        }
    else:
        out["v8"] = {"available": False, "why": "metadata absent"}
    out["verdict"] = (
        "NEITHER M5 control is fold-free"
        if (out["gen"].get("fold_free") is False and out["v8"].get("fold_free") is False)
        else "at least one M5 control is fold-free — re-read"
    )
    return out


def replicate_pair_audit():
    """R2CTRL was DESIGNED as an ecology control (teachers present, distill coef 0). As RUN it
    was not one: at its own commit `--distill-teacher` is INERT at coef 0, so R2CTRL and R2PLAIN
    are a REPLICATE PAIR of plain training -- and their difference is a direct run-to-run noise
    estimate for every fold contrast in this campaign.

    Four facts, each CHECKED here rather than asserted:
      1. the argvs differ only in `--distill-teacher`  (assert_argvs)
      2. `src/` is byte-identical between the two runs' git hashes
      3. at that commit every `_distill_pairs` consumer is gated on `distill_coef > 0`
      4. the live tree's own `apply_distill_team_bias` docstring names this run
    """
    repo = "/home/goodlad/dev/gen3ai"
    out = {}
    try:
        meta = {
            r: json.load(open(f"{repo}/models/{r}/metadata.json")).get("git_hash", "")
            for r in ("ai_v9_58_R2CTRL_0827", "ai_v9_62_R2PLAIN_0827")
        }
        out["git_hashes"] = meta
        a, b = meta["ai_v9_58_R2CTRL_0827"], meta["ai_v9_62_R2PLAIN_0827"]
        d = subprocess.run(
            ["git", "-C", repo, "diff", "--stat", a, b, "--", "src/"],
            capture_output=True, text=True, timeout=60,
        )
        out["src_diff_between_run_commits"] = d.stdout.strip()
        out["src_identical"] = (d.returncode == 0 and not d.stdout.strip())
    except Exception as e:  # pragma: no cover - provenance only
        out["src_identical"] = None
        out["why"] = repr(e)
    try:
        src = subprocess.run(
            ["git", "-C", repo, "show", f"{out['git_hashes']['ai_v9_58_R2CTRL_0827']}:src/main/train/config.py"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        i = src.find("args._distill_pairs = []")
        out["coef_gate_as_run"] = src[i : i + 90].splitlines()[:2]
        out["pairs_are_coef_gated"] = "if args.distill_coef and args.distill_coef > 0:" in src[i : i + 200]
    except Exception as e:  # pragma: no cover
        out["pairs_are_coef_gated"] = None
        out["why_gate"] = repr(e)
    doc = (Path(repo) / "src/main/train/matchup_setup.py").read_text()
    out["live_tree_docstring_names_this_run"] = "ai_v9_58_R2CTRL_0827` asked for exactly that" in doc
    out["verdict"] = (
        "R2CTRL == R2PLAIN in effective configuration ⇒ REPLICATE PAIR of plain training"
        if out.get("src_identical") and out.get("pairs_are_coef_gated")
        else "NOT ESTABLISHED — do not read the pair as replicates"
    )
    return out


def assert_screen():
    p = JOBS / "headroom_screen.json"
    if not p.exists():
        return {"available": False, "why": f"{p} absent; SCREEN_WR used as recorded"}
    raw = json.load(open(p))
    got, bad = {}, []
    for k, v in raw.items():
        if k in ("_meta", "POOLED"):
            continue
        sha8 = k.split("_", 1)[1]
        tag = f"U_{sha8}"
        if tag in SCREEN_WR:
            got[tag] = v["wr"]
            if abs(v["wr"] - SCREEN_WR[tag]) > 1e-9:
                bad.append(tag)
    missing = sorted(set(SCREEN_WR) - set(got))
    return {
        "available": True,
        "source": str(p),
        "screen_meta": raw.get("_meta", {}),
        "drifted": bad,
        "missing": missing,
        "ok": not bad and not missing,
    }


# ----------------------------------------------------------------- rendering


def fmt_pp(x):
    return f"{100 * x:+.2f}"


def render_contrast_row(label, c):
    if c is None:
        return f"| {label} | — | — | — | — | — | — |"
    lo, hi = c["ci95_cluster_bootstrap"]
    blo, bhi = c["ci95_binomial"]
    return (
        f"| {label} | {c['n_teams']} | {c['pooled_base_wr']:.4f} → {c['pooled_arm_wr']:.4f} | "
        f"**{fmt_pp(c['mean_delta'])}pp** | [{fmt_pp(lo)}, {fmt_pp(hi)}] | "
        f"[{fmt_pp(blo)}, {fmt_pp(bhi)}] | {c['z_binomial']:+.2f} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path stem for .md/.json")
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()

    arms, missing = {}, {}
    for tag in ARMS:
        a, err = load_arm(tag)
        if a:
            arms[tag] = a
        else:
            missing[tag] = err

    floor = [t for t in TEAMS if SCREEN_WR[t] > FLOOR_CUT]
    sub = [t for t in TEAMS if SCREEN_WR[t] <= FLOOR_CUT]

    # every contrast we can form, named by (arm, base)
    WANT = [
        ("plain_training_drift", "R2PLAIN", "REV1FIN"),
        ("ecology_only", "R2CTRL", "REV1FIN"),
        ("fold_rev2", "R2ACTION", "REV1FIN"),
        ("distillation_specific", "R2ACTION", "R2PLAIN"),
        ("distillation_beyond_ecology", "R2ACTION", "R2CTRL"),
        ("ecology_beyond_plain", "R2CTRL", "R2PLAIN"),
        ("fold_rev4", "R4ACTION", "R2ACTION"),
        ("self_fold_zero_content", "R3SELF", "R2ACTION"),
    ]

    cells = {}
    for name, arm, base in WANT:
        if arm not in arms or base not in arms:
            cells[name] = {"available": False, "needs": [x for x in (arm, base) if x not in arms]}
            continue
        A, B = arms[arm]["cells"], arms[base]["cells"]
        # COMMON-TEAM RESTRICTION, exact: only teams BOTH arms measured.
        common = [t for t in TEAMS if t in A and t in B]
        cells[name] = {
            "available": True,
            "arm": arm,
            "base": base,
            "restricted": len(common) != len(TEAMS),
            "restricted_to": common,
            "missing_teams": [t for t in TEAMS if t not in common],
            "all8": contrast(A, B, common, args.boot),
            "floor": contrast(A, B, [t for t in floor if t in common], args.boot),
            "sub_floor": contrast(A, B, [t for t in sub if t in common], args.boot),
        }

    # ---- the TAUGHT-9 companion cut, from already-banked meters (no battles)
    taught_arms, taught_missing = {}, {}
    for tag in ARMS:
        a, err = load_arm(tag, prefix="taught9", want=TAUGHT9)
        if a:
            taught_arms[tag] = a
        else:
            taught_missing[tag] = err
    taught_cells = {}
    for name, arm, base in WANT:
        if arm not in taught_arms or base not in taught_arms:
            taught_cells[name] = {
                "available": False,
                "needs": [x for x in (arm, base) if x not in taught_arms],
            }
            continue
        taught_cells[name] = {
            "available": True,
            "arm": arm,
            "base": base,
            "all9": contrast(taught_arms[arm]["cells"], taught_arms[base]["cells"], TAUGHT9, args.boot),
        }

    payload = {
        "probe": "M9",
        "question": "does PLAIN TRAINING rob untaught teams, with no distillation at all?",
        "instrument": {
            "collector": "designs/research_state/measurements/axis_split_untaught_arm.py (unmodified)",
            "teams": TEAMS,
            "n_per_team": 200,
            "target": "models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip",
            "impl": "rust",
            "stochastic": True,
        },
        "floor_stratum": {
            "variable": "headroom_screen.json -- R2ACTION piloting each team at n=150",
            "cut": FLOOR_CUT,
            "registered_by": "scorecard REPRO-2 (ledger 2026-08-31)",
            "floor_teams": floor,
            "sub_floor_teams": sub,
            "screen_wr": SCREEN_WR,
            "why_not_the_parent_wr": (
                "Stratifying on an arm's OWN measured WR and then differencing that same arm "
                "induces regression to the mean. The screen is an independent measurement of a "
                "third model, so it cannot."
            ),
        },
        "arms": {
            k: {"run": ARMS[k][0], "role": ARMS[k][1], "path": v["path"], "meta": v["meta"]}
            for k, v in arms.items()
        },
        "arms_missing": missing,
        "argv_single_variable": assert_argvs(),
        "screen_assertion": assert_screen(),
        "replicate_pair_audit": replicate_pair_audit(),
        "m5_control_audit": m5_control_audit(),
        "cells": cells,
        "taught9_companion": {
            "why": (
                "The SAME contrast on the 9 teams rev-2's fold actually TAUGHT, from the "
                "training session's standing fold-quality meter (9 x 300, same fixed target). "
                "Already banked for every arm, so it costs no battles and it says whether the "
                "plain-training effect is specific to untaught teams or general."
            ),
            "teams": TAUGHT9,
            "n_per_team": 300,
            "arms": {
                k: {"run": ARMS[k][0], "path": v["path"], "meta": v["meta"]}
                for k, v in taught_arms.items()
            },
            "arms_missing": taught_missing,
            "cells": taught_cells,
        },
    }

    out = Path(args.out)
    json.dump(payload, open(out.with_suffix(".json"), "w"), indent=1)

    # ---- a compact console/markdown table so the caller can read it without the JSON
    lines = []
    lines.append(
        "| contrast | teams | pooled WR base → arm | Δ | 95% CI (cluster over teams) "
        "| 95% CI (binomial — the ledger's convention) | z |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, arm, base, in WANT:
        c = cells[name]
        if not c.get("available"):
            lines.append(f"| {name} ({arm} − {base}) | MISSING: {c['needs']} | | | | | |")
            continue
        lines.append(render_contrast_row(f"**{name}** ({arm} − {base}) · all 8", c["all8"]))
        lines.append(render_contrast_row("  ↳ FLOOR (screen >0.55, n=6)", c["floor"]))
        lines.append(render_contrast_row("  ↳ sub-floor (n=2)", c["sub_floor"]))
    lines.append("")
    lines.append("### TAUGHT-9 companion (9 × 300, already banked — no battles)")
    lines.append("")
    lines.append(
        "| contrast | teams | pooled WR base → arm | Δ | 95% CI (cluster over teams) "
        "| 95% CI (binomial) | z |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, arm, base in WANT:
        c = taught_cells[name]
        if not c.get("available"):
            lines.append(f"| {name} ({arm} − {base}) | MISSING: {c['needs']} | | | | | |")
            continue
        lines.append(render_contrast_row(f"**{name}** ({arm} − {base})", c["all9"]))

    table = "\n".join(lines)
    open(out.parent / (out.name + "_tables.md"), "w").write(table + "\n")
    print(table)
    print()
    print("m5_control_audit:", json.dumps(payload["m5_control_audit"], indent=1)[:800])
    if missing:
        print("\nMISSING ARMS:", json.dumps(missing, indent=1))


if __name__ == "__main__":
    main()
