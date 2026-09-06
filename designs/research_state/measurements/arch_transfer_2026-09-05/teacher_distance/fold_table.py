"""THE FOLD TABLE — every gen-era fold with a banked untaught (off-slice) delta, RECOMPUTED.

Nothing here is copied out of the ledger. Every delta and every interval is recomputed from the
per-team win/games rows of the untaught-8 probe artifacts, so a transcription error in the ledger
cannot enter this measurement. Provenance (parent, teachers, coefficient, dose knobs) is read from
each run's own metadata.json `original_command` / `cli_args`, never typed.

REJECTED folds are printed with a reason. That list is part of the deliverable.

Run: python fold_table.py <out.json>   (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
MEAS = f"{MAIN}/designs/research_state/measurements"
IN = f"{HERE}/inputs"

# Where each arm's untaught-8 artifact lives. Every one of these was produced by the SAME probe
# (untaught_probe.py) at the SAME stamp: stochastic, opponent = rev-1's 24M snapshot, team set M,
# n=200/team. The stamp is re-verified per file below and a mismatch is fatal.
ART = {
    "R4ACTION":   f"{MEAS}/reuse_batch_2026-09-03/untaught_R4ACTION.json",
    "R3ACTION":   f"{MEAS}/axis_split_inputs/untaught_R3ACTION.json",
    "COMPFOLD":   f"{MEAS}/axis_split_inputs/untaught_COMPFOLD.json",
    "R2ACTION":   f"{MEAS}/reuse_batch_2026-09-03/untaught_R2ACTION.json",
    "REV1FIN":    f"{MEAS}/plain_training_robbery_inputs/untaught_REV1FIN.json",
    "B2_end":     f"{MEAS}/reuse_batch_2026-09-03/untaught_B2_end.json",
    "C1_end":     f"{MEAS}/reuse_batch_2026-09-03/untaught_C1_end.json",
    "N1_end":     f"{MEAS}/reuse_batch_2026-09-03/untaught_N1_end.json",
    "N2_end":     f"{MEAS}/reuse_batch_2026-09-03/untaught_N2_end.json",
    "R4DOSE12_end": f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE12_end.json",
    "R4DOSE6_end":  f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE6_end.json",
    "R4DOSE3_end":  f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE3_end.json",
    "TCFUNDA_end":  f"{IN}/untaught_TCFUNDA_end.json",
    "TCFUNDB_end":  f"{IN}/untaught_TCFUNDB_end.json",
    "TCUNFA_end":   f"{IN}/untaught_TCUNFA_end.json",
    "TCUNFB_end":   f"{IN}/untaught_TCUNFB_end.json",
    "TCUNFK6A_end": f"{IN}/untaught_TCUNFK6A_end.json",
    "TCUNFK6B_end": f"{IN}/untaught_TCUNFK6B_end.json",
}
# also banked for the p1M depth (used only for the depth-sensitivity column)
ART_P1M = {
    "B2": f"{MEAS}/reuse_batch_2026-09-03/untaught_B2_p1M.json",
    "C1": f"{MEAS}/reuse_batch_2026-09-03/untaught_C1_p1M.json",
    "N1": f"{MEAS}/reuse_batch_2026-09-03/untaught_N1_p1M.json",
    "N2": f"{MEAS}/reuse_batch_2026-09-03/untaught_N2_p1M.json",
    "R4DOSE12": f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE12_p1M.json",
    "R4DOSE6": f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE6_p1M.json",
    "R4DOSE3": f"{MEAS}/dose_cell_2026-09-02/untaught_R4DOSE3_p1M.json",
    "TCFUNDA": f"{IN}/untaught_TCFUNDA_p1M.json",
    "TCFUNDB": f"{IN}/untaught_TCFUNDB_p1M.json",
    "TCUNFA": f"{IN}/untaught_TCUNFA_p1M.json",
    "TCUNFB": f"{IN}/untaught_TCUNFB_p1M.json",
    "TCUNFK6A": f"{IN}/untaught_TCUNFK6A_p1M.json",
    "TCUNFK6B": f"{IN}/untaught_TCUNFK6B_p1M.json",
}

# fold tag -> (run dir, artifact key, PARENT artifact key)
FOLDS = [
    ("rev-4 / R4ACTION", "ai_v9_76_R4ACTION_0830",  "R4ACTION",     "R2ACTION"),
    ("rev-3 / R3ACTION", "ai_v9_70_R3ACTION_0828",  "R3ACTION",     "R2ACTION"),
    ("COMPFOLD",         "ai_v9_91_COMPFOLD_0831",  "COMPFOLD",     "R2ACTION"),
    ("rev-2 / R2ACTION", "ai_v9_59_R2ACTION_0827",  "R2ACTION",     "REV1FIN"),
    ("B2",               "ai_v9_140_B2_0901",       "B2_end",       "R2ACTION"),
    ("C1 (coef 0)",      "ai_v9_141_C1_0901",       "C1_end",       "R2ACTION"),
    ("N1",               "ai_v9_142_N1_0901",       "N1_end",       "R2ACTION"),
    ("N2",               "ai_v9_143_N2_0901",       "N2_end",       "R2ACTION"),
    ("R4DOSE12",         "ai_v9_150_R4DOSE12_0901", "R4DOSE12_end", "R2ACTION"),
    ("R4DOSE6",          "ai_v9_151_R4DOSE6_0901",  "R4DOSE6_end",  "R2ACTION"),
    ("R4DOSE3",          "ai_v9_152_R4DOSE3_0901",  "R4DOSE3_end",  "R2ACTION"),
    ("TC_FUND_A",        "ai_v9_160_TCFUNDA_0903",  "TCFUNDA_end",  "R2ACTION"),
    ("TC_FUND_B",        "ai_v9_161_TCFUNDB_0903",  "TCFUNDB_end",  "R2ACTION"),
    ("TC_UNF_A",         "ai_v9_162_TCUNFA_0903",   "TCUNFA_end",   "R2ACTION"),
    ("TC_UNF_B",         "ai_v9_163_TCUNFB_0903",   "TCUNFB_end",   "R2ACTION"),
    ("TC_UNF_K6_A",      "ai_v9_170_TCUNFK6A_0904", "TCUNFK6A_end", "R2ACTION"),
    ("TC_UNF_K6_B",      "ai_v9_171_TCUNFK6B_0904", "TCUNFK6B_end", "R2ACTION"),
]

REJECTED = [
    ("R3SELF", "no untaught cell was ever run (ledger, M9 FINAL: \"R3SELF's untaught cell also "
               "unrun\"); only its taught -8.96pp exists"),
    ("R4PLAIN", "does not exist — the named-and-priced matched plain control for rev-4 was never "
                "launched"),
    ("EXT_A (K=3 unfunded extension)", "still training at the time of this probe; no endpoint "
                                       "artifact"),
    ("R2PLAIN / R2CTRL", "NOT folds — distillation-free arms. They supply the replicate FLOOR, "
                         "and they have no teacher set, so no D_off exists for them"),
    ("the fdA/fdB/fdC/fdE/fdF and G1/G2 family (ai_v9_38/39/40/42/45/48/49)",
     "no untaught-8 artifact exists for ANY of them (searched the tree and the job dirs); they "
     "were scored on the TAUGHT side only, and they fork off rev-1 rather than off R2ACTION, so "
     "even a future untaught pass would be a sixth parent"),
    ("every rev-2/rev-3-era 'fleet fold' other than R2ACTION/R3ACTION",
     "no untaught-8 artifact on the standing stamp; probe Q's readings are a DIFFERENT team set "
     "and a greedy meter and may not be quoted beside these"),
]


def rows(path):
    d = json.load(open(path))
    meta = d["_meta"]
    ks = [k for k in d if k != "_meta"]
    w = np.array([d[k]["wins"] for k in ks], float)
    g = np.array([d[k]["games"] for k in ks], float)
    return ks, w, g, meta


def stamp_ok(meta):
    return (meta.get("n_per_team") == 200
            and meta.get("target", "").endswith("snapshot_000024000000.zip")
            and meta.get("pool") == 719)


def prov(run):
    p = f"{MD}/{run}/metadata.json"
    m = json.load(open(p))
    oc = m.get("original_command", "")
    ca = m.get("cli_args", {})
    spec = re.findall(r"--distill-teacher\s+(\S+)", oc)
    teachers, taught = [], []
    if spec:
        for chunk in spec[0].strip("'\"").split(";"):
            run_path, _, teams = chunk.partition(":")
            t = run_path.split("/")[-1]
            teachers.append(t)
            if teams.strip() == "*":
                tm = json.load(open(f"{MD}/{t}/metadata.json"))
                tt = re.findall(r"--trainee-teams\s+(\S+)",
                                tm.get("original_command", ""))
                teams = tt[0].strip("'\"") if tt else ""
            taught += [x.split("/")[-1].replace(".txt", "") for x in teams.split(",") if x]
    return {
        "run": run,
        "parent_model": (re.findall(r"--model\s+(\S+)", oc) or [None])[0],
        "teachers": teachers,
        "taught_teams": sorted(set(taught)),
        "n_taught_teams": len(set(taught)),
        "coef": ca.get("distill_coef"),
        "target": ca.get("distill_target"),
        "fork_lr": ca.get("fork_lr"),
        "gas": ca.get("grad_accum_steps"),
        "team_bias": ca.get("distill_team_bias"),
        "git_hash": (m.get("git_hash") or "")[:8],
    }


def main(out_path):
    rng = np.random.default_rng(20260905)
    base = {}
    for k in ("R2ACTION", "REV1FIN"):
        ks, w, g, meta = rows(ART[k])
        assert stamp_ok(meta), f"{k} stamp mismatch: {meta}"
        base[k] = (ks, w / g)

    table, seen_keys = [], None
    for tag, run, art, par in FOLDS:
        ks, w, g, meta = rows(ART[art])
        assert stamp_ok(meta), f"{art} stamp mismatch: {meta}"
        if seen_keys is None:
            seen_keys = ks
        assert ks == seen_keys, f"{art} team ORDER differs: {ks} vs {seen_keys}"
        pk, pwr = base[par]
        assert pk == ks, f"{art} vs parent {par}: team order differs"
        wr = w / g
        d = (wr - pwr) * 100.0
        # cluster bootstrap over the 8 untaught TEAMS (the real unit)
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        bs = d[idx].mean(axis=1)
        p = prov(run)
        table.append({
            "fold": tag, "artifact": os.path.relpath(ART[art], MAIN),
            "parent_artifact": par,
            "untaught_wr": float(wr.mean()), "parent_wr": float(pwr.mean()),
            "delta_pp": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_team_delta_pp": [float(x) for x in d],
            "teams_up": int((d > 0).sum()),
            "provenance": p,
        })
        print(f"{tag:20s} wr {wr.mean():.4f}  Δ {d.mean():+6.2f} "
              f"[{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}]  "
              f"up {int((d>0).sum())}/8  | teachers {len(p['teachers'])} "
              f"taught {p['n_taught_teams']:2d}  coef {p['coef']}  gas {p['gas']}  "
              f"fork_lr {p['fork_lr']}", flush=True)

    print("\nREJECTED:")
    for n, why in REJECTED:
        print(f"  {n}: {why}")

    out = {"_meta": {
        "statistic": "untaught-8 win-rate delta vs the fold's OWN parent, pp; per-team rows "
                     "recomputed from the probe artifacts, cluster-bootstrapped over the 8 teams "
                     "(20000 draws, seed 20260905)",
        "stamp": "stochastic · opponent rev-1 snapshot_000024000000 · team set M · n=200/team",
        "untaught_teams": seen_keys},
        "folds": table, "rejected": [{"fold": n, "reason": w} for n, w in REJECTED]}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/fold_table.json")
