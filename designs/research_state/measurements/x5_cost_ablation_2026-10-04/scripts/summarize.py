"""Join the launch reads and the ablation arms into the numbers the README quotes (X5 cost ablation, 2026-10-04).

    python summarize.py [--results results] [--out results/numbers.json] [--check]

Inputs, all COMMITTED beside the script: `results/<launch>.json` (`read_run.py`'s per-launch read, taken from
the launch's child log, TensorBoard and nvidia-smi in a job-scratch directory that is not kept), the phase / CPU-
sampler rows `results/raw/<launch>.{phases,cpu}.jsonl`, and `results/ablate_<arm>_r<k>.json` (`ablate.py`).
`--check` resolves them and exits non-zero on a missing one (`src/measurements_readout_gate_test.py`).

Launches: regime B (eval off, empty pool: the trainee plays the in-core bots; T2 serves the trainee only) =
blobB / fmB; regime A (a seeded 20-snapshot pool, self-play 90 %, T2 serves the trainee + 20 pool slots) =
blobA1, blobA2 / fmA3, fmA4 (both with `--behaviour-check warn`), run in that order (fmA1, fmA1b and fmA2 were refused — README). Every launch: `--arch production`
at N = 256 WITH the X26 ride-along heads (`--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5
--ridealong-opp 5 --ridealong-rnd-variants all`).

Statistics: the steady updates are the QUIET ones (windowed contention factor < 1.05; warm-up and the
compile-canary update excluded — `read_run.py`, standing rule 8). Median per arm over the pooled quiet updates,
a bootstrap 95 % CI of each median and of the DIFFERENCE (fixed_mass − blob), and each launch's own median.
"T2 service per host step" = `rust_env/flush_ms_per_host_step` (the host's graph launches) + `rust_env/
gpu_wait_ms_per_host_step` (the wait for every slot's results) — the whole T2 cost the collector pays per host
step (the budget's "T2 flush"; F-G-3 resolved in the README).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

REGIMES = {"B": {"blob": ["blobB"], "fixed_mass": ["fmB"]},
           "A": {"blob": ["blobA1", "blobA2"], "fixed_mass": ["fmA3", "fmA4"]}}
ABL_ARMS = ["blob", "fm", "no_other", "no_hypenc", "no_bisect", "no_argsort", "no_ext", "no_keybias",
            "no_hypatk", "no_permon", "all_off"]
ABL_REPS = (1, 2)


def boot_median(x, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.median(rng.choices(x, k=len(x))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def boot_diff(a, b, n=10000, seed=1):
    rng = random.Random(seed)
    d = sorted(st.median(rng.choices(a, k=len(a))) - st.median(rng.choices(b, k=len(b))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n)]


def play_walls(phases_path, cpu_path):
    ph = [json.loads(line) for line in open(phases_path)]
    col = sorted([r for r in ph if r["phase"] == "collect"], key=lambda r: r["t0"])
    samp = [json.loads(line) for line in open(cpu_path) if line.strip()]
    samp = [s for s in samp if "t" in s]
    out = []
    for i, c in enumerate(col):
        if i < 2:
            continue
        f = [s["factor"] for s in samp if c["t0"] <= s["t"] - s["dt"] and s["t"] <= c["t1"]]
        if not f or max(f) >= 1.05:
            continue
        out.append(round(c["t1"] - c["t0"], 3))
    return out


def headroom(line):
    m = re.search(r"headroom ([0-9,]+) MiB", line or "")
    return float(m.group(1).replace(",", "")) if m else None


def launch_row(r, d, t):
    tb = r["tb"]

    def med(k):
        return tb.get(k, {}).get("median_steady")
    fl, gw = med("rust_env/flush_ms_per_host_step"), med("rust_env/gpu_wait_ms_per_host_step")
    return {
        "startup_s_start_to_first_collect": r["startup_wall_s_start_to_first_collect"],
        "T2_up_s": float(r["T2_up_s"]), "R1_reset_prewarm_s": float(r["R1_prewarm_s"]),
        "startup_contended_windows": f'{r.get("startup_windows_ge_1.05")}/{r.get("startup_windows")}',
        "startup_max_contention_factor": round(r.get("startup_max_contention_factor") or 0, 2),
        "dry_update_s": r["updates"][0]["dur_s"],
        "update_steady_n": r["update_wall_s_steady"]["n"], "update_steady_median_s": r["update_wall_s_steady"]["median"],
        "update_steady_values": r["update_wall_s_steady"]["values"],
        "excluded_updates": [(u["i"], u["excluded"]) for u in r["updates"] if u["excluded"]],
        "train_ms_tb_median": med("train/train_ms"),
        "T2_flush_host_ms": fl, "T2_gpu_wait_ms": gw,
        "T2_service_ms": (fl + gw) if fl is not None and gw is not None else None,
        "core_ms_per_host_step": med("rust_env/core_ms_per_host_step"),
        "host_steps": med("rust_env/host_steps"),
        "p2_policy_decisions": med("rust_env/p2_policy_decisions"),
        "trainee_rows_per_host_step": med("rust_env/trainee_rows_per_host_step"),
        "updatefit_headroom_mib": headroom(r["dry_update_line"]),
        "updatefit_line": r["dry_update_line"],
        "cuda_device_free_mib": med("lifecycle/cuda_device_free_mib"),
        "cuda_update_peak_reserved_mib": med("lifecycle/cuda_update_peak_reserved_mib"),
        "cuda_update_peak_alloc_mib": med("lifecycle/cuda_update_peak_alloc_mib"),
        "startup_ledger": r.get("ledger"),
        "graphs_total": med("compile/graphs_total"),
        "recompiles_after_lock_max": max(tb.get("compile/recompiles_after_lock", {}).get("all") or [0]),
        "play_wall_s_quiet": play_walls(d / f"{t}.phases.jsonl", d / f"{t}.cpu.jsonl"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--out", default=str(here / "results" / "numbers.json"))
    ap.add_argument("--check", action="store_true", help="resolve every input, print the missing ones, compute nothing")
    ap.add_argument("--launches-only", action="store_true", help="skip the ablation arms (print, write nothing)")
    a = ap.parse_args()
    rdir = Path(a.results)
    d = rdir / "raw"
    tags = [t for reg in REGIMES.values() for ts in reg.values() for t in ts]
    need = [rdir / f"{t}.json" for t in tags] + [d / f"{t}.{k}.jsonl" for t in tags for k in ("phases", "cpu")] + \
           ([] if a.launches_only else [rdir / f"ablate_{arm}_r{k}.json" for arm in ABL_ARMS for k in ABL_REPS])
    missing = [str(p) for p in need if not p.is_file()]
    if a.check:
        print("\n".join(f"MISSING {m}" for m in missing) or f"all {len(need)} inputs resolve")
        sys.exit(1 if missing else 0)
    if missing:
        sys.exit("missing inputs:\n" + "\n".join(missing))
    out = {"launches": {}, "regimes": {}, "ablation": {}}
    for t in tags:
        out["launches"][t] = launch_row(json.load(open(rdir / f"{t}.json")), d, t)
    L = out["launches"]
    for reg, arms in REGIMES.items():
        o = {}
        for arm, ts in arms.items():
            ups = [v for t in ts for v in L[t]["update_steady_values"]]
            pw = [v for t in ts for v in L[t]["play_wall_s_quiet"]]
            o[arm] = {"update_n": len(ups), "update_median_s": st.median(ups), "update_median_ci95": boot_median(ups),
                      "per_launch_update_median_s": [L[t]["update_steady_median_s"] for t in ts],
                      "play_n": len(pw), "play_median_s": st.median(pw) if pw else None}
            for k in ("T2_flush_host_ms", "T2_gpu_wait_ms", "T2_service_ms", "updatefit_headroom_mib",
                      "cuda_update_peak_reserved_mib", "cuda_update_peak_alloc_mib", "T2_up_s", "R1_reset_prewarm_s",
                      "startup_s_start_to_first_collect", "p2_policy_decisions", "core_ms_per_host_step"):
                vals = [L[t][k] for t in ts]
                o[arm][k] = {"per_launch": vals, "mean": st.mean(vals) if all(v is not None for v in vals) else None}
        ub = [v for t in arms["blob"] for v in L[t]["update_steady_values"]]
        uf = [v for t in arms["fixed_mass"] for v in L[t]["update_steady_values"]]
        b, f = o["blob"], o["fixed_mass"]
        delta = {"update_median_s": f["update_median_s"] - b["update_median_s"],
                 "update_median_pct": 100 * (f["update_median_s"] / b["update_median_s"] - 1),
                 "update_median_diff_ci95_pct": tuple(100 * x / b["update_median_s"] for x in boot_diff(uf, ub))}
        for k in ("T2_flush_host_ms", "T2_gpu_wait_ms", "T2_service_ms"):
            delta[k + "_pct"] = 100 * (f[k]["mean"] / b[k]["mean"] - 1)
            delta[k] = f[k]["mean"] - b[k]["mean"]
        for k in ("updatefit_headroom_mib", "cuda_update_peak_reserved_mib", "cuda_update_peak_alloc_mib", "T2_up_s",
                  "R1_reset_prewarm_s", "startup_s_start_to_first_collect"):
            delta[k] = f[k]["mean"] - b[k]["mean"]
        if b["play_median_s"] and f["play_median_s"]:
            delta["play_median_pct"] = 100 * (f["play_median_s"] / b["play_median_s"] - 1)
        o["delta"] = delta
        out["regimes"][reg] = o
    if a.launches_only:
        print(json.dumps({"launches": out["launches"], "regimes": out["regimes"]}, indent=1, default=list))
        return
    # ---- ablation: the extractor's compiled fwd+bwd at B = 2,048, one process per arm and replicate
    ab = {arm: [json.load(open(rdir / f"ablate_{arm}_r{k}.json")) for k in ABL_REPS] for arm in ABL_ARMS}
    ms = {arm: st.mean(r["compiled_ms_median"] for r in rs) for arm, rs in ab.items()}
    mem = {arm: st.mean(r["peak_alloc_mib"] for r in rs) for arm, rs in ab.items()}
    fm, blob = ms["fm"], ms["blob"]
    rows = {}
    for arm in ABL_ARMS:
        rows[arm] = {"ms_per_microbatch": [r["compiled_ms_median"] for r in ab[arm]], "ms_mean": round(ms[arm], 3),
                     "compile_s": [r["first_call_compile_s"] for r in ab[arm]],
                     "peak_alloc_mib": round(mem[arm], 1),
                     "saving_vs_fm_ms": round(fm - ms[arm], 3) if arm not in ("blob", "fm") else None,
                     "saving_share_of_x5_pct": round(100 * (fm - ms[arm]) / (fm - blob), 1) if arm not in ("blob", "fm") else None,
                     "saving_per_update_s": round((fm - ms[arm]) * 480 / 1000, 2) if arm not in ("blob", "fm") else None,
                     "mem_saving_vs_fm_mib": round(mem["fm"] - mem[arm], 1) if arm != "fm" else None}
    out["ablation"] = {"x5_extractor_ms": round(fm - blob, 3), "x5_extractor_pct": round(100 * (fm / blob - 1), 1),
                       "x5_extractor_per_update_s": round((fm - blob) * 480 / 1000, 2), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, default=list))
    print(json.dumps({"regimes": out["regimes"], "ablation": out["ablation"]}, indent=1, default=list)[:12000])


if __name__ == "__main__":
    main()
