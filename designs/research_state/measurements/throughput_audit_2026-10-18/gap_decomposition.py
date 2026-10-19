"""The throughput audit's arithmetic: our cycle per env step, the gap to the published bots, and what each lever buys
(pure arithmetic over the INPUTS below; no GPU, no I/O beyond printing).

    python3 designs/research_state/measurements/throughput_audit_2026-10-18/gap_decomposition.py

Every input carries its provenance. "Step" = one TRAINEE decision (one row of the rollout buffer), the unit of
`time/fps` and of `rollout/collect_decisions`. Their figures are SELF-REPORTED forum numbers
(`designs/research_state/literature/showdown_bots_landscape_2026-10-18.md`); every quantity derived from them is an
ESTIMATE, and the ranges below say so.
"""
from __future__ import annotations

# ------------------------------------------------------------------------------------------------ inputs (ours)
#: X26 (`rb_x26_s2601`, RTX 3080 Ti, bf16 trunk, 256 envs, pool 20, self-play 0.90): medians over its last 30
#: rollouts at ~312M steps, read from its TensorBoard events 2026-10-18 ~20:10 PDT (README "Data").
X26 = {"train_s": 41.962, "collect_s": 26.065, "fps": 1346.0, "host_steps": 416.0, "rows": 98304,
           "gpu_wait_ms": 40.199, "core_ms": 13.334,
           "glue_ms": {"submit": 1.649, "flush": 2.365, "opp_draw": 0.447, "draw": 0.169, "write": 0.719, "post": 0.740},
           "fill_s": 0.269, "trainee_rows_per_host_step": 236.4, "p2_policy_rows": 91039.5, "games_ended": 2228.0,
           "ep_len": 44.8}
#: The learner's counted matmul-family FLOPs per row (CPU profiler, production micro-step at 2,048 rows):
#: `designs/endstate/design_compute_efficiency.md` §1.2 (`compute_efficiency_2026-10-16/result/cost_production.json`).
FWD_MFLOP, FWDBWD_MFLOP, EPOCHS = 94.8, 244.6, 10
PARAMS = 2_609_183          # learner params at HEAD (this folder's param_count.py; embeddings 32,352)
#: The frozen GPU suite (`gpu_bench_2026-10-18/suites/`): the bf16 update at the production shape, 1- and 3-epoch
#: medians -> fixed + marginal per epoch. 3090 (hu, 7900X) is the reference; the 600 W 5090 (mx, 9900X) the target.
BENCH = {"r3090": {"e1": 5.046, "marginal": 3.886, "est": 40.02, "wait_ms": 40.05},
             "r5090": {"e1": 3.308, "marginal": 1.703, "est": 18.64, "wait_ms": 18.81}}

# ------------------------------------------------------------------------------------------------ inputs (theirs)
MIKU = {"steps_per_s": 100_000, "grad_share": 0.70, "envs": 8192, "steps_per_update": 32, "games": 330e6, "hours": 48.3,
            "params": 8.7e6}
#: Peak dense BF16 tensor throughput of an RTX 5090, TFLOP/s. NVIDIA's sheet; GeForce parts may run FP32-accumulate
#: at half this rate (UNVERIFIED for Blackwell GeForce), so 209.5 is the GENEROUS bound for their side.
PEAK_5090 = 209.5
PEAK_3080TI = 68.2           # dense FP16/BF16 tensor with FP32 accumulate (Ampere GA102 whitepaper, 3080 Ti)


def ours() -> dict:
    c = dict(X26)
    cycle = c["rows"] / c["fps"]
    glue = sum(c["glue_ms"].values())
    wait_s = c["gpu_wait_ms"] * c["host_steps"] / 1e3
    core_s = c["core_ms"] * c["host_steps"] / 1e3
    glue_s = glue * c["host_steps"] / 1e3
    rest_collect = c["collect_s"] - wait_s - core_s - glue_s
    other = cycle - c["train_s"] - c["collect_s"]
    parts = {"update": c["train_s"], "gpu_wait": wait_s, "env_core": core_s, "python_glue": glue_s,
                 "collect_rest": rest_collect, "other": other}
    return {"cycle": cycle, "parts": parts, "glue_ms": glue}


def per_step_table(o: dict) -> None:
    rows = X26["rows"]
    print("| part of the cycle | s per 98,304-step cycle | µs per step | share |")
    print("|---|---|---|---|")
    for k, v in o["parts"].items():
        print(f"| {k} | {v:.2f} | {1e6 * v / rows:.1f} | {100 * v / o['cycle']:.1f} % |")
    print(f"| **total** | **{o['cycle']:.2f}** | **{1e6 * o['cycle'] / rows:.1f}** | 100 % |")


def hardware(o: dict) -> dict:
    """The 600 W 5090 on our cycle (Amdahl, the suite's own rule: the update and the GPU wait scale; the host does not)."""
    r, t = BENCH["r3090"], BENCH["r5090"]
    p = dict(o["parts"])
    p["update"] *= t["est"] / r["est"]
    p["gpu_wait"] *= t["wait_ms"] / r["wait_ms"]
    cyc = sum(p.values())
    return {"cycle": cyc, "parts": p, "speedup": o["cycle"] / cyc}


def epochs_update_s(e: int) -> float:
    """Our update at E epochs, from the 3090 suite's fixed + marginal split scaled to X26's measured E10."""
    r = BENCH["r3090"]
    fixed = r["e1"] - r["marginal"]
    scale = X26["train_s"] / r["est"]
    return scale * (fixed + e * r["marginal"])


def main() -> int:
    o = ours()
    print(f"OUR CYCLE (X26 @312M): {o['cycle']:.2f} s / 98,304 steps = {X26['rows'] / o['cycle']:,.0f} steps/s\n")
    per_step_table(o)
    hw = hardware(o)
    print(f"\n5090 (600 W) on our cycle: {hw['cycle']:.2f} s -> {X26['rows'] / hw['cycle']:,.0f} steps/s, "
          f"x{hw['speedup']:.2f}")
    for k, v in hw["parts"].items():
        print(f"  {k:12s} {v:6.2f} s  {1e6 * v / X26['rows']:6.1f} µs/step")

    us_t = 1e6 / MIKU["steps_per_s"]
    print(f"\nmikumiku37 as stated: {us_t:.1f} µs/step, of it {MIKU['grad_share'] * us_t:.1f} µs PPO gradient")
    gap = MIKU["steps_per_s"] * o["cycle"] / X26["rows"]
    print(f"face-value gap: x{gap:.1f}   after the card: x{gap / hw['speedup']:.1f}")
    games_s = MIKU["games"] / (MIKU["hours"] * 3600)
    print(f"their games/s {games_s:,.0f} => {MIKU['steps_per_s'] / games_s:.1f} 'steps' per game "
          f"(a Bo1 VGC game is ~6-12 turns: the stated figures are not one unit)")
    our_games_s = X26["games_ended"] / o["cycle"]
    print(f"our games/s {our_games_s:.1f} ({X26['rows'] / X26['games_ended']:.1f} trainee steps / game); "
          f"games/s gap x{games_s / our_games_s:.0f}")
    # the update side on the same card
    upd_ours_5090 = 1e6 * hw["parts"]["update"] / X26["rows"]
    print(f"\nUPDATE on a 5090: ours {upd_ours_5090:.1f} µs/step vs theirs {MIKU['grad_share'] * us_t:.1f}: "
          f"x{upd_ours_5090 / (MIKU['grad_share'] * us_t):.1f}")
    per_sample_mflop = EPOCHS * FWDBWD_MFLOP
    print(f"our learner work per sample: {EPOCHS} epochs x {FWDBWD_MFLOP} = {per_sample_mflop:,.0f} counted MFLOP")
    tflops_local = per_sample_mflop * 1e6 * X26["rows"] / X26["train_s"] / 1e12
    tflops_5090 = per_sample_mflop * 1e6 * X26["rows"] / BENCH["r5090"]["est"] / 1e12
    print(f"our achieved (counted) learner rate: {tflops_local:.1f} TFLOP/s on the 3080 Ti "
          f"({100 * tflops_local / PEAK_3080TI:.0f} % of {PEAK_3080TI}), {tflops_5090:.1f} on the 5090 "
          f"({100 * tflops_5090 / PEAK_5090:.1f} % of {PEAK_5090})")
    for mfu in (1.0, 0.5, 0.3):
        cap = MIKU["grad_share"] * us_t * 1e-6 * PEAK_5090 * 1e12 * mfu / 1e6
        print(f"  their per-sample learner work at {int(100 * mfu)} % utilisation: <= {cap:,.0f} MFLOP "
              f"(ours is x{per_sample_mflop / cap:.1f})")
    print("\ntheir per-forward FLOPs, a d=256 x 8-layer transformer (Wolfe_Alt6767's stated shape, 6.29M layer params):")
    for t in (8, 16, 24):
        f = 2 * 12 * 256 ** 2 * 8 * t / 1e6
        print(f"  T = {t:2d} tokens: ~{f:,.0f} MFLOP forward (ours {FWD_MFLOP})")
    print("\nTHE EPOCH LEVER on our cycle (3090 suite fixed + marginal, scaled to X26's E10):")
    for e in (10, 5, 3, 2, 1):
        u = epochs_update_s(e)
        cyc = o["cycle"] - X26["train_s"] + u
        print(f"  E{e:2d}: update {u:5.1f} s, cycle {cyc:5.1f} s, {X26['rows'] / cyc:,.0f} steps/s, x{o['cycle'] / cyc:.2f}")
    print(f"\nparams: ours {PARAMS:,} vs theirs {MIKU['params']:,.0f} (x{MIKU['params'] / PARAMS:.1f})")
    inf_rows = X26["rows"] + X26["p2_policy_rows"]
    wait_s = o["parts"]["gpu_wait"]
    print(f"inference rows per cycle {inf_rows:,.0f} ({inf_rows / X26['rows']:.2f} per trainee step) in "
          f"{wait_s:.1f} s of GPU wait = {1e6 * wait_s / inf_rows:.0f} µs/row; the learner's per-row-epoch "
          f"fwd+bwd is {1e6 * X26['train_s'] / (EPOCHS * X26['rows']):.1f} µs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
