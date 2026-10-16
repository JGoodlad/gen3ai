"""R-T5's equal-compute step (PREREG.md): reads time_to_step.py's JSON (argv[1]); T_ref = the mean over the 8 EB seeds of
the EFFECTIVE learn() wall seconds to 15,000,000 steps; per toy seed, the step s* where its effective curve reaches T_ref,
and the checkpoint nearest s* (by |step - s*|). JSON on stdout. Read-only."""
import glob, json, re, statistics, sys

d = json.load(open(sys.argv[1])); M = "/home/goodlad/dev/gen3ai/models"
eb = {k: v["t_eff_15M_s"] for k, v in d.items() if k.startswith("rb_bf_eb_")}
T = statistics.mean(eb.values()); SD = statistics.stdev(eb.values())


def step_at(curve, t):
    for (s0, t0), (s1, t1) in zip(curve, curve[1:]):
        if t0 <= t <= t1:
            return s0 + (s1 - s0) * (t - t0) / (t1 - t0) if t1 > t0 else s0


out = {"eb_t15M_s": eb, "T_ref_s": T, "T_ref_sd_s": SD, "toy": {}}
for run in ("rb_toy_ent_s4001", "rb_toy_ent_s4002", "rb_toy_ent_s4003"):
    s = step_at(d[run]["curve"], T)
    lo, hi = step_at(d[run]["curve"], T - SD), step_at(d[run]["curve"], T + SD)
    ck = []
    for p in glob.glob(f"{M}/{run}/checkpoints/checkpoint_*_steps.zip"):
        m = re.search(r"checkpoint_(\d+)_steps", p)
        if m: ck.append((int(m.group(1)), p))
    n = min(ck, key=lambda c: abs(c[0] - s))
    out["toy"][run] = {"s_star": round(s), "s_star_at_T_pm_1sd": [round(lo), round(hi)], "ckpt": n[1], "ckpt_step": n[0],
                       "toy_t15M_s": d[run]["t_eff_15M_s"], "toy_t100M_s": d[run].get("t_eff_100M_s"),
                       "speed_ratio_at_15M": round(T / d[run]["t_eff_15M_s"], 3)}
print(json.dumps(out, indent=1))
