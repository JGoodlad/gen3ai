"""§7 successor: stall-trajectory coverage + critic sign at the final decision.
Reads gen-13 eval traces + the training log. No battles, no model load.

RUN FROM THE MAIN CHECKOUT (/home/goodlad/dev/gen3ai) — models/ is gitignored and exists only
there, never in a worktree. Emits the JSON beside this file on stdout."""
import json, glob, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np

G13 = "models/ai_v9_15_gen13_hb_events_stack_0817"
G14 = "models/ai_v9_16_gen14_framedel_v91_0817"
CAP = 250  # MAX_TURNS

# ---- (a) TRAINING side: episode + decision share of cap-length trajectories ----
log = open(f"{G13}/launcher_child.log", errors="ignore").read()
# Status precedes Turns on the line; do NOT guess the order.
eps = re.findall(r"Episode Finished \|.*?Status: *(WIN|LOSS|DRAW).*?\| *Turns: *(\d+)", log)
if not eps:
    raise SystemExit("episode parse matched NOTHING - refusing to emit a zeros block")
train = [(int(t), o) for o, t in eps]
tr_cap = [e for e in train if e[0] >= CAP]
train_block = {
    "episodes": len(train),
    "cap_episodes": len(tr_cap),
    "cap_episode_frac": round(len(tr_cap) / len(train), 5) if train else None,
    "decisions_total": sum(t for t, _ in train),
    "decisions_cap": sum(t for t, _ in tr_cap),
    "cap_decision_frac": round(sum(t for t, _ in tr_cap) / sum(t for t, _ in train), 5) if train else None,
    "cap_outcomes": dict(Counter(o for _, o in tr_cap)),
    "outcomes_all": dict(Counter(o for _, o in train)),
}

# ---- (b)/(c) EVAL side: turn distribution + critic sign at the final decision ----
by = defaultdict(lambda: defaultdict(list))
vals = {"cap": [], "ordinary": []}
for s in glob.glob(f"{G13}/eval_traces/*/*/*_summary.json"):
    opp = os.path.basename(os.path.dirname(s))
    grp = "sentinel" if opp.startswith("sentinel") else "bot"
    try:
        m = json.load(open(s))["meta"]
    except Exception:
        continue
    r, t = m.get("result"), m.get("turns")
    if r not in ("WIN", "LOSS") or t is None:
        continue
    by[grp][r].append(int(t))
    if r != "LOSS":
        continue
    n = s.replace("_summary.json", "_states.npz")
    if not os.path.exists(n):
        continue
    try:
        z = np.load(n); v, w = z["values"], z["win_probs"]
    except Exception:
        continue
    if len(v):
        vals["cap" if t >= CAP else "ordinary"].append((float(v[-1]), float(w[-1])))

eval_block = {}
for grp in by:
    eval_block[grp] = {}
    for r in ("LOSS", "WIN"):
        arr = sorted(by[grp][r])
        if not arr:
            continue
        eval_block[grp][r] = {
            "n": len(arr), "median_turns": arr[len(arr) // 2],
            "p90_turns": arr[int(.9 * len(arr))], "max_turns": arr[-1],
            "cap_n": sum(1 for t in arr if t >= CAP),
            "cap_frac": round(sum(1 for t in arr if t >= CAP) / len(arr), 5),
        }

crit = {}
for k, rows in vals.items():
    if not rows:
        continue
    V = np.array([r[0] for r in rows]); W = np.array([r[1] for r in rows])
    crit[k] = {"n": len(rows), "V_mean": round(float(V.mean()), 3),
               "V_median": round(float(np.median(V)), 3),
               "V_positive_n": int((V > 0).sum()),
               "V_positive_frac": round(float((V > 0).mean()), 4),
               "pwin_mean": round(float(W.mean()), 4),
               "pwin_gt_half_n": int((W > .5).sum())}

# stats
a, b = crit["cap"]["V_positive_n"], crit["cap"]["n"] - crit["cap"]["V_positive_n"]
c, d = crit["ordinary"]["V_positive_n"], crit["ordinary"]["n"] - crit["ordinary"]["V_positive_n"]
N = a + b + c + d
hg = lambda A, B, C, D: comb(A + B, A) * comb(C + D, C) / comb(N, A + C)
fisher = sum(hg(x, a + b - x, a + c - x, d - (a - x)) for x in range(a, min(a + b, a + c) + 1))
PRE_CLOCK = 13 / 14
binom_tail = sum(comb(crit["cap"]["n"], k) * PRE_CLOCK ** k * (1 - PRE_CLOCK) ** (crit["cap"]["n"] - k)
                 for k in range(0, crit["cap"]["V_positive_n"] + 1))

out = {
    "measurement": "gen13_stall_coverage_and_critic_sign",
    "date": "2026-08-17",
    "run": G13,
    "cap_turns": CAP,
    "notes": [
        "Eval trace counts are NOT a win rate: losses are deliberately over-sampled by the eval quota.",
        "Eval denominator split by opponent class; bots end far shorter than pool sentinels, so the "
        "pooled number is not the comparison for a self-play training rollout.",
        "Value-loss MASS share NOT measured (needs train-loop instrumentation). Decision share is a "
        "LOWER bound on it, since stall residuals exceed average.",
    ],
    "training_rollout": train_block,
    "eval_turn_distribution": eval_block,
    "critic_sign_at_final_decision": crit,
    "stats": {
        "fisher_exact_one_sided_cap_vs_ordinary": round(fisher, 4),
        "P_le_observed_positives_under_preclock_rate": float(f"{binom_tail:.3e}"),
        "preclock_baseline": "13/14 positive V at the final decision of a timeout loss, pre gen3_deadline_clock_v1",
    },
}

# gen-14 early stall-rate cross-check (matched 46-min windows)
def stall_rate(run, minutes=46):
    ts = sorted(re.findall(r"(\d{8}_\d{6})", " ".join(os.listdir(f"{run}/stalls"))))
    if not ts:
        return None
    from datetime import datetime
    d0 = datetime.strptime(ts[0], "%Y%m%d_%H%M%S")
    n = sum(1 for t in ts
            if (datetime.strptime(t, "%Y%m%d_%H%M%S") - d0).total_seconds() <= minutes * 60)
    fps = re.findall(r"^\| *fps *\| *(\d+)", open(f"{run}/launcher_child.log", errors="ignore").read(), re.M)
    f = int(fps[-1]) if fps else None
    return {"stalls_in_window": n, "window_minutes": minutes, "fps_at_read": f,
            "per_1M_steps": round(n / (minutes * 60 * f / 1e6), 1) if f else None}

out["stall_rate_cross_check"] = {
    "gen13_final_segment": stall_rate(G13),
    "gen14_at_2M": stall_rate(G14),
    "caveat": "Confounded by training stage and pool strength; read only as NOT ELEVATED, never as an improvement.",
}
print(json.dumps(out, indent=2))
