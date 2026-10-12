"""Summarise the two policy_drift series into policy_drift_summary.json (numpy only; reads the out dirs)."""
import json
import sys
from pathlib import Path

import numpy as np

EPS = 1e-12
EDGES = (0.1, 0.3)
LABS = ["<0.1", "0.1-0.3", ">0.3"]
ARCH = Path.home() / "gen3ai_archive" / "policy_drift"
RUNS = {"rb_es_end2_s2001": "62085750", "rb_es_prod_s2001": "95d014fa"}


def kl(pa, pb, m):
    pa = np.where(m, pa, 0.0); pb = np.where(m, pb, 0.0)
    return (pa * (np.log(pa + EPS) - np.log(pb + EPS))).sum(-1)


def greedy(p, m):
    return np.where(m, p, -1.0).argmax(-1)


def margin(p, m):
    q = np.sort(np.where(m, p, 0.0), axis=-1)
    return q[:, -1] - q[:, -2]


def ref_block(p_new, p_old, m, choice, dstep):
    k = kl(p_new, p_old, m)
    fl = greedy(p_old, m) != greedy(p_new, m)
    idx = np.digitize(margin(p_old, m), EDGES)
    per = 1e6 / dstep
    b = {}
    for i, lab in enumerate(LABS):
        s = choice & (idx == i)
        b[lab] = {"n": int(s.sum()), "flip_rate": round(float(fl[s].mean()), 4) if s.any() else None}
    return {"delta_steps": int(dstep),
            "kl_mean": round(float(k.mean()), 5), "kl_median": round(float(np.median(k)), 5),
            "kl_mean_choice_states": round(float(k[choice].mean()), 5),
            "kl_mean_per_1M": round(float(k.mean()) * per, 5), "kl_median_per_1M": round(float(np.median(k)) * per, 5),
            "flip_rate": round(float(fl[choice].mean()), 4), "flip_rate_per_1M": round(float(fl[choice].mean()) * per, 4),
            "flip_by_old_margin": b}


def run_summary(run):
    od = ARCH / run
    meta = json.loads((od / "meta.json").read_text())
    probe = json.loads((od / "probe.json").read_text())
    with np.load(od / "probe.npz") as z:
        m = z["mask"] > 0.5
    choice = m.sum(-1) >= 2
    probs = {int(f.stem): np.load(f)["probs"] for f in sorted((od / "probs").glob("*.npz"))}
    rows = {int(json.loads(l)["step"]): json.loads(l) for l in (od / "rows.jsonl").read_text().splitlines() if l.strip()}
    steps = sorted(probs)
    anchor = steps[0]
    out = {"pin": RUNS[run], "out_dir": str(od), "probe_sha256": meta["probe_sha256"], "probe_states": int(len(m)),
           "probe_choice_states": int(choice.sum()), "probe_source": probe["source_ckpt"],
           "probe_battles": probe["battles"], "probe_seed_base": probe["seed_base"],
           "steps_recorded": steps, "row_status": {str(s): rows[s].get("status") for s in sorted(rows)},
           "anchor_step": anchor, "per_checkpoint": {}}
    max_dev = 0.0
    for i, s in enumerate(steps):
        if s < 4_500_000:
            continue
        prev = steps[i - 1]
        back_target = s - 10_000_000
        back = min(steps, key=lambda x: abs(x - back_target)) if back_target > 0 else None
        e = {"prev": dict(ref_block(probs[s], probs[prev], m, choice, s - prev), step=prev)}
        # cross-check against the tool's own row (same prev, same probe)
        tr = (rows[s].get("refs") or {}).get("prev")
        if tr and tr["step"] == prev:
            max_dev = max(max_dev, abs(tr["kl_mean"] - float(kl(probs[s], probs[prev], m).mean())))
        if back is not None:
            e["back_10M_nearest"] = dict(ref_block(probs[s], probs[back], m, choice, s - back), step=back)
        tb = (rows[s].get("refs") or {}).get("back")
        e["tool_back_ref_step"] = tb["step"] if tb else None
        e["anchor"] = dict(ref_block(probs[s], probs[anchor], m, choice, s - anchor), step=anchor)
        e["tool_verdict"] = rows[s].get("verdict")
        out["per_checkpoint"][str(s)] = e
    out["max_abs_dev_kl_prev_vs_tool_row"] = max_dev
    win = [str(s) for s in steps if 10_500_000 <= s <= 15_500_000]
    def avg(key):
        return round(float(np.mean([out["per_checkpoint"][s]["prev"][key] for s in win])), 5)
    out["avg_10_15M_per_1M"] = {"checkpoints": win, "kl_mean": avg("kl_mean_per_1M"),
                                "kl_median": avg("kl_median_per_1M"), "flip_rate": avg("flip_rate_per_1M")}
    win5 = [str(s) for s in steps if 5_500_000 <= s <= 10_500_000]
    out["avg_5_10M_per_1M"] = {"checkpoints": win5,
                               **{k: round(float(np.mean([out["per_checkpoint"][s]["prev"][kk] for s in win5])), 5)
                                  for k, kk in (("kl_mean", "kl_mean_per_1M"), ("kl_median", "kl_median_per_1M"),
                                                ("flip_rate", "flip_rate_per_1M"))}}
    return out


def main():
    res = {"what": "POLICY DRIFT (main.policy_drift, a DESCRIPTOR) over the two closing-test seeds' periodic "
                   "checkpoints, each loaded at its own pin; masked KL(current || ref) in nats over legal actions "
                   "(single-legal states included in kl_mean/median, as the tool does; kl_mean_choice_states "
                   "excludes them); flip = greedy action changed, over states with >= 2 legal actions, bucketed by "
                   "the OLDER policy's top1-top2 margin; per_1M = value * 1e6 / delta_steps (a linear "
                   "normalisation; consecutive checkpoints are ~1M apart so it is ~ the raw value). "
                   "back_10M_nearest = the recorded checkpoint nearest step-10M (the tool's own 'back' ref is the "
                   "latest at least 10M back, absent where the 1M checkpoint is a few hundred steps too late).",
           "commands": sys.argv[1:] and json.loads(Path(sys.argv[1]).read_text()) or None,
           "runs": {r: run_summary(r) for r in RUNS}}
    out = Path("/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a80c4703e73d9d0fa/designs/research_state/"
               "measurements/hidden_info_binding_2026-10-11/policy_drift_summary.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    for r, v in res["runs"].items():
        print(r, json.dumps(v["avg_10_15M_per_1M"]), json.dumps(v["avg_5_10M_per_1M"]))
        for s, e in v["per_checkpoint"].items():
            p = e["prev"]
            print(f"  {int(s):>11,} KLprev {p['kl_mean']:.4f}/{p['kl_median']:.4f} flip {p['flip_rate']:.3f} "
                  + " ".join(f"{k}:{b['flip_rate']}" for k, b in p["flip_by_old_margin"].items())
                  + (f" | back@{e['back_10M_nearest']['step']} KL {e['back_10M_nearest']['kl_mean']:.4f} flip "
                     f"{e['back_10M_nearest']['flip_rate']:.3f}" if "back_10M_nearest" in e else "")
                  + f" | anchor KL {e['anchor']['kl_mean']:.4f} flip {e['anchor']['flip_rate']:.3f}")


if __name__ == "__main__":
    main()
