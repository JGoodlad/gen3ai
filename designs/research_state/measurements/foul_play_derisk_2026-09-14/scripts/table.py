"""Render the per-session markdown table for the README from games.jsonl."""
import json
import os
import statistics as st

ROOT = "/home/goodlad/.claude/jobs/9ab51de6/tmp/foul_play"
games = [json.loads(l) for l in open(os.path.join(ROOT, "games.jsonl"))]
by = {}
for g in games:
    s = (g.get("fp_log") or "fp_s?.log").replace("fp_s", "").replace(".log", "")
    by.setdefault(s, []).append(g)

print("| session | our team | games | our wins | win rate | mean turns | max turns | our think ms (mean) | FP think s (mean) | FP visits/decision (mean) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in sorted(by, key=lambda x: int(x) if x.isdigit() else 99):
    rs = by[s]
    w = sum(1 for r in rs if r["our_win"])
    ft = [r["fp_think_s_mean"] for r in rs if r["fp_think_s_mean"]]
    vv = [r["fp_mcts_visits_mean"] for r in rs if r["fp_mcts_visits_mean"]]
    ot = [r["our_think_ms_mean"] for r in rs if r["our_think_ms_mean"]]
    print(f"| {s} | `{rs[0]['team_file']}` | {len(rs)} | {w} | {w/len(rs):.2f} | "
          f"{st.mean(r['turns'] for r in rs):.1f} | {max(r['turns'] for r in rs)} | "
          f"{st.mean(ot):.0f} | {st.mean(ft):.2f} | {st.mean(vv)/1e6:.2f} M |")
w = sum(1 for r in games if r["our_win"])
ft = [r["fp_think_s_mean"] for r in games if r["fp_think_s_mean"]]
vv = [r["fp_mcts_visits_mean"] for r in games if r["fp_mcts_visits_mean"]]
ot = [r["our_think_ms_mean"] for r in games if r["our_think_ms_mean"]]
print(f"| **all** | 8 pool teams | **{len(games)}** | **{w}** | **{w/len(games):.3f}** | "
      f"**{st.mean(r['turns'] for r in games):.1f}** | **{max(r['turns'] for r in games)}** | "
      f"**{st.mean(ot):.0f}** | **{st.mean(ft):.2f}** | **{st.mean(vv)/1e6:.2f} M** |")
