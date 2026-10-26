"""E1-0 POST HOC (written after the read; DESCRIPTIVE, no verdict): the split-half noise ceiling of the referees' rank
agreement over actions. Per bot state, Spearman over legal actions between T on 12 seeds and T on the other 12: self vs
self (seeds 16..27 vs 28..39), self vs bot (16..27 vs the bot's last 12), bot vs bot (first 12 vs last 12)."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_e10 as S
DEC = {"+": 1.0, "-": 0.0, "0": 0.5}
truth = {}
for l in open(Path.home()/"gen3ai_archive/hidden_info_maturity/rows/x26_225M/O/shard_0_of_1.jsonl"):
    r = json.loads(l); truth[r["id"]] = r
bot = {json.loads(l)["id"]: json.loads(l) for l in open(Path.home()/"gen3ai_archive/search_e1_0/rows/bot/rows.jsonl")}
ss, sb, sbb = [], [], []
for i, b in bot.items():
    t = truth[i]; acts = sorted(t["out"], key=int)
    m = lambda s: np.mean([DEC[c] for c in s])
    a1 = np.array([m(t["out"][a][16:28]) for a in acts]); a2 = np.array([m(t["out"][a][28:40]) for a in acts])
    b1 = np.array([m(b["out"][a][:12]) for a in acts]); b2 = np.array([m(b["out"][a][12:]) for a in acts])
    for lst, x, y in ((ss, a1, a2), (sb, a1, b2), (sbb, b1, b2)):
        v = S.spearman(x, y)
        if v is not None: lst.append(v)
for n, l in (("self half vs self half", ss), ("self half vs bot half", sb), ("bot half vs bot half", sbb)):
    print(n, "states", len(l), "median", round(float(np.median(l)), 3), "mean", round(float(np.mean(l)), 3))
