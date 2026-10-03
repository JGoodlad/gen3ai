import numpy as np
from scipy import stats
# M4: sigma from A - A2 (same seed, cross-pin)
d, lo, hi = -11.354, -13.73, -9.00
se = (hi - lo) / (2 * 1.96); s_run = np.sqrt((d**2 - se**2) / 2)
print(f"A-A2: meter SE {se:.2f}, sigma_run {s_run:.2f}")
pairs = {"A'-A2": (4.90, 0.66), "Wb-W": (3.69, 1.40), "A-A2": (11.35, se)}
for name, sel in (("first two", ["A'-A2", "Wb-W"]), ("all three seed/pin pairs", ["A'-A2", "Wb-W", "A-A2"])):
    v = np.mean([(pairs[k][0]**2 - pairs[k][1]**2) / 2 for k in sel]); dfv = len(sel)
    print(name, "pooled sigma", round(np.sqrt(v), 2), "95% CI", [round(np.sqrt(dfv * v / stats.chi2.ppf(q, dfv)), 2) for q in (0.975, 0.025)])
lv = np.array([34.458, 45.8125, 40.917, 43.104]); print("SD of E10 levels A,A2,A',B", lv.std(ddof=1).round(2))
# M7: joint power if the outside panel were a GATE (NI at the same delta), n=8/arm fixed, panel meter 300 games/run
for sig in (2.5, 3.43, 4.8):
    for games in (300, 1200):
        sm = 50 / np.sqrt(games)
        se = np.sqrt(2 * (sig**2 + sm**2) / 8); crit = stats.t.ppf(0.95, 14)
        p_panel = 1 - stats.nct.cdf(crit, 14, 3.5 / se)
        print(f"sig {sig} panel {games} games/run (meter SE {sm:.1f}): panel NI power {p_panel:.2f}")
