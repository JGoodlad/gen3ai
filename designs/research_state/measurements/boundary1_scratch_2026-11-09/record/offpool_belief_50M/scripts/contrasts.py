"""SEED-LEVEL factorial read of the 12 per-run panel numbers (supplement to the meter's team-clustered table).

Input: table_per_run.json (each run its own one-checkpoint arm; its `post_minus_prior.mean` is the panel number:
the mean over the panel's 72 teams of each team's mean). Output: per cell mean / SD, then main effects and matched
cell contrasts with a t interval from the 2 x 2 residual (pooled over the four cells; 4 x (3 - 1) = 8 df), 95 %.
The meter's own interval (a bootstrap over TEAMS) does not contain seed spread; this one contains seed spread and
not team spread. Neither is the other's substitute.
"""
import json, sys, math
from scipy import stats

per = json.load(open(sys.argv[1]))
cells = {"C": ["b1c_s5101", "b1c_s5102", "b1c_s5110"], "W": ["b1w_s5103", "b1w_s5104", "b1w_s5111"],
         "Br": ["b1br_s5105", "b1br_s5106", "b1br_s5112"], "WB": ["b1wb_s5107", "b1wb_s5108", "b1wb_s5113"]}
T = float(stats.t.ppf(0.975, 8))
out = {"df": 8, "t_crit_95": T, "panels": {}}
for panel, pd in per["panels"].items():
    out["panels"][panel] = {}
    for met, md in pd["metrics"].items():
        arms = md["arms"]
        for col in ("post_minus_prior", "post", "mix0.25_minus_prior"):
            v = {c: [arms[r][col]["mean"] for r in rs] for c, rs in cells.items()}
            mean = {c: sum(x) / 3 for c, x in v.items()}
            sd = {c: math.sqrt(sum((a - mean[c]) ** 2 for a in x) / 2) for c, x in v.items()}
            mse = sum(s * s for s in sd.values()) / 4

            def con(w):  # w: cell -> coefficient on the cell mean (each cell mean is of 3 runs)
                est = sum(w[c] * mean[c] for c in w)
                se = math.sqrt(mse * sum(w[c] ** 2 for c in w) / 3)
                return {"est": est, "lo": est - T * se, "hi": est + T * se, "se": se}

            cons = {
                "breadth_main (Br+WB)-(C+W)": con({"Br": .5, "WB": .5, "C": -.5, "W": -.5}),
                "width_main (W+WB)-(C+Br)": con({"W": .5, "WB": .5, "C": -.5, "Br": -.5}),
                "breadth @128 Br-C": con({"Br": 1, "C": -1}),
                "breadth @256 WB-W": con({"WB": 1, "W": -1}),
                "width @pool W-C": con({"W": 1, "C": -1}),
                "width @breadth WB-Br": con({"WB": 1, "Br": -1}),
            }
            out["panels"][panel].setdefault(met, {})[col] = {
                "cell_values": v, "cell_mean": mean, "cell_sd": sd, "mse": mse, "contrasts": cons}
json.dump(out, open(sys.argv[2], "w"), indent=1)
