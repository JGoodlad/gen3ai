"""Attribute x5_opp_mon_axis_test's 'no near-tie row on the golden obs' precondition: the unperturbed fixed_mass
learner's per-row move-selection tie gap on a given BUFFER, with the move prior optionally HELD at a file's values."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--buffer", required=True)
ap.add_argument("--prior", default=None)
a = ap.parse_args()
from agents.gen3_data import priors as P  # noqa: E402
if a.prior:
    t = json.loads(Path(a.prior).read_text())
    P.move_raw = lambda: t
import copy  # noqa: E402
from agents.model.hypothesis_set_test import _unperturbed_learner  # noqa: E402
from agents.model.hypothesis_set import near_tie_rows  # noqa: E402
from main.train.production_args import production_args  # noqa: E402
args = production_args()
args.belief_tokens = "fixed_mass"
fe = copy.deepcopy(_unperturbed_learner(args).policy).features_extractor.eval()
with np.load(a.buffer) as z:
    obs = {k[4:]: torch.as_tensor(z[k].reshape(-1, *z[k].shape[2:])[:64]) for k in z.files if k.startswith("obs:")}
with torch.no_grad():
    fe(obs)
hs = fe.last_hypothesis
g = hs.slot_moves_tie_gap
print(json.dumps({"buffer": a.buffer, "prior": a.prior or "tree", "near_tie_rows": int(near_tie_rows(hs).sum()),
                  "min_gap": float(g[torch.isfinite(g)].min())}))
