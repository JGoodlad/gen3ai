"""The gathered path vs the per-row pass on real rows (CPU): max |diff| on every hypothesis slot."""
import json
import sys

import numpy as np
import torch

from ablate import build
from agents.model.hypothesis_encode import gathered_hypothesis_tokens
from agents.model.hypothesis_tokens import hypothesis_ctx
from agents.model.parity_probe import PERTURB_SCALE, PERTURB_SEED, perturb_
from agents.observation.constants import TEAM_SIZE

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DT = sys.argv[2] if len(sys.argv) > 2 else "float32"
torch.manual_seed(0)
m = build("fm")
pol = m.policy
pol.eval()
perturb_(pol, seed=PERTURB_SEED, scale=PERTURB_SCALE)
if DT == "float64":
    pol.double()
fe = pol.features_extractor
with np.load("/tmp/x5pe/real_obs.npz") as z:
    obs = {k[4:]: torch.as_tensor(z[k][:B]) for k in z.files if k.startswith("obs:")}
if DT == "float64":
    obs = {k: (v.double() if v.dtype == torch.float32 else v) for k, v in obs.items()}
with torch.no_grad():
    ctx = fe.unpack(obs)
    role = fe.pokemon_encoder(ctx, fe.embeddings)
    hs = fe._build_hypothesis_species(ctx, role)
    hctx = hypothesis_ctx(ctx, hs, fe.layout)
    ref = fe.pokemon_encoder(hctx, fe.embeddings)[:, TEAM_SIZE:]
    new = gathered_hypothesis_tokens(fe.pokemon_encoder, fe.embeddings, ctx, hs.slot_species,
                                     fe.hypothesis_builder.dex_rows)
    hyp = hs.slot_is_hypothesis
    d = (ref - new).abs()[hyp]
    rel = d.max() / ref[hyp].abs().max()
print(json.dumps({"dtype": DT, "rows": B, "hyp_slots": int(hyp.sum()), "max_abs": d.max().item(),
                  "mean_abs": d.mean().item(), "ref_max_abs": ref[hyp].abs().max().item(),
                  "rel_to_max": rel.item(), "finite_all_slots": bool(torch.isfinite(new).all())}))
