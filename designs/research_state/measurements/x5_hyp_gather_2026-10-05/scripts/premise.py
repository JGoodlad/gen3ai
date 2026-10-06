"""PREMISE: which inputs reach the hypothesis PokemonEncoder encoding? (CPU, real rows.)

1. the SAME species under many real rows' contexts: are the hypothesis tokens bitwise equal?
2. the same rows with every ROW-LEVEL input of the encoder (turn/clock, weather, fainted, hazards,
   screens, the opponent's active context) set to ONE row's values: are they bitwise equal then?
3. is a hypothesis slot ever the opponent's active (the active-context scatter)?
"""
import dataclasses
import json
import sys

import numpy as np
import torch

from ablate import build
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


def per_species_spread(tok, hs):
    hyp = hs.slot_is_hypothesis
    sp = hs.slot_species[hyp]
    t = tok[:, TEAM_SIZE:][hyp]
    worst, n_multi, n_species_diff = 0.0, 0, 0
    for s in torch.unique(sp).tolist():
        ts = t[sp == s]
        if ts.shape[0] < 2:
            continue
        n_multi += 1
        d = (ts - ts[:1]).abs().max().item()
        worst = max(worst, d)
        n_species_diff += int(d > 0)
    return {"hidden_slots": int(hyp.sum()), "species_with_2plus": n_multi,
            "species_not_bitwise_equal": n_species_diff, "max_abs_diff": worst}


with torch.no_grad():
    ctx = fe.unpack(obs)
    role = fe.pokemon_encoder(ctx, fe.embeddings)
    hs = fe._build_hypothesis_species(ctx, role)
    hctx = hypothesis_ctx(ctx, hs, fe.layout)
    tok = fe.pokemon_encoder(hctx, fe.embeddings)
    r1 = per_species_spread(tok, hs)

    def const(x):
        return x[:1].expand_as(x).contiguous()
    hctx2 = dataclasses.replace(
        hctx, turn_feature=const(hctx.turn_feature), weather_feature=const(hctx.weather_feature),
        fainted_feature=const(hctx.fainted_feature), spikes_feature=const(hctx.spikes_feature),
        screen_feature=const(hctx.screen_feature), opp_ctx_raw=const(hctx.opp_ctx_raw))
    tok2 = fe.pokemon_encoder(hctx2, fe.embeddings)
    r2 = per_species_spread(tok2, hs)
    act = torch.nn.functional.one_hot(ctx.opp_active_local, TEAM_SIZE).bool()
    r3 = {"hypothesis_slot_is_opp_active": int((act & hs.slot_is_hypothesis).sum()), "rows": B}
    # which row-level inputs vary across these rows at all
    var = {k: bool((getattr(hctx, k) != getattr(hctx, k)[:1]).any())
           for k in ("turn_feature", "weather_feature", "fainted_feature", "spikes_feature",
                     "screen_feature", "opp_ctx_raw")}
out = {"dtype": DT, "real_contexts": r1, "row_level_inputs_held_constant": r2, "active": r3,
       "row_level_inputs_vary": var}
print(json.dumps(out, indent=1))
