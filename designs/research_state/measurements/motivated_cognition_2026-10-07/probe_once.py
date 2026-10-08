"""Feasibility probe (no results): which stashes / ctx fields exist at the pin, on a 3-battle bank slice."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

from agents.model.extra_obs_keys import zero_extra_obs
from main.belief_roles.__main__ import default_bank
from main.belief_roles.bank_rows import bank_rows_of
from main.belief_roles.forward import load_strict
from main.policy_spectrum.bank import load_bank

bank = load_bank(default_bank())
print("decisions", len(bank.decisions), "battles", len(bank.battles))
ids = {b.battle_id for b in bank.battles[:3]}
small = dataclasses.replace(bank, battles=bank.battles[:3],
                            decisions=[d for d in bank.decisions if d["battle"] in ids])
br = bank_rows_of(small, workers=1)
print("rows", br.rows.shape)
for z in sys.argv[1:]:
    m = load_strict(Path(z))
    fe = m.policy.features_extractor
    print(z, [n for n, _ in fe.named_children()])
    rows = torch.tensor(br.rows[:8])
    masks = torch.tensor(br.masks[:8].astype(np.float32))
    ob = {"observation": rows, "action_mask": masks}
    ob.update(zero_extra_obs(fe, batch=8))
    with torch.no_grad():
        ctx = fe.unpack(ob)
        dist = m.policy.get_distribution(ob)
    st = fe.stash
    print(" probs", dist.distribution.probs.shape)
    for k in ["move_belief_logits", "item_logits", "hp_type_logits", "spread_nature_logits", "spread_ev"]:
        v = getattr(st, k)
        print(" ", k, None if v is None else tuple(v.shape))
    print("  belief_logits", {k: tuple(v.shape) for k, v in (st.belief_logits or {}).items()})
    print("  ctx", ctx.species_ids.shape, ctx.item_ids.shape, ctx.our_active_idx[:4].tolist(),
          ctx.opp_active_local[:4].tolist(), ctx.all_move_ids.shape)
    print("  species row0", ctx.species_ids[0].tolist(), "items", ctx.item_ids[0].tolist(),
          "believed", ctx.opp_believed_mask[0].tolist())
    print("  tokens", small.decisions[0]["tokens"], small.decisions[0]["mask"])
