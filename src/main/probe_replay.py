"""Forensic replay probe: feed a REAL observation (captured during a battle) back
through a checkpoint and inspect why the policy chose what it did.

For a saved (summary.json, states.npz) pair and an invocation index, prints:
  - the recorded action distribution vs the live re-run (faithfulness check),
  - the active mon's per-move type-matchup the model saw,
  - an intervention sweep on the chosen move's matchup (0x..4x),
  - gradient saliency of the chosen action's logit over obs blocks.

Usage:
  python -m main.probe_replay <ckpt.zip> <battle_summary.json> <battle_states.npz> <inv_index>
"""
import sys, json
import numpy as np
import torch
from sb3_contrib import MaskablePPO

from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
import agents.observation.constants as C
from agents.action.constants import MOVE_START


def main():
    ckpt, summ_path, npz_path, inv_i = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    summ = json.load(open(summ_path))
    npz = np.load(npz_path)
    inv = summ["invocations"][inv_i]
    obs = npz["obs"][inv_i].astype(np.float32)
    if "has_state" in npz and not npz["has_state"][inv_i]:
        print(f"WARNING: invocation {inv_i} has no captured state."); return

    enc = Gen3ObservationEncoder(load_mappings())
    lay = enc.get_layout()
    RL = lay["reactive_layout"]
    MM_OFF = C.OFFSET_REACTIVE + RL["move_multiplier"]["offset"]   # active-move mults vs cur opp
    OM_OFF = C.OFFSET_REACTIVE + RL["our_matchups"]["offset"]      # [6 mon][4 move][6 opp]

    model = MaskablePPO.load(ckpt, device="cpu")
    policy = model.policy; policy.set_training_mode(False)

    # reconstruct the action mask from the summary's per-action validity
    acts = inv["actions"]
    labels = list(acts.keys())
    mask = np.array([1 if acts[k]["valid"] else 0 for k in labels], dtype=np.int8)

    def dist(o):
        ot = torch.as_tensor(o).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        d = policy.get_distribution({"observation": ot, "action_mask": mt})
        lg = d.distribution.logits.clone()
        masked = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
        return torch.softmax(masked, 1)[0].detach().numpy(), lg[0].detach().numpy()

    print(f"=== {summ_path}  inv {inv_i}  turn {inv['turn']} ===")
    print(f"our={inv['our']['species']}  opp={inv['opp']['species']}  chosen={inv['chosen']}")
    probs, logits = dist(obs)
    print("\naction            recorded   re-run(real obs)")
    for j, k in enumerate(labels):
        if mask[j]:
            print(f"  {k:18s} {acts[k]['prob']:>7}   {probs[j]*100:6.1f}%")

    # active mon = our team slot 0? find active our slot via species: assume slot 0 is active
    # (our active mon's per-Pokemon block is slot 0 only if active; otherwise we scan).
    # The reactive active-move multipliers (MM_OFF..+4) are vs the CURRENT opponent,
    # in available-moves order — directly the matchup the model saw for the active mon.
    print(f"\nactive-move type multipliers the model saw (×4 normalised): "
          f"{(obs[MM_OFF:MM_OFF+4]*4).round(2).tolist()}")

    # intervention: which move action is chosen? sweep its multiplier 0->4x.
    chosen = inv["chosen"]
    cj = labels.index(chosen)
    # labels are in action-index order (switches 0-5, moves 6-9, struggle 10), so
    # the move's request-slot — which is how vec[MM_OFF:MM_OFF+4] is ordered — is
    # exactly (action index - MOVE_START).
    if MOVE_START <= cj < MOVE_START + 4:
        slot = cj - MOVE_START
        sw_total = sum(p for j, p in enumerate(probs) if labels[j].startswith("switch"))
        print(f"\nP(all switches) at real obs = {sw_total*100:.1f}%")
        print(f"intervention — sweep '{chosen}' (request slot {slot}) multiplier:")
        for mult in [0.0, 1.0, 2.0, 4.0]:
            o = obs.copy(); o[MM_OFF + slot] = mult / 4.0
            p, _ = dist(o)
            print(f"   set {mult:.0f}x -> P({chosen})={p[cj]*100:5.1f}%  P(switches)={sum(p[j] for j in range(len(labels)) if labels[j].startswith('switch'))*100:4.1f}%")

    # saliency: d logit(chosen)/d obs by block
    ot = torch.as_tensor(obs).unsqueeze(0).requires_grad_(True)
    mt = torch.as_tensor(mask).unsqueeze(0)
    d = policy.get_distribution({"observation": ot, "action_mask": mt})
    cj = labels.index(chosen)
    d.distribution.logits[0, cj].backward()
    g = ot.grad[0].abs().numpy()
    def blk(name, lo, hi): print(f"   {name:30s} |grad|/dim={g[lo:hi].mean():.4f}  sum={g[lo:hi].sum():.2f}")
    print(f"\ngradient saliency of P({chosen}) logit (per-dim mean over ALL obs={g.mean():.4f}):")
    blk("active move_multipliers(4)", MM_OFF, MM_OFF + 4)
    blk("our_matchups(144)", OM_OFF, OM_OFF + 144)
    blk("our active pokemon block(99)", 0, 99)
    blk("turn-history block", lay["turn_history_offset"], lay["turn_history_offset"] + lay["n_history_turns"] * lay["turn_delta_dim"])


if __name__ == "__main__":
    main()
