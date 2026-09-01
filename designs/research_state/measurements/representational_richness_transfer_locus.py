"""M2 / part 2 — DELTA LOCUS: where a FOLD's parameter change lands (trunk vs head).

The plasticity forensics (`plasticity_forensics_v8_vs_gen_2026-08-28.md`, Phase A) measured this
for the EXPLOITER FORKS and found v8 trunk-heavy (G2 share 0.465) vs the gen era (0.278). Nobody
measured it for the FOLDS — which is the object the untaught-externality outcome is a property of.

This script reproduces that estimator verbatim (same group map, same SB3 triple-copy dedupe, same
share-of-total-squared-delta statistic) and applies it to fold-minus-parent pairs, plus the fork
anchors so the new numbers can be read on the old scale.

Run:  nice -n 15 python designs/research_state/measurements/representational_richness_transfer_locus.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Reads models/ READ-ONLY. Writes /tmp/m2rich/locus.json
"""
import io
import json
import os
import zipfile
from collections import OrderedDict

import torch

MODELS = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")
OUT = "/tmp/m2rich/locus.json"

# ---- group map: VERBATIM from tmp/plast_phaseA.py (2026-08-28) so the shares are comparable ----
GROUPS = [
    ("G1_input_encoders", [
        "embeddings", "pokemon_encoder", "entity_seats", "history_events",
        "zarch_encoder", "edge_bias", "damage_op", "assembler",
        "status_in_proj", "status_out_proj", "outgoing_proj", "refine_proj",
        "prefuse_proj",
    ]),
    ("G2_trunk", [
        "projection", "team_transformer", "pre_proj_norm", "film_pi", "film_vf",
    ]),
    ("G3_pools_value_trunk", [
        "cls_pool", "value_entity_pool", "value_projection", "value_pre_norm",
    ]),
    ("G4_aux_belief_heads", [
        "belief_head", "belief_slots", "move_belief", "spread_belief",
        "hp_type_belief_head", "item_belief_head", "hidden_opp_belief",
        "alpha_head", "beta_head", "cf_evid_head", "cf_shadow_head",
        "cf_twin_head_b", "cf_twin_head_c", "intent_conditional",
        "intent_move_cell", "intent_threshold_move", "pair_outcome_move",
        "pair_outcome_switch", "switch_branch", "conditional_threat",
        "win_head", "value_dist_head",
    ]),
]
TOP_GROUPS = [
    ("G5_mlp_extractor", ["mlp_extractor.policy_net", "mlp_extractor.value_net"]),
    ("G6_action_value_head", ["action_net", "pointer_head", "value_net"]),
]
SKIP_TOP = ("popart",)

# The two aggregate readings the hypothesis is stated in.
TRUNK_GROUPS = ("G2_trunk",)                       # the 0.47-vs-0.28 statistic
SHARED_GROUPS = ("G2_trunk", "G5_mlp_extractor")   # v8's 0.76
HEAD_GROUPS = ("G4_aux_belief_heads", "G6_action_value_head")


def group_of(key):
    if key.startswith("features_extractor."):
        sub = key[len("features_extractor."):]
        head = sub.split(".")[0]
        for gname, pats in GROUPS:
            if head in pats:
                return gname
        return "G0_unmapped_fe:" + head
    for gname, pats in TOP_GROUPS:
        for p in pats:
            if key.startswith(p):
                return gname
    if key.startswith(SKIP_TOP):
        return None
    return "G0_unmapped_top:" + key.split(".")[0]


def load_sd(path):
    z = zipfile.ZipFile(path)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu",
                    weights_only=False)
    out = OrderedDict()
    for k, v in sd.items():
        if k.startswith(("pi_features_extractor.", "vf_features_extractor.")):
            continue
        if not torch.is_floating_point(v):
            continue
        out[k] = v.float()
    return out


def num_timesteps(path):
    try:
        z = zipfile.ZipFile(path)
        return json.loads(z.read("data").decode("utf-8", "replace")).get("num_timesteps")
    except Exception:  # noqa: BLE001
        return None


def delta_profile(parent_sd, child_sd):
    shared = [k for k in parent_sd if k in child_sd
              and parent_sd[k].shape == child_sd[k].shape]
    g_d2, g_p2, g_n = {}, {}, {}
    for k in shared:
        g = group_of(k)
        if g is None:
            continue
        d = child_sd[k] - parent_sd[k]
        g_d2[g] = g_d2.get(g, 0.0) + float((d * d).sum())
        g_p2[g] = g_p2.get(g, 0.0) + float((parent_sd[k] * parent_sd[k]).sum())
        g_n[g] = g_n.get(g, 0) + d.numel()
    tot_d2 = sum(g_d2.values()) or 1e-30
    tot_p2 = sum(g_p2.values()) or 1e-30
    tot_n = max(1, sum(g_n.values()))
    prof = {g: {"rel_frob": (g_d2[g] ** 0.5) / (g_p2[g] ** 0.5 + 1e-12),
                "share_of_total_d2": g_d2[g] / tot_d2,
                "share_of_total_params": g_n[g] / tot_n,
                "n_params": g_n[g]} for g in sorted(g_d2)}
    agg = {
        "trunk_share": sum(g_d2.get(g, 0.0) for g in TRUNK_GROUPS) / tot_d2,
        "shared_share": sum(g_d2.get(g, 0.0) for g in SHARED_GROUPS) / tot_d2,
        "head_share": sum(g_d2.get(g, 0.0) for g in HEAD_GROUPS) / tot_d2,
    }
    return {
        "groups": prof,
        "aggregate": agg,
        "global_rel_frob": (tot_d2 ** 0.5) / (tot_p2 ** 0.5 + 1e-12),
        "n_shared_keys": len(shared),
        "n_parent_only_keys": len([k for k in parent_sd if k not in child_sd]),
        "n_child_only_keys": len([k for k in child_sd if k not in parent_sd]),
    }


def M(run, leaf):
    return f"{MODELS}/{run}/{leaf}"


def ck(run, step):
    return f"{MODELS}/{run}/checkpoints/checkpoint_{step}_steps.zip"


FIN = "final_model.zip"
FINI = "final_model_interrupted.zip"

# (label, parent_path, child_path, kind, untaught_externality_pp_or_None)
PAIRS = [
    # ---------------- FOLDS: the objects the externality is a property of ----------------
    ("FOLD v8_14 (3 teachers / 22 taught teams)",
     M("ai_v8_04_distill_4teacher_0722", FINI),
     M("ai_v8_14_distill3_0725", FINI), "fold", +5.42),
    ("FOLD rev-2 R2ACTION (5 teachers / 9 taught)",
     M("ai_v9_29_rev1_0823", FIN),
     M("ai_v9_59_R2ACTION_0827", FIN), "fold", -7.06),
    ("FOLD rev-3 R3ACTION (6 teachers / 12 taught)",
     M("ai_v9_59_R2ACTION_0827", FIN),
     M("ai_v9_70_R3ACTION_0828", FIN), "fold", -0.75),
    ("FOLD rev-4 R4ACTION (3 teachers)",
     M("ai_v9_59_R2ACTION_0827", FIN),
     M("ai_v9_76_R4ACTION_0830", FIN), "fold", None),
    ("FOLD COMPFOLD (composite)",
     M("ai_v9_59_R2ACTION_0827", FIN),
     M("ai_v9_91_COMPFOLD_0831", FIN), "fold", None),
    # ---------------- CONTROLS: what ordinary continued training's locus looks like -------
    ("CTRL R2CTRL (no fork, no distill, +3M)",
     M("ai_v9_29_rev1_0823", FIN),
     M("ai_v9_58_R2CTRL_0827", FIN), "control", None),
    ("CTRL R2PLAIN (+3M)",
     M("ai_v9_29_rev1_0823", FIN),
     M("ai_v9_62_R2PLAIN_0827", FIN), "control", None),
    # ---------------- FORK anchors: reproduce the published 0.465 / 0.278 -----------------
    ("FORK v8 semistall3 @matched",
     M("ai_v8_04_distill_4teacher_0722", FINI),
     ck("ai_v8_06_semistall_3team_exploiter_0722", 280748576), "fork_anchor", None),
    ("FORK v8 pool10 @matched",
     M("ai_v8_04_distill_4teacher_0722", FINI),
     ck("ai_v8_09_pool10_exploiter_0723", 280728930), "fork_anchor", None),
    ("FORK v8 defensive10 @matched",
     M("ai_v8_04_distill_4teacher_0722", FINI),
     ck("ai_v8_13_defensive10_exploiter_0725", 280716057), "fork_anchor", None),
    ("FORK gen F5a", M("ai_v9_29_rev1_0823", FIN), M("ai_v9_53_R2F5a_0826", FIN), "fork_anchor", None),
    ("FORK gen F5b", M("ai_v9_29_rev1_0823", FIN), M("ai_v9_54_R2F5b_0826", FIN), "fork_anchor", None),
    ("FORK gen F5c", M("ai_v9_29_rev1_0823", FIN), M("ai_v9_55_R2F5c_0826", FIN), "fork_anchor", None),
    ("FORK gen F5d", M("ai_v9_29_rev1_0823", FIN), M("ai_v9_56_R2F5d_0826", FIN), "fork_anchor", None),
    ("FORK gen F5e", M("ai_v9_29_rev1_0823", FIN), M("ai_v9_57_R2F5e_0826", FIN), "fork_anchor", None),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cache = {}

    def sd(p):
        if p not in cache:
            cache[p] = load_sd(p)
        return cache[p]

    res = {}
    for label, ppath, cpath, kind, ext in PAIRS:
        if not (os.path.exists(ppath) and os.path.exists(cpath)):
            res[label] = {"MISSING": [ppath, cpath]}
            print(f"MISSING {label}")
            continue
        prof = delta_profile(sd(ppath), sd(cpath))
        prof.update(kind=kind, parent=ppath, child=cpath,
                    parent_steps=num_timesteps(ppath),
                    child_steps=num_timesteps(cpath),
                    untaught_externality_pp=ext)
        res[label] = prof
        a = prof["aggregate"]
        print(f"{label:52s} trunk={a['trunk_share']:.3f} shared={a['shared_share']:.3f} "
              f"head={a['head_share']:.3f} relF={prof['global_rel_frob']:.4f} "
              f"keys={prof['n_shared_keys']}(+{prof['n_parent_only_keys']}p/"
              f"{prof['n_child_only_keys']}c) ext={ext}")
        # keep only ONE parent state_dict warm at a time
        if len(cache) > 3:
            for k in list(cache)[:-2]:
                if k != ppath:
                    del cache[k]
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
