"""Roll `component_cost.py`'s per-label attribution up into the COMPONENTS of design_compute_efficiency.md §2.

    python read_cost.py result/cost_production.json [--section micro_step|surrogate|infer_64|infer_256] [--labels]

Prints, per component: params, forward / backward FLOPs per row (MFLOP), the share of profiled CPU time
(forward self time + backward subtree time), and the share of counted FLOPs. Read-only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

#: (component, regex on the label). FIRST match wins; order matters.
RULES: List[Tuple[str, str]] = [
    ("edges: cell kernels (op physics -> edge cells)",
     r"meth:damage_op\.(pairwise_|discrete_(outgoing|incoming)_status|pair_outcome_coords)|meth:fe\._other_edge_cells"),
    ("edges: bias maps (cells -> [B,4,T,T])", r"mod:fe\.edge_bias"),
    ("damage op: incoming / content / facts",
     r"mod:fe\.(damage_op|op_content|prefuse_proj|op_worst_proj|mon_hazard_proj|eot_residual_rule|eot_residual_proj|"
     r"status_cure_rule|status_cure_proj|effective_stats_proj)|meth:damage_op\.|meth:fe\.(_op_content_rows|_spread_hp_damage)"),
    ("move resolution", r"mod:fe\.move_resolution_cell|meth:fe\._flat_consumer_ops"),
    ("opponent intent (flat pointer)", r"mod:fe\.flat_intent_head"),
    ("beliefs: hypothesis tokens + T0 prior",
     r"mod:fe\.(hypothesis_builder|t0_species_prior)|meth:fe\.(_build_hypothesis_species|_attach_move_group)"),
    ("beliefs: move / HP-type / item / spread heads",
     r"mod:fe\.(move_belief|hp_type_belief_head|item_belief_head|spread_belief)|meth:fe\.(_apply_move_belief|_typed_hp_posterior)"),
    ("beliefs: BeliefHead aux readout", r"mod:fe\.belief_head"),
    ("beliefs: HiddenOppBeliefPool", r"mod:fe\.hidden_opp_belief"),
    ("entity seats (E3/E4/E5)", r"mod:fe\.entity_seats"),
    ("history: event seats encoder", r"mod:fe\.history_events"),
    ("trunk round 1", r"mod:fe\.team_transformer\.transformer_layers\.0"),
    ("trunk round 2", r"mod:fe\.team_transformer\.transformer_layers\.1"),
    ("trunk round 3 (extra)", r"mod:fe\.team_transformer\.extra_rounds"),
    ("trunk: glue (token assembly, bias build)", r"mod:fe\.team_transformer"),
    ("static per-mon tokens + board + obs facts",
     r"mod:fe\.(pokemon_encoder|embeddings|unpack|obs_facts_inject|move_actor_proj|move_target_proj)"),
    ("heads: CLS pools", r"mod:fe\.cls_pool"),
    ("heads: critic (win head + value_entity_pool)", r"mod:fe\.(win_head|value_entity_pool)"),
    ("heads: policy tower (pre_proj_norm, projection, mlp_extractor)",
     r"mod:fe\.(assembler|pre_proj_norm|projection|activation|policy_query)|mod:mlp_extractor"),
    ("heads: pointer action head", r"mod:pointer_head"),
    ("extractor glue (forward_internal itself)", r"mod:fe$"),
    ("losses / masking / optimizer glue", r"\(outside|\(accumulate_grad|\(backward: unmatched"),
]


def component(label: str) -> str:
    for name, rx in RULES:
        if re.search(rx, label):
            return name
    return "OTHER: " + label


def roll(by_label: Dict[str, Dict[str, float]], rows: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for lab, d in by_label.items():
        c = out[component(lab)]
        for k, v in d.items():
            c[k] += v
    tf = sum(d.get("fwd_self_us", 0) for d in out.values()) or 1
    tb = sum(d.get("bwd_us", 0) for d in out.values()) or 1
    tfl = sum(d.get("fwd_flops", 0) + d.get("bwd_flops", 0) for d in out.values()) or 1
    for d in out.values():
        d["fwd_mflop_row"] = d.get("fwd_flops", 0) / rows / 1e6
        d["bwd_mflop_row"] = d.get("bwd_flops", 0) / rows / 1e6
        d["fwd_time_share"] = d.get("fwd_self_us", 0) / tf
        d["bwd_time_share"] = d.get("bwd_us", 0) / tb
        d["time_share"] = (d.get("fwd_self_us", 0) + d.get("bwd_us", 0)) / (tf + tb)
        d["flop_share"] = (d.get("fwd_flops", 0) + d.get("bwd_flops", 0)) / tfl
    return {k: dict(v) for k, v in out.items()}


def params_by_component(params: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for lab, n in params.items():
        if lab.startswith("("):
            continue
        out[component(lab)] += n
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--section", default="micro_step")
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.path))
    sec = a.section
    rows = int(sec.split("_")[1]) if sec.startswith("infer_") else d["rows"]
    if sec + "_by_label" not in d:
        raise SystemExit(f"no section {sec!r} in {a.path}")
    r = roll(d[sec + "_by_label"], rows)
    pc = params_by_component(d["params"])
    print(f"{d['model']}  section={sec} rows={rows} params={d['params'].get('(total, unique)')}")
    for k in sorted(d):
        if k.startswith(sec) and ("unprofiled" in k or "wall" in k):
            print(f"  {k}: {d[k]}")
    tot_f = sum(v["fwd_mflop_row"] for v in r.values())
    tot_b = sum(v["bwd_mflop_row"] for v in r.values())
    print(f"  counted MFLOP/row: forward {tot_f:.2f}  backward {tot_b:.2f}")
    print(f"{'component':62s} {'params':>9s} {'fwdMF':>7s} {'bwdMF':>7s} {'FLOP%':>6s} {'fwdT%':>6s} {'bwdT%':>6s} {'T%':>6s} {'aten':>6s}")
    for k, v in sorted(r.items(), key=lambda kv: -kv[1]["time_share"]):
        print(f"{k[:62]:62s} {pc.get(k, 0):9d} {v['fwd_mflop_row']:7.2f} {v['bwd_mflop_row']:7.2f} "
              f"{100 * v['flop_share']:6.1f} {100 * v['fwd_time_share']:6.1f} {100 * v['bwd_time_share']:6.1f} "
              f"{100 * v['time_share']:6.1f} {v.get('fwd_aten_calls', 0):6.0f}")
    if a.labels:
        for lab in sorted(d[sec + "_by_label"]):
            print("   ", component(lab)[:30], "<-", lab)


if __name__ == "__main__":
    main()
