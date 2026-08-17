"""GENERATED tables for designs/ARCHITECTURE.md — modules, head inputs, and the flag table.

WHY GENERATED. The hand-derived §2 module list, §3 head-input tables and §6 flag table went stale
twice within one week of generation turnover (the file itself records both). Like
`delivery_graph.py`, this module INSTANTIATES `Gen3FeaturesExtractor` from
`designs/production_config.json` resolved against HEAD and reads the tables off the live instance,
so a table that disagrees with the code is a failing test (`arch_tables_test.py`) rather than prose
someone eventually notices is wrong.

It rewrites ONLY the content between the three marker pairs in `designs/ARCHITECTURE.md`:

    <!-- BEGIN GENERATED: modules -->      / <!-- END GENERATED: modules -->
    <!-- BEGIN GENERATED: head-inputs -->  / <!-- END GENERATED: head-inputs -->
    <!-- BEGIN GENERATED: flag-table -->   / <!-- END GENERATED: flag-table -->

Everything outside the markers is hand-written and never touched.

Usage:
    python -m agents.model.arch_tables            # rewrite the marker sections in place
    python -m agents.model.arch_tables --check    # exit 1 + diff summary if the committed
                                                  # sections no longer match HEAD (staleness gate)
"""
from __future__ import annotations

import argparse
import difflib
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

# The same committed config every derived artifact keys on (delivery graph, arch viewer,
# compile gate). Reused rather than re-derived so the artifacts cannot disagree on the source.
from agents.model.delivery_graph import _DEFAULT_CONFIG

_ARCH_MD = os.path.join(os.path.dirname(_DEFAULT_CONFIG), "ARCHITECTURE.md")

SECTIONS = ("modules", "head-inputs", "flag-table")

# Optional modules worth calling ABSENT by name in §2 when the config does not build them.
# (label as shown, dotted attribute on the instance)
_ABSENT_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("belief_slots", "belief_slots"),
    ("belief_head", "belief_head"),
    ("spread_belief", "spread_belief"),
    ("hidden_opp_belief", "hidden_opp_belief"),
    ("win_head", "win_head"),
    ("value_dist_head", "value_dist_head"),
    ("alpha_head", "alpha_head"),
    ("beta_head", "beta_head"),
    ("intent_value_reduce", "intent_value_reduce"),
    ("value_threat_inject", "cls_pool.value_threat_proj"),
)

# Architecture toggle -> the module/effect that proves it is doing work on the instance.
# Only toggles whose presence IS a module are mapped; pure knobs (modes, k values, floors)
# have no single attribute and stay unmapped (truthy => ACTIVE).
_TOGGLE_MODULE: Dict[str, str] = {
    "damage_op": "damage_op",
    "damage_matrices_incoming": "damage_op",
    "damage_outgoing": "damage_op",
    "edge_bias_families": "edge_bias",
    "entity_tail_seats": "entity_seats",
    "entity_topk_seats": "entity_seats",
    "move_belief_mode": "move_belief",
    "hp_belief_mode": "hp_type_belief_head",
    "opp_belief_slots": "belief_slots",
    "opp_belief_cls_k": "hidden_opp_belief",
    "opp_intent": "alpha_head",
    "opp_intent_grad_mode": "alpha_head",
    "species_prior_fusion": "belief_head",
    "spread_belief": "spread_belief",
    "spread_belief_nature": "spread_belief",
    "win_prob_mode": "win_head",
    "value_dist_mode": "value_dist_head",
    "value_dist_bins": "value_dist_head",
    "value_dist_vmin": "value_dist_head",
    "value_dist_vmax": "value_dist_head",
    "value_threat_inject": "cls_pool.value_threat_proj",
    "intent_value_reduce": "intent_value_reduce",
}

# Training-loss coefficient -> the module that consumes it. THE INERT LOGIC LIVES HERE, in one
# reviewable dict: a nonzero coef whose module is None is INERT (head not built); a zero coef
# whose module exists is INERT (the loss is off, the module trains through other paths).
# `None` marks a core train-loop term with no gating module (vf_coef, value_tail_weight).
_COEF_MODULE: Dict[str, Optional[str]] = {
    "move_belief_coef": "move_belief",
    "move_belief_latent_coef": "move_belief",
    "hp_type_belief_coef": "hp_type_belief_head",
    "opp_belief_aux_coef": "belief_head",
    "spread_belief_coef": "spread_belief",
    "win_prob_coef": "win_head",
    "value_dist_coef": "value_dist_head",
    "item_belief_coef": "item_belief_head",     # v83
    "vf_coef": None,
    "value_tail_weight": None,
}

_FALSY_STRINGS = {"none", "off", ""}


def _is_off(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in _FALSY_STRINGS
    return not bool(value)


def _resolve(fe: Any, dotted: str) -> Any:
    obj = fe
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# --------------------------------------------------------------------------------- instantiation
def load_config(config_path: str = _DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(config_path) as fh:
        return cast(Dict[str, Any], json.load(fh))


def build_extractor(config_path: str = _DEFAULT_CONFIG) -> Tuple[Any, Dict[str, Any]]:
    """The same build seam `delivery_graph.build_graph` and the compile gate use: config fields
    filtered by the live constructor signature, historical floor reconciled at construction.

    DELEGATES to `delivery_graph.build_extractor` rather than repeating it — two copies of a
    construction ritual is two things to keep in step, and these tables and that graph describing
    different instances of the model is precisely the drift both exist to prevent.
    """
    from agents.model.delivery_graph import build_extractor as _build
    fe, cfg, _layout = _build(config_path)
    return fe, cfg


# ------------------------------------------------------------------------------------ §2 modules
def _wrap_interpunct(names: List[str], width: int = 95) -> str:
    lines: List[str] = []
    cur = ""
    for name in names:
        piece = name if not cur else cur + " · " + name
        if len(piece) > width and cur:
            lines.append(cur + " ·")
            cur = name
        else:
            cur = piece
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def modules_section(fe: Any) -> str:
    names = [n for n, _ in fe.named_children()]
    absent = [label for label, attr in _ABSENT_CANDIDATES if _resolve(fe, attr) is None]
    out = "```\n" + _wrap_interpunct(names) + "\n```\n"
    if absent:
        out += ("\nNotably **absent** (`None` on the instance): "
                + ", ".join(f"`{a}`" for a in absent) + ".")
    return out


# -------------------------------------------------------------------------------- §3 head inputs
def head_input_parts(fe: Any) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """The concat parts of each head, IN ORDER, derived from the live instance.

    The order mirrors `ProjectionAssembler.forward` (pi: pools, active, non_matchup_rest, then the
    optional hidden-opp belief; vf: value pool, non_matchup_rest, optional active readout, optional
    hidden-opp belief, optional seed readout) plus `forward_internal`'s post-assembler
    `intent_value_reduce` concat (vf only). Totals are asserted against the real `in_features`
    by `_assert_totals`, so a drift in either the order or the parts fails loudly.
    """
    from agents.model import features_extractor as fx

    D = fx.D_MODEL
    nmr = fe.team_transformer._non_matchup_rest_dim

    pi: List[Tuple[str, int, str]] = [
        ("`our_team_pooled`", D, "`CLSPool.our_cls` over our 6 refined tokens"),
        ("`their_team_pooled`", D, "`CLSPool.their_cls` over their 6"),
        ("`our_active_refined`", D, "our active slot's refined token"),
        ("`non_matchup_rest`", nmr, "global env + board scalars (`_non_matchup_rest_dim`)"),
    ]
    vf: List[Tuple[str, int, str]] = [
        ("`value_pooled`", D, "`CLSPool.value_cls` over **all 12** team tokens"),
        ("`non_matchup_rest`", nmr, "shared with pi"),
    ]
    if fe.hidden_opp_belief is not None:
        row = ("`hidden_opp_belief`", fe.opp_belief_cls_k * D,
               f"`HiddenOppBeliefPool` — k={fe.opp_belief_cls_k} × `D_MODEL`")
        pi.append(row)
        vf.append(row)
    sr = fe.assembler.seed_readout
    if sr is not None and fe.damage_op is not None:
        vf.append(("seed readout", sr.k * sr.dim,
                   f"`MultiSeedValueReadout` — k={sr.k} × {sr.dim} over `OpTensors.incoming_rows`"))
    # gen3_value_pooled_routes_v1 (v89): the value routes are ADDITIVE injections into
    # `value_pooled` — width-neutral for the projections, so they do NOT appear as vf concat
    # parts here; the flag table + delivery graph carry them.
    return pi, vf


def _assert_totals(fe: Any, pi: List[Tuple[str, int, str]], vf: List[Tuple[str, int, str]]) -> None:
    pi_total = sum(d for _, d, _ in pi)
    vf_total = sum(d for _, d, _ in vf)
    if pi_total != fe.projection.in_features:
        raise AssertionError(
            f"pi parts sum to {pi_total} but projection.in_features is "
            f"{fe.projection.in_features} — head_input_parts no longer matches "
            f"ProjectionAssembler.forward. Parts: {pi}")
    if vf_total != fe.value_projection.in_features:
        raise AssertionError(
            f"vf parts sum to {vf_total} but value_projection.in_features is "
            f"{fe.value_projection.in_features} — head_input_parts no longer matches "
            f"ProjectionAssembler.forward / forward_internal. Parts: {vf}")


def _head_table(title: str, linear: Any, parts: List[Tuple[str, int, str]], total_attr: str) -> str:
    lines = [f"**`{title}` — `Linear({linear.in_features}, {linear.out_features})`** "
             "(LayerNorm → Linear → ReLU). Input concat, in order:", "",
             "| Part | Dims | Source |", "|---|---|---|"]
    for name, dims, source in parts:
        lines.append(f"| {name} | {dims} | {source} |")
    lines.append(f"| **total** | **{linear.in_features}** | == `{total_attr}.in_features`, "
                 "asserted at generation |")
    return "\n".join(lines)


def head_inputs_section(fe: Any) -> str:
    pi, vf = head_input_parts(fe)
    _assert_totals(fe, pi, vf)
    return (_head_table("pi_projection", fe.projection, pi, "projection")
            + "\n\n"
            + _head_table("vf_projection", fe.value_projection, vf, "value_projection"))


# --------------------------------------------------------------------------------- §6 flag table
def _toggle_status(fe: Any, key: str, value: Any) -> str:
    if _is_off(value):
        return "OFF"
    attr = _TOGGLE_MODULE.get(key)
    if attr is None:
        return "ACTIVE"
    if _resolve(fe, attr) is None:
        return f"INERT — `{attr}` is None"
    return "ACTIVE"


def _coef_status(fe: Any, key: str, value: Any) -> str:
    if key not in _COEF_MODULE:
        raise KeyError(
            f"{key!r} looks like a training-loss coefficient but has no entry in "
            "_COEF_MODULE — add its consuming module (or None for a core term) so the "
            "INERT derivation stays honest.")
    attr = _COEF_MODULE[key]
    on = not _is_off(value)
    if attr is None:                                     # core train-loop term
        return "ACTIVE" if on else "OFF"
    built = _resolve(fe, attr) is not None
    if on and built:
        return "ACTIVE"
    if on and not built:
        return f"INERT — no `{attr}`"
    if built:
        return f"INERT — coef 0, `{attr}` built"
    return "OFF"


def _fmt_value(value: Any) -> str:
    return f"`{json.dumps(value)}`"


def flag_table_section(fe: Any, cfg: Dict[str, Any]) -> str:
    from agents.model.features_extractor import Gen3FeaturesExtractor

    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    coef_keys = sorted(k for k in cfg if k.endswith("_coef") or k == "value_tail_weight")
    toggle_keys = sorted(k for k in cfg if k in sig and k not in coef_keys)

    lines = ["| Flag | Production value | Status |", "|---|---|---|"]
    for key in toggle_keys:
        lines.append(f"| `{key}` | {_fmt_value(cfg[key])} | {_toggle_status(fe, key, cfg[key])} |")
    for key in coef_keys:
        lines.append(f"| `{key}` | {_fmt_value(cfg[key])} | {_coef_status(fe, key, cfg[key])} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ marker machinery
def _begin(name: str) -> str:
    return f"<!-- BEGIN GENERATED: {name} -->"


def _end(name: str) -> str:
    return f"<!-- END GENERATED: {name} -->"


def extract_section(text: str, name: str) -> str:
    b, e = _begin(name), _end(name)
    if text.count(b) != 1 or text.count(e) != 1:
        raise AssertionError(
            f"marker pair for {name!r} must appear exactly once in {_ARCH_MD} "
            f"(BEGIN×{text.count(b)}, END×{text.count(e)})")
    return text.split(b, 1)[1].split(e, 1)[0].strip("\n")


def fill_section(text: str, name: str, body: str) -> str:
    b, e = _begin(name), _end(name)
    extract_section(text, name)                          # validates the pair exists exactly once
    pre = text.split(b, 1)[0]
    post = text.split(e, 1)[1]
    return pre + b + "\n" + body.rstrip("\n") + "\n" + e + post


def generate_sections(config_path: str = _DEFAULT_CONFIG) -> Dict[str, str]:
    fe, cfg = build_extractor(config_path)
    return {
        "modules": modules_section(fe),
        "head-inputs": head_inputs_section(fe),
        "flag-table": flag_table_section(fe, cfg),
    }


def render(text: str, sections: Dict[str, str]) -> str:
    for name in SECTIONS:
        text = fill_section(text, name, sections[name])
    return text


# --------------------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="model config to build from (default: designs/production_config.json)")
    p.add_argument("--check", action="store_true",
                   help="exit 1 with a diff summary if the committed marker sections no longer "
                        "match what HEAD generates (staleness gate)")
    a = p.parse_args(argv)

    with open(_ARCH_MD) as fh:
        current = fh.read()
    sections = generate_sections(a.config)

    if a.check:
        stale = 0
        for name in SECTIONS:
            have = extract_section(current, name)
            want = sections[name].strip("\n")
            if have != want:
                stale += 1
                print(f"STALE section {name!r} in {os.path.relpath(_ARCH_MD)}:")
                for line in difflib.unified_diff(have.splitlines(), want.splitlines(),
                                                 "committed", "generated", lineterm="", n=1):
                    print(f"  {line}")
        if stale:
            print(f"\n{stale} generated section(s) drifted — regenerate with:")
            print("  python -m agents.model.arch_tables")
            return 1
        print(f"OK {os.path.relpath(_ARCH_MD)} generated sections match HEAD")
        return 0

    new = render(current, sections)
    if new != current:
        with open(_ARCH_MD, "w") as fh:
            fh.write(new)
        print(f"rewrote generated sections in {os.path.relpath(_ARCH_MD)}")
    else:
        print(f"{os.path.relpath(_ARCH_MD)} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
