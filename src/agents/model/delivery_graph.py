"""GENERATED delivery digraph — who delivers what, to whom, through which physical channel.

WHY THIS IS GENERATED. `designs/ai_v3/README.md` is the in-repo proof that hand-drawn architecture
diagrams rot: it is a frozen historical artifact carrying a 1309-dim obs and attention paths that
no longer exist, and it went stale because nothing forced it to change when the code did. This
module builds the same picture by INSTANTIATING the extractor from a `model_config.json` and
reading the live modules, so a diagram that disagrees with the code is a test failure
(`delivery_graph_test.py`) rather than a document someone eventually notices is wrong.

THE POINT IS THE EDGE TYPING, NOT THE MODULE CALL ORDER. A call graph would say "the op runs, then
the transformer runs". That is not the question the architecture actually poses. The question is:
when a computed physics fact reaches a decision, WHAT DOES IT PHYSICALLY CARRY? Attention weights
are softmax-normalised, so an edge bias can only move *who attends to whom* — the number it writes
into the residual stream is a RATIO within its row, never an absolute ("53% of max HP"). Only token
content and per-action cells can carry an absolute. Two families delivering "the same fact" through
different channels are therefore not substitutes, and a graph that does not distinguish them is
lying about the architecture. The reasoning is in
`designs/learning/shortcut_learning_and_feature_delivery.md`.

    EDGE TYPE   CARRIES                                        CHANNEL
    ---------   --------------------------------------------   --------------------------------
    bias        a RATIO (softmax-normalised attention logit)   EdgeBias -> attention bias tensor
    content     an ABSOLUTE, as token content                  seat construction / residual inject
    concat      an ABSOLUTE, at the head input                 ProjectionAssembler concat
    cell        an ABSOLUTE, PER-ACTION                        DamageOperator.pointer_cells
    aux         nothing — training-only supervision            stashed logits -> PPO aux losses

`aux` edges must NEVER terminate at `pi_projection`, `vf_projection`, or any pointer logit. That is
leak-safety, and `delivery_graph_test.py` asserts it — turning a property that was previously an
argument in prose into a test.

Usage:
    python -m agents.model.delivery_graph                        # print a summary
    python -m agents.model.delivery_graph --dot designs/architecture_graph.dot
    python -m agents.model.delivery_graph --json <path>
    python -m agents.model.delivery_graph --config <a run's model_config.json>

The default config is `designs/production_config.json`, a verbatim copy of the live gen-3 run's
`model_config.json`. It is committed so the graph (and `designs/ARCHITECTURE.md`, derived the same
way) is reproducible without reaching outside the repo into `models/`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional

# Edge-type vocabulary. Extending this is a deliberate act: each entry is a claim about a distinct
# PHYSICAL delivery channel, not a convenience label.
EDGE_TYPES = ("bias", "content", "concat", "cell", "aux")

# Sinks an `aux` edge may never reach (the leak-safety invariant).
FORWARD_SINKS = ("pi_projection", "vf_projection")

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "designs", "production_config.json",
)


# ----------------------------------------------------------------------------- graph construction
def _node(nid: str, kind: str, **attrs) -> Dict[str, Any]:
    return {"id": nid, "kind": kind, **attrs}


def _edge(src: str, dst: str, etype: str, width: Optional[int], const: str,
          **attrs) -> Dict[str, Any]:
    assert etype in EDGE_TYPES, f"unknown edge type {etype!r}"
    return {"src": src, "dst": dst, "type": etype, "width": width,
            "source_constant": const, **attrs}


def build_graph(config_path: str = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Instantiate the extractor from `config_path` and read the delivery graph off the live modules.

    Every width and every seat index below is READ, not assumed: the op's sub-block widths come from
    its own constants, the seat indices from `TeamTransformer._total_tokens` + `EntityMoveSeats`,
    the projection widths from the built `nn.Linear`s. A structural change therefore moves the graph.
    """
    import gymnasium as gym
    import numpy as np

    from agents.model import damage_op as dop
    from agents.model import features_extractor as fx
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.model.damage_tables import sanitize_historical_move_floor

    cfg = json.load(open(config_path))
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in sig}
    # v65: pre-v65 configs recorded move_candidate_floor 0.0; keep the committed copy verbatim and
    # reconcile at construction (see the helper's docstring for why not in _migrate_config).
    sanitize_historical_move_floor(kwargs)
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs).eval()
    if hasattr(fe, "disable_observation_debugger"):
        fe.disable_observation_debugger()

    op = fe.damage_op
    seats = fe.entity_seats
    T = fx.TEAM_SIZE
    D = fx.D_MODEL
    MOVE_NET_HIDDEN = fx.MOVE_NET_HIDDEN
    N_HP_TYPES = 16                                  # the typed-Hidden-Power axis

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # --- SEATS -------------------------------------------------------------------------------
    # Absolute seat indices are the load-bearing fact: everything after the global token is
    # `base + offset`, which is what makes the base slices position-stable.
    base = fe.team_transformer._total_tokens          # 2*TEAM_SIZE + N_HISTORY_TURNS + 1
    for i in range(T):
        nodes.append(_node(f"our_mon[{i}]", "seat", index=i, token_type="OUR_TEAM"))
    for j in range(T):
        nodes.append(_node(f"opp_mon[{j}]", "seat", index=T + j, token_type="THEIR_TEAM"))
    for h in range(fx.N_HISTORY_TURNS):
        nodes.append(_node(f"history[{h}]", "seat", index=2 * T + h, token_type="HISTORY"))
    g_idx = 2 * T + fx.N_HISTORY_TURNS
    nodes.append(_node("global", "seat", index=g_idx, token_type="GLOBAL"))

    n_e3 = 4
    for k in range(n_e3):
        nodes.append(_node(f"E3_move[{k}]", "seat", index=base + k, token_type="OUR_MOVE"))
    k_e4 = seats.topk_seats
    for c in range(k_e4):
        nodes.append(_node(f"E4_threat[{c}]", "seat", index=base + n_e3 + c,
                           token_type="THEIR_THREAT"))
    n_e5 = T if seats.tail_seats else 0
    for t in range(n_e5):
        nodes.append(_node(f"E5_tail[{t}]", "seat", index=base + n_e3 + k_e4 + t,
                           token_type="THEIR_THREAT+tail_marker"))

    # --- COMPUTE / SINK NODES ----------------------------------------------------------------
    nodes.append(_node("damage_op", "operator", out_dim=op.out_dim,
                       incoming_dim=op.incoming_dim))
    if fe.move_belief is not None:
        nodes.append(_node("move_belief", "belief_head",
                           out_dim=fe.move_belief.move_head.out_features))
    if fe.hp_type_belief_head is not None:
        nodes.append(_node("hp_type_belief", "belief_head",
                           out_dim=fe.hp_type_belief_head.type_head.out_features))
    if fe.spread_belief is not None:
        nodes.append(_node("spread_belief", "belief_head"))
    if fe.belief_head is not None:
        nodes.append(_node("species_belief", "belief_head"))

    nodes.append(_node("pi_projection", "head",
                       in_features=fe.projection.in_features,
                       out_features=fe.projection.out_features))
    nodes.append(_node("vf_projection", "head",
                       in_features=fe.value_projection.in_features,
                       out_features=fe.value_projection.out_features))
    for k in range(n_e3):
        nodes.append(_node(f"pointer.move_logit[{k}]", "logit", action_index=T + k))
    for j in range(T):
        nodes.append(_node(f"pointer.switch_logit[{j}]", "logit", action_index=j))
    nodes.append(_node("pointer.struggle_logit", "logit", action_index=T + n_e3))

    # --- BIAS EDGES: the edge families ---------------------------------------------------------
    # Row/col groups come from EdgeBias.forward's placement; the cell width from _EDGE_FAMILIES.
    # Every family writes its block AND the transpose (a separate head-set per direction).
    _ROUTES = {
        "d1": ("E3_move", "opp_mon"),
        "s1": ("E3_move", "opp_mon"),
        "c1": ("E3_move", "opp_mon"),
        "c2": ("E3_move", "opp_mon"),
        "c3": ("E3_move", "opp_mon"),
        "c5": ("E3_move", "our_mon"),
        "c4": ("E3_move", "global"),
        "d3": ("E4_threat", "our_mon"),
        "s3": ("E4_threat", "our_mon"),
        "d2": ("our_mon", "opp_mon"),      # opp ACTIVE column only (one-hot delivery)
        "d4": ("our_mon", "opp_mon"),      # opp active column pre-zeroed by the kernel
        "v": ("our_mon", "opp_mon"),
        "t": ("our_mon", "opp_mon"),
        "x": ("mon", "global"),            # BOTH sides
        "g": ("mon", "global"),            # BOTH sides
        "h": ("our_mon", "opp_mon"),       # H-A2 pair-history tendencies (obs-fed)
        "r": ("event", "mon"),             # H-C reference edges (event seats × all 12 mons)
    }
    _NOTES = {
        "d2": "opp ACTIVE column only (batch-varying, one-hot outer product)",
        "d4": "opp ACTIVE column pre-zeroed — that quadrant is d3's",
        "x": "written for our mons AND opp mons",
        "g": "written for our mons AND opp mons",
        "h": "obs-fed compiled pair-history (gen3_pair_history_v1) — the GPU cannot recompute it",
        "r": "structural [is_actor, is_target] identity to the live tokens (gen3_event_ref_edges_v1)",
    }
    # COVERAGE CHECK over the model's whole table, not just the families this config ENABLES. An
    # unrouted family already raised a bare `KeyError: '<fam>'` from the loop below — but only once
    # some config turned it on, so v79's `h` sat unrouted (and silently absent from the graph) from
    # the day it was built until gen-12 enabled it. Checking the full table means the next family
    # fails the moment it is DEFINED, with a message that says what to add. Same class as the CLI
    # validator's hand-copied family set (a69708e).
    _unrouted = set(fx._EDGE_FAMILIES) - set(_ROUTES)
    if _unrouted:
        raise KeyError(
            f"delivery_graph._ROUTES is missing edge families {sorted(_unrouted)} — every family "
            f"in features_extractor._EDGE_FAMILIES needs a (row, col) route here, or the graph "
            f"omits it until some config happens to enable it.")

    def _members(group: str) -> List[str]:
        if group == "our_mon":
            return [f"our_mon[{i}]" for i in range(T)]
        if group == "opp_mon":
            return [f"opp_mon[{j}]" for j in range(T)]
        if group == "mon":
            return [f"our_mon[{i}]" for i in range(T)] + [f"opp_mon[{j}]" for j in range(T)]
        if group == "E3_move":
            return [f"E3_move[{k}]" for k in range(n_e3)]
        if group == "E4_threat":
            return [f"E4_threat[{c}]" for c in range(k_e4)]
        if group == "global":
            return ["global"]
        raise KeyError(group)

    fams = sorted(fe.edge_bias.families) if fe.edge_bias is not None else []
    for fam in fams:
        cell = fx._EDGE_FAMILIES[fam]
        row_g, col_g = _ROUTES[fam]
        for src in _members(row_g):
            for dst in _members(col_g):
                edges.append(_edge(src, dst, "bias", cell, f"_EDGE_{fam.upper()}_CELL",
                                   family=fam, bidirectional=True,
                                   note=_NOTES.get(fam, "")))

    # --- CONTENT EDGES: what becomes / enters a token ------------------------------------------
    # Seat construction. These projections are ordinary trainable Linears (new information), not
    # the zero-init residual convention.
    for k in range(n_e3):
        edges.append(_edge("pokemon_encoder.move_tokens", f"E3_move[{k}]", "content",
                           fx.MOVE_NET_HIDDEN[1], "MOVE_NET_HIDDEN[1]",
                           via="EntityMoveSeats.move_seat_proj",
                           note="request-slot order — seat k IS action logit 6+k"))
    for c in range(k_e4):
        edges.append(_edge("damage_op", f"E4_threat[{c}]", "content",
                           fx.MOVE_LATENT_DIM + 3, "MOVE_LATENT_DIM + 3",
                           via="EntityMoveSeats.threat_seat_proj",
                           note="[latent, belief w, accuracy, is_phys]; idx detached, w differentiable"))
    for t in range(n_e5):
        edges.append(_edge("damage_op", f"E5_tail[{t}]", "content", 4, "_EDGE_TAIL_CELL(4)",
                           via="EntityMoveSeats.tail_proj",
                           note="[p_tail, worst_phys, worst_spec, revealed] — beyond-top-K mass"))

    # Residual injections onto existing seats.
    if fe.prefuse_proj is not None:
        for i in range(T):
            edges.append(_edge("damage_op", f"our_mon[{i}]", "content", dop._DMG_PER_MON,
                               "_DMG_PER_MON", via="prefuse_proj", zero_init=True,
                               note="per-our-mon incoming row; the ONE absolute-magnitude route "
                                    "into the trunk"))
    if fe.move_belief is not None:
        for j in range(T):
            edges.append(_edge("move_belief", f"opp_mon[{j}]", "content", D,
                               "D_MODEL", via="MoveBelief.reinject",
                               note="soft-embedded believed moveset"))
    if fe.hp_type_belief_head is not None:
        for j in range(T):
            edges.append(_edge("hp_type_belief", f"opp_mon[{j}]", "content", D,
                               "D_MODEL", via="HPTypeBelief.reinject_proj",
                               note="presence-gated expected typed-HP embedding"))
    if fe.spread_belief is not None:
        for j in range(T):
            edges.append(_edge("spread_belief", f"opp_mon[{j}]", "content", D,
                               "D_MODEL", via="SpreadBelief.reinject"))

    # The belief the op consumes (differentiable in w — this is the gradient path that trains the
    # move belief in a config where its BCE coefficient is zero).
    if fe.move_belief is not None:
        edges.append(_edge("move_belief", "damage_op", "content",
                           fe.move_belief.move_head.out_features, "layout['max_moves']",
                           note="the composed typed posterior; differentiable in w"))

    # Entity token -> its own logit (the pointer head's equivariant read).
    for k in range(n_e3):
        edges.append(_edge(f"E3_move[{k}]", f"pointer.move_logit[{k}]", "content",
                           fe.pointer_move_token_dim, "pointer_move_token_dim",
                           via="PointerNativeActionHead.move_proj",
                           note="the REFINED seat, post-attention"))
    for j in range(T):
        edges.append(_edge(f"our_mon[{j}]", f"pointer.switch_logit[{j}]", "content", D,
                           "D_MODEL", via="PointerNativeActionHead.switch_proj",
                           note="the post-transformer team token"))
    for dst in ([f"pointer.move_logit[{k}]" for k in range(n_e3)]
                + [f"pointer.switch_logit[{j}]" for j in range(T)]
                + ["pointer.struggle_logit"]):
        edges.append(_edge("pi_projection", dst, "content", fx.NET_ARCH[-1], "NET_ARCH[-1]",
                           via="PointerNativeActionHead.ctx_proj (latent_pi, via mlp_extractor)",
                           note="the shared decision context — everything in pi_projection "
                                "conditions every pointer score"))

    # --- CONCAT EDGES: the head inputs ---------------------------------------------------------
    # `pooled` distinguishes the attention-collapsed routes from the unpooled absolute ones.
    nmr = fe.team_transformer._non_matchup_rest_dim
    for i in range(T):
        edges.append(_edge(f"our_mon[{i}]", "pi_projection", "concat", D, "D_MODEL",
                           via="CLSPool.our_cls", pooled=True))
        edges.append(_edge(f"our_mon[{i}]", "vf_projection", "concat", D, "D_MODEL",
                           via="CLSPool.value_cls", pooled=True))
    for j in range(T):
        edges.append(_edge(f"opp_mon[{j}]", "pi_projection", "concat", D, "D_MODEL",
                           via="CLSPool.their_cls", pooled=True))
        edges.append(_edge(f"opp_mon[{j}]", "vf_projection", "concat", D, "D_MODEL",
                           via="CLSPool.value_cls", pooled=True))
    edges.append(_edge("our_active_refined", "pi_projection", "concat", D, "D_MODEL",
                       pooled=False,
                       note="our active's refined token; vf does NOT read it unless "
                            "value_active_readout"))
    if fe.assembler.value_active_readout:
        edges.append(_edge("our_active_refined", "vf_projection", "concat", D, "D_MODEL",
                           pooled=False))
    for head in FORWARD_SINKS:
        # gen3_ctx_dedup_v1: the active-context concat is DELETED from both heads — the ctx
        # rides the active tokens (E2 injection) + the global token. The op's flat block is
        # likewise NOT a head input since gen3_no_concat_v1; its per-head routes are below.
        edges.append(_edge("non_matchup_rest", head, "concat", nmr,
                           "GLOBAL_ENV_DIM + board scalars", pooled=False,
                           note="the one head input with NO token route a pool reads — its "
                                "only other delivery is the global token"))
    if fe.assembler.seed_readout is not None:
        edges.append(_edge("damage_op", "vf_projection", "concat",
                           fx.VALUE_SEED_K * fx.VALUE_SEED_DIM, "VALUE_SEED_K * VALUE_SEED_DIM",
                           via="MultiSeedValueReadout (k seed queries over the per-our-mon "
                               "incoming rows, OpTensors.incoming_rows)", pooled=True,
                           note="the critic's magnitude window after the concat's death"))
    if fe.intent_value_reduce is not None:
        edges.append(_edge("damage_op", "vf_projection", "concat",
                           fx.INTENT_VALUE_REDUCE_DIM, "INTENT_VALUE_REDUCE_DIM",
                           via="IntentValueReduce (alpha-weighted pair-cell rows)", pooled=True,
                           zero_init=True))
    if fe.hidden_opp_belief is not None:
        for head in FORWARD_SINKS:
            edges.append(_edge("hidden_opp_belief", head, "concat",
                               fe.opp_belief_cls_k * fx.D_MODEL, "opp_belief_cls_k * D_MODEL",
                               via="HiddenOppBeliefPool", pooled=True))
    if getattr(fe, "value_entity_pool", None) is not None:
        # gen3_unified_value_readout_v1 (v80, Stage-3 T3-DELIVER): the critic's ONE entity pool
        # — the designed successor of the seed/threat bolt-on routes above (OFF in production
        # until the critic-route audit adjudicates them). Sources = every team token (+ the op's
        # per-our-mon rows when the op exists), so it draws with the per-mon edge shape.
        _uvr_via = ("UnifiedValueReadout (UVR_K queries, per-source type embeddings, "
                    "zero-init out projection)")
        for i in range(T):
            edges.append(_edge(f"our_mon[{i}]", "vf_projection", "concat",
                               fx.UVR_OUT_DIM, "UVR_OUT_DIM", via=_uvr_via,
                               pooled=True, zero_init=True))
            edges.append(_edge(f"opp_mon[{i}]", "vf_projection", "concat",
                               fx.UVR_OUT_DIM, "UVR_OUT_DIM", via=_uvr_via,
                               pooled=True, zero_init=True))
        if fe.damage_op is not None:
            edges.append(_edge("damage_op", "vf_projection", "concat",
                               fx.UVR_OUT_DIM, "UVR_OUT_DIM",
                               via=_uvr_via + " — the per-our-mon incoming-row source",
                               pooled=True, zero_init=True))

    # --- CELL EDGES: per-action physics --------------------------------------------------------
    if op.pointer_move_cell_dim:
        for k in range(n_e3):
            edges.append(_edge("damage_op", f"pointer.move_logit[{k}]", "cell",
                               op.pointer_move_cell_dim, "_PTR_MOVE_CELL",
                               note="[low, high, crit, pko, p_land, known, sec x"
                                    f"{dop._N_OUT_SECONDARY}] for THIS request slot"))
    if getattr(fe, "intent_move_cell", None) is not None:
        # gen3_intent_move_cell_v1 (G3): the alpha-conditioned c2 re-delivery — a per-action
        # ABSOLUTE on the move logits, weighted by the T2 alpha publication (OFF in production
        # until the G3 gate run).
        for k in range(n_e3):
            edges.append(_edge("damage_op", f"pointer.move_logit[{k}]", "cell",
                               fx.INTENT_MOVE_CELL_DIM, "INTENT_MOVE_CELL_DIM",
                               via="IntentMoveCell (published alpha × c2 operands)",
                               zero_init=True,
                               note="alpha-expected burn/sleep consequence + raw c2 columns "
                                    "+ alpha_stay, for THIS request slot"))
    if op.pointer_switch_cell_dim:
        for j in range(T):
            edges.append(_edge("damage_op", f"pointer.switch_logit[{j}]", "cell",
                               op.pointer_switch_cell_dim,
                               "_PTR_SWITCH_CELL_IN" + (" + _PTR_SWITCH_CELL_OAX"
                                                        if op.matrices_outgoing_all else ""),
                               note="incoming row + CB tail"
                                    + (" + OAX attacker row" if op.matrices_outgoing_all
                                       else " (NO offense read — matrices_outgoing_all is off)")))

    # --- AUX EDGES: training-only supervision --------------------------------------------------
    # These terminate at loss sinks and NOWHERE else. delivery_graph_test asserts it.
    aux_specs = []
    if fe.move_belief is not None and float(cfg.get("move_belief_coef", 0.0)) > 0:
        aux_specs.append(("move_belief", "loss.move_belief_bce", "known_moves"))
    if fe.move_belief is not None and float(cfg.get("move_belief_latent_coef", 0.0)) > 0:
        aux_specs.append(("move_belief", "loss.move_latent_grading", "known_moves"))
    if fe.hp_type_belief_head is not None and float(cfg.get("hp_type_belief_coef", 0.0)) > 0:
        aux_specs.append(("hp_type_belief", "loss.hp_type_ce", "hp_type_label"))
    if fe.belief_head is not None and float(cfg.get("opp_belief_aux_coef", 0.0)) > 0:
        aux_specs.append(("species_belief", "loss.belief_aux", "belief_species/belief_moves"))
    if fe.spread_belief is not None and float(cfg.get("spread_belief_coef", 0.0)) > 0:
        aux_specs.append(("spread_belief", "loss.spread_belief", "belief_spread"))
    for src, sink, label in aux_specs:
        nodes.append(_node(sink, "aux_loss", label_key=label))
        edges.append(_edge(src, sink, "aux", None, "training-only obs Dict key",
                           label_key=label,
                           note="privileged label; never enters the forward"))

    # ---------------------------------------------------------------- T0: the front of the pipeline
    # Until now this graph began at the ROLE TOKENS and everything upstream collapsed into a handful
    # of auto-created `input` pseudo-nodes — 4 nodes for the entire obs -> encoder -> belief stack,
    # while the seat/edge half carried 50+. That is backwards relative to where the pipeline's
    # complexity actually lives, and it made the T0 tier (which the tier contract now enforces)
    # invisible in the one picture people read.
    #
    # Everything here is DERIVED, not asserted: the blocks come from the schema's validated tiling
    # (`build_schema(layout).slices()`, whose construction already proves the slices tile the vector
    # exactly), and each leg appears only when the live config actually builds it. So the graph
    # cannot drift from the obs layout without the snapshot diff showing it.
    from agents.observation.schema import build_schema
    _slices = build_schema(layout).slices()
    _tops = [(k, v) for k, v in _slices.items() if "." not in k]
    for name, sl in _tops:
        nodes.append(_node(f"obs.{name}", "obs_block", offset=sl.start,
                           width=sl.stop - sl.start, stage="T0"))
        edges.append(_edge(f"obs.{name}", "obs_unpack", "content", sl.stop - sl.start,
                           f"schema.slices()['{name}']",
                           note="validated tiling — the slice map is proven exact at construction"))
    # Sub-slices of a block, so the global/reactive groups are legible rather than one opaque width.
    for name, sl in _slices.items():
        if "." not in name:
            continue
        parent, _, leaf = name.partition(".")
        nodes.append(_node(f"obs.{name}", "obs_group", offset=sl.start,
                           width=sl.stop - sl.start, parent=f"obs.{parent}", stage="T0", leaf=leaf))

    # Phase nodes are DERIVED from the live module tree, not listed. An earlier cut hardcoded these
    # names and their tier, which would have kept drawing a structure that no longer existed the
    # moment anyone renamed a submodule — precisely the rot a generated artifact exists to prevent.
    # Now: the attribute must be a real child of the live `PokemonEncoder`, and the tier is read
    # from `tier_contract.TIER_OF` rather than asserted, so the graph and the enforced tier ordering
    # cannot disagree.
    from agents.model.tier_contract import TIER_OF, UNTIERED_CHILDREN

    def _tier(owner: str) -> str:
        """The stage label, read from the enforced contract — never asserted here.

        `pre-T0` is not a fudge: `unpack` is DELIBERATELY in `UNTIERED_CHILDREN` because it
        "produces the tier-0 INPUT, ahead of every tier". Distinguishing that from an unknown
        module matters — an unrecognised owner should look wrong, and a deliberately untiered one
        should not.
        """
        t = TIER_OF.get(owner)
        if t is not None:
            return f"T{t}"
        return "pre-T0" if owner in UNTIERED_CHILDREN else "UNDECLARED"

    nodes.append(_node("obs_unpack", "phase", stage=_tier("unpack"),
                       note="stateless; peels the flat vector into the ExtractorContext tensors"))
    _pe = fe.pokemon_encoder
    for attr, note in (
        ("move_network", "shared MLP over EVERY move slot (MOVE_NET_HIDDEN)"),
        ("move_self_attn", "within-Pokemon move attention (MHA + LayerNorm residual)"),
        ("role_encoder", "ROLE_ENCODER_HIDDEN MLP -> the 12 role tokens"),
    ):
        if not hasattr(_pe, attr):
            raise AssertionError(
                f"PokemonEncoder has no `{attr}` — the delivery graph names a submodule that no "
                "longer exists. Update the phase list rather than letting the picture describe a "
                "structure the code does not have.")
        nodes.append(_node(f"pokemon_encoder.{attr}", "phase",
                           stage=_tier("pokemon_encoder"), note=note))
    edges.append(_edge("obs_unpack", "pokemon_encoder.move_network", "content",
                       layout["move_slot_dim"] if "move_slot_dim" in layout else MOVE_NET_HIDDEN[0],
                       "per-move slot width", note="per-move slots"))
    edges.append(_edge("pokemon_encoder.move_network", "pokemon_encoder.move_self_attn",
                       "content", MOVE_NET_HIDDEN[-1], "MOVE_NET_HIDDEN[-1]"))
    edges.append(_edge("pokemon_encoder.move_self_attn", "pokemon_encoder.role_encoder",
                       "content", MOVE_NET_HIDDEN[-1], "MOVE_NET_HIDDEN[-1]"))
    edges.append(_edge("obs_unpack", "pokemon_encoder.role_encoder", "content",
                       layout["pokemon_full_dim"] if "pokemon_full_dim" in layout else D,
                       "POKEMON_FULL_DIM",
                       note="species/item/ability/type embeddings + the E2 active-ctx injection"))
    for i in range(T):
        edges.append(_edge("pokemon_encoder.role_encoder", f"our_mon[{i}]", "content",
                           D, "D_MODEL"))
        edges.append(_edge("pokemon_encoder.role_encoder", f"opp_mon[{i}]", "content",
                           D, "D_MODEL"))

    # The T0 belief legs, with the distinction that matters: a leg either REINJECTS into the tokens
    # the transformer will refine, or it is a side READOUT that never enters the forward. Collapsing
    # those two into one arrow is what made "is this belief actually used?" unanswerable by eye.
    if fe.belief_slots is not None:
        nodes.append(_node("belief_slots", "phase", stage="T0",
                           note="swaps unrevealed opp role tokens for learned unknown-mon queries"))
        for j in range(T):
            edges.append(_edge("belief_slots", f"opp_mon[{j}]", "content", D,
                               "D_MODEL", note="in-place; pre-transformer"))
    if fe.move_belief is not None:
        for j in range(T):
            edges.append(_edge("move_belief", f"opp_mon[{j}]", "content", D, "D_MODEL",
                               note="REINJECT — the believed moveset co-refines through attention"))
    if getattr(fe, "t0_species_prior", None) is not None:
        nodes.append(_node("t0_species_prior", "belief_head", stage="T0",
                           note="team-composition species belief; parameter-free"))
        edges.append(_edge("t0_species_prior", "damage_op", "content",
                           layout["max_species"], "layout['max_species']",
                           note="P(species | revealed team) — replaces the STATIC usage prior"))
    if fe.spread_belief is not None:
        edges.append(_edge("spread_belief", "damage_op", "content", 5, "5 stats",
                           note="believed opp spread; without it the op prices a fictional "
                                "maximally-invested opponent"))
    if getattr(fe, "hp_type_belief_head", None) is not None:
        edges.append(_edge("hp_type_belief", "move_belief", "content", N_HP_TYPES, "16 HP types",
                           note="composes the typed Hidden Power posterior"))

    # Input pseudo-nodes referenced above but not yet declared.
    declared = {n["id"] for n in nodes}
    for nid in sorted({e["src"] for e in edges} | {e["dst"] for e in edges}):
        if nid not in declared:
            nodes.append(_node(nid, "input"))

    meta = {
        "config": os.path.basename(config_path),
        "arch_signature": cfg.get("arch_signature"),
        "config_version": cfg.get("config_version"),
        "obs_dim": layout["total_dim"],
        "n_tokens": base + seats.n_seats,
        "base_seats": base,
        "extra_seats": seats.n_seats,
        "edge_bias_families": sorted(fams),
        "op_out_dim": op.out_dim,
        "pi_projection_in": fe.projection.in_features,
        "vf_projection_in": fe.value_projection.in_features,
        "edge_types": list(EDGE_TYPES),
    }
    nodes.sort(key=lambda n: (n["kind"], n["id"]))
    edges.sort(key=lambda e: (e["type"], e["src"], e["dst"], e.get("via") or ""))
    return {"meta": meta, "nodes": nodes, "edges": edges}


# ------------------------------------------------------------------------------------ DOT output
_STYLE = {
    "bias":    ('color="#b5453a"', "dashed", "RATIO"),
    "content": ('color="#2f6f3e"', "solid", "ABS"),
    "concat":  ('color="#31538f"', "solid", "ABS"),
    "cell":    ('color="#7d5bab"', "bold", "ABS/action"),
    "aux":     ('color="#8a8a8a"', "dotted", "train-only"),
}
_KIND_SHAPE = {
    "seat": 'shape=box, style="rounded,filled", fillcolor="#eef3fa"',
    "operator": 'shape=box3d, style=filled, fillcolor="#ffe9a8"',
    "belief_head": 'shape=box, style=filled, fillcolor="#e6dcf3"',
    "head": 'shape=box, style="filled,bold", fillcolor="#d7ead9"',
    "logit": 'shape=ellipse, style=filled, fillcolor="#fce8e6"',
    "aux_loss": 'shape=note, style=filled, fillcolor="#eeeeee"',
    "input": 'shape=plaintext',
    # T0 front-of-pipeline kinds. Distinct shapes so the obs SUBSTRATE reads differently from the
    # modules that consume it — the point of expanding this region was to make the early stage
    # legible, and drawing it in the seat palette would defeat that.
    "obs_block": 'shape=folder, style=filled, fillcolor="#dbeddb"',
    "obs_group": 'shape=folder, style="filled,dashed", fillcolor="#eef7ee"',
    "phase": 'shape=component, style=filled, fillcolor="#e8eef7"',
}


def to_dot(graph: Dict[str, Any], collapse: bool = True) -> str:
    """Graphviz DOT. `collapse` groups indexed seats (our_mon[0..5] -> our_mon x6) so the picture
    is readable; the JSON snapshot always carries the full per-index graph."""
    import re

    def norm(nid: str) -> str:
        return re.sub(r"\[\d+\]", "", nid) if collapse else nid

    m = graph["meta"]
    out = [
        "digraph gen3_delivery {",
        "  // GENERATED by src/agents/model/delivery_graph.py — do not edit by hand.",
        f"  // config={m['config']}  arch={m['arch_signature']}  v{m['config_version']}",
        "  rankdir=LR; splines=spline; fontname=\"Helvetica\"; compound=true;",
        "  node [fontname=\"Helvetica\", fontsize=10];",
        "  edge [fontname=\"Helvetica\", fontsize=8];",
        "",
        "  labelloc=\"t\";",
        f'  label=<<b>gen3 delivery graph</b><br/>{m["arch_signature"]} · config v{m["config_version"]}'
        f' · obs {m["obs_dim"]} · {m["n_tokens"]} tokens · op out_dim {m["op_out_dim"]}<br/>'
        '<font point-size="9">edge colour = what it PHYSICALLY carries: '
        '<font color="#b5453a">bias=RATIO</font> · '
        '<font color="#2f6f3e">content=ABS</font> · '
        '<font color="#31538f">concat=ABS</font> · '
        '<font color="#7d5bab">cell=ABS/action</font> · '
        '<font color="#8a8a8a">aux=never in forward</font></font>>;',
        "",
    ]

    counts: Dict[str, int] = {}
    for n in graph["nodes"]:
        counts[norm(n["id"])] = counts.get(norm(n["id"]), 0) + 1
    seen = set()
    for n in graph["nodes"]:
        nid = norm(n["id"])
        if nid in seen:
            continue
        seen.add(nid)
        c = counts[nid]
        label = nid + (f" ×{c}" if c > 1 else "")
        if n["kind"] == "head":
            label += f'\\n{n["in_features"]}→{n["out_features"]}'
        elif n["kind"] == "operator":
            label += f'\\nout_dim {n["out_dim"]}'
        out.append(f'  "{nid}" [{_KIND_SHAPE[n["kind"]]}, label="{label}"];')
    out.append("")

    # Collapse parallel edges, keeping one representative label + a multiplicity.
    agg: Dict[Any, Dict[str, Any]] = {}
    for e in graph["edges"]:
        key = (norm(e["src"]), norm(e["dst"]), e["type"], e.get("via"), e.get("family"))
        rec = agg.setdefault(key, {"n": 0, "e": e})
        rec["n"] += 1
    for (src, dst, etype, via, family), rec in sorted(agg.items(), key=lambda kv: str(kv[0])):
        e = rec["e"]
        color, style, tag = _STYLE[etype]
        bits = [family or etype]
        if e["width"] is not None:
            bits.append(f'w={e["width"]}')
        if rec["n"] > 1:
            bits.append(f'×{rec["n"]}')
        lbl = " ".join(bits)
        extra = ", dir=both" if e.get("bidirectional") else ""
        out.append(f'  "{src}" -> "{dst}" [{color}, style={style}, label="{lbl}", '
                   f'tooltip="{tag} · {e["source_constant"]}"{extra}];')
    out.append("}")
    return "\n".join(out) + "\n"


_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "delivery_graph_snapshot.json")


def _check(graph: Dict[str, Any]) -> int:
    """Compare the live build against the committed snapshot and say WHAT moved.

    `delivery_graph_test.py` already fails on drift, but a test only runs when someone runs the
    suite — and the snapshot sits UPSTREAM of the viewer, so a stale one makes every artifact
    below it confidently wrong while `build_arch_viewer --check` still passes (it compares the
    HTML against the snapshot, so both sides are stale together).

    Prints the structural diff rather than just "STALE", because the snapshot is a TRIPWIRE: the
    thing you need to see is *which* nodes and edges moved, so an unintended architecture change
    is obvious rather than merely detected.
    """
    if not os.path.exists(_SNAPSHOT_PATH):
        print(f"MISSING {_SNAPSHOT_PATH}")
        return 1
    with open(_SNAPSHOT_PATH) as fh:
        old = json.load(fh)
    want = json.dumps(graph, indent=2, sort_keys=True) + "\n"
    with open(_SNAPSHOT_PATH) as fh:
        if fh.read() == want:
            print(f"OK {os.path.relpath(_SNAPSHOT_PATH)} matches the live code")
            return 0

    print(f"STALE {os.path.relpath(_SNAPSHOT_PATH)} — the live code no longer matches it.")
    for key in sorted(set(graph["meta"]) | set(old.get("meta", {}))):
        a, b = old.get("meta", {}).get(key), graph["meta"].get(key)
        if a != b:
            print(f"  meta  {key}: {a!r} -> {b!r}")
    ekey = lambda e: (e["type"], e["src"], e["dst"], e.get("family") or "")   # noqa: E731
    for label, a_set, b_set in (
        ("node", {n["id"] for n in old.get("nodes", [])}, {n["id"] for n in graph["nodes"]}),
        ("edge", {ekey(e) for e in old.get("edges", [])}, {ekey(e) for e in graph["edges"]}),
    ):
        for gone in sorted(a_set - b_set)[:12]:
            print(f"  -{label}  {gone}")
        for new in sorted(b_set - a_set)[:12]:
            print(f"  +{label}  {new}")
        dropped = max(len(a_set - b_set) - 12, 0) + max(len(b_set - a_set) - 12, 0)
        if dropped:
            print(f"  ... and {dropped} more {label} changes")
    print("\nIf that is an INTENTIONAL architecture change, regenerate:")
    print("  python -m agents.model.delivery_graph \\")
    print("      --dot designs/architecture_graph.dot \\")
    print("      --json src/agents/model/delivery_graph_snapshot.json")
    print("  python -m agents.model.build_arch_viewer      # rebuilds FROM the fresh snapshot")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="model_config.json to build from (default: designs/production_config.json)")
    p.add_argument("--dot", help="write Graphviz DOT here")
    p.add_argument("--json", dest="json_out", help="write the JSON snapshot here")
    p.add_argument("--no-collapse", action="store_true",
                   help="DOT: keep every indexed seat as its own node")
    p.add_argument("--check", action="store_true",
                   help="exit 1 if the committed snapshot no longer matches the live code, and "
                        "print WHAT changed structurally (drift gate)")
    p.add_argument("--sync-config", metavar="RUN_MODEL_CONFIG",
                   help="Refresh designs/production_config.json FROM a run's model_config.json "
                        "(the direction of truth: run -> mirror), then build from it. The "
                        "per-generation ritual in one flag — pair with --dot/--json, then rebuild "
                        "the viewer (python -m agents.model.build_arch_viewer).")
    a = p.parse_args(argv)

    if a.sync_config:
        # The mirror refresh the compile gate's arch_signature assertion polices: the run dir's
        # model_config.json is the immutable per-run record (written at creation, never mutated);
        # designs/production_config.json is only its committed mirror for the tools (this graph,
        # the arch viewer, the compile gate). Copy verbatim — never hand-edit the mirror.
        with open(a.sync_config) as fh:
            cfg = json.load(fh)
        for required in ("arch_signature", "config_version", "total_dim"):
            if required not in cfg:
                raise SystemExit(f"--sync-config source lacks {required!r} — not a model_config.json")
        with open(_DEFAULT_CONFIG, "w") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"synced {_DEFAULT_CONFIG} <- {a.sync_config} "
              f"({cfg['arch_signature']} v{cfg['config_version']}, obs {cfg['total_dim']})")
        a.config = _DEFAULT_CONFIG

    g = build_graph(a.config)
    if a.check:
        return _check(g)
    if a.dot:
        open(a.dot, "w").write(to_dot(g, collapse=not a.no_collapse))
        print(f"wrote {a.dot}")
    if a.json_out:
        open(a.json_out, "w").write(json.dumps(g, indent=2, sort_keys=True) + "\n")
        print(f"wrote {a.json_out}")
    if not a.dot and not a.json_out:
        m = g["meta"]
        print(f"{m['arch_signature']} · config v{m['config_version']} · obs {m['obs_dim']} · "
              f"{m['n_tokens']} tokens · op out_dim {m['op_out_dim']}")
        print(f"{len(g['nodes'])} nodes, {len(g['edges'])} edges")
        by_type: Dict[str, int] = {}
        for e in g["edges"]:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        for t in EDGE_TYPES:
            print(f"  {t:8s} {by_type.get(t, 0):5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
