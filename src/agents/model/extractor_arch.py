"""ONE place that turns parsed CLI args into `Gen3FeaturesExtractor` kwargs.

WHY THIS MODULE EXISTS. `train_rl_agent.py` used to assemble this dict TWICE — once on the fresh-run
path (`extractor_kwargs`) and once on the resume path (`_load_extractor_kwargs`) — as two ~45-line
blocks of `d["flag"] = args.flag`. They were verified identical at the time this was factored out (42
keys, same set, same sources), but nothing enforced that: adding a v51 toggle to one and not the other
would mean a resume silently version-checks against a DIFFERENT arch than it builds. That is the
`gen3_resume_optimizer_realign_bug` failure shape — a mismatch that surfaces as a confusing tensor
error much later, or not at all.

There is now a third consumer that made the duplication untenable: the forkserver preload
(`agents.model.compile_preload`) must build the SAME extractor the env workers will, in a fresh
interpreter that never parsed argv, so it needs this mapping as DATA rather than as inline statements.

`ARCH_ARG_KEYS` is that data: extractor-kwarg name → the `args` attribute it reads. Anything needing a
derivation (a coef>0 enable signal, a private computed field) lives in `_DERIVED`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# extractor kwarg -> args attribute. Straight pass-throughs; keep alphabetical within a version group
# so a new toggle is obvious in review.
ARCH_ARG_KEYS: Dict[str, str] = {
    "attend_unrevealed_opponents": "attend_unrevealed_opponents",
    "opp_belief_cls_k": "opp_belief_cls_k",
    "value_active_readout": "value_active_readout",
    "move_belief_mode": "move_belief_mode",
    "damage_op": "damage_op",
    "damage_reattend": "damage_reattend",                                   # v31
    "damage_outgoing": "damage_outgoing",                                   # v23
    "move_candidate_floor": "move_candidate_floor",                         # v23
    "move_latent": "move_latent",                                           # v24
    "spread_belief": "spread_belief",                                       # v25
    "spread_belief_nature": "spread_belief_nature",                         # v40
    "spread_belief_nature_marginalize": "spread_belief_nature_marginalize",  # v40
    "move_prior_fusion": "move_prior_fusion",                               # v20
    "move_belief_prefuse": "move_belief_prefuse",                           # v32
    "move_belief_single_compute": "move_belief_single_compute",             # v47
    "damage_candidate_k": "damage_candidate_k",                             # v49
    "damage_op_prefuse": "damage_op_prefuse",                               # v50 (v51 pointer-native: no key — unconditional)
    "entity_topk_seats": "entity_topk_seats",                               # v54 (E3 unconditional; this is the E4 K)
    "edge_bias_families": "edge_bias_families",                             # v56 (layer swap unconditional; this is the family set)
    "win_prob_mode": "win_prob_mode",                                       # v22
    "pubval_mode": "pubval_mode",                                           # v43
    "value_dist_mode": "value_dist_mode",                                   # v29
    "value_dist_bins": "value_dist_bins",
    "value_dist_vmin": "value_dist_vmin",
    "value_dist_vmax": "value_dist_vmax",
    "damage_topk_k": "damage_topk_k",                                       # v30
    "damage_refine_rounds": "damage_refine_rounds",                         # v33
    "damage_matrices_outgoing": "damage_matrices_outgoing",                 # v34
    "damage_matrices_incoming": "damage_matrices_incoming",                 # v35
    "damage_matrices_outgoing_all": "damage_matrices_outgoing_all",         # v39
    "threat_refine_outgoing": "threat_refine_outgoing",                     # v36
    "threat_unrevealed_outgoing": "threat_unrevealed_outgoing",             # v36
    "threat_prob_outspeed": "threat_prob_outspeed",                         # v36
    "threat_status_refine": "threat_status_refine",                         # v37
    "hp_belief_mode": "hp_belief_mode",                                     # v53
    "belief_grad_mode": "belief_grad_mode",                                 # v41
    "zarch_film": "zarch_film",                                             # v44
    "zarch_dim": "zarch_dim",
    "zarch_lut": "zarch_lut",                                               # v46
    "zarch_lut_init_std": "zarch_lut_init_std",
}

# Kwargs that are NOT a plain attribute read. Each is a callable over `args`.
_DERIVED = {
    # coef>0 is the enable signal for these three; the COEF itself is a training hparam set on the
    # model, but the BOOL is the version-checked arch toggle.
    "opp_belief_slots": lambda a: getattr(a, "opp_belief_aux_coef", 0.0) > 0.0,
    "opp_belief_latent": lambda a: getattr(a, "opp_belief_latent_coef", 0.0) > 0.0,
    # Computed upstream by the --zarch-lut validation (a roster table, not a CLI value).
    "zarch_lut_rosters": lambda a: getattr(a, "_zarch_lut_rosters", None),
}


def build_extractor_arch_kwargs(args, base: Optional[Dict[str, Any]] = None,
                                log_level: Any = None) -> Dict[str, Any]:
    """Layout kwargs (`base`) + every version-checked architecture toggle read off `args`.

    `base` is normally `Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()` — the obs
    layout half. Pass None to get only the arch half (what the compile pre-warm wants when it builds
    its own layout). `log_level` is threaded only on the fresh-run path; it is a diagnostic, not an
    arch field, so it is omitted when None rather than being written as a null.
    """
    kwargs: Dict[str, Any] = dict(base) if base else {}
    for kwarg, attr in ARCH_ARG_KEYS.items():
        kwargs[kwarg] = getattr(args, attr)
    for kwarg, fn in _DERIVED.items():
        kwargs[kwarg] = fn(args)
    if log_level is not None:
        kwargs["log_level"] = log_level
    return kwargs


def arch_kwargs_to_plain(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-serialisable subset, for handing an arch across a process boundary.

    The forkserver preload receives its config through the environment, so anything unpicklable or
    huge (the layout's numpy mapping tables, the zarch roster table) is dropped — the preload only
    needs the toggles that change the traced GRAPH. Dropping a key means the preload builds a
    slightly different extractor and the worker's guards miss, which costs a recompile but is never
    incorrect, so this errs toward dropping.
    """
    plain: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in ("zarch_lut_rosters", "log_level"):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            plain[k] = v
    return plain
