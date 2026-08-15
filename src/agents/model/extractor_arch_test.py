"""Drift guards for `agents.model.extractor_arch` — the single arch-kwargs table.

`train_rl_agent` used to build this dict twice, inline, once on the fresh path and once on the resume
path. Nothing tied them together, so a new toggle could land on one and not the other and a resume
would version-check an arch it did not build. These tests exist to keep that from coming back.
"""
import inspect
import json
import re
from pathlib import Path

import pytest

from agents.model import extractor_arch as EA
from agents.model.features_extractor import Gen3FeaturesExtractor

_TRAIN_PY = Path(__file__).resolve().parents[2] / "main" / "train_rl_agent.py"


def _extractor_params():
    return set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)


def test_every_mapped_kwarg_is_a_real_extractor_parameter():
    """A typo or a removed toggle would otherwise be passed straight into `Gen3FeaturesExtractor`
    and blow up at model-build time (fresh runs) or, worse, only on resume."""
    params = _extractor_params()
    unknown = sorted(k for k in list(EA.ARCH_ARG_KEYS) + list(EA._DERIVED) if k not in params)
    assert not unknown, f"extractor_arch maps kwargs the extractor does not accept: {unknown}"


def test_no_inline_extractor_kwargs_assignments_remain():
    """THE anti-duplication guard. Both call sites must go through `build_extractor_arch_kwargs`; a
    stray `extractor_kwargs["new_toggle"] = args.new_toggle` re-opens the drift this replaced."""
    src = _TRAIN_PY.read_text()
    stray = re.findall(r'\b\w*extractor_kwargs\["([a-z_0-9]+)"\]\s*=\s*args\.', src)
    assert not stray, (
        f"train_rl_agent.py assigns arch kwargs inline again: {sorted(set(stray))}. "
        f"Add them to agents.model.extractor_arch.ARCH_ARG_KEYS instead, so the fresh and resume "
        f"paths cannot diverge."
    )


def test_both_paths_use_the_shared_builder():
    src = _TRAIN_PY.read_text()
    assert src.count("build_extractor_arch_kwargs(") >= 3, (
        "expected the fresh path, the resume path and the compile pre-load to share the builder"
    )


def test_builder_output_constructs_an_extractor(monkeypatch):
    """End-to-end shape check: whatever the table produces must be accepted by the extractor."""
    import types
    params = _extractor_params()
    args = types.SimpleNamespace()
    # Defaults chosen to be individually valid; the point is key/typing, not a working arch.
    for attr in EA.ARCH_ARG_KEYS.values():
        setattr(args, attr, False)
    args.move_belief_mode = "off"
    args.belief_grad_mode = "shaping"
    args.win_prob_mode = "none"
    args.pubval_mode = "none"
    args.value_dist_mode = "none"
    args.opp_intent_grad_mode = "detached"
    args.hp_belief_mode = "composed"
    args.edge_bias_families = "off"
    for attr in ("opp_belief_cls_k", "value_dist_bins", "damage_topk_k",
                 "damage_candidate_k", "consequence_topk", "entity_topk_seats"):
        setattr(args, attr, 0)
    for attr in ("move_candidate_floor", "value_dist_vmin", "value_dist_vmax"):
        setattr(args, attr, 0.0)
    args.opp_belief_aux_coef = 0.0
    args.opp_intent_coef = 0.0

    kwargs = EA.build_extractor_arch_kwargs(args)
    assert set(kwargs) <= params, sorted(set(kwargs) - params)
    assert kwargs["opp_belief_slots"] is False
    args.opp_belief_aux_coef = 0.1
    assert EA.build_extractor_arch_kwargs(args)["opp_belief_slots"] is True


def test_log_level_is_omitted_unless_given():
    import types
    args = types.SimpleNamespace(**{a: False for a in EA.ARCH_ARG_KEYS.values()})
    args.opp_belief_aux_coef = 0.0
    assert "log_level" not in EA.build_extractor_arch_kwargs(args)
    assert EA.build_extractor_arch_kwargs(args, log_level="periodic")["log_level"] == "periodic"


def test_plain_form_is_json_serialisable_and_drops_unpicklables():
    """The forkserver preload receives the arch through the environment, so it must survive JSON and
    must not try to carry the numpy layout tables."""
    import types
    args = types.SimpleNamespace(**{a: False for a in EA.ARCH_ARG_KEYS.values()})
    args.opp_belief_aux_coef = 0.0
    args.opp_intent_coef = 0.0
    kwargs = EA.build_extractor_arch_kwargs(args, base={"layout": {"np": object()}},
                                            log_level="periodic")
    plain = EA.arch_kwargs_to_plain(kwargs)
    assert "layout" not in plain
    assert "log_level" not in plain
    json.dumps(plain)                                  # must not raise
    # the toggles that actually change the traced graph DO survive
    assert "damage_op" in plain and "threat_prob_outspeed" in plain
