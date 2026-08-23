"""The CONTRACT the two prober decompositions rest on: `main.prober.engine` + `main.prober.session`.

On 2026-08-23 `engine.py` (3,058 lines) and `session.py` (2,573) each became a PACKAGE of the same
name whose `__init__.py` is a pure re-export hub — the `features_extractor.py` / `main/train/`
precedent. Three things make that safe rather than merely tidier, none of them self-enforcing:

**1. Both hubs still export everything they used to.** `from main.prober.engine import <anything>`
and `from main.prober.session import <anything>` have to keep resolving — for the web app, the
JSON CLI, `probe_replay`, `endofrun`, the teacher's selection funnel, three fuzz tests that reach
in for `_has_state`, and the ~120 call sites in `engine_test.py`. The lists below are the
module-level *definitions* of each file as it stood at the commit before the split, recovered by
AST rather than typed by hand. They may GROW; a name leaving one is a deliberate deletion that
should fail here first.

⚠️ Pure import BINDINGS are deliberately not pinned (`engine.np`, `engine.re`, `session.os`, and
the ~25 engine names `session.py` re-imported for its own use). They were reachable as module
attributes before and are not now — nothing in the tree reads them that way, and pinning them
would freeze one module's private imports as another's public surface.

**2. No submodule imports its own hub.** The hub imports every submodule; a submodule importing
the hub back closes a cycle whose symptom is a partially-initialised module at import time — an
`AttributeError` on a name that plainly exists, from a traceback pointing at the wrong file.

**3. Every submodule stands up on its own.** A module that only imports once its siblings have is
a cycle that has not bitten yet.

Milliseconds — attribute lookups and an AST walk. No marker; runs in the fast inner loop.
"""
import ast
import importlib
import pathlib

import pytest

import main.prober.engine as engine_hub
import main.prober.session as session_hub

_ENGINE_DIR = pathlib.Path(engine_hub.__file__).parent
_SESSION_DIR = pathlib.Path(session_hub.__file__).parent

# Every module-level name `src/main/prober/engine.py` DEFINED at 84b4122, the commit before the
# decomposition. Recovered by AST, transcribed verbatim.
_PRE_SPLIT_ENGINE = (
    "ActionRow", "BELIEF_FIRED_PKO", "BELIEF_NAME_CAVEAT", "BELIEF_UNDERREAD_PKO",
    "BeliefSlotView", "BeliefTrajectoryPoint", "BeliefTrajectoryView", "BeliefTruthView",
    "BeliefView", "BoardView", "CANT_PHRASE", "CRITIC_CONFIDENT_V", "ExclusiveBeliefView",
    "ExclusiveSlotView", "FAINTED_HP", "HEALTHY_HP", "IncomingBeliefView", "InterventionRow",
    "InterventionSweep", "InvocationAnalysis", "LOSS_TAXONOMY", "MatchupView", "MonState",
    "MoveBeliefView", "NO_EFFECT_TEXT", "OppFullMon", "OppFullTeamView", "OppIntentCandidate",
    "OppIntentOption", "OppIntentView", "OppMonTruth", "OppMoveBelief", "SETUP_MOVES",
    "STALL_NEAR_CAP", "SWITCH_OPTION", "Saliency", "SaliencyBlock", "SideBoard",
    "SpreadBeliefView", "SpreadSlotBelief", "SpreadStatRow", "SwitchInOutgoingRow",
    "SwitchInOutgoingView", "ThreatView", "TraceMeta", "UNCERTAIN_THRESHOLD", "ValueDistView",
    "ValueView", "WP_EVEN_DEFAULT", "WinProbView", "_BENCH_RE", "_BP0_DAMAGING",
    "_CURABLE_STATUSES", "_Cat", "_FAINT_EVENT_RE", "_L2_GRID", "_LOG_BLOCK_RE", "_MATCHUP_DIM",
    "_MAX_MOVES", "_MOVE_ID_TO_NUM", "_MOVE_NUM_TO_ID", "_MOVE_PLACEHOLDER_RE", "_SENTINEL_RE",
    "_SENT_IN", "_SEP", "_SPECIES_MAPS", "_SPREAD_BASE_KEY", "_SPREAD_COLS",
    "_SPREAD_PRIOR_CACHE", "_STATUS_EVENT_RE", "_SWEEP_MULTIPLIERS", "_SWITCH_IN_HIT_MAX_HP",
    "_TEAM_SIZE", "_active_slot", "_as_ptype", "_auc", "_beta_candidate", "_derived_hp",
    "_derived_stat", "_display_hp", "_dist_quantile", "_entropy_bits", "_f", "_faithfulness",
    "_has_state", "_hp_frac_from_str", "_intervention_sweep", "_is_attack", "_kfold_indices",
    "_logistic_fit", "_loss_pct", "_matches_move", "_matchups", "_merge_team", "_move_display",
    "_move_id_to_num", "_move_maps", "_multiplier_meaningful", "_no_effect_reason",
    "_norm_move", "_norm_species", "_npz_array", "_npz_value", "_npz_win_prob", "_oof_predict",
    "_opp_actual_action", "_our_items", "_parse_bench", "_parse_outcome_action", "_pct",
    "_retype_hp", "_ridge_fit", "_saliency", "_saliency_from_grad", "_selection_score",
    "_side_board", "_sigmoid", "_softmax", "_species_maps", "_spread_prior_means",
    "_standardize", "_switch_prob_sum", "_team_entry", "_threats", "_timeline_for",
    "_true_derived_spread", "_value_saliency", "_was_winning", "analyze_invocation",
    "attribute_turning_point", "awareness_text", "belief_view_from_logits", "build_belief",
    "build_belief_trajectory", "build_belief_truth", "build_board", "build_exclusive_belief",
    "build_meta", "build_opp_full_team", "build_opp_intent", "build_our_hp_types",
    "build_result_timeline", "build_spread_belief", "build_switch_in_outgoing",
    "build_value_dist", "cant_phrase", "decode_incoming_belief", "fit_probe",
    "has_curable_status", "history_slot_saliency", "is_status_cure", "move_belief_view",
    "move_order_from_protocol", "opp_intent_text", "opp_voluntary_switch", "opponent_rank",
    "parse_pct", "parse_protocol_log", "protocol_action_fate", "protocol_for_turn",
    "protocol_move_result", "revealed_opp_species", "self_cure_options", "sort_opponents",
    "summary_flags", "surprise_phrase", "timeline_entry_text",
)

# Same, for `src/main/prober/session.py`.
_PRE_SPLIT_SESSION = (
    "ProbeSession", "_DEFAULT_GAMMA", "_FIXED_DAMAGE_MOVES", "_MAX_CACHED_MODELS",
    "_PROBE_TARGETS", "_active_str", "_base_spe", "_belief_pko_group", "_big_hit_label",
    "_calibration_stats", "_choice_dict", "_chosen_prob", "_discounted_returns", "_dmg_label",
    "_faint_healthy_label", "_faint_label", "_faster_group", "_faster_label", "_loop_aggregate",
    "_mon_dict", "_opp_intent_dict", "_opp_move_id", "_opp_status_move_label",
    "_opp_switch_label", "_prov", "_r", "_rate_by", "_ratio", "_reliability_curve",
    "_reliability_gap_at", "_short_id", "_side_dict",
)


@pytest.mark.parametrize("name", _PRE_SPLIT_ENGINE)
def test_the_engine_hub_still_exports_every_pre_split_name(name):
    assert hasattr(engine_hub, name), (
        f"`main.prober.engine.{name}` no longer resolves. The hub's whole job is that the "
        f"decomposition changed no import path — re-export it from whichever "
        f"`main/prober/engine/` module now owns it, or delete this entry deliberately if the "
        f"name is genuinely gone."
    )


@pytest.mark.parametrize("name", _PRE_SPLIT_SESSION)
def test_the_session_hub_still_exports_every_pre_split_name(name):
    assert hasattr(session_hub, name), (
        f"`main.prober.session.{name}` no longer resolves. Re-export it from whichever "
        f"`main/prober/session/` module now owns it, or delete this entry deliberately."
    )


def test_the_engine_hub_re_exports_from_every_module():
    """A module the hub forgot is a module no historical import path can reach.

    Not a size check — a COVERAGE one: every `.py` beside the hub must be named in one of its
    re-export blocks, so adding a module without wiring it up fails here rather than at the first
    caller that happens to want a name from it.
    """
    modules = {p.stem for p in _ENGINE_DIR.glob("*.py") if p.stem != "__init__"}
    src = (_ENGINE_DIR / "__init__.py").read_text(encoding="utf-8")
    missing = sorted(m for m in modules if f"main.prober.engine.{m} import" not in src)
    assert not missing, f"engine hub re-exports nothing from: {missing}"


def test_the_session_hub_covers_every_helper_module():
    """The mixin modules are reached through `ProbeSession`; the helper modules must be named."""
    src = (_SESSION_DIR / "__init__.py").read_text(encoding="utf-8")
    for helper in ("core", "serialize", "stats", "probe_targets"):
        assert f"main.prober.session.{helper} import" in src, (
            f"session hub re-exports nothing from `{helper}`")


@pytest.mark.parametrize("pkg_dir,pkg", [("engine", "main.prober.engine"),
                                         ("session", "main.prober.session")])
def test_no_submodule_imports_its_own_hub(pkg_dir, pkg):
    """The cycle guard. Submodules may import each other; none may import the package itself."""
    directory = _ENGINE_DIR if pkg_dir == "engine" else _SESSION_DIR
    offenders = []
    for path in sorted(directory.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            if mod == pkg or (mod or "").startswith(f"{pkg} "):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"submodule(s) import the `{pkg}` hub back: {offenders}. That closes an import cycle — "
        f"the hub imports every submodule — and its symptom is an AttributeError on a name that "
        f"plainly exists. Import from the sibling module that OWNS the name instead."
    )


@pytest.mark.parametrize("pkg_dir,pkg", [("engine", "main.prober.engine"),
                                         ("session", "main.prober.session")])
def test_every_submodule_imports_on_its_own(pkg_dir, pkg):
    directory = _ENGINE_DIR if pkg_dir == "engine" else _SESSION_DIR
    for path in sorted(directory.glob("*.py")):
        if path.name != "__init__.py":
            importlib.import_module(f"{pkg}.{path.stem}")


def test_probe_session_carries_every_command_family():
    """`ProbeSession` is assembled from mixins — a family dropped from the bases vanishes silently.

    The class was 2,068 lines before the split, so splitting the module without splitting the
    class would not have got under the size bound. The cost of that is a base list, and a base
    list can lose an entry without any import failing.
    """
    from main.prober.session import ProbeSession

    for method in ("run_summary", "battles", "decision_table", "battle_overview", "battle_turns",
                   "scan", "triage", "awareness_scan", "loops", "analyze", "find", "falsify",
                   "lookahead", "better_line", "replay_counterfactual", "falsify_scan",
                   "calibration", "probe", "switch_vs_info", "history_saliency", "probe_model"):
        assert callable(getattr(ProbeSession, method, None)), (
            f"`ProbeSession.{method}` is gone — a mixin dropped out of the base list in "
            f"`session/core.py`, and nothing else in the tree would fail at import time.")
