"""Pure forensic-replay analysis — the single source of truth for every prober surface.

No printing, no Textual, no file IO beyond reading the already-loaded summary dict / npz arrays.
Every torch call goes through the injected ``model`` object (see ``model.ProbeModel``), so the
whole engine is testable with a fake.

The analysis for one invocation (decision point):
  - faithfulness: recorded action probs (from the summary) vs a live re-run,
  - matchups: the active mon's per-move type multipliers the model saw,
  - intervention: sweep the chosen move's matchup 0×→4× and watch P shift,
  - saliency: |d logit(chosen) / d obs| aggregated per obs block.

**This package is a re-export HUB and keeps `main.prober.engine`'s whole public surface.** It was
one 3,058-line module until 2026-08-23; every name it ever exported still resolves from it, so no
caller changed. The precedent is `features_extractor.py` / `main/train/`: one file per concern,
the original import path kept.

THE MODULE MAP — a strict DAG, leaves first:

    views.py       every frozen dataclass an analysis returns (the DATA MODEL; no numpy, no IO)
    util.py        the shared leaves: percent parsing, npz access, species keys, move predicates
    opponents.py   opponent-NAME ordering (sentinels first, strongest first)
    flags.py       per-invocation flags + the cure-option ("heal ≠ cure") helpers
    protocol.py    reading the RAW Showdown protocol out of a trace's `*_replay.html`
    board.py       the BOARD read-model
    timeline.py    the RESULT timeline — re-attributed, one line per action
    beliefs.py     species / move / exclusive-species beliefs + the refinement trajectory
    intent.py      the α/β opponent-intent read + `awareness_text`
    spread.py      believed vs TRUE derived spreads (the DamageOperator's stat input)
    switch_in.py   forced-switch OUTGOING damage per bench candidate
    decode.py      the obs/model decoding units (faithfulness, matchups, intervention, saliency)
    analyze.py     `analyze_invocation` — the top-level entry — plus `build_meta` / value-dist
    taxonomy.py    loss attribution: the turning-point category table
    probes.py      representation probing (`fit_probe`)
"""

from __future__ import annotations

# Every import below is a live re-export: `from main.prober.engine import <anything>` resolves
# exactly as it did when this was one module. Ordered leaves-first, mirroring the map above.
from main.prober.engine.views import (   # noqa: F401 — re-export hub
    ActionRow, BELIEF_NAME_CAVEAT, BeliefSlotView, BeliefTrajectoryPoint, BeliefTrajectoryView,
    BeliefTruthView, BeliefView, BoardView, ExclusiveBeliefView, ExclusiveSlotView,
    IncomingBeliefView, InterventionRow, InterventionSweep, InvocationAnalysis, MatchupView,
    MonState, MoveBeliefView, OppFullMon, OppFullTeamView, OppIntentCandidate, OppIntentOption,
    OppIntentView, OppMonTruth, OppMoveBelief, Saliency, SaliencyBlock, SideBoard,
    SpreadBeliefView, SpreadSlotBelief, SpreadStatRow, SwitchInOutgoingRow, SwitchInOutgoingView,
    ThreatView, TraceMeta, ValueDistView, ValueView, WinProbView,
)
from main.prober.engine.util import (   # noqa: F401 — re-export hub
    _BP0_DAMAGING, _has_state, _loss_pct, _multiplier_meaningful, _norm_species, _npz_array,
    _npz_value, _npz_win_prob, _pct, parse_pct,
)
from main.prober.engine.opponents import (   # noqa: F401 — re-export hub
    _SENTINEL_RE, opponent_rank, sort_opponents,
)
from main.prober.engine.flags import (   # noqa: F401 — re-export hub
    _CURABLE_STATUSES, UNCERTAIN_THRESHOLD, has_curable_status, is_status_cure,
    self_cure_options, summary_flags,
)
from main.prober.engine.protocol import (   # noqa: F401 — re-export hub
    _LOG_BLOCK_RE, move_order_from_protocol, parse_protocol_log, protocol_action_fate,
    protocol_for_turn, protocol_move_result,
)
from main.prober.engine.board import (   # noqa: F401 — re-export hub
    _BENCH_RE, _MOVE_PLACEHOLDER_RE, _merge_team, _our_items, _parse_bench, _retype_hp,
    _side_board, _team_entry, build_board, build_our_hp_types,
)
from main.prober.engine.timeline import (   # noqa: F401 — re-export hub
    _FAINT_EVENT_RE, _SENT_IN, _SEP, _STATUS_EVENT_RE, _SWITCH_IN_HIT_MAX_HP, _is_attack,
    _no_effect_reason, _parse_outcome_action, _timeline_for, CANT_PHRASE, NO_EFFECT_TEXT,
    build_result_timeline, cant_phrase, opp_voluntary_switch, surprise_phrase,
    timeline_entry_text,
)
from main.prober.engine.beliefs import (   # noqa: F401 — re-export hub
    _MAX_MOVES, _MOVE_ID_TO_NUM, _MOVE_NUM_TO_ID, _SPECIES_MAPS, _entropy_bits, _move_id_to_num,
    _move_maps, _norm_move, _softmax, _species_maps, belief_view_from_logits, build_belief,
    build_belief_trajectory, build_belief_truth, build_exclusive_belief, build_opp_full_team,
    move_belief_view, revealed_opp_species,
)
from main.prober.engine.intent import (   # noqa: F401 — re-export hub
    _beta_candidate, _matches_move, _move_display, _opp_actual_action, SWITCH_OPTION,
    awareness_text, build_opp_intent, opp_intent_text,
)
from main.prober.engine.spread import (   # noqa: F401 — re-export hub
    _SPREAD_BASE_KEY, _SPREAD_COLS, _SPREAD_PRIOR_CACHE, _derived_stat, _spread_prior_means,
    _true_derived_spread, build_spread_belief,
)
from main.prober.engine.switch_in import (   # noqa: F401 — re-export hub
    _as_ptype, _derived_hp, _hp_frac_from_str, build_switch_in_outgoing,
)
from main.prober.engine.decode import (   # noqa: F401 — re-export hub
    _MATCHUP_DIM, _SWEEP_MULTIPLIERS, _TEAM_SIZE, _active_slot, _display_hp, _faithfulness,
    _intervention_sweep, _matchups, _saliency, _saliency_from_grad, _switch_prob_sum, _threats,
    _value_saliency, decode_incoming_belief, history_slot_saliency,
)
from main.prober.engine.analyze import (   # noqa: F401 — re-export hub
    _dist_quantile, analyze_invocation, build_meta, build_value_dist,
)
from main.prober.engine.taxonomy import (   # noqa: F401 — re-export hub
    _Cat, _f, _was_winning, BELIEF_FIRED_PKO, BELIEF_UNDERREAD_PKO, CRITIC_CONFIDENT_V,
    FAINTED_HP, HEALTHY_HP, LOSS_TAXONOMY, SETUP_MOVES, STALL_NEAR_CAP, WP_EVEN_DEFAULT,
    attribute_turning_point,
)
from main.prober.engine.probes import (   # noqa: F401 — re-export hub
    _L2_GRID, _auc, _kfold_indices, _logistic_fit, _oof_predict, _ridge_fit, _selection_score,
    _sigmoid, _standardize, fit_probe,
)
