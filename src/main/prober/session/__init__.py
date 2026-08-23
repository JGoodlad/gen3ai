"""Programmatic access to the probing infrastructure — for agents and scripts.

``ProbeSession`` is a thin, framework-agnostic facade over discovery + engine that
returns **JSON-serializable** dicts, so an agent can investigate a model's
behaviour without a UI. A typical investigation:

    sess = ProbeSession("models/run_.../")
    sess.run_summary()                       # orient: steps, opponents, win/loss, identity
    sess.battles(outcome="loss", step=8_000_000)   # pick battles to look at
    sess.scan(outcome="loss", opponent="aggressive_v2")  # MODEL-FREE: worst turn PER battle, ranked
    sess.battle_overview(battle_id)          # MODEL-FREE digest: per-decision rows + `notable`
    sess.find(battle_id, "value_drop", limit=5)    # rank decisions by where V(s) cratered
    sess.find(battle_id, "disagree")         # decisions the loaded model disagrees with
    sess.analyze(battle_id, inv)             # full forensic analysis of one decision

A ``battle_id`` is either the trace's ``*_summary.json`` path (as returned by
``battles()``/``run_summary``) or a short ``step_<N>/<Opponent>/<outcome>_<idx>``
id. Model loading uses the same exact→nearest→recent ladder as the web front end, cached.
The matching CLI is ``python -m main.prober.query``.

**This package is a re-export HUB and keeps `main.prober.session`'s whole public surface.** It was
one 2,573-line module until 2026-08-23; `ProbeSession` and every module-level helper any caller or
test ever imported still resolve from it, so no import path changed. `ProbeSession` is assembled
from one MIXIN per command family — the class alone was 2,068 lines, so splitting the module
without splitting the class would not have got under the bound.

THE MODULE MAP:

    core.py            `ProbeSession` — construction, shared internals, the resolution ladder
    reading.py         MODEL-FREE orientation: run_summary · battles · decision_table ·
                       battle_overview · battle_turns
    scans.py           RUN-LEVEL model-free folds: scan · awareness_scan · loops · triage
    trace_io.py        the trace's sibling files (protocol log, privileged teams, our HP types)
    analysis.py        the per-decision deep read: analyze (loads the model) · find
    counterfactual.py  falsify · lookahead · better_line · replay_counterfactual
    aggregate.py       falsify_scan · calibration — the two run-level counterfactual folds
    probes.py          probe · switch_vs_info · history_saliency
    serialize.py       the JSON-shaping leaves
    stats.py           the pure statistics (loop aggregation, discounted returns, reliability)
    probe_targets.py   the representation-probe target table
"""

from __future__ import annotations

from main.prober.session.core import (   # noqa: F401 — re-export hub
    ProbeSession, _DEFAULT_GAMMA, _MAX_CACHED_MODELS,
)
from main.prober.session.serialize import (   # noqa: F401 — re-export hub
    _active_str, _choice_dict, _chosen_prob, _mon_dict, _opp_intent_dict, _r, _short_id,
    _side_dict,
)
from main.prober.session.stats import (   # noqa: F401 — re-export hub
    _calibration_stats, _discounted_returns, _loop_aggregate, _ratio, _rate_by,
    _reliability_curve, _reliability_gap_at,
)
from main.prober.session.probe_targets import (   # noqa: F401 — re-export hub
    _PROBE_TARGETS, _base_spe, _belief_pko_group, _big_hit_label, _dmg_label,
    _faint_healthy_label, _faint_label, _faster_group, _faster_label, _FIXED_DAMAGE_MOVES,
    _opp_move_id, _opp_status_move_label, _opp_switch_label, _prov,
)
