"""`gen3_clean_world_config_v1` — the four flags that make the CLEAN-WORLD reward reachable.

Spec: `designs/research_state/measurements/no_progress_tax_review_2026-08-29.md` §5 (probe N).
The target composition is **1 TERMINAL ∈ {+1, −1}, draw −1, ZERO PBRS, ZERO BIAS**, with a frozen
win-prob potential as the only dense signal. Before this change the flag surface could not express
it, and the reason is worth restating because it is not obvious from either flag's name:
`--all-shaping-pbrs` does TWO jobs at once — it folds five potentials AND it is
`_bias_term_active`'s master gate — so turning it off silences the potentials while REVIVING 25
BIAS terms. "No hand PBRS **and** no BIAS" sat in a hole between the two settings.

What each group pins:

* **DEFAULT BYTE-IDENTITY** — the whole point of shipping this mid-campaign. Four new fields, and a
  flagless launch must compose exactly as it did: 1 TERMINAL / 7 PBRS / 1 BIAS, ±30, draw −35.
* **THE CLEAN CONFIG SMOKE** — one named flag set, checked against the reward's OWN census rather
  than against a list retyped here. A census that agreed with a hand-written expectation but not
  with the folds would be worthless; `reward_class_composition` reads `_pbrs_term_active` /
  `_bias_term_active`, and the folds now call the SAME predicate, so agreement is structural.
* **THE FOLD/CENSUS UNIFICATION** — the census and the eight `_fold_*_pbrs` early-returns were two
  hand-maintained copies of one set of conditions. A revert-catcher pins that every PBRS field the
  census calls inactive really does stay 0.0 through a full `process_turn_reward`.
* **THE TERMINAL** — ±`victory_value` at both sites, including the PRE-CAP TIE, which shared the
  decisive-loss branch and was the one `-VICTORY_VALUE` literal probe N's B3 named.
* **THE OUTCOME ORDERING** — the single largest hazard in the clean arm. `draw_penalty` better than
  `-victory_value` makes running the 250-turn clock out the best non-winning outcome, so a losing
  agent's optimal play is to stall. Not an error (an arm may want it), but it must be SAID.
* **THE VERSION RECORD** — four resume-immutable fields; a config that records one meaning and a
  reward that reads another is the drift class `reward_defaults_test.py` exists for.

Run:
    python -m pytest src/agents/training/clean_world_config_test.py -q
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import dataclasses

import pytest

from agents.model.model_version import (
    _REWARD_FIELD_FLAGS,
    _REWARD_IMMUTABLE_FIELDS,
    ModelVersion,
    ModelVersionError,
)
from agents.model.model_version.constants import MODEL_CONFIG_VERSION
from agents.model.model_version.migrations import _migrate_config
from agents.training.reward_manager import (
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    reward_class_composition,
)
from agents.training.reward_weights import VICTORY_VALUE
from main.train_rl_agent import build_parser

#: THE CLEAN-WORLD FLAG SET, verbatim, in one place so the doc, the smoke and any future launch
#: quote the same string. `--win-prob-*` is the dense signal and is exercised by
#: `winprob_pbrs_test.py`; the reward composition is what this file is about.
CLEAN_WORLD_REWARD_FLAGS = ["--no-hand-shaping", "--victory-value", "1.0",
                            "--draw-penalty", "-1.0"]

_NEW_FIELDS = ("hand_shaping", "pbrs_material", "pbrs_belief", "victory_value")


def _args(argv):
    return build_parser().parse_args(list(argv))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 1. DEFAULT BYTE-IDENTITY — four new fields, zero change to a flagless run
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_flagless_default_composition_is_unchanged():
    """1 TERMINAL / 7 PBRS / 1 BIAS — the validated ai_v8 composition, before and after."""
    comp = reward_class_composition(RewardConfig())
    assert (comp["terminal"], comp["pbrs"], comp["bias"]) == (1, 7, 1)
    assert comp["bias_terms"] == ["no_progress_tax"]
    assert set(comp["pbrs_terms"]) == {"pbrs_material", "pbrs_belief", "pbrs_status",
                                       "pbrs_hazard", "pbrs_boost", "pbrs_opp_boosts", "pbrs_roar"}


def test_every_new_field_defaults_to_todays_behaviour():
    rc = RewardConfig()
    assert (rc.hand_shaping, rc.pbrs_material, rc.pbrs_belief) == (True, True, True)
    assert rc.victory_value == 30.0


def test_victory_value_defaults_to_the_module_constant_it_was_promoted_from():
    """The constant stays the single source of the DEFAULT. Two independent declarations of the
    same number is exactly the drift `_REWARD_IMMUTABLE_FIELDS` vs `RewardConfig` is pinned for."""
    assert RewardConfig().victory_value == VICTORY_VALUE == 30.0


def test_the_parser_defaults_reproduce_the_dataclass_defaults():
    a = _args([])
    for name in _NEW_FIELDS:
        assert getattr(a, name) == getattr(RewardConfig(), name), name
    assert RewardConfig.from_args(a) == RewardConfig()


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 2. THE CLEAN CONFIG SMOKE — the target composition, read off the reward's OWN census
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_clean_world_flag_set_produces_exactly_one_terminal_and_nothing_else():
    """THE headline. `{terminal ±1, draw −1, zero PBRS, zero BIAS}` — and every number here comes
    from `reward_class_composition`, not from a list retyped beside it."""
    rc = RewardConfig.from_args(_args(CLEAN_WORLD_REWARD_FLAGS))
    comp = reward_class_composition(rc)
    assert (comp["terminal"], comp["pbrs"], comp["bias"]) == (1, 0, 0)
    assert comp["pbrs_terms"] == [] and comp["bias_terms"] == []
    assert rc.victory_value == 1.0          # win +1, decisive loss AND pre-cap tie −1
    assert rc.draw_penalty == -1.0          # the owner's draw = loss ruling
    assert rc.hand_shaping is False


def test_no_combination_of_the_PRE_EXISTING_flags_could_reach_it():
    """The build gap, as an assertion — so a future reader does not "simplify" `hand_shaping` away.

    `--all-shaping-pbrs` is ALSO `_bias_term_active`'s master gate, so the two halves of the target
    are anti-correlated across it: ON ⇒ 5 potentials live, OFF ⇒ 25 BIAS terms live. Adding
    `--stall-pbrs` and both `--drop-*` flags does not close it either.
    """
    for cfg in (RewardConfig(all_shaping_pbrs=True, stall_pbrs=True,
                             pbrs_material=False, pbrs_belief=False),
                RewardConfig(all_shaping_pbrs=False, stall_pbrs=True,
                             pbrs_material=False, pbrs_belief=False,
                             drop_redundant_bias=True, drop_switch_bias=True)):
        comp = reward_class_composition(cfg)
        assert comp["pbrs"] + comp["bias"] > 0, (
            "a pre-existing-flag combination reached 0 PBRS + 0 BIAS — if that is now genuinely "
            "possible, --hand-shaping's rationale has changed and this file must be rewritten, not "
            "the assertion loosened")


@pytest.mark.parametrize("field,term", [("pbrs_material", "pbrs_material"),
                                        ("pbrs_belief", "pbrs_belief")])
def test_each_new_individual_flag_removes_its_own_term_and_only_it(field, term):
    base = reward_class_composition(RewardConfig())["pbrs_terms"]
    off = reward_class_composition(RewardConfig(**{field: False}))["pbrs_terms"]
    assert set(base) - set(off) == {term}


def test_the_individual_flags_are_INDEPENDENT_of_all_shaping_pbrs():
    """Probe N's warning, as a test: the new gates must not be wired through `asp`, whose OFF state
    revives the BIAS class. Both directions are checked so neither implication can creep in."""
    assert reward_class_composition(
        RewardConfig(all_shaping_pbrs=False))["pbrs_terms"] == ["pbrs_material", "pbrs_belief"]
    assert reward_class_composition(
        RewardConfig(pbrs_material=False, pbrs_belief=False))["bias_terms"] == ["no_progress_tax"]


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 3. THE FOLD / CENSUS UNIFICATION — the census must describe what the folds actually emit
# ──────────────────────────────────────────────────────────────────────────────────────────────

_CENSUS_CASES = [
    ("default", RewardConfig()),
    ("clean", RewardConfig(hand_shaping=False)),
    ("no-material", RewardConfig(pbrs_material=False)),
    ("no-belief", RewardConfig(pbrs_belief=False)),
    ("no-both", RewardConfig(pbrs_material=False, pbrs_belief=False)),
    ("v9", RewardConfig(all_shaping_pbrs=False)),
    ("fully-pbrs", RewardConfig(all_shaping_pbrs=True, stall_pbrs=True)),
]


def _emitted(cfg, n_turns=4):
    """Every reward field a real `process_turn_reward` sequence makes NON-ZERO under `cfg`.

    Several turns, with damage / a faint / boosts / hazards, so the potentials actually MOVE — a
    single static turn leaves every ΔΦ at 0 and would let an ungated fold pass as gated.
    """
    from agents.training.progress_clock import ProgressClock
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live
    clock = ProgressClock()
    mgr = Gen3RewardManager(config=cfg, progress_clock=clock)
    seen = set()
    # Each board moves a DIFFERENT potential: HP (Φ_mat, Φ_belief), spikes (Φ_hazard), an opponent
    # boost (Φ_boost / Φ_opp_boosts / Φ_roar), a status (Φ_status). A single static turn leaves
    # every ΔΦ at 0 and would let an ungated fold pass as gated.
    def _board(t):
        live = _full_team_live(our_hp=1.0 - 0.15 * t, opp_alive=6 - (t // 2))
        if t >= 1:
            live.opp.side_conditions = {"spikes": min(3, t)}
        if t >= 2:
            live.ours.active.boosts = {"atk": t - 1}
            live.opp.active.boosts = {"spa": t - 1}
        if t >= 3:
            live.opp.active.status = "par"
        return live

    for t in range(n_turns):
        clock.last_penalty = -0.15 if t else 0.0      # exercise the BIAS tilt too
        mgr.process_turn_reward(_Battle(_board(t), turn=t + 1), _delta())
        seen |= {n for n, v in dataclasses.asdict(mgr._last_breakdown).items() if v}
    return seen


@pytest.mark.parametrize("name,cfg", _CENSUS_CASES, ids=[c[0] for c in _CENSUS_CASES])
def test_a_term_the_census_calls_INACTIVE_never_emits(name, cfg):
    """The census is a promise about the FOLDS; this is the promise checked against them.

    `_hand_pbrs_on` delegating to `_pbrs_term_active` makes them one declaration, so this is the
    revert-catcher: re-inline a hand-written `if not self.config.all_shaping_pbrs` into a fold and
    the two can silently disagree again — which is the state this change found them in.
    """
    inactive = (set(RewardBreakdown.registry_fields(RewardClass.PBRS))
                - set(reward_class_composition(cfg)["pbrs_terms"]))
    assert not (inactive & _emitted(cfg)), f"{name}: census says off, the fold emitted anyway"


def test_no_hand_shaping_emits_NOTHING_but_the_terminal():
    """The strongest statement of the clean composition: not "the census says zero" but "over a
    sequence of real turns, no PBRS and no BIAS field ever became non-zero"."""
    emitted = _emitted(RewardConfig(hand_shaping=False))
    shaping = set(RewardBreakdown.registry_fields(RewardClass.PBRS)) | \
        set(RewardBreakdown.registry_fields(RewardClass.BIAS))
    assert not (emitted & shaping), sorted(emitted & shaping)
    # ...and the SAME sequence under the defaults does emit shaping, so the check is not vacuous.
    assert _emitted(RewardConfig()) & shaping


def test_the_default_keeps_the_one_tilt_that_no_hand_shaping_removes():
    """The distinction between the two flags, in one line: --all-shaping-pbrs deliberately KEEPS
    `no_progress_tax` as the acknowledged anti-stall bias; --no-hand-shaping takes it too."""
    assert reward_class_composition(RewardConfig())["bias_terms"] == ["no_progress_tax"]
    assert reward_class_composition(RewardConfig(hand_shaping=False))["bias_terms"] == []


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 4. THE TERMINAL — ±victory_value at BOTH sites, pre-cap tie included (B2 + B3)
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _win_loss(cfg, *, turn, **outcome) -> float:
    """Run the REAL `process_turn_reward` and read `win_loss` off the breakdown it produced.

    Reuses `reward_test_fakes`'s battle/live fixtures rather than re-implementing the branch —
    a test that recomputes the expression it is checking proves only that it can copy it.
    """
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live
    mgr = Gen3RewardManager(config=cfg)
    mgr.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())   # seed _prev_phi_mat
    mgr.process_turn_reward(_Battle(_full_team_live(**outcome), turn=turn), _delta())
    return mgr._last_breakdown.win_loss


@pytest.mark.parametrize("victory", [30.0, 1.0])
def test_a_win_scores_plus_victory_and_a_decisive_loss_minus_it(victory):
    cfg = RewardConfig(victory_value=victory)
    assert _win_loss(cfg, turn=20, our_alive=3, opp_alive=0, won=True, finished=True) == victory
    assert _win_loss(cfg, turn=20, our_alive=0, opp_alive=3, lost=True, finished=True) == -victory


@pytest.mark.parametrize("victory", [30.0, 1.0])
def test_the_PRE_CAP_TIE_rides_the_same_config_field(victory):
    """B3: `finished and not won and not lost and turn < cap` was a hardcoded `-VICTORY_VALUE`
    literal sharing the decisive-loss branch. It now reads `victory_value` — the default is
    preserved AND the ±1 arm is coherent instead of scoring a tie at −30 beside a −1 loss."""
    assert _win_loss(RewardConfig(victory_value=victory), turn=20, finished=True) == -victory


def test_a_TIMEOUT_still_takes_draw_penalty_not_the_terminal():
    from agents.training.reward_manager import _TIMEOUT_TURN_CAP
    assert _win_loss(RewardConfig(victory_value=1.0, draw_penalty=-1.0),
                     turn=_TIMEOUT_TURN_CAP, lost=True, finished=True) == -1.0
    assert _win_loss(RewardConfig(victory_value=1.0, draw_penalty=-1.2),
                     turn=_TIMEOUT_TURN_CAP + 5, lost=True, finished=True) == -1.2


def test_the_default_terminal_is_still_exactly_plus_or_minus_thirty():
    """The byte-identity claim at the one place a run's return scale is set."""
    assert _win_loss(RewardConfig(), turn=20, our_alive=3, opp_alive=0,
                     won=True, finished=True) == 30.0
    assert _win_loss(RewardConfig(), turn=20, our_alive=0, opp_alive=3,
                     lost=True, finished=True) == -30.0


# ──────────────────────────────────────────────────────────────────────────────────────────────
# 5. THE OUTCOME ORDERING — the clean arm's largest hazard, stated once and loudly
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_draw_equal_to_loss_is_expressible_and_is_what_the_clean_set_asks_for():
    """The owner's ruling. `--draw-penalty` already carried the value; what was missing was a
    `victory_value` for it to be equal to."""
    rc = RewardConfig.from_args(_args(CLEAN_WORLD_REWARD_FLAGS))
    assert rc.draw_penalty == -rc.victory_value


def _resolve(argv):
    from main.train.config import resolve_config
    parser = build_parser()
    return resolve_config(parser.parse_args(["--steps", "1", "--debug", *argv]), parser)


@pytest.mark.parametrize("argv,needle", [
    (["--victory-value", "0"], "must be > 0"),
    (["--victory-value", "-1"], "must be > 0"),
    (["--win-prob-pbrs-source", "x.zip"], "requires --win-prob-pbrs-coef"),
])
def test_the_config_gates_refuse_the_ways_this_can_be_wrong(argv, needle, capsys):
    with pytest.raises(SystemExit):
        _resolve(argv)
    assert needle in capsys.readouterr().err


def test_a_draw_better_than_a_loss_WARNS_rather_than_refusing(capsys):
    """It is a legitimate thing to want, so it is not an error — but with every anti-stall term
    removed nothing else in the clean arm opposes running the clock out, so it must be said."""
    _resolve(["--victory-value", "1.0", "--draw-penalty", "0.0"])
    out = capsys.readouterr().out
    assert "ORDERING" in out and "stall" in out


def test_the_clean_flag_set_itself_does_NOT_warn(capsys):
    _resolve(CLEAN_WORLD_REWARD_FLAGS)
    assert "ORDERING" not in capsys.readouterr().out



# ──────────────────────────────────────────────────────────────────────────────────────────────
# 6. THE VERSION RECORD — resume-immutable, migrated, and agreeing with RewardConfig
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_every_new_field_is_resume_immutable_and_names_its_flag():
    for name in _NEW_FIELDS:
        assert name in _REWARD_IMMUTABLE_FIELDS, name
        assert name in _REWARD_FIELD_FLAGS, name


def test_the_immutable_defaults_agree_with_RewardConfig():
    """Two separate declarations of one field set: a divergence would make an ABSENT field mean one
    thing to the reward and another to the version record."""
    rc = {f.name: f.default for f in dataclasses.fields(RewardConfig)}
    for name in _NEW_FIELDS:
        assert _REWARD_IMMUTABLE_FIELDS[name] == rc[name], name


def test_a_pre_v105_config_migrates_to_TODAYS_behaviour_rather_than_refusing():
    """Not a guess about the past: the flags did not exist, the two potentials were unconditional,
    and `victory_value` was the module constant in every run ever."""
    assert MODEL_CONFIG_VERSION >= 105
    out = _migrate_config({"config_version": 104})
    assert out["hand_shaping"] is True and out["pbrs_material"] is True
    assert out["pbrs_belief"] is True and out["victory_value"] == 30.0
    assert out["win_prob_pbrs_source"] is None
    assert out["config_version"] == MODEL_CONFIG_VERSION
    # a recorded value migrates UNTOUCHED
    assert _migrate_config({"config_version": 104, "victory_value": 1.0})["victory_value"] == 1.0


def _saved_version(**reward_fields):
    """A ModelVersion as a run with these reward fields recorded it (reward_defaults_test's shape)."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    return dataclasses.replace(v, **reward_fields)


def test_resuming_a_clean_world_run_against_default_flags_is_a_LOUD_refusal():
    """A run's reward must never flip underneath it — and the error has to name the flags to
    re-pass, or the check is a wall rather than a gate."""
    saved = _saved_version(hand_shaping=False, victory_value=1.0, draw_penalty=-1.0)
    with pytest.raises(ModelVersionError) as e:
        saved.check_reward_config(RewardConfig())
    msg = str(e.value)
    assert "--no-hand-shaping" in msg and "--victory-value 1.0" in msg
    # The strongest form of "actionable": take the flags back out of the message, feed them to the
    # REAL parser, and the config they build must pass the very check that produced the message.
    fix = msg.split("re-pass `")[1].split("`")[0]
    saved.check_reward_config(RewardConfig.from_args(_args(fix.split())))


def test_a_clean_world_run_resumes_flaglessly_under_ITS_OWN_flags():
    saved = _saved_version(hand_shaping=False, victory_value=1.0, draw_penalty=-1.0)
    saved.check_reward_config(RewardConfig.from_args(_args(CLEAN_WORLD_REWARD_FLAGS)))
