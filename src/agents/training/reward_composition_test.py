"""`agents/training/reward_composition.py` — the recorded composition, and the INERT-flag list.

🚨 **THE DEFECT THIS FILE EXISTS FOR, observed on the LIVE `ai_v12_01_winprob_critic` arm.**
`model_config.json` records `all_shaping_pbrs=True`, `pbrs_material=True`, `pbrs_belief=True` — their
argparse defaults, faithfully recorded — while the same run's startup announcer prints
`1 TERMINAL + 0 PBRS + 0 BIAS (none — fully policy-invariant)`. Both statements are correct and they
disagree, because `--no-hand-shaping` makes all three unreachable without changing what any of them
RECORDS. A reader who opens the config concludes shaping was on.

Two answers ship, and the split matters: the ANNOUNCED LINE is recorded into `metadata.json`'s
`reward_composition` block (so a run's composition survives the log rotation that takes the launcher
child log), and `inert_reward_flags` is written into `model_config.json` **beside** the values it
describes — never in place of them, because `check_reward_config` compares each recorded value
against the resuming argv's, and that argv still says `all_shaping_pbrs=True`.

Run:
    pytest src/agents/training/reward_composition_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import json

import pytest

from agents.training.reward_composition import (
    _FLAG_TERMS,
    format_reward_composition,
    inert_reward_flags,
    reward_class_composition,
    reward_composition_block,
)
from agents.training.reward_manager import RewardConfig

#: The live arm's reward composition, verbatim from `design_winprob_only_critic.md` §5.4.
_WINPROB = dict(hand_shaping=False, victory_value=1.0, draw_penalty=0.0, terminal_indicator=True)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE DEFECT, named
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_three_flags_the_live_arm_records_TRUE_are_reported_INERT():
    """The exact reading that misled: all three are `True` in the config AND all three are
    unreachable. The list is what closes the gap between those two true statements."""
    cfg = RewardConfig(**_WINPROB)
    assert (cfg.all_shaping_pbrs, cfg.pbrs_material, cfg.pbrs_belief) == (True, True, True), (
        "the premise: these record their defaults, so the artifact reads as shaping-on")
    comp = reward_class_composition(cfg)
    assert (comp["pbrs"], comp["bias"]) == (0, 0), "...while the census says nothing is emitted"

    inert = inert_reward_flags(cfg)
    for flag in ("all_shaping_pbrs", "pbrs_material", "pbrs_belief"):
        assert flag in inert, f"{flag} records True and can emit nothing — it must be named"


def test_draw_penalty_is_inert_under_the_indicator_terminal_and_LIVE_without_it():
    """The MAGNITUDE-shaped half, which no term-activity table could express: under
    `--terminal-indicator` the terminal pays `+victory_value` on a win and `0.0` on a loss, a tie
    AND a timeout alike, so the flag's NUMBER is simply never read — the term is still emitted."""
    assert "draw_penalty" in inert_reward_flags(RewardConfig(**_WINPROB))
    shaped = RewardConfig(hand_shaping=False, victory_value=1.0, draw_penalty=-1.0)
    assert "draw_penalty" not in inert_reward_flags(shaped), (
        "without the indicator terminal the draw magnitude is read and must not be called inert")


def test_the_default_production_config_reports_only_genuinely_dead_flags():
    """The no-false-positive direction. Under the production composition the hand potentials are
    LIVE, so none of them may appear; what does appear is the weight-gated family sitting at 0 and
    the drop flag whose two terms `--all-shaping-pbrs` has already zeroed."""
    inert = inert_reward_flags(RewardConfig())
    for live in ("all_shaping_pbrs", "pbrs_material", "pbrs_belief", "draw_penalty",
                 "no_progress_penalty", "mat_alive_weight"):
        assert live not in inert, f"{live} is LIVE under the default composition"
    assert set(inert) == {"switch_bias_weight", "self_ko_hp_penalty", "drop_redundant_bias"}


def test_bias_additivity_is_deliberately_ABSENT_even_with_an_empty_bias_class():
    """It is inert in EFFECT (the refund is identically 0 with nothing to refund) but by having
    nothing to act on rather than by a gate. Listing it would turn this into "flags that happen not
    to matter", which is a different and much weaker claim."""
    assert "bias_additivity" not in inert_reward_flags(RewardConfig(**_WINPROB))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# IT IS DERIVED FROM THE FOLDS' OWN GATES — never a second copy of them
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_every_flag_in_the_table_names_only_REAL_registry_terms():
    """The one hand-maintained fact here is which switch reaches which term. A typo would silently
    make a flag look permanently inert (no term of that name is ever active)."""
    from agents.training.reward_manager import RewardBreakdown

    known = set(RewardBreakdown._REGISTRY)
    for flag, terms in _FLAG_TERMS.items():
        assert terms, f"{flag} governs no terms — it would report inert forever"
        unknown = set(terms) - known
        assert not unknown, f"{flag} names terms that are not in the registry: {sorted(unknown)}"


def test_a_flag_is_inert_EXACTLY_when_the_folds_own_predicates_say_so():
    """The anti-drift property: recompute the verdict straight from `_pbrs_term_active` /
    `_bias_term_active` — the SAME predicates `Gen3RewardManager._hand_pbrs_on` delegates to — and
    require agreement across four very different compositions."""
    from agents.training.reward_composition import _bias_term_active, _pbrs_term_active
    from agents.training.reward_manager import RewardBreakdown, RewardClass

    configs = [RewardConfig(), RewardConfig(**_WINPROB),
               RewardConfig(all_shaping_pbrs=False), RewardConfig(stall_pbrs=True)]
    for cfg in configs:
        got = set(inert_reward_flags(cfg)) - {"draw_penalty"}
        expected = set()
        for flag, terms in _FLAG_TERMS.items():
            live = False
            for t in terms:
                cls = RewardBreakdown._REGISTRY[t]
                live = live or (_pbrs_term_active(cfg, t) if cls is RewardClass.PBRS
                                else _bias_term_active(cfg, t))
            if not live:
                expected.add(flag)
        assert got == expected, f"drifted on {cfg}"


def test_it_is_sorted_and_stable_so_two_configs_diff_cleanly():
    a = inert_reward_flags(RewardConfig(**_WINPROB))
    assert a == sorted(a) == sorted(set(a))
    assert a == inert_reward_flags(RewardConfig(**_WINPROB))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE metadata.json BLOCK — additive, and it carries the ANNOUNCED line
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_block_is_ADDITIVE_over_the_census_every_existing_reader_uses():
    """`model_build`'s two gamma asserts read `["pbrs"]`; `callbacks` and the `reward/` export read
    the three term lists. None of them may move."""
    for cfg in (RewardConfig(), RewardConfig(**_WINPROB), RewardConfig(all_shaping_pbrs=False)):
        census, block = reward_class_composition(cfg), reward_composition_block(cfg)
        for key, value in census.items():
            assert block[key] == value, f"{key} moved"
        assert set(block) - set(census) == {
            "composition_line", "class_shares", "inert_reward_flags"}


def test_the_block_carries_the_announcers_string_VERBATIM():
    """The launch PRINTS this and nothing kept it — a launcher rotates the child log away. It is
    the announcer's own output rather than a re-render, so the recorded line and the printed line
    cannot drift."""
    cfg = RewardConfig(**_WINPROB)
    line = reward_composition_block(cfg)["composition_line"]
    assert line == format_reward_composition(cfg)
    assert "1 TERMINAL + 0 PBRS + 0 BIAS" in line and "fully policy-invariant" in line


def test_the_class_shares_partition_the_active_terms():
    for cfg in (RewardConfig(), RewardConfig(**_WINPROB), RewardConfig(all_shaping_pbrs=False)):
        block = reward_composition_block(cfg)
        total = block["terminal"] + block["pbrs"] + block["bias"]
        assert sum(block["class_shares"].values()) == pytest.approx(1.0)
        for k in ("terminal", "pbrs", "bias"):
            assert block["class_shares"][k] == pytest.approx(block[k] / total)
    assert reward_composition_block(RewardConfig(**_WINPROB))["class_shares"]["terminal"] == 1.0


def test_the_block_is_JSON_SERIALIZABLE_because_metadata_json_carries_it():
    json.dumps(reward_composition_block(RewardConfig(**_WINPROB)))


def test_the_launch_records_the_BLOCK_and_not_the_bare_census():
    """The seam: `train_rl_agent` must build the richer block, or the announced line reaches
    nothing. Read off the source, because the alternative is a whole training run."""
    from main.train import entry_source

    src = entry_source()
    assert "reward_composition = reward_composition_block(reward_config)" in src
    assert "reward_class_composition(reward_config)" not in src, (
        "the bare census would drop the line, the shares and the inert list")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE model_config.json ANNOTATION — beside the values, and it must not break a resume
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _real_layout():
    """The REAL encoder layout — `from_layout_and_policy_kwargs` reads the embedding dims off it,
    so a stub dict is not a shortcut, it is a KeyError. Cached, because building it loads the data
    singletons."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _version_for(cfg):
    from agents.model.model_version import ModelVersion
    return ModelVersion.from_layout_and_policy_kwargs(
        _real_layout(), {"net_arch": [512, 512]}, reward_config=cfg)


def _write_config(tmp_path, cfg):
    from agents.model.snapshot import save_model_snapshot

    version = _version_for(cfg)
    save_model_snapshot(str(tmp_path), version)
    with open(tmp_path / "model_config.json") as f:
        return version, json.load(f)


def test_the_config_file_carries_the_annotation_BESIDE_the_unchanged_values(tmp_path):
    """Both halves in one assertion: the annotation is present AND the three flags still record
    exactly what they always did."""
    _version, data = _write_config(tmp_path, RewardConfig(**_WINPROB))
    assert "all_shaping_pbrs" in data["inert_reward_flags"]
    for flag in ("all_shaping_pbrs", "pbrs_material", "pbrs_belief"):
        assert data[flag] is True, "the recorded VALUE must not move — see check_reward_config"


def test_a_config_carrying_the_annotation_still_LOADS(tmp_path):
    """`from_json_file` does `cls(**data)`, which TypeErrors on a stale key — so the annotation is
    POPped by the version-INDEPENDENT sanitizer rather than becoming a field."""
    from agents.model.model_version import ModelVersion

    version, _data = _write_config(tmp_path, RewardConfig(**_WINPROB))
    loaded = ModelVersion.from_json_file(str(tmp_path / "model_config.json"))
    assert loaded.all_shaping_pbrs is True
    assert not hasattr(loaded, "inert_reward_flags")
    assert loaded.arch_signature == version.arch_signature


def test_the_annotated_config_RESUMES_at_the_defaults_the_resuming_argv_rebuilds(tmp_path):
    """🚨 THE REASON THE VALUES ARE NOT REWRITTEN. `check_reward_config` compares each recorded
    value against the one `RewardConfig.from_args` builds from the RESUMING argv, and that argv
    still carries `all_shaping_pbrs=True` — its default. A config recording False would FATAL every
    restart of the very run the annotation describes, and that run restarts every three hours."""
    from agents.model.model_version import ModelVersion

    cfg = RewardConfig(**_WINPROB)
    _version, _data = _write_config(tmp_path, cfg)
    saved = ModelVersion.from_json_file(str(tmp_path / "model_config.json"))
    saved.check_reward_config(cfg)          # the resuming process rebuilds the same config

    # ...and the guard is not vacuous: a genuinely different reward still FATALs.
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError):
        saved.check_reward_config(RewardConfig(victory_value=30.0, draw_penalty=-35.0))


def test_a_SHAPED_config_records_the_key_too_so_ABSENT_never_reads_as_UNKNOWN(tmp_path):
    """A shaped run's list is short but PRESENT. An absent key and an empty list would otherwise be
    the same on disk, and only one of them is a measurement."""
    _version, data = _write_config(tmp_path, RewardConfig())
    assert data["inert_reward_flags"] == ["drop_redundant_bias", "self_ko_hp_penalty",
                                          "switch_bias_weight"]


def test_to_json_itself_stays_exactly_asdict(tmp_path):
    """Several tests round-trip `ModelVersion(**json.loads(v.to_json()))` with no migration, so the
    annotation belongs to the WRITER, not to the serializer."""
    import dataclasses

    from agents.model.model_version import ModelVersion

    v = _version_for(RewardConfig(**_WINPROB))
    assert json.loads(v.to_json()) == dataclasses.asdict(v)
    ModelVersion(**json.loads(v.to_json()))     # must not TypeError
