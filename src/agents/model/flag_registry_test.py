"""The consistency gate: every `flag_registry` row agrees with all five hand-synced surfaces.

A model-relevant toggle has to be spelled out in five places (argparse / `_resolve` /
`ARCH_ARG_KEYS` / `current_model_version` / the `ModelVersion` field). Two of those are now
GENERATED from the registry, so they cannot drift; these tests VALIDATE the other three, and every
failure names the missing site rather than the symptom.

The failure modes this is here to catch, all of which have happened:

  * a toggle in `ARCH_ARG_KEYS` but not on `ModelVersion` — the extractor builds it, the recorded
    config does not know about it, so a resume version-checks against an arch it does not build
  * a toggle with an argparse entry but no `_resolve` line — a FLAGLESS resume silently reverts it
    to OFF and then FATALs at `check_compatible` (or, worse, does not)
  * a toggle on `ModelVersion` but not a `current_model_version` keyword — an eval/self-play worker
    rebuilds a toggle-OFF gate and FATALs on the run's OWN snapshots
  * a `config_only` toggle that still has an argparse entry — the demotion did not actually happen,
    so the "frozen" value is still settable and the frozen default is a lie

The argparse surface is read from the `add_argument` CALLS rather than by grepping for a string —
a flag that exists only inside a comment must not count as present. It reads `main.train`'s
`entry_source()` (the hub + every phase module) rather than one file: the parser moved to
`main/train/parser.py` and `_resolve` to `main/train/config.py` on 2026-08-22, and a probe naming a
single path would have gone silently VACUOUS instead of red.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from agents.model import extractor_arch as EA
from agents.model.flag_registry import (
    REGISTRY, BY_NAME, Tier, Klass, cli_flags, config_only_flags, recorded_flags,
)
from agents.model.model_version import ModelVersion
from agents.model.snapshot import current_model_version

# The entry point is a PACKAGE since 2026-08-22 (`main/train/` + the `train_rl_agent.py`
# hub). `entry_source()` is its one canonical text, so a gate that READS the entry point
# keeps reading all of it instead of silently emptying when a phase moves.
from main.train import entry_source


# --------------------------------------------------------------------------------- surface probes
def _train_source() -> str:
    return entry_source()


def _argparse_option_strings() -> set:
    """Every option string `train_rl_agent`'s parser accepts.

    `main()` builds the parser inline, so rather than import-and-run it (which would spin up an
    asyncio loop and a training entry point) the `add_argument(...)` calls are extracted from the
    source and evaluated for their POSITIONAL string arguments only. Comments and help text are
    invisible to this by construction — it reads the call's arguments, not the file's text.
    """
    src = _train_source()
    opts: set = set()
    for call in re.finditer(r"parser\.add_argument\(\s*((?:\"[^\"]*\"\s*,?\s*)+)", src):
        for lit in re.finditer(r"\"(--[a-zA-Z0-9_-]+)\"", call.group(1)):
            opts.add(lit.group(1))
    assert "--model" in opts and "--steps" in opts, (
        "the add_argument scan found no recognisable flags — the parser's shape changed and this "
        "probe needs updating (it is a gate, so a silently-empty result must fail loudly)")
    return opts


def _resolved_names() -> set:
    """Every name passed to the `_resolve("name", default)` inheritance helper."""
    return set(re.findall(r"_resolve\(\s*\"([a-z0-9_]+)\"", _train_source()))


def _argparse_action_for(dest: str):
    """The LIVE argparse action writing `dest` — built, not source-scanned.

    The option-string probe above reads source because it only needs literals; a DEFAULT has to
    come from the constructed parser, since it can be an expression. `build_parser()` exists
    precisely so the parser can be inspected without running a training job.
    """
    from main.train.parser import build_parser
    for action in build_parser()._actions:
        if action.dest == dest:
            return action
    return None


def _current_model_version_params() -> set:
    return set(inspect.signature(current_model_version).parameters)


def _model_version_fields() -> set:
    return {f.name for f in dataclasses.fields(ModelVersion)}


# ------------------------------------------------------------------------------------- the checks
@pytest.mark.parametrize("flag", cli_flags(), ids=lambda f: f.name)
def test_cli_flags_have_an_argparse_entry(flag):
    opts = _argparse_option_strings()
    assert flag.cli_flag in opts, (
        f"MISSING SITE — argparse: flag_registry declares {flag.name!r} as tier=cli, but "
        f"train_rl_agent.py's parser has no {flag.cli_flag!r} entry.\n"
        f"Either add the argparse entry, or demote the row to tier=config_only "
        f"(and give it the frozen default a CLI-launched run should get).")


@pytest.mark.parametrize("flag", cli_flags(), ids=lambda f: f.name)
def test_cli_flags_have_a_resolve_line(flag):
    """A cli-tier toggle without `_resolve` silently reverts to OFF on a flagless resume."""
    assert flag.arg in _resolved_names(), (
        f"MISSING SITE — _resolve: flag_registry declares {flag.name!r} as tier=cli, but "
        f"train_rl_agent.py has no `_resolve({flag.arg!r}, ...)` line.\n"
        f"Without it a FLAGLESS resume (`--model X --steps N`) does not inherit the saved value: "
        f"it falls back to the argparse default and the run either FATALs at check_compatible or "
        f"silently trains a different architecture.")


# The argparse default that MUST be None for the `_resolve` above to be reachable. Two entries
# are exempt because their CLI surface is a DIFFERENT flag that desugars into them, and the
# desugaring itself sets them to None when that flag is absent (see `resolve_config`).
_DESUGARED_ARGS = {"damage_matrices_outgoing", "damage_matrices_incoming"}


@pytest.mark.parametrize("flag", [f for f in cli_flags() if f.arg not in _DESUGARED_ARGS],
                         ids=lambda f: f.name)
def test_cli_flags_argparse_default_is_none(flag):
    """The `_resolve` line above is only REACHABLE when the argparse default is None.

    `_resolve(name, default)` fires on `getattr(args, name) is None`. An argparse entry that
    defaults to anything else therefore makes its own `_resolve` line DEAD CODE — the flagless
    resume reads the argparse default, never the checkpoint — while
    `test_cli_flags_have_a_resolve_line` still passes, because the line is PRESENT.

    That is not hypothetical. When this gate was written it failed on FIVE live flags:
    `value_threat_inject` (`store_true, default=False` — ON in the gen-17 production config, so a
    flagless resume of production would have FATALed at check_compatible), `opp_intent_coef`
    (`default=0.0`, and `opp_intent` is DERIVED from it, so the same), and the three v98/v99
    counterfactual heads `cf_evidential` / `cf_twin_heads` / `cf_shadow_critic`. Every one of them
    had the `_resolve` line the presence test asks for, and in every one of them it did nothing.
    """
    action = _argparse_action_for(flag.arg)
    assert action is not None, (
        f"no argparse action writes dest={flag.arg!r} — {flag.name!r} is tier=cli, so one must")
    assert action.default is None, (
        f"DEAD _resolve — argparse: {flag.cli_flag} defaults to {action.default!r}, not None, so "
        f"`_resolve({flag.arg!r}, ...)` in `main.train.config` can never fire and a FLAGLESS "
        f"resume silently reverts {flag.name!r} to that default.\n"
        f"Fix the DEFAULT, not the _resolve line: use `default=None` (with `action=BoolFlag` for a "
        f"bool, so `--no-{flag.cli_flag.lstrip('-')}` can still turn it off explicitly) and let "
        f"`_resolve` supply the OFF value for a fresh run.")


@pytest.mark.parametrize("flag", config_only_flags(), ids=lambda f: f.name)
def test_config_only_flags_have_no_cli_surface(flag):
    """The demotion has to be real: a config_only toggle must not be settable at launch."""
    opts = _argparse_option_strings()
    assert flag.cli_flag not in opts, (
        f"EXTRA SITE — argparse: flag_registry declares {flag.name!r} as tier=config_only "
        f"(frozen at {flag.default!r}), but {flag.cli_flag!r} is still an argparse entry.\n"
        f"A settable 'frozen' default is a lie — delete the argparse entry, or move the row back "
        f"to tier=cli.")
    assert flag.arg not in _resolved_names(), (
        f"EXTRA SITE — _resolve: {flag.name!r} is tier=config_only but still has a "
        f"`_resolve({flag.arg!r}, ...)` line, which reads an args attribute that no longer exists.")


@pytest.mark.parametrize("flag", recorded_flags(), ids=lambda f: f.name)
def test_recorded_flags_have_a_model_version_field(flag):
    assert flag.name in _model_version_fields(), (
        f"MISSING SITE — ModelVersion: flag_registry declares {flag.name!r} as "
        f"class={flag.klass.value}, which is RECORDED, but it is not a ModelVersion field.\n"
        f"The extractor would build it while model_config.json stayed silent, so a resume "
        f"version-checks against an architecture it does not build. Add the field, bump "
        f"MODEL_CONFIG_VERSION, and add the _migrate_config default.")


@pytest.mark.parametrize("flag", recorded_flags(), ids=lambda f: f.name)
def test_recorded_flags_are_current_model_version_keywords(flag):
    assert flag.name in _current_model_version_params(), (
        f"MISSING SITE — current_model_version: {flag.name!r} is recorded on ModelVersion but is "
        f"not a keyword of snapshot.current_model_version().\n"
        f"Eval / self-play workers rebuild their load gate through that function; without the "
        f"keyword the worker's gate is toggle-OFF and FATALs on the run's OWN pool snapshots.")


@pytest.mark.parametrize("flag", REGISTRY, ids=lambda f: f.name)
def test_every_toggle_reaches_the_extractor_exactly_once(flag):
    """Each row lands in exactly one of the three kwarg sources — never two, never none."""
    sources = [
        name for name, present in (
            ("ARCH_ARG_KEYS", flag.name in EA.ARCH_ARG_KEYS),
            ("_DERIVED", flag.name in EA._DERIVED),
            ("FROZEN_ARCH_KWARGS", flag.name in EA.FROZEN_ARCH_KWARGS),
        ) if present
    ]
    if flag.tier is Tier.CONSTRUCTOR_ONLY:
        assert sources == [], (
            f"{flag.name!r} is tier=constructor_only but appears in {sources} — a "
            f"constructor-only toggle is reachable ONLY by constructing the module.")
        return
    assert len(sources) == 1, (
        f"{flag.name!r} reaches the extractor via {sources or 'NOTHING'} — expected exactly one "
        f"source. derived={flag.derived}, tier={flag.tier.value}.")


def test_derived_key_set_matches_the_registry():
    """`_DERIVED`'s values are callables so it stays hand-written — its KEYS are still pinned."""
    declared = {f.name for f in REGISTRY if f.derived}
    assert set(EA._DERIVED) == declared, (
        f"extractor_arch._DERIVED keys {sorted(EA._DERIVED)} disagree with the registry rows "
        f"marked derived=True {sorted(declared)}. A derived kwarg with no registry row is invisible "
        f"to every check in this file.")


def test_generated_arch_arg_keys_are_identity_mappings():
    """`ARCH_ARG_KEYS` maps kwarg -> args attribute; the registry's premise is that they agree."""
    odd = {k: v for k, v in EA.ARCH_ARG_KEYS.items() if k != v}
    assert odd == {}, (
        f"ARCH_ARG_KEYS has non-identity rows {odd}. That is representable (a row may set "
        f"source_arg), but it breaks the one-name invariant the ModelVersion / "
        f"current_model_version checks rely on — add the row to an explicit allowlist here first.")


def test_registry_covers_every_extractor_arch_kwarg():
    """The reverse direction: nothing reaches the extractor without a registry row."""
    reached = set(EA.ARCH_ARG_KEYS) | set(EA._DERIVED) | set(EA.FROZEN_ARCH_KWARGS)
    assert reached == {f.name for f in REGISTRY}, (
        f"extractor kwargs without a registry row: {sorted(reached - set(BY_NAME))}; "
        f"registry rows that reach nothing: {sorted(set(BY_NAME) - reached)}")


def test_config_only_defaults_are_what_the_extractor_would_receive():
    """A frozen value must survive `build_extractor_arch_kwargs`, including a stale args attribute.

    The demotion leaves the ATTRIBUTE reachable in some paths (a resumed namespace, a test stub),
    and a frozen toggle that a leftover attribute could override would be frozen in name only.
    """
    import types
    stub = types.SimpleNamespace(**{f.arg: None for f in cli_flags()})
    for f in config_only_flags():                 # the stale-attribute hazard, planted
        setattr(stub, f.arg, "STALE")
    stub.opp_belief_aux_coef = 0.0
    stub.opp_intent_coef = 0.0
    built = EA.build_extractor_arch_kwargs(stub)
    for f in config_only_flags():
        assert built[f.name] == f.default, (
            f"{f.name!r} is tier=config_only frozen at {f.default!r}, but "
            f"build_extractor_arch_kwargs produced {built[f.name]!r} — a leftover args attribute "
            f"is overriding the frozen value.")


def test_arch_toggles_from_args_matches_the_extractor_kwargs():
    """The version gate and the extractor must see ONE toggle set (they were two hand-kept lists)."""
    import types
    stub = types.SimpleNamespace(**{f.arg: None for f in cli_flags()})
    stub.opp_belief_aux_coef = 0.0
    stub.opp_intent_coef = 0.0
    toggles = EA.arch_toggles_from_args(stub)
    built = EA.build_extractor_arch_kwargs(stub)
    assert set(toggles) == {f.name for f in REGISTRY}
    for k, v in toggles.items():
        assert built[k] == v, f"{k!r}: version gate sees {v!r}, the extractor is built with {built[k]!r}"


def test_since_versions_are_within_the_schema_history():
    from agents.model.model_version import MODEL_CONFIG_VERSION
    bad = [f.name for f in REGISTRY if not (1 <= f.since <= MODEL_CONFIG_VERSION)]
    assert not bad, f"registry rows with an out-of-range `since`: {bad}"


def test_resume_immutable_flags_are_excluded_from_check_compatible():
    """The class is not decorative: it decides WHICH gate, and the wrong gate breaks league play.

    A `resume_immutable` toggle has a bit-identical forward, so gating it in `check_compatible`
    (which runs on EVERY load, including frozen eval / pool / distill opponents) would be a false
    rejection. The check is a source scan of `check_compatible` for a `self.<name> !=` compare.
    """
    src = inspect.getsource(ModelVersion.check_compatible)
    for f in REGISTRY:
        compared = re.search(rf"self\.{re.escape(f.name)}\s*!=", src) is not None
        if f.klass is Klass.RESUME_IMMUTABLE:
            assert not compared, (
                f"{f.name!r} is class=resume_immutable but check_compatible compares it. That "
                f"gate runs on every frozen-opponent load, whose forward is identical regardless "
                f"— it would reject the run's own snapshots. Move it to a dedicated check_*.")
        elif f.klass is Klass.STRUCTURAL:
            assert compared, (
                f"{f.name!r} is class=structural but check_compatible does not compare it, so "
                f"nothing rejects a resume that flips it. Add the compare, or reclassify the row.")


def test_generated_doc_is_not_stale():
    """`designs/flag_registry.md` is generated; a drifted table is a failing test, not stale prose."""
    from agents.model import flag_registry as FR
    with open(FR._DOC) as fh:
        current = fh.read()
    for name, want in FR.generate_sections().items():
        assert FR.extract_section(current, name) == want.strip("\n"), (
            f"designs/flag_registry.md section {name!r} is stale — regenerate with:\n"
            f"  python -m agents.model.flag_registry")
