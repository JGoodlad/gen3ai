"""THE ARCH SURFACE — "is this the architecture you MEANT?", on all three surfaces at once.

WHY THIS FILE EXISTS (2026-09-06, 25,131 s = ~7 GPU-hours). The first win-prob-critic arm was
launched from
the 37 forwarded tokens of a design document's command block — the critic flags and the
hyperparameters, none of the production feature flags. `python -m main.checkargs` said "still
launches"; `python -m main.launcher --dry-run` said "would launch"; both were right, and the run
trained a near-bare network for 24.4M steps and six checkpoints — it
was still holding the GPU when it was discovered a SECOND time. The drift gate that WOULD have caught it
(`arch_tables_test::test_production_config_matches_newest_run`) fires only when someone runs the
suite, which happened hours after the GPU started.

The central test here is therefore §1: the REAL recorded argv, read off
`models/ai_v12_01_winprob_critic/metadata.json` when the archive is on this box, must be REFUSED
and must name at least twenty keys. Everything else guards the ways that refusal could be lost —
an umbrella that does not close it, a consent flag that is not recorded, a fork that is wrongly
gated, and two surfaces that stop agreeing.

Run: python -m pytest src/main/train/arch_surface_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import contextlib
import io
import json
import os
import shlex

import pytest

from agents.model.flag_registry import BY_NAME, Klass, Tier, arch_surface_flags
from main.train import arch_surface


# ------------------------------------------------------------------------------- THE INCIDENT ARGV
#: The 2026-09-06 arm, reconstructed from its own `metadata.json` when this box has the archive and
#: falling back to this verbatim copy otherwise. The fallback is the string that WAS recorded — a
#: literal, so the test means the same thing on a machine with no `models/`.
INCIDENT_ARGV_FALLBACK = (
    "--run-name ai_v12_01_winprob_critic --restart-interval-hours 3 --pin-commit e798c13a "
    "--steps 75000000 --n-envs 64 --batch-size 4096 --grad-accum-steps 16 --n-epochs 10 "
    "--n-steps 2048 --lr 0.0003 --ent-coef 0.02 --device cuda --log-level periodic "
    "--critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0 "
    "--draw-penalty 0 --vf-coef 0.5 --self-play"
)

INCIDENT_RUN = "ai_v12_01_winprob_critic"


def _incident_argv() -> list:
    """The incident's own recorded `original_command`, minus the program name — else the copy."""
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is not None:
        meta = os.path.join(str(root), INCIDENT_RUN, "metadata.json")
        if os.path.isfile(meta):
            try:
                with open(meta) as fh:
                    cmd = json.load(fh).get("original_command")
                if cmd:
                    # argv[0] is the launcher's __main__.py; the rest is the command.
                    return shlex.split(cmd)[1:]
            except Exception:                            # noqa: BLE001 — fall back to the literal
                pass
    return shlex.split(INCIDENT_ARGV_FALLBACK)


def _namespace(argv):
    """The argv as EVERY surface sees it: parsed, marked explicit, critic-resolved, desugared.

    Exactly `combination_checks_test._namespace`'s order, and for its reason — the launch path,
    `main.checkargs` and `--dry-run` all build the namespace this way, and a test that skipped one
    step would judge a config nobody executes. `desugar_umbrella_flags` is also where `--arch
    production` is applied, so this function covers the umbrella too.
    """
    from main.train.config import desugar_umbrella_flags, resolve_critic_mode
    from main.train_rl_agent import build_parser
    parser = build_parser()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        ns, _rest = parser.parse_known_args(argv)
        ns._explicit_flags = frozenset(d for d, v in vars(ns).items() if v is not None)
        ns._saved_config_present = False
        resolve_critic_mode(ns, None)
        desugar_umbrella_flags(ns)
    return ns


def _report(argv, *, fresh=True):
    ns = _namespace(argv)
    return ns, arch_surface.report(
        ns,
        fresh=fresh,
        allowed=bool(getattr(ns, "allow_nonproduction_arch", False)),
        umbrella=getattr(ns, "arch", None),
    )


# =============================================================================================
# 1. THE INCIDENT — the argv that cost the GPU-hours is REFUSED, naming what it left out
# =============================================================================================

def test_the_incident_argv_is_refused_naming_at_least_twenty_keys():
    argv = _incident_argv()
    _ns, rep = _report(argv)
    assert rep.refuses, (
        "the 2026-09-06 argv must be REFUSED — it is the whole reason this guard exists.\n"
        + "\n".join(arch_surface.report_lines(rep)))
    assert len(rep.diffs) >= 20, (
        f"only {len(rep.diffs)} key(s) reported; the incident's run differs from the production "
        f"mirror on far more than that: {[d.name for d in rep.diffs]}")


def test_the_incident_refusal_names_the_families_the_run_actually_lost():
    """Not merely 'twenty keys' — the SPECIFIC families the owner noticed were missing.

    `belief/aux_loss` absent from TensorBoard is what surfaced the incident, so the belief slots
    must be in the list; the edge families, entity seats, event window, intent heads and value
    pools are the other five the post-mortem enumerated.
    """
    _ns, rep = _report(_incident_argv())
    named = {d.name for d in rep.diffs}
    for key in ("opp_belief_slots", "edge_bias_families", "entity_topk_seats", "history_events",
                "opp_intent", "value_entity_pool"):
        assert key in named, f"{key} is not reported, but the incident's run had it off"


def test_the_refusal_message_says_how_to_fix_it():
    _ns, rep = _report(_incident_argv())
    text = "\n".join(arch_surface.report_lines(rep))
    assert "--arch production" in text
    assert arch_surface.ALLOW_FLAG in text
    assert "2026-09-06" in text, "the message must cite the incident that motivated it"


def test_every_reported_diff_carries_BOTH_values():
    """A guard that says 'these keys differ' without saying HOW is a guard nobody can act on."""
    prod = arch_surface.load_production_config()
    _ns, rep = _report(_incident_argv())
    for d in rep.diffs:
        assert d.production == prod[d.name]
        assert d.resolved != d.production
        assert repr(d.resolved) in d.line() and repr(d.production) in d.line()


# =============================================================================================
# 2. THE UMBRELLA — `--arch production` closes it, leaving only the critic family
# =============================================================================================

def test_the_incident_argv_plus_arch_production_passes():
    argv = _incident_argv() + ["--arch", "production"]
    ns, rep = _report(argv)
    assert not rep.refuses, "\n".join(arch_surface.report_lines(rep))
    assert not rep.diffs, [d.line() for d in rep.diffs]
    assert ns.arch == "production"


def test_what_the_umbrella_leaves_open_is_EXACTLY_critic_readouts_and_doses():
    """Everything `--arch production` does not set must be a CRITIC readout, a resume-immutable
    value, or a declared SUPERVISION DOSE — never a structural toggle it silently skipped."""
    from agents.model.flag_registry import Family
    doses = {f"--{f.coef_arg.replace('_', '-')}" for f in arch_surface.arch_surface_flags()
             if f.coef_arg}
    for flag, _value in arch_surface.unapplied_production_keys():
        if flag in doses:
            continue                     # a training dose, declared by `ModelFlag.coef_arg`
        name = next(f.name for f in BY_NAME.values() if f.cli_flag == flag)
        row = BY_NAME[name]
        assert row.family is Family.CRITIC or row.klass in (
            Klass.TRAINING_COEF, Klass.RESUME_IMMUTABLE), (
            f"{name} is skipped by the umbrella but is neither critic-family nor a "
            f"training/resume-immutable value — it would be a silent architecture hole")


def test_the_umbrella_NAMES_the_supervision_doses_it_does_not_set():
    """The 2026-09-06 failure, one layer down: `--arch production` builds the production network
    with `move_belief_coef` / `spread_belief_coef` at their 0.0 fresh defaults where production
    trains them at 0.05. The umbrella may leave them alone — a dose is training, not architecture —
    but it must never leave them UNSAID, or its own block reads as coverage."""
    named = {flag for flag, _ in arch_surface.unapplied_production_keys()}
    for flag in ("--move-belief-coef", "--spread-belief-coef", "--item-belief-coef",
                 "--hp-type-belief-coef", "--move-belief-latent-coef"):
        assert flag in named, f"{flag} is a production dose the umbrella neither sets nor names"

    rep = _report(_incident_argv() + ["--arch", "production"])[1]
    text = "\n".join(arch_surface.report_lines(rep))
    assert "--move-belief-coef 0.05" in text, text


def test_the_umbrella_does_not_fight_the_winprob_critic():
    """`--critic winprob` REFUSES `--value-dist-mode` and `--value-from-dist`, both of which the
    production mirror has ON. If the umbrella applied them it would turn every critic arm into a
    launch refusal — the reason the critic family is excluded by declaration."""
    from main.train.combination_checks import failing_checks
    ns = _namespace(_incident_argv() + ["--arch", "production"])
    broken = [c.name for c in failing_checks(ns)]
    assert not broken, f"--arch production made a winprob command incoherent: {broken}"


def test_the_umbrella_records_its_source_by_CONTENT_hash():
    from main.train.config import desugar_umbrella_flags
    ns = _namespace(["--steps", "100", "--arch", "production"])
    tag = getattr(ns, "arch_source", None)
    assert tag and tag.startswith("production_config@"), tag
    assert tag == arch_surface.arch_source_tag()
    # A content hash, not a commit: it must change when the FILE changes and not otherwise.
    assert arch_surface.production_blob_sha() in tag.split("@", 1)[1] or \
        tag.split("@", 1)[1] == arch_surface.production_blob_sha()[:12]
    del desugar_umbrella_flags


def _raw_namespace(argv):
    """Parsed and NOTHING else — no desugars. `--unified-moves` defaults to 'both' on a fresh run
    and sets `damage_op` / `move_latent` / `damage_topk_k` itself, so a desugared namespace would
    make the umbrella look like it applied fewer keys than it covers."""
    from main.train_rl_agent import build_parser
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        ns, _rest = build_parser().parse_known_args(argv)
    return ns


def test_the_umbrella_applies_every_surface_key_the_mirror_carries():
    """The count is the claim `--arch production` makes; a silently-shrinking one is the failure."""
    ns = _raw_namespace(["--steps", "100"])
    applied = dict(arch_surface.apply_production_arch(ns))
    covered = set(arch_surface.production_surface_keys())
    # Every surface key production carries is either applied here, or was already non-None on a
    # bare namespace (a `config_only` row, which has no argparse dest at all).
    config_only = {f.name for f in arch_surface_flags() if f.tier is not Tier.CLI}
    assert covered - set(applied) <= config_only, sorted(covered - set(applied) - config_only)
    assert len(applied) >= 30, f"only {len(applied)} applied — the mirror carries {len(covered)}"


# =============================================================================================
# 3. PRECEDENCE — an explicit flag beats the umbrella, and is then reported
# =============================================================================================

def test_an_explicit_zero_beats_the_umbrella_and_is_then_refused():
    """`--arch production --entity-topk-seats 0`: the operator's value wins (it is a default, not
    a lock) AND the guard reports the resulting drift, so an ablation is still visible."""
    argv = _incident_argv() + ["--arch", "production", "--entity-topk-seats", "0"]
    ns, rep = _report(argv)
    assert ns.entity_topk_seats == 0, "the umbrella overwrote an explicitly typed value"
    names = {d.name for d in rep.diffs}
    assert "entity_topk_seats" in names, "the explicit override was not reported as drift"
    assert rep.refuses, "an explicit non-production value must still be gated"


def test_that_same_command_passes_with_the_consent_flag():
    argv = (_incident_argv()
            + ["--arch", "production", "--entity-topk-seats", "0",
               arch_surface.ALLOW_FLAG])
    ns, rep = _report(argv)
    assert rep.allowed and not rep.refuses
    assert {d.name for d in rep.diffs} == {"entity_topk_seats"}
    del ns


def test_the_umbrella_leaves_a_typed_edge_family_alone():
    argv = ["--steps", "100", "--arch", "production", "--edge-bias-families", "h"]
    ns, _rep = _report(argv)
    assert ns.edge_bias_families == "h"


# =============================================================================================
# 4. CONSENT — `--allow-nonproduction-arch` passes AND is recorded
# =============================================================================================

def test_the_consent_flag_passes_the_incident_argv():
    argv = _incident_argv() + [arch_surface.ALLOW_FLAG]
    _ns, rep = _report(argv)
    assert rep.diffs, "the diff must still be COMPUTED and printed — consent is not blindness"
    assert not rep.refuses
    text = "\n".join(arch_surface.report_lines(rep))
    assert "EXPLICIT choice" in text


def test_the_consent_is_a_recorded_field_on_ModelVersion():
    """`we meant it` has to survive on disk, or the next reader is in the incident's position."""
    import dataclasses

    from agents.model.model_version import ModelVersion
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    assert "arch_source" in fields


def test_arch_source_is_provenance_only_and_gates_nothing():
    """It must never join `check_compatible` — a provenance string that can refuse a load would
    make an ablation's checkpoint unloadable by a tool that did not type the flag."""
    from agents.model.model_version import compat
    import inspect
    src = inspect.getsource(compat)
    assert "arch_source" not in src, (
        "arch_source appears in the compatibility gate — it is PROVENANCE and gates nothing")


def test_a_pre_v111_config_migrates_with_arch_source_absent_meaning_unknown():
    from agents.model.model_version.constants import ARCH_SIGNATURE, MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    data = {"config_version": 109, "arch_signature": ARCH_SIGNATURE}
    out = _migrate_config(dict(data))
    assert out["config_version"] == MODEL_CONFIG_VERSION
    assert out["arch_source"] is None, "a pre-v111 run must read 'not recorded', never a guess"


# =============================================================================================
# 5. A FORK IS EXEMPT — but still prints the diff
# =============================================================================================

def test_a_fork_is_not_gated_but_the_diff_is_still_printed():
    _ns, rep = _report(_incident_argv(), fresh=False)
    assert rep.diffs, "the comparison must still run on a fork"
    assert not rep.refuses, "a fork INHERITS its parent's surface — it must not be gated"
    text = "\n".join(arch_surface.report_lines(rep))
    assert "INFO only" in text and "FORK/RESTART" in text
    assert "REFUSED" not in text


def test_the_umbrella_itself_is_refused_on_a_resume():
    """A fork that APPLIED production would replace inheritance with a mirror its parent may never
    have matched — a check_compatible FATAL at best."""
    from main.train.combination_checks import failing_checks
    ns = _namespace(["--steps", "100", "--arch", "production",
                     "--model", "models/p/final_model.zip"])
    assert "arch_umbrella_is_fresh_only" in [c.name for c in failing_checks(ns)]


# =============================================================================================
# 6. THE KEY SET IS DERIVED — never a hand list
# =============================================================================================

def test_the_surface_is_derived_from_the_registry_and_covers_the_mirror():
    prod = arch_surface.load_production_config()
    surface = arch_surface_flags()
    assert surface, "the surface is empty — the registry query is broken"
    missing = [f.name for f in surface if f.name not in prod]
    assert not missing, (
        f"designs/production_config.json does not record {missing}. Either the mirror is stale or "
        f"a new structural toggle is not written to model_config.json — both are real defects.")


def test_no_hand_written_key_list_exists_in_the_module():
    """The one property that keeps this guard from rotting: a literal key list would go stale the
    first time a toggle landed, and the guard would silently under-report."""
    import inspect
    src = inspect.getsource(arch_surface)
    body = src.split('"""', 2)[-1]              # drop the module docstring's prose
    for name in ("edge_bias_families", "entity_topk_seats", "history_events"):
        assert f'"{name}"' not in body, (
            f"{name} is written as a literal in arch_surface.py — the key set must come from "
            f"flag_registry.arch_surface_flags(), never a hand list")


def test_every_excluded_class_is_excluded_by_its_own_declaration():
    from agents.model.flag_registry import REGISTRY, Family
    surface = {f.name for f in arch_surface_flags()}
    for f in REGISTRY:
        if f.name in surface:
            assert f.klass is Klass.STRUCTURAL and f.family is Family.ARCH
        elif f.klass is Klass.STRUCTURAL and f.tier is not Tier.CONSTRUCTOR_ONLY:
            assert f.family is Family.CRITIC, (
                f"{f.name} is structural and NOT on the arch surface, but is not declared "
                f"family=CRITIC — the exclusion must be a declaration, not an accident")


# =============================================================================================
# 7. THE THREE SURFACES AGREE — one function, three readers
# =============================================================================================

def _checkargs_verdict(argv):
    from main.checkargs import check
    return check(list(argv)).get("arch")


@pytest.mark.parametrize("case,argv_extra,should_refuse", [
    ("incident", [], True),
    ("umbrella", ["--arch", "production"], False),
    ("consent", [arch_surface.ALLOW_FLAG], False),
])
def test_checkargs_and_the_direct_report_agree(case, argv_extra, should_refuse):
    argv = _incident_argv() + argv_extra
    direct = _report(argv)[1]
    via_checkargs = _checkargs_verdict(argv)
    assert via_checkargs is not None, f"{case}: checkargs computed no arch report"
    assert via_checkargs.refuses == direct.refuses == should_refuse, case
    assert {d.name for d in via_checkargs.diffs} == {d.name for d in direct.diffs}, case


def test_checkargs_exits_nonzero_on_the_incident_argv(capsys):
    from main.checkargs import main as checkargs_main
    argv = [t for t in _incident_argv() if t not in ("--pin-commit", "e798c13a")]
    rc = checkargs_main(["--argv", " ".join(argv)])
    out = capsys.readouterr().out
    assert rc != 0, "checkargs said the incident argv still launches"
    assert "ARCH SURFACE" in out
    assert "this command still launches" not in out


def test_checkargs_exits_zero_with_the_umbrella(capsys):
    """WITHOUT the incident's `--pin-commit`: `--arch` does not exist at e798c13a, so a pinned
    check correctly refuses it (the pinned-parser rule, and a real constraint — see the runbook).
    The question here is whether the ARCH SURFACE half passes."""
    from main.checkargs import main as checkargs_main
    argv = [t for t in _incident_argv() if t not in ("--pin-commit", "e798c13a")]
    rc = checkargs_main(["--argv", " ".join(argv + ["--arch", "production"])])
    out = capsys.readouterr().out
    assert rc == 0, out[-2000:]
    assert "every ARCH-surface key matches" in out


def test_a_PINNED_argv_is_advisory_and_a_FRESH_one_REFUSES(capsys):
    """THE ASYMMETRY, on the surface an operator actually runs.

    `designs/production_config.json` is THIS tree's mirror. A run pinned to another commit is built
    by THAT commit's registry, its flags and its own mirror — and `--arch` does not exist before
    2026-09-06, so the remedy the refusal offers is not even available there. Refusing it would be
    the same false POSITIVE `gen3_pinned_argv_parser_v1` fixed for the parser. So a pinned argv is
    ADVISORY (printed, never gated) and an un-pinned FRESH one at HEAD REFUSES.

    The diff is computed and printed on BOTH — dropping it on the pinned path is the other half of
    that lesson, and is what would make the guard silently stop working the day a batch pins.
    """
    from main.checkargs import main as checkargs_main
    bare = [t for t in _incident_argv() if t not in ("--pin-commit", "e798c13a")]

    rc_fresh = checkargs_main(["--argv", " ".join(bare)])
    fresh_out = capsys.readouterr().out
    assert rc_fresh != 0 and "✗ REFUSED" in fresh_out

    rc_pinned = checkargs_main(["--argv", " ".join(bare), "--pin", "e798c13a"])
    pinned_out = capsys.readouterr().out
    assert rc_pinned == 0, "the arch surface GATED a pinned argv — it must only report there"
    assert "ARCH SURFACE" in pinned_out, "the pinned path DROPPED the diff instead of demoting it"
    assert "ADVISORY" in pinned_out
    assert "✗ REFUSED: a FRESH run" not in pinned_out
    assert "builds the wrong architecture" not in pinned_out


def test_the_arch_verdict_never_shares_a_line_with_a_combination_refusal(capsys):
    """A refused flag COMBINATION is loud and PRE-launch; arch drift is silent and POST-launch. The
    2026-09-06 relaunch hit both at once (nine flags the win-prob critic subsumes, on top of the
    stripped surface), so the two must read as two findings — separate blocks, separate closing
    lines — or a reader who fixes the loud one believes they have fixed the silent one."""
    from main.checkargs import main as checkargs_main
    argv = ("--steps 100 --critic winprob --no-hand-shaping --terminal-indicator "
            "--victory-value 1.0 --draw-penalty 0 --use-popart --value-from-dist")
    rc = checkargs_main(["--argv", argv])
    out = capsys.readouterr().out
    assert rc != 0
    assert "refused combinations" in out and "ARCH SURFACE" in out
    # neither subsumed flag may appear as arch drift...
    arch_block = out.split("ARCH SURFACE", 1)[1]
    for subsumed in ("use_popart", "value_from_dist", "win_prob_coef", "value_tail_weight"):
        assert subsumed not in arch_block, f"{subsumed} reported as arch drift"
    # ...and the arch-only closing line must NOT fire when a combination also failed, or the two
    # verdicts would blur into one.
    assert "builds the wrong architecture" not in out


def test_dry_run_and_checkargs_reach_the_same_verdict(monkeypatch, tmp_path, capsys):
    """The EXECUTING surface and the offline one must not disagree — that split is the whole
    failure mode (`checkargs` clean, the launch dead; here it would be `--dry-run` clean and the
    launcher refusing three seconds later)."""
    import main.launcher.dry_run as dry_run_mod
    monkeypatch.chdir(tmp_path)
    argv = _incident_argv()
    # Strip the launcher-owned flags the launcher parses out before forwarding.
    child_args = [t for t in argv if t not in ("--restart-interval-hours", "3",
                                               "--pin-commit", "e798c13a")]
    rc = dry_run_mod.dry_run(
        child_args, interval_hours=0, pin=False, sync_to_main=False, pin_commit=None,
        grace_minutes=0, max_crash_restarts=0, nice=0, out=lambda _s: None)
    from main.exit_codes import TrainExitCode
    assert rc == int(TrainExitCode.FATAL_CONFIG), "--dry-run accepted the incident argv"
    assert _checkargs_verdict(child_args).refuses
    capsys.readouterr()


def test_dry_run_accepts_the_umbrella(monkeypatch, tmp_path):
    import main.launcher.dry_run as dry_run_mod
    monkeypatch.chdir(tmp_path)
    child_args = [t for t in _incident_argv()
                  if t not in ("--restart-interval-hours", "3", "--pin-commit", "e798c13a")]
    rc = dry_run_mod.dry_run(
        child_args + ["--arch", "production"], interval_hours=0, pin=False, sync_to_main=False,
        pin_commit=None, grace_minutes=0, max_crash_restarts=0, nice=0, out=lambda _s: None)
    assert rc == 0


def test_dry_run_prints_the_applied_keys_under_the_umbrella(monkeypatch, tmp_path):
    import main.launcher.dry_run as dry_run_mod
    monkeypatch.chdir(tmp_path)
    lines = []
    dry_run_mod.dry_run(
        ["--steps", "100", "--arch", "production"], interval_hours=0, pin=False,
        sync_to_main=False, pin_commit=None, grace_minutes=0, max_crash_restarts=0, nice=0,
        out=lines.append)
    text = "\n".join(lines)
    assert "--arch production" in text
    assert "ARCH-surface key(s) applied" in text


# =============================================================================================
# 8. THE MIRROR ITSELF
# =============================================================================================

def test_the_blob_sha_matches_gits_own_hash_of_the_file():
    """`production_blob_sha` computes git's blob hash WITHOUT git; if the two ever disagreed, the
    recorded `arch_source` would name a content nobody could look up."""
    import subprocess
    from utils.paths import repo_root
    try:
        out = subprocess.run(
            ["git", "hash-object", arch_surface.PRODUCTION_CONFIG_PATH],
            cwd=str(repo_root()), capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001 — no git here
        pytest.skip("git unavailable")
    assert out == arch_surface.production_blob_sha()


def test_a_clean_production_argv_reports_a_match_and_says_how_many_keys():
    ns = _namespace(["--steps", "100", "--arch", "production"])
    rep = arch_surface.report(ns, fresh=True, umbrella="production")
    text = "\n".join(arch_surface.report_lines(rep))
    assert "✓ every ARCH-surface key matches" in text
    part = arch_surface.surface_partition()
    assert f"{part['arch']} of {part['total']} registry toggles" in text


# =============================================================================================
# 9. THE COMPARED-KEY COUNT IS RECONCILED, not merely smaller
# =============================================================================================

def test_the_surface_partition_is_exhaustive_and_printed():
    """A guard that compares FEWER keys than a reader's own count is the failure mode that
    matters — the reader cannot tell an excluded row from a forgotten one. So the difference is
    arithmetic (39 arch + 7 critic + 3 non-structural = 49 registry rows on 2026-09-06) and every
    printed block carries it. A row that lands without a `family` decision breaks this identity
    rather than quietly moving the surface."""
    from agents.model.flag_registry import REGISTRY
    part = arch_surface.surface_partition()
    assert part["arch"] + part["critic"] + part["non_structural"] == part["total"] == len(REGISTRY)
    assert part["arch"] == len(arch_surface_flags())
    rep = arch_surface.report(_namespace(_incident_argv()), fresh=True)
    text = "\n".join(arch_surface.report_lines(rep))
    assert f"{part['critic']} critic readouts" in text
    assert f"{part['non_structural']} non-structural rows" in text


# =============================================================================================
# 10. THE RECORDED CONFIG — the guard reads a model_config.json, not only an argv
# =============================================================================================
#
# The hand-rolled check that validated the corrected relaunch compared the RECORDED config, not the
# argv, and the two namespace SHAPES differ for a derived row: an argv carries `opp_intent_coef`
# (there is no `--opp-intent` flag) while `model_config.json` carries `opp_intent`. Reading only the
# coefficient made the guard report `opp_intent False` against a config whose recorded value is
# `True` — a false finding on the LIVE run. Both shapes are read now, and these two tests are the
# reason. They need `models/`, so they SKIP without it (never a hardcoded /home path).

def _recorded_config(run: str):
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is None:
        pytest.skip("no models/ archive in this checkout")
    path = os.path.join(str(root), run, "model_config.json")
    if not os.path.isfile(path):
        pytest.skip(f"{run} is not in this box's archive")
    import types
    with open(path) as fh:
        return types.SimpleNamespace(**json.load(fh))


#: The ARCH-surface keys the 2026-09-06 arm actually lost, read off its own recorded config on
#: 2026-09-06. The literal is the point: a fixture that recomputed the answer would pass against a
#: guard that had stopped looking.
INCIDENT_LOST_KEYS = frozenset({
    "opp_belief_cls_k", "opp_belief_slots", "move_belief_mode", "spread_belief",
    "damage_matrices_outgoing", "spread_belief_nature", "entity_topk_seats", "edge_bias_families",
    "entity_tail_seats", "value_threat_inject", "opp_intent", "species_prior_fusion",
    "t0_species_prior", "intent_move_cell", "value_entity_pool_full", "history_events",
    "value_entity_pool", "item_belief", "intent_threshold", "intent_conditional",
    "pair_outcome_cell", "pair_outcome_switch", "switch_branch_cell", "conditional_threat_cell",
    "op_drop_renders", "op_believed_lean",
})


def test_the_incident_RECORDED_config_names_exactly_the_keys_it_lost():
    ns = _recorded_config(INCIDENT_RUN)
    got = {d.name for d in arch_surface.diff_against_production(ns)}
    assert got == INCIDENT_LOST_KEYS, (
        f"missing from the finding: {sorted(INCIDENT_LOST_KEYS - got)}; "
        f"unexpected: {sorted(got - INCIDENT_LOST_KEYS)}")


def test_the_CORRECTED_relaunch_reads_clean_against_the_mirror():
    """`ai_v12_02_winprob_critic` is the relaunch that carries the production surface. Its recorded
    config must show ZERO arch-surface drift — the same verdict the hand-rolled check reached over
    all 49 registry rows, from a guard that compares the 39 it declares."""
    ns = _recorded_config("ai_v12_02_winprob_critic")
    diffs = arch_surface.diff_against_production(ns)
    assert not diffs, [d.line() for d in diffs]


def test_every_key_the_incident_lost_is_either_REFUSED_or_NAMED():
    """The completeness claim, against the ledger's own count.

    `2026-09-06 · INCIDENT` records **31 keys differ, 29 of them architecture** between
    `ai_v12_01_winprob_critic` and the mirror. This asserts that NOT ONE of those 31 can pass
    unmentioned: each is either on the ARCH surface (refused, with both values), or the
    `--arch` block NAMES it as a dose it does not set, or it is the enable COEFFICIENT of a
    surface row that is itself refused (`opp_belief_aux_coef` ⇒ `opp_belief_slots`).

    Silence is the failure mode, not a wrong verdict: the run's whole defect was that every
    validator it passed had nothing to say about this class.
    """
    import types
    ns = _recorded_config(INCIDENT_RUN)
    prod = arch_surface.load_production_config()
    cfg = vars(ns)
    differing = {k for k in prod if k in cfg and cfg[k] != prod[k]}
    assert len(differing) == 31, sorted(differing)      # the ledger's count, re-measured

    refused = {d.name for d in arch_surface.diff_against_production(types.SimpleNamespace(**cfg))}
    named = {f.lstrip("-").replace("-", "_")
             for f, _ in arch_surface.unapplied_production_keys(prod)}
    enable_args = {f.arg for f in arch_surface_flags() if f.derived and f.name in refused}
    unmentioned = differing - refused - named - enable_args
    assert not unmentioned, (
        f"{sorted(unmentioned)} differ from production on the arm that burned ~7 GPU-hours and "
        f"NOTHING would tell the operator — the guard must refuse it or name it")
    assert len(refused) == 26, sorted(refused)


def test_a_derived_row_is_read_from_a_recorded_bool_as_well_as_a_coefficient():
    import types
    from agents.model.flag_registry import BY_NAME as _BY
    row = _BY["opp_intent"]
    assert row.derived and row.arg == "opp_intent_coef"
    # the RECORDED shape: the bool, no coefficient anywhere
    val, src = arch_surface.resolved_value(row, types.SimpleNamespace(opp_intent=True))
    assert (val, src) == (True, "recorded")
    # the ARGV shape: the coefficient, no bool
    val, src = arch_surface.resolved_value(row, types.SimpleNamespace(opp_intent_coef=0.05))
    assert (val, src) == (True, "argv")
    # neither: the registry default, which is what a fresh run builds
    val, src = arch_surface.resolved_value(row, types.SimpleNamespace())
    assert (val, src) == (False, "default")
