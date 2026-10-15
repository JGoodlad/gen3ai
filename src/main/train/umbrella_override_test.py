"""gen3_umbrella_no_silent_override_v1 (T31, 2026-10-15): an UMBRELLA flag never silently overrides a
sub-flag that already holds a different value.

The incident (`designs/endstate/design_x26_readiness.md` F1 / I23): under the fresh-run default
`--unified-moves both`, a typed `--damage-op false` was turned back ON — with `move_prior_fusion` and
`damage_outgoing` — and `main.checkargs`, which judges the RESOLVED namespace, reported "every ARCH-surface
key matches". The operator's ablation was production with no trace on any surface.

The fix REFUSES the conflict (`config.UmbrellaOverride`, FATAL_CONFIG at launch, the same verdict in
`checkargs`) — never "the umbrella wins", and not "the typed value wins" either (the umbrella's OTHER
writes would then make a combination refused under a message about a different flag). Every test below
fails on revert: the old desugar wrote the umbrella's value over the typed one and raised nothing.
"""
import contextlib
import io

import pytest

from main.train.parser import build_parser


def _desugar(argv):
    from main.train.config import desugar_umbrella_flags
    ns = build_parser().parse_args(["--steps", "1", *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        desugar_umbrella_flags(ns)
    return ns


def _conflicts(argv):
    from main.train.config import UmbrellaOverride
    with pytest.raises(UmbrellaOverride) as ei:
        _desugar(argv)
    return {flag: (held, want, umb) for flag, held, want, umb in ei.value.conflicts}


# ------------------------------------------------------------------------- the refused conflicts
def test_a_typed_no_damage_op_under_the_default_umbrella_is_refused_not_turned_back_on():
    got = _conflicts(["--no-damage-op"])
    assert got == {"--damage-op": (False, True, "--unified-moves both (the fresh-run default)")}


def test_a_typed_no_damage_op_under_arch_production_is_refused():
    """The exact argv of the incident: `--arch production` + one typed ablation flag."""
    got = _conflicts(["--arch", "production", "--no-damage-op"])
    assert set(got) == {"--damage-op"}


def test_every_conflicting_sub_flag_is_named_at_once():
    got = _conflicts(["--unified-moves", "both", "--no-damage-op", "--no-move-prior-fusion",
                      "--no-damage-outgoing"])
    assert set(got) == {"--damage-op", "--move-prior-fusion", "--damage-outgoing"}
    assert {umb for _, _, umb in got.values()} == {"--unified-moves both"}     # typed: no "default" tag


def test_a_typed_damage_outgoing_under_the_incoming_level_is_refused():
    """'incoming' writes damage_outgoing=False — a typed True is the same class."""
    got = _conflicts(["--unified-moves", "incoming", "--damage-outgoing"])
    assert got == {"--damage-outgoing": (True, False, "--unified-moves incoming")}


def test_a_typed_unified_damage_off_under_an_active_unified_moves_is_refused():
    """`--unified-damage` used to default to 'off', so a TYPED 'off' was indistinguishable from the
    default and was silently raised to the --unified-moves level. It now defaults to None."""
    got = _conflicts(["--unified-damage", "off"])
    assert got == {"--unified-damage": ("off", "both", "--unified-moves both (the fresh-run default)")}


def test_a_typed_damage_topk_zero_beside_an_incoming_matrix_is_refused_not_raised_to_five():
    got = _conflicts(["--unified-moves", "off", "--damage-op", "--damage-matrices", "incoming",
                      "--damage-topk", "0"])
    assert set(got) == {"--damage-topk"}


def test_the_refusal_message_names_the_flag_and_the_fix():
    from main.train.config import UmbrellaOverride
    with pytest.raises(UmbrellaOverride, match=r"--damage-op is False.*--unified-moves off"):
        _desugar(["--no-damage-op"])


# ------------------------------------------------------------------------- what still resolves
def test_production_resolves_unchanged():
    ns = _desugar(["--arch", "production"])
    assert (ns.unified_moves, ns.unified_damage) == ("both", "both")
    assert (ns.damage_op, ns.move_prior_fusion, ns.damage_outgoing) == (True, True, True)
    from main.train.arch_surface import load_production_config
    assert ns.damage_topk_k == load_production_config()["damage_topk_k"]


def test_a_sub_flag_typed_to_the_umbrellas_own_value_is_no_conflict():
    ns = _desugar(["--damage-op", "--move-prior-fusion", "--damage-outgoing"])
    assert (ns.damage_op, ns.move_prior_fusion, ns.damage_outgoing) == (True, True, True)


def test_unified_moves_off_lets_the_typed_sub_flags_stand():
    """The toy T-ENT's route (`rb_toy_ent_s4001`): `--unified-moves off` + the typed OFF values."""
    ns = _desugar(["--unified-moves", "off", "--no-damage-op", "--no-damage-outgoing",
                   "--no-move-prior-fusion", "--damage-matrices", "off", "--damage-topk", "0"])
    assert ns.unified_damage == "off"          # the resolved value every reader sees (was the default)
    assert (ns.damage_op, ns.move_prior_fusion, ns.damage_outgoing, ns.damage_topk_k) == (False, False, False, 0)


def test_a_typed_non_off_unified_damage_level_still_wins():
    ns = _desugar(["--unified-moves", "both", "--unified-damage", "incoming"])
    assert ns.unified_damage == "incoming" and ns.damage_outgoing is False


def test_a_resume_never_desugars_so_a_recorded_off_op_is_inherited_not_refused():
    ns = _desugar(["--model", "models/x/final_model.zip"])
    assert ns.unified_moves == "off" and ns.unified_damage == "off" and ns.damage_op is None


# ------------------------------------------------------------------------- checkargs reports it
def test_checkargs_reports_the_refusal_instead_of_every_key_matching():
    from main import checkargs
    res = checkargs.check(["--arch", "production", "--steps", "1000", "--no-damage-op"])
    assert res["umbrella_refusal"] and "--damage-op is False" in res["umbrella_refusal"]
    assert res["arch"] is None                 # no ARCH report on a namespace the launch never builds


def test_checkargs_exits_fatal_config_on_the_refusal():
    from main import checkargs
    from main.exit_codes import TrainExitCode
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = checkargs.main(["--argv", "--arch production --steps 1000 --no-damage-op"])
    assert rc == int(TrainExitCode.FATAL_CONFIG)
    assert "REFUSED at launch (FATAL_CONFIG)" in out.getvalue()
    assert "every ARCH-surface key matches" not in out.getvalue()
