"""The launcher must not have an opinion about the compile flags — it must FORWARD them.

Same shape as `default_port_test.py`, and for the same reason: two surfaces can disagree about a
default, and when they do nothing fails — the run just silently gets the other one's answer. The
port test catches a drift in a launcher-INJECTED default; this one catches the opposite failure,
a launcher that quietly eats a child flag.

Two ways that happens, both pinned below:

1. `_strip_launcher_args` removes launcher-owned flags by exact match. A future `--no-compile-*`
   entry in that list would drop the owner's fallback on the floor.
2. argparse abbreviation-matches an unknown token against the parser's KNOWN options, and the
   launcher parses with `parse_known_args`. `--no-compile-opponents` sitting next to the
   launcher's own `--no-pin` is exactly the neighbourhood where that bites.

The launcher owns NO compile default and must not acquire one: `--compile-opponents` /
`--compile-opponents-preload` / `--compile-trainer` are defaulted once, in `train_rl_agent`'s
parser, and the launcher's job is to be transparent to them.
"""

from main.launcher.checkpoint import _strip_launcher_args
from main.launcher.run import build_launcher_parser

_COMPILE_FLAGS = [
    "--compile-opponents", "--no-compile-opponents",
    "--compile-opponents-preload", "--no-compile-opponents-preload",
    "--compile-opponents-strict",
    "--compile-trainer", "--no-compile-trainer",
]


def test_strip_launcher_args_forwards_every_compile_flag():
    argv = ["--restart-interval-hours", "3"] + _COMPILE_FLAGS + ["--steps", "100"]
    out = _strip_launcher_args(argv)
    for flag in _COMPILE_FLAGS:
        assert flag in out, f"{flag} was stripped and never reached the child"
    assert "--restart-interval-hours" not in out       # the launcher-owned one still goes


def test_the_launcher_parser_claims_no_compile_flag():
    """Interrogates the REAL parser, not a copy — the point is to catch a launcher flag added
    later whose name abbreviation-collides with one of these."""
    parser = build_launcher_parser()
    for flag in _COMPILE_FLAGS:
        known, unknown = parser.parse_known_args([flag, "--steps", "100"])
        assert flag in unknown, (
            f"the launcher parser consumed {flag} — it belongs to train_rl_agent, and a swallowed "
            "flag is a flag that appears to do nothing")


def test_the_launcher_owns_no_compile_default():
    """A launcher-side default would be a second source of truth for a value train_rl_agent
    already defaults — the `child_uses_bridge` drift class, one flag over."""
    ns, _ = build_launcher_parser().parse_known_args(["--steps", "100"])
    for name in ("compile_opponents", "compile_trainer", "compile_opponents_preload"):
        assert not hasattr(ns, name), (
            f"the launcher parser now defines {name}; the trainer's parser is the single source "
            "of truth for it")


def test_compile_flags_survive_a_full_launcher_style_argv():
    """End to end over a realistic production command: launcher flags out, everything else through
    in order."""
    argv = [
        "--restart-interval-hours", "3", "--nice", "10", "--no-pin",
        "--steps", "25000000", "--n-envs", "48", "--device", "cuda",
        "--no-compile-trainer", "--compile-opponents-strict",
    ]
    out = _strip_launcher_args(argv)
    assert out == ["--steps", "25000000", "--n-envs", "48", "--device", "cuda",
                   "--no-compile-trainer", "--compile-opponents-strict"]
