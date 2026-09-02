"""The launcher must FORWARD the fork pool-seed flags, not have an opinion about them.

Same shape as `compile_flag_forwarding_test.py`, and the stake is higher: these two flags are what
stand between a launcher-managed fold and the empty-pool footgun that cost the 2026-08-18 TD-aux
cell and nearly cost the 2026-09-02 dose cell. A `--no-fork-pool-seed` the launcher swallowed would
re-seed a run its operator deliberately told not to; an `--allow-empty-pool` it swallowed would turn
every restart of a consenting run into a `FATAL_CONFIG` loop.

Note the argparse-abbreviation half is not hypothetical here: `--no-fork-pool-seed` sits in the same
`--no-*` neighbourhood as the launcher's own `--no-pin`.
"""

from main.launcher.checkpoint import _strip_launcher_args
from main.launcher.run import build_launcher_parser

_POOL_SEED_FLAGS = [
    "--fork-pool-seed", "--no-fork-pool-seed",
    "--allow-empty-pool", "--no-allow-empty-pool",
]


def test_strip_launcher_args_forwards_every_pool_seed_flag():
    argv = ["--restart-interval-hours", "3"] + _POOL_SEED_FLAGS + ["--steps", "100"]
    out = _strip_launcher_args(argv)
    for flag in _POOL_SEED_FLAGS:
        assert flag in out, f"{flag} was stripped and never reached the child"
    assert "--restart-interval-hours" not in out       # the launcher-owned one still goes


def test_the_launcher_parser_claims_no_pool_seed_flag():
    """Interrogates the REAL parser — the point is to catch a launcher flag added later whose
    name abbreviation-collides with one of these (`--no-pin` is the near neighbour)."""
    parser = build_launcher_parser()
    for flag in _POOL_SEED_FLAGS:
        _known, unknown = parser.parse_known_args([flag, "--steps", "100"])
        assert flag in unknown, (
            f"the launcher parser consumed {flag} — it belongs to train_rl_agent, and a swallowed "
            "flag is a flag that appears to do nothing")


def test_the_launcher_owns_no_pool_seed_default():
    """A launcher-side default would be a second source of truth for a value train_rl_agent
    already defaults — the `child_uses_bridge` drift class, two flags over."""
    ns, _ = build_launcher_parser().parse_known_args(["--steps", "100"])
    for name in ("fork_pool_seed", "allow_empty_pool"):
        assert not hasattr(ns, name), (
            f"the launcher parser now defines {name}; the trainer's parser is the single source "
            "of truth for it")


def test_the_trainer_parser_defaults_are_seed_on_and_consent_off():
    """The defaults ARE the guard: auto-seed on, and no implicit consent to the bot fallback."""
    from main.train_rl_agent import build_parser
    ns = build_parser().parse_args(["--steps", "1"])
    assert ns.fork_pool_seed is True
    assert ns.allow_empty_pool is False
    off = build_parser().parse_args(["--steps", "1", "--no-fork-pool-seed", "--allow-empty-pool"])
    assert off.fork_pool_seed is False
    assert off.allow_empty_pool is True
