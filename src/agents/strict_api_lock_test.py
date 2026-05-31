"""Phase-4 lock: the static guard that makes the strict battle-API standard stick.

Background (`designs/ai_v4/todo_live_battle.md`). Our code is meant to read all battle
state through the vetted, fuzzed read-models we own — `LiveView` (current board),
`TurnView` (history, event-folded), and `LegalActions` (server-authoritative legality) —
behind `StrictBattleView`. poke-env keeps running underneath as the state engine, but it is
an implementation detail, not something the consumer clusters poke at directly. Likewise the
four spec-defined value-enums flow through the `agents.enums` seam, never imported from
poke_env directly.

Until now that standard was convention-only. This test is the enforcement: it AST-walks the
consumer modules under `observation/`, `action/`, `training/`, `inference/` and FAILS on

  (a) a raw read of a stateful `battle.<attr>` (the temporal/overwrite-on-every-line fields
      that belong behind the read-models), and
  (b) a `from poke_env … import` of one of the four accepted value-enums (they must come from
      `agents.enums`).

AST, not regex: a regex flags docstrings, comments, and unrelated `.team`/`.weather` on
non-battle objects. The walk only sees real attribute-access nodes whose base is the
`battle` identifier, and real `ImportFrom` nodes.

A small, per-(file, attr) allowlist records the handful of reads the migration audit
confirmed are *intended seams* (not incomplete migration) — each justified inline. Because
the allowlist is keyed by attribute, a NEW kind of raw read (a different stateful attr, or a
read in a fresh file) still fails. A `test_allowlist_has_no_dead_entries` guard keeps the
allowlist from rotting: if a refactor removes an allowlisted read, the stale entry must be
pruned rather than left to silently widen the guard.
"""

import ast
import pathlib

import pytest

# src/agents/ — this test lives at the agents root so it can walk all four clusters.
AGENTS_ROOT = pathlib.Path(__file__).resolve().parent
CONSUMER_DIRS = ("observation", "action", "training", "inference")

# Stateful `battle` attributes that must flow through LiveView / TurnView / LegalActions.
# (Meta scalars — turn / battle_tag / finished / won / lost / wait — are deliberately NOT
# here: the strict view re-exposes them verbatim, so reading them off the battle is fine.)
STATEFUL_BATTLE_ATTRS = frozenset({
    "active_pokemon", "opponent_active_pokemon",
    "available_moves", "available_switches",
    "last_request",
    "team", "opponent_team",
    "weather", "side_conditions", "opponent_side_conditions",
    "force_switch", "trapped",
    "_player_role",
})

# The four accepted value-enums. They must be imported from `agents.enums`, never poke_env.
VALUE_ENUMS = frozenset({"PokemonType", "Status", "MoveCategory", "Weather"})

# Documented boundary seams excluded from the walk entirely (by relpath under AGENTS_ROOT):
#   * action/serialize.py — the one sanctioned Choice -> BattleOrder serialization touch
#     (it reads battle.available_moves / available_switches to resolve a Choice to a
#     poke-env order; a future fully-owned Player would re-derive this).
#   * training/battle_snapshot.py — the per-decision decision-capture seam (BattleContext).
#     It still reads poke-env directly for the fields the poke-env-gap fuzz tests probe
#     (we_moved_first / our_move_crit / *_last_effectiveness) and for the Hidden-Power
#     tracker's opp_last_damaging_event; a future fully-owned Player would re-derive these.
#     (The old battle_context.py — the diff-based heuristic reader — was deleted by the
#     Phase-5 TurnDelta event-fold; the heuristic TurnDelta.build it fed is gone.)
EXCLUDED_RELPATHS = frozenset({
    "action/serialize.py",
    "training/battle_snapshot.py",
})

# Per-(relpath, attr) allowlist of KNOWN-LEGITIMATE residual raw reads. Each is an intended
# seam the audit confirmed, NOT incomplete migration. Keyed by attribute (not line number)
# so it survives line shifts but a NEW stateful attr in the same file still fails.
ALLOWLISTED_BATTLE_READS = frozenset({
    # observation/reactive.py — the effectiveness hot loop (288 matchup cells + the active-
    # move block) stays on the raw battle ON PURPOSE: our own Hidden Power keeps its typed id
    # (hiddenpowerfire) only on the raw Move object, the loop is keyed on PokemonType enums
    # (LiveView exposes strings), and alignment_test pins each cell to move.type. The team /
    # opponent_team reads are the documented `live is None` unit-test/plain-Battle fallback
    # for fainted counts, byte-identical to the LiveView path.
    ("observation/reactive.py", "available_moves"),
    ("observation/reactive.py", "active_pokemon"),
    ("observation/reactive.py", "opponent_active_pokemon"),
    ("observation/reactive.py", "team"),
    ("observation/reactive.py", "opponent_team"),

    # observation/base.py — get_team_list(), the team-list helper that FEEDS the reactive
    # matchup matrices. It must return raw Pokemon (with raw Move objects) so get_sorted_moves
    # preserves typed Hidden Power; migrating it to LiveView would collapse own HP and break
    # the same alignment_test pin as reactive. Same intended seam, hoisted into the helper.
    ("observation/base.py", "team"),
    ("observation/base.py", "opponent_team"),
    ("observation/base.py", "opponent_active_pokemon"),

    # observation/state_encoder.py — `mon is battle.opponent_active_pokemon`, an identity
    # check used to set the per-mon active flag. It compares object identity against the same
    # poke-env mon being iterated, so it is intrinsically a raw-object check, not state we
    # could re-source from LiveView (which holds primitives, no Pokemon back-reference).
    ("observation/state_encoder.py", "opponent_active_pokemon"),
})


def _is_scaffolding(path: pathlib.Path) -> bool:
    """Tests, fuzz harnesses, and benchmarks construct and poke at battles for
    measurement/validation — like `*_test.py`, they are not production consumer modules and
    are out of scope for the lock. The stem-substring check also catches `fuzz_test_unit.py`
    (which does not end in `_test.py`)."""
    stem = path.stem
    return "test" in stem or "benchmark" in stem


def _consumer_modules():
    """Yield (relpath:str, path:Path) for every production consumer module in the walk."""
    for d in CONSUMER_DIRS:
        for path in sorted((AGENTS_ROOT / d).glob("*.py")):
            if _is_scaffolding(path):
                continue
            relpath = path.relative_to(AGENTS_ROOT).as_posix()
            if relpath in EXCLUDED_RELPATHS:
                continue
            yield relpath, path


def _battle_reads(tree: ast.AST):
    """Every `battle.<attr>` access where <attr> is a stateful battle field.
    Yields (attr, lineno). Matches only when the base is the `battle` identifier — so
    unrelated `.team`/`.weather` on other objects are not flagged."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in STATEFUL_BATTLE_ATTRS
            and isinstance(node.value, ast.Name)
            and node.value.id == "battle"
        ):
            yield node.attr, node.lineno


def _poke_env_enum_imports(tree: ast.AST):
    """Every `from poke_env… import {value-enum}`. Yields (sorted_names, lineno)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("poke_env"):
            hits = sorted(a.name for a in node.names if a.name in VALUE_ENUMS)
            if hits:
                yield hits, node.lineno


def test_no_raw_stateful_battle_reads_in_consumers():
    """(a) No consumer module reads a stateful `battle.<attr>` outside the allowlist."""
    violations = []
    for relpath, path in _consumer_modules():
        tree = ast.parse(path.read_text(), filename=relpath)
        for attr, lineno in _battle_reads(tree):
            if (relpath, attr) not in ALLOWLISTED_BATTLE_READS:
                violations.append(f"  {relpath}:{lineno}  battle.{attr}")

    assert not violations, (
        "Raw stateful battle reads found outside the allowlist — route these through "
        "LiveView / TurnView / LegalActions (via battle.strict_view()), or, if this is a "
        "genuine intended seam, add a commented (relpath, attr) entry to "
        "ALLOWLISTED_BATTLE_READS:\n" + "\n".join(violations)
    )


def test_no_poke_env_value_enum_imports_in_consumers():
    """(b) No consumer module imports the four value-enums from poke_env — use agents.enums."""
    violations = []
    for relpath, path in _consumer_modules():
        tree = ast.parse(path.read_text(), filename=relpath)
        for names, lineno in _poke_env_enum_imports(tree):
            violations.append(f"  {relpath}:{lineno}  from poke_env import {', '.join(names)}")

    assert not violations, (
        "Value-enums imported directly from poke_env — import them from `agents.enums` "
        "instead (the accepted-enums seam):\n" + "\n".join(violations)
    )


def test_allowlist_has_no_dead_entries():
    """Keep the allowlist honest: every allowlisted (relpath, attr) must still correspond to
    a real raw read. A stale entry (left after a refactor removed the read) silently widens
    the guard, so it must be pruned."""
    live_reads = set()
    excluded_now = set()
    for relpath, path in _consumer_modules():
        tree = ast.parse(path.read_text(), filename=relpath)
        for attr, _ in _battle_reads(tree):
            live_reads.add((relpath, attr))

    # An allowlist entry can also go dead if its file is newly excluded — surface that too.
    for relpath, _attr in ALLOWLISTED_BATTLE_READS:
        if relpath in EXCLUDED_RELPATHS:
            excluded_now.add((relpath, _attr))

    dead = (set(ALLOWLISTED_BATTLE_READS) - live_reads) - excluded_now
    assert not dead, (
        "Stale ALLOWLISTED_BATTLE_READS entries no longer match any raw read — prune them so "
        "the allowlist stays minimal:\n"
        + "\n".join(f"  {relpath}  battle.{attr}" for relpath, attr in sorted(dead))
    )


def test_walk_actually_covers_the_known_consumers():
    """Sanity-pin the walk: if a path/exclusion change silently emptied the module set, the
    guard above would pass vacuously. Assert the clusters we expect are present."""
    found = {relpath for relpath, _ in _consumer_modules()}
    for expected in (
        "observation/reactive.py",
        "observation/state_encoder.py",
        "action/mapper.py",
        "action/mask_generator.py",
        "training/reward_manager.py",
        "training/gen3_env.py",
        "inference/player.py",
    ):
        assert expected in found, f"{expected} missing from the consumer walk — exclusions too broad?"
    # And the documented seams are genuinely excluded.
    assert "action/serialize.py" not in found
    assert "training/battle_snapshot.py" not in found
