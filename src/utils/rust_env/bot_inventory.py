"""THE SCRIPTED-BOT INVENTORY — every bot class a pool plays, where, what it reads, what it draws
(M5 Lane F, `designs/endstate/program_rust_core.md` §2 M5).

The table of record for porting the scripted bots into the Rust env core (`src/rust_env/src/bots/`).
`bot_inventory_test.py` (routine) derives every roster FROM THE CODE and fails when:

* a class appears in a roster (the training pool, the eval roster)
  with no row here — a new bot must be inventoried before it plays;
* a row's ``used_by`` disagrees with the rosters (a bot dropped from a pool reads as dropped here);
* a ``Player`` subclass is defined in one of the bot modules with no row (defined-but-unused bots
  are listed with ``used_by=()`` so "not in any pool" is a stated fact, not an omission);
* (the exploiter keep-bots mix == the training roster check read the Python env core's `env_factory` /
  `MaskableAgentWrapper`, deleted in deletion pass U3; the Rust plan builds the floor from the same
  roster, `rust_env_setup._bot_names`);
* a row marked PORTED names a Rust bot the crate does not define.

THE STATE a bot reads is ONE object: the opponent side's poke-env ``Battle`` (in training
``env.battle2``, agent2's own battle; in eval / anchors the bot player's own battle).
The Rust port reads the env core's ``BoardReading`` for that side — the port of that object — through
the bot VIEW (`utils.rust_env.bot_view`), whose equality is checked at every banked decision.

THE RANDOMNESS, per stream (the seed route is the PRODUCTION one; every stream is the process-wide
``random`` module unless a seed arrives):

* ``choice`` — ``Player._choice_rng`` (`gen3_player_choice_rng_v1`): ``choose_random_move``, every
  bot's fallback and RandomPlayer's whole policy, and ``DEFAULT_CHOICE_CHANCE`` on a rejected order.
  Seeded by ``rng_seed=`` or ``$GEN3AI_PLAYER_SEED``; production passes neither.
* ``protect`` — the two stallers' Protect coin (``_protect_rng``): ``protect_seed=`` or
  ``$GEN3AI_STALLER_SEED``; production passes neither.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The roster SITES a bot can be used from (the test derives each from the code).
SITES = {
    "train": "the training floor roster — `main/train/matchup_setup.py` OPPONENT_CLASSES (also the "
             "exploiter --exploiter-keep-bots mix, mapped to the core's bots by `rust_env_setup._bot_names`)",
    "eval": "the eval roster — `agents/training/eval_roster.py` _EVAL_OPPONENT_SPECS (also "
            "`main.anchors` `bot:<name>` and the prober's replay, both through that table)",
}


@dataclass(frozen=True)
class BotRow:
    name: str            # display name (`eval_callback.opponent_name`)
    cls: str             # dotted class path
    used_by: tuple       # keys of SITES
    rng: tuple           # streams the bot can draw from ("choice", "protect")
    reads: str           # what it reads off the Battle (beyond what every bot reads)
    rust: str            # the Rust bot (`bots::Kind::<X>`), or "" while not ported


#: Every bot the fallbacks share: ``choose_random_move`` reads ``valid_orders`` (``wait``,
#: ``trapped``, ``force_switch``, ``available_switches``, ``available_moves``) and draws one ``choice``.
ROWS: tuple = (
    BotRow("random", "poke_env.player.baselines.RandomPlayer", ("eval",), ("choice",),
           "valid_orders only — one choice draw per decision (its WHOLE policy)", "Random"),
    BotRow("heuristic", "poke_env.player.baselines.SimpleHeuristicsPlayer",
           ("train", "eval"), ("choice",),
           "matchup (types, base spe, HP), _stat_estimation (base stats, boosts), own stats "
           "(_should_switch_out), hazards (side conditions), setup (self_setup_boosts: Target.SELF boosts or a "
           "non-Ghost's Curse — dead until the F-LF-1 fix, 2026-09-29), fallback draw", "Heuristic"),
    BotRow("heuristic2", "agents.opponents.Gen3HeuristicV2Player", ("train", "eval"), ("choice",),
           "damage calc (base stats, boosts, HP, ability, status FRZ), revealed opp moves, own bench movesets, "
           "hazards, recovery, setup (Target.SELF or non-Ghost Curse; dead until the F-LF-1 fix), status immunity", "HeuristicV2"),
    BotRow("staller", "agents.opponents.Gen3StallerPlayer", ("train", "eval"), ("choice", "protect"),
           "opp status, Protect coin when TOX, recovery, side_conditions (any), best damage v1, best switch v1",
           "Staller"),
    BotRow("staller_v2", "agents.opponents.Gen3StallerV2Player", ("train", "eval"), ("choice", "protect"),
           "+ revealed opp damage (pivot), status immunity, last_move (is_last_used), damage v2, switch v2",
           "StallerV2"),
    BotRow("aggressive", "agents.opponents.Gen3AggressivePlayer", ("train", "eval"), ("choice",),
           "damage v1 over damaging moves, forced-switch by base atk/spa", "Aggressive"),
    BotRow("aggressive_v2", "agents.opponents.Gen3AggressiveV2Player", ("train", "eval"), ("choice",),
           "KO calc, matchup, immunity escape, damage v2, switch v2", "AggressiveV2"),
    BotRow("setup_sweep", "agents.opponents.Gen3SetupSweepPlayer", ("train", "eval"), ("choice",),
           "matchup, boosts, setup (Target.SELF or non-Ghost Curse; dead until the F-LF-1 fix), damage v1, switch v1", "SetupSweep"),
    BotRow("setup_sweep_v2", "agents.opponents.Gen3SetupSweepV2Player", ("train", "eval"), ("choice",),
           "KO calc, revealed opp damage, matchup, setup (Target.SELF or non-Ghost Curse; dead until the F-LF-1 fix), damage v2, switch v2", "SetupSweepV2"),
    BotRow("max_base_power", "poke_env.player.baselines.MaxBasePowerPlayer", (), ("choice",),
           "max base_power; in NO pool (upstream poke-env baseline, defined only)", ""),
)


def by_class() -> dict:
    return {r.cls: r for r in ROWS}


def by_name() -> dict:
    return {r.name: r for r in ROWS}


def ported() -> tuple:
    return tuple(r for r in ROWS if r.rust)
