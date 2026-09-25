"""The PRE-REGISTERED plan of the cutover stress: every stream, its units and its target.

These numbers ARE the cutover tier (`designs/endstate/program_rust_core.md` §3, the CUTOVER row,
reformulated 2026-09-24 for owner option 2: COVERAGE-and-COUNT, not hours, because the cores are
shared with the training queue). They were written BEFORE any stress result was read; changing one
after the stress started is a new registration (the driver refuses a registration that differs from
the one on disk).

A stream is a list of UNITS; a unit is one process, minutes long, one durable row. Unit ``i`` of a
stream is a pure function of ``(stream, i)`` (plus the pool size and the ladder tier size, both
stamped in the registration), so a resumed driver re-derives exactly the units it skipped.

Key recipes (every battle is re-runnable alone from its row: the key, the two team indices, the
source):

* the player RNGs are ``1000+key`` / ``2000+key``, the sim seed ``[11+key, 22+key, 33+key, 44+key]``
  (``rust_core_parity.play``), so ``key`` must be unique ACROSS streams — each stream owns a key base
  — and at most :data:`MAX_KEY` (every seed word is a 16-bit int: the bridge REFUSES a larger one);
* ``ladder_full_a`` keeps the parity harness's own recipe (key ``k`` plays full-tier teams ``2k`` /
  ``2k+1``, key base 0), so its first 300 keys ARE the MILESTONE tier's ladder battles and the named
  ``LADDER_KNOWN_DIVERGENCES`` reproduce byte for byte; ``ladder_full_b`` plays the same pairs with
  the SLOTS SWAPPED, so every one of the 22,813 teams is played from both player slots and read by
  both viewers from each.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA = "gen3_core_cutover_stress_v1"

#: The largest key the sim-seed recipe accepts (``44 + key`` must fit a 16-bit seed word).
MAX_KEY = 65535 - 44

#: Pool offsets: team ``i`` meets ``i+d`` for each ``d`` (mod the pool), so every team plays each
#: offset once from EACH slot (as ``i`` it is p1, as ``i+d`` it is p2) — 12 distinct partners per
#: slot, spread from its neighbour to the far side of the pool.
POOL_RANDOM_OFFSETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
POOL_POLICY_OFFSETS = (1, 7, 61, 360)

#: The ladder tier the full-coverage streams play, and the one the policy stream plays.
LADDER_FULL_TIER = "full"
LADDER_POLICY_TIER = "milestone"


@dataclass(frozen=True)
class Stream:
    name: str
    kind: str                 # parity | corpora | fuzz | soak | envn
    battles: int              # the registered n (battles, fuzz battles, soak / env episodes)
    per_unit: int
    target: str               # the human statement §3 carries
    params: Dict = field(default_factory=dict)
    #: A capability the PIN must have for this stream to run (``None``: every pin). A stream whose
    #: requirement is missing is REPORTED as not runnable at this pin, never silently dropped.
    requires: Optional[str] = None

    @property
    def units(self) -> int:
        return math.ceil(self.battles / self.per_unit)

    def unit_range(self, i: int) -> range:
        if not 0 <= i < self.units:
            raise IndexError(f"{self.name}: unit {i} of {self.units}")
        return range(i * self.per_unit, min((i + 1) * self.per_unit, self.battles))


def streams(pool_n: int, ladder_full_n: int, ladder_policy_n: int) -> List[Stream]:
    """Every stream of the registration, in a fixed order."""
    lf = math.ceil(ladder_full_n / 2)
    out = [
        # ---- slices E / V / T / O, both viewers, every decision (+ live == offline) ----------
        Stream("ladder_full_a", "parity", lf, 40,
               f"all {ladder_full_n} full-tier LADDER teams, each once (teams 2k/2k+1), seeded-random",
               dict(source="ladder", tier=LADDER_FULL_TIER, policy=False, swap=False, key_base=0)),
        Stream("ladder_full_b", "parity", lf, 40,
               "the same pairs with the player SLOTS SWAPPED (every team from both slots)",
               dict(source="ladder", tier=LADDER_FULL_TIER, policy=False, swap=True, key_base=12_000)),
        Stream("pool_random", "parity", pool_n * len(POOL_RANDOM_OFFSETS), 40,
               f"every pool team x {len(POOL_RANDOM_OFFSETS)} partners from EACH slot, seeded-random",
               dict(source="pool", policy=False, offsets=POOL_RANDOM_OFFSETS, key_base=24_000)),
        Stream("pool_policy", "parity", pool_n * len(POOL_POLICY_OFFSETS), 20,
               f"every pool team x {len(POOL_POLICY_OFFSETS)} partners, the `production` policy at T=1",
               dict(source="pool", policy=True, offsets=POOL_POLICY_OFFSETS, key_base=33_000)),
        Stream("ladder_policy", "parity", ladder_policy_n, 20,
               f"the {ladder_policy_n} MILESTONE-tier ladder teams from both slots, the `production` policy",
               dict(source="ladder", tier=LADDER_POLICY_TIER, policy=True, key_base=36_000)),
        Stream("procedural_random", "parity", 4000, 40,
               "4,000 battles over 8,000 fresh PROCEDURAL teams, seeded-random",
               dict(source="procedural", policy=False, seed_base=7_000_000, key_base=37_000)),
        Stream("procedural_policy", "parity", 500, 20,
               "500 battles over 1,000 fresh PROCEDURAL teams, the `production` policy",
               dict(source="procedural", policy=True, seed_base=8_000_000, key_base=41_000)),
        Stream("corpora", "corpora", 1, 1,
               "the 22-scenario protocol corpus x 2 + every byte-fuzz fixture (slice E)", {}),
        # ---- the four Rust-vs-Node A/B fuzzers (each's own green-gate definition) ------------
        *_fuzz("state", "ab_fuzz.js", [], (("ladder", 10_000), ("ourandom", 3_000), ("pool", 1_000))),
        *_fuzz("proto", "ab_fuzz.js", ["--protocol", "--format", "gen3ou"],
               (("ladder", 10_000), ("ourandom", 3_000), ("pool", 1_000))),
        # (`bridge_ab_fuzz.js` and `gen_sim_bridge_diff.js` have no `ourandom` mode — their second
        # surface is the pool; amendment 1 of the registration, 2026-09-24.)
        *_fuzz("bridge", "bridge_ab_fuzz.js", ["--format", "gen3ou"],
               (("ladder", 5_000), ("pool", 2_000), ("trapping", 1_000))),
        *_fuzz("sbdiff", "gen_sim_bridge_diff.js", ["--format", "gen3ou", "--persistent"],
               (("ladder", 2_000), ("pool", 1_000)), per_unit=50),
        # ---- the SOAK: long-lived training-transport bridge children --------------------------
        Stream("soak_transport", "soak", 6 * 10_000, 10_000,
               "6 bridge children x 10,000 episodes each (~4.6x a production child's 3-h life of "
               "~2,150 battles), child + env RSS sampled every 250 episodes",
               dict(obs_source="python")),
        # ---- slice N (env level) and the core-obs soak: need `--obs-source core` in the pin ----
        Stream("envn_pool_random", "envn", 3000, 25,
               "3,000 training-shaped episodes, pool, seeded-random trainee, both obs sources in lockstep",
               dict(source="pool", policy=False, key_base=42_000), requires="obs_source_core"),
        Stream("envn_pool_policy", "envn", 1000, 20,
               "1,000 episodes, pool, the `production` policy trainee, both obs sources in lockstep",
               dict(source="pool", policy=True, key_base=45_000), requires="obs_source_core"),
        Stream("envn_ladder", "envn", 1000, 25,
               "1,000 episodes, full-tier ladder teams, seeded-random, both obs sources in lockstep",
               dict(source="ladder", policy=False, key_base=46_000), requires="obs_source_core"),
        Stream("envn_procedural", "envn", 500, 25,
               "500 episodes, procedural teams, seeded-random, both obs sources in lockstep",
               dict(source="procedural", policy=False, key_base=47_000), requires="obs_source_core"),
        Stream("soak_core_obs", "soak", 4 * 10_000, 10_000,
               "4 bridge children x 10,000 episodes under --obs-source core, RSS sampled",
               dict(obs_source="core"), requires="obs_source_core"),
    ]
    names = [s.name for s in out]
    assert len(names) == len(set(names)), names
    return out


def _fuzz(tag: str, script: str, extra: List[str], modes: Tuple[Tuple[str, int], ...],
          per_unit: int = 100) -> List[Stream]:
    out = []
    for mode, n in modes:
        argv = list(extra) + ["--mode", mode]
        if mode == "ladder":
            argv += ["--ladder-tier", LADDER_FULL_TIER]
        out.append(Stream(f"fz_{tag}_{mode}", "fuzz", n, per_unit,
                          f"{script} {' '.join(argv)}: {n:,} battles, 0 non-allowlisted",
                          dict(script=script, argv=argv, seed_base=_FUZZ_SEED_BASE[tag])))
    return out


#: Master seeds: unit ``i`` of a fuzz stream runs ``--master-seed seed_base + i`` (mode is part of
#: the stream, so the same seed in two modes is two different battle sets).
_FUZZ_SEED_BASE = {"state": 610_000, "proto": 620_000, "bridge": 630_000, "sbdiff": 640_000}


def unit_id(stream: str, i: int) -> str:
    return f"{stream}.{i:05d}"


def parse_unit_id(uid: str) -> Tuple[str, int]:
    name, _, i = uid.rpartition(".")
    return name, int(i)


# ---------------------------------------------------------------------------------------------
# the battles of a parity unit
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ParityBattle:
    """One parity battle: the RNG key, the two team indices in the stream's team list, a label."""

    key: int
    t1: int
    t2: int
    label: str


def parity_battles(stream: Stream, i: int, n_teams: int) -> List[ParityBattle]:
    """The battles of unit ``i``. ``n_teams`` is the stream's team-list size (the pool, the ladder
    tier, or ``2 x unit battles`` for a procedural unit, whose teams are drawn per unit)."""
    p = stream.params
    out = []
    for b in stream.unit_range(i):
        if p["source"] == "ladder" and "swap" in p:
            k = b
            a, c = (2 * k) % n_teams, (2 * k + 1) % n_teams
            if p["swap"]:
                a, c = c, a
            key = p["key_base"] + k
            out.append(ParityBattle(key, a, c, f"ladder{'B' if p['swap'] else 'A'}_{k}"))
        elif p["source"] == "ladder":            # the policy stream: pairs, both slot orders
            k, swap = divmod(b, 2)
            a, c = (2 * k) % n_teams, (2 * k + 1) % n_teams
            if swap:
                a, c = c, a
            out.append(ParityBattle(p["key_base"] + b, a, c, f"policy_ladder_{k}{'s' if swap else ''}"))
        elif p["source"] == "pool":
            offs = p["offsets"]
            t, j = divmod(b, len(offs))
            out.append(ParityBattle(p["key_base"] + b, t % n_teams, (t + offs[j]) % n_teams,
                                    f"{'policy_' if p['policy'] else ''}pool_{t}_{offs[j]}"))
        elif p["source"] == "procedural":
            m = b - stream.unit_range(i).start
            out.append(ParityBattle(p["key_base"] + b, 2 * m, 2 * m + 1,
                                    f"{'policy_' if p['policy'] else ''}procedural_{b}"))
        else:
            raise ValueError(f"{stream.name}: unknown source {p['source']!r}")
    return out


def procedural_seed(stream: Stream, i: int) -> int:
    """The procedural generator's seed for unit ``i`` (its ``2 x per_unit`` teams)."""
    from utils.team_sources import PROCEDURAL_SEED

    return PROCEDURAL_SEED + stream.params["seed_base"] + i
