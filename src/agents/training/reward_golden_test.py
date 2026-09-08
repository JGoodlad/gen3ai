"""`gen3_reward_golden_v1` — THE REWARD STREAM, BIT FOR BIT, AS A COLLECTED TEST.

Every field of every `RewardBreakdown`, for every decision of 30 real bridge battles, under SIX
reward compositions, hashed. If the number the trainer optimizes changes, this says so — and says
it in the routine gate, not in a one-off script somebody remembers to run.

**Where it came from.** It was built as a throwaway during the 2026-09-07 `reward_manager.py`
decomposition (`b0b3a253`, 1,990 → 808 lines across four modules) to prove the refactor moved no
bytes: BEFORE and AFTER hashed identically at
``9463dc242efde3fabadd652db4bef6e158eb24b5a0e53680e29be077b6a1926f``. A byte-identity reference
that exists only in one agent's scratch directory protects exactly one refactor; promoted here it
protects every future one, which is the whole difference between an artifact and a habit.

**REPRODUCIBLE BY CONSTRUCTION — the four clauses, all of them.** `designs/ops/testing.md`: a
pytest-collected test wants the SAME battle every run, and `random.seed(k)` is not enough because
two players share the global module RNG and the bridge interleaves their `choose_move` calls. So:
fixed teams by pool index · a per-player `RandomState` / `random.Random` (never the module RNG) ·
a fixed sim PRNG seed · **`concurrency=1`**, the clause that gets missed (at concurrency 3 two runs
of the same measurement differed by up to +0.043). Verified end-to-end: the hash reproduces
bit-for-bit in a FRESH worktree with an independently checked-out ``data/`` (2026-09-07).

⚠️ **Why not `obs_roundtrip_fuzz_test.record_fixture_battle`**, which testing.md names as the source
of record for a collected test's battle. That helper plays its battle and hands back the OBS
artifacts *afterwards* — a reconstruction record, a summary, an npz of observation rows. The reward
stream is not in any of them: a `RewardBreakdown` is produced by folding a live `Gen3RewardManager`
across the battle AS it happens, and this test needs six different folds (six compositions) over the
same board. So it reuses the helper's RECIPE rather than its return value — the same
:class:`SeededRandomPlayer` opponent, the same teams-by-index, the same
``seed=[11+key, 22+key, 33+key, 44+key]``, the same ``concurrency=1``. The reproducibility clauses
are identical; only the thing being recorded differs.

**Tier: `sim`, and deliberately NOT `slow`.** It plays real battles in-process through the
`deps/pokemon-showdown` bridge, so `sim` is what it NEEDS. It costs **19.7 s of CALL time measured
under pytest beside a live training run AND a concurrent `-m slow -n 2` suite** (2026-09-07; 24.5 s
as a standalone script), and `slow` means *minutes, not seconds* — the marker that decides
routine cost. A reward-stream golden belongs in the routine gate for the same reason the six-battle
obs golden does: it is battle-backed AND cheap, and the one time this tree put a cheap battle-backed
linchpin behind a cost wall it rode main RED three separate times.

Regenerate (only when a reward change is INTENDED) and record the new hash in the ledger:

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/agents/training/reward_golden_test.py --write

which rewrites `reward_golden.json` beside this file with the new hash and the commit it was
produced at. It is never a routine step: the hash moving means the number the trainer optimizes
moved, and that is a research event before it is a maintenance one.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.training.battle_snapshot import BattleContext
from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer
from agents.training.reward_manager import Gen3RewardManager, RewardConfig
from agents.training.slot_registry import SlotRegistry
from agents.training.turn_delta import TurnDelta
from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from utils.bridge.local_battle_runner import run_local_battles
from utils.paths import repo_root, src_path
from utils.team_loader import TeamLoader

pytestmark = pytest.mark.sim

BATTLE_FORMAT = "gen3ou"

#: The recorded hash and its provenance, beside this file.
MANIFEST = Path(__file__).with_name("reward_golden.json")

#: Battle keys 0..N_BATTLES-1 are drawn from the team pool; key N_BATTLES is the MIXED_TEAM mirror.
N_BATTLES = 4

#: name -> RewardConfig kwargs. Each names WHAT it covers, so a hash change can be LOCALISED to a
#: composition instead of "the reward moved". Together they reach every fold family: the PBRS
#: potentials, the additive BIAS regime, the bias-refund, the redesign levers, the fully-PBRS arm,
#: and the win-prob era's terminal-indicator-only world.
COMPOSITIONS = {
    "production_default":  dict(),
    "additive_bias":       dict(all_shaping_pbrs=False),
    "redesign_levers":     dict(all_shaping_pbrs=False, bias_redesign=True,
                                switch_bias_weight=0.5, self_ko_hp_penalty=1.0,
                                bias_additivity=0.5, mat_alive_weight=1.5),
    "drops_and_stall":     dict(all_shaping_pbrs=False, drop_redundant_bias=True,
                                drop_switch_bias=True, no_progress_penalty=0.25),
    "fully_pbrs":          dict(all_shaping_pbrs=True, stall_pbrs=True),
    "clean_world":         dict(hand_shaping=False, pbrs_material=False, pbrs_belief=False,
                                terminal_indicator=True, victory_value=1.0,
                                no_progress_tax_armed=True),
}


class _StubClock:
    """A DETERMINISTIC stand-in for `ProgressClock` — this pins the reward manager's arithmetic,
    not the clock's. Ticked once per decision so `no_progress_tax` (BIAS) and Φ_progress (PBRS)
    both carry non-zero values instead of the clock-absent 0.0."""

    def __init__(self) -> None:
        self.n = 0
        self.last_penalty = 0.0

    def tick(self) -> None:
        self.n += 1
        self.last_penalty = -0.15 if (self.n % 3 == 0) else 0.0

    def value(self) -> float:
        return ((self.n * 7) % 13) / 13.0

    def reset(self) -> None:
        self.n = 0
        self.last_penalty = 0.0


class _BattleState:
    def __init__(self, config: RewardConfig):
        self.clock = _StubClock()
        self.mgr = Gen3RewardManager(config=config, progress_clock=self.clock)
        self.prev_ctx = None
        self.last_action = None
        self.prev_cursor = 0
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()


class GoldenPlayer(Player):
    """Drives ONE reward manager per battle and records every field of every breakdown.

    ⚠️ An exception inside `choose_move` is COLLECTED, never `os._exit`-ed and never swallowed: the
    script this was promoted from could afford to kill the process, a pytest worker cannot. The
    battle is then finished with the default order and `errors` is asserted EMPTY before the hash is
    compared — a harness that died mid-battle must not be reported as a reward change."""

    def __init__(self, *args, config: RewardConfig, rng_seed: int, sink: list, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._rng = np.random.RandomState(rng_seed)
        self._sink = sink
        self._per_battle: dict = {}
        self.decisions = 0
        self.errors: list = []

    def _state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState(self._config)
        return self._per_battle[tag]

    def _fold(self, battle, state, curr_ctx) -> None:
        events = battle.events_since(state.prev_cursor)
        delta = TurnDelta.build_from_events(state.prev_ctx, curr_ctx, state.last_action, events)
        state.clock.tick()
        state.mgr.process_turn_reward(battle, delta)
        bd = state.mgr._last_breakdown
        self.decisions += 1
        row = " ".join(f"{n}={float(getattr(bd, n)).hex()}"
                       for n in type(bd).field_names())
        self._sink.append(f"{self.decisions:05d} total={float(bd.total).hex()} {row}")

    def _battle_finished_callback(self, battle) -> None:
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.ones(11, dtype=np.int8)
            curr_ctx = BattleContext.from_battle(battle, mask, state.our_slots, state.opp_slots)
            self._fold(battle, state, curr_ctx)
        except Exception as e:                       # pragma: no cover - diagnostic
            self._sink.append(f"FINAL_TURN_SKIPPED {tag}: {e}")
            self.errors.append(f"final turn {tag}: {e!r}")
        finally:
            self._per_battle.pop(tag, None)

    def choose_move(self, battle):
        try:
            state = self._state(battle)
            mask = Gen3ActionMasker.get_mask(battle)
            curr_ctx = BattleContext.from_battle(battle, mask, state.our_slots, state.opp_slots)
            if state.prev_ctx is not None and state.last_action is not None:
                self._fold(battle, state, curr_ctx)
            valid = np.where(mask == 1)[0]
            choice = int(valid[self._rng.randint(len(valid))])
            if not battle.finished:
                state.mgr.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice
                state.prev_cursor = battle.event_cursor
            return Gen3ActionMapper.action_to_order(choice, battle)
        except Exception as e:                       # pragma: no cover - diagnostic
            self.errors.append(f"{battle.battle_tag} turn {battle.turn}: {e!r}\n"
                               f"{traceback.format_exc()}")
            return self.choose_default_move()


# A deliberately broad team on BOTH sides for one battle per composition — Spikes/Roar,
# Calm Mind/Dragon Dance, Explosion, Will-O-Wisp/Toxic/Thunder Wave, Protect — so the boost,
# phaze, hazard-cap and self-KO families are exercised, not just what the pool happens to draw.
MIXED_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 240 HP / 244 Def / 24 Spe
Bold Nature
- Surf
- Ice Beam
- Calm Mind
- Rest

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Dragon Dance

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Will-O-Wisp
- Explosion

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Protect
- Toxic

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break
"""


@contextlib.contextmanager
def _in_repo_root():
    """`TeamLoader` resolves ``data/teams`` — and, inside itself, ``data/<entry>`` — relative to the
    CWD. A collected test may be invoked from anywhere, and a pool that silently comes back EMPTY
    would turn this golden into a test of nothing, so the cwd is pinned rather than assumed."""
    prev = os.getcwd()
    os.chdir(repo_root())
    try:
        yield
    finally:
        os.chdir(prev)


async def _one(name: str, config: RewardConfig, key: int, sink: list, teams=None):
    """One reproducible battle: fixed teams, per-player RNG, fixed sim seed, concurrency=1."""
    if teams is None:
        pool = TeamLoader().get_all_teams()
        assert pool, "no gen3ou teams under data/teams — the golden would hash an empty pool"
        teams = (pool[key % len(pool)], pool[(key + 3) % len(pool)])
    t1, t2 = teams
    tag = f"{name[:3]}{key}"
    trainee = GoldenPlayer(
        config=config, rng_seed=1000 + key, sink=sink,
        battle_format=BATTLE_FORMAT, team=t1, battle_class=Gen3Battle,
        account_configuration=AccountConfiguration(f"RG{tag}t", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = SeededRandomPlayer(
        rng_seed=2000 + key,
        battle_format=BATTLE_FORMAT, team=t2, battle_class=Gen3Battle,
        account_configuration=AccountConfiguration(f"RG{tag}o", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    await run_local_battles(trainee, opp, 1,
                            seed=[11 + key, 22 + key, 33 + key, 44 + key],
                            concurrency=1)
    return trainee.decisions, trainee.errors


async def build_golden(n_battles: int = N_BATTLES, progress=False):
    """Play the whole sweep and return ``(body, sweeps, errors)``.

    ``sweeps`` is ``{"<composition> <label>": {"decisions": n, "sha256": ...}}`` — a PER-SWEEP
    hash as well as the whole-body one, so a mismatch names the composition and the battle that
    moved instead of only saying "the reward changed". That is the difference between a golden
    that starts an investigation and one that ends it.
    """
    lines: list = []
    sweeps: dict = {}
    errors: list = []
    for name, kwargs in COMPOSITIONS.items():
        config = RewardConfig(**kwargs)
        for key in range(n_battles + 1):
            # key == n_battles is the MIXED_TEAM mirror battle (broad signal coverage).
            teams = (MIXED_TEAM, MIXED_TEAM) if key == n_battles else None
            sink: list = []
            n, errs = await _one(name, config, key, sink, teams)
            errors.extend(errs)
            label = "mixed" if key == n_battles else f"key={key}"
            header = f"### {name} {label} decisions={n}"
            lines.append(header)
            lines.extend(sink)
            sweeps[f"{name} {label}"] = {
                "decisions": n,
                "sha256": hashlib.sha256(("\n".join([header] + sink) + "\n").encode()).hexdigest(),
            }
            if progress:
                print(f"  {name:<20} {label:<8} {n:4d} decisions", flush=True)
    return "\n".join(lines) + "\n", sweeps, errors


_CHANGED = """\
the reward sequence changed.

    recorded  {want}
    measured  {got}

The recorded stream is {rows} rows over {n_sweeps} battle x composition sweeps
({manifest}, produced at commit {commit} on {produced_on}).

WHICH SWEEPS MOVED ({n_moved} of {n_sweeps}):
{moved}

If this is UNINTENDED, it is a reward regression: something in `reward_manager.py`,
`reward_bias_terms.py`, `reward_potentials.py` or `reward_config.py` changed the number the
trainer optimizes. The composition names above localise it — a term that only exists in one
composition moves only that composition's hash; a change to the fold SEQUENCE moves them all.

If it is INTENDED, regenerate and record it in the ledger:

    export PYTHONPATH=$PYTHONPATH:src
    python3 {script} --write

and append a ledger entry saying WHICH term moved and why — a golden hash that changes without a
ledger line is a golden nobody can audit."""


def _load_manifest() -> dict:
    assert MANIFEST.exists(), (
        f"{MANIFEST} is missing — it is a COMMITTED artifact recording the reward stream's hash. "
        f"Restore it with `git checkout -- {MANIFEST}`, or regenerate with "
        f"`python3 {src_path('agents', 'training', 'reward_golden_test.py')} --write`.")
    doc = json.loads(MANIFEST.read_text())
    assert doc.get("schema") == "gen3_reward_golden_v1", f"{MANIFEST}: unexpected schema"
    return doc


def test_the_reward_stream_matches_the_recorded_golden():
    """The whole point: 2,772 decisions x every `RewardBreakdown` field x 6 compositions, hashed."""
    want = _load_manifest()
    with _in_repo_root():
        body, sweeps, errors = asyncio.run(build_golden(want["n_battles"]))

    # Preconditions, ASSERTED rather than branched on — a harness that quietly played fewer
    # battles, or died mid-battle, must not be reported as "the reward changed".
    assert not errors, ("the golden harness errored, so the hash means nothing:\n"
                        + "\n".join(errors))
    assert sorted(sweeps) == sorted(want["sweeps"]), (
        f"played {len(sweeps)} sweeps, recorded {len(want['sweeps'])}")
    drift = {k: (v["decisions"], want["sweeps"][k]["decisions"]) for k, v in sweeps.items()
             if v["decisions"] != want["sweeps"][k]["decisions"]}
    assert not drift, (
        "the BATTLES diverged before the reward did — decision counts differ from the recording "
        f"(played, recorded): {drift}. That is a simulator / team-pool / seed change, not a "
        "reward change, and the hash below would be meaningless.")

    rows = body.splitlines()
    assert len(rows) == want["rows"], f"{len(rows)} rows, recorded {want['rows']}"
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != want["sha256"]:
        moved = [k for k, v in sorted(sweeps.items())
                 if v["sha256"] != want["sweeps"][k]["sha256"]]
        pytest.fail(_CHANGED.format(
            want=want["sha256"], got=digest, rows=want["rows"], n_sweeps=len(sweeps),
            n_moved=len(moved), moved="\n".join(f"    {k}" for k in moved) or "    (none — the "
            "per-sweep hashes all match, so the concatenation itself changed)",
            manifest=MANIFEST.name, commit=want["produced_at_commit"],
            produced_on=want["produced_on"],
            script=src_path("agents", "training", "reward_golden_test.py")))


def test_every_composition_in_the_sweep_is_a_constructible_reward_config():
    """Cheap and independent of the battles: a renamed lever must fail HERE, with its own name,
    rather than as an opaque hash mismatch 25 seconds later."""
    for name, kwargs in COMPOSITIONS.items():
        cfg = RewardConfig(**kwargs)
        for k, v in kwargs.items():
            assert getattr(cfg, k) == v, f"{name}: RewardConfig.{k} did not take {v!r}"


def _write_manifest(n_battles: int) -> None:
    from utils.git import get_git_hash
    import datetime
    with _in_repo_root():
        body, sweeps, errors = asyncio.run(build_golden(n_battles, progress=True))
    assert not errors, "\n".join(errors)
    digest = hashlib.sha256(body.encode()).hexdigest()
    doc = {
        "schema": "gen3_reward_golden_v1",
        "sha256": digest,
        "produced_at_commit": get_git_hash(),
        "produced_on": datetime.date.today().isoformat(),
        "n_battles": n_battles,
        "rows": len(body.splitlines()),
        "compositions": sorted(COMPOSITIONS),
        "sweeps": sweeps,
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\nSHA256 : {digest}\nwrote  : {MANIFEST}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_manifest(N_BATTLES)
    else:
        with _in_repo_root():
            _body, _sweeps, _errs = asyncio.run(build_golden(N_BATTLES, progress=True))
        print(f"\nrows   : {len(_body.splitlines())}")
        print(f"SHA256 : {hashlib.sha256(_body.encode()).hexdigest()}")
        print(f"errors : {_errs or 'none'}")
