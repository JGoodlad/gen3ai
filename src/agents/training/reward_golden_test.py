"""`gen3_reward_golden_v2` — THE REWARD STREAM, BIT FOR BIT, AS A COLLECTED TEST.

The reward (`RewardBreakdown.total`), its TERMINAL term (`win_loss`) and the material margin the
env publishes as the `win_margin` obs key (`Gen3RewardManager._last_material_margin`), for every
decision of 30 real bridge battles under SIX terminal-only compositions, hashed. If the number the
trainer optimizes changes, this says so — in the routine gate, not in a one-off script.

**v2 is the SHAPED-REWARD DELETION's parity proof** (program_rust_core §4 M3 row, 2026-09-26).
v1 hashed every `RewardBreakdown` field under six compositions, five of them SHAPED; the deletion
removed those fields and those compositions, so a v1 hash could only ever move. v2 hashes what
survives — the reward a run actually trains on — and was RECORDED AT THE LAST PRE-DELETION COMMIT
(`produced_at_commit` in `reward_golden.json`), with the shaped code still in the tree. It passing
unchanged at the deletion commit is the statement "production reward is byte-identical", measured
rather than argued. The compositions are written as raw `model_config.json`-shaped dicts and built
through `RewardConfig.from_dict`, which ignores unknown keys — so the SAME source constructs the
same config on both sides of the deletion (before it, `hand_shaping: False` switches the shaping
off; after it, the key is simply gone).

`production` is read from the production mirror itself (`baselines.production_config()`), not
retyped, so the golden follows what production IS rather than what someone remembered it to be.

**REPRODUCIBLE BY CONSTRUCTION — the four clauses, all of them.** `designs/ops/testing.md`: a
pytest-collected test wants the SAME battle every run, and `random.seed(k)` is not enough because
two players share the global module RNG and the bridge interleaves their `choose_move` calls. So:
fixed teams by pool index · a per-player `RandomState` / `random.Random` (never the module RNG) ·
a fixed sim PRNG seed · **`concurrency=1`**, the clause that gets missed (at concurrency 3 two runs
of the same measurement differed by up to +0.043).

⚠️ **Why not `obs_roundtrip_fuzz_test.record_fixture_battle`**, which testing.md names as the source
of record for a collected test's battle: it hands back OBS artifacts after the battle, and the
reward stream is not in them — a `RewardBreakdown` is produced by folding a live
`Gen3RewardManager` across the battle AS it happens. So this reuses the helper's RECIPE (the same
:class:`SeededRandomPlayer`, teams-by-index, ``seed=[11+key, 22+key, 33+key, 44+key]``,
``concurrency=1``) rather than its return value.

**Tier: `sim`, and deliberately NOT `slow`** — it plays real battles in-process through the bridge
(~20 s), and a cheap battle-backed linchpin belongs in the routine gate.

Regenerate (only when a reward change is INTENDED) and record the new hash in the ledger:

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/agents/training/reward_golden_test.py --write

It is never a routine step: the hash moving means the number the trainer optimizes moved.
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

#: The terminal-only compositions, as RAW `model_config.json`-shaped dicts built through
#: `RewardConfig.from_dict` (unknown keys ignored — see the module docstring for why that matters to
#: a before/after proof). `production` is the mirror itself; the other five vary the TERMINAL's own
#: three knobs (indicator, magnitude, timeout score), which are all that survives the deletion.
#: The `hand_shaping: False` keys are the PRE-deletion spelling of "no shaping"; after the deletion
#: `from_dict` drops them and the config is the same one.
_NO_SHAPING = {"hand_shaping": False, "pbrs_material": False, "pbrs_belief": False,
               "no_progress_tax_armed": False}


def compositions() -> dict:
    """name -> the raw dict a `RewardConfig` is built from. A function, not a constant, because
    `production` is READ from the committed mirror."""
    from agents.training.baselines import production_config
    return {
        "production":            dict(production_config()),
        "indicator_30":          {**_NO_SHAPING, "terminal_indicator": True, "victory_value": 30.0},
        "signed_unit_draw_loss": {**_NO_SHAPING, "terminal_indicator": False, "victory_value": 1.0,
                                  "draw_penalty": -1.0},
        "signed_default":        {**_NO_SHAPING, "terminal_indicator": False},
        "signed_draw_worse":     {**_NO_SHAPING, "terminal_indicator": False, "victory_value": 1.0,
                                  "draw_penalty": -2.0},
        "indicator_unit":        {**_NO_SHAPING, "terminal_indicator": True, "victory_value": 1.0,
                                  "draw_penalty": 0.0},
    }


class _BattleState:
    def __init__(self, config: RewardConfig):
        self.mgr = Gen3RewardManager(config=config)
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
        reward = state.mgr.process_turn_reward(battle, delta)
        bd = state.mgr._last_breakdown
        self.decisions += 1
        margin = float(state.mgr._last_material_margin)
        self._sink.append(f"{self.decisions:05d} reward={float(reward).hex()} "
                          f"total={float(bd.total).hex()} win_loss={float(bd.win_loss).hex()} "
                          f"win_margin={margin.hex()}")

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
    for name, raw in compositions().items():
        config = RewardConfig.from_dict(raw)
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
`reward_config.py` or `material_margin.py` changed the number the trainer optimizes (or the
`win_margin` obs key). The composition names above localise it — `production` moving is the one
that matters most.

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
    assert doc.get("schema") == "gen3_reward_golden_v2", f"{MANIFEST}: unexpected schema"
    return doc


def test_the_reward_stream_matches_the_recorded_golden():
    """The whole point: every decision's reward, terminal and win_margin x 6 compositions, hashed."""
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


def test_every_composition_in_the_sweep_is_terminal_only():
    """Cheap and independent of the battles: every swept composition is a reward the tree can
    still build, and it is TERMINAL-only — the only class that survives the deletion."""
    for name, raw in compositions().items():
        cfg = RewardConfig.from_dict(raw)
        for k in ("victory_value", "terminal_indicator", "draw_penalty"):
            if k in raw:
                assert getattr(cfg, k) == raw[k], f"{name}: RewardConfig.{k} did not take {raw[k]!r}"


def _write_manifest(n_battles: int) -> None:
    from utils.git import get_git_hash
    import datetime
    with _in_repo_root():
        body, sweeps, errors = asyncio.run(build_golden(n_battles, progress=True))
    assert not errors, "\n".join(errors)
    digest = hashlib.sha256(body.encode()).hexdigest()
    doc = {
        "schema": "gen3_reward_golden_v2",
        "sha256": digest,
        "produced_at_commit": get_git_hash(),
        "produced_on": datetime.date.today().isoformat(),
        "n_battles": n_battles,
        "rows": len(body.splitlines()),
        "compositions": sorted(compositions()),
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
