"""``live_view()`` memo fuzz — the memo'd path is BYTE-IDENTICAL to fresh construction.

``gen3_live_view_memo_v1``. Runs real battles in-process via the local BattleStream bridge
(no server) and, at **every decision**, proves two things against a full rebuild oracle:

  1. **View identity** — ``battle.live_view()`` (memo'd) ``==`` ``LiveView.from_battle``
     (fresh), field for field. Checked BEFORE and AFTER the decision's obs build, so a memo
     that survives one mutation too long inside the decision is caught too.
  2. **Observation byte-identity** — the full 2501-dim obs vector encoded with the memo warm
     is bit-for-bit equal to the same encode with the memo forcibly cleared. This is the
     contract that matters: the memo is a value-neutral refactor, so a single differing
     float is a failure.

The oracle is a REBUILD, never a second read of the memo — the two paths share
``LiveView.from_battle``, so the comparison is scheduler-vs-scheduler, not a fork that can
drift.

``--format gen3randombattle`` swaps the corpus to random battles, which is where the
whole-slot-invalidation family lives: Transform (Ditto), Forecast/Castform forme change,
the Deoxys/Unown formes, the wrap family and Baton Pass. The gen3ou pool exercises the
common path (reveals, item consumption, hazards, boosts, status, faints, phazing) at higher
density. Both are run in the gate; the corpus a green run actually covered is PRINTED, so a
pass cannot silently mean "the interesting lines never arrived".

⚠️ **Check 2 is gen3ou-only, and it says so out loud.** The obs encoder is scoped to gen3ou
and FAILS LOUD on anything outside it (a randbats Conversion raises
``UnknownVolatileError: typechange``) — that is the encoder's own coverage tripwire, not a
memo defect, so ``--format gen3randombattle`` runs check 1 alone and PRINTS that it did.
Silently swallowing the encoder's crash would have turned this gate into one that reports a
pass while measuring half of what it claims.

Run directly (no server needed):
    python src/agents/battle/live_view_memo_fuzz_test.py [n_battles]
    python src/agents/battle/live_view_memo_fuzz_test.py 20 --format gen3randombattle
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import traceback
from collections import Counter
from typing import Dict, List

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions, LiveView
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# Event kinds whose invalidation edge is the interesting one (whole-slot rewrites and the
# reveal/consume families). Reported as corpus coverage, not asserted — a gen3ou pool run
# will not contain a Transform, and pretending otherwise would make the gate lie.
_WATCHED = (
    EventKind.TRANSFORM, EventKind.FORMECHANGE, EventKind.ENDITEM, EventKind.ITEM,
    EventKind.ABILITY, EventKind.SETHP, EventKind.DRAG, EventKind.SWITCH,
    EventKind.VOLATILE_START, EventKind.FAINT, EventKind.STATUS, EventKind.WEATHER,
    EventKind.SIDE, EventKind.BOOST, EventKind.CANT,
)


def _diff_views(a: LiveView, b: LiveView) -> List[str]:
    """Field-level diff of two LiveViews (for a readable failure, not for the assert)."""
    out: List[str] = []

    def cmp(path, x, y):
        if dataclasses.is_dataclass(x) and dataclasses.is_dataclass(y):
            for f in dataclasses.fields(x):
                cmp(f"{path}.{f.name}", getattr(x, f.name), getattr(y, f.name))
        elif isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
            for i, (xi, yi) in enumerate(zip(x, y)):
                cmp(f"{path}[{i}]", xi, yi)
        elif x != y:
            out.append(f"{path}: memo={x!r} fresh={y!r}")

    cmp("live", a, b)
    return out or [f"live: memo={a!r} fresh={b!r}"]


class _MemoCheckPlayer(Player):
    """Plays random legal moves while shadow-verifying the memo at every decision."""

    def __init__(self, *args, check_obs: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_obs = check_obs
        self.obs_enc = get_observation_encoder(load_mappings())
        self._trackers: Dict[str, EpisodeTracker] = {}
        self.decisions = 0
        self.view_checks = 0
        self.obs_checks = 0
        self.failures: List[str] = []
        self.kinds: Counter = Counter()

    def _tracker(self, tag: str) -> EpisodeTracker:
        tr = self._trackers.get(tag)
        if tr is None:
            tr = EpisodeTracker(history_cap=1)
            self._trackers[tag] = tr
        return tr

    def _check_view(self, battle, when: str) -> None:
        memo = battle.live_view()
        fresh = LiveView.from_battle(battle)
        self.view_checks += 1
        if memo != fresh:
            self.failures.append(
                f"[{battle.battle_tag} t{battle.turn} {when}] live_view diverged:\n    "
                + "\n    ".join(_diff_views(memo, fresh)[:12])
            )

    def choose_move(self, battle):
        try:
            return self._checked(battle)
        except Exception:
            self.failures.append(traceback.format_exc())
            return self.choose_random_move(battle)

    def _checked(self, battle):
        tag = battle.battle_tag
        tr = self._tracker(tag)
        for e in battle.events_since(getattr(tr, "_last_cursor", 0)):
            if e.kind in _WATCHED:
                self.kinds[e.kind.name] += 1

        self._check_view(battle, "pre")

        legal = LegalActions.from_battle(battle)
        mask = np.asarray(Gen3ActionMasker.get_mask(battle, legal=legal)).astype(np.int8)
        if int(mask.sum()) == 0:
            return self.choose_random_move(battle)

        tr.record(battle, mask, legal=legal)
        tr.update_progress_clock(battle, legal)

        def _encode():
            return self.obs_enc.encode(
                battle, hp_tracker=tr.hidden_power_tracker, legal=legal,
                progress_clock=tr.progress_clock, recency=tr.recency,
                pair_history=tr.pair_history, event_window=tr.event_window,
            )

        if self.check_obs:
            warm = np.asarray(_encode(), dtype=np.float32)
            battle._live_view_memo = None      # force every consumer to rebuild
            cold = np.asarray(_encode(), dtype=np.float32)
            self.obs_checks += 1
            if warm.shape != cold.shape or not np.array_equal(warm, cold):
                bad = (np.flatnonzero(warm != cold)[:10] if warm.shape == cold.shape
                       else np.array([]))
                self.failures.append(
                    f"[{tag} t{battle.turn}] OBS DIVERGED memo-warm vs fresh-rebuild: "
                    f"shapes {warm.shape} vs {cold.shape}; first differing offsets "
                    f"{bad.tolist()} "
                    + "; ".join(f"@{i}: {warm[i]!r} vs {cold[i]!r}" for i in bad)
                )

        self._check_view(battle, "post")

        valid = [i for i, v in enumerate(mask) if v]
        idx = int(np.random.choice(valid)) if valid else 0
        tr.advance(idx)
        self.decisions += 1
        order = self.choose_random_move(battle)
        return order

    def _battle_finished_callback(self, battle):
        # The terminal board moves via won_by/tied, which never reach parse_message —
        # the door-3 edge, checked once per battle on the real object.
        try:
            self._check_view(battle, "finished")
        except Exception:
            self.failures.append(traceback.format_exc())
        self._trackers.pop(battle.battle_tag, None)


async def main(n_battles: int, battle_format: str) -> bool:
    # The obs encoder is gen3ou-scoped and fail-loud outside it (a randbats Conversion
    # raises UnknownVolatileError: typechange). That is the ENCODER's coverage tripwire,
    # not a memo defect — so check 2 is skipped, loudly, rather than swallowed.
    check_obs = battle_format == "gen3ou"
    if battle_format == "gen3ou":
        loader = TeamLoader()
        pool = loader.get_sample_teams() or loader.get_all_teams()
        if not pool:
            raise RuntimeError("no gen3ou teams found under data/teams")
        team = Gen3Teambuilder(pool)
    else:
        team = None  # random battles: the sim builds the teams

    p1 = _MemoCheckPlayer(
        account_configuration=AccountConfiguration("MemoFuzzA", None),
        battle_format=battle_format, team=team, start_listening=False,
        server_configuration=LocalhostServerConfiguration,
        check_obs=check_obs,
    )
    p1._battle_class = Gen3Battle
    p2 = RandomPlayer(
        account_configuration=AccountConfiguration("MemoFuzzB", None),
        battle_format=battle_format, team=team, start_listening=False,
        server_configuration=LocalhostServerConfiguration,
    )
    await run_local_battles(p1, p2, n_battles=n_battles)

    print("=" * 70)
    print(f"format={battle_format}  battles={n_battles}  decisions={p1.decisions}")
    print(f"live_view identity checks : {p1.view_checks}")
    if check_obs:
        print(f"obs byte-identity checks  : {p1.obs_checks}")
    else:
        print("obs byte-identity checks  : SKIPPED — the obs encoder is gen3ou-scoped and "
              "fail-loud outside it; run the gen3ou arm for check 2")
    print("corpus (watched event kinds seen): "
          + (", ".join(f"{k}={v}" for k, v in sorted(p1.kinds.items())) or "NONE"))
    if p1.decisions == 0 or (check_obs and p1.obs_checks == 0):
        print("FAIL — no decisions validated (did any battle actually run?)")
        return False
    if p1.failures:
        print(f"FAIL — {len(p1.failures)} divergence(s); first 5:")
        for f in p1.failures[:5]:
            print("  " + f)
        return False
    print("PASS — memo'd live_view == fresh construction"
          + ("; obs bit-for-bit identical." if check_obs else " (view check only)."))
    return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="live_view() memo byte-identity fuzz")
    ap.add_argument("n", nargs="?", type=int, default=20, help="battles to play")
    ap.add_argument("--format", default="gen3ou",
                    help="gen3ou (pool teams) or gen3randombattle (forme/Transform corpus)")
    a = ap.parse_args()
    sys.exit(0 if asyncio.run(main(a.n, a.format)) else 1)
