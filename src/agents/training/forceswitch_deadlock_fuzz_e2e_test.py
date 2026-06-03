"""Force-switch request-delivery DEADLOCK — reproduction + regression guard (needs a live 9XXX server).

WHAT THIS CATCHES
-----------------
A latent concurrency bug in poke-env's gym-env request handshake (present verbatim in upstream
``poke_env`` 0.15.0) that can wedge a training env forever. ``_AsyncQueue.race_get``
(``poke_env/environment/env.py``) races a per-agent ``queue.get()`` against the ``_waiting`` /
``_trying_again`` coordination events. Two ways it can drop a request that the server has ALREADY
delivered into the queue:

  1. STRANDING — ``asyncio.wait(FIRST_COMPLETED)`` returns the instant ANY waiter completes, so an
     already-set (STALE) event wins before the equally-ready ``queue.get()`` is even scheduled.
     ``race_get`` returns ``None`` and the env marks the agent "not to move" while its request
     sits unread in the queue.
  2. ORPHAN THEFT — ``race_get`` ``cancel()``s the pending ``queue.get()``; a bare cancel can
     leave it alive just long enough for a later ``put`` to wake it, where it dequeues and
     DISCARDS the request (counted in ``qsize`` but gone).

Either way the agent's order is never sent, the server goes silent, and the env hangs. The
``_trying_again`` event goes stale because ``env.step`` clears it only on the
``race_get``-returned-``None`` path — but a re-request makes the battle non-``None``, skipping
that clear, so the event lingers and later wins a race it shouldn't.

This is a cross-thread TIMING race (POKE_LOOP delivering protocol messages vs the training thread
running ``env.step``), so it is rare — ~1 in 8,600 battles in production self-play. ``--widen``
sleeps inside the decision (releasing the GIL so POKE_LOOP runs concurrently) to widen the window
and make it reliably observable (~1 in 70) — the same technique as ``racing_player_fuzz_e2e_test``.

HOW IT REPRODUCES
-----------------
Drives the EXACT training stack — ``Gen3Env`` + ``MaskableAgentWrapper`` + an opponent polled on
the training thread — via the gym API against a live server. Both sides lead a Choice-Band
Arena-Trap Dugtrio (the trigger scenario): each turn they attempt a switch (under Arena Trap the
server rejects it → ``[Unavailable choice]`` → sets the soon-to-be-stale ``_trying_again``), then
Earthquake, so a (near-)simultaneous faint hands one side a ``forceSwitch`` and the other a
``wait`` together — the exact request pair whose force-switch gets stranded. Self-play mirror
matches are what made this common in training; against fixed heuristic bots it almost never fired.

HOW TO RUN (use a private 9XXX port — NEVER 8000/8001)
-----------------------------------------------------
    npm run showdown -- 9124          # separate shell
    export PYTHONPATH=$PYTHONPATH:src
    GEN3_RACE_GET_TIMEOUT_S=8 GEN3_RACE_TRACE=1 \
      python src/agents/training/forceswitch_deadlock_fuzz_e2e_test.py [--port 9124] [--battles 500] [--widen 0.015]

  - ``GEN3_RACE_GET_TIMEOUT_S=8`` makes the watchdog fail fast (default 120s) instead of hanging.
  - ``GEN3_RACE_TRACE=1`` dumps the wedged battle's cross-thread interleaving on a trip.
  - ``--widen 0.015`` + a few hundred battles is needed to surface the race on buggy code.

PASS = every battle ran to completion, no silent-stall ``ShowdownException`` (the bug is fixed).
FAIL = the silent-stall watchdog fired = the deadlock reproduced.

GUARDS the fix in ``poke_env/environment/env.py``: ``_AsyncQueue.race_get`` now settles the
cancelled ``queue.get()`` (recovering its item, never orphaning it) and prefers a queued battle
over a stale event, and ``env.step`` clears ``_trying_again`` the moment its agent receives a
battle. Unit-level coverage of the same two failure modes is in ``async_queue_disconnect_test.py``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.exceptions import ShowdownException
from poke_env.player import Player
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.wrappers import MaskableAgentWrapper
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Both players lead a Choice-Band Arena-Trap Dugtrio with Earthquake as move slot 0 (action
# index 6). Minimal-HP Dugtrio so CB Earthquake is a clean OHKO on the mirror → a turn-1 faint
# that hands one side a forceSwitch and the other a wait simultaneously.
_DUGTRIO_LEAD_TEAM = """\
Dugtrio @ Choice Band
Ability: Arena Trap
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Earthquake
- Rock Slide
- Aerial Ace
- Beat Up

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Surf
- Calm Mind
- Ice Beam
- Rest

Snorlax @ Leftovers
Ability: Immunity
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Body Slam
- Curse
- Rest
- Earthquake

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Spikes
- Roar
- Drill Peck
- Rest

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Claydol @ Leftovers
Ability: Levitate
EVs: 252 HP / 4 Atk / 252 SpD
Impish Nature
- Earthquake
- Psychic
- Rapid Spin
- Explosion
"""

_EARTHQUAKE_ACTION = 6  # MOVE_START (6) + move slot 0; Earthquake is slot 0 of the lead Dugtrio.


class _TrapBaiter(Player):
    """Opponent brain that reproduces the bug's precondition: TRY to switch whenever a switch is
    offered (under Arena Trap the server rejects it → ``[Unavailable choice]`` → ``_trying_again``
    set), and otherwise Earthquake (the OHKO that produces the mid-turn faint). The rejected
    switch (which leaves a stale ``_trying_again``) plus the faint (the wait/forceSwitch pair) are
    what wedge ``race_get``. Returns a synchronous order (SingleAgentWrapper polls it on the
    training thread).

    ``widen_s`` sleeps inside the decision (releasing the GIL so POKE_LOOP runs concurrently) to
    WIDEN the cross-thread window and make the rare timing race reliably observable."""

    def __init__(self, *args, widen_s: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._widen_s = widen_s

    def choose_move(self, battle):
        if self._widen_s:
            time.sleep(self._widen_s)
        if battle.available_switches:                       # bait the trap: attempt a switch
            return self.create_order(battle.available_switches[0])
        for m in (battle.available_moves or []):            # trapped → Earthquake (OHKO)
            if m.id == "earthquake":
                return self.create_order(m)
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_default_move()


def _build(port: int, widen_s: float = 0.0):
    mappings = load_mappings()
    sc = localhost_server_configuration(port)
    ts = int(time.time()) % 100000
    env = Gen3Env(
        mappings,
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(_DUGTRIO_LEAD_TEAM),
        server_configuration=sc,
        account_configuration1=AccountConfiguration(f"FsDl{ts}", "password"),
    )
    opp = _TrapBaiter(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(_DUGTRIO_LEAD_TEAM),
        server_configuration=sc,
        account_configuration=AccountConfiguration(f"FsOpp{ts}", "password"),
        start_listening=False,  # opponent is a brain; the env's agent2 is the connection
        widen_s=widen_s,
    )
    wrapped = MaskableAgentWrapper(env, opp)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, env


def _pick_action(obs) -> int:
    """Bait the trap to match production: a switch action (0–5) when the mask offers one (under
    Arena Trap the server rejects it → ``_trying_again``), else Earthquake (action 6), else first
    legal."""
    mask = np.asarray(obs["action_mask"])
    for i in range(min(6, len(mask))):          # 0–5 are switches; attempt one if offered
        if mask[i]:
            return i
    if _EARTHQUAKE_ACTION < len(mask) and mask[_EARTHQUAKE_ACTION]:
        return _EARTHQUAKE_ACTION
    legal = [i for i in range(len(mask)) if mask[i]]
    return legal[0] if legal else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9124)
    ap.add_argument("--battles", type=int, default=6)
    ap.add_argument("--widen", type=float, default=0.0,
                    help="opponent-decision + pre-step sleep (s) to widen the cross-thread race window")
    args = ap.parse_args()

    print(f"Force-switch deadlock repro — {args.battles} battles — :{args.port} — widen={args.widen*1000:.0f}ms")
    wrapped, env = _build(args.port, widen_s=args.widen)
    finished = 0
    deadlock = None
    try:
        for ep in range(args.battles):
            obs, _ = wrapped.reset()
            done = False
            steps = 0
            while not done and steps < 400:
                if args.widen:
                    time.sleep(args.widen * random.random())  # jitter the trainee-step timing too
                a = _pick_action(obs)
                obs, _r, term, trunc, _info = wrapped.step(a)
                done = bool(term or trunc)
                steps += 1
            finished += 1
            print(f"  battle {ep + 1:>2}/{args.battles}: completed in {steps} steps")
    except ShowdownException as e:
        msg = str(e)
        if "silent battle stall" in msg:
            deadlock = msg
            print(f"\n>>> DEADLOCK REPRODUCED after {finished} clean battle(s): watchdog fired <<<")
        else:
            raise
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("=" * 68)
    if deadlock is not None:
        # The race-trace dump (if GEN3_RACE_TRACE=1) follows the first line in the exception text.
        print("❌ FORCE-SWITCH DEADLOCK FUZZ: race_get stranded a queued request "
              "(the wait/forceSwitch handshake wedged).")
        print("   " + deadlock.splitlines()[0])
        sys.exit(1)
    print(f"✅ FORCE-SWITCH DEADLOCK FUZZ PASSED — {finished}/{args.battles} battles "
          f"completed, no silent stall.")
    sys.exit(0)


if __name__ == "__main__":
    main()
