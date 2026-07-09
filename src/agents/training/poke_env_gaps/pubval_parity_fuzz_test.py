"""Bridge fuzz test for V_pub feature PARITY (gen3_pubval_aux_v1) — the live path == the corpus parser.

The public-value aux head's target is only meaningful if the LIVE feature fold
(``pub_side_from_live`` over the LiveView, what Gen3Env evaluates each decision) computes EXACTLY
what the CALIBRATION corpus parser (``parse_replay_log``, what the frozen logistic was trained on)
computes for the same board. A drift between them is a silent GIGO: the head would be regressed
toward a V_pub evaluated on differently-defined features than it was calibrated on. Unit tests pin
each side on hand-built states; this drives BOTH sides over real Showdown battles via the
in-process BattleStream bridge (no server) and asserts, at every decision:

  1. PARITY — fold the trainee's own accumulated protocol stream through the corpus parser (with a
     synthetic ``|turn|`` line appended so the parser snapshots its CURRENT folded state) and
     compare every ``PubSide`` field against ``pub_side_from_live`` on the same instant's LiveView.
     Counts exact; HP sums atol 1e-3 (both sides fold the same lines); weather case-normalized.
  2. END-TO-END — the env's emitted ``pubval_target`` (the value the aux loss trains toward) equals
     the frozen artifact's prediction on the live features. Checked with a small mismatch budget:
     the bridge pump can advance the battle between the obs emit and this re-read (the documented
     opponent-decision race), so an occasional stale comparison is timing, not GIGO.

Run directly (no server; in-process bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/pubval_parity_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import sys
import traceback
from collections import Counter

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.pubval import PubValModel, features, parse_replay_log, pub_side_from_live
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
_HP_ATOL = 1e-3          # both sides fold the SAME protocol lines → near-exact
_E2E_ATOL = 1e-4         # obs pubval_target vs predict(live features) when no race
_E2E_MISMATCH_BUDGET = 0.01   # tolerated fraction of end-to-end checks lost to the pump race


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _build_bridge_env(teams, idx: int, lines: list):
    env = Gen3Env(
        load_mappings(),
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"PubValFz{idx}", None),
        start_listening=False,
        emit_pubval_target=True,       # the REAL env path under test (loads data/gen3_pubval.json)
    )
    # Hook the trainee's message handler BEFORE attaching the bridge: attach_bridge_transport
    # captures the BOUND `player._handle_battle_message` reference at attach time
    # (bridge_session.py `on_battle_message=`), so a wrapper installed after would never fire —
    # the exact vacuous-pass failure this ordering prevents.
    _capture_stream(env, lines)
    attach_bridge_transport(env, battle_format=BATTLE_FORMAT, persistent=True, recycle_every=10000)
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"PubValFzOpp{idx}", None),
        start_listening=False,
    )
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, env


def _capture_stream(env, lines: list):
    """Append every protocol line the TRAINEE receives (its one-sided stream) to ``lines`` —
    the same grammar the replay corpus carries, so the corpus parser folds it verbatim."""
    orig = env.agent1._handle_battle_message

    async def wrapped(split_messages):
        for msg in split_messages:
            lines.append("|" + "|".join(str(x) for x in msg[1:]) if len(msg) > 1 else "|")
        await orig(split_messages)

    env.agent1._handle_battle_message = wrapped


def _parser_now(lines: list, turn: int):
    """Fold the captured stream to NOW: append a synthetic |turn| so the parser snapshots its
    current running state (it only snapshots at |turn| boundaries)."""
    text = "\n".join(lines) + f"\n|turn|{turn}\n"
    positions, _w, _r, _ = parse_replay_log(text)
    return positions[-1] if positions else None


def _assert_side_parity(live_side, parser_side, tag: str, b: int, step: int):
    lp = pub_side_from_live(live_side)
    for field in ("alive", "revealed", "spikes", "statusc"):
        lv, pv = getattr(lp, field), getattr(parser_side, field)
        assert lv == pv, (f"PARITY[{tag}.{field}] battle {b} step {step}: live={lv} parser={pv}\n"
                          f"live={lp}\nparser={parser_side}")
    for field in ("active_hp", "known_hp", "boost"):
        lv, pv = getattr(lp, field), getattr(parser_side, field)
        assert abs(lv - pv) <= _HP_ATOL, (
            f"PARITY[{tag}.{field}] battle {b} step {step}: live={lv:.6f} parser={pv:.6f}\n"
            f"live={lp}\nparser={parser_side}")


def main(n_battles: int = 30) -> int:
    teams = _teams()
    rng = np.random.default_rng(0)
    lines: list = []
    wrapped, env = _build_bridge_env(teams, idx=1, lines=lines)
    model = PubValModel.load()

    stats = Counter()
    e2e_mismatch = 0
    t0 = __import__("time").time()
    try:
        for b in range(n_battles):
            lines.clear()
            obs, _ = wrapped.reset()
            for step in range(600):
                b1 = getattr(env, "battle1", None)
                if b1 is not None and lines:
                    live = b1.live_view()
                    role = str(getattr(b1, "player_role", None) or "p1")
                    snap = _parser_now(list(lines), int(live.turn))
                    if snap is not None:
                        _t, p1s, p2s, weather = snap
                        ours, theirs = (p1s, p2s) if role == "p1" else (p2s, p1s)
                        # (1) PARITY — every PubSide field, both sides.
                        _assert_side_parity(live.ours, ours, "ours", b, step)
                        _assert_side_parity(live.opp, theirs, "opp", b, step)
                        lw = (live.weather.weather or "none").lower()
                        pw = (weather or "none").lower()
                        assert lw == pw, f"PARITY[weather] battle {b} step {step}: live={lw} parser={pw}"
                        stats["parity_checks"] += 1
                        # (2) END-TO-END — the emitted target == predict(live features) (race-budgeted).
                        emitted = float(obs["pubval_target"][0])
                        assert float(obs["pubval_mask"][0]) == 1.0
                        want = model.predict(features(
                            pub_side_from_live(live.ours), pub_side_from_live(live.opp),
                            int(live.turn), live.weather.weather))
                        if abs(emitted - want) > _E2E_ATOL:
                            e2e_mismatch += 1
                        stats["e2e_checks"] += 1
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if legal.size else 0
                obs, _r, terminated, truncated, _info = wrapped.step(action)
                if terminated or truncated:
                    break
            else:
                raise AssertionError("episode did not finish within 600 steps")
            stats["battles"] += 1
    except Exception:
        traceback.print_exc()
        print(f"\nFAILED after {stats['battles']} battles / {stats['parity_checks']} parity checks")
        return 1

    if stats["parity_checks"] < n_battles * 5:
        # Anti-vacuous guard: a hook/plumbing regression that silences the capture must FAIL,
        # not pass with 0 checks (the exact failure the attach-order note above documents).
        print(f"FAIL: only {stats['parity_checks']} parity checks over {stats['battles']} battles — "
              "the stream capture is not seeing traffic (vacuous run).")
        return 1
    rate = e2e_mismatch / max(1, stats["e2e_checks"])
    print(f"\n{stats['battles']} battles | {stats['parity_checks']} parity checks — ALL EXACT | "
          f"e2e target checks {stats['e2e_checks']} (race mismatches {e2e_mismatch} = {rate:.2%}) | "
          f"{__import__('time').time() - t0:.0f}s")
    if rate > _E2E_MISMATCH_BUDGET:
        print(f"FAIL: end-to-end mismatch rate {rate:.2%} > {_E2E_MISMATCH_BUDGET:.0%} — that is not "
              "the pump race, the env target computation drifted from the artifact.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
