"""The lock-in / recharge fixtures against the REAL Showdown (`gen3_lockin_cancelled_repeat_v1`, `gen3_recharge_expiry_v1`,
GIGO audit 2026-10-26 B-C1 and the re-run of its targeted games).

Each fixture is a few-turn SCRIPT (a `START` + `CHOOSE` lines, the core's own input-log form) replayed through BOTH the Node
bridge (Showdown, the oracle) and the Rust core (`ffi.trace`), and the two sides' complete line streams (`|t:|` stamps and the
request `rqid` normalised away) must be IDENTICAL. The scenarios are the ones the audit's 100 targeted games hit, shrunk to the
smallest board that reaches them, and each asserts its precondition on the Node stream first (the cancel really happened) so a
fixture that stops reaching the surface fails instead of passing vacuously:

* a 3-turn Thrash roll whose turn-2 repeat is CANCELLED by a Ghost's immunity ends the lock QUIETLY (the engine kept it a turn);
* a 2-turn roll cancelled the same way still ends in the fatigue confusion;
* a PROTECT-blocked repeat REFRESHES the lock (gen 4's `protect.onTryHit`);
* a user Spored AFTER its last locked repeat is confused although asleep (the duration tick precedes the sleep branch);
* a Hyper Beam user whose recharge action was cancelled by a faint is free the next turn (`mustrecharge` is `duration: 2`).

The pins with the sim's final PRNG seeds are `src/rust_sim/tests/regression_test.rs` (LI3, RC1); this is the live differential, and
the standing guard is `gen_sim_bridge_diff.js --mode lockin` (`designs/rust_sim/fuzzers.md`). Marked `integration` (it spawns the
Node bridge; needs `deps/pokemon-showdown`).
"""
from __future__ import annotations

import base64
import json
import select
import subprocess
import time
from typing import Dict, List, Tuple

import pytest

from utils.contention import scale_timeout
from utils.paths import repo_path
from utils.rust_env import ffi

pytestmark = [pytest.mark.sim, pytest.mark.integration]

USER = "Tauros||none|intimidate|thrash,splash|Hardy|85,85,85,85,85,85|M||||"
FOE = ("Snorlax||none|sturdy|splash,protect|Hardy|85,85,85,85,85,85|M||||]"
       "Gengar||none|levitate|splash|Hardy|85,85,85,85,85,85|M||||")
FOE_SPORE = "Parasect||none|effectspore|splash,spore|Hardy|85,85,85,85,85,85|M||||"
HB_USER = "Snorlax||none|sturdy|hyperbeam,splash|Hardy|85,85,85,85,85,85|M||||"
HB_FOE = ("Blissey||none|naturalcure|splash,explosion|Hardy|85,85,85,85,85,85|F||||]"
          "Chansey||none|naturalcure|splash|Hardy|85,85,85,85,85,85|F||||")


def _script(seed: str, p1: str, p2: str, commands: List[str]) -> str:
    start = {"formatid": "gen3customgame", "seed": seed,
             "p1": {"name": "P1", "team": p1}, "p2": {"name": "P2", "team": p2}}
    # a trace replays a FINISHED episode, so the fixture ends with p1's forfeit (the core's own `FORCELOSE` form)
    return "START " + json.dumps(start, separators=(",", ":")) + "\n" + "\n".join(commands + ["FORCELOSE p1"]) + "\n"


def _turns(p1_moves: List[str], p2_moves: List[str]) -> List[str]:
    out: List[str] = []
    for a, b in zip(p1_moves, p2_moves):
        out += [f"CHOOSE p1 {a}", f"CHOOSE p2 {b}"]
    return out


#: name -> (script, the Node-stream facts that prove the fixture reaches its surface, the facts that must be ABSENT)
CASES: Dict[str, Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = {
    "three_turn_roll_cancelled_by_immunity_ends_quietly": (
        _script("2,2,2,2", USER, FOE, _turns(["move 1"] * 3, ["move 1", "switch 2", "move 1"])),
        ("[from] lockedmove", "|-immune|p2a: Gengar"), ("|-start|p1a: Tauros|confusion",)),
    "two_turn_roll_cancelled_by_immunity_still_confuses": (
        _script("1,1,1,1", USER, FOE, _turns(["move 1"] * 3, ["move 1", "switch 2", "move 1"])),
        ("[from] lockedmove", "|-immune|p2a: Gengar", "|-start|p1a: Tauros|confusion|[fatigue]"), ()),
    "protect_blocked_repeat_refreshes_the_lock": (
        _script("2,2,2,2", USER, FOE, _turns(["move 1"] * 3, ["move 1", "move 2", "move 1"])),
        ("|-activate|p2a: Snorlax|Protect", "|-start|p1a: Tauros|confusion|[fatigue]"), ()),
    "user_spored_after_its_last_repeat_is_confused_asleep": (
        _script("1,1,1,1", USER, FOE_SPORE, _turns(["move 1"] * 3, ["move 1", "move 2", "move 1"])),
        ("|-status|p1a: Tauros|slp|[from] move: Spore", "|-start|p1a: Tauros|confusion|[fatigue]"), ()),
    "recharge_cancelled_by_a_faint_expires": (
        _script("2,2,2,2", HB_USER, HB_FOE,
                ["CHOOSE p1 move 1", "CHOOSE p2 move 1", "CHOOSE p1 move 1", "CHOOSE p2 move 2",
                 "CHOOSE p2 switch 2", "CHOOSE p1 move 1", "CHOOSE p2 move 1"]),
        ("|-mustrecharge|p1a: Snorlax", "|faint|p2a: Blissey"), ("|cant|p1a: Snorlax|recharge",)),
}


def _norm(line: str) -> str:
    if line.startswith("|request|"):
        v = json.loads(line[len("|request|"):])
        v.pop("rqid", None)
        return "|request|" + json.dumps(v, sort_keys=True)
    return line


def _keep(line: str) -> bool:
    return bool(line) and not line.startswith("|t:|") and line != "|"


def _node_streams(script: str) -> Dict[str, List[str]]:
    """The Node bridge's per-side stream for `script`: fed at once, read until it has been silent for a stretch."""
    bridge = str(repo_path("src", "utils", "bridge", "local_sim_bridge.js"))
    p = subprocess.Popen(["node", bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out: Dict[str, List[str]] = {"p1": [], "p2": []}
    try:
        p.stdin.write(script)
        p.stdin.flush()
        quiet = scale_timeout(2.5)           # a starved Node child is slower, not wedged: scale the idle window
        end_first = time.time() + scale_timeout(60.0)
        last = time.time()
        got_any = False
        while True:
            r, _, _ = select.select([p.stdout], [], [], 0.1)
            if r:
                ln = p.stdout.readline()
                if not ln:
                    break
                last = time.time()
                got_any = True
                if ln[:3] in ("p1 ", "p2 "):
                    out[ln[:2]] += base64.b64decode(ln[3:]).decode().split("\n")
                elif ln.startswith("__ERR__"):
                    pytest.fail("the Node bridge failed: " + base64.b64decode(ln[8:]).decode()[:300])
            elif got_any and time.time() - last > quiet:
                break
            elif not got_any and time.time() > end_first:
                pytest.fail("the Node bridge produced no output")
    finally:
        p.kill()
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_rust_core_equals_showdown_on_the_lock_in_and_recharge_fixtures(name):
    script, present, absent = CASES[name]
    node = _node_streams(script)
    lib = ffi.load(ffi.default_path())
    traced = ffi.trace(lib, script, commit="a", label="lockin-parity")
    for side in ("p1", "p2"):
        rust = [_norm(json.loads(l)["text"]) for l in traced["records"][side].splitlines()[1:]
                if _keep(json.loads(l)["text"])]
        node_lines = [_norm(x) for x in node[side] if _keep(x)]
        if side == "p1":
            text = "\n".join(node_lines)
            for needle in present:           # NON-VACUITY: Showdown really reached the surface
                assert needle in text, f"{name}: the fixture no longer reaches {needle!r} on Showdown:\n{text[-1500:]}"
            for needle in absent:
                assert needle not in text, f"{name}: Showdown shows {needle!r}:\n{text[-1500:]}"
        k = next((i for i in range(min(len(rust), len(node_lines))) if rust[i] != node_lines[i]), min(len(rust), len(node_lines)))
        assert rust == node_lines, (
            f"{name} [{side}]: the Rust core diverges from Showdown at line {k}\n"
            f"  RUST: {rust[k][:300] if k < len(rust) else None}\n  NODE: {node_lines[k][:300] if k < len(node_lines) else None}")
