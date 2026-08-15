"""rust-vs-node bridge PARITY — the correctness signal for `--use-bridge=rust`.

The Rust `sim_bridge` binary is a drop-in for the Node `local_sim_bridge.js`, byte-for-byte
protocol-validated at the chunk/stdout level by the crate's own
`src/rust_sim/harness/gen_sim_bridge_diff.js` (which drives NAME-based choices off the Node
bridge's `|request|` frames and diffs the per-side chunk streams — 36 ok / 0 diverged on the
gender-pinned corpus). This test is the PYTHON-integration complement: it drives the REAL RL
transport path — `run_local_battles(..., impl=…)` → `local_battle_runner` → the subprocess
spawn / base64 framing / demux / poke-env parse — so it catches an integration bug (plumbing,
framing, the persistent-child lifecycle, a poke-env parse divergence) a Rust-internal byte-diff
can't, and confirms poke-env's own name-serialized choices (`/choose move earthquake`,
`move hiddenpowerice`, `/choose switch <species>`) resolve through the Rust bridge.

HONEST SCOPE — the Rust bridge's move coverage. The pokesim port models a large-but-INCOMPLETE
gen3 move/ability set and FAIL-LOUDS (`__ERR__`) on anything unmodeled (Aromatherapy, Wish,
Baton Pass, …) rather than silently desync. Real `gen3ou` sample teams routinely carry an
unmodeled move, so a random-real-team battle through the Rust bridge can crash on it. This test
therefore drives battles ONE AT A TIME and treats an unmodeled-move `__ERR__` as a SKIP (not a
failure), reporting the completion rate — so it measures TRANSPORT correctness (the wiring this
change adds) over the battles the Rust engine can actually play, without being dominated by the
port's move-coverage gap. `--use-bridge=rust` is only safe for a run whose teams stay inside the
port's modeled universe; the startup path documents/warns this.

Two checks (both restricted to the battles that completed under BOTH impls):

1. **Integration smoke** — the SAME two deterministic heuristic bots play battles under
   `impl="rust"`; assert a healthy fraction FINISH with no *transport* error. Three outcome
   classes per battle: OK (played to completion), a `__ERR__ … is not modeled` **skip** (the port's
   move-coverage gap — not a transport bug), or a **transport error** (a real correctness bug that
   must be 0 — a plumbing/framing crash, OR a poke-env PARSE `ValueError` from a Rust protocol-
   CONTENT divergence, e.g. an emitted mon-identifier token that differs from node's and overflows
   the team). This check would have caught the move-name/switch-species resolution bug this wiring
   surfaced (poke-env serializes choices by move-id / species-name, not slot number, so a slot-only
   Rust parser crashed with `unsupported choice "move hiddenpowerice"`).

2. **Statistical win-rate parity at `seed=None`** — the TRAINING regime (training runs
   `seed=None`). Over the completed battles, assert node and rust p1 win-rates agree within a
   tolerance. `seed=None` (not a fixed seed) is the right parity check because an EXACT per-battle
   seed match on real `gen3ou` teams is subject to a DOCUMENTED port seed-convention gap: the
   port's `run_full_battle` omits the sim's turn-0 construction endTurn (the per-mon gender
   `sample(['M','F'])` + speed-tie construction shuffles), so a FIXED seed on gender-unspecified
   teams can diverge the turn-1 line — see `advance_seed_for_construction` in
   `src/rust_sim/src/bridge.rs` + `CLAUDE.md`. With `seed=None` there is no shared reference, so
   the two impls are the SAME battle distribution and their win-rates must match statistically.
   (The gender-pinned EXACT byte-parity is the crate harness's job, cited above.)

Run directly as a script (builds the rust binary via the helper if the
`$POKESIM_SIM_BRIDGE_BIN` override isn't set):

    export PYTHONPATH=$PYTHONPATH:src
    python src/utils/bridge/bridge_impl_parity_test.py [n_battles]

Under pytest it also exposes `test_rust_node_bridge_parity`, SKIPPED unless a pre-built rust
binary is available (`$POKESIM_SIM_BRIDGE_BIN` set, or `src/rust_sim/target/release/sim_bridge`
already present) — so the default unit suite never triggers a cargo build (nor rebuilds the
shared `src/rust_sim/target/`). Provide a binary to exercise it as a real parity check.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from poke_env import AccountConfiguration

from agents.opponents import Gen3HeuristicV2Player, Gen3AggressiveV2Player
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge import local_battle_runner
from utils.bridge.local_battle_runner import run_local_battles
from utils.contention import describe_contention

BATTLE_FORMAT = "gen3ou"
# Short per-battle timeout so an unmodeled-move HANG (some Rust fail-loud paths emit __ERR__ to
# one side but never deliver the other side's request → the demux blocks) is skipped in seconds,
# not the 180s production default. A healthy in-process bridge battle is < 2s.
#
# It bounds the IDLE GAP between protocol chunks, not the battle's duration — the distinction
# that makes it survive a saturation event. The hang it exists to catch emits NO chunks, so 20s of
# silence catches it just as fast as a 20s duration cap did, while a battle that is merely STARVED
# keeps emitting and finishes. MEASURED 2026-08-14, as a duration cap, on a box saturated by a
# `cargo build --release`: 8 of 12 battles scored as timeouts and the test FAILED, none wedged.
# `local_battle_runner` still scales it by measured contention (`ProgressDeadline` re-scales at
# every check), so a spike mid-battle widens the budget rather than manufacturing a failure.
_PARITY_BATTLE_IDLE_BUDGET = 20.0

# Above this fraction of timed-out battles the surviving sample is not worth a verdict.
_TIMEOUT_INCONCLUSIVE_FRAC = 0.25


def _build_pair(all_teams, tag: str):
    """Two deterministic heuristic bots (different archetypes for a livelier game), built
    start_listening=False so no websocket opens — the bridge is the only transport."""
    p1 = Gen3HeuristicV2Player(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(all_teams),
        account_configuration=AccountConfiguration(f"ParH{tag}", "password"),
        start_listening=False,
    )
    p2 = Gen3AggressiveV2Player(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(all_teams),
        account_configuration=AccountConfiguration(f"ParA{tag}", "password"),
        start_listening=False,
    )
    return p1, p2


_UNMODELED_MARKER = "is not modeled"  # the pokesim fail-loud message for an unmodeled mechanic


def _play_series(impl: str, n_battles: int, all_teams, tag: str):
    """Play up to ``n_battles`` under ``impl`` at ``seed=None`` (fresh sim RNG per battle — the
    training regime), ONE AT A TIME so an unmodeled-move `__ERR__` (Rust engine fail-loud) skips
    only that battle instead of aborting the run.

    Returns (p1_wins, finished, skipped_unmodeled, transport_errors, timed_out).

    ``timed_out`` is its OWN bucket, and that is the whole point (see the timeout handler
    below): a timeout is a statement about the BOX, never about the port's move coverage.
    """
    p1, p2 = _build_pair(all_teams, tag)
    skipped = transport_err = timed_out = 0
    # Bound the IDLE GAP so an unmodeled-move HANG skips fast (restored after). This used to
    # shrink the TOTAL per-battle cap instead, which bounded the wrong thing: it also killed
    # battles that were merely slow, and beside the live trainer that scored 4 of 12 (33%) as
    # timeouts at ~1.4x contention. The hang it exists to catch produces NO protocol chunks, so
    # the idle gap catches it just as fast without punishing a healthy starved battle.
    saved = local_battle_runner._BATTLE_IDLE_BUDGET
    local_battle_runner._BATTLE_IDLE_BUDGET = _PARITY_BATTLE_IDLE_BUDGET
    try:
        for _ in range(n_battles):
            try:
                asyncio.run(
                    run_local_battles(p1, p2, 1, battle_format=BATTLE_FORMAT, impl=impl)
                )
            except RuntimeError as e:
                if _UNMODELED_MARKER in str(e):
                    skipped += 1  # the port doesn't model this team's move → skip, not a transport bug
                    continue
                transport_err += 1  # a real bridge/plumbing error — surfaced by the caller
                continue
            except ValueError as e:
                # A poke-env PARSE error (e.g. "team already has 6 pokemons" from a Pressure
                # target resolved off a `|move|` line) is a Rust protocol-CONTENT divergence — the
                # emitted mon-identifier token differs from node's. NOT an unmodeled-move skip: it
                # is a real correctness bug that must be 0. Count it as a transport error so the
                # caller FAILS on it.
                transport_err += 1
                continue
            except (asyncio.TimeoutError, TimeoutError):
                # A per-battle timeout. Some Rust unmodeled-move fail-loud paths emit __ERR__ to
                # one side but never deliver the other's request → the demux hangs, so a timeout
                # CAN mean an unmodeled move.
                #
                # It can equally mean the box was busy. This used to be counted as `skipped`, and
                # that conflation produced a genuinely dangerous artifact: beside a live training
                # run, 39 of 40 battles timed out and the suite reported them as "skipped
                # (unmodeled move)" — a clean-looking result that blamed the PORT for the load
                # average, and left the win-rate parity check to draw its conclusion from the one
                # surviving battle. A timeout is never a semantic outcome; it gets its own bucket
                # so the caller can tell the two apart and say which happened.
                timed_out += 1
                continue
    finally:
        local_battle_runner._BATTLE_IDLE_BUDGET = saved
    return p1.n_won_battles, p1.n_finished_battles, skipped, transport_err, timed_out


def _starvation_verdict(label: str, timed_out: int, attempted: int) -> None:
    """Refuse to report a result computed from a starved sample — and name the cause.

    Sized as a FRACTION of the attempted battles, not an absolute count: one timeout in 60 is
    a plausible wedge worth noting, while a third of them is a statement about the machine.
    """
    if timed_out == 0 or attempted == 0:
        return
    frac = timed_out / attempted
    print(f"[parity] ⚠️  {label}: {timed_out}/{attempted} battles TIMED OUT ({frac:.0%}). "
          f"{describe_contention()}")
    if frac >= _TIMEOUT_INCONCLUSIVE_FRAC:
        raise AssertionError(
            f"{label} INCONCLUSIVE: {timed_out}/{attempted} battles ({frac:.0%}) timed out, so "
            f"the surviving sample is not representative and no parity claim can be made from "
            f"it. This is a statement about the BOX, not about the port's move coverage — it "
            f"used to be silently counted as an 'unmodeled move' skip, which turned a starved "
            f"run into a clean-looking pass. {describe_contention()}")


def run(n_battles: int = 60, tol: float = 0.20) -> None:
    loader = TeamLoader()
    all_teams = loader.get_all_teams()

    # --- Check 1: rust integration smoke — battles finish, no TRANSPORT error ---
    print(f"[parity] rust integration smoke: up to {n_battles} battles …")
    r_wins, r_fin, r_skip, r_terr, r_to = _play_series("rust", n_battles, all_teams, "Rs")
    print(f"[parity] rust: {r_fin} finished, {r_skip} skipped (unmodeled move), "
          f"{r_to} timed out, {r_terr} transport errors (p1-wins {r_wins})")
    if r_terr > 0:
        raise AssertionError(
            f"rust bridge integration FAILED: {r_terr} TRANSPORT errors (a plumbing / framing / "
            f"lifecycle / parse bug — NOT an unmodeled-move skip).")
    _starvation_verdict("rust smoke", r_to, n_battles)
    if r_fin == 0:
        raise AssertionError(
            f"rust bridge integration INCONCLUSIVE: 0/{n_battles} battles completed "
            f"({r_skip} hit unmodeled moves, {r_to} timed out). Point at a modeled-team pool to "
            f"exercise it. {describe_contention()}")
    print(f"✅ [parity] rust smoke: {r_fin} battle(s) played to completion with no transport "
          f"error (the move-name/switch-species transport path works end-to-end).")

    # --- Check 2: statistical win-rate parity at seed=None (the training regime) ---
    print(f"[parity] win-rate parity: up to {n_battles} battles/impl at seed=None …")
    n_wins, n_fin, n_skip, n_terr, n_to = _play_series("node", n_battles, all_teams, "Ns")
    r2_wins, r2_fin, r2_skip, r2_terr, r2_to = _play_series("rust", n_battles, all_teams, "Rp")
    if n_terr or r2_terr:
        raise AssertionError(
            f"transport error during parity (node {n_terr}, rust {r2_terr}) — investigate.")
    # Before comparing win rates: a starved arm's surviving battles are a biased sample (the
    # ones that happened to get scheduled), and comparing it against the other arm measures the
    # scheduler. Refuse rather than report.
    _starvation_verdict("node parity arm", n_to, n_battles)
    _starvation_verdict("rust parity arm", r2_to, n_battles)
    if n_fin == 0 or r2_fin == 0:
        raise AssertionError("parity INCONCLUSIVE: no completed battles on one side.")
    node_wr = n_wins / n_fin
    rust_wr = r2_wins / r2_fin
    print(f"[parity] node  p1 win-rate: {node_wr:.3f} ({n_wins}/{n_fin}, {n_skip} skipped, "
          f"{n_to} timed out)")
    print(f"[parity] rust  p1 win-rate: {rust_wr:.3f} ({r2_wins}/{r2_fin}, {r2_skip} skipped, "
          f"{r2_to} timed out)")

    # Tolerance: sampling noise on the completed binomials. Use max(tol, 2·pooled-SE) so a small
    # completed-N is judged on its actual variance — both play the SAME distribution, so a real
    # transport bug shifts the win-rate far beyond noise.
    p = (n_wins + r2_wins) / (n_fin + r2_fin)
    se = math.sqrt(p * (1 - p) * (1 / n_fin + 1 / r2_fin)) if 0 < p < 1 else 0.0
    bound = max(tol, 2.0 * se)
    delta = abs(node_wr - rust_wr)
    if delta > bound:
        raise AssertionError(
            f"rust vs node win-rate parity FAILED: |{node_wr:.3f} − {rust_wr:.3f}| = "
            f"{delta:.3f} > tolerance {bound:.3f} (2·SE={2 * se:.3f}). A real transport "
            f"divergence, not sampling noise — investigate.")
    print(f"✅ [parity] win-rate parity: |Δ|={delta:.3f} ≤ tolerance {bound:.3f} "
          f"(2·SE={2 * se:.3f}) — node ≈ rust at seed=None over the completed battles.")
    print("✅ [parity] rust bridge is a correct drop-in for the RL transport over its modeled "
          "move universe (exact per-battle byte-parity is proven separately by "
          "src/rust_sim/harness/gen_sim_bridge_diff.js).")


# --- pytest entry (skipped unless a pre-built rust binary is available) ---------------------

def _prebuilt_rust_available() -> bool:
    if os.environ.get("POKESIM_SIM_BRIDGE_BIN"):
        return True
    shared = Path(__file__).resolve().parents[2] / "rust_sim" / "target" / "release" / "sim_bridge"
    return shared.is_file()


import pytest

# gen3 test tiers (MEASURED 2026-08-14): 72.6 s across 14 tests, and the most CONTENTION-FRAGILE
# test in the tree — it plays 12 battles under per-battle timeouts, so a busy box turns it into a
# wall of TIMEOUTs rather than a result (observed twice, both times a fresh worktree paying for a
# cargo release build). `slow` is the honest statement of that: this is the rust/node STABILITY
# gate, run deliberately — before a ship, in CI, or when touching the bridge — not on every edit.
pytestmark = [pytest.mark.sim, pytest.mark.slow]


@pytest.mark.integration
def test_rust_node_bridge_parity():
    # Marked `integration`: it spawns node + rust bridge subprocesses and plays real battles, so
    # the default `-m "not integration"` unit run excludes it (and it's ALSO skipped without a
    # pre-built rust binary). Run it via the full suite or as a script.
    if not _prebuilt_rust_available():
        pytest.skip(
            "no pre-built rust sim_bridge binary (set POKESIM_SIM_BRIDGE_BIN or build "
            "src/rust_sim); skipping to avoid a cargo build in the unit suite")
    # A modest N keeps the unit-suite cost bounded (each battle spawns a fresh sim subprocess ~5s,
    # and an unmodeled-move hang costs up to _PARITY_BATTLE_IDLE_BUDGET); the tolerance widens to 2·SE
    # for the small completed-N. The script default plays more for a tighter statistical bound.
    run(n_battles=12)


# --- SEED handling parity (gen3_bridge_seedless_fixed_seed_v1 / gen3_bridge_seed_forms_v1) ---
#
# The four seed defects all shipped because every gate on this path — including check 2 above,
# which deliberately runs at seed=None but only compares WIN RATES — passed an array seed or
# looked at aggregates. These pin the two things the aggregates cannot see: that a SEEDLESS rust
# battle is a NEW battle each time, and that every seed SPELLING means the same battle on both
# impls (so a node-recorded battle replayed on rust is the same battle).

def _fixed_pair(team: str):
    """Two RandomPlayers on ONE fixed team — with `random.seed` pinned per battle, the whole
    trajectory is a function of the sim's dice alone, so a protocol digest isolates the seed.

    The usernames are CONSTANT on purpose: they are echoed in the `|player|` protocol lines,
    so a per-impl tag would make every digest differ for a reason that has nothing to do with
    the seed."""
    from poke_env.player import RandomPlayer
    return (
        RandomPlayer(battle_format=BATTLE_FORMAT, team=team, start_listening=False,
                     account_configuration=AccountConfiguration("SeedParA", "password")),
        RandomPlayer(battle_format=BATTLE_FORMAT, team=team, start_listening=False,
                     account_configuration=AccountConfiguration("SeedParB", "password")),
    )


def _battle_digest(impl: str, seed, team: str, py_seed: int = 7):
    """Play ONE battle and return (protocol digest, had_recon). The digest strips `|t:|`
    wall-clock stamps (emission-time, in poke-env's MESSAGES_TO_IGNORE — invisible to state)."""
    import hashlib
    import random
    from utils.bridge.reconstruction import pop_record
    random.seed(py_seed)
    p1, p2 = _fixed_pair(team)
    sink: list = []
    asyncio.run(run_local_battles(p1, p2, 1, battle_format=BATTLE_FORMAT, seed=seed,
                                  chunk_sink=sink, impl=impl))
    body = "\n".join(
        f"{side}|" + "\n".join(l for l in chunk.split("\n") if not l.startswith("|t:|"))
        for side, chunk in sink
    )
    battle = next(iter(p1._battles.values()))
    # Anti-vacuity: a digest over an empty/aborted stream would make every comparison below
    # pass for the wrong reason.
    assert battle.finished and battle.turn >= 5 and len(sink) >= 20, (
        f"{impl}/seed={seed!r}: degenerate battle (finished={battle.finished}, "
        f"turn={battle.turn}, chunks={len(sink)}) — the digest would be meaningless")
    return hashlib.sha256(body.encode()).hexdigest(), pop_record(battle.battle_tag) is not None


@pytest.mark.integration
def test_seedless_rust_battles_are_distinct_and_recorded():
    """B1/B2: a seedless START must MINT a seed (not reuse a constant), and therefore still
    emit `__RECON__`. Pre-fix, three seedless rust battles hashed IDENTICALLY and none carried
    a record — i.e. every training episode and eval game replayed one dice stream."""
    if not _prebuilt_rust_available():
        pytest.skip("no pre-built rust sim_bridge binary")
    team = TeamLoader().get_all_teams()[0]
    seen = set()
    for i in range(3):
        digest, had_recon = _battle_digest("rust", None, team, py_seed=7)
        assert had_recon, "a seedless rust battle emitted no __RECON__ (forensics go dark)"
        seen.add(digest)
    assert len(seen) == 3, (
        f"{len(seen)} distinct battles out of 3 SEEDLESS rust battles — the child is replaying "
        f"one fixed dice stream (gen3_bridge_seedless_fixed_seed_v1)")


@pytest.mark.integration
@pytest.mark.parametrize("seed", [
    [1, 2, 3, 4],            # the JSON array form (the only one the old parser read)
    "1,2,3,4",               # the STRING form `new PRNG()` requires for a re-roll
    "gen5,0001000200030004",  # the same seed, hex-packed
    "sodium,deadbeef",       # the ChaCha20 backend (Showdown's default)
])
def test_seed_forms_reproduce_the_same_battle_on_rust_and_node(seed):
    """B3: every seed spelling the Node bridge accepts must run the SAME battle on rust.
    Pre-fix, the two STRING forms were silently dropped by rust's array-only parser and ran a
    different battle than the caller asked for — so replaying a node-recorded battle on rust
    quietly answered a different question."""
    if not _prebuilt_rust_available():
        pytest.skip("no pre-built rust sim_bridge binary")
    team = TeamLoader().get_all_teams()[0]
    node_digest, _ = _battle_digest("node", seed, team)
    rust_digest, _ = _battle_digest("rust", seed, team)
    assert node_digest == rust_digest, (
        f"seed {seed!r}: rust ran a DIFFERENT battle than node "
        f"({rust_digest[:16]} vs {node_digest[:16]})")


# --- CONTENTION: a timeout is never a semantic outcome (gen3_contention_robust_timeouts_v1) ---
#
# These are pure (no bridge, no marker) so they run in the DEFAULT unit gate — the gate that
# excludes the integration test they protect. That is deliberate: the starved-run artifact these
# pin was invisible precisely because nothing cheap ever checked this file's classification.


def test_timeouts_do_not_inflate_the_unmodeled_skip_count():
    """The regression itself. `skipped` means 'the port cannot play this move'; a timeout means
    'the box did not get around to it'. Conflating them let 39/40 starved battles be reported as
    a move-coverage gap."""
    import inspect
    src = inspect.getsource(_play_series)
    handler = src.split("except (asyncio.TimeoutError, TimeoutError):")[1]
    assert "timed_out += 1" in handler
    assert "skipped += 1" not in handler, (
        "the timeout handler must not increment the unmodeled-move skip counter — that is the "
        "conflation that made a CPU-starved run look like a clean pass")


def test_starvation_verdict_is_silent_when_nothing_timed_out():
    _starvation_verdict("arm", timed_out=0, attempted=60)  # must not raise


def test_starvation_verdict_tolerates_an_isolated_timeout():
    """One wedge in 60 is a real signal worth printing, not grounds to void the run."""
    _starvation_verdict("arm", timed_out=1, attempted=60)  # must not raise


def test_starvation_verdict_refuses_a_starved_sample():
    """The 39/40 case must now REFUSE rather than report a parity verdict computed from the
    single battle that happened to survive."""
    with pytest.raises(AssertionError, match="INCONCLUSIVE") as ei:
        _starvation_verdict("rust parity arm", timed_out=39, attempted=40)
    msg = str(ei.value)
    assert "about the BOX" in msg, "must name the cause, not leave the reader guessing"
    assert "load average" in msg, "must carry the self-diagnosis"


def test_starvation_threshold_is_a_fraction_not_a_count():
    """Scaling with the attempted count is what keeps the rule meaningful at both n=12 (the
    pytest entry) and n=60 (the script default).

    ⚠️ The label says SYNTHETIC because this test PRINTS. Under `-s` its warning lands in the same
    stream as the real series lines, in the same format, and reads as a measurement of the run —
    it cost a real investigation on 2026-08-15, where `⚠️ small run: 4/12 battles TIMED OUT (33%)`
    was taken for a live starvation reading while every actual series that run reported 0. A
    diagnostic that cannot be told apart from a measurement is a bad diagnostic.
    """
    _starvation_verdict("SYNTHETIC unit-test sample", timed_out=2, attempted=12)   # 17% — under
    with pytest.raises(AssertionError):
        _starvation_verdict("SYNTHETIC unit-test sample", timed_out=4, attempted=12)  # 33% — over


def test_parity_battle_timeout_is_scaled_by_contention(monkeypatch):
    """The TOTAL backstop must still go through the contention scaler.

    It is no longer this file's primary bound — that is now the IDLE gap (`_BATTLE_IDLE_BUDGET`),
    because a total cap cannot tell a starved battle from a wedged one. But the backstop survives
    for livelock, so it must keep scaling: unscaled, 20 s is under two seconds of real work on a
    loaded box.
    """
    from utils import contention

    saved = local_battle_runner._PER_BATTLE_TIMEOUT
    try:
        local_battle_runner._PER_BATTLE_TIMEOUT = 20.0
        monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
        contention._cached = None
        assert local_battle_runner._per_battle_timeout() == pytest.approx(20.0)

        monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "3")
        contention._cached = None
        assert local_battle_runner._per_battle_timeout() == pytest.approx(60.0)
    finally:
        local_battle_runner._PER_BATTLE_TIMEOUT = saved
        contention._cached = None


# ===========================================================================================
# CHOOSE-path parity — the fatal-`__ERR__` class (`gen3_bridge_choose_path_parity_v1`).
#
# An `__ERR__` frame is NOT a recoverable in-band error: `BridgeSession._dispatch` raises on
# it, `_persistent_read_loop` retires the reader and calls `_signal_transport_dead()`, and every
# in-flight `step()` raises `ShowdownException: Showdown websocket dropped …`. One `__ERR__`
# therefore kills a whole training run. So ANY CHOOSE the Node bridge tolerates, the Rust bridge
# must tolerate too — this is the gate on that, driven at the RAW protocol level (no poke-env,
# no battle driver) so it is deterministic and runs in seconds.
#
# Two real production failures live here, both `--use-bridge=rust`-only:
#   1. `CHOOSE <side> default` / `pass` — poke-env's `DefaultBattleOrder` / `PassBattleOrder`.
#      `Player._handle_battle_request` (`player.py:384`) sends `/choose default` after a
#      rejected request with probability `DEFAULT_CHOICE_CHANCE = 1/1000`, so this fires as a
#      RATE, independent of box load — matching the observed ~8-minute crash at load 5 and at
#      load 31 alike. Showdown's `Side.choose` accepts both tokens; `bridge.rs::parse_choice`
#      accepted only `move `/`switch `.
#   2. A stray CHOOSE after `__END__` on a PERSISTENT child — the child resets itself at
#      `__END__`, and `BridgeSession._dispatch` fires poke-env's feeds as UN-AWAITED tasks, so a
#      late answer to the ending battle's last `|request|` arrives with no battle live. Node
#      ignores it (`local_sim_bridge.js`: `if (streams && streams[side])`); Rust flushed anyway
#      and returned `no battle in progress (missing START)`.
# ===========================================================================================

_BOTH_IMPLS = ["node", "rust"]


class _RawChild:
    """A raw pipe to one sim_bridge child — the line protocol and nothing else.

    ONE background reader thread feeds a queue. (A per-call reader thread abandoned on timeout
    stays blocked in `read` and swallows the NEXT line, which silently turns a real `__ERR__`
    into an apparent success — that bug cost a full debugging cycle while writing this.)
    """

    def __init__(self, impl: str):
        import queue as _queue
        import subprocess
        import threading

        from utils.bridge.sim_bridge_bin import bridge_spawn_argv

        self.proc = subprocess.Popen(
            bridge_spawn_argv(impl),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._lines: "_queue.Queue[str]" = _queue.Queue()
        self._empty = _queue.Empty

        def pump():
            for raw in self.proc.stdout:
                self._lines.put(raw.decode().rstrip("\n"))
            self._lines.put("")  # EOF sentinel

        threading.Thread(target=pump, daemon=True).start()
        threading.Thread(target=lambda: [None for _ in self.proc.stderr], daemon=True).start()

    def send(self, line: str) -> None:
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def readline(self, timeout: float):
        """One stdout line, ``""`` on EOF, or ``None`` if the child stayed silent."""
        try:
            return self._lines.get(timeout=timeout)
        except self._empty:
            return None

    def drain(self, quiet_for: float = 3.0):
        """Every frame the child has queued, read until it goes quiet."""
        out = []
        while True:
            line = self.readline(timeout=quiet_for)
            if line is None:
                return out
            out.append(line)
            if line == "":
                return out

    def close(self) -> None:
        try:
            self.send("END")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _decode_frame(frame: str) -> str:
    import base64
    return base64.b64decode(frame.split(" ", 1)[1]).decode()


def _err_text(frame: str) -> str:
    return _decode_frame(frame)


def _start_json(teams) -> str:
    import json
    tb = Gen3Teambuilder(teams)
    return json.dumps({
        "formatid": BATTLE_FORMAT,
        "persistent": True,
        "p1": {"name": "P1", "team": tb.yield_team()},
        "p2": {"name": "P2", "team": tb.yield_team()},
    })


def _choice_for(chunk: str):
    """A legal choice token for a `|request|` chunk, or None if the side owes nothing."""
    import json
    for line in chunk.split("\n"):
        if not line.startswith("|request|") or len(line) <= len("|request|"):
            continue
        req = json.loads(line[len("|request|"):])
        if req.get("wait"):
            return None
        mons = (req.get("side") or {}).get("pokemon") or []
        if req.get("forceSwitch"):
            for i, mon in enumerate(mons, start=1):
                if not mon.get("active") and "0 fnt" not in (mon.get("condition") or ""):
                    return f"switch {i}"
            return None
        active = (req.get("active") or [None])[0]
        if active:
            for i, mv in enumerate(active.get("moves", []), start=1):
                if not mv.get("disabled") and mv.get("pp", 1) > 0:
                    return f"move {i}"
            return "move 1"
    return None


@pytest.mark.integration
@pytest.mark.parametrize("impl", _BOTH_IMPLS)
@pytest.mark.parametrize("token", ["default", "pass"])
def test_poke_env_fallback_choice_tokens_never_produce_a_fatal_err(impl, token):
    """`CHOOSE <side> default` / `pass` must NEVER come back as `__ERR__` on either impl.

    These are tokens poke-env really sends (`DefaultBattleOrder` / `PassBattleOrder`), and
    Showdown's `Side.choose` accepts both. An in-band `|error|` frame is a fine answer — that is
    recoverable and is what Node does for an illegal `pass`. An out-of-band `__ERR__` is not: it
    kills the reader and the run.
    """
    if impl == "rust" and not _prebuilt_rust_available():
        pytest.skip("no pre-built rust sim_bridge binary")
    child = _RawChild(impl)
    try:
        child.send("START " + _start_json(TeamLoader().get_all_teams()))
        opening = child.drain()
        assert any(_choice_for(_decode_frame(f)) for f in opening if not f.startswith("__")), (
            f"{impl}: no opening |request| to answer — the fixture, not the bridge, is broken")

        child.send(f"CHOOSE p1 {token}")
        for frame in child.drain(quiet_for=5.0):
            assert not frame.startswith("__ERR__"), (
                f"{impl}: `CHOOSE p1 {token}` produced a FATAL __ERR__ "
                f"({_err_text(frame)!r}). poke-env sends this token "
                f"({'player.py:384, p=1/1000 after a rejected request' if token == 'default' else 'PassBattleOrder'}), "
                f"and an __ERR__ retires BridgeSession's reader → _signal_transport_dead() → "
                f"ShowdownException in every in-flight step(). Node accepts it; so must rust.")
            assert frame != "", f"{impl}: the child EXITED on `CHOOSE p1 {token}`"
    finally:
        child.close()


@pytest.mark.integration
@pytest.mark.parametrize("impl", _BOTH_IMPLS)
@pytest.mark.parametrize("end_via", ["natural", "forfeit"])
def test_stray_choose_after_battle_end_is_ignored_on_a_persistent_child(impl, end_via):
    """A CHOOSE arriving after `__END__` on a persistent child must be a silent NO-OP.

    The child resets itself at `__END__`, but `BridgeSession._dispatch` fires poke-env's feeds
    as un-awaited tasks, so a late answer to the ending battle's last `|request|` routinely
    lands with no battle live and the old battle tag still registered. Node's
    `if (streams && streams[side])` drops it. Rust used to fall through to `flush_new_chunks`
    and return `no battle in progress (missing START)` → `__ERR__` → dead run.
    """
    if impl == "rust" and not _prebuilt_rust_available():
        pytest.skip("no pre-built rust sim_bridge binary")
    child = _RawChild(impl)
    try:
        child.send("START " + _start_json(TeamLoader().get_all_teams()))
        decisions = 0
        forfeited = False
        while True:
            line = child.readline(timeout=30.0)
            assert line is not None, f"{impl}: child went silent before __END__"
            assert line != "", f"{impl}: child exited before __END__"
            if line == "__END__":
                break
            assert not line.startswith("__ERR__"), \
                f"{impl}: unexpected __ERR__ mid-battle: {_err_text(line)!r}"
            if line.startswith("__RECON__"):
                continue
            side, choice = line.split(" ", 1)[0], _choice_for(_decode_frame(line))
            if choice is None:
                continue
            decisions += 1
            assert decisions < 4000, f"{impl}: battle never ended (driver fixture bug)"
            if end_via == "forfeit" and not forfeited and decisions >= 6 and side == "p2":
                forfeited = True
                child.send("FORCELOSE p1")   # the training seam's reset-mid-battle path
                continue
            child.send(f"CHOOSE {side} {choice}")

        child.send("CHOOSE p2 move 1")       # the stray late answer
        late = child.drain(quiet_for=3.0)
        assert late == [], (
            f"{impl}/{end_via}: a stray CHOOSE after __END__ produced "
            f"{[_err_text(f) if f.startswith('__ERR__') else f[:40] for f in late]} "
            f"instead of being ignored. On the rust impl this was the ~8-minute "
            f"`--use-bridge=rust` training crash.")
        assert child.proc.poll() is None, f"{impl}: the child exited on a stray CHOOSE"
    finally:
        child.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(n)
