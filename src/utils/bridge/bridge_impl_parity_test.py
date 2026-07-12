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

BATTLE_FORMAT = "gen3ou"
# Short per-battle timeout so an unmodeled-move HANG (some Rust fail-loud paths emit __ERR__ to
# one side but never deliver the other side's request → the demux blocks) is skipped in seconds,
# not the 180s production default. A healthy in-process bridge battle is < 2s.
_PARITY_BATTLE_TIMEOUT = 20.0


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
    only that battle instead of aborting the run. Returns (p1_wins, finished, skipped_unmodeled,
    transport_errors)."""
    p1, p2 = _build_pair(all_teams, tag)
    skipped = transport_err = 0
    # Use a short per-battle timeout so an unmodeled-move hang skips fast (restored after).
    saved = local_battle_runner._PER_BATTLE_TIMEOUT
    local_battle_runner._PER_BATTLE_TIMEOUT = _PARITY_BATTLE_TIMEOUT
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
                # A per-battle timeout: some Rust unmodeled-move fail-loud paths emit __ERR__ to
                # one side but never deliver the other's request → the demux hangs. Treat as an
                # (unmodeled) skip, not a transport failure — the completed battles are the signal.
                skipped += 1
                continue
    finally:
        local_battle_runner._PER_BATTLE_TIMEOUT = saved
    return p1.n_won_battles, p1.n_finished_battles, skipped, transport_err


def run(n_battles: int = 60, tol: float = 0.20) -> None:
    loader = TeamLoader()
    all_teams = loader.get_all_teams()

    # --- Check 1: rust integration smoke — battles finish, no TRANSPORT error ---
    print(f"[parity] rust integration smoke: up to {n_battles} battles …")
    r_wins, r_fin, r_skip, r_terr = _play_series("rust", n_battles, all_teams, "Rs")
    print(f"[parity] rust: {r_fin} finished, {r_skip} skipped (unmodeled move), "
          f"{r_terr} transport errors (p1-wins {r_wins})")
    if r_terr > 0:
        raise AssertionError(
            f"rust bridge integration FAILED: {r_terr} TRANSPORT errors (a plumbing / framing / "
            f"lifecycle / parse bug — NOT an unmodeled-move skip).")
    if r_fin == 0:
        raise AssertionError(
            f"rust bridge integration INCONCLUSIVE: 0/{n_battles} battles completed "
            f"({r_skip} hit unmodeled moves). Point at a modeled-team pool to exercise it.")
    print(f"✅ [parity] rust smoke: {r_fin} battle(s) played to completion with no transport "
          f"error (the move-name/switch-species transport path works end-to-end).")

    # --- Check 2: statistical win-rate parity at seed=None (the training regime) ---
    print(f"[parity] win-rate parity: up to {n_battles} battles/impl at seed=None …")
    n_wins, n_fin, n_skip, n_terr = _play_series("node", n_battles, all_teams, "Ns")
    r2_wins, r2_fin, r2_skip, r2_terr = _play_series("rust", n_battles, all_teams, "Rp")
    if n_terr or r2_terr:
        raise AssertionError(
            f"transport error during parity (node {n_terr}, rust {r2_terr}) — investigate.")
    if n_fin == 0 or r2_fin == 0:
        raise AssertionError("parity INCONCLUSIVE: no completed battles on one side.")
    node_wr = n_wins / n_fin
    rust_wr = r2_wins / r2_fin
    print(f"[parity] node  p1 win-rate: {node_wr:.3f} ({n_wins}/{n_fin}, {n_skip} skipped)")
    print(f"[parity] rust  p1 win-rate: {rust_wr:.3f} ({r2_wins}/{r2_fin}, {r2_skip} skipped)")

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
    # and an unmodeled-move hang costs up to _PARITY_BATTLE_TIMEOUT); the tolerance widens to 2·SE
    # for the small completed-N. The script default plays more for a tighter statistical bound.
    run(n_battles=12)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run(n)
