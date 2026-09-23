"""Phase-0 (d): LOC of every link in the Python re-derivation chain, grouped the way the README's
inventory table reads. Counts NON-TEST source lines (`wc -l` semantics, whole file) at HEAD.

    python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/inventory_loc.py

The groups are the inventory's rows; each names the files it counts so a later run on a later
tree is the same measurement.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")

GROUPS = {
    "1 poke-env battle-state parse (vendored fork, battle/ only)": [
        "poke_env/battle/abstract_battle.py", "poke_env/battle/battle.py",
        "poke_env/battle/pokemon.py", "poke_env/battle/move.py", "poke_env/battle/effect.py",
        "poke_env/battle/field.py", "poke_env/battle/side_condition.py",
        "poke_env/battle/weather.py", "poke_env/battle/status.py",
        "poke_env/battle/pokemon_type.py"],
    "2 event layer (Gen3Battle + schema)": [
        "agents/battle/gen3_battle.py", "agents/battle/battle_event.py"],
    "3 read-models (LiveView/TurnView/LegalActions/StrictBattleView)": [
        "agents/battle/live_view.py", "agents/battle/turn_view.py",
        "agents/battle/strict_view.py"],
    "4 second constructors: view adapter + light event fold": [
        "agents/battle/view_adapter.py", "agents/battle/event_fold.py"],
    "5 per-decision folds + trackers": [
        "agents/training/turn_delta.py", "agents/training/battle_snapshot.py",
        "agents/training/episode_tracker.py", "agents/training/choice_band_tracker.py",
        "agents/training/hidden_power_tracker.py", "agents/training/reward_tracker.py",
        "agents/training/clone_pins.py", "agents/observation/wish_belief.py",
        "agents/observation/sleep_belief.py"],
    "6 encoder (obs)": [
        "agents/observation/state_encoder.py", "agents/observation/assembler.py",
        "agents/observation/base.py", "agents/observation/constants.py",
        "agents/observation/schema.py", "agents/observation/pokemon.py",
        "agents/observation/moves.py", "agents/observation/items.py",
        "agents/observation/abilities.py", "agents/observation/types.py",
        "agents/observation/species.py", "agents/observation/global_env.py",
        "agents/observation/active_context.py", "agents/observation/reactive.py",
        "agents/observation/incoming_damage.py", "agents/observation/incoming_damage_encoder.py",
        "agents/observation/turn_delta_encoder.py", "agents/observation/gen3_effects.py",
        "agents/observation/true_team.py", "agents/observation/belief_labels.py"],
    "7 legality / action (mask, mapper, serialize)": [
        "agents/action/mask_generator.py", "agents/action/mapper.py",
        "agents/action/serialize.py", "agents/action/choice.py",
        "agents/action/ordering_integrity.py", "agents/action/constants.py"],
    "8 transport: bridge client + node drivers": [
        "utils/bridge/local_battle_runner.py", "utils/bridge/bridge_session.py",
        "utils/bridge/battle_stream_client.py", "utils/bridge/sim_bridge_bin.py",
        "utils/bridge/local_sim_bridge.js", "utils/bridge/search_driver.js",
        "utils/bridge/replay_driver.js", "utils/bridge/replay_kernels.js",
        "utils/bridge/search_session.py", "utils/bridge/reconstruction.py"],
    "9 search materializers (protocol road + view road)": [
        "agents/training/obs_materializer.py", "agents/training/view_successor.py"],
    "10 env process layer (env, wrappers, vec-env, forkserver/compile preload)": [
        "agents/training/gen3_env.py", "agents/training/wrappers.py",
        "agents/training/async_vec_env.py", "main/train/env_factory.py",
        "agents/model/compile_opponents.py", "agents/model/compile_preload.py",
        "agents/model/compile_prewarm.py", "agents/training/snapshot_pool.py"],
    "11 player-side inference (per-env opponent forward)": ["agents/inference/player.py"],
}

RUST = {
    "rust port (src/rust_sim/src, all .rs)": "rust_sim/src",
}


def loc(rel: str) -> int:
    p = os.path.join(SRC, rel)
    if not os.path.exists(p):
        return -1
    with open(p, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def main() -> int:
    total = 0
    missing = []
    for g, files in GROUPS.items():
        n = 0
        for f in files:
            x = loc(f)
            if x < 0:
                missing.append(f)
            else:
                n += x
        total += n
        print(f"{n:>7}  {g}  ({len(files)} files)")
    print(f"{total:>7}  TOTAL python/js chain")
    for g, d in RUST.items():
        n = 0
        for dp, _dn, fns in os.walk(os.path.join(SRC, d)):
            for fn in fns:
                if fn.endswith(".rs"):
                    with open(os.path.join(dp, fn)) as fh:
                        n += sum(1 for _ in fh)
        print(f"{n:>7}  {g}")
    if missing:
        print("MISSING (not counted):", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
