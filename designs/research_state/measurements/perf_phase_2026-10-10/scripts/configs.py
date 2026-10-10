"""The configurations the perf-phase GPU window measures (2026-10-10). Every argv carries the closing test's ride-along
flags (design_endstate_closing_test.md §4), so the eager tail is the one the live chain trains.

    python configs.py --check          # resolve every argv like a launch (CPU, no GPU), print the levers that differ
    python configs.py --argv E         # print one argv (for the window's driver)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RIDE = "--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
X = "--allow-nonproduction-arch"

#: The COST ATTRIBUTION ladder (orchestrator, 2026-10-10): production -> static -> +facts -> +depth -> the end state,
#: and the exact-KO cost inside the end state.
CONFIGS = {
    "P": f"--arch production {RIDE}",
    "S": f"--arch production --token-encoding static {X} {RIDE}",                 # static tokens only, depth 2
    "SF": f"--arch static_recovery --trunk-layers 2 {X} {RIDE}",                  # + the recovery facts, depth 2
    "R": f"--arch static_recovery {RIDE}",                                        # + the third trunk round
    "E": f"--arch endstate {RIDE}",                                               # + the rest of the end state
    "EK": f"--arch endstate --ko-ramp ramp {X} {RIDE}",                           # the end state minus exact KO
}

KEYS = ("token_encoding", "trunk_layers", "mon_hazard_cost", "move_actor_state", "switch_hazard_cost",
        "eot_residual", "move_resolution", "speed_physics", "value_threat_inject", "op_reduction", "obs_facts",
        "ko_ramp", "move_set_closure", "batch_size", "grad_accum_steps", "n_epochs", "grad_checkpointing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--argv")
    a = ap.parse_args()
    if a.argv:
        print(CONFIGS[a.argv])
        return
    import shlex
    from update_bench import resolved_args
    for tag, argv in CONFIGS.items():
        ns = resolved_args(shlex.split(argv), "cpu")
        print(tag, {k: getattr(ns, k, "?") for k in KEYS})


if __name__ == "__main__":
    main()
