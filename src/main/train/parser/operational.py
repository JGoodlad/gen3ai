"""The `# --- Operational Flags ---` section: what to run, where to write it, and the
`--debug` smoke.

Lifted VERBATIM out of the old single-file `parser.py` (lines 108-178); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.constants import SMOKE_EVAL_BATTLES, SMOKE_STEPS
from main.train.parser.base import BoolFlag


def add_operational_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--run-dir", type=str, help="Run folder to write checkpoints into (set by launcher on resume)")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None,
                        help="A MEMORABLE name for a fresh run → writes to models/<name>/ instead of "
                             "a date-stamped models/run_<timestamp>/. Must be a single name "
                             "(letters/digits/._-, no slashes). Refuses to overwrite an existing run "
                             "of that name (pick another, or --model to resume it). Ignored when "
                             "--run-dir is set (launcher resume). For --exploiter, defaults to "
                             "'exploiter_vs_<target>' if you don't name it.")
    parser.add_argument("--eval-only", action=BoolFlag, default=False, help="Skip training and only evaluate")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action=BoolFlag, default=False, help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--debug-eval", "--debug_eval", dest="debug_eval", action=BoolFlag, default=False,
                        help="Run evaluation under --debug. By default a --debug smoke run skips ALL eval "
                             "(both the periodic eval callback AND the final win-rate eval) so it needs no "
                             "eval opponents / Showdown eval connection and stays light on CPU. Pass "
                             "--debug-eval to exercise the eval pipeline in a smoke run. No effect on real "
                             "(non-debug) runs, which always eval.")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--async-rollout", "--async_rollout", dest="async_rollout",
                        action=BoolFlag, default=False,
                        help="Non-barrier async rollout collection: keep every env worker "
                             "continuously in-flight and forward whichever are ready, instead of "
                             "barriering on the slowest env each step (AsyncSubprocVecEnv + an "
                             "on-policy async collect_rollouts that overlaps the GPU forward with "
                             "CPU env-stepping). Off by default; ignored under --debug. With async, "
                             "right-size --n-envs nearer the core count (16) rather than oversubscribing.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)")
    parser.add_argument("--showdown-port", type=int, default=None,
                        help="Local Showdown server port (default 8000). Sets the port for the trainee, "
                             "eval, and self-play clients. Start the server on the matching port, "
                             "e.g. npm run showdown -- <port>.")
    parser.add_argument("--use-bridge", type=str, default="rust",
                        choices=["off", "node", "rust"],
                        help="In-process BattleStream bridge transport for BOTH training AND eval "
                             "(no Showdown server, no port, no /challenge storm, deterministic). "
                             "'rust' (DEFAULT) = the byte-compatible src/rust_sim sim_bridge binary "
                             "(built via cargo; override with POKESIM_SIM_BRIDGE_BIN) — measured "
                             "1.41x node's throughput at --n-envs 48 with a ~25x smaller child "
                             "(9 MB RSS vs ~224 MB). 'node' = the Node local_sim_bridge.js, kept as "
                             "the explicit A/B arm and for the parity harness. 'off' = the websocket "
                             "transport, which needs a running Showdown server on --showdown-port. "
                             "NOTE: 'rust' now emits __RECON__ (gen3_bridge_recon_record_v1, on a "
                             "seedless battle too) and supports resumeReseed "
                             "(gen3_bridge_resume_reseed_v1), so the forensic reconstruction and "
                             "counterfactual paths work on rust. The OFFLINE search/replay drivers "
                             "are on rust too (gen3_rust_search_driver_v1 / "
                             "gen3_rust_replay_driver_v1 — one search_driver binary serves both "
                             "verb families), so --search-teacher no longer requires 'node'; the "
                             "run's impl is threaded into the teacher workers. 'rust' also "
                             "fail-louds on an unmodeled move.")
    parser.add_argument(
        "--self-play-use-cpu",
        action=BoolFlag,
        default=True,
        help="Load self-play opponent snapshots on CPU instead of the training device. "
             "Default True: avoids one CUDA context per SubprocVecEnv worker (~300-600 MB each), "
             "which would otherwise OOM the GPU at high --n-envs. Opponent inference is batch-1 "
             "no_grad, so CPU is plenty fast. Pass --no-self-play-use-cpu to load them on --device.",
    )
    parser.add_argument("--eval-battles", type=int, default=None,
                        help="Battles per FINAL-evaluation opponent. Default 100, but AUTO-SCALED "
                             f"down to {SMOKE_EVAL_BATTLES} when --steps < {SMOKE_STEPS:,} (a smoke "
                             "run), because a 9-opponent x 100-battle final eval costs many minutes "
                             "and a 2k-step policy produces no signal worth that. An explicit value "
                             "always wins.")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")
