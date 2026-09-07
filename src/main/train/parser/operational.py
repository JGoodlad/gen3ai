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
    parser.add_argument(
        "--tb-inherit",
        action=BoolFlag,
        default=True,
        help="gen3_tb_inherit_v1 — on a FORK, copy the parent's SCALAR TensorBoard events at steps "
             "<= fork_step into this run's tb/, so its charts read as one continuous curve from "
             "step 0 instead of starting mid-air at fork_step. A fork's global step continues the "
             "parent's counter and TensorBoard merges every event file in a run dir by step, so "
             "this is pure bookkeeping over points the parent already logged — nothing is "
             "recomputed. Truncated at fork_step (the parent usually trained past the fork), "
             "scalars only, idempotent via tb/INHERITED_FROM.json (a launcher restart never "
             "re-copies), and a no-op on a fresh run or a same-run restart. Costs a few hundred KB. "
             "Pass --no-tb-inherit to opt out — worth doing for a large sibling FLEET under an "
             "UNCURATED logdir, where 8 exploiters off one target then draw 8 identical prefixes "
             "in every chart (under `main.tb_curate` this is exactly what you want).")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")
    # --- gen3_arch_surface_guard_v1 (2026-09-06) — THE ARCH SURFACE ---------------------------
    # "it launches" and "it is the experiment" are INDEPENDENT checks. A validator that only ever
    # answered the first let a 38-token argv train a near-bare architecture for ~7 GPU-hours /
    # 24.4M steps (31 keys off production). These two flags are the answer to the OTHER question.
    parser.add_argument("--arch", type=str, default=None, choices=["production"],
                        help="Apply an ENTIRE architecture surface as if every flag had been "
                             "typed. 'production' reads designs/production_config.json — the same "
                             "mirror the generated ARCHITECTURE.md tables and the compile gate key "
                             "on — and sets every structural toggle in it that this argv leaves "
                             "unset, so an explicitly-typed flag still wins. It does NOT set the "
                             "training coefficients (the belief-supervision doses) or the CRITIC "
                             "readouts: those are what an experiment varies, and the startup block "
                             "lists them so their absence is visible. Refused on a resume, which "
                             "INHERITS its parent's surface instead. Records "
                             "arch_source=production_config@<content hash> in model_config.json.")
    parser.add_argument("--allow-nonproduction-arch", action="store_true",
                        help="Consent to a FRESH run whose architecture differs from "
                             "designs/production_config.json. Without it such a launch is REFUSED, "
                             "naming every differing key with both values (--dry-run, "
                             "`python -m main.checkargs` and the launcher itself all refuse "
                             "identically). Use it for a deliberate ablation; the choice is "
                             "recorded in model_config.json's arch_source and in metadata's "
                             "cli_args. A fork/restart never needs it.")
