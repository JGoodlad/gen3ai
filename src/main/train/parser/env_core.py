"""`# --- THE ENV CORE (M5 Lane G) ---`: which environment the trainer's rollout runs on.

``--env-core python`` (the DEFAULT, unchanged) is today's ``SubprocVecEnv`` of ``Gen3Env`` workers.
``--env-core rust`` runs the rollout on the M5 Rust env core — N envs in ONE core behind the process
(or FFI) front end, the trainee and every policy opponent forwarded through the inference service
(T2), the scripted bots played inside the core — through the COMPLETE-GAME collector (order
constraint 6 of ``designs/endstate/program_rust_core.md``). Design and hazards:
``designs/training/rust_collector.md``. The CUTOVER to it as the default is a separate, later decision.

Runtime-only flags: none of them is recorded in ``model_config.json`` or inherited on a resume (a
launcher restart re-sends the argv, so they persist across one run's restarts); ``metadata.json``
records the env core a process ran under (``env_core``). Every collector flag defaults to ``None`` =
"not typed", resolved by ``main.train.rust_env_setup.resolve_env_core_args`` (the defaults the help
strings state), so a collector flag typed on the python core is REFUSED rather than silently inert
(``combination_checks``' ``env_core_flags_need_the_rust_core``).
"""
import argparse

from main.train.parser.base import BoolFlag  # noqa: F401 — the family's shared pieces


def add_env_core_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their `--help` order."""
    parser.add_argument("--env-core", "--env_core", dest="env_core", choices=("python", "rust"), default="python",
                        help="Which env the rollout runs on (M5 Lane G). 'python' (DEFAULT) = today's "
                             "SubprocVecEnv of Gen3Env workers. 'rust' = the M5 Rust env core: N envs in one "
                             "core (process front end), trainee + policy opponents through the inference "
                             "service, bots in the core, the COMPLETE-GAME collector (--rollout-trigger). "
                             "Refuses every flag whose path it does not serve yet, by name, at startup.")
    parser.add_argument("--rollout-trigger", "--rollout_trigger", dest="rollout_trigger",
                        choices=("complete_game", "window"), default=None,
                        help="--env-core rust only. 'complete_game' (DEFAULT, the owner's collector, "
                             "2026-09-29): every row of a game is buffered until the game ends, GAE and the "
                             "win-prob labels run on COMPLETE games, and an update fires once the buffer holds "
                             "--rollout-target-samples completed-game rows; games in progress carry over, "
                             "each row keeps its behaviour log-prob and policy version, and NO row is dropped "
                             "or down-weighted for age (PPO's per-row ratio corrects it). 'window' = today's "
                             "n_steps-per-env schedule (the rollout-level parity tool).")
    parser.add_argument("--rollout-target-samples", "--rollout_target_samples", dest="rollout_target_samples",
                        type=int, default=None,
                        help="The complete-game trigger's TARGET sample count (rows per update). 0 (default) = "
                             "--n-steps x --n-envs, today's rollout size. Must be a multiple of lcm("
                             "--batch-size, --n-envs) (no ragged micro-batch, and the buffer keeps its "
                             "[n_steps, n_envs] shape).")
    parser.add_argument("--rollout-target-band", "--rollout_target_band", dest="rollout_target_band",
                        type=str, default=None,
                        help="'LO,HI' — the band inside which an adaptive-batch controller may move the target "
                             "between updates (the hook; the SIZING study picks the numbers). HI sizes the row "
                             "arena at startup. Default: the target alone.")
    parser.add_argument("--rust-env-front", "--rust_env_front", dest="rust_env_front", choices=("proc", "ffi"),
                        default=None,
                        help="The Rust env's front end: 'proc' (DEFAULT, crash isolation: a core fault is a "
                             "typed error + a respawn, the learner's optimizer and GPU context survive) or "
                             "'ffi' (in process).")
    parser.add_argument("--rust-env-threads", "--rust_env_threads", dest="rust_env_threads", type=int, default=None,
                        help="Worker threads inside the Rust env core (default 8; the core scaled 5.8x at 8).")
    parser.add_argument("--rust-env-profile", "--rust_env_profile", dest="rust_env_profile",
                        choices=("release", "selfcheck"), default=None,
                        help="Which build of THIS checkout's Rust env core to load (default 'release'; "
                             "'selfcheck' = the emission self-check build every test runs, slower).")
    parser.add_argument("--rust-env-refusal-budget", "--rust_env_refusal_budget", dest="rust_env_refusal_budget",
                        type=int, default=None,
                        help="Quarantined battles the core may bank before the batch fails "
                             "(RefusalBudgetExceeded: a refusal STORM is systemic).")
    parser.add_argument("--rust-env-respawn-budget", "--rust_env_respawn_budget", dest="rust_env_respawn_budget",
                        type=int, default=None,
                        help="Process-front-end child deaths a run survives (each cuts every game in progress "
                             "and RESETs the core; counted in PROC_SPAWNS_AFTER_FREEZE). Beyond it: FATAL.")
    parser.add_argument("--version-pinning", "--version_pinning", dest="version_pinning",
                        choices=("off", "per_game"), default=None,
                        help="The FIRST staleness remedy, OFF by default (owner 2026-09-29: enable only if the "
                             "staleness/* measurements say old rows are harmful). 'per_game' = a game is played "
                             "start-to-finish by the policy version it began with, held in its own T2 slot "
                             "(--trainee-slots of them), so every game is single-policy data.")
    parser.add_argument("--trainee-slots", "--trainee_slots", dest="trainee_slots", type=int, default=None,
                        help="T2 slots for the trainee's versions (default 1; 3 under --version-pinning "
                             "per_game). A pinned update with no free slot is a typed failure.")
    parser.add_argument("--t2-buckets", "--t2_buckets", dest="t2_buckets", type=str, default=None,
                        help="Inference batch buckets, comma-separated. Default (8, --n-envs): 8 for an opponent "
                             "slot's 1–6 rows, N for the trainee's batch (F-LE-9, decided 2026-09-29).")
    parser.add_argument("--t2-lanes", "--t2_lanes", dest="t2_lanes", type=int, default=None,
                        help="Inference lanes (CUDA streams); 0 (default) = min(slots, 8) on CUDA, 1 on CPU.")
    parser.add_argument("--t2-backend", "--t2_backend", dest="t2_backend", choices=("graph", "eager", "aot"),
                        default=None,
                        help="Inference backend; default 'graph' on CUDA, 'eager' on CPU.")
    parser.add_argument("--opponent-sampling", "--opponent_sampling", dest="opponent_sampling",
                        choices=("keyed", "generator"), default=None,
                        help="How a policy opponent's stochastic action is drawn under --env-core rust: 'keyed' "
                             "(DEFAULT, gen3_keyed_draw_v1: a counter-based draw keyed by (run seed, env, episode, "
                             "decision) — one vectorised op, exactly replayable; F-LE-8) or 'generator' (one torch "
                             "generator per env per opponent — today's RLPlayer stream, bit for bit).")
    parser.add_argument("--rust-eval-envs", "--rust_eval_envs", dest="rust_eval_envs", type=int, default=None,
                        help="--env-core rust only (M5 Lane H): envs of the EVAL core, declared at startup "
                             "(default 64). An eval cycle plays each shard unit's games in order on one env, "
                             "so about --eval-games / --eval-shard-games x opponents envs run it in one wave.")
    parser.add_argument("--behaviour-check", "--behaviour_check", dest="behaviour_check",
                        choices=("off", "warn", "fatal"), default=None,
                        help="K9(b) BEHAVIOUR-POLICY CONSISTENCY: before any optimizer step of every update, "
                             "the learner's recomputed log pi(a|s) for rows at the CURRENT policy version must "
                             "equal the stored behaviour log-prob (fp32: DETERMINISTIC — rows within a relative margin 2e-4 of a declared "
                             "selection / threshold cutoff are excluded, every other row's |d| < 1e-4, the excluded share < 0.15; "
                             "--matmul-precision high: the micro-batch's "
                             "p99 |d| < 3.6e-3 AND its max < 0.071, the max FATAL only on 4 consecutive updates, and REFUSED "
                             "with 'fatal' under --env-core rust — one measured table, "
                             "rust_rollout/consistency.BEHAVIOUR_GATES). Default 'fatal' on both env "
                             "cores: under rust Lane G's pre-loop probe (per-row policy versions), under python the "
                             "first micro-batch's own forward (K9, M5 Lane K — no extra forward).")
