"""`# --- THE ENV CORE (M5 Lane G) ---`: the Rust env core's collector, T2 and eval-core knobs.

The Rust env core is the ONLY env core: N envs in ONE core behind the process (or FFI) front end, the
trainee and every policy opponent forwarded through the inference service (T2), the scripted bots
played inside the core, the COMPLETE-GAME collector (order constraint 6 of
``designs/endstate/program_rust_core.md``). Design and hazards: ``designs/training/rust_collector.md``.

THERE IS NO ``--env-core`` FLAG (deletion pass P11b, 2026-10-03; ``designs/deleted_flags.md``): the
Python env core (``Gen3Env`` workers behind a ``SubprocVecEnv``) was DELETED in U3 and ``rust`` was its
one legal value, so a typed ``--env-core`` is refused with the reason by ``ExplainingParser``. What
stays keyed on the RECORD, not on a flag: ``metadata.json`` records the env core a process ran under
(``env_core``, read back by ``rust_env_setup.recorded_env_core``), and a python-era checkpoint that is
resumed or forked moves onto the Rust core ANNOUNCED as a core switch (``env_core_switch_line``) — a
shaped-critic one is REFUSED (deletion pass D4, ``rust_env_setup.refuse_python_era_checkpoint``: run it
pinned to its own commit).

Every collector flag defaults to ``None`` = "not typed", resolved by
``main.train.rust_env_setup.resolve_env_core_args`` (the defaults the help strings state).
"""
import argparse

from main.train.parser.base import BoolFlag  # noqa: F401 — the family's shared pieces


def add_env_core_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their `--help` order."""
    parser.add_argument("--rollout-trigger", "--rollout_trigger", dest="rollout_trigger",
                        choices=("complete_game", "window"), default=None,
                        help="'complete_game' (DEFAULT, the owner's collector, "
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
    parser.add_argument("--t2-opponent-bucket-cap", "--t2_opponent_bucket_cap", dest="t2_opponent_bucket_cap",
                        type=int, default=None,
                        help="The largest T2 bucket a NON-trainee slot (opponents, eval's slots) captures "
                             "(gen3_slot_bucket_caps_v1); rows beyond it are chunked. Default 64 (and 64 joins "
                             "the default buckets when --n-envs > 64): each lane's CUDA-graph pool is sized by its "
                             "largest capture, so only the trainee's lane holds the N-row bucket's (8 x 232 MiB "
                             "at N = 256 before). 0 = uncapped (every slot captures every bucket).")
    parser.add_argument("--t2-lanes", "--t2_lanes", dest="t2_lanes", type=int, default=None,
                        help="Inference lanes (CUDA streams); 0 (default) = min(slots, 8) on CUDA, 1 on CPU.")
    parser.add_argument("--t2-backend", "--t2_backend", dest="t2_backend", choices=("graph", "eager", "aot"),
                        default=None,
                        help="Inference backend; default 'graph' on CUDA, 'eager' on CPU.")
    parser.add_argument("--opponent-sampling", "--opponent_sampling", dest="opponent_sampling",
                        choices=("keyed", "generator"), default=None,
                        help="How a policy opponent's stochastic action is drawn 'keyed' "
                             "(DEFAULT, gen3_keyed_draw_v1: a counter-based draw keyed by (run seed, env, episode, "
                             "decision) — one vectorised op, exactly replayable; F-LE-8) or 'generator' (one torch "
                             "generator per env per opponent — today's RLPlayer stream, bit for bit).")
    parser.add_argument("--rust-eval-envs", "--rust_eval_envs", dest="rust_eval_envs", type=int, default=None,
                        help="(M5 Lane H) envs of the EVAL core, declared at startup "
                             "(default 64). An eval cycle plays each shard unit's games in order on one env, "
                             "so about --eval-games / --eval-shard-games x opponents envs run it in one wave.")
    parser.add_argument("--behaviour-check", "--behaviour_check", dest="behaviour_check",
                        choices=("off", "warn", "fatal"), default=None,
                        help="K9(b) BEHAVIOUR-POLICY CONSISTENCY: before any optimizer step of every update, "
                             "the learner's recomputed log pi(a|s) for rows at the CURRENT policy version must "
                             "equal the stored behaviour log-prob (fp32, the only matmul precision: DETERMINISTIC — rows within a "
                             "relative margin 2e-4 of a declared selection / threshold cutoff are excluded, every other "
                             "row's |d| < 1e-4, the excluded share < 0.15 — one measured table, "
                             "rust_rollout/consistency.BEHAVIOUR_GATE). Default 'fatal' on both env "
                             "cores: under rust Lane G's pre-loop probe (per-row policy versions), under python the "
                             "first micro-batch's own forward (K9, M5 Lane K — no extra forward).")
