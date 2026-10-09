"""The `# --- Operational Flags ---` section: what to run, where to write it, and the
`--debug` smoke.

Lifted VERBATIM out of the old single-file `parser.py` (lines 108-178); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_operational_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--run-dir", type=str, help="Run folder to write checkpoints into (set by launcher on resume)")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None,
                        help="A MEMORABLE name for a fresh run → writes to models/<name>/ instead of "
                             "a date-stamped models/<era>_run_<timestamp>/ (<era> = the two-letter era code, e.g. rb; "
                             "utils.era). Take the era prefix yourself (rb_x26_s1001): a name without it is "
                             "accepted as typed with a warning. Must be a single name "
                             "(letters/digits/._-, no slashes). Refuses to overwrite an existing run "
                             "of that name (pick another, or --model to resume it). Ignored when "
                             "--run-dir is set (launcher resume). For --exploiter, defaults to "
                             "'<era>_exploiter_vs_<target>' if you don't name it.")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action=BoolFlag, default=False, help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--debug-eval", "--debug_eval", dest="debug_eval", action=BoolFlag, default=False,
                        help="Run evaluation under --debug. By default a --debug smoke run skips ALL eval "
                             "(both the periodic eval callback AND the final win-rate eval) so it needs no "
                             "eval opponents / Showdown eval connection and stays light on CPU. Pass "
                             "--debug-eval to exercise the eval pipeline in a smoke run. No effect on real "
                             "(non-debug) runs, which always eval.")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)")
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
                             "unset, so an explicitly-typed flag still wins. It ALSO applies the "
                             "production TRAINING RECIPE (main.train.recipe_surface: the mirror's "
                             "`recipe` block — n_envs, batch/accumulation, epochs, LR, clip, "
                             "entropy, self-play — plus the reward values and the "
                             "belief-supervision doses), every untyped knob as if typed. The "
                             "remaining critic READOUTS are not applied (the win-prob critic implies "
                             "them). Refused on a resume, which INHERITS its parent's surface "
                             "instead (a same-run restart of such a run restores the recipe from "
                             "its own metadata.json). Records "
                             "arch_source=production_config@<content hash> in model_config.json.")
    parser.add_argument("--allow-nonproduction-arch", action="store_true",
                        help="Consent to a FRESH run whose architecture differs from "
                             "designs/production_config.json. Without it such a launch is REFUSED, "
                             "naming every differing key with both values (--dry-run, "
                             "`python -m main.checkargs` and the launcher itself all refuse "
                             "identically). Use it for a deliberate ablation; the choice is "
                             "recorded in model_config.json's arch_source and in metadata's "
                             "cli_args. A fork/restart never needs it.")
    # --- K10(a) THE RECIPE SURFACE (main.train.recipe_surface) -------------------------------
    parser.add_argument("--allow-nonproduction-recipe", action="store_true",
                        help="Consent to a FRESH run whose TRAINING RECIPE differs from the "
                             "production mirror's `recipe` block on a knob this argv did NOT type "
                             "(the value is the parser's default, not a choice). Without it such "
                             "a launch is REFUSED by --dry-run, `python -m main.checkargs` and the "
                             "launcher, naming every untyped knob. A TYPED differing value never "
                             "needs it — typing is how an arm states its lever. Recorded in "
                             "metadata's cli_args. A fork/restart never needs it.")
    # --- T23 THE DESKTOP-GPU REFUSAL (utils.desktop_gpu) ----------------------------------------
    parser.add_argument("--allow-desktop-gpu", action="store_true",
                        help="Consent to a CUDA run while a display process (gnome-shell, Xorg, "
                             "Xwayland, a display manager, ...) holds the GPU. Without it such a "
                             "launch exits FATAL_CONFIG naming the process, its VRAM and the fix "
                             "(`sudo systemctl stop gdm.service`); the launcher's --dry-run reports "
                             "the same verdict. For dev / short runs only: the desktop's VRAM "
                             "(~0.8 GiB) is gone from the run. Recorded in metadata.json "
                             "(cli_args.allow_desktop_gpu + cli_args._desktop_gpu). `--debug` (CPU) "
                             "is exempt without it.")
    # --- THE DISK-SPACE GUARD (utils.disk_guard) -------------------------------------------------
    parser.add_argument("--allow-low-disk", action="store_true",
                        help="Consent to a run whose REQUIRED disk space (checkpoints still to be written + "
                             "eval traces + compile cache + log allowances + a margin, derived from the run "
                             "itself and printed) exceeds the free space on the run archive's filesystem. "
                             "Without it such a launch exits FATAL_CONFIG naming the shortfall; the "
                             "launcher's --dry-run reports the same verdict. Also stands the in-run STOP "
                             "down (free < one more checkpoint => clean exit FATAL_DISK 8); the low-disk "
                             "warnings stay. Dev / short runs only. Recorded in metadata.json "
                             "(cli_args.allow_low_disk + cli_args._disk_guard). `--debug` is exempt.")
