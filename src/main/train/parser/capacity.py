"""The `# --- LIVE CAPACITY TELEMETRY (gen3_capacity_telemetry_v1) ---` section.

Lifted VERBATIM out of the old single-file `parser.py` (lines 892-979); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_capacity_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- LIVE CAPACITY TELEMETRY (gen3_capacity_telemetry_v1) --------------------------------
    # Three continuous saturation early-warnings that ride the train loop instead of being probed
    # offline: the PLASTICITY CANARY, the HALF-BATCH TRUNK-GRADIENT COSINE, and the FIXED-PROBE
    # FEATURE VELOCITY. All TRAINING-only and, uniquely, not even a loss weight — nothing here
    # enters `loss` or writes `.grad`, so the policy's updates are bit-identical on or off. They
    # are recorded on `ModelVersion` for PROVENANCE and `_resolve`-inherited on a flagless resume
    # (the `td_aux_coef` / `cf_records` class), never gated. Detail:
    # `agents/training/capacity_telemetry.py` and `src/agents/training/CLAUDE.md`.
    parser.add_argument("--capacity-telemetry", "--capacity_telemetry", dest="capacity_telemetry",
                        action=BoolFlag, default=None,
                        help="Log the `capacity/*` saturation early-warnings: the PLASTICITY CANARY "
                             "(a detached head refitting K=4 seeded synthetic targets, one of which "
                             "is re-seeded every --canary-reset-steps), the HALF-BATCH TRUNK COSINE "
                             "(does the batch fight itself?) and the FEATURE VELOCITY on a frozen "
                             "probe batch (do the functions still move?). Default OFF — off is "
                             "byte- AND cost-identical (no head, no optimizer, no probe batch, no "
                             "extra forward or backward). ON costs <3%% of the train step. Read every "
                             "scalar as a TREND, never as a level. ⚠️ The canary's state is NOT "
                             "checkpointed: it re-inits on every resume/launcher restart, so its "
                             "curves restart there too.")
    parser.add_argument("--canary-reset-steps", "--canary_reset_steps", dest="canary_reset_steps",
                        type=int, default=None,
                        help="ENV steps between plasticity-canary resets (default 1000000). Each "
                             "reset re-seeds ONE of the K=4 synthetic targets, round-robin; the "
                             "re-fit that follows is the supply-side measurement, read off "
                             "capacity/canary_recovery at a MATCHED capacity/canary_age. Too small "
                             "and the head never converges between resets; too large and the run "
                             "yields two points.")
    parser.add_argument("--capacity-cosine-every", "--capacity_cosine_every",
                        dest="capacity_cosine_every", type=int, default=None,
                        help="Minibatches between half-batch trunk-gradient cosine measurements "
                             "(default 50). The probe costs two half-batch forward+backwards ≈ one "
                             "extra full one, so 50 amortizes it to ~2%% of the train step. 0 = off "
                             "(the other two probes keep running).")
    parser.add_argument("--capacity-velocity-every", "--capacity_velocity_every",
                        dest="capacity_velocity_every", type=int, default=None,
                        help="train() calls between feature-velocity measurements (default 50). One "
                             "no_grad forward of a frozen 256-row probe batch. 0 = off.")
