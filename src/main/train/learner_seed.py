"""THE LEARNER'S GLOBAL STREAMS ON A RESUME OR A FORK (`gen3_learner_segment_seed_v1`).

THE DEFECT (`designs/research_state/measurements/gigo_audit_datapath_2026-10-26`). A loaded learner's
`_setup_model` seeds Python `random`, NumPy and torch from the seed SAVED IN THE CHECKPOINT. On a fork that
is the PARENT's seed: every X35 replicate (`--seed 3501` … `3505`, forked from `rb_x26_s2601`) carries
`seed 2601` in every checkpoint it wrote, so `--seed` never reached the learner, and every fork of one parent
— and every restart segment of one run — replayed the SAME global stream. Its one steady-state consumer is
the PPO minibatch permutation (`RolloutBuffer.get` → `np.random.permutation`, `global_rng_guard`), so the
"independent" replicates shuffled their (independent) buffers in one shared order. The DATA were never
shared: the collector's teams, battle seeds and keyed draws come from `segment_seed(--seed, num_timesteps)`.

THE RULE. After the load, before anything draws: the checkpoint records the RUN's declared seed
(`model.seed = --seed`), and the global streams are seeded from `learner_segment_seed(--seed,
num_timesteps)` — a keyed hash, like the collector's `segment_seed` (its own stream, 13), so a restart never
replays its first segment's permutations either. A fresh run is unchanged (its constructor seeds `--seed`).

THE GUARD. `check_declared_seed` refuses (`FatalConfigError`) a learner about to train whose recorded seed is
not the run's `--seed` — it fails on a revert of the reseed (the loaded seed is the parent's).
"""
from __future__ import annotations

from typing import Any, Optional

from main.exit_codes import FatalConfigError

#: The keyed-hash stream of the learner's per-segment global seed (the collector's `segment_seed` is 9, the
#: respawn segment 11, the teams' 2 / 3).
STREAM_LEARNER_SEGMENT = 13


class UndeclaredLearnerSeed(FatalConfigError):
    """The learner about to train records a seed other than the run's declared `--seed`."""


def declared_seed(seed: Optional[int]) -> int:
    return int(seed or 0)


def learner_segment_seed(seed: Optional[int], num_timesteps: int) -> int:
    """The global-stream seed of the process segment that starts at ``num_timesteps`` (< 2**31: NumPy's
    legacy global seed takes 32 bits)."""
    from agents.training import keyed_draw as KD

    return int(KD.draw_keys(declared_seed(seed) & 0x7FFFFFFFFFFFFFFF, STREAM_LEARNER_SEGMENT, 0,
                            int(num_timesteps), 0)) & 0x7FFFFFFF


def seed_loaded_learner(model: Any, seed: Optional[int]) -> int:
    """A resumed / forked learner: record the run's seed and seed the global streams for THIS segment.
    Returns the segment seed."""
    s = learner_segment_seed(seed, int(getattr(model, "num_timesteps", 0) or 0))
    model.seed = declared_seed(seed)
    model.set_random_seed(s)
    return s


def check_declared_seed(model: Any, seed: Optional[int]) -> None:
    """The guard (module docs)."""
    got = getattr(model, "seed", None)
    if declared_seed(got) != declared_seed(seed):
        raise UndeclaredLearnerSeed(
            f"[LearnerSeed] FATAL: the learner records seed {got!r} but the run declares --seed {declared_seed(seed)} "
            "— a loaded checkpoint's saved seed (a fork's PARENT's) seeds the global streams unless "
            "`main.train.learner_seed.seed_loaded_learner` runs after the load (gen3_learner_segment_seed_v1)")
