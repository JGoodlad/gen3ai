"""THE argparse surface — `build_parser()` and its three custom argparse pieces.

Split out of `train_rl_agent.py` so the parser can be read (and inspected by `main.checkargs`)
without loading the training entry point. `train_rl_agent` re-exports every name here, so
`from main.train_rl_agent import build_parser` still resolves.

**On the shape.** This was ONE 1,967-line file, 33 lines from `file_size_gate_test.py`'s 2,000
hard bound, defended in prose as "197 flags read top-to-bottom exactly as `--help` renders
them". The reading order is preserved and the defence is not retracted — it is just no longer a
reason to keep one file, because it is a property of `build_parser()`'s CALL ORDER, not of the
bytes' being contiguous. Each family module holds its section VERBATIM, in its original relative
order, and `build_parser()` calls them in the original order; `--help` is byte-identical.

    base.py              `optional_float` / `str2bool` / `BoolFlag` — the shared argparse pieces
    operational.py       `# --- Operational Flags ---`
    hyperparameters.py   `# --- Hyperparameter Flags (Optimized for GPU) ---`
    reward.py            `# --- Reward config ---` (resume-immutable, value-checked)
    clean_world.py       `# --- gen3_clean_world_config_v1 ---` + the PPO clip / belief /
                         damage-op / compile / entity-seat flags declared under it
    teacher.py           `# --- SEARCH-AS-TEACHER ---` + `# --- THE WIN-PROB ONE-PLY TEACHER ---`
    cf_grounding.py      `# --- COUNTERFACTUAL VALUE GROUNDING ---`
    value_heads.py       the EVIDENTIAL BETA / TWIN HEADS + SHADOW CRITIC / PER-ACTION Q sections
    capacity.py          `# --- LIVE CAPACITY TELEMETRY ---`
    distillation.py      `# --- ADVANTAGE-GATED / ACTION-FORM DISTILLATION + the RANK TRIPWIRE ---`
    eval_subprocess.py   `# --- Subprocess eval ---` (workers, self-play pool, exploiter, teams)

Adding a flag means editing ONE family module — and appending it at the end of that family's
function, since the position inside a family is the position in `--help`.
"""
import argparse

from main.train.parser.base import (   # noqa: F401 — re-export hub
    BoolFlag, optional_float, str2bool, _BOOL_FALSE, _BOOL_TRUE,
)
from main.train.parser.capacity import add_capacity_flags
from main.train.parser.cf_grounding import add_cf_grounding_flags
from main.train.parser.clean_world import add_clean_world_flags
from main.train.parser.distillation import add_distillation_flags
from main.train.parser.eval_subprocess import add_eval_subprocess_flags
from main.train.parser.hyperparameters import add_hyperparameter_flags
from main.train.parser.operational import add_operational_flags
from main.train.parser.reward import add_reward_flags
from main.train.parser.teacher import add_teacher_flags
from main.train.parser.value_heads import add_value_head_flags

__all__ = ["optional_float", "str2bool", "BoolFlag", "build_parser",
           "_BOOL_TRUE", "_BOOL_FALSE"]


def build_parser() -> argparse.ArgumentParser:
    """THE argument parser, as data — built outside `main()` so it can be INSPECTED without
    running a training job.

    Extracted for two reasons, both of which cost real time before it existed:

    * A run's recorded `launcher_command` outlives the flags in it. Relaunching gen-12's
      argv on v89 died on `--pubval-*` (deleted at v88) — one flag at a time, since argparse
      reports only the first error, and only by actually starting the trainer.
    * `--help` was itself broken (an unescaped `%` rendered as a `%o` conversion), so there
      was no offline way to ask what the parser accepts. Nothing rendered the help strings,
      so nothing caught it.

    `python -m main.checkargs` is the consumer; `checkargs_test.py` renders every help string.

    **The call order below IS the `--help` order** — argparse renders optionals in the order
    they were added. Reordering these calls, or moving a flag between families, silently
    rewrites `--help`; keep an addition inside its family.
    """
    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")

    add_operational_flags(parser)
    add_hyperparameter_flags(parser)
    add_reward_flags(parser)
    add_clean_world_flags(parser)
    add_teacher_flags(parser)
    add_cf_grounding_flags(parser)
    add_value_head_flags(parser)
    add_capacity_flags(parser)
    add_distillation_flags(parser)
    add_eval_subprocess_flags(parser)

    return parser
