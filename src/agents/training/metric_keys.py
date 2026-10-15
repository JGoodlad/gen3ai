"""The STARTUP metric-key check (`gen3_metric_key_collision_v1`).

THE CLASS. The learner's stdout table (`train_logger.HumanOutputFormat`, sb3's) cut a key to 33 characters +
"..." and RAISED when two keys cut alike — mid-run, at the first dump that held both. Twice it killed every
run of an architecture at its first dump: F-XC-5 (`flat_switch_target_recall_top1_{bot,pool}`, every
fixed_mass self-play run) and 2026-10-15 (`cell/move_resolution_restored_move_{weight,grad}_norm` and the
`restored_switch` / `status_switch` pairs: every PRODUCTION run at a commit >= 65b71f6a, at step ~202k on
the GPU). Both escaped because the tests checked the metric DICT, never the formatter.

THE FIX, two halves:

1. the formatter is COLLISION-PROOF: a long key renders as `train_logger.truncate_label`'s
   ``<head>~<digest>~<tail>``, and a residual clash prints the full key — a display never stops a run;
2. THIS module refuses, at STARTUP, a key set that collides in ANY output format
   (`train_logger.check_metric_keys`: one key with two producers, one stdout row for two keys, one
   TensorBoard tag for two keys) — `MetricKeyCollisionError`, FATAL_CONFIG (`main.exit_codes`).

WHICH KEYS. The families whose NAMES are derived from the network — the ones that differ between
architectures, which is where both incidents lived — are listed from the built extractor
(`grad_balance.cell_family_keys` / `edge_family_keys`, both halves of every pair, one entry per producer).
On CUDA the update-fit gate's dry first update (`update_fit.dry_update`) also hands over EVERY key that real
update recorded (the first update of a process is always a diagnostic update, so every gated probe's tags
are in it). Off CUDA there is no dry update; the CPU smoke's dumps are the remaining check
(`main/train/debug_shape_smoke_integration_test.py` asserts >= 2 dumps carrying `cell/*`).
"""
from __future__ import annotations

from typing import Any, Iterable, List

from agents.training.grad_balance import cell_family_keys, edge_family_keys
from agents.training.train_logger import check_metric_keys


def network_metric_keys(model: Any) -> List[str]:
    """The `cell/*` and `edge/*` keys this model's extractor can write, one entry per producer."""
    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None:
        return []
    return cell_family_keys(fe) + edge_family_keys(fe)


def check_startup_metric_keys(model: Any, recorded: Iterable[str] = ()) -> int:
    """Refuse (`MetricKeyCollisionError`) a key set that collides in any output format: the network's
    derived families, plus ``recorded`` — the keys a dry update put on the logger's bus (deduplicated
    against the derived list: a recorded key IS one of the derived producers, not a second). Returns
    the number of distinct keys checked."""
    derived = network_metric_keys(model)
    keys = derived + sorted(set(recorded) - set(derived))
    check_metric_keys(keys, where="at startup: the network's cell/edge families"
                      + (" + the dry first update's recorded keys" if recorded else ""))
    return len(set(keys))
