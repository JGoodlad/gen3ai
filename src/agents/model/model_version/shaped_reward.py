"""A checkpoint TRAINED WITH A SHAPED REWARD, recognised — so a resume or fork of one REFUSES.

`gen3_shaped_reward_deletion_v1` (config v122, program_rust_core §4 M3 row, owner-approved
2026-09-26). The hand-shaped reward path — the eight PBRS potentials, the ~25 BIAS terms, the
bias-additivity refund, the no-progress tax — was DELETED, and with it the 14 recorded fields
below. Production had trained on the terminal alone since the win-prob era.

**THE DECISION (owner, 2026-09-26): a resume or fork of a shaped checkpoint REFUSES LOUDLY.** It
must never continue silently on the terminal alone: the objective would change underneath the run
while its name, its TensorBoard series and its lineage all said it was the same experiment. The fix
the refusal names is to run it PINNED to a commit that still has the shaped path — any commit at or
before :data:`LAST_SHAPED_COMMIT` (the checkpoint's own recorded `git_hash` is the natural pin, and
the launcher pins a restart to it by default).

⚠️ **TODAY THIS IS BELT-AND-BRACES.** `MIGRATION_FLOOR` (121) already refuses every pre-v121
checkpoint on HEAD, and every v121 run on record is a win-prob-era arm that recorded
`hand_shaping=False`. It is made explicit anyway because the floor refuses for a DIFFERENT reason
(architecture), and the day the floor is walked back or a v121 shaped arm exists, a silent
terminal-only continuation is exactly the failure nothing else would catch.

WHERE IT IS ENFORCED — only where the reward matters:

* `main.train.config.resolve_config` — every `--model` launch (resume AND fork), before the
  inheritance sweep reads a single recorded flag;
* `main.checkargs` — the offline answer to "does this launch", from the same predicate.

NOT on a frozen load (eval opponent, self-play pool, teacher, prober): a frozen forward never reads
the reward, so the deleted fields simply POP in `_migrate_config`, and a shaped-trained snapshot
remains a perfectly good opponent.

The predicate is the deleted census's own gates, restated over the RAW recorded dict (the migration
pops the fields, so it must read what the file SAYS before any migration runs): a config was shaped
iff `hand_shaping` was on (it gated every potential and the whole BIAS class; absent means a pre-v105
config, whose terms were unconditional), or the one BIAS term `--arm-no-progress-tax` could re-arm
under `--no-hand-shaping` was reachable.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agents.model.model_version.constants import ModelVersionError

#: The last commit whose tree still has the shaped reward path — the one BEFORE the deletion. Any
#: commit at or before it can resume a shaped checkpoint (`--pin-commit <sha>`).
LAST_SHAPED_COMMIT = "029cee837a36025e6191dd635b29852839faa30f"

#: The first config version written WITHOUT the shaped fields. A config at or above it that lacks
#: `hand_shaping` is terminal-only by construction; one below it that lacks the key predates v105,
#: when every hand term was unconditional.
SHAPED_DELETION_VERSION = 122

#: Every recorded field the deletion removed → the CLI flag that set it. Declared here (the lowest
#: layer that needs it) so the migration's POP list, this predicate and the freshness docs cannot
#: disagree about what went. The names are stored WITHOUT their `--` and prefixed below: a bare
#: `"--flag"` constant in a production module counts as LIVE CLI surface to
#: `src/claude_md_freshness_gate_test.py`, and these are exactly the flags that no longer are.
_DELETED_FLAG_NAMES: Dict[str, str] = {
    "bias_additivity": "bias-additivity",
    "mat_alive_weight": "mat-alive-weight",
    "bias_redesign": "bias-redesign",
    "switch_bias_weight": "switch-bias-weight",
    "self_ko_hp_penalty": "self-ko-hp-penalty",
    "drop_redundant_bias": "drop-redundant-bias",
    "drop_switch_bias": "drop-switch-bias",
    "all_shaping_pbrs": "all-shaping-pbrs",
    "stall_pbrs": "stall-pbrs",
    "no_progress_penalty": "no-progress-penalty",
    "hand_shaping": "hand-shaping",
    "pbrs_material": "pbrs-material",
    "pbrs_belief": "pbrs-belief",
    "no_progress_tax_armed": "arm-no-progress-tax",
}
DELETED_SHAPED_REWARD_FIELDS: Dict[str, str] = {
    field: "--" + name for field, name in _DELETED_FLAG_NAMES.items()}


class ShapedRewardCheckpointError(ModelVersionError):
    """A resume or fork of a checkpoint trained with the DELETED shaped reward. Typed, so a caller
    (and a test) can tell it from every other version refusal; a `ModelVersionError`, so every
    existing `except ModelVersionError` exits FATAL_CONFIG on it."""

    def __init__(self, message: str, *, evidence: List[str], config_path: Optional[str]):
        super().__init__(message)
        self.evidence = list(evidence)
        self.config_path = config_path


def shaped_reward_evidence(raw: Dict[str, Any]) -> List[str]:
    """The recorded facts proving `raw` (a `model_config.json` dict, UNMIGRATED) trained with a
    shaped reward — each as ``"field=value (--flag)"``. Empty ⇔ the reward was the terminal alone."""
    version = raw.get("config_version", 1)
    if "hand_shaping" in raw:
        hand = bool(raw["hand_shaping"])
    else:
        # Absent: a post-deletion config (terminal-only by construction) or a pre-v105 one, whose
        # hand terms had no gate at all (v105's migration defaulted exactly this to True).
        hand = not (isinstance(version, int) and version >= SHAPED_DELETION_VERSION)
    out: List[str] = []
    if hand:
        how = "recorded" if "hand_shaping" in raw else f"implied by config_version {version}"
        out.append(f"hand_shaping=True ({how}; --hand-shaping — every hand PBRS potential and "
                   "BIAS term was live)")
        return out
    # `--no-hand-shaping` zeroed everything EXCEPT a re-armed no-progress tax, whose own gate was
    # (bias_redesign or all_shaping_pbrs) and not stall_pbrs.
    if bool(raw.get("no_progress_tax_armed", False)):
        reachable = ((bool(raw.get("bias_redesign", False)) or bool(raw.get("all_shaping_pbrs", True)))
                     and not bool(raw.get("stall_pbrs", False)))
        if reachable:
            out.append("no_progress_tax_armed=True (--arm-no-progress-tax — the no-progress tax "
                       "BIAS term was live)")
    return out


def refusal_message(evidence: List[str], config_path: Optional[str]) -> str:
    where = config_path or "its model_config.json"
    lines = "\n".join(f"  · {e}" for e in evidence)
    return (
        f"this checkpoint was TRAINED WITH A SHAPED REWARD ({where}):\n{lines}\n"
        "The shaped reward path was DELETED (gen3_shaped_reward_deletion_v1, config v"
        f"{SHAPED_DELETION_VERSION}, program_rust_core §4 M3). Resuming or forking it on this code "
        "would continue it on the TERMINAL ALONE — a different objective under the same run name — "
        "so it is refused rather than silently switched.\n"
        f"Fix: run it pinned to a pre-deletion commit, ≤ {LAST_SHAPED_COMMIT[:8]} "
        f"(`--pin-commit {LAST_SHAPED_COMMIT[:8]}`, or the checkpoint's own recorded git_hash, "
        "which the launcher pins a restart to by default). A fresh win-indicator run is a NEW "
        "experiment, not a continuation — start it without --model.")


def saved_config_path(model_path: str) -> Optional[str]:
    """The `model_config.json` a resume from `model_path` reads — the checkpoint's directory, then
    its parent (a checkpoint in `<run>/checkpoints/` beside a run-level config). None if neither."""
    dirs = [os.path.abspath(model_path)] if os.path.isdir(model_path) else []
    try:
        from agents.model.snapshot import _resolve_paths
        _, cfg_dir = _resolve_paths(model_path)
    except Exception:                                   # noqa: BLE001 — no zip: fall back
        cfg_dir = os.path.dirname(os.path.abspath(model_path))
    dirs += [cfg_dir, os.path.dirname(cfg_dir)]
    for d in dirs:
        cand = os.path.join(d, "model_config.json")
        if os.path.exists(cand):
            return cand
    return None


def check_not_shaped(config_path: Optional[str]) -> None:
    """Raise :class:`ShapedRewardCheckpointError` when the config at `config_path` recorded a shaped
    reward. A missing path is not a verdict (the caller's own unreadable-config handling applies);
    an unparseable file is left to the loader that will fail on it loudly anyway."""
    if not config_path or not os.path.exists(config_path):
        return
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    evidence = shaped_reward_evidence(raw)
    if evidence:
        raise ShapedRewardCheckpointError(refusal_message(evidence, config_path),
                                          evidence=evidence, config_path=config_path)
