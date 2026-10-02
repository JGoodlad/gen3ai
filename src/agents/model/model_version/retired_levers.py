"""LEVERS THAT WERE DELETED IN THE POST-SWITCH DELETION PASS, recognised in a recorded config.

`designs/ops/deletion_pass_manifest.md` §2 (owner-approved 2026-10-02: delete with the Python core,
port none), plus K2's retired TF32 (R9, owner 2026-10-01). Each lever below left `ModelVersion` — `_migrate_config` POPs its recorded field from any
config, whatever vintage wrote it (`cls(**data)` TypeErrors on a stale key) — and this module is where
the deletion is JUDGED, in the shape `shaped_reward.py` set for the deleted shaped reward:

* **STRUCTURAL levers** — their ON value named PARAMETERS or a critic route the surviving code cannot
  rebuild (a PopArt normalizer's buffers, the distributional value head's Linear, the `value_from_dist`
  critic route, the privileged true-team readout, the dense auxiliary head). A config recording one ON is REFUSED on EVERY load (`_migrate_config` calls
  :func:`refuse_structural`): popping it would hand SB3 an unplaceable state_dict, or run a checkpoint
  under a critic it was not trained with, with every shape check green.
* **TRAINING-ONLY levers** — they scaled a loss or edited the reward stream and touched no forward pass,
  so a frozen load (an eval opponent, a pool snapshot, a teacher, the prober) loses nothing by the
  POP. The one path where the recorded value MATTERS is a RESUME or FORK, which would keep TRAINING
  without the lever — a different objective under the same run name. Those refuse BEFORE the
  migration pops the evidence, from the raw file (:func:`check_no_retired_levers`, called by
  `main.train.config.resolve_config` and `main.checkargs`), and name the fix: run it PINNED to a
  commit that still has the lever (:attr:`RetiredLever.last_commit`).

⚠️ **TODAY THIS IS BELT-AND-BRACES.** `MIGRATION_FLOOR` (121) already refuses every pre-v121
checkpoint on HEAD, and every v121+ run on record (2026-10-02: 15 of them) recorded every lever below
OFF. It is made explicit for the shaped reward's reason: the day the floor is walked back, or a run
from another box appears, a silent continuation without the lever is exactly the failure nothing else
would catch.

Flag names are stored WITHOUT their `--` and prefixed below: a bare `"--flag"` string constant in a
production module counts as LIVE CLI surface to `src/claude_md_freshness_gate_test.py`, and these are
exactly the flags that no longer are. **Later deletion units APPEND their levers to :data:`RETIRED`.**

**TF32 (`matmul_precision`, deletion pass K2) is a RUNTIME knob, so its record is `metadata.json`'s, not
`model_config.json`'s:** `check_no_retired_levers` overlays that file's `RUNTIME_RECORD_FIELDS` onto the raw
config before it judges the table, so the one refusal path covers it. It is training-only (no forward pass
carries it; a frozen load loses nothing), so it refuses a RESUME or FORK only, and no `ModelVersion` field
or config bump was involved.

**Levers with NO recorded field are not in the table, and cannot be.** `--defensive-entropy-boost` /
`--bait-entropy-boost` (deletion pass L2) were never written to `model_config.json` (they were
training-only, never inherited on a flagless resume, and so never `_resolve`d), so there is no raw
config to read an "ON" from; their only trace is `metadata.json`'s `cli_args` / `original_command`,
and an unpinned resume of such a run fails argparse on the inherited argv instead
(`designs/ops/deletion_pass_manifest.md` §6 finding 8 — the flag census decides whether that is
softened). `designs/deleted_flags.md` names the pin for each.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from agents.model.model_version.constants import ModelVersionError

#: The last commit whose tree still has the L1 levers (self-PBRS, frozen-φ PBRS, PopArt, the
#: distributional value head, `value_from_dist`, the CVaR value-tail weight, the win-prob aux BCE
#: coefficient) — the tip when deletion unit L1 branched. Any commit at or before it can resume a run
#: that recorded one (`--pin-commit <sha>`).
LAST_COMMIT_L1 = "3bc3e77ed2a6883324262a6844cb97f1e8c59377"

#: The first config version written WITHOUT the L1 fields.
L1_DELETION_VERSION = 131

#: The last commit whose tree still has the L2 levers (the win-prob λ-return target, the R-rollout
#: Monte-Carlo target and its anchor weight, the dense auxiliary head, the privileged true-team value
#: route, both entropy boosts) — the tip when deletion unit L2 branched, which is the commit that
#: shipped L1 (`475bd817`). Any commit at or before it can resume a run that recorded one.
LAST_COMMIT_L2 = "475bd817bdf1da3d055036641a29fe7540ec090d"

#: The first config version written WITHOUT the L2 fields.
L2_DELETION_VERSION = 132

#: The last commit whose tree still has TF32 (`--matmul-precision high` and its three parity rules) — the
#: tip when deletion unit K2 branched, which is the commit that shipped L2 (`615a764f`). Any commit at or
#: before it can resume a run that recorded `matmul_precision: high` (`--pin-commit <sha>`).
LAST_COMMIT_K2 = "615a764fdb7e05abfcc1575797e2c83eb61c3ea1"

#: Recorded in `metadata.json` (top level, rewritten on every save), not in `model_config.json`: the
#: RUNTIME knobs a retired lever may have set. `check_no_retired_levers` overlays them onto the raw config.
RUNTIME_RECORD_FIELDS: tuple = ("matmul_precision",)


class RetiredLever(NamedTuple):
    field: str                                   # the recorded ModelVersion field that left the config
    flag: str                                    # the CLI flag that set it, WITHOUT the leading `--`
    active: Callable[[Dict[str, Any]], bool]     # was this lever ON in the recorded RAW config?
    structural: bool                             # ON named parameters / a route: refuse on EVERY load
    what: str                                    # what ON meant (the refusal's evidence line)
    last_commit: str                             # the last commit whose tree still has the lever
    unit: str = "L1"                             # the deletion unit that removed it (for the refusal's text)
    version: int = L1_DELETION_VERSION           # the config version that first wrote without it


def _truthy(field: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda raw: bool(raw.get(field, False))


def _nonzero(field: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda raw: float(raw.get(field, 0.0) or 0.0) != 0.0


def _below_one(field: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda raw: float(raw.get(field, 1.0) if raw.get(field) is not None else 1.0) < 1.0


def _above_one(field: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda raw: float(raw.get(field, 1.0) if raw.get(field) is not None else 1.0) > 1.0


def _winprob_aux_coef_moved(raw: Dict[str, Any]) -> bool:
    """`win_prob_coef` weighted the win-prob head's AUXILIARY BCE under the shaped critic; under the
    win-prob critic the head's BCE is the value loss at `vf_coef` and the coefficient was unused. Its
    value only mattered where there was a head, a non-win-prob critic and a non-default weight."""
    if str(raw.get("critic", "shaped")) == "winprob":
        return False
    if str(raw.get("win_prob_mode", "none")) == "none":
        return False
    return abs(float(raw.get("win_prob_coef", 1.0) if raw.get("win_prob_coef") is not None else 1.0)
               - 1.0) > 1e-12


#: Every retired lever, in the order a refusal reports them.
RETIRED: tuple = (
    RetiredLever("use_popart", "use-popart", _truthy("use_popart"), True,
                 "use_popart=True (PopArt value-target normalization: running mu/sigma buffers and a "
                 "normalized value head)", LAST_COMMIT_L1),
    RetiredLever("value_dist_mode", "value-dist-mode",
                 lambda raw: str(raw.get("value_dist_mode", "none") or "none") != "none", True,
                 "value_dist_mode != 'none' (the distributional value head: an HL-Gauss Linear in the "
                 "state_dict)", LAST_COMMIT_L1),
    RetiredLever("value_from_dist", "value-from-dist", _truthy("value_from_dist"), True,
                 "value_from_dist=True (the critic was the distributional E[Z], not the scalar value "
                 "head)", LAST_COMMIT_L1),
    RetiredLever("value_tail_weight", "value-tail-weight", _nonzero("value_tail_weight"), False,
                 "value_tail_weight != 0 (a CVaR-blended value loss)", LAST_COMMIT_L1),
    RetiredLever("win_prob_coef", "win-prob-coef", _winprob_aux_coef_moved, False,
                 "win_prob_coef != 1.0 under the shaped critic (the win-prob head's auxiliary BCE "
                 "weight)", LAST_COMMIT_L1),
    RetiredLever("win_prob_pbrs_coef", "win-prob-pbrs-coef", _nonzero("win_prob_pbrs_coef"), False,
                 "win_prob_pbrs_coef != 0 (self-PBRS: gamma*phi(s') - phi(s) from the win-prob head "
                 "added to the reward)", LAST_COMMIT_L1),
    RetiredLever("win_prob_pbrs_source", "win-prob-pbrs-source",
                 lambda raw: bool(raw.get("win_prob_pbrs_source")), False,
                 "win_prob_pbrs_source set (the frozen checkpoint supplying the self-PBRS potential)",
                 LAST_COMMIT_L1),
    RetiredLever("win_prob_pbrs_frozen", "win-prob-pbrs-frozen",
                 lambda raw: bool(raw.get("win_prob_pbrs_frozen")), False,
                 "win_prob_pbrs_frozen set (frozen-phi PBRS: a frozen win-prob head reshaping the "
                 "policy's advantages)", LAST_COMMIT_L1),
    # ---- deletion pass L2 (config v132) -----------------------------------------------------
    RetiredLever("value_true_team", "value-true-team", _truthy("value_true_team"), True,
                 "value_true_team=True (the privileged true-opponent-team value route: a "
                 "TrueTeamValueReadout in the state_dict, reading the `opp_true_team` obs key)",
                 LAST_COMMIT_L2, "L2", L2_DELETION_VERSION),
    RetiredLever("dense_aux", "win-prob-dense-aux", _truthy("dense_aux"), True,
                 "dense_aux=True (the dense auxiliary head: a DenseAuxHead in the state_dict)",
                 LAST_COMMIT_L2, "L2", L2_DELETION_VERSION),
    RetiredLever("win_prob_dense_aux", "win-prob-dense-aux", _nonzero("win_prob_dense_aux"), False,
                 "win_prob_dense_aux != 0 (the dense auxiliary loss's weight)",
                 LAST_COMMIT_L2, "L2", L2_DELETION_VERSION),
    RetiredLever("win_prob_lambda", "win-prob-lambda", _below_one("win_prob_lambda"), False,
                 "win_prob_lambda < 1.0 (the win-prob BCE trained against a lambda-return target "
                 "instead of the terminal bit)", LAST_COMMIT_L2, "L2", L2_DELETION_VERSION),
    RetiredLever("win_prob_rollout_target", "win-prob-rollout-target",
                 _nonzero("win_prob_rollout_target"), False,
                 "win_prob_rollout_target != 0 (a fraction of the rollout buffer's states labelled "
                 "by an R-rollout Monte-Carlo win fraction)", LAST_COMMIT_L2, "L2",
                 L2_DELETION_VERSION),
    RetiredLever("win_prob_rollout_weight", "win-prob-rollout-weight",
                 _above_one("win_prob_rollout_weight"), False,
                 "win_prob_rollout_weight > 1.0 (a per-row loss weight on the rollout-anchored rows)",
                 LAST_COMMIT_L2, "L2", L2_DELETION_VERSION),
    # ---- deletion pass K2 (no config bump: a metadata.json-recorded runtime knob) -----------
    RetiredLever("matmul_precision", "matmul-precision",
                 lambda raw: str(raw.get("matmul_precision") or "highest") != "highest", False,
                 "matmul_precision != 'highest' (TF32 matmuls in the trainer process: the run's numerics "
                 "were ~1e-2 coarser than fp32, and TF32 was retired)",
                 LAST_COMMIT_K2, "K2", L2_DELETION_VERSION),
)

#: Recorded fields that left the config but are INERT without one of the levers above (an atom count,
#: a support, a loss weight for a head that no longer exists): popped silently, any value.
INERT_RETIRED_FIELDS: tuple = ("value_dist_bins", "value_dist_vmin", "value_dist_vmax",
                               "value_dist_coef",
                               # L2: the lambda truncation convention and the rollout count / mode are
                               # inert without their (retired) lever above.
                               "win_prob_lambda_truncated", "win_prob_rollout_r",
                               "win_prob_rollout_mode")

#: Every field `_migrate_config` pops, however its value reads.
RETIRED_FIELDS: tuple = tuple(r.field for r in RETIRED) + INERT_RETIRED_FIELDS

#: field -> the CLI flag (with its `--`), for messages. Built from the table so it cannot drift.
RETIRED_FIELD_FLAGS: Dict[str, str] = {r.field: "--" + r.flag for r in RETIRED}


class RetiredLeverCheckpointError(ModelVersionError):
    """A resume or fork of a checkpoint that recorded a DELETED lever ON. Typed, so a caller (and a
    test) can tell it from every other version refusal; a `ModelVersionError`, so every existing
    `except ModelVersionError` exits FATAL_CONFIG on it."""

    def __init__(self, message: str, *, evidence: List[str], config_path: Optional[str],
                 last_commit: str):
        super().__init__(message)
        self.evidence = list(evidence)
        self.config_path = config_path
        #: A pin at or before this commit still has EVERY lever the evidence names.
        self.last_commit = last_commit


def retired_lever_evidence(raw: Dict[str, Any], *, structural_only: bool = False) -> List[RetiredLever]:
    """The retired levers `raw` (a `model_config.json` dict, UNMIGRATED) recorded ON, in table order."""
    return [r for r in RETIRED if (r.structural or not structural_only) and r.active(raw)]


def refuse_structural(raw: Dict[str, Any]) -> None:
    """`_migrate_config`'s half: refuse, on EVERY load, a config whose recorded ON lever named
    parameters or a critic route this code cannot rebuild."""
    for r in retired_lever_evidence(raw, structural_only=True):
        raise ModelVersionError(
            f"{r.what} is no longer supported: the lever was DELETED (deletion pass {r.unit}, config v"
            f"{r.version}; --{r.flag}), and its ON value named parameters or a critic route "
            "the surviving code cannot rebuild, so this checkpoint's weights cannot be loaded as "
            "they were trained.\n"
            f"To re-read it, use the git_hash in its own metadata.json (any commit at or before "
            f"{r.last_commit[:8]} has the lever).")


def refusal_message(evidence: List[RetiredLever], config_path: Optional[str]) -> str:
    where = config_path or "its model_config.json"
    lines = "\n".join(f"  · {e.what}  (--{e.flag})" for e in evidence)
    pin = evidence[0].last_commit
    commits = sorted({e.last_commit[:8] for e in evidence})
    return (
        f"this checkpoint was TRAINED WITH A DELETED LEVER ({where}):\n{lines}\n"
        "These levers were DELETED (deletion pass, designs/ops/deletion_pass_manifest.md §1-2). "
        "Resuming or forking it on this code would continue it WITHOUT the lever — a different "
        "objective or different numerics under the same run name — so it is refused rather than "
        "silently switched.\n"
        f"Fix: run it pinned to a commit that still has the lever, ≤ {' / '.join(commits)} "
        f"(`--pin-commit {pin[:8]}`, or the checkpoint's own recorded git_hash, which the launcher "
        "pins a restart to by default). A fresh run without the lever is a NEW experiment, not a "
        "continuation — start it without --model.")


def check_no_retired_levers(config_path: Optional[str]) -> None:
    """Raise :class:`RetiredLeverCheckpointError` when the config at `config_path` recorded a
    deleted lever ON. A missing path is not a verdict (the caller's own unreadable-config handling
    applies); an unparseable file is left to the loader that will fail on it loudly anyway."""
    if not config_path or not os.path.exists(config_path):
        return
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    # the runtime knobs live in the run's metadata.json, beside the model_config.json
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(config_path)), "metadata.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict):
        raw = {**raw, **{k: meta[k] for k in RUNTIME_RECORD_FIELDS if k in meta}}
    evidence = retired_lever_evidence(raw)
    if evidence:
        raise RetiredLeverCheckpointError(
            refusal_message(evidence, config_path),
            evidence=[f"{e.what} (--{e.flag})" for e in evidence], config_path=config_path,
            last_commit=evidence[0].last_commit)
