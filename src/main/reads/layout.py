"""A read's STATE directory (``designs/tools/reads.md`` §4): where every verb reads and writes. Outside the repo and
``models/``; one per read; nothing in it is ever edited by hand.

    <state>/spec.json                   the spec, byte-copied at plan (its sha256 is the read's identity)
    <state>/code.json                   the playing checkout's commit at plan (a run from another commit refuses)
    <state>/repins.jsonl                APPEND-ONLY: each repin (main.reads.repin) that accepted another playing commit
    <state>/models.json                 {label: resolved + copied model}, written as each model SETTLES
    <state>/ckpts/<label>/              the sha-checked copies (+ model_config.json, sidecar)
    <state>/anchor/<group>/<label>/     a main.x35.kakuna state per (group, model): plan.json, cells/, units.jsonl
    <state>/s/<group>/                  members/ (R's copies), ledger/ (the eval ledger), units.jsonl, logs/
    <state>/cells.json                  assemble's output
    <state>/void.json                   the registered VOID gate, run before decide (main.reads.rules.void_gate)
    <state>/inputs.json, decision.json  decide's input and output
    <state>/record.json                 the record (main.reads.record)
    <state>/<lane>.lock                 the lane's in-state supervisor flock (with the PATH lock outside the state,
                                        main.reads.supervisors); settle.lock serialises model copies
"""
from __future__ import annotations

import os
from pathlib import Path


class StateError(RuntimeError):
    """A state directory that is not this read's, or sits where a read must never write."""


def check_state_dir(state: Path) -> Path:
    """Refuse a state under the repo checkout or the run archive (READ-ONLY on ``models/``)."""
    from utils.paths import main_models_dir, repo_root

    s = Path(os.path.abspath(os.path.expanduser(str(state))))
    for root, what in ((repo_root(), "the repo checkout"), (main_models_dir(), "the run archive models/")):
        if root is None:
            continue
        r = Path(os.path.abspath(str(root)))
        if s == r or r in s.parents:
            raise StateError(f"{s}: a read's state is never inside {what} ({r})")
    return s


def spec_file(state: Path) -> Path:
    return state / "spec.json"


def models_file(state: Path) -> Path:
    return state / "models.json"


def code_file(state: Path) -> Path:
    return state / "code.json"


def ckpt_dir(state: Path, label: str) -> Path:
    return state / "ckpts" / label


def anchor_state(state: Path, group: str, label: str) -> Path:
    return state / "anchor" / group / label


def s_dir(state: Path, group: str) -> Path:
    return state / "s" / group


def s_ledger(state: Path, group: str) -> Path:
    return s_dir(state, group) / "ledger"


def cells_file(state: Path) -> Path:
    return state / "cells.json"


def inputs_file(state: Path) -> Path:
    return state / "inputs.json"


def void_file(state: Path) -> Path:
    return state / "void.json"


def decision_file(state: Path) -> Path:
    return state / "decision.json"


def record_file(state: Path) -> Path:
    return state / "record.json"


def repins_file(state: Path) -> Path:
    return state / "repins.jsonl"


def repin_lock(state: Path) -> Path:
    return state / "repin.lock"


def lane_lock(state: Path, lane: str) -> Path:
    return state / f"{lane}.lock"
