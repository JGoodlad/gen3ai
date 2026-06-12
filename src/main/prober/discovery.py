"""Trace discovery — pure filesystem, no torch.

Walks the ``eval_traces`` tree a training run writes and builds a browsable
(step → opponent → battle) model for the TUI. The scan is glob-only: it parses
step / opponent / outcome / index from path strings and never opens the JSON or
npz (per-trace metadata is loaded lazily when a battle is selected), so opening a
run with thousands of traces stays instant.

Also resolves the checkpoint to probe with (override > best_model > latest).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

# Roster order mirrors the launcher's _METRIC_ORDER so the prober lists opponents
# the same way the training dashboard does; unknown names fall after, alphabetically.
_OPPONENT_ORDER = [
    "random", "heuristic", "heuristic2", "staller", "staller_v2",
    "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2",
]
# Eval writes either `<outcome>_<idx>` (un-sharded) or `<outcome>_s<shard>_<idx>`
# (the work-stealing eval's per-shard namespacing, `agents.training.eval_callback`'s
# `{outcome}_s{shard}_{idx}` trace_tag). Match both; the `s<shard>_` infix is optional.
_FNAME_RE = re.compile(r"^(win|loss)_(?:s(\d+)_)?(\d+)_summary\.json$")
_STEP_RE = re.compile(r"step_(\d+)")
# Filenames the eval writes (contract with agents.training.eval_callback —
# duplicated here so the prober doesn't import the heavy training module).
_MANIFEST_NAME = "eval_manifest.json"
_SNAPSHOT_NAME = "snapshot.zip"
_CKPT_STEP_RE = re.compile(r"(?:checkpoint|forced)_(\d+)_steps\.zip$")


@dataclass(frozen=True)
class BattleTrace:
    summary_path: str
    npz_path: "str | None"   # None when the _states.npz is missing
    outcome: str             # "win" | "loss" | "?"
    index: int
    opponent: str
    step: int

    @property
    def label(self) -> str:
        idx = f"{self.index:03d}" if self.index else "?"
        return f"{self.outcome} {idx}"


@dataclass(frozen=True)
class OpponentGroup:
    name: str
    battles: "tuple[BattleTrace, ...]"


@dataclass(frozen=True)
class StepGroup:
    step: int
    opponents: "tuple[OpponentGroup, ...]"


@dataclass(frozen=True)
class TraceTree:
    root: str
    run_dir: "str | None"
    steps: "tuple[StepGroup, ...]"
    manifests: "dict[int, dict]"   # step → eval_manifest.json contents (if present)

    @property
    def is_empty(self) -> bool:
        return not self.steps

    def all_battles(self) -> "list[BattleTrace]":
        return [b for s in self.steps for o in s.opponents for b in o.battles]

    def manifest_for(self, step: int) -> "dict | None":
        return self.manifests.get(step)


@dataclass(frozen=True)
class ModelChoice:
    """The checkpoint the prober will load for a given eval step, and why."""
    path: "str | None"     # None when nothing is resolvable
    tier: str              # "override" | "exact" | "nearest" | "recent" | "none"
    detail: str            # human-readable label for the badge
    manifest: "dict | None"
    step: int              # the eval step this choice is for

    @property
    def is_exact(self) -> bool:
        return self.tier in ("exact", "override")


# ---------------------------------------------------------------------------
# Path parsing helpers
# ---------------------------------------------------------------------------

def _opponent_sort_key(name: str):
    try:
        return (0, _OPPONENT_ORDER.index(name))
    except ValueError:
        return (1, name)


def _outcome_sort_key(outcome: str) -> int:
    return {"win": 0, "loss": 1}.get(outcome, 2)


def _parse_trace(summary_path: str) -> BattleTrace:
    base = os.path.basename(summary_path)
    m = _FNAME_RE.match(base)
    if m:
        # Fold the shard into `index` so two shards' same-idx traces (loss_s1_005 /
        # loss_s3_005) stay DISTINCT (unique short_id / sort key). Un-sharded files
        # have shard 0 → index == idx, byte-identical to the pre-shard behaviour.
        outcome = m.group(1)
        shard = int(m.group(2)) if m.group(2) else 0
        index = shard * 1000 + int(m.group(3))
    else:
        outcome, index = "?", 0
    step_m = _STEP_RE.search(summary_path)
    step = int(step_m.group(1)) if step_m else 0
    # opponent = the immediate parent dir, unless it is a step_* / eval_traces dir
    parent = os.path.basename(os.path.dirname(summary_path))
    opponent = parent if not (parent.startswith("step_") or parent == "eval_traces") else "?"
    npz_path = summary_path[: -len("_summary.json")] + "_states.npz"
    return BattleTrace(
        summary_path=summary_path,
        npz_path=npz_path if os.path.exists(npz_path) else None,
        outcome=outcome,
        index=index,
        opponent=opponent,
        step=step,
    )


def _infer_run_dir(path: str) -> "str | None":
    """The run directory (the one containing ``eval_traces/``), if discoverable."""
    p = os.path.abspath(path)
    chain = [p]
    while True:
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
        chain.append(p)
    for anc in chain:
        if os.path.basename(anc) == "eval_traces":
            return os.path.dirname(anc)
        if os.path.isdir(os.path.join(anc, "eval_traces")):
            return anc
    return None


def _summary_paths(path: str) -> "list[str]":
    """All ``*_summary.json`` under the given path (run dir / eval_traces / file)."""
    if os.path.isfile(path):
        return [path] if path.endswith("_summary.json") else []
    # A run dir → descend into eval_traces; otherwise glob the dir itself.
    eval_root = os.path.join(path, "eval_traces")
    root = eval_root if os.path.isdir(eval_root) else path
    return sorted(glob.glob(os.path.join(root, "**", "*_summary.json"), recursive=True))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_trace_tree(path: str) -> TraceTree:
    """Build the (step → opponent → battle) tree for a run dir, eval_traces dir,
    or a single ``*_summary.json``."""
    traces = [_parse_trace(p) for p in _summary_paths(path)]

    by_step: "dict[int, dict[str, list[BattleTrace]]]" = {}
    for t in traces:
        by_step.setdefault(t.step, {}).setdefault(t.opponent, []).append(t)

    run_dir = _infer_run_dir(path)
    steps = []
    manifests: "dict[int, dict]" = {}
    for step in sorted(by_step):
        opponents = []
        for opp in sorted(by_step[step], key=_opponent_sort_key):
            battles = sorted(
                by_step[step][opp],
                key=lambda b: (_outcome_sort_key(b.outcome), b.index),
            )
            opponents.append(OpponentGroup(name=opp, battles=tuple(battles)))
        steps.append(StepGroup(step=step, opponents=tuple(opponents)))
        m = read_eval_manifest(run_dir, step)
        if m is not None:
            manifests[step] = m

    return TraceTree(root=os.path.abspath(path), run_dir=run_dir,
                     steps=tuple(steps), manifests=manifests)


def read_eval_manifest(run_dir: "str | None", step: int) -> "dict | None":
    """Read ``<run_dir>/eval_traces/step_<N>/eval_manifest.json`` if present."""
    if not run_dir:
        return None
    mpath = os.path.join(run_dir, "eval_traces", f"step_{step}", _MANIFEST_NAME)
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_checkpoints(run_dir: "str | None") -> "list[tuple[int, str]]":
    """Step-bearing persisted checkpoints `(step, path)` under run_dir, ascending."""
    if not run_dir:
        return []
    out = []
    for p in glob.glob(os.path.join(run_dir, "*.zip")):
        m = _CKPT_STEP_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_model_config(run_dir: "str | None") -> "dict | None":
    """Read ``<run_dir>/model_config.json`` if present (for an arch badge)."""
    if not run_dir:
        return None
    cfg_path = os.path.join(run_dir, "model_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve_checkpoint(run_dir: "str | None", override: "str | None" = None) -> str:
    """Resolve the checkpoint to probe: override > best_model > latest.

    Raises ``FileNotFoundError`` with an actionable message when none is found.
    """
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"--ckpt not found: {override}")
        return override
    if run_dir:
        best = os.path.join(run_dir, "best_model", "best_model.zip")
        if os.path.exists(best):
            return best
        # Reuse the launcher's run-dir-aware checkpoint finder (latest.txt → max step).
        from main.launcher.checkpoint import find_latest_checkpoint

        latest = find_latest_checkpoint(run_dir, run_dir=run_dir)
        if latest:
            return latest
    raise FileNotFoundError(
        "No checkpoint found. Pass --ckpt <path/to/model.zip> "
        f"(looked under run dir: {run_dir!r})."
    )


# Tier preferences the prober UI cycles through (see resolve_model_for_step).
TIERS = ("auto", "nearest", "recent")


def _exact_snapshot(tree: TraceTree, step: int) -> "str | None":
    """The bit-exact eval snapshot for `step` if the manifest retained it and it exists."""
    m = tree.manifest_for(step)
    if not (m and m.get("snapshot") and tree.run_dir):
        return None
    p = os.path.join(tree.run_dir, "eval_traces", f"step_{step}", m["snapshot"])
    return p if os.path.exists(p) else None


def _nearest_checkpoint(tree: TraceTree, step: int) -> "tuple[str, int] | None":
    ckpts = list_checkpoints(tree.run_dir)
    if not ckpts:
        return None
    cstep, path = min(ckpts, key=lambda sp: (abs(sp[0] - step), -sp[0]))
    return path, cstep


def _most_recent(tree: TraceTree) -> "str | None":
    try:
        return resolve_checkpoint(tree.run_dir, None)
    except FileNotFoundError:
        return None


def resolve_model_for_step(
    tree: TraceTree, step: int, override: "str | None" = None, tier: str = "auto",
) -> ModelChoice:
    """Pick the checkpoint to load for an eval `step` and explain the choice.

    Ladder (tier="auto"): the bit-exact retained eval snapshot → the nearest persisted
    checkpoint by |Δstep| → the most-recent (best_model/latest). `tier="nearest"` skips
    exact; `tier="recent"` forces best/latest. An explicit `override` always wins.
    """
    manifest = tree.manifest_for(step)
    if override:
        return ModelChoice(override, "override", f"override: {os.path.basename(override)}",
                           manifest, step)

    if tier == "auto":
        exact = _exact_snapshot(tree, step)
        if exact:
            return ModelChoice(exact, "exact", f"exact eval snapshot · step {step:,}",
                               manifest, step)

    if tier in ("auto", "nearest"):
        near = _nearest_checkpoint(tree, step)
        if near:
            path, cstep = near
            delta = abs(cstep - step)
            tag = "exact match" if delta == 0 else f"Δ{delta:,}"
            return ModelChoice(path, "nearest",
                               f"nearest checkpoint · step {cstep:,} ({tag})", manifest, step)

    recent = _most_recent(tree)
    if recent:
        return ModelChoice(recent, "recent",
                           f"most recent · {os.path.basename(recent)}", manifest, step)

    return ModelChoice(None, "none", "no checkpoint found (pass --ckpt)", manifest, step)
