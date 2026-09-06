"""WHO FORKED WHOM — the run's ancestry, as a first-class recorded block.

WHY THIS EXISTS. Every exploiter, distillation fold, funding fork and dose arm in this programme is
a FORK of some parent checkpoint, and every comparison the ledger makes (rev-2/3/4 forking one
common base; teachers forked from the target; folds forked from the parent) is a claim about that
graph. Until now the graph was recoverable only by REGEXing `--model` out of a recorded shell
command (`metadata.json`'s `original_command`) — brittle in the obvious ways (a renamed flag, a
quoted path, a `--model=X` spelling) and silent in the worst way: a failed parse reads exactly like
a fresh run. `metadata.json` now states the answer instead of implying it.

THE BLOCK. `metadata.json` grows one top-level `lineage` key::

    "lineage": {
      "schema": 1,
      "role": "fold",                 # exploiter | fold | fork | fresh
      "fork_step": 28115184,          # this run's STARTING num_timesteps
      "recorded_at": "...Z",
      "fork_parent": {                # null on a fresh run
        "path": "models/<run>/final_model.zip",     # exactly as --model gave it
        "resolved_path": "/abs/real/path.zip",
        "run_dir": "/abs/<run>", "run_name": "<run>",
        "git_hash": "...", "arch_signature": "...", "model_config_version": 107,
        "num_timesteps": 28115184, "sha256": "...", "created_at": "...Z"
      },
      "teachers": [ {…same shape…} ],  # every --distill-teacher
      "exploiter_target": {…} | null,  # --exploiter
      "ancestry": [ {run_name, run_dir, git_hash, arch_signature, fork_step, role, source}, … ],
      "ancestry_stop": {"at": "<run>", "reason": "…"}
    }

IMMUTABILITY. Written ONCE, at fork creation, and preserved verbatim across every restart and every
checkpoint — the SAME mechanism `original_command` uses (`agents.model.snapshot.save_model_snapshot`
reads the existing value first and the existing value always wins). A launcher restart therefore
cannot re-point a run's parent at the drifted checkpoint the launcher swapped `--model` to
(`main.launcher.checkpoint.resolve_fork_resume_model`) — the exact failure the distill anchor has a
whole module of prose defending against.

FORK vs RESTART is decided by `main.train.fork_lr.is_same_run_checkpoint`, IMPORTED rather than
re-derived: a `--model` outside the run dir is a FORK, a `<run>/checkpoints/*.zip` (or a legacy
`<run>/*.zip`) is a RESTART. A second predicate for the same question is a second answer waiting to
disagree.

LEGACY RUNS. Every run on disk today predates the block. `fork_parent(run_dir)` returns the recorded
parent when there is one and otherwise DERIVES it from `original_command`, printing
`[lineage] WARNING: derived from original_command (legacy run, pre-lineage)`. So consumers get one
call, the regex lives in exactly one place, and the derived answer is never mistaken for a recorded
one (`ForkParent.derived` says which). `python -m main.lineage --backfill` writes a derived block
into a legacy run's metadata (marked `"derived": true`), dry-run by default.

TORCH-FREE by design — `main.lineage` reads runs whose architecture has drifted past the current
code, which is most of `models/`. A checkpoint's `num_timesteps` comes from the SB3 zip's plain-JSON
`data` member via `zipfile`, never by loading the model.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shlex
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

#: Bumped when the block's SHAPE changes. Readers may key off it; writers always emit the current.
LINEAGE_SCHEMA = 1

#: How far `ancestry` walks before giving up. A generation chain is a handful of links; anything
#: deeper is a bug or a cycle, and both must terminate.
MAX_ANCESTRY_DEPTH = 24

#: The argv spellings argparse accepts for the fork parent (mirrors `distill_anchor_callback`).
_MODEL_FLAGS = ("--model", "--model_path", "--model-path")
_EXPLOITER_FLAGS = ("--exploiter",)
_TEACHER_FLAGS = ("--distill-teacher", "--distill_teacher")

_LEGACY_WARNING = "[lineage] WARNING: derived from original_command (legacy run, pre-lineage)"


# --------------------------------------------------------------------------------------------
# plain filesystem reads
# --------------------------------------------------------------------------------------------
def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        return None
    return obj if isinstance(obj, dict) else None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def sha256_file(path: str, *, chunk: int = 1 << 20) -> Optional[str]:
    """The file's SHA-256, or None if it cannot be read. ~0.1 s on a 36 MB checkpoint."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def checkpoint_num_timesteps(zip_path: str) -> Optional[int]:
    """`num_timesteps` out of an SB3 checkpoint, WITHOUT importing torch or loading the model.

    SB3 stores its python-level state as a plain-JSON `data` member inside the zip, so `zipfile` +
    `json` is the whole read. Falls back to the `checkpoint_<N>_steps.zip` filename convention when
    the zip is unreadable (a groomed-away checkpoint still names its own step)."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read("data"))
        val = data.get("num_timesteps")
        if isinstance(val, (int, float)):
            return int(val)
    except Exception:  # noqa: BLE001 — not a zip, absent, or an SB3 layout we don't know
        pass
    parts = os.path.basename(zip_path).replace(".zip", "").split("_")
    for i, part in enumerate(parts):
        if part.isdigit() and i + 1 < len(parts) and parts[i + 1] == "steps":
            return int(part)
    return None


def resolve_model_path(path: str) -> str:
    """A `--model`-style path made absolute, reaching across to the MAIN checkout when needed.

    `models/` is not committed and exists only in the main checkout, so a `models/<run>/x.zip`
    recorded by a training run does not resolve from a worktree. Falls back to
    `utils.paths.main_models_dir()`; an unresolvable path comes back as a plain abspath so the
    caller records what it was given rather than a guess."""
    if not path:
        return path
    if os.path.exists(path):
        return os.path.realpath(path)
    norm = path.replace("\\", "/")
    tail = norm[len("models/"):] if norm.startswith("models/") else None
    try:
        from utils.paths import main_models_dir
        root = main_models_dir()
    except Exception:  # noqa: BLE001 — no git, no models archive; not an error here
        root = None
    if root is not None:
        for cand in ([os.path.join(str(root), tail)] if tail else []) + [os.path.join(str(root), path)]:
            if os.path.exists(cand):
                return os.path.realpath(cand)
    return os.path.abspath(path)


def run_dir_of(model_path: str) -> Optional[str]:
    """The RUN DIRECTORY a checkpoint belongs to, or None.

    Uses `load_model_snapshot`'s own convention — the zip's directory, then one level up (which is
    what covers `<run>/checkpoints/<name>.zip`, `<run>/best_model/…` and `<run>/warmstart/…`) —
    and picks the first that carries a `metadata.json` or a `model_config.json`."""
    if not model_path:
        return None
    apath = resolve_model_path(model_path)
    first = apath if os.path.isdir(apath) else os.path.dirname(apath)
    for d in (first, os.path.dirname(first)):
        if d and (os.path.exists(os.path.join(d, "metadata.json"))
                  or os.path.exists(os.path.join(d, "model_config.json"))):
            return d
    return first or None


# --------------------------------------------------------------------------------------------
# the recorded shapes
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ForkParent:
    """One model REFERENCE — the fork parent, a distill teacher, or an exploiter target.

    `derived` is False for a reference read out of a recorded `lineage` block and True for one
    inferred from `original_command`. It is the field that keeps a legacy guess distinguishable
    from a recorded fact, and every consumer that reports provenance should say which it got.

    WHICH FILE WAS ACTUALLY LOADED (`gen3_last_snapshot_resolution_v1`). A teacher / target / parent
    is usually named as a run DIRECTORY, and a directory is not a file: it is resolved through
    `agents.training.fixed_opponent_pool.resolve_model_ref`, whose answer depends on what the run
    left on disk. Until 2026-09-06 that answer was the BOT-WIN-RATE `best_model/best_model.zip`, and
    for 2 of 8 R5F teachers it was a ~0.93M-step export rather than the ~2.93M final — with nothing
    recording it (ledger 2026-09-06, probe H8). So the block now states it:

      * `resolved_file`           — the `.zip` the resolver picked, absolute
      * `resolved_num_timesteps`  — its `num_timesteps` (`None` = the zip declares none: unknown)
      * `resolution_rung`         — the rung that fired: `explicit_step` / `explicit_zip` /
                                    `latest_txt` / `highest_checkpoint` / `final_model` /
                                    `best_model_fallback`, or `unresolved`
      * `resolution_rule`         — its coarse class: `explicit_step` / `explicit_zip` /
                                    `last_snapshot` / `best_model_fallback`, or `unresolved`

    `None` on these four means NOT RECORDED — a pre-`gen3_last_snapshot_resolution_v1` run, which
    loaded under the OLD rule and left no trace of which file it got. `unresolved` means this run
    DID try and the path resolved to nothing. The two are different facts and readers must not
    collapse them: `main.lineage` prints "resolved file not recorded (pre
    gen3_last_snapshot_resolution_v1)" for the first and never re-resolves a legacy reference under
    today's rule, which would present a current answer as history.

    `resolved_path` / `num_timesteps` / `sha256` keep their older, narrower meaning — they describe
    the literal path as given, so for a `.zip` reference they agree with the resolved fields and for
    a run-DIRECTORY reference they are about the directory (i.e. mostly `None`)."""

    path: str
    resolved_path: Optional[str] = None
    run_dir: Optional[str] = None
    run_name: Optional[str] = None
    git_hash: Optional[str] = None
    arch_signature: Optional[str] = None
    model_config_version: Optional[int] = None
    num_timesteps: Optional[int] = None
    sha256: Optional[str] = None
    created_at: Optional[str] = None
    derived: bool = False
    resolved_file: Optional[str] = None
    resolved_num_timesteps: Optional[int] = None
    resolution_rung: Optional[str] = None
    resolution_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if not (k == "derived" and not v)}

    @classmethod
    def from_dict(cls, obj: Dict[str, Any], *, derived: bool = False) -> "ForkParent":
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in (obj or {}).items() if k in fields}
        kwargs.setdefault("path", str((obj or {}).get("path") or ""))
        if derived:
            kwargs["derived"] = True
        return cls(**kwargs)  # type: ignore[arg-type]


def _resolve_reference(path: str) -> "Tuple[Optional[str], Optional[int], str, str]":
    """`(resolved_file, num_timesteps, rung, rule)` for a model reference, through THE resolver.

    Calls `agents.training.fixed_opponent_pool.resolve_model_ref` — the same function every load
    path uses — so a recorded provenance and the file the run actually opens cannot disagree.
    Imported lazily because that module reaches back here for `checkpoint_num_timesteps`.

    A reference that resolves nowhere (an archived run, a typo, a coef-0 control arm's absent
    teacher) comes back as `("unresolved", "unresolved")` rather than raising: recording provenance
    must never be able to fail a launch."""
    from agents.training.fixed_opponent_pool import resolve_model_ref
    from agents.training.run_spec import split_run_spec
    try:
        # Resolve the DIRECTORY half against the main checkout first (`models/` is not committed
        # and does not exist in a worktree), then hand the step back to the resolver.
        spec_path, spec_step = split_run_spec(path, what="run spec")
        ref = resolve_model_ref(resolve_model_path(spec_path), spec_step, warn=False)
    except Exception:  # noqa: BLE001 — unresolvable / malformed: a fact, not this module's error
        return None, None, "unresolved", "unresolved"
    return ref.zip_path, ref.num_timesteps, ref.rung, ref.rule


def describe_model(path: str, *, hash_file: bool = True, resolve_ref: bool = True) -> ForkParent:
    """Everything recordable about a model reference, from the files it and its run already wrote.

    Total: a path that resolves nowhere still produces a `ForkParent` carrying the path as given —
    provenance must record what it was told even when it cannot confirm it.

    `resolve_ref=False` LEAVES THE FOUR RESOLUTION FIELDS UNSET, and that is not an optimisation:
    it is what the LEGACY path passes. Deriving a pre-`gen3_last_snapshot_resolution_v1` run's
    teachers from its `original_command` and then resolving them under TODAY'S rule would print a
    current answer as if it were what that run loaded, when the honest answer is that the run
    recorded nothing."""
    resolved = resolve_model_path(path)
    rdir = run_dir_of(path)
    meta = _read_json(os.path.join(rdir, "metadata.json")) if rdir else None
    cfg = _read_json(os.path.join(rdir, "model_config.json")) if rdir else None
    exists = os.path.isfile(resolved)
    cfgv = (cfg or {}).get("config_version")
    rfile, rsteps, rung, rule = (_resolve_reference(path) if resolve_ref
                                 else (None, None, None, None))
    return ForkParent(
        path=path,
        resolved_path=resolved,
        run_dir=rdir,
        run_name=os.path.basename(os.path.normpath(rdir)) if rdir else None,
        git_hash=(meta or {}).get("git_hash"),
        arch_signature=(cfg or {}).get("arch_signature"),
        model_config_version=int(cfgv) if isinstance(cfgv, (int, float)) else None,
        num_timesteps=checkpoint_num_timesteps(resolved) if exists else None,
        sha256=sha256_file(resolved) if (exists and hash_file) else None,
        created_at=_iso(os.path.getmtime(resolved)) if exists else None,
        resolved_file=rfile, resolved_num_timesteps=rsteps,
        resolution_rung=rung, resolution_rule=rule,
    )


# --------------------------------------------------------------------------------------------
# parsing a recorded command (the LEGACY path, and the only regex-ish code in the tree)
# --------------------------------------------------------------------------------------------
def _flag_value(toks: List[str], flags: Tuple[str, ...]) -> Optional[str]:
    for i, tok in enumerate(toks):
        for flag in flags:
            if tok == flag and i + 1 < len(toks):
                return toks[i + 1]
            if tok.startswith(flag + "="):
                return tok[len(flag) + 1:]
    return None


def parse_command(command: str) -> Dict[str, Optional[str]]:
    """`{model, exploiter, distill_teacher}` out of a recorded shell command. Total — a malformed
    command yields all-None rather than raising, because a lineage read must never break a caller."""
    try:
        toks = shlex.split(command or "")
    except ValueError:
        return {"model": None, "exploiter": None, "distill_teacher": None}
    return {
        "model": _flag_value(toks, _MODEL_FLAGS),
        "exploiter": _flag_value(toks, _EXPLOITER_FLAGS),
        "distill_teacher": _flag_value(toks, _TEACHER_FLAGS),
    }


def teacher_paths(spec: Optional[str]) -> List[str]:
    """The TEACHER paths in a `--distill-teacher` spec, reusing the one parser that owns its grammar
    (`agents.training.distill_spec`). The `*` wildcard is stubbed rather than resolved — lineage
    wants the teachers, not their team lists, and resolving would touch the filesystem."""
    if not spec:
        return []
    try:
        from agents.training.distill_spec import parse_distill_teacher_spec
        return [t for t, _teams in parse_distill_teacher_spec(spec, resolve_wildcard=lambda _p: ["*"])]
    except Exception:  # noqa: BLE001 — a malformed spec is not this module's error to raise
        return []


def read_original_command(run_dir: str) -> Optional[str]:
    """`<run_dir>/metadata.json`'s immutable `original_command`, or None."""
    meta = _read_json(os.path.join(run_dir or "", "metadata.json")) or {}
    val = meta.get("original_command")
    return val if isinstance(val, str) and val.strip() else None


def read_num_timesteps(run_dir: str) -> Optional[int]:
    """`<run_dir>/metadata.json`'s top-level `num_timesteps` — HOW FAR THE RUN TRAINED — or None.

    Unlike `original_command` this key is "latest", overwritten by every save
    (`agents.model.snapshot.save_model_snapshot`). `None` means the run predates the key (or was
    saved without a step being known) and the answer is genuinely UNKNOWN — never 0, and never
    silently substituted from the checkpoint zip, which this JSON-only reader will not open.
    """
    meta = _read_json(os.path.join(run_dir or "", "metadata.json")) or {}
    val = meta.get("num_timesteps")
    return int(val) if isinstance(val, (int, float)) else None


# --------------------------------------------------------------------------------------------
# building the block (the WRITE side)
# --------------------------------------------------------------------------------------------
def role_for(*, model_path: Optional[str], exploiter: Optional[str],
             distill_teacher: Optional[str]) -> str:
    """`fresh` | `exploiter` | `fold` | `fork`. `--exploiter` wins over `--distill-teacher`: a
    double-sided exploiter is an exploiter that also distils, and its TARGET is the fact that
    identifies it."""
    if not model_path:
        return "fresh"
    if exploiter:
        return "exploiter"
    if distill_teacher:
        return "fold"
    return "fork"


def build_lineage(*, model_path: Optional[str], model_dir: str,
                  exploiter: Optional[str] = None, distill_teacher: Optional[str] = None,
                  fork_step: Optional[int] = None, hash_parent: bool = True,
                  derived: bool = False) -> Optional[Dict[str, Any]]:
    """The `lineage` block for THIS process, or None when there is nothing new to write.

    None means SAME-RUN RESTART: the run already stated its lineage at fork time and that statement
    is immutable, so a restart contributes nothing. The predicate is imported from
    `main.train.fork_lr`, never re-derived.

    A FRESH run (no `--model`) writes the explicit null form — `fork_parent: null, role: "fresh",
    ancestry: []` — because "the block is absent" and "this run has no parent" are different facts
    and only one of them is a measurement."""
    from main.train.fork_lr import is_same_run_checkpoint

    now = datetime.now(timezone.utc).isoformat()
    if not model_path:
        block: Dict[str, Any] = {
            "schema": LINEAGE_SCHEMA, "role": "fresh", "fork_parent": None,
            "fork_step": int(fork_step or 0), "recorded_at": now,
            "teachers": [], "exploiter_target": None, "ancestry": [],
        }
        if derived:
            block["derived"] = True
        return block

    if is_same_run_checkpoint(model_path, model_dir):
        return None

    # `derived` is the BACKFILL / legacy path: it re-reads an old run's recorded command, so it
    # must not claim to know which file that run resolved (see `describe_model`'s `resolve_ref`).
    _res = not derived
    parent = describe_model(model_path, hash_file=hash_parent, resolve_ref=_res)
    step = fork_step if fork_step is not None else parent.num_timesteps
    chain, stop = ancestry_from_parent(parent)
    block = {
        "schema": LINEAGE_SCHEMA,
        "role": role_for(model_path=model_path, exploiter=exploiter,
                         distill_teacher=distill_teacher),
        "fork_parent": parent.to_dict(),
        "fork_step": int(step) if step is not None else None,
        "recorded_at": now,
        "teachers": [describe_model(t, hash_file=False, resolve_ref=_res).to_dict()
                     for t in teacher_paths(distill_teacher)],
        "exploiter_target": (describe_model(exploiter, hash_file=False, resolve_ref=_res).to_dict()
                             if exploiter else None),
        "ancestry": chain,
    }
    if stop:
        block["ancestry_stop"] = stop
    if derived:
        block["derived"] = True
    return block


def build_lineage_from_command(command: str, *, model_dir: str, hash_parent: bool = True,
                               fork_step: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """`build_lineage` over a recorded `original_command` — the BACKFILL and legacy-derive entry.

    Marked `derived: true` in the block it returns, so a backfilled run never reads as one that
    recorded its own lineage at fork time."""
    parsed = parse_command(command)
    return build_lineage(model_path=parsed["model"], model_dir=model_dir,
                         exploiter=parsed["exploiter"], distill_teacher=parsed["distill_teacher"],
                         fork_step=fork_step, hash_parent=hash_parent, derived=True)


# --------------------------------------------------------------------------------------------
# reading it back (the ACCESSOR — one call, legacy handled in one place)
# --------------------------------------------------------------------------------------------
def read_block(run_dir: str) -> Optional[Dict[str, Any]]:
    """The raw recorded `lineage` block, or None. No derivation, no warning."""
    meta = _read_json(os.path.join(run_dir or "", "metadata.json")) or {}
    block = meta.get("lineage")
    return block if isinstance(block, dict) else None


def fork_parent(run_dir: str, *, warn: bool = True) -> Optional[ForkParent]:
    """THE ACCESSOR. The run's fork parent — recorded if it has one, DERIVED from
    `original_command` otherwise (with a printed warning), None if neither names one.

    A fresh run returns None, as does a legacy run whose command names no `--model`. Consumers that
    need to tell those apart read `role_of(run_dir)`."""
    block = read_block(run_dir)
    if block is not None:
        parent = block.get("fork_parent")
        return ForkParent.from_dict(parent, derived=bool(block.get("derived"))) if parent else None
    cmd = read_original_command(run_dir)
    if not cmd:
        return None
    got = parse_command(cmd)["model"]
    if not got:
        return None
    if warn:
        print(f"{_LEGACY_WARNING}: {run_dir}", file=sys.stderr)
    # resolve_ref=False: a legacy run loaded under the OLD (best_model-first) rule and recorded
    # nothing. Resolving it now would answer with TODAY'S rule and read as history.
    return dataclasses.replace(describe_model(got, hash_file=False, resolve_ref=False),
                               derived=True)


def role_of(run_dir: str, *, warn: bool = False) -> Optional[str]:
    """The run's `role`, recorded or derived. None when the run states nothing at all."""
    block = read_block(run_dir)
    if block is not None:
        role = block.get("role")
        return str(role) if role else None
    cmd = read_original_command(run_dir)
    if not cmd:
        return None
    if warn:
        print(f"{_LEGACY_WARNING}: {run_dir}", file=sys.stderr)
    p = parse_command(cmd)
    return role_for(model_path=p["model"], exploiter=p["exploiter"],
                    distill_teacher=p["distill_teacher"])


def _node(run_dir: Optional[str], parent: ForkParent) -> Dict[str, Any]:
    block = read_block(run_dir) if run_dir else None
    return {
        "run_name": parent.run_name,
        "run_dir": parent.run_dir,
        "git_hash": parent.git_hash,
        "arch_signature": parent.arch_signature,
        "fork_step": (block or {}).get("fork_step"),
        "role": role_of(run_dir, warn=False) if run_dir else None,
        "model_path": parent.path,
        "source": "lineage" if block is not None else ("original_command" if run_dir else None),
    }


def ancestry_from_parent(parent: ForkParent, *, max_depth: int = MAX_ANCESTRY_DEPTH,
                         ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """`(chain, stop)` walking UP from an immediate parent, nearest ancestor first.

    `stop` says where the chain went dark and why — a chain that simply ends at a fresh root and a
    chain truncated by a missing directory are different facts, and a bare list conflates them.
    Cycle-safe on realpaths (a corrupted or hand-edited block can name its own descendant)."""
    chain: List[Dict[str, Any]] = []
    seen = set()
    cur: Optional[ForkParent] = parent
    while cur is not None:
        rdir = cur.run_dir
        key = os.path.realpath(rdir) if rdir else f"?{cur.path}"
        if key in seen:
            return chain, {"at": cur.run_name or cur.path, "reason": "CYCLE — already visited"}
        seen.add(key)
        chain.append(_node(rdir, cur))
        if len(chain) >= max_depth:
            return chain, {"at": cur.run_name or cur.path,
                           "reason": f"depth limit {max_depth} reached"}
        if not rdir or not os.path.isdir(rdir):
            return chain, {"at": cur.run_name or cur.path,
                           "reason": "parent run directory not on disk"}
        nxt = fork_parent(rdir, warn=False)
        if nxt is None:
            role = role_of(rdir, warn=False)
            reason = ("root: the run records no parent (fresh)" if role == "fresh"
                      else "root: no lineage block and no --model in original_command")
            return chain, {"at": cur.run_name or cur.path, "reason": reason}
        cur = nxt
    return chain, None


def ancestry(run_dir: str, *, max_depth: int = MAX_ANCESTRY_DEPTH, warn: bool = True,
             ) -> List[Dict[str, Any]]:
    """The run's full ancestry chain, nearest ancestor first. `[]` for a fresh or unknown run.

    Recorded when the run has a `lineage` block (which already stores the chain it saw at fork
    time — but this RE-WALKS, so a chain recorded before an ancestor gained its own block gets the
    current answer); derived through the accessor otherwise."""
    parent = fork_parent(run_dir, warn=warn)
    if parent is None:
        return []
    chain, _stop = ancestry_from_parent(parent, max_depth=max_depth)
    return chain


def ancestry_stop(run_dir: str, *, max_depth: int = MAX_ANCESTRY_DEPTH,
                  ) -> Optional[Dict[str, Any]]:
    """Where `ancestry(run_dir)` went dark, and why. None for a run with no parent at all."""
    parent = fork_parent(run_dir, warn=False)
    if parent is None:
        return None
    _chain, stop = ancestry_from_parent(parent, max_depth=max_depth)
    return stop


# --------------------------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------------------------
def check_links(run_dir: str) -> List[str]:
    """Every broken thing about this run's immediate link, as human-readable lines. `[]` = clean.

    Three failures are checkable from what is on disk: the parent directory is gone, the parent
    FILE's sha256 no longer matches what was recorded (the checkpoint was replaced), and the
    architecture signature changed across the link (a fork whose parent is a different
    architecture cannot have loaded it — the recorded parent is wrong)."""
    out: List[str] = []
    parent = fork_parent(run_dir, warn=False)
    if parent is None:
        return out
    if not parent.run_dir or not os.path.isdir(parent.run_dir):
        out.append(f"parent run directory missing: {parent.run_dir or parent.path}")
    resolved = parent.resolved_path or resolve_model_path(parent.path)
    if resolved and not os.path.isfile(resolved):
        out.append(f"parent checkpoint missing on disk: {resolved}")
    elif parent.sha256 and resolved:
        actual = sha256_file(resolved)
        if actual and actual != parent.sha256:
            out.append(f"sha256 MISMATCH on {resolved} — recorded {parent.sha256[:12]}…, "
                       f"on disk {actual[:12]}…")
    cfg = _read_json(os.path.join(run_dir, "model_config.json")) or {}
    mine, theirs = cfg.get("arch_signature"), parent.arch_signature
    if mine and theirs and mine != theirs:
        out.append(f"arch_signature changed across the link: parent {theirs!r} → this run {mine!r}")
    return out
