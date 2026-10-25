"""Which file a read plays, and its bytes (``designs/tools/reads.md`` §3). ``models/`` is only READ here.

THE SETTLED RULE (a ``final`` ref). A run's final is ``<run>/final_model.zip`` and nothing else: it is SETTLED iff
the file exists, its recorded step is >= ``min_step``, nothing has written it for ``quiet_s`` seconds, and (when the
spec names a ``done_marker``) the marker file holds the line. Until then the answer is :class:`NotReady`, never a
checkpoint: x26c's rule fell back to "the newest quiet checkpoint >= 480M" and the Kakuna chain twice started on a
checkpoint 81k steps short while the end chain was still writing the final (x26c FINDING 2). A read that wants a
checkpoint names it as a ``pinned`` ref with its sha256.

THE COPY. A resolved model is copied into the read's state (``copy_checked``): the copy must hash to the source, the
source to a pinned sha256, and the copy carries the arch provenance ``main.h2h`` / ``main.anchors`` resolve beside a
zip -- the checkpoint's own ``.json`` sidecar and the run's ``model_config.json`` (the zip's dir, then its parent).
No ``model_config.json`` is a refusal at COPY time (D1 look 2's void ``h_d``: copied finals without it refused every
unit, d1_read FINDING 4). Every later use re-hashes the copy (:func:`verify`).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from main.reads.spec import Model, ModelRef


class ModelError(RuntimeError):
    """A model reference that cannot be honoured (named; nothing is played on it)."""


@dataclass(frozen=True)
class NotReady:
    why: str


@dataclass(frozen=True)
class Resolved:
    """A model's source file, its sha256 and recorded step (before the copy)."""

    label: str
    source: str
    sha256: str
    step: int


def sha256_file(path: "str | os.PathLike[str]") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_step(zip_path: "str | os.PathLike[str]") -> int:
    """The step a checkpoint records (``fixed_opponent_pool.resolve_model_ref``); none is a :class:`ModelError`."""
    from agents.training.fixed_opponent_pool import resolve_model_ref

    try:
        r = resolve_model_ref(str(zip_path), warn=False)
    except (FileNotFoundError, ValueError) as exc:
        raise ModelError(f"{zip_path}: does not resolve as a model ({exc})") from exc
    if r.num_timesteps is None:
        raise ModelError(f"{zip_path}: no recorded step")
    return int(r.num_timesteps)


def spec_path(p: str) -> Path:
    """A spec's path: absolute as written; ``models/...`` under the MAIN checkout's run archive; else repo-relative."""
    from utils.paths import main_models_dir, repo_path

    if os.path.isabs(p):
        return Path(p)
    head, _, rest = p.partition("/")
    if head == "models":
        root = main_models_dir()
        if root is None:
            raise ModelError(f"{p}: no run archive here (main_models_dir() is None)")
        return root / rest
    return repo_path(p)


def settled_final(ref: ModelRef, *, now: Optional[float] = None,
                  step_of: Callable[[str], int] = recorded_step) -> "Path | NotReady":
    """THE SETTLED RULE (module docstring) for a ``final`` ref."""
    assert ref.kind == "final" and ref.run is not None and ref.min_step is not None
    t = time.time() if now is None else now
    z = spec_path(ref.run) / "final_model.zip"
    if not z.is_file():
        return NotReady(f"{z}: no final yet")
    age = t - z.stat().st_mtime
    if age < ref.quiet_s:
        return NotReady(f"{z}: written {age:.0f} s ago (< {ref.quiet_s} s quiet)")
    if ref.done_marker is not None:
        f, line = ref.done_marker
        try:
            text = spec_path(os.path.expanduser(f)).read_text()
        except OSError:
            return NotReady(f"done marker {f}: unreadable")
        if line not in text:
            return NotReady(f"done marker {f}: no line {line!r}")
    step = step_of(str(z))
    if step < ref.min_step:
        return NotReady(f"{z}: step {step:,} < {ref.min_step:,}")
    return z


def resolve(m: Model, *, now: Optional[float] = None,
            step_of: Callable[[str], int] = recorded_step) -> "Resolved | NotReady":
    """The source file of ``m`` (pinned: its path, hashed against the spec; final: the settled final)."""
    if m.ref.kind == "pinned":
        assert m.ref.path is not None
        p = spec_path(m.ref.path)
        if not p.is_file():
            raise ModelError(f"{m.label}: pinned file {p} is missing")
        sha = sha256_file(p)
        if sha != m.ref.sha256:
            raise ModelError(f"{m.label}: {p} sha256 {sha[:12]} != the spec's {str(m.ref.sha256)[:12]}")
        return Resolved(m.label, str(p), sha, step_of(str(p)))
    z = settled_final(m.ref, now=now, step_of=step_of)
    if isinstance(z, NotReady):
        return z
    return Resolved(m.label, str(z), sha256_file(z), step_of(str(z)))


def copy_checked(src: Path, dst_dir: Path, want_sha: Optional[str] = None) -> "tuple[Path, str]":
    """Copy ``src`` (read-only) into ``dst_dir`` with its provenance sidecars (module docstring)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    src_sha = sha256_file(src)
    if want_sha is not None and src_sha != want_sha:
        raise ModelError(f"{src}: sha256 {src_sha} differs from the expected {want_sha}")
    config = next((d / "model_config.json" for d in (src.parent, src.parent.parent)
                   if (d / "model_config.json").is_file()), None)
    if config is None:
        raise ModelError(f"{src}: no model_config.json beside it or in its run dir (h2h / anchors refuse such a zip)")
    before = src.stat()
    if not dst.exists() or sha256_file(dst) != src_sha:
        tmp = dst.with_suffix(".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    after = src.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size) or sha256_file(src) != src_sha:
        raise ModelError(f"{src}: written while it was copied (mtime / size / sha moved): not settled")
    got = sha256_file(dst)
    if got != src_sha:
        raise ModelError(f"{dst}: the copy's sha256 {got} differs from the source's {src_sha}")
    side = src.with_suffix(".json")
    if side.is_file():
        shutil.copyfile(side, dst_dir / side.name)
    shutil.copyfile(config, dst_dir / "model_config.json")
    return dst, got


def copy_resolved(r: Resolved, dst_dir: Path) -> Dict[str, Any]:
    """Copy a resolved model; the state's ``models.json`` entry."""
    dst, sha = copy_checked(Path(r.source), dst_dir, r.sha256)
    return {"label": r.label, "source": r.source, "sha256": sha, "step": r.step, "copy": str(dst),
            "copied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def verify(entry: Dict[str, Any]) -> None:
    """Re-hash a copied model before it is played or read (a changed byte is a refusal)."""
    got = sha256_file(entry["copy"])
    if got != entry["sha256"]:
        raise ModelError(f"{entry['label']}: {entry['copy']} sha256 {got[:12]} != recorded {entry['sha256'][:12]}")
