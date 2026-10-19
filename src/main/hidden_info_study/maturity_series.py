"""THE MATURITY BATTERY's checkpoint SERIES: which checkpoints of a named run it reads, how each is copied out, and the
resumable-work rule its queue uses (`designs/endstate/design_hidden_info_binding.md` §12.13.7; backlog T35).

The battery re-runs the stage and dose reads at fixed checkpoints of ONE deep run, every checkpoint loaded on CPU at
the run's own PIN commit (+ an optional PAD commit for the world playouts, as Tier 1 needed for its 15M pins; a run
whose pin already holds the gender draw pad `98146ddb` plays its worlds at the pin itself).

- :data:`SERIES`         the registered runs (name -> :class:`Series`: run dir name, pin, pad, anchor spacing).
- :func:`checkpoint_files`  the run's ``checkpoints/checkpoint_<step>_steps.zip`` files, by step.
- :func:`anchors`        THE TRIGGER: per multiple ``m`` of the spacing (``first``, ``first + every``, ...), the FIRST
                         checkpoint file whose step lies in ``[m, m + every)``. A multiple with no file in that window
                         has no anchor (reported, never filled from a neighbour).
- :func:`pending_anchors`   the anchors the series has not registered yet (the trigger's "new work").
- :func:`plateau_anchor`    the series' DECLARED plateau checkpoint (§10.5's criterion; a NAMED anchor off the 25M
                         grid, labelled ``<series>_P<step in M>M``): the exact file at the declared step, its sha256 checked
                         against the declaration (PREREGISTRATION amendment A1 of the X26 battery).
- :func:`copy_checkpoint`   a READ-ONLY copy out of the run dir, sha256 of the source before and after and of the copy,
                         the copy made read-only; refuses a destination under the run archive.
- :func:`select_new_work`   ``--limit`` caps NEW work only (the E2 fix `6a1cf301`: capping the id list first re-picks
                         done ids forever once they are done).

PURE: stdlib only, no ``main`` / ``agents`` import, so the pin-side drivers load this exact file by path (the
``hand_eval_pin.py`` pattern of this package) while ``PYTHONPATH`` names the pin's ``src``.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)_steps\.zip$")
ANCHOR_EVERY = 25_000_000


@dataclass(frozen=True)
class Series:
    """One deep run the battery reads. ``pad`` is the commit the WORLD playouts run at when it differs from ``pin``
    (None = the pin itself)."""

    name: str
    run: str
    pin: str
    pad: str | None = None
    first: int = ANCHOR_EVERY
    every: int = ANCHOR_EVERY
    #: the DECLARED plateau checkpoint's exact step and sha256 (None = no plateau declared)
    plateau: int | None = None
    plateau_sha256: str | None = None

    def play_commit(self) -> str:
        return self.pad or self.pin


#: the registered series (a new run is one entry; its pin is the commit every one of its checkpoints was saved at)
SERIES: dict[str, Series] = {
    # P* = 280M, declared by the orchestrator 2026-10-19 (the plateau step; PREREGISTRATION amendment A1)
    "x26": Series(name="x26", run="rb_x26_s2601", pin="9c0334890d05785bbfadcb2ecbe3fe2d548efe9b",
                  plateau=280_000_170,
                  plateau_sha256="814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"),
}


class SeriesError(RuntimeError):
    """A checkpoint, copy or series refused (never a silent skip)."""


def checkpoint_files(run_dir: Path) -> list[tuple[int, Path]]:
    """[(step, path)] of ``run_dir/checkpoints/checkpoint_<step>_steps.zip``, ascending by step."""
    d = Path(run_dir) / "checkpoints"
    out = []
    for p in d.iterdir() if d.is_dir() else ():
        m = CHECKPOINT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def anchor_label(series: str, multiple: int) -> str:
    """``x26_025M``: the series and the anchor multiple in millions (zero-padded, so labels sort by step)."""
    if multiple % 1_000_000:
        raise SeriesError(f"anchor multiple {multiple} is not a whole million")
    return f"{series}_{multiple // 1_000_000:03d}M"


def anchors(files: Sequence[tuple[int, Path]], every: int = ANCHOR_EVERY,
            first: int = ANCHOR_EVERY) -> dict[int, tuple[int, Path]]:
    """{multiple: (step, path)}: for every multiple m = first, first + every, ... up to the largest step, the FIRST
    checkpoint with m <= step < m + every. A multiple with no checkpoint in its window is absent."""
    if every <= 0 or first <= 0:
        raise SeriesError("the anchor spacing and the first anchor must be positive")
    srt = sorted(files)
    if not srt:
        return {}
    out: dict[int, tuple[int, Path]] = {}
    m = first
    top = srt[-1][0]
    while m <= top:
        hit = next(((s, p) for s, p in srt if m <= s < m + every), None)
        if hit is not None:
            out[m] = hit
        m += every
    return out


def plateau_label(series: Series) -> str:
    """``x26_P280M``: the series and the plateau step in whole millions (rounded down), marked ``P`` so it never
    collides with a 25M-grid label."""
    if series.plateau is None:
        raise SeriesError(f"{series.name}: no plateau declared")
    return f"{series.name}_P{series.plateau // 1_000_000:03d}M"


def plateau_anchor(files: Sequence[tuple[int, Path]], series: Series) -> tuple[str, tuple[int, Path]]:
    """(label, (step, path)) of the series' DECLARED plateau checkpoint: the file at EXACTLY the declared step (never
    the nearest one) whose sha256 equals the declaration."""
    lab = plateau_label(series)
    hit = next(((s, p) for s, p in files if s == series.plateau), None)
    if hit is None:
        raise SeriesError(f"{series.name}: no checkpoint file at the declared plateau step {series.plateau}")
    got = sha256_file(hit[1])
    if got != series.plateau_sha256:
        raise SeriesError(f"{hit[1]}: sha256 {got} is not the declared plateau's {series.plateau_sha256}")
    return lab, hit


def pending_anchors(files: Sequence[tuple[int, Path]], series: Series,
                    registered: Iterable[str]) -> dict[str, tuple[int, Path]]:
    """{label: (step, path)} of the series' anchors whose label is not yet registered, in step order."""
    reg = set(registered)
    out: dict[str, tuple[int, Path]] = {}
    for m, hit in sorted(anchors(files, series.every, series.first).items()):
        lab = anchor_label(series.name, m)
        if lab not in reg:
            out[lab] = hit
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _under(p: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(p)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def copy_checkpoint(src: Path, dst: Path, forbid_under: Sequence[Path] = ()) -> dict:
    """Copy ``src`` to ``dst`` READ-ONLY on both ends: the source is only opened for reading, its sha256 is taken
    before and after the copy (a file that changed while being copied refuses), the copy's sha256 must equal it, and
    the copy is made read-only (0444). A destination under any of ``forbid_under`` (the run archive) refuses; an
    existing destination with the SAME sha is kept (idempotent), a different one refuses."""
    src, dst = Path(src), Path(dst)
    for root in forbid_under:
        if _under(dst, root):
            raise SeriesError(f"{dst}: a copy may not be written under {root}")
    if not src.is_file():
        raise SeriesError(f"{src}: no such checkpoint file")
    before = sha256_file(src)
    if dst.exists():
        have = sha256_file(dst)
        if have != before:
            raise SeriesError(f"{dst} exists with sha {have}, the source is {before}")
        return {"src": str(src), "dst": str(dst), "sha256": before, "bytes": dst.stat().st_size, "copied": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    after = sha256_file(src)
    got = sha256_file(tmp)
    if after != before:
        tmp.unlink()
        raise SeriesError(f"{src} changed while being copied ({before} -> {after})")
    if got != before:
        tmp.unlink()
        raise SeriesError(f"the copy of {src} has sha {got}, the source {before}")
    os.chmod(tmp, 0o444)
    os.replace(tmp, dst)
    return {"src": str(src), "dst": str(dst), "sha256": before, "bytes": dst.stat().st_size, "copied": True}


def select_new_work(ids: Sequence[str], done: set[str] | Mapping[str, object],
                    limit: int | None) -> list[str]:
    """The ids one call works on: ``ids`` minus the done ones, in the given order, THEN capped at ``limit``. Capping
    first (``ids[:limit]`` before dropping the done ones) re-picks the same finished ids on every call, so the queue
    never advances (E2's phase1 stall, fixed in `6a1cf301`)."""
    todo = [x for x in ids if x not in done]
    return todo[:limit] if limit else todo
