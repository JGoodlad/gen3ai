"""Persistent manual-review annotations (a *funky* flag + a timestamped note log) per decision.

Backs the prober TUI's review mode: while walking a model's own games turn-by-turn, the human
marks decisions that look off and jots why. Notes are an **append log** — each saved comment is a
``{ts, text}`` entry, so the history (and *when* each was added) is preserved rather than
overwritten. Stored as ``<run_dir>/review_notes.json`` keyed by the trace's run-relative path + the
invocation index, so notes survive across sessions and can be exported to markdown. Legacy entries
that stored a single ``note`` string are read transparently. Pure (json + os + datetime only) —
no Textual, unit-testable on its own (the clock is injectable).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Tuple


def _default_clock() -> str:
    """Local wall-clock stamp to the minute — what the note log shows next to each comment."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class ReviewStore:
    """Load/mutate/persist per-decision review annotations for one run dir."""

    FILENAME = "review_notes.json"

    def __init__(self, run_dir: "str | None", *, clock: "Callable[[], str] | None" = None) -> None:
        self._run_dir = run_dir
        self._path = os.path.join(run_dir, self.FILENAME) if run_dir else None
        self._clock = clock or _default_clock
        self._data: Dict[str, Dict[str, dict]] = {}
        if self._path and os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._data = loaded
            except (OSError, ValueError):
                self._data = {}

    # -- reads ---------------------------------------------------------------
    def _entry(self, battle_id: str, inv: int) -> dict:
        return self._data.get(battle_id, {}).get(str(inv), {})

    @staticmethod
    def _has_notes(e: dict) -> bool:
        return bool(e.get("notes") or e.get("note"))

    def flag(self, battle_id: str, inv: int) -> bool:
        return bool(self._entry(battle_id, inv).get("flag"))

    def notes(self, battle_id: str, inv: int) -> List[Tuple[str, str]]:
        """The append log: ``[(timestamp, text), …]`` in the order added. A legacy single
        ``note`` string surfaces as one leading entry with an empty timestamp."""
        e = self._entry(battle_id, inv)
        out: List[Tuple[str, str]] = []
        legacy = e.get("note")
        if legacy:
            out.append(("", str(legacy)))
        for n in e.get("notes", []) or []:
            text = str(n.get("text", ""))
            if text:
                out.append((str(n.get("ts", "")), text))
        return out

    def note(self, battle_id: str, inv: int) -> str:
        """The most recent note's text (back-compat / glyph presence). '' when none."""
        log = self.notes(battle_id, inv)
        return log[-1][1] if log else ""

    def has_annotation(self, battle_id: str, inv: int) -> bool:
        e = self._entry(battle_id, inv)
        return bool(e.get("flag") or self._has_notes(e))

    def annotated_invs(self, battle_id: str) -> List[int]:
        """Invocation indices in this battle that carry a flag or any note, ascending."""
        b = self._data.get(battle_id, {})
        return sorted(int(k) for k, v in b.items() if v.get("flag") or self._has_notes(v))

    def all_annotations(self) -> List[Tuple[str, int, bool, List[Tuple[str, str]]]]:
        """``(battle_id, inv, flag, note_log)`` for every annotated decision across the run,
        sorted by (battle_id, inv) — the export order. ``note_log`` is the ``(ts, text)`` list."""
        out: List[Tuple[str, int, bool, List[Tuple[str, str]]]] = []
        for bid, b in self._data.items():
            for k, v in b.items():
                if v.get("flag") or self._has_notes(v):
                    out.append((bid, int(k), bool(v.get("flag")), self.notes(bid, int(k))))
        return sorted(out, key=lambda t: (t[0], t[1]))

    # -- writes (each persists immediately — the file is tiny) ----------------
    def add_note(self, battle_id: str, inv: int, text: str, *, ts: "str | None" = None) -> None:
        """Append a timestamped comment to a decision's log. Empty/whitespace text is a no-op."""
        text = (text or "").strip()
        if not text:
            return
        e = self._data.setdefault(battle_id, {}).setdefault(str(inv), {})
        e.setdefault("notes", []).append({"ts": ts or self._clock(), "text": text})
        self._save()

    def set(self, battle_id: str, inv: int, *, flag: "bool | None" = None,
            note: "str | None" = None) -> None:
        """Set the flag and/or APPEND a note (kept for callers/tests). Note appends go through
        the same log as :meth:`add_note`; pass an empty note to set the flag alone."""
        b = self._data.setdefault(battle_id, {})
        e = b.setdefault(str(inv), {})
        if flag is not None:
            e["flag"] = bool(flag)
        if note is not None and note.strip():
            e.setdefault("notes", []).append({"ts": self._clock(), "text": note.strip()})
        # Prune empties so annotated_invs / the file stay clean.
        if not e.get("flag") and not self._has_notes(e):
            b.pop(str(inv), None)
            if not b:
                self._data.pop(battle_id, None)
        self._save()

    def toggle_flag(self, battle_id: str, inv: int) -> bool:
        new = not self.flag(battle_id, inv)
        self.set(battle_id, inv, flag=new)
        return new

    def _save(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=1, ensure_ascii=False)
        except OSError:
            pass

    # -- export --------------------------------------------------------------
    def export_markdown(self) -> "str | None":
        """Write every annotation to ``<run_dir>/review_notes.md`` and return the path
        (None if there's no run dir). Overwrites — the json is the source of truth."""
        if not self._run_dir:
            return None
        rows = self.all_annotations()
        lines = ["# Manual review notes", "",
                 f"{len(rows)} annotated decision(s).", ""]
        cur = None
        for bid, inv, flag, note_log in rows:
            if bid != cur:
                lines += ["", f"## {bid}", ""]
                cur = bid
            mark = "⚑ " if flag else ""
            lines.append(f"- **inv {inv}** {mark}".rstrip())
            for ts, text in note_log:
                stamp = f"`{ts}` " if ts else ""
                lines.append(f"  - {stamp}{text}")
        out = os.path.join(self._run_dir, "review_notes.md")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            return None
        return out
