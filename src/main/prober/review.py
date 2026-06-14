"""Persistent manual-review annotations (a *funky* flag + a free-text note) per decision.

Backs the prober TUI's review mode: while walking a model's own games turn-by-turn, the human
marks decisions that look off and jots why. Stored as ``<run_dir>/review_notes.json`` keyed by the
trace's run-relative path + the invocation index, so notes survive across sessions and can be
exported to markdown. Pure (json + os only) — no Textual, unit-testable on its own.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple


class ReviewStore:
    """Load/mutate/persist per-decision review annotations for one run dir."""

    FILENAME = "review_notes.json"

    def __init__(self, run_dir: "str | None") -> None:
        self._run_dir = run_dir
        self._path = os.path.join(run_dir, self.FILENAME) if run_dir else None
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

    def flag(self, battle_id: str, inv: int) -> bool:
        return bool(self._entry(battle_id, inv).get("flag"))

    def note(self, battle_id: str, inv: int) -> str:
        return str(self._entry(battle_id, inv).get("note", ""))

    def has_annotation(self, battle_id: str, inv: int) -> bool:
        e = self._entry(battle_id, inv)
        return bool(e.get("flag") or e.get("note"))

    def annotated_invs(self, battle_id: str) -> List[int]:
        """Invocation indices in this battle that carry a flag or a note, ascending."""
        b = self._data.get(battle_id, {})
        return sorted(int(k) for k, v in b.items() if v.get("flag") or v.get("note"))

    def all_annotations(self) -> List[Tuple[str, int, bool, str]]:
        """``(battle_id, inv, flag, note)`` for every annotated decision across the run,
        sorted by (battle_id, inv) — the export order."""
        out: List[Tuple[str, int, bool, str]] = []
        for bid, b in self._data.items():
            for k, v in b.items():
                if v.get("flag") or v.get("note"):
                    out.append((bid, int(k), bool(v.get("flag")), str(v.get("note", ""))))
        return sorted(out, key=lambda t: (t[0], t[1]))

    # -- writes (each persists immediately — the file is tiny) ----------------
    def set(self, battle_id: str, inv: int, *, flag: "bool | None" = None,
            note: "str | None" = None) -> None:
        b = self._data.setdefault(battle_id, {})
        e = b.setdefault(str(inv), {})
        if flag is not None:
            e["flag"] = bool(flag)
        if note is not None:
            e["note"] = note
        # Prune empties so annotated_invs / the file stay clean.
        if not e.get("flag") and not e.get("note"):
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
        for bid, inv, flag, note in rows:
            if bid != cur:
                lines += ["", f"## {bid}", ""]
                cur = bid
            mark = "⚑ " if flag else ""
            lines.append(f"- **inv {inv}** {mark}{note}".rstrip())
        out = os.path.join(self._run_dir, "review_notes.md")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            return None
        return out
