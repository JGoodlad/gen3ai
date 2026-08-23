"""The trace's SIBLING files — protocol log, privileged teams, our Hidden Power types.

File IO deliberately kept OUT of the pure engine: the engine takes these as arguments, this is
where they are loaded from disk. Every reader is best-effort and returns None/empty rather than
raising, because a websocket-era trace simply has no `reconstruction.json`.
"""

from __future__ import annotations

import os

from main.prober.engine import build_our_hp_types, parse_protocol_log, protocol_for_turn


class _TraceIOMixin:
    def _protocol_lines(self, b) -> list:
        """The WHOLE parsed protocol log for a battle, from its `*_replay.html` sibling.

        Separate from `_protocol_for` because a turn-by-turn read wants every turn's slice: going
        through `_protocol_for` per decision would re-read and re-parse the same file once per turn
        (249 times on the longest battle). Empty when the replay file is absent/unreadable."""
        replay = b.summary_path[: -len("_summary.json")] + "_replay.html"
        if not os.path.exists(replay):
            return []
        try:
            with open(replay, encoding="utf-8") as f:
                return list(parse_protocol_log(f.read()))
        except Exception:  # noqa: BLE001 — best-effort
            return []

    def _protocol_for(self, b, turn) -> list:
        """Raw Showdown protocol lines for a decision's turn, from the trace's `*_replay.html` sibling
        (the browser-watchable log) — so the JSON `analyze` carries the exact events the summary
        collapses. Empty when the replay file is absent/unreadable."""
        replay = b.summary_path[: -len("_summary.json")] + "_replay.html"
        if not os.path.exists(replay):
            return []
        try:
            with open(replay, encoding="utf-8") as f:
                return list(protocol_for_turn(parse_protocol_log(f.read()), int(turn or 0)))
        except Exception:  # noqa: BLE001 — best-effort
            return []

    def _our_hp_types(self, b) -> "dict | None":
        """OUR team's typed Hidden Power per species from the trace's `reconstruction.json` sibling
        (`{norm_species: 'hiddenpower(bug)'}`), so a bare own HP types in the board — None for
        websocket/older traces. Best-effort; mirrors the TUI's `_load_our_hp_types`."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            return build_our_hp_types(rec.team_details(side)) if side else None
        except Exception:  # noqa: BLE001 — privileged team is best-effort; degrade to bare HP
            return None

    def _opp_team_details(self, b) -> "tuple | None":
        """The opponent's PRIVILEGED `team_details()` (species + evs/ivs/nature/…) from the trace's
        `reconstruction.json` sibling — feeds both the slot-matched belief truth (species) and the
        spread-belief truth (derived stats). Returns `(species_tuple, details_list)` or `None` for
        websocket/older traces. Mirrors the TUI's `_load_opp_team` / `_load_opp_team_details`."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            if not side:
                return None
            opp = "p2" if side == "p1" else "p1"
            details = rec.team_details(opp)
            return tuple(m["species"] for m in details), details
        except Exception:  # noqa: BLE001 — privileged truth is best-effort; degrade to no truth
            return None

    def _our_team_details(self, b) -> "list | None":
        """OUR (trainee's) PRIVILEGED `team_details()` from the trace's `reconstruction.json` sibling
        — feeds the forced-switch switch-in-outgoing panel (true spreads + movesets). `None` for
        websocket/older traces. Mirrors `_opp_team_details` on our own side."""
        recon = b.summary_path[: -len("_summary.json")] + "_reconstruction.json"
        if not os.path.exists(recon):
            return None
        try:
            from utils.bridge.reconstruction import ReconstructionRecord
            rec = ReconstructionRecord.load(recon)
            side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
            return rec.team_details(side) if side else None
        except Exception:  # noqa: BLE001 — privileged team is best-effort
            return None
