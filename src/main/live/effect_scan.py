"""The drift gate's ENCODER (source) check on the RUST reader (P6 of the poke-env retirement).

Replays only show what a day's games happened to do, so the effect class is ALSO derived from the Showdown SOURCE
the public server runs: :mod:`agents.observation.gen3_effect_sources` scans every ``add('-start' | '-activate' |
'-singleturn' | '-singlemove', …)`` the gen3 format executes (a TEXT scan of the checkout, its computed arguments
expanded by its declared table) into concrete protocol lines. This module EXECUTES each line on the reader the live
client runs — a spectator-style chain (``main.live.reader.LiveReader`` → ``live_reader`` →
``pokesim::side_reader::SideReader``) fed a fixed two-mon preamble and then the line — and ENCODES the result
(``PROBE``): a line the reader refuses, or a volatile the encoder cannot classify (``UnknownVolatileError``), is a
finding, exactly where the live client would halt mid-battle.

It replaces the PYTHON execution of the same lines (``gen3_effect_sources.effect_ids_for_line`` on a ``Gen3Battle`` +
``gen3_effects.encode_volatiles``), which read the client nobody runs after P6. Identity (2026-10-08): on the pinned
``deps/pokemon-showdown`` and on a master checkout (51ad80fa) both executions judged all 93 concrete lines alike (93
clean each), and the Rust probe refuses a fabricated effect (``bogusvolatile``, ``move: Bogus Move``, ``Bogus``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

#: The battle every line is executed on (``gen3_effect_sources._fresh_battle``'s, as protocol text): two actives
#: and one revealed opposing move, so a line that indexes the target's moves (Leppa Berry) can resolve it. The
#: reader is a SPECTATOR (no team), viewer p1; every line targets ``p2a: Snorlax``.
PREAMBLE: Tuple[str, ...] = (
    "|player|p1|p1user||", "|player|p2|p2user||", "|teamsize|p1|6", "|teamsize|p2|6", "|gametype|singles",
    "|gen|3", "|start", "|switch|p1a: Zappy|Zapdos, L100|100/100", "|switch|p2a: Snorlax|Snorlax, L100, M|100/100",
    "|turn|1", "|move|p2a: Snorlax|Tackle|p1a: Zappy")
TARGET = "p2a: Snorlax"

#: The silent effect sites: a ``|move|`` that starts an effect with no announcing line (Minimize).
SILENT_LINES: Tuple[str, ...] = ("|move|p2a: Snorlax|Minimize|p2a: Snorlax",)


def line_text(keyword: str, effect: str, extra: Tuple[str, ...] = ()) -> str:
    return "|".join(["", keyword, TARGET, effect, *extra])


def probe_lines(lines: List[str]) -> Dict[str, str]:
    """Each line → the reader's refusal or the encoder's failure (absent = clean). One reader process, re-opened per
    line; a reader that died is respawned for the next line (the death is that line's finding)."""
    from main.live.reader import LiveReader, ReaderRefusal

    bad: Dict[str, str] = {}
    reader = LiveReader()
    try:
        for text in lines:
            try:
                reader.open("p1", "p1user", None)
                reader.feed(list(PREAMBLE))
                reader.feed([text])
                err = reader.probe()
                if err is not None:
                    bad[text] = f"encode: {err}"
            except ReaderRefusal as exc:
                bad[text] = f"refused ({exc.kind}): {exc}"
                reader.close()
                reader = LiveReader()
    finally:
        reader.close()
    return bad


def source_lines(showdown_root: Path) -> Dict[str, List[str]]:
    """Every concrete effect line the gen3 sim at ``showdown_root`` can announce → its ``file:line`` sources. Raises
    ``gen3_effect_sources.UnresolvedDynamicEffect`` on a computed argument with no declared expansion."""
    from agents.observation import gen3_effect_sources as S

    out: Dict[str, List[str]] = {}
    for (kw, eff, extra), ems in sorted(S.concrete_lines(S.scan_emissions(showdown_root)).items()):
        out.setdefault(line_text(kw, eff, extra), []).extend(f"{e.file}:{e.line}" for e in ems)
    for text in SILENT_LINES:
        out.setdefault(text, []).append("(silent: a |move| that starts an effect)")
    return out


def check(showdown_root: str) -> int:
    """The drift gate's ENCODER (source) check: 0 = every derived line reads and encodes clean, 1 = findings."""
    from agents.observation import gen3_effect_sources as S

    root = Path(showdown_root)
    try:
        chain = S.mod_chain(root)
        if chain != S.GEN3_MOD_CHAIN:
            print(f"[drift] ✗ gen3's mod chain changed: {chain} (the scan walks {S.GEN3_MOD_CHAIN}) — update "
                  "gen3_effect_sources.GEN3_MOD_CHAIN")
            return 1
        lines = source_lines(root)
    except S.UnresolvedDynamicEffect as exc:
        print(f"[drift] ✗ encoder (source): {exc}")
        return 1
    bad = probe_lines(sorted(lines))
    print(f"[drift] encoder (source): {len(lines)} effect lines derived from {root}, each read + encoded by the Rust "
          "reader")
    if bad:
        print("[drift] ✗ effect lines the live reader REFUSES or cannot ENCODE (the live client would halt):")
        for text, why in sorted(bad.items()):
            print(f"     {text}  ({', '.join(lines[text][:2])}): {why[:200]}")
        print("[drift]   fix: classify the effect in the Rust reader / encoder (a slot, or not-a-volatile with where "
              "its information lives), with a revert-failing test.")
        return 1
    print("[drift] ✓ encoder (source): every derived effect line reads and encodes clean.")
    return 0
