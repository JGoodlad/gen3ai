"""MIRRORED TEAM PAIRS for an external-anchor read (T17, ``gen3_mirrored_pairs_v1``, ``--mirrored-pairs``).

A pair is two consecutive battles of one HALF (one challenger, so the seats do not move): battle ``2k`` our
side pilots ``A_k`` (drawn from OUR team source) against the peer on ``B_k`` (drawn from the PEER's team
set); battle ``2k+1`` the teams are HANDED OVER — we pilot ``B_k``, the peer pilots ``A_k``. The front end
(``--pair-seeds``) keys each battle's seed by its unordered team pair, so both games of a pair share one
seed whatever ran between them. Team-draw luck then cancels inside the pair.

Nothing here is trusted on faith. Each side's team for each battle is RECORDED (our draw log; the peer's
``team_draws``), and so is every battle's seed (the front end's ``--pair-log``); a pair counts only when
both games finished, both sides played the planned teams in the planned order, and the two battles shared
a seed. Anything else VOIDS the pair, by name — a half pair is not a measurement.

THE PAIR IS THE UNIT: the read's interval is the pentanomial one over pairs
(:mod:`agents.training.mirrored_pairs`), a game worth 2 / 1 / 0 half-points for a win / draw / loss, where
a battle our side forfeited at the turn limit is a DRAW (the trainer's own convention,
``trace_result.classify_result``: a timeout pays the draw penalty, never a decisive loss).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.training import mirrored_pairs as MP

SCHEMA = MP.SCHEMA


class MirrorError(RuntimeError):
    """A mirrored read that cannot be planned honestly (an odd half, no legal team on a side)."""


@dataclass
class HalfMirror:
    """One half's planned pairs: our side's (label, text) per battle, the peer's team FILE per battle."""

    half: str
    ours: List[Tuple[str, str]] = field(default_factory=list)
    theirs: List[Path] = field(default_factory=list)
    sequence_path: Optional[Path] = None

    @property
    def n_games(self) -> int:
        return len(self.ours)


def _valid(items: Sequence[Tuple[str, str]], battle_format: str) -> List[Tuple[str, str]]:
    """Only the teams OUR validator passes — the sequence builder must not drop one silently (that would
    shift every later battle onto the wrong team)."""
    from utils.bridge.team_validator import validate_teams_locally

    verdicts = validate_teams_locally(battle_format, [t for _, t in items])
    return [it for it, v in zip(items, verdicts) if v.get("valid")]


def plan_half(half: str, n_games: int, our_items: Sequence[Tuple[str, str]],
              their_items: Sequence[Tuple[str, str]], *, seed: int, out_dir: Path,
              battle_format: str = "gen3ou", validate: bool = True) -> HalfMirror:
    """Draw ``n_games // 2`` pairings (``A_k`` from ``our_items``, ``B_k`` from ``their_items``) with a
    private RNG keyed by ``(seed, half)``, write every team as a file under ``out_dir/<half>/mirror_teams``,
    and the peer's battle-ordered file list as ``peer_team_sequence.json``. Deterministic in its inputs."""
    if n_games % 2:
        raise MirrorError(f"half {half}: {n_games} battles is not a whole number of mirrored pairs")
    ours_ok = _valid(our_items, battle_format) if validate else list(our_items)
    theirs_ok = _valid(their_items, battle_format) if validate else list(their_items)
    if not ours_ok or not theirs_ok:
        raise MirrorError(f"half {half}: no legal team on {'our' if not ours_ok else 'the peer'} side")
    rng = random.Random(f"{SCHEMA}:{seed}:{half}")
    d = Path(out_dir) / half / "mirror_teams"
    d.mkdir(parents=True, exist_ok=True)
    hm = HalfMirror(half=half)
    for k in range(n_games // 2):
        (a_name, a_text), (b_name, b_text) = rng.choice(ours_ok), rng.choice(theirs_ok)
        fa = d / f"p{k:04d}_ours_{_safe(a_name)}_team"
        fb = d / f"p{k:04d}_theirs_{_safe(b_name)}_team"
        # The PEER plays this file in the swapped battle — with the Hidden Power IVs spelled out.
        fa.write_text(with_hp_ivs(a_text) + "\n")
        fb.write_text(b_text.strip() + "\n")
        hm.ours += [(fa.name, a_text), (fb.name, b_text)]
        hm.theirs += [fb, fa]
    hm.sequence_path = Path(out_dir) / half / "peer_team_sequence.json"
    hm.sequence_path.write_text(json.dumps([str(p) for p in hm.theirs], indent=1))
    return hm


_IV_NAMES = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


def with_hp_ivs(text: str) -> str:
    """A Showdown export with its gen-3 HIDDEN POWER IVs WRITTEN OUT.

    Our pool's exports leave the Hidden Power IVs implicit — OUR builder derives them
    (``gen3_utils.fix_gen3_hp_ivs``) — but the PEER parses the file with its own stack, which does not,
    and the front end then REJECTS the team ("Forretress has Hidden Power Ghost, but its IVs are for
    Hidden Power Psychic"; measured on the first mirrored read, 2026-10-01). So every team we hand the
    peer carries the IVs our builder would have used. A mon that already states IVs is left alone."""
    from poke_env.teambuilder import Teambuilder

    from utils.gen3_utils import fix_gen3_hp_ivs

    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    mons = fix_gen3_hp_ivs(Teambuilder.parse_showdown_team(text.strip()))
    if len(blocks) != len(mons):
        raise MirrorError(f"cannot align {len(blocks)} export blocks with {len(mons)} parsed Pokemon")
    out = []
    for block, mon in zip(blocks, mons):
        lines = block.strip().splitlines()
        ivs = list(mon.ivs or [31] * 6)
        if any(v != 31 for v in ivs) and not any(ln.strip().startswith("IVs:") for ln in lines):
            iv_line = "IVs: " + " / ".join(f"{v} {n}" for v, n in zip(ivs, _IV_NAMES) if v != 31)
            at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("- ")), len(lines))
            lines.insert(at, iv_line)
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in str(name))[:60]


def game_points(rec: Any) -> Optional[int]:
    """One battle's half-points to OUR side (``BattleRecord``): None when it did not finish."""
    if not getattr(rec, "finished", None):
        return None
    if getattr(rec, "hit_forfeit_limit", False):
        return MP.DRAW_POINTS                    # a turn-limit forfeit is a DRAW, as in training
    result = getattr(rec, "result", None)
    return {"win": MP.WIN_POINTS, "tie": MP.DRAW_POINTS, "loss": MP.LOSS_POINTS}.get(result)


def read_pair_log(path: Optional[Path]) -> Dict[str, List[int]]:
    """``{battle tag: seed}`` from the front end's ``--pair-log`` (empty when absent)."""
    out: Dict[str, List[int]] = {}
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
            out[str(r["tag"])] = list(r["seed"])
        except (ValueError, KeyError):
            continue
    return out


def score_half(hm: HalfMirror, records: Sequence[Any], our_draws: Sequence[Optional[str]],
               their_draws: Sequence[str], seeds: Dict[str, List[int]]) -> Dict[str, Any]:
    """The half's pentanomial, with every VOIDED pair counted by reason."""
    counts = MP.empty_counts()
    void: Dict[str, int] = {}

    def bad(reason: str) -> None:
        void[reason] = void.get(reason, 0) + 1

    for k in range(hm.n_games // 2):
        i, j = 2 * k, 2 * k + 1
        if j >= len(records):
            bad("missing_game")
            continue
        planned_ours = (hm.ours[i][0], hm.ours[j][0])
        planned_theirs = (hm.theirs[i].name, hm.theirs[j].name)
        if tuple(our_draws[i:j + 1]) != planned_ours:
            bad("our_team_off_plan")
            continue
        if tuple(their_draws[i:j + 1]) != planned_theirs:
            bad("peer_team_off_plan")
            continue
        s0, s1 = seeds.get(records[i].battle_tag), seeds.get(records[j].battle_tag)
        if s0 is None or s0 != s1:
            bad("seed_not_shared")
            continue
        p0, p1 = game_points(records[i]), game_points(records[j])
        if p0 is None or p1 is None:
            bad("unfinished_game")
            continue
        counts[p0 + p1] += 1
    return {"half": hm.half, "pair_counts": counts, "voided": void,
            "n_voided": sum(void.values()), "n_planned": hm.n_games // 2}


def summary(halves: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The read's mirrored-pair block: the pooled pentanomial + its PAIR-level interval, and the voids."""
    if not halves:
        return None
    pooled = MP.pooled(h["pair_counts"] for h in halves)
    block = MP.summary(pooled) or {}
    void: Dict[str, int] = {}
    for h in halves:
        for r, n in h["voided"].items():
            void[r] = void.get(r, 0) + n
    block.update({"n_voided": sum(void.values()), "voided": void,
                  "n_planned": sum(h["n_planned"] for h in halves), "by_half": list(halves)})
    return block
