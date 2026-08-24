"""Pool-consistent DETERMINIZATION of the opponent's never-revealed team slots.

This is a productionised port of the hidden-information-floor probe's `hif_lib` (the
`tmp/hif_*.py` scratch family, 2026-08-22), which measured the irreducible hidden-info
variance floor and validated exactly this construction: **535/535 and 615/615 sampled
worlds reproduced the our-side protocol prefix byte-for-byte.** Nothing here is new
science; it is the same surgery with the scratch paths (cached JSON pool, hardcoded
worktree roots) replaced by the live data facade.

Why the construction is EXACT rather than a heuristic
-----------------------------------------------------
* eval draws the opponent's team **uniformly** from the 719-team validated pool
  (`TeamLoader().get_all_teams()` -> `Gen3Teambuilder` -> `random.choice`), so "uniform
  over pool teams consistent with the revealed half" IS the posterior under that
  population — not an approximation of one;
* a mon that has never been ACTIVE keeps its ORIGINAL index in `side.pokemon` for the
  whole battle (Showdown's switch only ever swaps index 0 with the target), and every
  recorded opponent ``switch N`` command targets a mon that thereby becomes active =
  revealed. So replacing never-revealed slots can never re-point a recorded command.

The one real hazard is the PRNG: ``Pokemon``'s constructor calls ``battle.sample(['M','F'])``
for a mon whose packed gender field is empty AND whose species has no fixed dex gender.
Swapping such a mon for a genderless one (or vice versa) SHIFTS the whole dice stream and
the prefix stops reproducing. :func:`_gender_matched` keeps the draw COUNT identical — and
the caller still VERIFIES every world by replaying the prefix and asserting our-side
protocol byte-identity (:func:`prefix_matches`), because a guard that is not checked is a
belief.

⚠️ **WHAT THIS DOES NOT HIDE — the honest arm's known leak.** A REVEALED mon keeps its TRUE
set (EVs, nature, item, ability). It has to: those have already shown themselves in the
damage numbers of the prefix, so changing them changes the observed prefix, which is what
the identity gate forbids. So the "honest" search knows more than a real player would about
mons it has seen. The precedent measured how much that costs: adding every UNUSED move of
every revealed mon (axis M) moved the floor by **-0.0011 [-0.0102, +0.0086]** — a tight null
against the slot axis it strictly contains. The never-revealed SLOT channel is the hidden
information that matters. :func:`swap_unused_moves` implements axis M anyway, for a caller
that wants to shrink the leak further at the cost of a lower gate pass-rate.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

# The reveal tags: a mon that appears on the field by any route is revealed.
_REVEAL_TAGS = ("switch", "drag", "replace")


def to_id(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# packed-team surgery
# ---------------------------------------------------------------------------


class MonSet:
    """ONE packed-team entry, kept as its raw field list so surgery is lossless."""

    __slots__ = ("fields", "species", "rolls_gender")

    def __init__(self, packed_mon: str, gender_tbl: Dict[str, str],
                 aliases: Optional[Dict[str, str]] = None):
        f = packed_mon.split("|")
        while len(f) < 12:
            f.append("")
        self.fields = f
        raw = f[1] or f[0]
        sid = to_id(raw)
        if aliases and sid in aliases:
            sid = to_id(aliases[sid])
        self.species = sid
        # sample(['M','F']) fires only when the packed gender is blank AND the species has no
        # fixed dex gender. That is the ONLY construction-time draw whose COUNT depends on
        # which mon sits in the slot.
        self.rolls_gender = (f[7] not in ("M", "F", "N")) and not gender_tbl.get(sid)

    def packed(self) -> str:
        return "|".join(self.fields)

    def with_fields(self, **kw) -> "MonSet":
        c = MonSet.__new__(MonSet)
        c.fields = list(self.fields)
        for idx, val in kw.items():
            c.fields[int(idx[1:])] = val          # f7=... -> fields[7]
        c.species, c.rolls_gender = self.species, self.rolls_gender
        return c

    def with_gender(self, g: str) -> "MonSet":
        return self.with_fields(f7=g)

    @property
    def moves(self) -> List[str]:
        return [to_id(x) for x in self.fields[4].split(",") if x]


def split_team(packed: str, gender_tbl: Dict[str, str],
               aliases: Optional[Dict[str, str]] = None) -> List[MonSet]:
    return [MonSet(m, gender_tbl, aliases) for m in packed.split("]")]


def join_team(mons: Sequence[MonSet]) -> str:
    return "]".join(m.packed() for m in mons)


def species_gender_table() -> Dict[str, str]:
    """``{species_id: dex gender}`` — ``""`` when the species has no fixed gender (and so
    would consume a construction PRNG draw). Read from the data facade's gen-3 pokedex, not
    a cached scratch JSON."""
    from poke_env.data import GenData

    return {sid: (entry.get("gender") or "")
            for sid, entry in GenData.from_gen(3).pokedex.items()}


def _gender_matched(donor: MonSet, original: MonSet, gender_tbl: Dict[str, str]) -> Optional[MonSet]:
    """A copy of ``donor`` that consumes the SAME number of construction PRNG draws as
    ``original``, or ``None`` if that is impossible. See the module header."""
    ambiguous = not gender_tbl.get(donor.species)      # no fixed dex gender -> a blank rolls
    if original.rolls_gender:                          # 1 draw -> the donor must roll too
        return donor.with_gender("") if ambiguous else None
    # 0 draws -> the donor must not roll: pin an explicit gender on an ambiguous species,
    # leave a fixed-gender / genderless species alone (the dex answers, no sample).
    return donor.with_gender("M") if ambiguous else donor


# ---------------------------------------------------------------------------
# revealed detection (OUR-SIDE protocol only — never the referee view)
# ---------------------------------------------------------------------------


def strip_ts(lines: Sequence[str]) -> List[str]:
    """Drop the wall-clock ``|t:|`` lines the sim stamps at emission time. They are state-
    and obs-invisible (poke-env's ``MESSAGES_TO_IGNORE``) and are the one thing a faithful
    replay legitimately differs on."""
    return [ln for ln in lines if not ln.startswith("|t:|")]


def chunks_to_lines(chunks: Sequence[str]) -> List[str]:
    return [ln for c in chunks for ln in c.split("\n")]


def reveal_events(lines: Sequence[str], opp_side: str) -> List[Tuple[int, str]]:
    """``(line_index, species_id)`` for every OPPONENT mon that appears on the field, in order.

    Species is read from the DETAILS field, never the nickname — this pool carries localized
    nicknames (``Triopikeur`` = Dugtrio) and a nickname-keyed read would name nothing."""
    out: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if not ln.startswith("|"):
            continue
        p = ln.split("|")
        if len(p) < 4 or p[1] not in _REVEAL_TAGS or not p[2].startswith(opp_side):
            continue
        out.append((i, to_id(p[3].split(",")[0])))
    return out


def revealed_species(lines: Sequence[str], opp_side: str) -> set:
    """Every opponent species we have SEEN in ``lines`` (our-side protocol so far)."""
    return {sp for (_i, sp) in reveal_events(lines, opp_side)}


def used_moves_by_species(lines: Sequence[str], opp_side: str) -> Dict[str, set]:
    """``{species_id: {move_id, ...}}`` for every OPPONENT move seen on the field.

    Tracks the active nickname->species binding from the switch lines, because ``|move|``
    names the ACTOR by nickname and this pool carries localized nicknames."""
    active: Dict[str, str] = {}
    used: Dict[str, set] = defaultdict(set)
    for ln in lines:
        if not ln.startswith("|"):
            continue
        p = ln.split("|")
        if len(p) < 4:
            continue
        if p[1] in _REVEAL_TAGS and p[2].startswith(opp_side):
            active[p[2].split(":", 1)[1].strip()] = to_id(p[3].split(",")[0])
        elif p[1] == "move" and p[2].startswith(opp_side):
            sp = active.get(p[2].split(":", 1)[1].strip())
            if sp:
                used[sp].add(to_id(p[3]))
    return dict(used)


def norm_move(mv: str) -> str:
    """Collapse the 16 typed Hidden Powers onto the bare id.

    The protocol names the move ``Hidden Power`` (-> ``hiddenpower``) while the packed set
    says ``hiddenpowerflying``, so an id-equality "was this move used?" test says NO for every
    HP ever thrown. Not cosmetic: the precedent measured **440 of 615 axis-M worlds failing
    the verify gate on this alone**, because it let the builder swap a USED move out and
    invalidate the recorded command that threw it."""
    return "hiddenpower" if mv.startswith("hiddenpower") else mv


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------


def load_pool(teams: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """``(packed_teams, gender_table)`` — the pool exactly as training/eval draws from it.

    ``teams`` injects a pool (tests, a pinned subset); ``None`` builds the real one.
    """
    gender_tbl = species_gender_table()
    if teams is not None:
        return list(teams), gender_tbl
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    return list(Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams), gender_tbl


# ---------------------------------------------------------------------------
# determinization
# ---------------------------------------------------------------------------


def build_determinizations(
    base_team: List[MonSet],
    revealed: set,
    pool_teams: Sequence[Sequence[MonSet]],
    gender_tbl: Dict[str, str],
    *,
    k: int,
    rng: random.Random,
    exclude_true: bool = False,
) -> Tuple[List[dict], dict]:
    """K alternative completions of the opponent's NEVER-REVEALED slots.

    A donor is a WHOLE other pool team; its slots (minus any species already on the target
    team) fill the hidden positions in order. Donor preference is TIERED so the strict
    pool-posterior is used whenever it exists:

      tier 1  the donor's roster CONTAINS every revealed species (uniform over pool teams
              consistent with the revealed half -> the exact posterior)
      tier 2  the donor contains at least one revealed species
      tier 3  any donor

    The tier is reported per world and never silently mixed. ``exclude_true`` drops the true
    completion from the sample — the FLOOR probe wanted alternatives ONLY; a search wants the
    posterior it actually believes, so the default keeps it.
    """
    hidden_slots = [i for i, m in enumerate(base_team) if m.species not in revealed]
    stats = {"n_hidden": len(hidden_slots), "hidden_slots": hidden_slots,
             "tier1_donors": 0, "tier2_donors": 0, "n_built": 0}
    if not hidden_slots or k <= 0:
        return [], stats

    rev = set(revealed)
    tiers: Tuple[List[int], List[int], List[int]] = ([], [], [])
    for j, dm in enumerate(pool_teams):
        sp = {m.species for m in dm}
        if rev and rev <= sp:
            tiers[0].append(j)
        elif rev & sp:
            tiers[1].append(j)
        else:
            tiers[2].append(j)
    stats["tier1_donors"], stats["tier2_donors"] = len(tiers[0]), len(tiers[1])

    order: List[Tuple[int, int]] = []
    for tier, lst in ((1, tiers[0]), (2, tiers[1]), (3, tiers[2])):
        idxs = list(lst)
        rng.shuffle(idxs)
        order.extend((tier, j) for j in idxs)

    true_sig = tuple(sorted(base_team[i].species for i in hidden_slots))
    dets: List[dict] = []
    used_sigs: set = set()
    fixed_species = {base_team[i].species for i in range(len(base_team)) if i not in hidden_slots}
    for tier, j in order:
        if len(dets) >= k:
            break
        donor = pool_teams[j]
        taken, chosen = set(fixed_species), []
        for slot in hidden_slots:
            pick = None
            for dm in donor:
                if dm.species in taken:
                    continue
                cand = _gender_matched(dm, base_team[slot], gender_tbl)
                if cand is None:
                    continue
                pick = cand
                break
            if pick is None:
                break
            taken.add(pick.species)
            chosen.append((slot, pick))
        if len(chosen) != len(hidden_slots):
            continue
        sig = tuple(sorted(p.species for _s, p in chosen))
        if sig in used_sigs or (exclude_true and sig == true_sig):
            continue
        used_sigs.add(sig)
        newteam = list(base_team)
        for slot, pick in chosen:
            newteam[slot] = pick
        dets.append({"tier": tier, "packed": join_team(newteam), "hidden_species": list(sig)})
    stats["n_built"] = len(dets)
    return dets, stats


def swap_unused_moves(team: List[MonSet], revealed: set, used: Dict[str, set],
                      bank: Dict[str, list], rng: random.Random) -> Tuple[List[MonSet], int]:
    """Axis M — replace every UNUSED move slot of every REVEALED mon with a pool alternative
    for that species. Returns ``(new_team, n_slots_changed)``.

    Nothing in the one-sided stream depends on an unused move (PP is invisible for the
    opponent and a move's presence is never announced), so the prefix must still reproduce —
    and the gate checks that rather than assuming it. ``hiddenpower`` is excluded as a
    REPLACEMENT: its type rides the set's IVs, which stay fixed here, so injecting one would
    build a set we did not mean to build.
    """
    out: List[MonSet] = []
    n_changed = 0
    for m in team:
        if m.species not in revealed:
            out.append(m)
            continue
        moves = m.moves
        seen = {norm_move(x) for x in used.get(m.species, set())}
        sets = bank.get(m.species) or []
        cands: List[str] = []
        for donor_set in (rng.sample(sets, k=min(len(sets), 4)) if sets else []):
            cands.extend(mv for mv in donor_set if not mv.startswith("hiddenpower"))
        newmoves, ci = list(moves), 0
        for i, mv in enumerate(moves):
            if norm_move(mv) in seen:
                continue
            while ci < len(cands):
                pick = cands[ci]
                ci += 1
                if pick not in newmoves:
                    newmoves[i] = pick
                    n_changed += 1
                    break
        out.append(m.with_fields(f4=",".join(newmoves)))
    return out, n_changed


def species_move_bank(pool_teams: Sequence[Sequence[MonSet]]) -> Dict[str, list]:
    """Per species, the pool's SETS for it — a list of move-lists, one per pool occurrence.

    ⚠️ Deliberately samples whole SETS, not individual moves from a flat frequency bank. The
    flat version was tried in the precedent and is a MEASUREMENT ARTIFACT: a species' pooled
    bank runs 20-40 moves, most niche, so a uniform draw builds sets no one plays — which made
    the determinized opponents WEAKER and collapsed the across-world variance, so the move axis
    read SMALLER than the slot axis it strictly contains (impossible for a real posterior)."""
    banks: Dict[str, list] = defaultdict(list)
    for t in pool_teams:
        for m in t:
            if m.moves:
                banks[m.species].append(m.moves)
    return dict(banks)


# ---------------------------------------------------------------------------
# record surgery + the verification gate
# ---------------------------------------------------------------------------


def record_with_team(record, side: str, packed: str, tag_suffix: str = ""):
    """A copy of ``record`` whose ``side`` packed team is ``packed``.

    Only the ``>player`` line is touched — ``start_options()`` / ``players()`` are the ONLY
    ``input_log`` readers (``replay_kernels.js::writeStart`` likewise), and ``commands`` are
    side/choice pairs whose referents are unchanged (see the module header)."""
    new_log = []
    for line in record.input_log:
        head = f">player {side} "
        if line.startswith(head):
            payload = json.loads(line[len(head):])
            payload["team"] = packed
            new_log.append(head + json.dumps(payload))
        else:
            new_log.append(line)
    tag = f"{record.battle_tag}{tag_suffix}" if record.battle_tag else record.battle_tag
    return replace(record, input_log=tuple(new_log), battle_tag=tag)


def turn_marker_index(lines: Sequence[str], turn: int) -> Optional[int]:
    """Index of the bare ``|turn|<n>`` line, or ``None``. Exact-match on purpose: ``|turn|1``
    must not be found by a prefix test inside ``|turn|12``."""
    want = f"|turn|{turn}"
    for i, ln in enumerate(lines):
        if ln.rstrip() == want:
            return i
    return None


def prefix_through_turn(lines: Sequence[str], turn: int) -> Optional[List[str]]:
    """Everything up to and INCLUDING the ``|turn|<turn>`` marker.

    Index-based, not turn-arithmetic: everything strictly before the marker has already happened
    (the lead, every prior pivot, a forced replacement at the end of turn T-1), and everything
    after it is turn T's resolution — i.e. after the decision we are about to search."""
    idx = turn_marker_index(lines, turn)
    return None if idx is None else list(lines[: idx + 1])


def prefix_matches(observed_lines: Sequence[str], replayed_chunks: Sequence[str],
                   turn: Optional[int] = None) -> bool:
    """THE GATE. A determinized world is kept only when the OUR-SIDE protocol it regenerates is
    byte-identical (modulo ``|t:|``) to what we actually saw.

    This is what makes the construction a measurement rather than a hope: the precedent ran 535
    and 615 worlds through it with ZERO mismatches, and every world that fails is DROPPED WITH A
    COUNTER, never silently kept.

    ``turn`` scopes BOTH sides to the ``|turn|<turn>`` marker. It is not optional in practice and
    the reason is worth stating: the live stream has run to wherever the battle currently is (and
    carries this turn's ``|request|``), while a search root's prefix stops at the start of turn T.
    Comparing the two whole would fail on trailing content that is not a disagreement about the
    battle — a gate that fires on everything is the same as no gate, because the caller then falls
    back on every decision and the arm silently becomes the control. A missing marker on either
    side is a genuine mismatch and returns ``False``.
    """
    obs = strip_ts(observed_lines)
    rep = strip_ts(chunks_to_lines(replayed_chunks))
    if turn is not None:
        obs_p, rep_p = prefix_through_turn(obs, turn), prefix_through_turn(rep, turn)
        if obs_p is None or rep_p is None:
            return False
        obs, rep = obs_p, rep_p
    return obs == rep
