"""ROLES and SUBSTITUTE PAIRS, from SMOGON data only (`design_x5_belief_tokens.md` §4.1, §4.2 R3).

A role ρ is "carries move m". The role set is DATA-DRIVEN and pre-registered: every move whose Smogon
expected carriers per team, ``Σ_s usage(s) · P(m | s)``, is at least :data:`ROLE_BAR` (0.25), where

* ``usage(s)`` = 6 × the normalised Smogon usage share (``dex_ids.build_species_usage_prior``, from the
  chaos RATING-WEIGHTED set total W = ``Σ Abilities``) — the expected number of copies of ``s`` per team, the T0 prior's own marginal;
* ``P(m | s)`` = ``gen3_data.priors.moves(s)`` — the chaos ``Moves`` per species (§4.1's "from
  ``priors.moves``"). The 16 typed Hidden Powers collapse into ONE role, num 237 ("carries Hidden
  Power"), as ``build_move_prior_logits`` sums them into the 237 presence channel: a set runs at most one.

A SUBSTITUTE PAIR (R3) is two species ``(s1, s2)`` with ``P(m | s) ≥`` :data:`SUBSTITUTE_CARRY` (0.5) for a
shared role ``m`` and a teammate lift below 1 in BOTH directions (they co-occur less than chance) —
the lift is the Smogon chaos ``Teammates`` log-lift the T0 prior uses (``build_species_cooccur_prior``;
a pair Smogon never recorded in either direction has lift exactly 0 = "no information", so it never
qualifies).

**Nothing here reads the team pool** (owner rule 2026-08-15: the pool may MEASURE, never ship as a
prior). The data sources are declared in :data:`SOURCES`; ``roles_test`` runs the derivation in a fresh
interpreter under an ``open`` audit hook and fails if any other data file is read.

**Rule 8.** A role whose carriers sit within :data:`BAR_EPS` of the bar, a species whose carry rate sits
within :data:`BAR_EPS` of :data:`SUBSTITUTE_CARRY`, and a pair whose log-lift sits within :data:`BAR_EPS`
of 0 are EXCLUDED (never decided by rounding) and listed in ``excluded``.

F-X5-41 FIXED (2026-10-04, ``gen3_smogon_prior_denominator_v1``): ``gen3_move_priors.json`` divided the
RATING-WEIGHTED chaos ``Moves`` by the UNWEIGHTED ``Raw count`` (Skarmory Spikes 0.547; per-species sums
0.41–3.75). It is now ``Moves / W`` (Spikes 0.997, sums 4 less the empty-slot mass), and the role set moved
from 15 roles (sha ``4e3ab394…``) to 29 (``daba9995…``; Rapid Spin, Baton Pass, Recover, Softboiled, … join,
none leave). The role set is stamped with its sha, so ``infer`` refuses to pool a read across the fix.
F-X5-47 FIXED (2026-10-04, ``gen3_smogon_species_usage_weighted_v1``): ``usage(s)`` was the UNWEIGHTED
``Raw count`` share (Shuckle's share ×5.2 too high, Tyranitar's 10 % too low); it is now the W share, so the role
set's sha moved again (``infer`` refuses to pool across it).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

#: §4.1's pre-registered bar on a role's Smogon expected carriers per team.
ROLE_BAR = 0.25
#: §4.2 R3: both species carry the shared role at least this often (Smogon P(m | s)).
SUBSTITUTE_CARRY = 0.5
#: Rule-8 exclusion half-width at each pre-registration boundary.
BAR_EPS = 1e-6
#: The team size the usage share is scaled by (expected copies per team).
TEAM = 6
#: The typeless Hidden Power num every typed variant collapses into.
HP_NUM = 237
#: The model's species / move num axes (`Gen3ObservationEncoder.get_layout()`), checked at use.
N_SPECIES = 400
N_MOVES = 400

#: The ONLY data files the derivation may read (``data/pokemon/``; all Smogon-derived, plus the dex
#: files that map ids to nums; the move dex reads the type chart). ``roles_test`` asserts the derivation opens nothing else under ``data/``.
SOURCES = ("gen3_move_priors.json", "gen3_smogon_stats.json", "gen3_teammate_priors.json",
           "gen3_species.json", "gen3_moves.json", "gen3_type_chart.json")


@dataclass(frozen=True)
class Role:
    num: int                 # the move num (237 for any Hidden Power)
    move: str                # the move id (``hiddenpower`` for 237)
    carriers: float          # Smogon expected carriers per team


@dataclass(frozen=True)
class SubstitutePair:
    s1: int                  # species nums, s1 < s2
    s2: int
    name1: str
    name2: str
    role: int                # the shared role's move num
    carry1: float            # Smogon P(role | s1)
    carry2: float
    log_lift_12: float       # log lift of s2 given s1 revealed, and the reverse
    log_lift_21: float


@dataclass(frozen=True)
class RoleSet:
    roles: Tuple[Role, ...]
    pairs: Tuple[SubstitutePair, ...]
    excluded: Dict[str, list]

    @property
    def nums(self) -> List[int]:
        return [r.num for r in self.roles]

    def sha256(self) -> str:
        body = json.dumps({"roles": [asdict(r) for r in self.roles],
                           "pairs": [asdict(p) for p in self.pairs],
                           "bar": ROLE_BAR, "carry": SUBSTITUTE_CARRY}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def to_json(self) -> dict:
        return {"bar": ROLE_BAR, "substitute_carry": SUBSTITUTE_CARRY, "sha256": self.sha256(),
                "roles": [asdict(r) for r in self.roles], "pairs": [asdict(p) for p in self.pairs],
                "excluded": self.excluded, "sources": list(SOURCES)}


def _move_num(move_id: str) -> int:
    from agents import gen3_data

    if move_id.startswith("hiddenpower"):
        return HP_NUM
    md = gen3_data.moves.get(move_id)
    if md is None:
        raise KeyError(f"move {move_id!r} has no dex row")
    return int(md.num)


def smogon_carry_table(n_species: int = N_SPECIES) -> Dict[int, Dict[int, float]]:
    """``{species num: {role num: Smogon P(m | s)}}`` from ``priors.moves``, typed Hidden Powers summed
    into 237 (BASE forms only — a forme shares its base's num; the num-keyed rule)."""
    from agents import gen3_data

    out: Dict[int, Dict[int, float]] = {}
    for sid in gen3_data.species.base_form_ids():
        sd = gen3_data.species.get(sid)
        if sd is None or not (0 < sd.num < n_species):
            continue
        row: Dict[int, float] = {}
        for mid, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(mid)
            if md is None:
                continue
            num = _move_num(mid)
            row[num] = row.get(num, 0.0) + float(p)
        if row:
            out[int(sd.num)] = row
    return out


def derive(n_species: int = N_SPECIES) -> RoleSet:
    """The pre-registered role set and substitute pairs (module docstring). Deterministic: roles in
    move-num order, pairs in (role, s1, s2) order."""
    from agents import gen3_data
    from agents.model.belief_tables import build_species_cooccur_prior
    from agents.model.dex_ids import build_species_usage_prior

    usage = build_species_usage_prior(n_species).double().numpy() * TEAM       # expected copies / team
    carry = smogon_carry_table(n_species)
    expected: Dict[int, float] = {}
    for snum, row in carry.items():
        for m, p in row.items():
            expected[m] = expected.get(m, 0.0) + float(usage[snum]) * p
    excluded: Dict[str, list] = {"roles_near_bar": [], "species_near_carry": [], "pairs_near_lift": []}
    roles = []
    for m in sorted(expected):
        e = expected[m]
        if abs(e - ROLE_BAR) < BAR_EPS:
            excluded["roles_near_bar"].append(m)
            continue
        if e >= ROLE_BAR:
            mid = "hiddenpower" if m == HP_NUM else _move_id_of(m)
            roles.append(Role(num=m, move=mid, carriers=round(e, 12)))
    _, log_lift = build_species_cooccur_prior(n_species)
    ll = log_lift.double().numpy()
    names = {int(gen3_data.species.get(s).num): s for s in gen3_data.species.base_form_ids()  # type: ignore[union-attr]
             if gen3_data.species.get(s) is not None}
    pairs = []
    for r in roles:
        carriers = []
        for snum in sorted(carry):
            p = carry[snum].get(r.num, 0.0)
            if abs(p - SUBSTITUTE_CARRY) < BAR_EPS:
                excluded["species_near_carry"].append([r.num, snum])
            elif p >= SUBSTITUTE_CARRY:
                carriers.append((snum, p))
        for i, (a, pa) in enumerate(carriers):
            for b, pb in carriers[i + 1:]:
                l_ab, l_ba = float(ll[b, a]), float(ll[a, b])     # log_lift[s, t] = evidence t gives s
                if l_ab == 0.0 or l_ba == 0.0:                    # unrecorded: no information, never
                    continue                                      # "less than chance"
                if abs(l_ab) < BAR_EPS or abs(l_ba) < BAR_EPS:
                    excluded["pairs_near_lift"].append([r.num, a, b])
                    continue
                if l_ab < 0.0 and l_ba < 0.0:
                    pairs.append(SubstitutePair(s1=a, s2=b, name1=names.get(a, str(a)),
                                                name2=names.get(b, str(b)), role=r.num,
                                                carry1=round(pa, 12), carry2=round(pb, 12),
                                                log_lift_12=round(l_ab, 12), log_lift_21=round(l_ba, 12)))
    return RoleSet(roles=tuple(roles), pairs=tuple(pairs), excluded=excluded)


def _move_id_of(num: int) -> str:
    from agents import gen3_data

    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        if md is not None and int(md.num) == num:
            return str(mid)
    raise KeyError(f"no move with num {num}")


def prior_move_probs(n_species: int = N_SPECIES, n_moves: int = N_MOVES):
    """``[n_species, n_moves]`` float64 numpy: the network's OWN Smogon move prior as probabilities
    (``build_move_prior_logits`` — ``priors.moves`` + learnset legality + the legal-unobserved floor;
    typed HP summed into 237). The prior column's and OTHER's ``P(m | s)``."""
    import torch

    from agents.model.belief_tables import build_move_prior_logits

    return torch.sigmoid(build_move_prior_logits(n_species, n_moves).double()).numpy()
