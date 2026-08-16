"""Team-composition SPECIES prior — "given what the opponent has shown, what's in a hidden slot".

Every other belief leg in this codebase fuses a PRIOR with a learned DELTA
(``posterior_logits = prior_logits(context) + head_delta``); the SPECIES belief is the one
exception — a bare ``Linear(D_MODEL, n_species)`` cold-starting ~uniform over ~400 nums. This
derives the missing prior from the pool the runtime actually trains on
(``data/teams/`` via :class:`utils.team_loader.TeamLoader`), as a committed calibration artifact
in the ``gen3_team_archetypes`` / ``gen3_bot_elo_anchors`` / ``gen3_pubval`` pattern:

  ``data/teams/gen3_species_priors.json``
    * ``meta``      — n_teams, git hash, date, the species-id vocabulary basis, smoothing
    * ``marginal``  — ``species_id -> P(species appears on a team)``
    * ``cooccur``   — ``species_id -> {teammate_id -> P(species on team | teammate on team)}``
    * ``counts``    — the raw team/pair counts the two above were derived from (provenance +
      the exact fallback for an unobserved pair)

The consumer-facing primitive is the PURE :func:`species_prior_logits`: a set of REVEALED
opponent species → per-species conditional log-probabilities, num-indexed like every other model
table (``table[species.num]``, so it lines up with ``layout['max_species']`` and
``BeliefHead.species_head``). The math is naive Bayes over pairwise co-occurrence:

    log P(s | R) ∝ log P(s) + Σ_{r ∈ R} log[ P(s | r) / P(s) ]

i.e. the marginal plus a sum of pairwise log-LIFTs — which degrades to exactly the marginal when
nothing is revealed, and to exactly the marginal again for a teammate that carries no information
(``P(s|r) == P(s)``). Species Clause is applied as a hard constraint: a species already revealed
cannot also occupy a hidden slot, so it is floored out.

CLI (writes the artifact + prints a HELD-OUT accuracy evaluation):
    export PYTHONPATH=$PYTHONPATH:src
    python -m agents.training.species_priors [--out data/teams/gen3_species_priors.json]
                                             [--no-eval] [--folds 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from agents.gen3_data import species as g3species

# NOTE: poke-env is imported LAZILY inside `team_species` (the only consumer). Importing it at
# module scope starts poke-env's global asyncio loop THREAD — the same import-side effect that
# made `set_forkserver_preload` hang a 48-env run. (Historically this module was also reached
# from the model layer via `build_species_cooccur_prior`; since 2026-08-15 that prior is
# SMOGON-sourced — owner rule: priors are never pool-based — and this module is a pool-ANALYSIS
# tool only, e.g. the belief-coupling measurement. Its numpy estimator remains the pool's
# naive-Bayes reference.)

ARTIFACT_PATH = os.path.join("data", "teams", "gen3_species_priors.json")

# Shrinkage strength for the CONDITIONAL estimate: P(s|t) = (n_st + m·P(s)) / (n_t + m).
# A teammate seen on few teams shrinks its conditional back toward the marginal (log-lift → 0),
# so a 2-team coincidence can't masquerade as a strong pairwise signal. m is in "pseudo-teams".
SMOOTHING_M = 5.0

# The out-of-vocabulary / excluded floor for the emitted logit vector. NOT -inf: the vector is
# meant to be ADDED to a learned delta, and -inf poisons a softmax gradient. log(1e-9) is far
# below any real pool probability (the rarest species sits at 1/719 ≈ 1.4e-3).
FLOOR_LOGIT = math.log(1e-9)

# The num-indexed width every model buffer uses (``state_encoder.load_mappings``'s
# ``layout['max_species']``). Kept as a default only — pass n_species explicitly from the layout.
DEFAULT_N_SPECIES = 400


# ── pool parsing ─────────────────────────────────────────────────────────────────

def team_species(team_str: str) -> List[str]:
    """The BASE-form species ids of one Showdown export (pure; fail-loud on an unparseable team).

    An alternate FORME is normalized to its ``base_species`` — the artifact's vocabulary is the
    num-keyed basis (``gen3_data.species.base_form_ids()``), because a forme shares its base's
    national-dex ``num`` and every num-indexed consumer would collapse them anyway."""
    from poke_env.data import to_id_str            # lazy — see the module-scope note
    from poke_env.teambuilder import Teambuilder

    mons = Teambuilder.parse_showdown_team(team_str)
    if not mons:
        raise ValueError("unparseable team (no mons)")
    out: List[str] = []
    for mon in mons:
        sid = to_id_str(mon.species or mon.nickname or "")
        sd = g3species.get(sid)
        if sd is None:
            raise ValueError(f"unknown gen3 species id {sid!r} in team")
        out.append(sd.base_species or sd.id)
    # Species Clause: a legal team never repeats a species. Dedupe defensively (order-preserving)
    # so a malformed export can't double-count one species into the marginal.
    seen, uniq = set(), []
    for sid in out:
        if sid not in seen:
            seen.add(sid)
            uniq.append(sid)
    return uniq


# ── the estimator (pure) ─────────────────────────────────────────────────────────

def build_species_priors(teams: Sequence[str], *, smoothing_m: float = SMOOTHING_M) -> dict:
    """Derive the artifact BODY (``marginal`` / ``cooccur`` / ``counts``) from a list of team
    exports. Pure — no I/O, no git, no clock — so the held-out evaluation can refit it per fold."""
    rosters = [team_species(t) for t in teams]
    return build_species_priors_from_rosters(rosters, smoothing_m=smoothing_m)


def build_species_priors_from_rosters(rosters: Sequence[Sequence[str]], *,
                                      smoothing_m: float = SMOOTHING_M) -> dict:
    """The same estimator over already-parsed rosters (the hot path for cross-validation)."""
    n_teams = len(rosters)
    if n_teams == 0:
        raise ValueError("no teams")
    counts: Dict[str, int] = {}
    pair: Dict[str, Dict[str, int]] = {}
    for roster in rosters:
        uniq = sorted(set(roster))
        for sid in uniq:
            counts[sid] = counts.get(sid, 0) + 1
        for i, a in enumerate(uniq):
            row = pair.setdefault(a, {})
            for b in uniq[:i] + uniq[i + 1:]:
                row[b] = row.get(b, 0) + 1

    marginal = {sid: n / n_teams for sid, n in counts.items()}
    cooccur: Dict[str, Dict[str, float]] = {}
    for s, p_s in marginal.items():
        row = {}
        for t, n_t in counts.items():
            if t == s:
                continue
            n_st = pair.get(s, {}).get(t, 0)
            if n_st == 0:
                continue  # unobserved pair → the loader recomputes the shrunk value from counts
            row[t] = (n_st + smoothing_m * p_s) / (n_t + smoothing_m)
        cooccur[s] = row

    return {
        "marginal": marginal,
        "cooccur": cooccur,
        "counts": {"n_teams": n_teams, "team_counts": counts},
        "smoothing_m": smoothing_m,
    }


class SpeciesPriorTable:
    """A light, PURE accessor over an artifact dict — the object the model layer will hold.

    Holds no file handle and no global state, so a unit test (or a CV fold) constructs one from a
    synthetic dict in a line."""

    def __init__(self, marginal: Dict[str, float], cooccur: Dict[str, Dict[str, float]],
                 team_counts: Dict[str, int], n_teams: int, smoothing_m: float = SMOOTHING_M):
        self.marginal = dict(marginal)
        self.cooccur = {s: dict(r) for s, r in cooccur.items()}
        self.team_counts = dict(team_counts)
        self.n_teams = int(n_teams)
        self.smoothing_m = float(smoothing_m)
        self.species = tuple(sorted(self.marginal))
        self._nums = {sid: (g3species.get(sid).num if g3species.get(sid) else 0)
                      for sid in self.species}

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_artifact(cls, artifact: dict) -> "SpeciesPriorTable":
        counts = artifact.get("counts", {})
        return cls(artifact["marginal"], artifact["cooccur"],
                   counts.get("team_counts", {}), counts.get("n_teams", 0),
                   artifact.get("smoothing_m", SMOOTHING_M))

    @classmethod
    def from_teams(cls, teams: Sequence[str], *, smoothing_m: float = SMOOTHING_M):
        return cls.from_artifact(build_species_priors(teams, smoothing_m=smoothing_m))

    @classmethod
    def from_rosters(cls, rosters: Sequence[Sequence[str]], *, smoothing_m: float = SMOOTHING_M):
        return cls.from_artifact(
            build_species_priors_from_rosters(rosters, smoothing_m=smoothing_m))

    # -- probabilities ---------------------------------------------------------
    def num(self, species_id: str) -> int:
        return self._nums.get(species_id, 0)

    def conditional(self, species_id: str, teammate_id: str) -> float:
        """``P(species_id on team | teammate_id on team)``, shrunk toward the marginal.

        An UNOBSERVED pair is not stored (the artifact keeps only ``n_st > 0``); its value is the
        shrinkage limit ``m·P(s) / (n_t + m)`` recomputed here, so a missing entry is a value, not
        a hole. An unknown teammate carries no information → the marginal."""
        p_s = self.marginal.get(species_id, 0.0)
        if teammate_id == species_id:
            return p_s
        row = self.cooccur.get(species_id)
        if row is not None and teammate_id in row:
            return row[teammate_id]
        n_t = self.team_counts.get(teammate_id)
        if n_t is None:
            return p_s          # unknown teammate → no evidence
        return (self.smoothing_m * p_s) / (n_t + self.smoothing_m)

    def log_lift(self, species_id: str, teammate_id: str) -> float:
        """``log[ P(s|t) / P(s) ]`` — the pairwise evidence term, 0 when the teammate is
        uninformative, negative for an anti-correlated pair."""
        p_s = self.marginal.get(species_id, 0.0)
        if p_s <= 0.0:
            return 0.0
        p_st = self.conditional(species_id, teammate_id)
        if p_st <= 0.0:
            # Shrinkage keeps this positive for any species with a nonzero marginal; guard anyway.
            return FLOOR_LOGIT
        return math.log(p_st) - math.log(p_s)

    # -- the consumer primitive ------------------------------------------------
    def conditional_log_probs(self, revealed_ids: Iterable[str], *,
                              exclude_revealed: bool = True) -> Dict[str, float]:
        """``{species_id: log P(species | revealed)}`` over the pool vocabulary (sums to 1).

        Naive Bayes: ``log P(s) + Σ_r log-lift(s, r)``, renormalized. With no revealed species
        this is EXACTLY the marginal."""
        rev = [r for r in dict.fromkeys(revealed_ids)]
        scores: Dict[str, float] = {}
        for s in self.species:
            p_s = self.marginal[s]
            if p_s <= 0.0:
                continue
            if exclude_revealed and s in rev:
                continue
            scores[s] = math.log(p_s) + sum(self.log_lift(s, r) for r in rev)
        if not scores:
            return {}
        hi = max(scores.values())
        tot = sum(math.exp(v - hi) for v in scores.values())
        norm = hi + math.log(tot)
        return {s: v - norm for s, v in scores.items()}

    def logit_vector(self, revealed_ids: Iterable[str], *, n_species: int = DEFAULT_N_SPECIES,
                     exclude_revealed: bool = True) -> np.ndarray:
        """The num-indexed ``[n_species]`` float32 log-probability vector the model adds to its
        head delta. Nums outside the pool vocabulary (and the revealed species, by Species Clause)
        carry :data:`FLOOR_LOGIT` — a finite floor, never ``-inf``."""
        out = np.full(n_species, FLOOR_LOGIT, dtype=np.float32)
        for sid, lp in self.conditional_log_probs(
                revealed_ids, exclude_revealed=exclude_revealed).items():
            num = self.num(sid)
            if 0 < num < n_species:
                out[num] = max(lp, FLOOR_LOGIT)
        return out

    def rank(self, revealed_ids: Iterable[str], *, exclude_revealed: bool = True) -> List[str]:
        """Candidate species ids, most-likely first (the evaluation + inspection view)."""
        lp = self.conditional_log_probs(revealed_ids, exclude_revealed=exclude_revealed)
        return [s for s, _ in sorted(lp.items(), key=lambda kv: (-kv[1], kv[0]))]


# ── the loader (facade convention: lazy singleton, fail-loud) ─────────────────────

_ARTIFACT_CACHE: Optional[dict] = None
_TABLE_CACHE: Optional[SpeciesPriorTable] = None


def load_species_priors(path: str = ARTIFACT_PATH) -> dict:
    """The committed artifact, lazy singleton (raise if missing/empty)."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found — generate it with "
                f"`python -m agents.training.species_priors`")
        with open(path) as fh:
            data = json.load(fh)
        if not data.get("marginal"):
            raise ValueError(f"{path} has no marginal")
        _ARTIFACT_CACHE = data
    return _ARTIFACT_CACHE


def species_prior_table(path: str = ARTIFACT_PATH) -> SpeciesPriorTable:
    """The committed artifact as a :class:`SpeciesPriorTable`, lazy singleton."""
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        _TABLE_CACHE = SpeciesPriorTable.from_artifact(load_species_priors(path))
    return _TABLE_CACHE


def species_prior_logits(revealed_ids: Iterable[Union[str, int]], *,
                         table: Optional[SpeciesPriorTable] = None,
                         n_species: int = DEFAULT_N_SPECIES,
                         exclude_revealed: bool = True,
                         path: str = ARTIFACT_PATH) -> np.ndarray:
    """PURE (given a ``table``): revealed opponent species → ``[n_species]`` num-indexed
    conditional log-probabilities, ready to ADD to the species head's delta.

    ``revealed_ids`` accepts species ID STRINGS or national-dex NUMS (the form the model has), in
    any mix; unknown entries are ignored as evidence. With nothing revealed the result is the
    marginal. Falls back to the committed artifact when no ``table`` is given."""
    tbl = table if table is not None else species_prior_table(path)
    resolved: List[str] = []
    if revealed_ids is not None:
        num_to_id = {tbl.num(s): s for s in tbl.species}
        for r in revealed_ids:
            if isinstance(r, str):
                sd = g3species.get(r)
                sid = (sd.base_species or sd.id) if sd is not None else r
                if sid in tbl.marginal:
                    resolved.append(sid)
            elif r:  # a num; 0 is the UNKNOWN-species sentinel → no evidence
                sid = num_to_id.get(int(r))
                if sid is not None:
                    resolved.append(sid)
    return tbl.logit_vector(resolved, n_species=n_species, exclude_revealed=exclude_revealed)


# ── held-out evaluation ───────────────────────────────────────────────────────────

def evaluate(rosters: Sequence[Sequence[str]], *, folds: int = 5, seed: int = 0,
             ks: Sequence[int] = (0, 1, 2, 3), repeats: int = 3,
             smoothing_m: float = SMOOTHING_M) -> dict:
    """K-fold held-out accuracy at guessing ONE hidden mon of a test team.

    For every test team and every one of its 6 mons as the HIDDEN target, ``k`` of the other 5 are
    revealed and the model ranks the candidates. Two arms share the identical candidate set (both
    drop the revealed species — Species Clause), so the difference isolates the CO-OCCURRENCE
    signal rather than the exclusion:

      * ``marginal``    — rank by ``P(s)`` alone (the baseline the current head would converge to)
      * ``conditional`` — rank by ``P(s | revealed)``

    The fit NEVER sees a test team (fold-disjoint), so these are honest generalization numbers."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rosters))
    fold_of = {int(i): int(f % folds) for f, i in enumerate(idx)}

    res = {k: {"n": 0, "marg_top1": 0, "marg_top3": 0, "cond_top1": 0, "cond_top3": 0}
           for k in ks}
    for f in range(folds):
        train = [rosters[i] for i in range(len(rosters)) if fold_of[i] != f]
        test = [rosters[i] for i in range(len(rosters)) if fold_of[i] == f]
        tbl = SpeciesPriorTable.from_rosters(train, smoothing_m=smoothing_m)
        for roster in test:
            for ti in range(len(roster)):
                target = roster[ti]
                others = [s for j, s in enumerate(roster) if j != ti]
                for k in ks:
                    n_rep = 1 if k == 0 else repeats
                    for _ in range(n_rep):
                        rev = list(rng.choice(others, size=k, replace=False)) if k else []
                        cond = tbl.rank(rev)
                        marg = sorted((s for s in tbl.species if s not in rev),
                                      key=lambda s: (-tbl.marginal[s], s))
                        cell = res[k]
                        cell["n"] += 1
                        cell["marg_top1"] += int(bool(marg) and marg[0] == target)
                        cell["marg_top3"] += int(target in marg[:3])
                        cell["cond_top1"] += int(bool(cond) and cond[0] == target)
                        cell["cond_top3"] += int(target in cond[:3])
    out = {}
    for k, c in res.items():
        n = max(c["n"], 1)
        out[k] = {"n": c["n"],
                  "marginal_top1": c["marg_top1"] / n, "marginal_top3": c["marg_top3"] / n,
                  "conditional_top1": c["cond_top1"] / n, "conditional_top3": c["cond_top3"] / n}
    return out


# ── the CLI: derive + validate + write ────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=ARTIFACT_PATH)
    ap.add_argument("--smoothing-m", type=float, default=SMOOTHING_M)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    from datetime import datetime, timezone

    from utils.git import get_git_hash
    from utils.team_loader import TeamLoader

    loader = TeamLoader()
    teams = loader.get_all_teams()
    rosters = [team_species(t) for t in teams]
    body = build_species_priors_from_rosters(rosters, smoothing_m=args.smoothing_m)
    tbl = SpeciesPriorTable.from_artifact(body)

    artifact = {
        "meta": {
            "n_teams": body["counts"]["n_teams"],
            "n_species": len(body["marginal"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_hash": get_git_hash(),
            "source": "data/teams/ via utils.team_loader.TeamLoader (the training pool)",
            "vocabulary": (
                "species ID strings on the BASE-form basis "
                "(agents.gen3_data.species.base_form_ids(); an alternate forme is normalized to "
                "its base_species, since a forme shares its base's national-dex num). "
                "`species_nums` maps each id to that num — the index every model table "
                "(layout['max_species']) uses."),
            "smoothing": (
                "P(s|t) = (n_st + m*P(s)) / (n_t + m), m = pseudo-teams of shrinkage toward the "
                "marginal. Pairs with n_st == 0 are omitted from `cooccur`; their value is the "
                "shrinkage limit m*P(s)/(n_t+m), recomputed by SpeciesPriorTable.conditional."),
            "smoothing_m": args.smoothing_m,
            "consumer": ("agents.training.species_priors.species_prior_logits(revealed_ids) -> "
                         "np.ndarray[n_species] of conditional log-probs, num-indexed"),
        },
        "species_nums": {sid: tbl.num(sid) for sid in tbl.species},
        "marginal": body["marginal"],
        "cooccur": body["cooccur"],
        "counts": body["counts"],
        "smoothing_m": args.smoothing_m,
    }

    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(artifact, fh, indent=1, sort_keys=True)
        size_kb = os.path.getsize(args.out) / 1024
        print(f"wrote {args.out}: {artifact['meta']['n_teams']} teams, "
              f"{artifact['meta']['n_species']} species, {size_kb:.0f} KB")

    top = sorted(tbl.marginal.items(), key=lambda kv: -kv[1])[:10]
    print("\ntop-10 marginal P(species on a team):")
    for sid, p in top:
        print(f"  {sid:>14} {p:.3f}  (n={tbl.team_counts[sid]})")

    print("\nstrongest pairwise log-lifts (|lift| over pairs both seen on >= 30 teams):")
    lifts: List[Tuple[float, str, str]] = []
    for s in tbl.species:
        if tbl.team_counts[s] < 30:
            continue
        for t in tbl.species:
            if t == s or tbl.team_counts[t] < 30:
                continue
            lifts.append((tbl.log_lift(s, t), s, t))
    lifts.sort(key=lambda x: -x[0])
    for lift, s, t in lifts[:5]:
        print(f"  +{lift:.2f}  P({s} | {t}) = {tbl.conditional(s, t):.3f} vs "
              f"P({s}) = {tbl.marginal[s]:.3f}")
    for lift, s, t in lifts[-5:]:
        print(f"  {lift:.2f}  P({s} | {t}) = {tbl.conditional(s, t):.3f} vs "
              f"P({s}) = {tbl.marginal[s]:.3f}")

    if not args.no_eval:
        print(f"\nheld-out evaluation ({args.folds}-fold, seed {args.seed}, "
              f"{args.repeats} reveal-draws per (team, target, k>0)):")
        ev = evaluate(rosters, folds=args.folds, seed=args.seed, repeats=args.repeats,
                      smoothing_m=args.smoothing_m)
        print(f"  {'k revealed':>10} {'n':>7}  {'marg@1':>7} {'marg@3':>7}  "
              f"{'cond@1':>7} {'cond@3':>7}")
        for k in sorted(ev):
            c = ev[k]
            print(f"  {k:>10} {c['n']:>7}  {c['marginal_top1']:>7.3f} {c['marginal_top3']:>7.3f}"
                  f"  {c['conditional_top1']:>7.3f} {c['conditional_top3']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
