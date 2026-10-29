"""(Re)build ``x62_recomb_v1``: X62's RECOMBINATION corpus, 12,000 new gen3ou sixes made only of the pool's own
STRICT-class mon sets, none sharing more than 2 sets with any one pool team.

    python -m utils.team_corpora.build_x62_recomb [--check]

Spec of record: ``designs/endstate/design_x62_team_recombination.md`` §2 (2.1 the draw, 2.2 STRICT Hidden Power,
2.3 legality, 2.4 size / tiers / manifest). It generalises rc72b's panel builder
(``hctx_rc72_2026-10-26/scripts/build_rc72.py``) to a 12,000-team corpus; the prototype whose numbers §2.2 quotes is
``x62_design_2026-10-26/revision_2026-10-27/strict_draw_sim.py``. **The pool supplies every SET, the pool's species
counts the profile, Smogon only the teammate coherence** (``agents.gen3_data.priors``: the lift). Nothing reads this
corpus as a prior (``utils.team_corpora``'s rule); it is training data, like ``n1_proc_v1``.

THE DRAW of candidate ``j`` (one ``random.Random(stream_seed("corpus", j))``, as in ``build_n1_proc``):

1. SPECIES. Slot 1 with weight ``c_pool(s)``; each later slot among the species not yet on the team (Species Clause by
   construction) with weight ``c_pool(s) * geomean_t lift(t, s)`` over the team's species ``t``, ``lift(t, s) =
   P(s | t) / share(s)`` from Smogon's teammate and usage tables, clipped to [1/4, 4] (rc72b's rules). ``c_pool(s)`` is
   the species' count over the pool's mons, restricted to the species with a STRICT-class occurrence (the prototype's
   profile: see :data:`PROFILE_RULE`).
2. SET. One STRICT-class pool OCCURRENCE of the drawn species, uniform over its occurrences, copied VERBATIM except
   the NICKNAME, which is stripped (:func:`strip_nickname`: ``Nick (Species) (G) @ Item`` -> ``Species (G) @ Item``; a
   head line with no nickname is left alone). STRICT class at the slot = the text filter :func:`strict_text_bad` (a
   Hidden Power naming a type other than Dark with no ``IVs:`` line is never drawn); ``hp_set_check`` (step 4) is the
   authority.
3. DISTANCE REFUSALS at draw time: a slot whose partial team would share > :data:`MAX_SHARED_POOL` (2) mon sets with
   ONE pool team, or >= :data:`HELDOUT_REFUSE` (3) with ONE team of a held-out panel (``heldout72``,
   ``heldout72_hpfix``, ``forum72``, ``p72``, the sealed ``forum216b``), of ``n1p72`` or of ``rc72b``, is redrawn
   (species and set), up to :data:`SLOT_TRIES`; then the candidate fails (``slot_exhausted``). A mon SET is
   ``gen3_team_overlap_v1``'s (species + sorted moves, Hidden Power typed).
4. WHOLE-TEAM CHECKS, in this order (a failure moves to the next candidate; the manifest counts each):
   ``pool_species_six`` (the six species of a pool team), ``level`` (a ``Level:`` line other than 100, the label
   guard), ``validator`` (the pinned ``TeamValidator('gen3ou')`` via ``ladder_corpus_filter.js``; the One Boost
   Passer Clause is the team rule a recombination can break), ``pack_level`` (Showdown's PACKED level,
   ``build_w_forum.level_defects``), ``hidden_power`` (STRICT, :func:`hp_bad_strict` over ``hp_set_check.js``: a Hidden
   Power that does not play its named type at 70 as written, or whose IVs the validator rewrites), ``duplicate`` (the
   six mon sets of a team already accepted) and ``within_gt4`` (> :data:`WITHIN_MAX` shared with ONE accepted team).
   The last two run in CANDIDATE ORDER after the batch's Node calls, so the output does not depend on :data:`BATCH`.

After the draw the engine (``scan_move_probe``) is asked about every species / move / item / ability the corpus
carries and the build THROWS on an unplayable one (decided by this corpus's ENGINE PIN, empty: a pool set the engine
cannot run is a crash, not a refusal), and the readers' label guard (``check_label_encodable``) re-checks the output.

The corpus is the first :data:`N_CORPUS` accepted candidates in candidate order; tiers are PREFIXES (``K1`` ⊂ ``K3``
⊂ ``ALL``). Deterministic: the same pool, panels, priors, validator and seeds give the same bytes (``--check``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path, repo_root
from utils.team_corpora import (CORPORA, DATA_NAME, MANIFEST_NAME, SCHEMA, check_label_encodable, encode_rows,
                                 label_path_defects, tier_sha256)

NAME = "x62_recomb_v1"
GENERATOR = "gen3_x62_recombination_v1"
BUILD_SEED = 6201
N_CORPUS = 12_000
#: Prefix tiers (name -> count), smallest first (§2.4: K1 ⊂ K3 ⊂ ALL serve X64's curve and a later dose read).
TIERS = (("K1", 1_000), ("K3", 3_000), ("ALL", N_CORPUS))
DEFAULT_TIER = "ALL"
#: rc72b's ``MAX_SHARED_TRAIN``: the lever's definition (familiar sets in a new six).
MAX_SHARED_POOL = 2
#: REVIEW A's revealed-fit radius (B2): >= 3 shared with ONE held-out / n1p72 / rc72b team is refused.
HELDOUT_REFUSE = 3
#: At most 4 shared with any corpus team already accepted (near-copies do not spend dose).
WITHIN_MAX = 4
LIFT_CLIP = (0.25, 4.0)
SLOT_TRIES = 80
#: Candidates drawn per Node call (no effect on the output or the counts, see :func:`accept`).
BATCH = 2_000
#: A stream that has not filled after this many candidates is a broken generator, not a slow one.
MAX_CANDIDATES = 40_000
CANDIDATE_RULES = ("slot_exhausted", "pool_species_six", "level", "validator", "pack_level", "hidden_power",
                   "duplicate", "within_gt4")
SLOT_RULES = ("train_overlap", "heldout_fit")
#: The panels refused at radius 3 beside the registry's held-out set.
EXTRA_PANELS = ("n1p72", "rc72b")
N1P72_REL = "designs/research_state/measurements/x61_n1_breadth_2026-10-25/teams_n1p72"
N1P72_MANIFEST_REL = "designs/research_state/measurements/x61_n1_breadth_2026-10-25/n1p72_manifest.json"
#: The other breadth corpora the manifest reports near-duplicates against (§2.4, review m8; reported, never refused).
NEAR_DUP_CORPORA = (("w_forum_v1", "ABC"), ("d3_setmut_v1", "ALL"), ("n1_proc_v1", "ALL"))
PROFILE_RULE = ("c_pool(s) = the species' count over ALL the pool's mons (the prototype strict_draw_sim.py's profile), "
                "for the species with >= 1 STRICT-class occurrence; DEVIATION from §2.1's wording 'over the pool's "
                "STRICT-class mons', declared in the builder's report")
PRIOR_FILES = ("gen3_teammate_priors.json", "gen3_smogon_stats.json")
HP = "hiddenpower"
_HP_TEXT = re.compile(r"hidden power\s*\[?\s*([a-z]+)", re.I)
#: ``Nick (Species) (G)`` / ``Nick (Species)``: a nickname head (the species group is never a bare gender).
_NICK_HEAD = re.compile(r"^(?P<nick>.*?\S)\s*\((?P<species>[^()]+)\)\s*(?P<gender>\((?:M|F)\))?\s*$")


class BuildError(RuntimeError):
    """The build cannot stand behind its output (an unplayable id, a level defect, a stream that does not fill)."""


# ------------------------------------------------------------------------------------------------ the pure core
def stream_seed(*parts: object) -> int:
    """A 64-bit seed from ``BUILD_SEED`` and the parts (sha256; independent of Python's hash randomisation)."""
    h = hashlib.sha256("|".join(str(p) for p in (BUILD_SEED,) + parts).encode()).hexdigest()
    return int(h[:16], 16)


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def pool_source_id(text: str) -> str:
    """A pool team's content id (rc72b's ``pool_<sha12>``)."""
    return "pool_" + sha_text(text)[:12]


def strict_text_bad(block: str) -> bool:
    """The slot's STRICT text filter: True iff the block names a Hidden Power type other than Dark and carries no IV
    line (all-31 IVs give Dark at 70, so that one plays as written). ``hp_set_check`` stays the authority."""
    low = block.lower()
    if "hidden power" not in low or "ivs:" in low:
        return False
    m = _HP_TEXT.search(block)
    return not (m and m.group(1).lower() == "dark")


def strip_nickname(block: str) -> str:
    """The block with its nickname removed from the head line (§2.1 step 2, review m9): ``Nick (Species) (G) @ Item``
    -> ``Species (G) @ Item``; a head with no nickname (``Species @ Item``, ``Species (G) @ Item``) is unchanged, and so
    is every other line."""
    first, nl, rest = block.partition("\n")
    head, at, item = first.partition("@")
    m = _NICK_HEAD.match(head.strip())
    if not m or m.group("species").strip() in ("M", "F"):
        return block
    new = m.group("species").strip() + (" " + m.group("gender") if m.group("gender") else "")
    return (new + (" @" + item if at else "")) + nl + rest


def weighted_pick(rng: random.Random, entries: Sequence[Tuple[str, float]]) -> str:
    total = sum(w for _, w in entries)
    r = rng.random() * total
    for k, w in entries:
        r -= w
        if r < 0:
            return k
    return entries[-1][0]


def split_blocks(text: str) -> List[str]:
    return [b.strip("\n") for b in re.split(r"\n[ \t]*\n+", text.strip()) if b.strip()]


def pool_occurrences(pool: Sequence[str]) -> Dict[str, List[Tuple[str, Any, str]]]:
    """``{species: [(verbatim block, MonSet, source content id), ...]}`` over every mon of every pool team, in the
    pool's (content-sorted) order (rc72b's)."""
    from main.anchors.team_overlap import parse_mon

    out: Dict[str, List[Tuple[str, Any, str]]] = collections.defaultdict(list)
    for t in pool:
        sid = pool_source_id(t)
        for b in split_blocks(t):
            m = parse_mon(b)
            if m is not None:
                out[m.species].append((b, m, sid))
    return dict(out)


def lift_table(species: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """``lift(t, s) = P(s | t) / share(s)`` from Smogon's teammates over usage share, clipped (rc72b's)."""
    from agents.gen3_data import priors

    usage = priors.species_usage()
    tot = sum(usage.values())
    out = {}
    for t in species:
        tm = priors.teammates(t)
        for s in species:
            if s == t:
                continue
            share = usage.get(s, 0.0) / tot
            raw = (tm.get(s, 0.0) / share) if (tm and share > 0) else 1.0
            out[(t, s)] = min(max(raw, LIFT_CLIP[0]), LIFT_CLIP[1])
    return out


def max_shared(refs: Dict[str, Any], mons: Sequence[Any]) -> int:
    from main.anchors.team_overlap import Team

    team = Team(label="cand", mons=tuple(mons))
    return max((ref.max_shared(team)[0] for ref in refs.values()), default=0)


def draw_team(j: int, profile: Dict[str, int], occ: Dict[str, List[Tuple[str, Any, str]]],
              lift: Dict[Tuple[str, str], float], refs: Dict[str, Dict[str, Any]], fails: collections.Counter,
              stream: str = "corpus") -> Optional[List[Tuple[str, Any, str]]]:
    """Candidate ``j``'s six ``(verbatim block, MonSet, source id)``, or None (a slot exhausted its tries).
    ``fails`` counts the slot refusals (``train_overlap`` / ``heldout_fit``). Deterministic in (``BUILD_SEED``,
    ``stream``, ``j``)."""
    rng = random.Random(stream_seed(stream, j))
    team: List[Tuple[str, Any, str]] = []
    for _slot in range(6):
        used = [m.species for _, m, _ in team]
        cands = []
        for s in sorted(profile):
            if s in used:
                continue
            g = math.exp(sum(math.log(lift[(t, s)]) for t in used) / len(used)) if used else 1.0
            cands.append((s, profile[s] * g))
        for _try in range(SLOT_TRIES):
            s = weighted_pick(rng, cands)
            pick = occ[s][rng.randrange(len(occ[s]))]
            mons = [m for _, m, _ in team] + [pick[1]]
            if max_shared(refs["train"], mons) > MAX_SHARED_POOL:
                fails["train_overlap"] += 1
                continue
            if max_shared(refs["held"], mons) >= HELDOUT_REFUSE:
                fails["heldout_fit"] += 1
                continue
            team.append(pick)
            break
        else:
            return None
    return team


def hp_bad_strict(verdict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """rc72b's ``hp_bad`` in its ``strict`` form (X37a's rule): the Hidden Power sets of one ``hp_set_check.js``
    verdict that do not play their named type at 70 AS WRITTEN, or whose IVs the validator rewrites."""
    return [s for s in verdict.get("sets", []) if s.get("hp_move") and (
        s.get("hp_type") != s["hp_move"].split()[-1] or s.get("power") != 70
        or s.get("ivs_after_validate") != s.get("ivs_in"))]


def tvd(a: collections.Counter, b: collections.Counter) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in set(a) | set(b))


# ------------------------------------------------------------------------------------------------- the inputs
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def n1p72_items() -> List[Tuple[str, str]]:
    """X61's held-out procedural panel (not in the registry), each file checked against its manifest's sha256."""
    from main.anchors.session import load_team_texts

    man = json.loads(repo_path(*N1P72_MANIFEST_REL.split("/")).read_text())
    d = repo_path(*N1P72_REL.split("/"))
    for name, rec in man["files"].items():
        if _sha((d / name).read_bytes()) != rec["sha256"]:
            raise BuildError(f"n1p72: {name} does not match {N1P72_MANIFEST_REL}")
    items = load_team_texts(d)
    if sorted(n for n, _ in items) != sorted(man["files"]):
        raise BuildError(f"n1p72: {d} holds files outside its manifest")
    return items


def panel_items() -> Dict[str, List[Tuple[str, str]]]:
    """Every panel refused at radius 3: the registry's held-out set, n1p72 and the diagnostic rc72b (both registries
    verified)."""
    from main import heldout_panels as HPN

    HPN.verify()
    out = {p: list(HPN.panel_team_texts(p)) for p in HPN.panel_names(held_out_only=True)}
    out["n1p72"] = n1p72_items()
    out["rc72b"] = list(HPN.panel_team_texts("rc72b"))
    return out


class Inputs:
    """The pool, its STRICT occurrences, the profile, the lift and the references (deterministic)."""

    def __init__(self) -> None:
        from main.anchors import team_overlap as TO
        from main.anchors.session import pool_team_texts

        self.pool_texts = sorted(pool_team_texts(), key=sha_text)
        self.pool = [TO.parse_team(t, pool_source_id(t)) for t in self.pool_texts]
        self.occ_all = pool_occurrences(self.pool_texts)
        occ = {s: [x for x in v if not strict_text_bad(x[0])] for s, v in self.occ_all.items()}
        self.species_lost = sorted(s for s, v in occ.items() if not v)
        self.occ = {s: v for s, v in occ.items() if v}
        all_counts = collections.Counter(m.species for t in self.pool for m in t.mons)
        self.profile = {s: c for s, c in all_counts.items() if s in self.occ}
        self.lift = lift_table(sorted(self.profile))
        self.panels = panel_items()
        self.held = {p: TO.Reference(TO.parse_items(items)) for p, items in self.panels.items()}
        self.refs = {"train": {"pool": TO.Reference(self.pool)}, "held": self.held}
        self.pool_six = {frozenset(t.species6) for t in self.pool}


# ------------------------------------------------------------------------------------------------- the build
def accept(inp: Inputs, n: int, stream: str = "corpus") -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    """The first ``n`` candidates of ``stream`` passing every rule, in candidate order; the candidates drawn; and the
    refusal tallies over exactly those candidates (so neither depends on :data:`BATCH`)."""
    from main.anchors import team_overlap as TO
    from utils.bridge.hp_set_check import check_teams
    from utils.team_corpora.build_w_forum import level_defects, showdown_rows

    out: List[Dict[str, Any]] = []
    per: Dict[int, Tuple[str, collections.Counter]] = {}
    why_validator: Dict[int, str] = {}
    acc_keys: set = set()
    within_idx: Dict[Any, List[int]] = collections.defaultdict(list)
    j, drawn = 0, None
    while drawn is None:
        if j >= MAX_CANDIDATES:
            raise BuildError(f"stream {stream!r}: {len(out)} of {n} accepted after {j} candidates")
        cands = []
        for jj in range(j, j + BATCH):
            loc: collections.Counter = collections.Counter()
            got = draw_team(jj, inp.profile, inp.occ, inp.lift, inp.refs, loc, stream)
            if got is None:
                per[jj] = ("slot_exhausted", loc)
                continue
            mons = [m for _b, m, _s in got]
            if frozenset(m.species for m in mons) in inp.pool_six:
                per[jj] = ("pool_species_six", loc)
                continue
            text = "\n\n".join(strip_nickname(b) for b, _m, _s in got).strip()   # the corpus contract: stripped
            if label_path_defects(text):
                per[jj] = ("level", loc)
                continue
            per[jj] = ("pending", loc)
            cands.append({"j": jj, "id": f"x62r_{jj:05d}", "text": text, "sources": [s for _b, _m, s in got],
                          "team": TO.Team(label=f"x62r_{jj:05d}", mons=tuple(mons))})
        j += BATCH
        sd = showdown_rows({c["id"]: c["text"] for c in cands}) if cands else {}
        hp = check_teams([c["text"] for c in cands]) if cands else []
        for c, hv in zip(cands, hp):
            r = sd[c["id"]]
            loc = per[c["j"]][1]
            if r["verdict"] != "ok":
                per[c["j"]] = ("validator", loc)
                why_validator[c["j"]] = f"{r['verdict']}: {str(r.get('why', ''))[:80]}"
                continue
            if level_defects(c["id"], c["text"], r["packed"]):
                per[c["j"]] = ("pack_level", loc)
                continue
            if hp_bad_strict(hv):
                per[c["j"]] = ("hidden_power", loc)
                continue
            key = c["team"].key
            if key in acc_keys:
                per[c["j"]] = ("duplicate", loc)
                continue
            h: collections.Counter = collections.Counter()
            for k in key:
                for i in within_idx[k]:
                    h[i] += 1
            mx = max(h.values()) if h else 0
            if mx > WITHIN_MAX:
                per[c["j"]] = ("within_gt4", loc)
                continue
            per[c["j"]] = ("accepted", loc)
            c.update(sd=r, within_max=mx)
            for k in key:
                within_idx[k].append(len(out))
            acc_keys.add(key)
            out.append(c)
            if len(out) == n:
                drawn = c["j"] + 1
                break
    counted = [per[i] for i in range(drawn)]
    tallies = {
        "candidates_drawn": drawn,
        "candidate_refusals": {r: sum(1 for o, _ in counted if o == r) for r in CANDIDATE_RULES},
        "slot_refusals": {r: sum(loc[r] for _o, loc in counted) for r in SLOT_RULES},
        "validator_failures_by_reason": dict(sorted(collections.Counter(
            v for i, v in why_validator.items() if i < drawn).items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return out, drawn, tallies


def _quantiles(xs: Sequence[float], qs=(0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)) -> Dict[str, float]:
    s = sorted(xs)
    return {f"q{int(round(q * 100)):02d}": round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 4) for q in qs}


def describe(rows: Sequence[Dict[str, Any]], inp: Inputs) -> Dict[str, Any]:
    """Descriptive statistics (no rule reads them), §2.4's list."""
    from main.anchors import team_overlap as TO
    from utils import team_corpora as TC

    teams = [c["team"] for c in rows]
    sp_c = collections.Counter(m.species for t in teams for m in t.mons)
    sp_p = collections.Counter(m.species for t in inp.pool for m in t.mons)
    set_c = collections.Counter(m.key for t in teams for m in t.mons)
    set_p = collections.Counter(m.key for t in inp.pool for m in t.mons)
    tc, tp = sum(set_c.values()), sum(set_p.values())
    drawable = {m.key for v in inp.occ.values() for _b, m, _s in v}
    ratio = {k: (set_c.get(k, 0) / tc) / (set_p[k] / tp) for k in set_p}
    withdrawn = sorted(set(set_p) - drawable)

    def hp_share(c: collections.Counter) -> float:
        return round(sum(v for k, v in c.items() if any(x.startswith(HP) for x in k[1])) / sum(c.values()), 4)

    near: Dict[str, Dict[str, int]] = {}
    for name, ref in {**inp.refs["train"], **inp.held}.items():
        near[name] = {str(k): v for k, v in sorted(collections.Counter(ref.max_shared(t)[0] for t in teams).items())}
    dups: Dict[str, Any] = {}
    for name, tier in NEAR_DUP_CORPORA:
        c = TC.load_corpus(name, tier)
        ref = TO.Reference(TO.parse_items(list(zip(c.ids, c.texts))))
        d = collections.Counter(ref.max_shared(t)[0] for t in teams)
        dups[f"{name}@{tier}"] = {"max_shared_hist": {str(k): v for k, v in sorted(d.items())},
                                  "at_4": d.get(4, 0), "at_ge_5": sum(v for k, v in d.items() if k >= 5)}
    top = sp_p.most_common(10)
    return {
        "n_teams": len(teams),
        "strict_filter": {
            "pool_mons": sum(len(v) for v in inp.occ_all.values()),
            "drawable_mons": sum(len(v) for v in inp.occ.values()),
            "removed_mons": sum(len(v) for v in inp.occ_all.values()) - sum(len(v) for v in inp.occ.values()),
            "pool_distinct_sets": len(set_p), "drawable_distinct_sets": len(drawable),
            "species_lost": inp.species_lost, "drawable_species": len(inp.occ),
        },
        "withdrawn_sets": {"n": len(withdrawn), "sets": [[k[0], list(k[1])] for k in withdrawn],
                           "rule": "a pool set with no STRICT-class occurrence (every occurrence names a non-Dark "
                                   "Hidden Power with no IVs line): never drawn"},
        "distinct_sets_drawn": len(set_c), "species_drawn": len(sp_c),
        "species_tvd_vs_pool": round(tvd(sp_c, sp_p), 4),
        "set_exposure_tvd_vs_pool_occurrences": round(tvd(set_c, set_p), 4),
        "set_exposure_ratio_quantiles": {
            "rule": "per pool set: (corpus share of slots) / (pool share of slots)",
            "all_pool_sets": _quantiles(list(ratio.values())),
            "drawable_sets": _quantiles([v for k, v in ratio.items() if k in drawable])},
        "slots_with_hidden_power": {"pool": hp_share(set_p), "corpus": hp_share(set_c)},
        "top_pool_species_share_pool_to_corpus": {s: [round(sp_p[s] / tp, 4), round(sp_c[s] / tc, 4)] for s, _ in top},
        "within_corpus_max_shared_with_an_earlier_team": {
            str(k): v for k, v in sorted(collections.Counter(c["within_max"] for c in rows).items())},
        "max_shared_hist": near,
        "exact_pool_species_six": sum(frozenset(t.species6) in inp.pool_six for t in teams),
        "near_duplicates_vs_breadth_corpora": dups,
    }


def build() -> Tuple[bytes, Dict[str, Any]]:
    from main import heldout_panels as HPN
    from main.anchors.team_overlap import OVERLAP_VERSION
    from utils.ladder_corpus.build import engine_verdicts
    from utils.team_corpora import engine_pin

    inp = Inputs()
    corpus, drawn, tallies = accept(inp, N_CORPUS)
    verdicts = engine_verdicts([c["sd"] for c in corpus])
    unplayable = engine_pin.unplayable(NAME, verdicts)
    if any(unplayable.values()):
        raise BuildError(f"the corpus carries ids the engine cannot run: {unplayable}")
    check_label_encodable([c["id"] for c in corpus], [c["text"] for c in corpus], f"the {NAME} build")
    rows = [{"i": i, "id": c["id"], "tier": next(t for t, k in TIERS if i < k), "candidate": c["j"],
             "sources": c["sources"], "sha256": _sha(c["text"].encode()), "text": c["text"]}
            for i, c in enumerate(corpus)]
    data = encode_rows(rows)
    texts = [c["text"] for c in corpus]
    reg_path = HPN.registry_path()
    diag_path = repo_path(*HPN.DIAGNOSTIC_REL.split("/"))
    manifest = {
        "schema": SCHEMA, "name": NAME, "generator": GENERATOR,
        "what": "X62's recombination corpus: new gen3ou sixes made only of the pool's STRICT-class mon sets (verbatim, "
                "nickname stripped), species drawn by the pool's profile with Smogon teammate lift, <= 2 sets shared "
                "with any one pool team; TRAINING DATA, never a prior",
        "built_by": "python -m utils.team_corpora.build_x62_recomb",
        "design": "designs/endstate/design_x62_team_recombination.md §2",
        "prototype": "designs/research_state/measurements/x62_design_2026-10-26/revision_2026-10-27/strict_draw_sim.py",
        "seeds": {"build": BUILD_SEED, "rule": "one random.Random per (stream 'corpus', candidate j): stream_seed = "
                                               "sha256('BUILD_SEED|corpus|j')"},
        "params": {"n": N_CORPUS, "max_shared_pool": MAX_SHARED_POOL, "heldout_refuse": HELDOUT_REFUSE,
                   "within_max": WITHIN_MAX, "lift_clip": list(LIFT_CLIP), "slot_tries": SLOT_TRIES,
                   "profile": PROFILE_RULE, "matcher": OVERLAP_VERSION},
        "draw": {"species": "slot 1 by c_pool(s); each later slot among species not yet chosen by c_pool(s) x geomean_t "
                            "lift(t, s), lift = P(s | t) / share(s) (Smogon), clipped to [1/4, 4]",
                 "set": "one STRICT-class pool occurrence of the species, uniform over occurrences, verbatim except "
                        "the nickname (stripped)",
                 "slot_refusals": "> 2 shared with ONE pool team; >= 3 with ONE team of a held-out panel, n1p72 or "
                                  f"rc72b; redrawn up to {SLOT_TRIES} tries, then the candidate fails",
                 "candidate_rules": list(CANDIDATE_RULES)},
        "hidden_power": {"rule": "STRICT (rc72b's hp_bad strict via utils.bridge.hp_set_check): the named type at 70 "
                                 "as written, IVs not rewritten by the validator",
                         "slot_text_filter": "a Hidden Power naming a type other than Dark with no IVs line is never "
                                             "drawn"},
        "priors": {f: _sha(repo_path("data", "pokemon", f).read_bytes()) for f in PRIOR_FILES},
        "pool": {"n_teams": len(inp.pool_texts),
                 "content_sha256": _sha("\x00".join(sorted(inp.pool_texts)).encode()),
                 "profile": dict(sorted(inp.profile.items(), key=lambda kv: (-kv[1], kv[0])))},
        "panels": {"refused_at_radius_3": {p: {"n": len(v), "content_sha256": _sha("\x00".join(
                       sorted(t for _, t in v)).encode())} for p, v in inp.panels.items()},
                   "panel_registry": {"path": str(reg_path.relative_to(repo_root())), "sha256": _sha(reg_path.read_bytes())},
                   "diagnostic_registry": {"path": HPN.DIAGNOSTIC_REL, "sha256": _sha(diag_path.read_bytes())},
                   "n1p72_manifest": {"path": N1P72_MANIFEST_REL,
                                      "sha256": _sha(repo_path(*N1P72_MANIFEST_REL.split("/")).read_bytes())}},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js, pinned "
                     "deps/pokemon-showdown; level 100 by Showdown's pack; hp_set_check.js",
        "engine": {"oracle": "scan_move_probe", "output_unplayable": unplayable,
                   "rule": "every id the corpus carries is asked; an unplayable one THROWS (the pin is empty)"},
        **tallies,
        "describe": describe(corpus, inp),
        "tiers": {t: {"n": k, "sha256": tier_sha256(texts[:k]), "source_tiers": [x for x, kk in TIERS if kk <= k]}
                  for t, k in TIERS},
        "default_tier": DEFAULT_TIER, "data_file": DATA_NAME, "data_sha256": _sha(data),
    }
    return data, manifest


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    data, manifest = build()
    out = CORPORA[NAME]
    if a.check:
        same = ((out / DATA_NAME).exists() and (out / DATA_NAME).read_bytes() == data
                and (out / MANIFEST_NAME).read_text() == _dump(manifest))
        print("reproduces the committed corpus" if same else "DOES NOT reproduce the committed corpus")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA_NAME).write_bytes(data)
    (out / MANIFEST_NAME).write_text(_dump(manifest))
    print(json.dumps({k: v["n"] for k, v in manifest["tiers"].items()}), "drawn", manifest["candidates_drawn"],
          json.dumps(manifest["candidate_refusals"]), json.dumps(manifest["slot_refusals"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
