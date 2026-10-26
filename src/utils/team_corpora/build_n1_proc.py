"""(Re)build ``n1_proc_v1``: X61 / N1's PROCEDURAL corpus, 12,000 whole gen3ou teams drawn from Smogon usage alone,
and its held-out procedural panel ``n1p72``.

    python -m utils.team_corpora.build_n1_proc [--check]

Design: ``designs/research_state/literature/sota_survey_2026-10-26.md`` §5 N1 (Procgen-scale team breadth from step 0:
"the gap closes at as many as 10,000 levels"; Metamon's 11k procedural teams) and ``design_beat_metamon_path.md`` §6.5b.
The lever is ``--team-mix n1_proc_v1@ALL:0.7 --team-mix-sides both`` (``main.x61`` arm ``toy_n1``): with probability 0.7
per episode and side the team is a procedural team, else the side's usual pool draw.

**Every input is SMOGON-derived (owner rule: priors trace to Smogon), so this is NOT a human-data source** in the sense of
``designs/endstate/human_data_architecture.md`` (no replay, no human-built team, no human decision): the generator reads
only the ``agents.gen3_data.priors`` facade (the 12-month chaos usage: ``species_usage``, ``teammates``, ``moves``,
``items``, ``ability``, ``spreads``) and types Hidden Power through :func:`agents.gen3_data.hp_repair.repair_team`
(``gen3_hp_repair_v1``, the one rule, F-DV-4). The pool is read only to refuse an exact copy. Nothing reads this corpus
as a prior (``utils.team_corpora``'s rule). The JS fuzz generator ``ou_random_teams.js`` (P72's provenance) is the same
idea in another language; this one is Python so it reads the facade, and it is NOT byte-compatible with it.

THE DRAW of one candidate (one ``random.Random`` per (stream, candidate index), :func:`stream_seed`):

1. SPECIES. The lead by Smogon usage (the rating-weighted set total W). Each next mon by the MIXTURE of the chosen mons'
   Smogon teammate conditionals, ``sum_c P(x | c)`` over eligible species not yet chosen (Species Clause by
   construction); usage if that mass is 0. Eligible: a usage entry, a move prior with >= 4 engine-playable moves, an
   item, an ability and a spread prior, and the engine plays the species.
2. SET, per mon (D3's rule, ``build_d3_setmut``): four distinct moves sequentially without replacement by
   ``P(move in set)``, the 16 typed Hidden Powers ONE candidate with their summed mass; the item by usage
   (``nothing`` = no item); the ability by usage; nature + EVs as ONE usage spread; level 100 (no ``Level:`` line); IVs
   all 31 unless Hidden Power, which :func:`hp_repair.repair_team` types (side by EVs / nature / moves, type by the
   species' Smogon HP usage on that side, the validator's IVs).

THE RULES, in order (a failing candidate is dropped and the next index drawn; the manifest counts each rule):

* ``level``: a ``Level:`` line other than 100 (:func:`utils.team_corpora.label_path_defects`; never fires, the guard);
* ``pool_exact``: the same six mon sets as a pool team (``gen3_team_overlap_v1``);
* ``panel``: a near neighbour of a HELD-OUT panel AT REVIEW A's RADIUS: >= :data:`PANEL_MIN_SHARED` (3) shared mon sets
  with one team of ``heldout72``, ``heldout72_hpfix``, ``forum72``, ``p72`` or the sealed ``forum216b`` (the registry's
  held-out set), or of ``n1p72`` (the corpus only). The registry's own radius is 4 (``main.heldout_panels``), which the
  ``--team-mix`` startup guard checks; ``human_replay_quality_2026-10-25/REVIEW_A_data.md`` B2 found 4 looser than the
  panels' "unfamiliar" region (<= 2 fits) and registered 3. A procedural team is fully known, so a "revealed fit" is a
  shared mon set (species + sorted moves, Hidden Power typed). The in-pool panels ``home72`` / ``away`` are not
  excluded (their teams are pool teams);
* ``validator``: the pinned ``TeamValidator('gen3ou')`` rejects it, or Showdown cannot import six Pokemon
  (``ladder_corpus_filter.js``);
* ``duplicate``: the same six mon sets as a team already accepted (in candidate order; the panel and the corpus share
  one key set).

The ENGINE is a PREFILTER, as in D3: a species, move, item or ability the Rust engine cannot run (``scan_move_probe``)
is never drawn (the move mass renormalised; the dropped mass is in the manifest), and the build re-asks the engine
about every id the output carries and THROWS on any unplayable one.

THE TWO OUTPUTS, both deterministic (the same priors, pool, registry, engine, validator and seeds give the same bytes;
``--check``):

* ``n1p72``, the HELD-OUT PROCEDURAL PANEL: the first :data:`N_PANEL` accepted candidates of stream ``panel``, written as
  72 pastes + a manifest under :data:`PANEL_REL`. It is drawn FIRST and the corpus excludes its near neighbours, so it
  measures generalisation to new draws of the corpus's OWN generator (Procgen's held-out levels). It is not yet in
  ``heldout_panels.json`` (X61 PREREG FINDING): a Kakuna read plays it by directory; ``main.h2h --panel`` cannot.
* ``n1_proc_v1``: the first :data:`N_CORPUS` accepted candidates of stream ``corpus``, in acceptance order (already a
  seeded random order). Tiers are PREFIXES (``K1`` ⊂ ``K3`` ⊂ ``K10`` ⊂ ``ALL``), so a later dose read can train on
  1k / 3k / 10k / 12k teams of one draw.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path, repo_root
from utils.team_corpora import (CORPORA, DATA_NAME, MANIFEST_NAME, SCHEMA, check_label_encodable, encode_rows,
                                 label_path_defects, tier_sha256)
from utils.team_corpora.build_d3_setmut import (HP, MOVE_SLOTS, SpeciesPrior, _engine_unplayable, draw_moves,
                                                item_name, move_name, species_prior, weighted_pick)

NAME = "n1_proc_v1"
GENERATOR = "gen3_n1_procedural_team_v1"
BUILD_SEED = 6101
N_CORPUS = 12_000
N_PANEL = 72
#: Prefix tiers (name -> count), smallest first; ``ALL`` is the whole corpus.
TIERS = (("K1", 1_000), ("K3", 3_000), ("K10", 10_000), ("ALL", N_CORPUS))
DEFAULT_TIER = "ALL"
#: REVIEW A's radius (B2): a team sharing >= 3 mon sets with one held-out panel team is that team's neighbour.
PANEL_MIN_SHARED = 3
#: Candidates drawn per validator call (a Node process per batch; no effect on the output, see :func:`_accept`).
BATCH = 2_000
#: A stream that has not filled after this many candidates is a broken generator, not a slow one.
MAX_CANDIDATES = 40_000
RULES = ("level", "pool_exact", "panel", "validator", "duplicate")
PANEL_NAME = "n1p72"
PANEL_REL = "designs/research_state/measurements/x61_n1_breadth_2026-10-25/teams_n1p72"
PANEL_MANIFEST_REL = "designs/research_state/measurements/x61_n1_breadth_2026-10-25/n1p72_manifest.json"
PANEL_SCHEMA = "x61_n1p72_manifest_v1"
PRIOR_FILES = ("gen3_smogon_stats.json", "gen3_teammate_priors.json", "gen3_move_priors.json", "gen3_item_priors.json",
               "gen3_ability_priors.json", "gen3_spread_priors.json", "gen3_hidden_power_priors.json")
_STAT_LABEL = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


class BuildError(RuntimeError):
    """The build cannot stand behind its output (an unplayable id, a level defect, a stream that does not fill)."""


# ------------------------------------------------------------------------------------------------ the pure core
def stream_seed(*parts: object) -> int:
    """A 64-bit seed from ``BUILD_SEED`` and the parts (sha256; independent of Python's hash randomisation)."""
    h = hashlib.sha256("|".join(str(p) for p in (BUILD_SEED,) + parts).encode()).hexdigest()
    return int(h[:16], 16)


@dataclass(frozen=True)
class Universe:
    """The eligible species with their draw tables, in sorted id order (the draw order is part of the seed contract)."""

    species: Tuple[str, ...]
    usage: Tuple[float, ...]
    names: Dict[str, str]
    sets: Dict[str, SpeciesPrior]
    abilities: Dict[str, Tuple[Tuple[str, float], ...]]
    teammates: Dict[str, Dict[str, float]]


def build_universe(bad: Dict[str, frozenset]) -> Universe:
    """Every species the generator may draw, from the Smogon facade, minus what the engine cannot run (``bad[kind]``)."""
    from agents.gen3_data import abilities as ab_dex
    from agents.gen3_data import priors, species

    keep, names, sets, abil, mates = [], {}, {}, {}, {}
    usage = priors.species_usage()
    for sid in sorted(usage):
        sp = species.get(sid)
        if sp is None or sid in bad["species"]:
            continue
        pr = species_prior(sid, bad["move"], bad["item"])
        ab = []
        for a, p in sorted(priors.ability(sid).items()):
            if p > 0.0 and a not in bad["ability"] and ab_dex.get(a) is not None:
                ab.append((a, float(p)))
        if not pr.eligible or not ab:
            continue
        keep.append(sid)
        names[sid] = sp.name
        sets[sid] = pr
        abil[sid] = tuple(ab)
        mates[sid] = {k: float(v) for k, v in priors.teammates(sid).items() if v > 0.0}
    return Universe(tuple(keep), tuple(float(usage[s]) for s in keep), names, sets, abil, mates)


def draw_species(rng: random.Random, u: Universe) -> List[str]:
    """Six distinct species: the lead by usage, each next by the summed teammate conditionals of the chosen ones."""
    chosen = [weighted_pick(rng, list(zip(u.species, u.usage)))]
    while len(chosen) < 6:
        taken = set(chosen)
        cand = []
        for s in u.species:
            if s in taken:
                continue
            w = sum(u.teammates[c].get(s, 0.0) for c in chosen)
            if w > 0.0:
                cand.append((s, w))
        if not cand:
            cand = [(s, w) for s, w in zip(u.species, u.usage) if s not in taken]
        chosen.append(weighted_pick(rng, cand))
    return chosen


def ability_name(aid: str) -> str:
    from agents.gen3_data import abilities

    d = abilities.get(aid)
    if d is None:
        raise BuildError(f"the ability prior names {aid!r}, which the ability dex does not know")
    return d.name


def draw_set(rng: random.Random, sid: str, u: Universe, hp_seed: int) -> Tuple[str, Dict[str, Any]]:
    """One mon's paste block and its record (module docstring step 2)."""
    from agents.gen3_data.hp_repair import repair_team

    pr = u.sets[sid]
    mv = draw_moves(rng, pr.moves)
    item = weighted_pick(rng, list(pr.items))
    ability = weighted_pick(rng, list(u.abilities[sid]))
    si = weighted_pick(rng, [(str(i), s[2]) for i, s in enumerate(pr.spreads)])
    nature, evs, _ = pr.spreads[int(si)]
    out = [u.names[sid] + ("" if item == "nothing" else f" @ {item_name(item)}"), f"Ability: {ability_name(ability)}"]
    ev_text = " / ".join(f"{e} {lab}" for e, lab in zip(evs, _STAT_LABEL) if e)
    if ev_text:
        out.append(f"EVs: {ev_text}")
    out.append(f"{nature} Nature")
    out += [f"- {move_name(m)}" for m in mv]
    block = "\n".join(out)
    if HP in mv:
        block, recs = repair_team(block, hp_seed)
        if len(recs) != 1:
            raise BuildError(f"the HP repair typed {len(recs)} sets of one drawn set: {block!r}")
        block = block.strip("\n")
    return block, {"species": sid, "moves": sorted(mv), "item": item, "ability": ability, "nature": nature,
                   "evs": list(evs), "hidden_power": HP in mv}


def draw_team(stream: str, j: int, u: Universe) -> Tuple[str, List[Dict[str, Any]]]:
    """Candidate ``j`` of ``stream``: ``(paste, per-mon records)``. Deterministic in (``BUILD_SEED``, stream, j)."""
    rng = random.Random(stream_seed(stream, j))
    blocks, recs = [], []
    for slot, sid in enumerate(draw_species(rng, u)):
        b, r = draw_set(rng, sid, u, stream_seed(stream, j, "hp", slot))
        blocks.append(b)
        recs.append(r)
    return "\n\n".join(blocks), recs


# ------------------------------------------------------------------------------------------------- the build
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _Refs:
    """The exclusion references: the pool (exact), the held-out panels and (for the corpus) ``n1p72`` at radius 3."""

    def __init__(self, pool_texts: Sequence[str]):
        from main import heldout_panels as HPN
        from main.anchors.team_overlap import Reference, parse_items

        self.pool = Reference(parse_items([(f"pool_{i}", t) for i, t in enumerate(pool_texts)]))
        self.panels = {p: Reference(parse_items(HPN.panel_team_texts(p))) for p in HPN.panel_names(held_out_only=True)}

    def add_panel(self, name: str, items: Sequence[Tuple[str, str]]) -> None:
        from main.anchors.team_overlap import Reference, parse_items

        self.panels[name] = Reference(parse_items(items))

    def nearest(self, team: Any) -> Dict[str, int]:
        return {p: int(r.max_shared(team)[0]) for p, r in self.panels.items()}


def _accept(stream: str, n: int, u: Universe, refs: _Refs, keys: set, fails: Dict[str, int],
            why_validator: collections.Counter) -> Tuple[List[Dict[str, Any]], int]:
    """The first ``n`` candidates of ``stream`` that pass every rule, in candidate order, and the candidates drawn.
    Batching only groups the validator's Node calls: each candidate's verdict is its own, and duplicates resolve in
    candidate order, so the output does not depend on :data:`BATCH`."""
    from main.anchors.team_overlap import parse_team
    from utils.team_corpora.build_w_forum import level_defects, showdown_rows

    out: List[Dict[str, Any]] = []
    j = 0
    while len(out) < n:
        if j >= MAX_CANDIDATES:
            raise BuildError(f"stream {stream!r}: {len(out)} of {n} accepted after {j} candidates")
        cands = []
        for jj in range(j, j + BATCH):
            text, recs = draw_team(stream, jj, u)
            team = parse_team(text)
            if label_path_defects(text):
                fails["level"] += 1
                continue
            if refs.pool.keys[team.key]:
                fails["pool_exact"] += 1
                continue
            near = refs.nearest(team)
            if max(near.values()) >= PANEL_MIN_SHARED:
                fails["panel"] += 1
                continue
            cands.append({"j": jj, "id": f"{stream}_{jj:05d}", "text": text, "draws": recs, "key": team.key,
                          "near": near})
        j += BATCH
        sd = showdown_rows({c["id"]: c["text"] for c in cands}) if cands else {}
        for c in cands:
            r = sd[c["id"]]
            if r["verdict"] != "ok":
                fails["validator"] += 1
                why_validator[f"{r['verdict']}: {str(r.get('why', ''))[:80]}"] += 1
                continue
            if c["key"] in keys:
                fails["duplicate"] += 1
                continue
            if level_defects(c["id"], c["text"], r["packed"]):
                raise BuildError(f"{c['id']}: Showdown packs a non-100 level for a procedural team")
            c["sd"] = r
            keys.add(c["key"])
            out.append(c)
            if len(out) == n:
                return out, c["j"] + 1
    return out, j


def describe(rows: Sequence[Dict[str, Any]], u: Universe, refs: _Refs) -> Dict[str, Any]:
    """Descriptive statistics (no rule reads them): species coverage against usage, set breadth, panel distance."""
    from main.anchors.team_overlap import parse_team

    n = len(rows)
    sp_count = collections.Counter(d["species"] for r in rows for d in r["draws"])
    tot_w = sum(u.usage)
    # Smogon "usage %" of a species = its share of team slots (W / (sum W / 6)), the quantity a team frequency estimates
    usage_pct = {s: 6.0 * w / tot_w for s, w in zip(u.species, u.usage)}
    freq = {s: sp_count.get(s, 0) / n for s in u.species}
    xs = [usage_pct[s] for s in u.species]
    ys = [freq[s] for s in u.species]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    pear = cov / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    top = sorted(u.species, key=lambda s: -usage_pct[s])
    mon_sets = collections.Counter()
    for r in rows:
        for m in parse_team(r["text"]).mons:
            mon_sets[m.key] += 1
    pool_shared = collections.Counter(int(refs.pool.max_shared(parse_team(r["text"]))[0]) for r in rows)
    near_hist = {p: dict(sorted(collections.Counter(r["near"][p] for r in rows).items())) for p in rows[0]["near"]}
    near_max = dict(sorted(collections.Counter(max(r["near"].values()) for r in rows).items()))
    species6 = collections.Counter(frozenset(d["species"] for d in r["draws"]) for r in rows)
    return {
        "n_teams": n,
        "species": {
            "eligible": len(u.species), "drawn": sum(1 for s in u.species if sp_count.get(s)),
            "never_drawn": sorted(s for s in u.species if not sp_count.get(s)),
            "pearson_team_freq_vs_smogon_usage_pct": round(pear, 4),
            "top20_usage_pct_vs_team_freq": {s: [round(usage_pct[s], 4), round(freq[s], 4)] for s in top[:20]},
            "teams_per_species_min_median_max": [min(sp_count.values()), sorted(sp_count.values())[len(sp_count) // 2],
                                                 max(sp_count.values())],
        },
        "distinct_species6": len(species6),
        "distinct_mon_sets": len(mon_sets),
        "mon_sets_seen_once": sum(1 for v in mon_sets.values() if v == 1),
        "max_shared_mon_sets_with_pool_hist": dict(sorted(pool_shared.items())),
        "max_shared_mon_sets_with_heldout_panel_hist": near_hist,
        "max_shared_mon_sets_with_any_heldout_panel_hist": near_max,
        "sets_with_hidden_power": sum(1 for r in rows for d in r["draws"] if d["hidden_power"]),
        "itemless_sets": sum(1 for r in rows for d in r["draws"] if d["item"] == "nothing"),
    }


def build() -> Tuple[bytes, Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    """``(corpus data, corpus manifest, {panel file: text}, panel manifest)``."""
    from agents.gen3_data import priors
    from agents.gen3_data.hp_repair import REPAIR_VERSION
    from main import heldout_panels as HPN
    from main.anchors.team_overlap import OVERLAP_VERSION
    from utils.ladder_corpus.build import KINDS, engine_verdicts
    from utils.team_loader import TeamLoader

    HPN.verify()
    pool_texts = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    usage = priors.species_usage()
    all_moves = sorted({m for s in usage for m in priors.moves(s) if m})
    all_items = sorted({i for s in usage for i in priors.items(s) if i != "nothing"})
    all_abil = sorted({a for s in usage for a in priors.ability(s)})
    bad = {"species": frozenset(_engine_unplayable("species", sorted(usage))),
           "move": frozenset(_engine_unplayable("move", all_moves)),
           "item": frozenset(_engine_unplayable("item", all_items)),
           "ability": frozenset(_engine_unplayable("ability", all_abil))}
    u = build_universe(bad)
    dropped_mass = {s: round(sum(p for m, p in priors.moves(s).items() if m in bad["move"]) / float(MOVE_SLOTS), 6)
                    for s in u.species}

    refs = _Refs(pool_texts)
    keys: set = set()
    fails = {"panel": {r: 0 for r in RULES}, "corpus": {r: 0 for r in RULES}}
    why = {"panel": collections.Counter(), "corpus": collections.Counter()}
    panel, drawn_panel = _accept("panel", N_PANEL, u, refs, keys, fails["panel"], why["panel"])
    panel_items = [(f"n1p_{i:03d}.txt", c["text"]) for i, c in enumerate(panel)]
    refs.add_panel(PANEL_NAME, panel_items)
    corpus, drawn_corpus = _accept("corpus", N_CORPUS, u, refs, keys, fails["corpus"], why["corpus"])

    verdicts = engine_verdicts([c["sd"] for c in panel + corpus])
    unplayable = {kind: sorted(i for i, x in verdicts[kind].items() if x != "ran") for kind, _ in KINDS}
    if any(unplayable.values()):
        raise BuildError(f"the output carries ids the engine cannot run: {unplayable}")
    check_label_encodable([c["id"] for c in corpus], [c["text"] for c in corpus], f"the {NAME} build")
    check_label_encodable([c["id"] for c in panel], [c["text"] for c in panel], f"the {PANEL_NAME} build")

    rows = [{"i": i, "id": c["id"], "tier": next(t for t, k in TIERS if i < k), "candidate": c["j"],
             "sha256": _sha(c["text"].encode()), "text": c["text"]} for i, c in enumerate(corpus)]
    data = encode_rows(rows)
    texts = [c["text"] for c in corpus]
    reg_path = HPN.registry_path()
    prior_sha = {f: _sha(repo_path("data", "pokemon", f).read_bytes()) for f in PRIOR_FILES}
    common = {
        "generator": GENERATOR,
        "seeds": {"build": BUILD_SEED, "rule": "one random.Random per (stream, candidate j): stream_seed = sha256("
                                               "'BUILD_SEED|stream|j'); Hidden Power's own stream adds ('hp', slot)"},
        "draw": {"species": "lead by Smogon usage W; each next by sum over chosen c of P(x | c) (Smogon teammates), "
                            "usage if that mass is 0; Species Clause by construction",
                 "moves": "4 distinct, sequentially without replacement, weight P(move in set); the 16 typed HP keys "
                          "are one 'Hidden Power' candidate with their summed mass",
                 "item": "Smogon item usage ('nothing' = no item)", "ability": "Smogon ability usage",
                 "nature_evs": "one draw of a Smogon usage spread (nature and EVs joint)", "level": 100,
                 "ivs": "all 31 unless Hidden Power", "hidden_power": REPAIR_VERSION},
        "priors": prior_sha,
        "engine": {"oracle": "scan_move_probe", "prefilter_unplayable": {k: sorted(v) for k, v in bad.items()},
                   "eligible_species": len(u.species),
                   "dropped_move_slot_mass": {s: m for s, m in dropped_mass.items() if m > 0},
                   "output_unplayable": unplayable},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js, pinned "
                     "deps/pokemon-showdown",
        "panel_registry": {"path": str(reg_path.relative_to(repo_root())), "sha256": _sha(reg_path.read_bytes()),
                           "excluded": HPN.panel_names(held_out_only=True),
                           "not_excluded": sorted(set(HPN.panel_names()) - set(HPN.panel_names(held_out_only=True))),
                           "near_neighbour_min_shared": PANEL_MIN_SHARED, "registry_radius": HPN.NEAR_NEIGHBOUR_MIN_SHARED,
                           "matcher": OVERLAP_VERSION,
                           "rule": "REVIEW A B2's radius (>= 3 shared mon sets with one team, one tighter than the "
                                   "registry's 4) against every held-out panel, and (the corpus) against n1p72; the "
                                   "pool by exact copy; home72 / away (in-pool) not excluded"},
        "pool": {"n_files": len(pool_texts),
                 "content_sha256": _sha("\x00".join(sorted(t.strip() for t in pool_texts)).encode())},
    }
    panel_files = {name: text + "\n" for name, text in panel_items}
    panel_manifest = dict(common, schema=PANEL_SCHEMA, name=PANEL_NAME, stream="panel", n=len(panel),
                          candidates_drawn=drawn_panel, rule_failures=fails["panel"],
                          validator_failures_by_reason=dict(sorted(why["panel"].items(), key=lambda kv: (-kv[1], kv[0]))),
                          files={name: {"candidate": c["j"], "sha256": _sha(panel_files[name].encode())}
                                 for (name, _), c in zip(panel_items, panel)},
                          describe=describe(panel, u, refs),
                          role="X61's held-out PROCEDURAL panel: drawn before the corpus from the same generator, "
                               "and the corpus excludes its >= 3-fit neighbours; held out from every training source; "
                               "NOT in heldout_panels.json (X61 PREREG FINDING)")
    manifest = dict(common, schema=SCHEMA, name=NAME,
                    what="X61 / N1's procedural corpus: whole gen3ou teams drawn from Smogon usage only (species by "
                         "usage + teammate co-usage; sets from the usage spreads); TRAINING DATA, never a prior; not a "
                         "human-data source (Smogon aggregates, the priors rule)",
                    built_by="python -m utils.team_corpora.build_n1_proc",
                    design="designs/research_state/literature/sota_survey_2026-10-26.md §5 N1; "
                           "design_beat_metamon_path.md §6.5b",
                    stream="corpus", candidates_drawn=drawn_corpus, rule_failures=fails["corpus"],
                    validator_failures_by_reason=dict(sorted(why["corpus"].items(), key=lambda kv: (-kv[1], kv[0]))),
                    held_out_procedural_panel={"name": PANEL_NAME, "dir": PANEL_REL, "manifest": PANEL_MANIFEST_REL},
                    describe=describe(corpus, u, refs),
                    tiers={t: {"n": k, "sha256": tier_sha256(texts[:k]), "source_tiers": [x for x, kk in TIERS if kk <= k]}
                           for t, k in TIERS},
                    default_tier=DEFAULT_TIER, data_file=DATA_NAME, data_sha256=_sha(data))
    return data, manifest, panel_files, panel_manifest


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    data, manifest, panel_files, panel_manifest = build()
    out = CORPORA[NAME]
    pdir = repo_path(*PANEL_REL.split("/"))
    pman = repo_path(*PANEL_MANIFEST_REL.split("/"))
    if a.check:
        same = ((out / DATA_NAME).exists() and (out / DATA_NAME).read_bytes() == data
                and (out / MANIFEST_NAME).read_text() == _dump(manifest)
                and pman.exists() and pman.read_text() == _dump(panel_manifest)
                and pdir.is_dir() and sorted(p.name for p in pdir.iterdir()) == sorted(panel_files)
                and all((pdir / k).read_text() == v for k, v in panel_files.items()))
        print("reproduces the committed corpus and panel" if same else "DOES NOT reproduce the committed corpus / panel")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA_NAME).write_bytes(data)
    (out / MANIFEST_NAME).write_text(_dump(manifest))
    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.iterdir():
        if old.name not in panel_files:
            old.unlink()
    for k, v in panel_files.items():
        (pdir / k).write_text(v)
    pman.write_text(_dump(panel_manifest))
    print(json.dumps({k: v["n"] for k, v in manifest["tiers"].items()}), "corpus", json.dumps(manifest["rule_failures"]),
          "drawn", manifest["candidates_drawn"], "| panel", json.dumps(panel_manifest["rule_failures"]),
          "drawn", panel_manifest["candidates_drawn"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
