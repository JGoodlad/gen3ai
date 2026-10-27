"""(Re)build ``d3_setmut_v1``: X41 / D3's SET-MUTATION corpus, pool teams with 1-2 mons' sets resampled from Smogon usage.

    python -m utils.team_corpora.build_d3_setmut [--check]

Design: ``designs/endstate/design_diversity_teams_opponents.md`` §4.5 (L5 (a), SET RESAMPLING: "keep the six species and
resample k mons' moves, item, EVs and nature from Smogon's per-species usage; Hidden Power follows the L1 repair rule";
"an offline pre-generated augmented corpus is simpler and pin-safe"), and ``design_beat_metamon_path.md`` §6.4. The
lever is ``--team-mix d3_setmut_v1:0.3 --team-mix-sides both`` (``main.x35.kit`` arm ``d3``): with probability 0.3 per
episode and side, the team is a MUTATED pool team instead of a pool team.

**Every input is SMOGON-derived (owner rule: priors trace to Smogon).** The new moves, item, nature and EVs are drawn
from the ``agents.gen3_data.priors`` facade (``gen3_move_priors.json``, ``gen3_item_priors.json``,
``gen3_spread_priors.json``: the 12-month chaos usage), and Hidden Power's type from ``gen3_hidden_power_priors.json``
through :func:`agents.gen3_data.hp_repair.repair_team` (``gen3_hp_repair_v1``, THE one function, F-DV-4). The pool
supplies only the SPECIES cores, the abilities and the untouched mons, exactly as the design says. Nothing reads this
corpus as a prior (``utils.team_corpora``'s rule).

THE MUTATION of one source team (a team of our 719-team pool, ``TeamLoader``), variant ``v``, attempt ``t``; one
``random.Random`` per (source, v, t), seeded by :func:`stream_seed` (``BUILD_SEED`` + the source's content id):

1. ``k`` uniform on :data:`K_CHOICES` (1 or 2); ``k`` distinct slots uniform among the team's ELIGIBLE mons (a species
   with a Smogon move prior of >= 4 playable moves, an item prior and a spread prior). A team with fewer than ``k``
   eligible mons mutates all it has (``k`` is recorded as the count actually mutated).
2. Per slot: FOUR distinct moves drawn sequentially without replacement, each with probability proportional to its
   ``P(move in set)`` (the procedural generator's rule, ``ou_random_teams.js`` ``genOuSet``); the 16 typed Hidden Power
   keys are ONE candidate ``Hidden Power`` carrying their summed mass (a set holds one HP). The ITEM is drawn by its
   usage (``nothing`` = no item); the NATURE and EVs are ONE draw of a usage spread (they are joint in the stats). The
   ABILITY, the nickname / gender / shiny / happiness lines are the source's; the IVs are the default all-31.
3. A drawn ``Hidden Power`` is written untyped and typed by :func:`hp_repair.repair_team` on that ONE set: the side by
   its EVs / nature / other moves, the type from Smogon's per-species HP usage on that side, the validator's IVs.
4. The mutated set must be a NEW mon set (``gen3_team_overlap_v1``: species + sorted moves, HP typed), so a variant
   shares exactly ``6 - k`` mon sets with its source; a draw that re-creates the source's moves fails the attempt.

THE RULES a candidate must pass, in order (a failure sends the (source, v) pair to its next attempt, up to
:data:`MAX_ATTEMPTS`; the manifest counts each rule's failures):

* ``unchanged``: a mutated slot drew its source's own move set (step 4);
* ``level``: a ``Level:`` line other than 100 (:func:`utils.team_corpora.label_path_defects`; the source pool is
  level-100 by ``TeamLoader``'s guard and a mutation writes no ``Level:`` line, so this never fires: it is the guard);
* ``pool_exact``: the same six mon sets as a pool team;
* ``panel``: an exact copy or near neighbour (>= 4 shared mon sets) of a HELD-OUT panel
  (``main.heldout_panels.contamination``, the ``--team-mix`` startup guard's own question). The IN-POOL panels
  ``home72`` / ``away`` are NOT excluded: D3 perturbs the pool, and the sample teams' cores are the familiar cores the
  lever targets (the CORE stratum is defined on them); DECLARED, a deliberate difference from ``w_forum_v1``;
* ``validator``: the pinned ``TeamValidator('gen3ou')`` rejects it, or Showdown cannot import six Pokemon
  (``ladder_corpus_filter.js``, the corpora's own filter);
* ``duplicate``: the same six mon sets as a variant already accepted (accepted in (round, source, v) order).

The ENGINE is not a rule but a PREFILTER: a move or item the Rust engine could not run at this corpus's build
(``scan_move_probe``'s answer then, pinned in ``engine_pins.json``, :mod:`utils.team_corpora.engine_pin`) is never drawn
(the drawn mass is renormalised, as the procedural generator does; the dropped mass is in the manifest), and the build
re-asks the engine about every id the corpus carries and THROWS on any unplayable one.

A (source, v) pair still failing after :data:`MAX_ATTEMPTS` is ``exhausted``. The kept variants are ordered by
(source id, v) and shuffled once with ``random.Random(ORDER_SEED)``; one tier, ``ALL``. The build is deterministic: the
same pool, priors, registry, engine pin, validator and seeds give the same bytes (``--check``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path, repo_root
from utils.showdown_id import to_id_str
from utils.team_corpora import (CORPORA, DATA_NAME, MANIFEST_NAME, SCHEMA, check_label_encodable, encode_rows,
                                 label_path_defects, tier_sha256)

NAME = "d3_setmut_v1"
BUILD_SEED = 4101
ORDER_SEED = 4102
#: Variants generated per source pool team (DECLARED: §4.5 names no corpus size; 4 puts the corpus at ≈ 2.9k teams,
#: the size of D1's ``w_forum_v1`` (1,981), so its startup validation and per-team draw count are D1's measured regime).
VARIANTS_PER_TEAM = 4
#: How many mons a variant mutates (§4.5: "k ∈ {1, 2}"), uniform.
K_CHOICES = (1, 2)
MAX_ATTEMPTS = 12
MOVE_SLOTS = 4
TIER = "ALL"
RULES = ("unchanged", "level", "pool_exact", "panel", "validator", "duplicate")
SOURCE_RULES = ("source_shape", "source_no_eligible_mon")
#: The committed Smogon-derived files the draws read (their sha256s are stamped on the manifest).
PRIOR_FILES = ("gen3_move_priors.json", "gen3_item_priors.json", "gen3_spread_priors.json",
               "gen3_hidden_power_priors.json", "gen3_smogon_stats.json")
#: Source lines a mutation KEEPS (everything else of the mutated set is redrawn or dropped).
_KEEP = re.compile(r"^\s*(Ability|Shiny|Happiness|Level)\s*:", re.IGNORECASE)
_NICK = re.compile(r"^(?P<nick>.+?)\s*\((?P<species>[^()]+)\)\s*(?:\((?:M|F)\))?\s*$")
_STAT_LABEL = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
HP = "hiddenpower"


class BuildError(RuntimeError):
    """The build cannot stand behind its output (an unplayable id, a level defect, a source it cannot read)."""


# ------------------------------------------------------------------------------------------------ the pure core
def stream_seed(*parts: object) -> int:
    """A 64-bit seed from ``BUILD_SEED`` and the parts (sha256; independent of Python's hash randomisation)."""
    h = hashlib.sha256("|".join(str(p) for p in (BUILD_SEED,) + parts).encode()).hexdigest()
    return int(h[:16], 16)


def split_blocks(text: str) -> List[str]:
    """A Showdown paste's set blocks, in order (blank-line separated)."""
    return [b.strip("\n") for b in re.split(r"\n[ \t]*\n+", text.strip()) if b.strip()]


def block_species(block: str) -> str:
    """The species id of a set block (``Nick (Species) (M) @ Item`` or ``Species @ Item``)."""
    head = block.split("\n", 1)[0].split("@", 1)[0].strip()
    m = _NICK.match(head)
    name = m.group("species") if m and m.group("species").strip() not in ("M", "F") else \
        re.sub(r"\s*\((?:M|F)\)\s*$", "", head)
    return to_id_str(name)


def weighted_pick(rng: random.Random, entries: Sequence[Tuple[str, float]]) -> str:
    """One key, with probability proportional to its weight (``entries`` in a fixed order; weights > 0)."""
    total = sum(w for _, w in entries)
    r = rng.random() * total
    for k, w in entries:
        r -= w
        if r < 0:
            return k
    return entries[-1][0]


@dataclass(frozen=True)
class SpeciesPrior:
    """The Smogon draw tables of one species, the engine-unplayable ids already removed (each sorted by key)."""

    moves: Tuple[Tuple[str, float], ...]     # HP collapsed into one ``hiddenpower`` candidate
    items: Tuple[Tuple[str, float], ...]     # ``nothing`` = no item
    spreads: Tuple[Tuple[str, Tuple[int, ...], float], ...]

    @property
    def eligible(self) -> bool:
        return len(self.moves) >= MOVE_SLOTS and bool(self.items) and bool(self.spreads)


def species_prior(species: str, bad_moves: frozenset = frozenset(), bad_items: frozenset = frozenset()) -> SpeciesPrior:
    """The draw tables of ``species`` from the Smogon facade (``agents.gen3_data.priors``)."""
    from agents.gen3_data import priors

    mv: Dict[str, float] = {}
    for m, p in priors.moves(species).items():
        if not m or m in bad_moves or p <= 0.0:
            continue
        key = HP if m.startswith(HP) else m
        mv[key] = mv.get(key, 0.0) + float(p)
    items = tuple(sorted((i, float(p)) for i, p in priors.items(species).items() if p > 0.0 and i not in bad_items))
    spreads = tuple((str(n), tuple(int(x) for x in evs), float(w)) for n, evs, w in priors.spreads(species) if w > 0.0)
    return SpeciesPrior(tuple(sorted(mv.items())), items, spreads)


def draw_moves(rng: random.Random, moves: Sequence[Tuple[str, float]]) -> List[str]:
    """FOUR distinct move ids, sequentially without replacement, each by its ``P(move in set)``."""
    pool = list(moves)
    out: List[str] = []
    while len(out) < MOVE_SLOTS:
        k = weighted_pick(rng, pool)
        out.append(k)
        pool = [e for e in pool if e[0] != k]
    return out


def move_name(mid: str) -> str:
    from agents.gen3_data import moves

    return "Hidden Power" if mid == HP else str(moves.raw()[mid]["name"])


def item_name(iid: str) -> str:
    from agents.gen3_data import items

    d = items.get(iid)
    if d is None:
        raise BuildError(f"the item prior names {iid!r}, which the item dex does not know")
    return d.name


def mutate_block(block: str, prior: SpeciesPrior, rng: random.Random, hp_seed: int) -> Tuple[str, Dict[str, Any]]:
    """One set, re-drawn (module docstring step 2-3): ``(new block, record)``."""
    from agents.gen3_data.hp_repair import repair_team

    lines = block.split("\n")
    head = lines[0].split(" @ ", 1)[0].split("@", 1)[0].rstrip()
    mv = draw_moves(rng, prior.moves)
    item = weighted_pick(rng, prior.items)
    si = weighted_pick(rng, [(str(i), s[2]) for i, s in enumerate(prior.spreads)])
    nature, evs, _ = prior.spreads[int(si)]
    out = [head + ("" if item == "nothing" else f" @ {item_name(item)}")]
    out += [ln.strip() for ln in lines[1:] if _KEEP.match(ln)]
    ev_text = " / ".join(f"{e} {lab}" for e, lab in zip(evs, _STAT_LABEL) if e)
    if ev_text:
        out.append(f"EVs: {ev_text}")
    out.append(f"{nature} Nature")
    out += [f"- {move_name(m)}" for m in mv]
    new = "\n".join(out)
    if HP in mv:
        new, recs = repair_team(new, hp_seed)
        if len(recs) != 1:
            raise BuildError(f"the HP repair typed {len(recs)} sets of one mutated set: {new!r}")
        new = new.strip("\n")
    return new, {"moves": sorted(mv), "item": item, "nature": nature, "evs": list(evs), "hidden_power": HP in mv}


def mutate_team(blocks: Sequence[str], priors_by_slot: Sequence[Optional[SpeciesPrior]], source_id: str, v: int,
                t: int) -> Optional[Tuple[str, List[int], List[Dict[str, Any]]]]:
    """Variant ``v``, attempt ``t`` of one source team: ``(text, mutated slots, records)``, or None when no slot is
    eligible. Deterministic in (``BUILD_SEED``, ``source_id``, ``v``, ``t``)."""
    eligible = [i for i, p in enumerate(priors_by_slot) if p is not None and p.eligible]
    if not eligible:
        return None
    rng = random.Random(stream_seed(source_id, v, t))
    k = min(rng.choice(K_CHOICES), len(eligible))
    slots = sorted(rng.sample(eligible, k))
    out = list(blocks)
    recs = []
    for s in slots:
        out[s], rec = mutate_block(blocks[s], priors_by_slot[s], rng, stream_seed(source_id, v, t, "hp", s))  # type: ignore[arg-type]
        recs.append(rec)
    return "\n\n".join(out), slots, recs


def source_id(text: str) -> str:
    """A pool team's content id (the pool's files are not part of the corpus identity; their content is)."""
    return "pool_" + hashlib.sha256(text.strip().encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------- the build
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _engine_unplayable(name: str, kind: str, ids: Sequence[str]) -> List[str]:
    """The ids of ``kind`` the engine could not run AT corpus ``name``'s BUILD (its pin, :mod:`utils.team_corpora.
    engine_pin`); the live oracle (``scan_move_probe``, the ladder corpus's call :func:`utils.ladder_corpus.build.
    engine_verdicts` for one kind) is still asked, and an engine that LOST coverage throws."""
    import subprocess

    from utils.ladder_corpus.build import scan_probe_bin
    from utils.team_corpora import engine_pin

    want = sorted(set(ids))
    if not want:
        return []
    p = subprocess.run([scan_probe_bin()], input="\n".join(want) + "\n", capture_output=True, text=True,
                       env=dict(os.environ, PROBE_KIND=kind), check=True)
    got = {}
    for line in p.stdout.splitlines():
        d = json.loads(line)
        got[d.get("id", d.get("move"))] = d["verdict"]
    if set(got) != set(want):
        raise BuildError(f"scan_move_probe answered {len(got)} of {len(want)} {kind} ids")
    return engine_pin.unplayable(name, {kind: got})[kind]


def build() -> Tuple[bytes, Dict[str, Any]]:
    from agents.gen3_data import priors
    from agents.gen3_data.hp_repair import REPAIR_VERSION
    from main import heldout_panels as HPN
    from main.anchors.team_overlap import OVERLAP_VERSION, Reference, parse_items, parse_team
    from utils.ladder_corpus.build import KINDS, engine_verdicts
    from utils.team_corpora.build_w_forum import level_defects, showdown_rows
    from utils.team_loader import TeamLoader

    HPN.verify()
    pool_texts = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    pool_ref = Reference(parse_items([(source_id(t), t) for t in pool_texts]))
    src_drops: Dict[str, List[str]] = {r: [] for r in SOURCE_RULES}
    seen_src, sources = set(), []
    for t in sorted(pool_texts, key=source_id):
        sid = source_id(t)
        if sid in seen_src:
            continue
        seen_src.add(sid)
        blocks = split_blocks(t)
        if len(blocks) != 6 or len(parse_team(t).mons) != 6:
            src_drops["source_shape"].append(sid)
            continue
        sources.append((sid, blocks))

    # the engine prefilter over every species the sources hold
    species = sorted({block_species(b) for _, bl in sources for b in bl})
    all_moves = sorted({m for s in species for m in priors.moves(s) if m})
    all_items = sorted({i for s in species for i in priors.items(s) if i != "nothing"})
    bad_moves = frozenset(_engine_unplayable(NAME, "move", all_moves))
    bad_items = frozenset(_engine_unplayable(NAME, "item", all_items))
    table = {s: species_prior(s, bad_moves, bad_items) for s in species}
    dropped_mass = {s: round(sum(p for m, p in priors.moves(s).items() if m in bad_moves) / 4.0, 6) for s in species}

    pending: List[Tuple[str, int]] = []
    by_src = dict(sources)
    pri = {sid: [table[block_species(b)] for b in bl] for sid, bl in sources}
    for sid, _ in sources:
        if not any(p.eligible for p in pri[sid]):
            src_drops["source_no_eligible_mon"].append(sid)
            continue
        pending += [(sid, v) for v in range(VARIANTS_PER_TEAM)]
    fails: Dict[str, int] = {r: 0 for r in RULES}
    why_validator: collections.Counter = collections.Counter()
    accepted: Dict[Tuple[str, int], Dict[str, Any]] = {}
    acc_keys = set()
    attempts_hist: collections.Counter = collections.Counter()
    for t in range(MAX_ATTEMPTS):
        if not pending:
            break
        cands: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for sid, v in pending:
            got = mutate_team(by_src[sid], pri[sid], sid, v, t)
            assert got is not None  # eligibility checked above
            text, slots, recs = got
            src_team, team = parse_team("\n\n".join(by_src[sid])), parse_team(text)
            if any(team.mons[s].key == src_team.mons[s].key for s in slots):
                fails["unchanged"] += 1
                continue
            if label_path_defects(text):
                fails["level"] += 1
                continue
            if pool_ref.keys[team.key]:
                fails["pool_exact"] += 1
                continue
            cid = f"d3m_{sid[5:]}_{v}"
            if HPN.contamination([(cid, text)]):
                fails["panel"] += 1
                continue
            cands[(sid, v)] = {"id": cid, "source": sid, "v": v, "attempt": t, "slots": slots, "k": len(slots),
                               "draws": recs, "text": text, "key": team.key}
        sd = showdown_rows({c["id"]: c["text"] for c in cands.values()}) if cands else {}
        for key in sorted(cands):
            c = cands[key]
            r = sd[c["id"]]
            if r["verdict"] != "ok":
                fails["validator"] += 1
                why_validator[f"{r['verdict']}: {str(r.get('why', ''))[:80]}"] += 1
                continue
            if c["key"] in acc_keys:
                fails["duplicate"] += 1
                continue
            if level_defects(c["id"], c["text"], r["packed"]):
                raise BuildError(f"{c['id']}: Showdown packs a non-100 level for a mutated pool team")
            c["sd"] = r
            acc_keys.add(c["key"])
            accepted[key] = c
            attempts_hist[t] += 1
        pending = [p for p in pending if p not in accepted]
    exhausted = sorted(f"{s}/{v}" for s, v in pending)

    # the engine, re-asked about every id the corpus carries (the prefilter's own check)
    verdicts = engine_verdicts([c["sd"] for c in accepted.values()])
    unplayable = {kind: sorted(i for i, x in verdicts[kind].items() if x != "ran") for kind, _ in KINDS}
    if any(unplayable.values()):
        raise BuildError(f"the corpus carries ids the engine cannot run: {unplayable}")

    ordered = [accepted[k] for k in sorted(accepted)]
    random.Random(ORDER_SEED).shuffle(ordered)
    check_label_encodable([c["id"] for c in ordered], [c["text"] for c in ordered], f"the {NAME} build")
    rows = [{"i": i, "id": c["id"], "tier": TIER, "source": c["source"], "variant": c["v"], "slots": c["slots"],
             "sha256": _sha(c["text"].encode()), "text": c["text"]} for i, c in enumerate(ordered)]
    data = encode_rows(rows)
    texts = [c["text"] for c in ordered]

    # descriptive only (no rule reads it)
    k_hist = collections.Counter(c["k"] for c in ordered)
    per_src = collections.Counter(c["source"] for c in ordered)
    shared = collections.Counter(6 - c["k"] for c in ordered)
    n_mut = sum(c["k"] for c in ordered)
    hp_mut = sum(1 for c in ordered for d in c["draws"] if d["hidden_power"])
    reg_path = HPN.registry_path()
    manifest = {
        "schema": SCHEMA,
        "name": NAME,
        "what": "X41 / D3's set-mutation corpus: our pool teams with 1-2 mons' moves, item, nature and EVs resampled "
                "from Smogon usage (Hidden Power typed by gen3_hp_repair_v1); the species cores, abilities and the "
                "other mons are the pool's; TRAINING DATA, never a prior",
        "built_by": "python -m utils.team_corpora.build_d3_setmut",
        "design": "designs/endstate/design_diversity_teams_opponents.md §4.5 (a); design_beat_metamon_path.md §6.4",
        "seeds": {"build": BUILD_SEED, "order": ORDER_SEED,
                  "rule": "one random.Random per (source id, variant, attempt): stream_seed = sha256('BUILD_SEED|"
                          "source|v|t'); Hidden Power's own stream adds ('hp', slot)"},
        "mutation": {"variants_per_team": VARIANTS_PER_TEAM, "k_choices": list(K_CHOICES), "max_attempts": MAX_ATTEMPTS,
                     "moves": "4 distinct, sequentially without replacement, weight P(move in set); the 16 typed HP "
                              "keys are one 'Hidden Power' candidate with their summed mass",
                     "item": "weight = Smogon item usage ('nothing' = no item)",
                     "nature_evs": "one draw of a Smogon usage spread (nature and EVs joint)",
                     "kept": "species, ability, nickname / gender / shiny / happiness lines; IVs all 31 unless HP",
                     "hidden_power": REPAIR_VERSION, "new_set": "every mutated slot is a new mon set "
                                                               f"({OVERLAP_VERSION}): a variant shares 6 - k sets with "
                                                               "its source"},
        "priors": {f: _sha(repo_path("data", "pokemon", f).read_bytes()) for f in PRIOR_FILES},
        "panel_registry": {"path": str(reg_path.relative_to(repo_root())), "sha256": _sha(reg_path.read_bytes()),
                           "excluded": HPN.panel_names(held_out_only=True),
                           "not_excluded": sorted(set(HPN.panel_names()) - set(HPN.panel_names(held_out_only=True))),
                           "near_neighbour_min_shared": HPN.NEAR_NEIGHBOUR_MIN_SHARED, "matcher": OVERLAP_VERSION,
                           "rule": "a variant hitting a HELD-OUT panel is refused; the in-pool panels (home72, away) "
                                   "are not excluded: D3 perturbs the pool's own cores"},
        "pool": {"n_files": len(pool_texts), "n_distinct_texts": len(seen_src), "sources": len(sources),
                 "source_drops": {r: {"n": len(v), "ids": v} for r, v in src_drops.items()},
                 "content_sha256": _sha("\x00".join(sorted(t.strip() for t in pool_texts)).encode())},
        "engine": {"oracle": "scan_move_probe", "prefilter_unplayable": {"move": sorted(bad_moves),
                                                                           "item": sorted(bad_items)},
                   "dropped_move_slot_mass": {s: m for s, m in dropped_mass.items() if m > 0},
                   "corpus_unplayable": unplayable},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js, pinned "
                     "deps/pokemon-showdown",
        "rule_failures": fails,
        "validator_failures_by_reason": dict(sorted(why_validator.items(), key=lambda kv: (-kv[1], kv[0]))),
        "attempts_to_accept": {str(k): attempts_hist[k] for k in sorted(attempts_hist)},
        "exhausted": {"n": len(exhausted), "ids": exhausted},
        "describe": {"k_hist": {str(k): k_hist[k] for k in sorted(k_hist)},
                     "shared_mon_sets_with_source_hist": {str(k): shared[k] for k in sorted(shared)},
                     "mutated_sets": n_mut, "mutated_sets_with_hidden_power": hp_mut,
                     "variants_per_source_hist": dict(sorted(collections.Counter(per_src.values()).items())),
                     "sources_with_a_variant": len(per_src)},
        "tiers": {TIER: {"n": len(texts), "sha256": tier_sha256(texts), "source_tiers": [TIER]}},
        "default_tier": TIER,
        "data_file": DATA_NAME,
        "data_sha256": _sha(data),
    }
    return data, manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    data, manifest = build()
    out = CORPORA[NAME]
    text = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if a.check:
        same = (out / DATA_NAME).exists() and (out / DATA_NAME).read_bytes() == data \
            and (out / MANIFEST_NAME).read_text() == text
        print("reproduces the committed corpus" if same else "DOES NOT reproduce the committed corpus")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA_NAME).write_bytes(data)
    (out / MANIFEST_NAME).write_text(text)
    print(json.dumps({k: v["n"] for k, v in manifest["tiers"].items()}), json.dumps(manifest["rule_failures"]),
          "exhausted", manifest["exhausted"]["n"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
