"""(Re)build ``w_forum_v1``: X38 / D1's pool-widening corpus, from the Smogon forum v2 collection's VERBATIM teams.

    python -m utils.team_corpora.build_w_forum [--check]

Design: ``designs/endstate/design_diversity_teams_opponents.md`` §4.1 (the TSR proposal: human-built forum teams
ahead of replay reconstructions, tier A first, a lenient strength cut, never X26's rating alone) and §5.6 (freeze,
then widen: the panel registry ``designs/research_state/heldout_panels.json`` was committed first, 8989c137).

The source is ``designs/research_state/measurements/smogon_forum_teams_v2_2026-10-18/`` (``manifest_teams.jsonl`` +
``teams/``). Every step is a stated rule, applied in this order; a team leaves at the FIRST rule that drops it and the
manifest counts each rule per tier:

1. ``status == kept`` and ``fidelity == verbatim`` (the paste site's own text; F2: transcriptions are weak evidence).
   The file's sha256 must equal the row's (a moved source file is a build error, never a different corpus).
2. **Hidden Power repair only where HP is untyped**: :func:`agents.gen3_data.hp_repair.repair_team` (``gen3_hp_repair_v1``,
   the ONE function ``heldout72_hpfix`` used; F-DV-4) with ``team_seed(HP_SEED, id)``. It touches only an UNTYPED Hidden
   Power whose IVs are absent or all 31; a typed one, or one whose IVs pick a type, is byte-identical.
3. ``pool_exact``: the same six MON SETS (``gen3_team_overlap_v1``) as a team of our current 719-team pool.
4. ``panel``: an exact copy or a NEAR NEIGHBOUR (≥ 4 shared mon sets with one team) of ANY registered panel
   (``main.heldout_panels.hits``; the in-pool home72 / away panels included).
5. ``crash_class``: a team carrying a move of an OPEN crash class (``CRASH_MOVES``). EMPTY since 2026-10-19: Transform +
   Hidden Power (the tracker crash, ``0cef9320a``) and Imprison (verified NOT a livelock, ``752aaf16a``) are closed, so
   the 8 teams they excluded are IN; the rule stays so a newly opened class can be re-added (§4.1 step 4(b), F-DV-8).
6. ``format``: the source row names a format other than ``gen3ou`` (``sub_format``; an empty one is the thread's own).
7. ``validator``: the pinned ``TeamValidator('gen3ou')`` rejects the REPAIRED text, or Showdown cannot import it as six
   Pokemon (``src/rust_sim/harness/ladder_corpus_filter.js``, the ladder corpus's own filter).
8. ``engine_unplayable``: a species / move / item / ability the Rust ENGINE did not run AT THIS CORPUS'S BUILD
   (``scan_move_probe``'s answer then, pinned in ``engine_pins.json``; :mod:`utils.team_corpora.engine_pin`). The live
   oracle is still asked: an engine that lost coverage throws, one that gained it changes nothing.
9. ``duplicate``: the same six mon sets as a team already kept (the source deduped before repair).
10. ``strength``: the lenient TSR cut. A team is dropped iff it carries a Kakuna-pilot rating below ``STRENGTH_CUT_ELO``
   in ``measurements/team_strength_2026-10-18/teams.tsv`` (joined by ``main.team_strength.panel.team_id``). An X26-only
   rating never drops a team (X26 rates off-meta teams low partly because it pilots them badly), and a team with no
   rating is KEPT and counted as ``unrated``.
11. **Curator cap** (F4): the largest integer ``c`` such that ``c <= CURATOR_SHARE_CAP`` × the capped total
    ``sum_k min(n_k, c)``; each curator keeps its first ``c`` teams in TIER order (A, then B, then C), within a tier in
    the seeded order below.
12. ``level``, LAST: a mon not at level 100 (Showdown's PACKED level, which must agree with the text's ``Level:`` lines,
    :func:`utils.team_corpora.label_path_defects`). The env's spread labels recompute stats at L100 and FAULT otherwise,
    and ``gen3ou`` caps the level without a floor, so the validator passes such a team: ``sfv2_5051`` (all six at
    level 5) crash-looped D1 s3501 (launch sheet AM-D1-FIX, 2026-10-22). It runs AFTER the cap on purpose: removing an
    id from a tier's seeded shuffle re-permutes the whole tier and the cap then re-selects a capped curator's teams
    (dropping ``sfv2_5051`` before the cap swapped 197 C teams, measured), so a late drop removes exactly the
    unencodable teams and keeps every other team and its order. ``label_guard.source_scan`` in the manifest lists
    every source team with a non-100 level and the rule that dropped it. The build then RE-CHECKS its output with the
    readers' throwing guard (:func:`utils.team_corpora.check_label_encodable`).

The kept teams are ordered tier A, then B, then C; inside each tier, ids sorted then shuffled once with
``random.Random(ORDER_SEED)``. The tiers are prefixes: ``A`` ⊂ ``AB`` ⊂ ``ABC`` (the default). The build is
deterministic: the same source, pool, registry, engine pin and seeds give the same bytes (``--check``).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.paths import repo_path, repo_root
from utils.team_corpora import (CORPORA, DATA_NAME, LABEL_LEVEL, MANIFEST_NAME, SCHEMA, check_label_encodable,
                                 encode_rows, label_path_defects, tier_sha256)

NAME = "w_forum_v1"
SRC_REL = "designs/research_state/measurements/smogon_forum_teams_v2_2026-10-18"
TSR_REL = "designs/research_state/measurements/team_strength_2026-10-18/teams.tsv"
HP_SEED = 20261019
ORDER_SEED = 3801
CURATOR_SHARE_CAP = 0.10
STRENGTH_CUT_ELO = -170.0
#: Moves of an OPEN crash class: a team carrying one is dropped (rule ``crash_class``). EMPTY since 2026-10-19: the two
#: classes that stood here are closed (Transform + Hidden Power belief crash, ``0cef9320a``; Imprison verified NOT a
#: livelock and pinned, ``752aaf16a``). Re-add a move here if a crash class is ever opened again, and rebuild.
CRASH_MOVES: Tuple[str, ...] = ()
TIER_ORDER = ("A", "B", "C")
TIERS = {"A": ("A",), "AB": ("A", "B"), "ABC": ("A", "B", "C")}
DEFAULT_TIER = "ABC"
#: The curator of a row whose source names none (one curator for the cap: unknown builders are not independent).
UNCREDITED = "(uncredited)"
RULES = ("pool_exact", "panel", "crash_class", "format", "validator", "engine_unplayable", "duplicate", "strength",
         "level")
#: The source ``sub_format`` values a team may carry (rule ``format``; "" = the gen3ou thread's own, unlabelled).
FORMATS = ("gen3ou", "")
FILTER_JS = repo_path("src", "rust_sim", "harness", "ladder_corpus_filter.js")

_MOVE_RE = re.compile(r"^\s*-\s*(.+?)\s*$")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def source_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The verbatim kept rows with their texts, and the source's provenance block."""
    src = repo_path(SRC_REL)
    man_path = src / "manifest_teams.jsonl"
    rows = [json.loads(ln) for ln in man_path.read_text().splitlines() if ln.strip()]
    kept = [r for r in rows if r["status"] == "kept"]
    out = []
    for r in kept:
        if r["fidelity"] != "verbatim":
            continue
        raw = (src / "teams" / f"{r['id']}.txt").read_bytes()
        if _sha(raw) != r["sha256"]:
            raise RuntimeError(f"{r['id']}: the source file's sha256 does not match manifest_teams.jsonl")
        out.append({"id": r["id"], "tier": r["tier"], "curator": r["curator"] or UNCREDITED, "category": r["category"],
                    "sub_format": r.get("sub_format", ""), "source_url": r["source_url"], "text": raw.decode().strip()})
    prov = {"dir": SRC_REL, "manifest_teams_sha256": _sha(man_path.read_bytes()),
            "kept": len(kept), "kept_verbatim": len(out), "kept_transcribed_dropped": len(kept) - len(out)}
    return out, prov


def packed_levels(packed: str) -> List[str]:
    """Each mon's level in a Showdown PACKED team (field 10 of ``nick|species|item|ability|moves|nature|evs|gender|ivs|
    shiny|level|...``; empty = 100)."""
    return [(m.split("|")[10] if len(m.split("|")) > 10 else "") or LABEL_LEVEL for m in packed.split("]")]


def level_defects(tid: str, text: str, packed: str) -> List[str]:
    """Rule ``level``: the non-100 levels of a team, from Showdown's packed team; the text's ``Level:`` lines must say
    the same (a disagreement is a build error, never a verdict)."""
    by_packed = sorted(f"Level: {lv}" for lv in packed_levels(packed) if lv != LABEL_LEVEL)
    by_text = sorted(label_path_defects(text))
    if by_packed != by_text:
        raise RuntimeError(f"{tid}: Showdown's packed levels {by_packed} disagree with the text's {by_text}")
    return by_packed


def moves_of(text: str) -> List[str]:
    from utils.showdown_id import to_id_str

    return [to_id_str(m.group(1).split("/")[0]) for ln in text.splitlines() if (m := _MOVE_RE.match(ln))]


def showdown_rows(texts: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """The ladder corpus's Showdown filter over ``{id: text}`` (import, six Pokemon, validator, packed, ids)."""
    with tempfile.TemporaryDirectory(prefix="w_forum_filter_") as d:
        for tid, text in texts.items():
            Path(d, f"{tid}.gen3ou_team").write_text(text + "\n")
        p = subprocess.run(["node", str(FILTER_JS), d], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ladder_corpus_filter.js failed: {p.stderr.strip()[-2000:]}")
    out = {}
    for ln in p.stdout.splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["file"].rsplit(".", 1)[0]] = r
    if set(out) != set(texts):
        raise RuntimeError(f"the Showdown filter answered {len(out)} of {len(texts)} teams")
    return out


def tsr_kakuna_ratings() -> Tuple[Dict[str, float], Dict[str, float]]:
    """``(kakuna_elo, x26_elo)`` by TSR team id, from the committed TSR table."""
    kak, x26 = {}, {}
    with open(repo_path(TSR_REL)) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("kakuna_elo"):
                kak[r["team_id"]] = float(r["kakuna_elo"])
            if r.get("x26_elo"):
                x26[r["team_id"]] = float(r["x26_elo"])
    return kak, x26


def curator_cap(counts: Dict[str, int], share: float) -> int:
    """The largest integer ``c`` with ``c <= share * sum_k min(n_k, c)`` (``c`` in 1..max n)."""
    best = 0
    for c in range(1, max(counts.values()) + 1):
        if c <= share * sum(min(n, c) for n in counts.values()):
            best = c
    if best == 0:
        raise RuntimeError("no curator cap satisfies the share rule")
    return best


def engine_unplayable(sd_rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Rule ``engine_unplayable``'s ids by kind over Showdown filter rows: this corpus's ENGINE PIN (the oracle's answer
    at its build), the live oracle still asked (:func:`utils.team_corpora.engine_pin.unplayable`)."""
    from utils.ladder_corpus.build import engine_verdicts
    from utils.team_corpora import engine_pin

    return engine_pin.unplayable(NAME, engine_verdicts(sd_rows))


def build() -> Tuple[bytes, Dict[str, Any]]:
    from agents.gen3_data.hp_repair import REPAIR_VERSION, repair_team, team_seed
    from main import heldout_panels as HP
    from main.anchors.team_overlap import OVERLAP_VERSION, Reference, parse_items, parse_team
    from main.team_strength.panel import team_id as tsr_team_id
    from utils.ladder_corpus.build import KINDS
    from utils.team_loader import TeamLoader

    HP.verify()
    rows, prov = source_rows()
    drops: Dict[str, collections.Counter] = {r: collections.Counter() for r in RULES}
    dropped_ids: Dict[str, List[str]] = {r: [] for r in RULES}

    def drop(rule: str, r: Dict[str, Any]) -> None:
        drops[rule][r["tier"]] += 1
        dropped_ids[rule].append(r["id"])

    # 2. HP repair where untyped
    n_repaired_teams, n_repaired_mons = 0, 0
    for r in rows:
        new, recs = repair_team(r["text"], team_seed(HP_SEED, r["id"]))
        if recs:
            n_repaired_teams += 1
            n_repaired_mons += len(recs)
            r["text"] = new.strip()
            r["hp_repaired"] = len(recs)

    pool = TeamLoader(str(repo_path("data", "teams"))).get_all_teams()
    pool_ref = Reference(parse_items([(f"pool{i:03d}", t) for i, t in enumerate(pool)]))
    survivors: List[Dict[str, Any]] = []
    for r in rows:
        team = parse_team(r["text"], r["id"])
        if pool_ref.keys[team.key]:
            drop("pool_exact", r)
            continue
        hits = HP.hits([(r["id"], r["text"])])
        if hits:
            r["panel_hits"] = sorted({h.panel for h in hits})
            drop("panel", r)
            continue
        if any(m in CRASH_MOVES for m in moves_of(r["text"])):
            drop("crash_class", r)
            continue
        if r["sub_format"] not in FORMATS:
            drop("format", r)
            continue
        survivors.append(r)

    # 6. validator + import (the REPAIRED text)
    sd = showdown_rows({r["id"]: r["text"] for r in survivors})
    ok = []
    for r in survivors:
        if sd[r["id"]]["verdict"] != "ok":
            drop("validator", r)
        else:
            ok.append(r)
    # 7. engine coverage
    unplayable = engine_unplayable([sd[r["id"]] for r in ok])
    why_engine: collections.Counter = collections.Counter()
    playable = []
    for r in ok:
        bad = [f"{kind}:{i}" for kind, key in KINDS for i in sd[r["id"]][key] if i in unplayable[kind]]
        if bad:
            why_engine.update(bad)
            drop("engine_unplayable", r)
        else:
            playable.append(r)
    # 8. duplicates after repair, in tier order then id
    playable.sort(key=lambda r: (TIER_ORDER.index(r["tier"]), r["id"]))
    seen, uniq = set(), []
    for r in playable:
        k = parse_team(r["text"]).key
        if k in seen:
            drop("duplicate", r)
            continue
        seen.add(k)
        uniq.append(r)
    # 9. the lenient strength cut (Kakuna rating only; X26-only ratings never drop)
    kak, x26 = tsr_kakuna_ratings()
    rated_kak = rated_x26_only = unrated = 0
    graded = []
    for r in uniq:
        tid = tsr_team_id(r["text"])
        if tid in kak:
            rated_kak += 1
            if kak[tid] < STRENGTH_CUT_ELO:
                drop("strength", r)
                continue
        elif tid in x26:
            rated_x26_only += 1
        else:
            unrated += 1
        graded.append(r)

    # 10. the curator cap, tier priority, seeded order inside a tier
    by_tier: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TIER_ORDER}
    for r in graded:
        by_tier[r["tier"]].append(r)
    for t in TIER_ORDER:
        ids = sorted(by_tier[t], key=lambda r: r["id"])
        random.Random(ORDER_SEED).shuffle(ids)
        by_tier[t] = ids
    ordered = [r for t in TIER_ORDER for r in by_tier[t]]
    counts = collections.Counter(r["curator"] for r in ordered)
    cap = curator_cap(dict(counts), CURATOR_SHARE_CAP)
    taken: collections.Counter = collections.Counter()
    capped_drop: collections.Counter = collections.Counter()
    capped = []
    for r in ordered:
        if taken[r["curator"]] >= cap:
            capped_drop[r["tier"]] += 1
            continue
        taken[r["curator"]] += 1
        capped.append(r)
    # 12. LAST: the label path encodes level 100 only (after the cap: no reshuffle, no re-selection)
    level_why: Dict[str, List[str]] = {}
    final = []
    for r in capped:
        why = level_defects(r["id"], r["text"], sd[r["id"]]["packed"])
        if why:
            level_why[r["id"]] = why
            drop("level", r)
            continue
        final.append(r)
    fate = {i: rule for rule, ids in dropped_ids.items() for i in ids}
    source_scan = {r["id"]: {"tier": r["tier"], "levels": sorted(set(d)), "dropped_by": fate.get(r["id"], "curator_cap")}
                   for r in rows if (d := label_path_defects(r["text"]))}

    # descriptive only (no rule reads it): the corpus against our pool and the sample reference's species cores
    sample_ref = [t for t in parse_items(list(HP.panel_team_texts("home72")) + list(HP.panel_team_texts("away")))
                  if len(t.mons) == 6]
    pool_hist: collections.Counter = collections.Counter()
    strata: Dict[str, collections.Counter] = {t: collections.Counter() for t in TIER_ORDER}
    for r in final:
        team = parse_team(r["text"])
        pool_hist[pool_ref.max_shared(team)[0]] += 1
        k = max(len(team.species6 & s.species6) for s in sample_ref)
        strata[r["tier"]]["CORE" if k == 6 else "NEAR" if k >= 4 else "FAR"] += 1
    describe = {"pool_max_shared_mon_sets_hist": {str(k): pool_hist.get(k, 0) for k in range(7)},
                "ladder_strata_by_tier": {t: dict(strata[t]) for t in TIER_ORDER},
                "strata_rule": "ladder_weights_v1's rule: best SPECIES overlap with one sample team (home72 + away): "
                               "CORE 6, NEAR 4-5, FAR <= 3 (descriptive)"}

    check_label_encodable([r["id"] for r in final], [r["text"] for r in final], f"the {NAME} build")
    out_rows = [{"i": i, "id": r["id"], "tier": r["tier"], "curator": r["curator"],
                 "sha256": _sha(r["text"].encode()), "text": r["text"]} for i, r in enumerate(final)]
    data = encode_rows(out_rows)
    texts = [r["text"] for r in final]
    tiers = {}
    for name, members in TIERS.items():
        n = sum(1 for r in final if r["tier"] in members)
        tiers[name] = {"n": n, "sha256": tier_sha256(texts[:n]), "source_tiers": list(members)}
    by_tier_curator = {t: dict(collections.Counter(r["curator"] for r in final if r["tier"] == t).most_common())
                       for t in TIER_ORDER}
    reg_path = HP.registry_path()
    manifest = {
        "schema": SCHEMA,
        "name": NAME,
        "what": "X38 / D1's pool-widening corpus: human-built gen3ou teams from the Smogon forum v2 collection "
                "(VERBATIM pastes), minus every registered panel and its near neighbours, minus our pool's exact "
                "teams; TRAINING DATA, never a prior",
        "built_by": "python -m utils.team_corpora.build_w_forum",
        "source": prov,
        "panel_registry": {"path": str(reg_path.relative_to(repo_root())), "sha256": _sha(reg_path.read_bytes()),
                           "panels": HP.panel_names(), "near_neighbour_min_shared": HP.NEAR_NEIGHBOUR_MIN_SHARED,
                           "matcher": OVERLAP_VERSION},
        "pool": {"n": len(pool), "rule": "exact six-mon-set copies of a pool team are dropped; pool NEAR "
                                         "neighbours are kept (they are new teams)"},
        "hp_repair": {"version": REPAIR_VERSION, "seed": HP_SEED,
                      "seed_rule": "team_seed(seed, source id); only an UNTYPED Hidden Power with absent / all-31 IVs "
                                   "is a target", "teams_repaired": n_repaired_teams, "mons_repaired": n_repaired_mons},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js on the REPAIRED "
                     "text, pinned deps/pokemon-showdown",
        "engine": {"oracle": "scan_move_probe", "unplayable_ids": unplayable,
                   "unplayable_by_id": dict(why_engine.most_common())},
        "crash_class_moves": list(CRASH_MOVES),
        "formats": list(FORMATS),
        "label_guard": {"rule": "a mon not at level 100 (Showdown's packed level, agreeing with the text) is dropped: "
                                "the env's spread labels recompute stats at L100 and FAULT otherwise (AM-D1-FIX); "
                                "every reader re-checks with utils.team_corpora.check_label_encodable",
                        "level": int(LABEL_LEVEL), "dropped": {k: sorted(set(v)) for k, v in sorted(level_why.items())},
                        "source_scan": dict(sorted(source_scan.items()))},
        "strength_cut": {"rule": f"drop iff a Kakuna-pilot TSR rating < {STRENGTH_CUT_ELO} Elo; an X26-only rating "
                                 "never drops; an unrated team is KEPT",
                         "table": TSR_REL, "table_sha256": _sha(repo_path(TSR_REL).read_bytes()),
                         "rated_kakuna": rated_kak, "rated_x26_only": rated_x26_only, "unrated_kept": unrated},
        "curator_cap": {"share": CURATOR_SHARE_CAP, "cap": cap,
                        "rule": "the largest c with c <= share x sum_k min(n_k, c); each curator keeps its first c "
                                "teams in tier order (A, B, C), seeded order inside a tier",
                        "dropped_by_tier": dict(capped_drop), "curators": len(taken),
                        "top": dict(taken.most_common(15))},
        "order": {"seed": ORDER_SEED, "rule": "tier A, then B, then C; ids sorted then shuffled once per tier"},
        "drops": {rule: {"by_tier": dict(drops[rule]), "n": sum(drops[rule].values())} for rule in RULES},
        "dropped_ids": dropped_ids,
        "by_tier_curator": by_tier_curator,
        "describe": describe,
        "tiers": tiers,
        "default_tier": DEFAULT_TIER,
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
    print(json.dumps({k: v["n"] for k, v in manifest["tiers"].items()}),
          json.dumps({k: v["n"] for k, v in manifest["drops"].items()}), "cap", manifest["curator_cap"]["cap"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
