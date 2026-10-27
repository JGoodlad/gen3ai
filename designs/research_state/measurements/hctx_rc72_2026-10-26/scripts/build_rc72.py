"""Build rc72, the RECOMBINED panel of the H-CTX test (PREREG.md §2): 72 new gen3ou teams made only of mon sets our
training pool holds, no team sharing more than 2 sets with any single team any arm trains on.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 \
        python designs/research_state/measurements/hctx_rc72_2026-10-26/scripts/build_rc72.py [--check] [--install]

It plays nothing. Every input is either our pool (``main.anchors.session.pool_team_texts``: the 719 teams P* trained
on, the source of every SET), home72's species counts (the profile), or Smogon (``agents.gen3_data.priors``: the
teammate co-occurrence that keeps a team coherent). Nothing here ships as a prior; the panel is an evaluation set.

THE DRAW of team ``i``, attempt ``a`` (one ``random.Random(stream_seed(i, a))``):

1. Six slots in order. Slot 1's species is drawn with weight ``c_h(s)``, its count over home72's 432 mons (only the
   41 species home72 holds can be drawn: "match home72's profile"). Each later slot's species, among those not yet on
   the team (Species Clause), with weight ``c_h(s) * geomean_t lift(t, s)`` over the team's species ``t``, where
   ``lift(t, s) = P(s | t) / share(s)`` (Smogon teammates over Smogon usage share), clipped to [1/4, 4]; a species
   with no teammate row reads lift 1.
2. The slot's SET: one OCCURRENCE of the species in the pool, uniform over occurrences (so a set is drawn as often
   as the pool repeats it), copied VERBATIM (nickname, item, ability, EVs, nature, IVs and moves as the pool team
   wrote them). The slot is refused, and redrawn (species and set, up to ``SLOT_TRIES`` times; then the attempt
   fails), when the partial team would:
   * share more than ``MAX_SHARED_TRAIN`` (2) mon sets (``gen3_team_overlap_v1``: species + sorted moves, Hidden Power
     typed) with ONE team of the pool or of a committed training corpus (``utils.team_corpora.CORPORA``: D1's
     ``w_forum_v1``, D3's ``d3_setmut_v1``, N1-P's ``n1_proc_v1``);
   * share ``HELDOUT_REFUSE`` (3) or more with ONE team of a held-out panel (``heldout_panels.panel_names(
     held_out_only=True)`` plus X61's unregistered ``n1p72``): REVIEW A's revealed-fit rule (B2), the X61 corpus's.
3. A complete candidate is then refused (the index moves to its next attempt) when Showdown's pinned
   ``TeamValidator('gen3ou')`` rejects it (``ladder_corpus_filter.js``, the corpora's own filter), Showdown packs a
   level other than 100 (the pack oracle, ``build_w_forum.level_defects``), a Hidden Power does not play its named
   type at 70 once validated (``hp_set_check.js``; :func:`hp_bad`), or it shares 4 or more sets with an rc72 team already
   accepted (the panel's own near-neighbour rule; teams are accepted in index order).

After the draw, the engine is asked about every move and item id the panel carries (``scan_move_probe``) and the
build THROWS on an unplayable one; the label path is checked (``check_label_encodable``). Deterministic: the same
pool, corpora, panels, priors and seeds give the same bytes (``--check``).

``--install`` writes the committed ``teams_rc72/``, the manifest, the anchors config, and the two byte copies the
symmetric Kakuna cell reads: ours (``READS_DIR/teams_rc72``) and Metamon's cache set ``gen3ai_rc72``. Idempotent;
refuses to overwrite different bytes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path, repo_root

HERE = Path(__file__).resolve().parent.parent
#: The two draws. ``rc72`` (RETIRED before any game counted): its Hidden Power rule let 71 sets through whose
#: text names a type but carries no IV line (all 31); the live server REJECTS such a team (``rc72__00`` failed twice,
#: "Aerodactyl has Hidden Power Bug, but its IVs are for Hidden Power Water"). ``rc72b``: X37a's strict rule.
PANELS = {"rc72": "validated", "rc72b": "strict"}
NAME = "rc72b"
HP_RULE = PANELS[NAME]
N_TEAMS = 72
BUILD_SEED = 7272
MAX_SHARED_TRAIN = 2
HELDOUT_REFUSE = 3
WITHIN_REFUSE = 4
LIFT_CLIP = (0.25, 4.0)
SLOT_TRIES = 80
MAX_ATTEMPTS = 60
N1P72_DIR = "designs/research_state/measurements/x61_n1_breadth_2026-10-25/teams_n1p72"
READS_DIR = Path("/home/goodlad/dev/gen3ai-reads/hctx_rc72_2026-10-26")
METAMON_TEAMS = Path("/home/goodlad/dev/metamon/cache/teams")
BASE_CONFIG = "designs/research_state/measurements/x37a_eval_matrix_2026-10-18/configs/anchors_x37a_forum72.json"
CONFIG_NAME = f"anchors_{NAME}_kakuna.json"
MANIFEST_NAME = f"{NAME}_manifest.json"
TEAMS_DIR = HERE / f"teams_{NAME}"


def set_panel(name: str) -> None:
    """Point the module at one draw (``--panel``): its name, Hidden Power rule and file names."""
    global NAME, HP_RULE, CONFIG_NAME, MANIFEST_NAME, TEAMS_DIR
    NAME, HP_RULE = name, PANELS[name]
    CONFIG_NAME, MANIFEST_NAME, TEAMS_DIR = f"anchors_{name}_kakuna.json", f"{name}_manifest.json", HERE / f"teams_{name}"


def stream_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in (BUILD_SEED,) + parts).encode()).hexdigest()
    return int(h[:16], 16)


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def weighted_pick(rng: random.Random, entries: Sequence[Tuple[str, float]]) -> str:
    total = sum(w for _, w in entries)
    r = rng.random() * total
    for k, w in entries:
        r -= w
        if r < 0:
            return k
    return entries[-1][0]


def split_blocks(text: str) -> List[str]:
    import re

    return [b.strip("\n") for b in re.split(r"\n[ \t]*\n+", text.strip()) if b.strip()]


# --------------------------------------------------------------------------------------------- the references
def references() -> Dict[str, Any]:
    from main import heldout_panels as HPN
    from main.anchors import team_overlap as TO
    from main.anchors.session import load_team_texts, pool_team_texts
    from utils import team_corpora as TC

    HPN.verify()
    pool = sorted(pool_team_texts(), key=sha_text)
    train = {"pool": TO.Reference([TO.parse_team(t, "pool_" + sha_text(t)[:12]) for t in pool])}
    for name in sorted(TC.CORPORA):
        c = TC.load_corpus(name, "ALL" if name != "w_forum_v1" else "ABC")
        train[name] = TO.Reference(TO.parse_items(list(zip(c.ids, c.texts))))
    held = {p: TO.Reference(TO.parse_items(HPN.panel_team_texts(p))) for p in HPN.panel_names(held_out_only=True)}
    held["n1p72"] = TO.Reference(TO.parse_items(load_team_texts(repo_path(N1P72_DIR))))
    home72 = TO.parse_items(HPN.panel_team_texts("home72"))
    return {"pool_texts": pool, "train": train, "held": held, "home72": home72}


def pool_occurrences(pool: Sequence[str]) -> Dict[str, List[Tuple[str, Any, str]]]:
    """``{species: [(verbatim block, MonSet, source content id), ...]}`` over every mon of every pool team."""
    from main.anchors.team_overlap import parse_mon

    out: Dict[str, List[Tuple[str, Any, str]]] = collections.defaultdict(list)
    for t in pool:
        sid = "pool_" + sha_text(t)[:12]
        for b in split_blocks(t):
            m = parse_mon(b)
            if m is not None:
                out[m.species].append((b, m, sid))
    return dict(out)


def lift_table(species: Sequence[str]) -> Dict[Tuple[str, str], float]:
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


# ------------------------------------------------------------------------------------------------- the draw
def max_shared(refs: Dict[str, Any], mons: Sequence[Any]) -> Tuple[int, Optional[str], Optional[str]]:
    from main.anchors.team_overlap import Team

    team = Team(label="cand", mons=tuple(mons))
    best = (0, None, None)
    for name, ref in refs.items():
        k, lab = ref.max_shared(team)
        if k > best[0]:
            best = (k, name, lab)
    return best


def draw_team(i: int, a: int, profile: Dict[str, int], occ: Dict[str, List[Tuple[str, Any, str]]],
              lift: Dict[Tuple[str, str], float], refs: Dict[str, Any], fails: collections.Counter
              ) -> Optional[List[Tuple[str, Any, str]]]:
    rng = random.Random(stream_seed(i, a))
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
            k, _n, _l = max_shared(refs["train"], mons)
            if k > MAX_SHARED_TRAIN:
                fails["train_overlap"] += 1
                continue
            k, _n, _l = max_shared(refs["held"], mons)
            if k >= HELDOUT_REFUSE:
                fails["heldout_fit"] += 1
                continue
            team.append(pick)
            break
        else:
            fails["slot_exhausted"] += 1
            return None
    return team


def hp_bad(verdict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """``strict`` (rc72b; X37a's rule): a Hidden Power that does not play its named type at 70 AS WRITTEN, or whose
    IVs the validator rewrites. 737 of the pool's 1,912 Hidden Power sets name a type and carry no IV line, and the
    live server rejects them (the batch validator only rewrites them, which is why ``validated`` let them through);
    home72 and F hold none. ``validated`` (rc72, retired): the named type once the validator has set the IVs."""
    if HP_RULE == "validated":
        return [s for s in verdict.get("sets", []) if s.get("hp_move") and (
            s.get("validator_hp_type") != s["hp_move"].split()[-1] or s.get("power") != 70)]
    return [s for s in verdict.get("sets", []) if s.get("hp_move") and (
        s.get("hp_type") != s["hp_move"].split()[-1] or s.get("power") != 70
        or s.get("ivs_after_validate") != s.get("ivs_in"))]


def build() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from main import heldout_panels as HPN
    from main.anchors import team_overlap as TO
    from main.anchors.session import team_set_digest
    from utils.bridge.hp_set_check import check_teams
    from utils.ladder_corpus.build import KINDS, engine_verdicts
    from utils.team_corpora import check_label_encodable
    from utils.team_corpora.build_w_forum import level_defects, showdown_rows

    refs = references()
    occ = pool_occurrences(refs["pool_texts"])
    profile = dict(collections.Counter(m.species for t in refs["home72"] for m in t.mons))
    missing = sorted(s for s in profile if s not in occ)
    if missing:
        raise SystemExit(f"home72 species with no pool occurrence: {missing}")
    lift = lift_table(sorted(profile))
    fails: collections.Counter = collections.Counter()
    accepted: Dict[int, Dict[str, Any]] = {}
    acc_teams: List[Any] = []
    pending = list(range(N_TEAMS))
    for a in range(MAX_ATTEMPTS):
        if not pending:
            break
        cands: Dict[int, Dict[str, Any]] = {}
        for i in pending:
            got = draw_team(i, a, profile, occ, lift, refs, fails)
            if got is None:
                continue
            text = "\n\n".join(b for b, _, _ in got) + "\n"
            cands[i] = {"id": f"{NAME}_{i:02d}", "attempt": a, "text": text,
                        "mons": [{"species": m.species, "moves": list(m.moves), "source": sid} for _, m, sid in got]}
        if cands:
            sd = showdown_rows({c["id"]: c["text"] for c in cands.values()})
            order = sorted(cands)
            hp = dict(zip(order, check_teams([cands[i]["text"] for i in order])))
        for i in sorted(cands):
            c = cands[i]
            r = sd[c["id"]]
            if r["verdict"] != "ok":
                fails["validator"] += 1
                continue
            if level_defects(c["id"], c["text"], r["packed"]):
                fails["level"] += 1
                continue
            if hp_bad(hp[i]):
                fails["hidden_power"] += 1
                continue
            team = TO.parse_team(c["text"], c["id"])
            if acc_teams and TO.Reference(acc_teams).max_shared(team)[0] >= WITHIN_REFUSE:
                fails["within_panel"] += 1
                continue
            c["sd"] = r
            accepted[i] = c
            acc_teams.append(team)
        pending = [i for i in pending if i not in accepted]
    if pending:
        raise SystemExit(f"{len(pending)} indices exhausted after {MAX_ATTEMPTS} attempts: {pending}")
    rows = [accepted[i] for i in range(N_TEAMS)]
    verdicts = engine_verdicts([r["sd"] for r in rows])
    unplayable = {kind: sorted(x for x, v in verdicts[kind].items() if v != "ran") for kind, _ in KINDS}
    if any(unplayable.values()):
        raise SystemExit(f"the panel carries ids the engine cannot run: {unplayable}")
    check_label_encodable([r["id"] for r in rows], [r["text"] for r in rows], "rc72")

    # descriptive (the PREREG reads none of it)
    teams = [TO.parse_team(r["text"], r["id"]) for r in rows]
    overlap = {name: TO.overlap(teams, ref) for name, ref in {**refs["train"], **refs["held"]}.items()}
    hist = {}
    for name, ref in {**refs["train"], **refs["held"]}.items():
        hist[name] = dict(sorted(collections.Counter(ref.max_shared(t)[0] for t in teams).items()))
    within = TO.Reference(teams)
    within_max = []
    for j, t in enumerate(teams):
        others = TO.Reference(teams[:j] + teams[j + 1:])
        within_max.append(others.max_shared(t)[0])
    rc_species = collections.Counter(m.species for t in teams for m in t.mons)
    pool_keys = collections.Counter(m.key for t in refs["train"]["pool"].teams for m in t.mons)
    set_freq = [pool_keys[m.key] for t in teams for m in t.mons]
    home_freq = [pool_keys[m.key] for t in refs["home72"] for m in t.mons]
    del within
    manifest = {
        "schema": "hctx_rc72_manifest_v1",
        "built_by": "designs/research_state/measurements/hctx_rc72_2026-10-26/scripts/build_rc72.py",
        "rules": "PREREG.md §2 and this script's docstring",
        "seeds": {"build": BUILD_SEED, "stream": "random.Random(int(sha256('BUILD_SEED|i|a')[:16], 16)) per "
                                                   "(team index i, attempt a)"},
        "params": {"n": N_TEAMS, "max_shared_train": MAX_SHARED_TRAIN, "heldout_refuse": HELDOUT_REFUSE,
                   "within_refuse": WITHIN_REFUSE, "lift_clip": list(LIFT_CLIP), "slot_tries": SLOT_TRIES,
                   "max_attempts": MAX_ATTEMPTS},
        "inputs": {"pool": {"n": len(refs["pool_texts"]),
                            "content_sha256": sha_text("\x00".join(sorted(refs["pool_texts"])))},
                   "train_references": {k: len(v.teams) for k, v in refs["train"].items()},
                   "heldout_references": {k: len(v.teams) for k, v in refs["held"].items()},
                   "panel_registry_sha256": HPN.registry_sha256(),
                   "priors": {f: hashlib.sha256(repo_path("data", "pokemon", f).read_bytes()).hexdigest()
                              for f in ("gen3_teammate_priors.json", "gen3_smogon_stats.json")},
                   "home72_profile": dict(sorted(profile.items(), key=lambda kv: (-kv[1], kv[0])))},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js, pinned "
                     "deps/pokemon-showdown; level 100 by Showdown's pack; hp_set_check.js",
        "rule_failures": dict(sorted(fails.items())),
        "attempts_to_accept": dict(sorted(collections.Counter(r["attempt"] for r in rows).items())),
        "teams": {r["id"]: {"sha256": sha_text(r["text"]), "attempt": r["attempt"], "mons": r["mons"]} for r in rows},
        "content_digest": team_set_digest([r["text"] for r in rows]),
        "describe": {
            "max_shared_hist": hist, "overlap": overlap, "within_panel_max_shared": dict(
                sorted(collections.Counter(within_max).items())),
            "species": {"n_distinct": len(rc_species),
                        "rc72": dict(sorted(rc_species.items(), key=lambda kv: (-kv[1], kv[0]))),
                        "l1_vs_home72": sum(abs(rc_species.get(s, 0) - profile.get(s, 0))
                                            for s in set(rc_species) | set(profile)) / (2 * 432)},
            "pool_repeats_of_each_set": {"rc72_median": sorted(set_freq)[len(set_freq) // 2],
                                         "home72_median": sorted(home_freq)[len(home_freq) // 2],
                                         "rc72_mean": sum(set_freq) / len(set_freq),
                                         "home72_mean": sum(home_freq) / len(home_freq)},
            "distinct_mon_sets": len({m.key for t in teams for m in t.mons}),
        },
    }
    if NAME != "rc72":          # the retired draw's manifest predates these two keys (it must still reproduce)
        manifest.update({"panel": NAME, "hidden_power_rule": HP_RULE})
    return rows, manifest


# --------------------------------------------------------------------------------------------- install
def write_if_same(p: Path, text: str) -> None:
    if p.exists():
        if p.read_text() != text:
            raise SystemExit(f"existing file differs: {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def config_text() -> str:
    cfg = json.loads(repo_path(BASE_CONFIG).read_text())
    cfg["_README"] = [f"H-CTX symmetric cell rc72 (hctx_rc72_2026-10-26): both sides pilot rc72 (our byte copy; "
                      f"Metamon set gen3ai_{NAME}), checked against the diagnostic panel registry before every unit."]
    cfg["opponents"]["metamon"]["team_sets"]["home"] = f"gen3ai_{NAME}"
    cfg["opponents"]["foulplay"]["team_dirs"]["home"] = str(READS_DIR / f"teams_{NAME}")
    cfg["our_team_sources"]["home"]["path"] = str(READS_DIR / f"teams_{NAME}")
    return json.dumps(cfg, indent=1) + "\n"


def install(rows: Sequence[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    dirs = ((TEAMS_DIR, ".txt"), (READS_DIR / f"teams_{NAME}", ".txt"),
            (METAMON_TEAMS / f"gen3ai_{NAME}" / "gen3ou", ".gen3ou_team"))
    for r in rows:
        for d, ext in dirs:
            write_if_same(d / f"{r['id']}{ext}", r["text"])
    for d, ext in dirs:
        if sorted(p.name for p in d.iterdir()) != sorted(r["id"] + ext for r in rows):
            raise SystemExit(f"{d} holds files outside the panel")
    write_if_same(HERE / MANIFEST_NAME, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    write_if_same(HERE / CONFIG_NAME, config_text())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed panel and manifest reproduce")
    ap.add_argument("--install", action="store_true", help="write the panel, manifest, config and byte copies")
    ap.add_argument("--panel", choices=sorted(PANELS), default="rc72b", help="which draw (rc72 is retired)")
    a = ap.parse_args(argv)
    set_panel(a.panel)
    rows, manifest = build()
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if a.check:
        same = (HERE / MANIFEST_NAME).exists() and (HERE / MANIFEST_NAME).read_text() == text and all(
            (TEAMS_DIR / f"{r['id']}.txt").exists() and (TEAMS_DIR / f"{r['id']}.txt").read_text() == r["text"]
            for r in rows)
        print("reproduces the committed panel" if same else "DOES NOT reproduce the committed panel")
        return 0 if same else 1
    if a.install:
        install(rows, manifest)
    print(json.dumps({"content_digest": manifest["content_digest"], "rule_failures": manifest["rule_failures"],
                      "attempts": manifest["attempts_to_accept"], "describe": {
                          k: v for k, v in manifest["describe"].items() if k != "overlap"}}, indent=1))
    print("root", repo_root())
    return 0


if __name__ == "__main__":
    sys.exit(main())
