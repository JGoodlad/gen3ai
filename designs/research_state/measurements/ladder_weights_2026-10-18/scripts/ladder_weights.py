"""LADDER WEIGHTS v1 — how much the ladder resembles the SAMPLE teams, as frozen strata weights (2026-10-18).

For `design_diversity_teams_opponents.md` §5.3 (the ladder-weighted headline). It plays nothing. It reads:
  * the ladder proxy: Metamon's `hl_05_26/gen3ou` (22,862 replay-derived teams; the archived copy);
  * the SAMPLE reference: home72 (`n0_endofrun_2026-09-27/teams_home72`, our `data/teams/sample`) ∪ Metamon's
    `competitive/gen3ou` (`away`);
  * heldout72 (`heldout_teams_kakuna_2026-10-18/teams_heldout72`) and, when present, the parallel agent's
    `heldout72_hpfix` copy, read-only.

For each ladder team it takes the best SPECIES overlap with any one sample team (0–6) and the best MON-SET overlap
(`gen3_team_overlap_v1`). The strata are CORE (all 6 species of one sample team), NEAR (4–5) and FAR (≤ 3). The
weights are each stratum's share of the ladder proxy, counted per file (a file is one replay team, so popular teams
weigh by how often they were played). It also assigns every heldout72 / heldout72_hpfix team to a stratum, which
gives X37a's per-stratum sub-panels.

The orchestrator's quick script (`~/.cache/gen3ai/hl_vs_samples.py`, a lighter parser) reads the same
distribution. This module is the versioned one: `gen3_team_overlap_v1` (formes, IV-typed Hidden Power).

Usage (rule 5b, CPU only):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 \
    python designs/research_state/measurements/ladder_weights_2026-10-18/scripts/ladder_weights.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from main.anchors import team_overlap as TO
from utils.paths import repo_path, repo_root

MEAS = repo_path("designs", "research_state", "measurements")
OUT = MEAS / "ladder_weights_2026-10-18" / "ladder_weights_v1.json"
LADDER = Path(os.path.expanduser("~/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou"))
REF_DIRS = {
    "home72": MEAS / "n0_endofrun_2026-09-27" / "teams_home72",
    "away": Path(os.path.expanduser("~/dev/metamon/cache/teams/competitive/gen3ou")),
}
PANELS = {
    "heldout72": MEAS / "heldout_teams_kakuna_2026-10-18" / "teams_heldout72",
    "heldout72_hpfix": Path(os.path.expanduser(
        "~/dev/gen3ai-reads/heldout72hp_kakuna_2026-10-18/teams_heldout72_hpfix")),
}
STRATA = (("CORE", lambda k: k == 6), ("NEAR", lambda k: 4 <= k <= 5), ("FAR", lambda k: k <= 3))


def team_files(d: Path) -> list:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix in (".txt", ".gen3ou_team"))


def digest(files: list) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def shown(d: Path) -> str:
    """A path as it is recorded: repo-relative inside this checkout (so the JSON does not name a worktree)."""
    try:
        return str(Path(d).resolve().relative_to(Path(repo_root()).resolve()))
    except ValueError:
        return str(d).replace(os.path.expanduser("~"), "~", 1)


def stratum(k: int) -> str:
    return next(name for name, pred in STRATA if pred(k))


def main() -> int:
    ref_files = [f for d in REF_DIRS.values() for f in team_files(d)]
    ref = [TO.parse_team(f.read_text(errors="ignore"), f.name) for f in ref_files]
    ref = [t for t in ref if len(t.mons) == 6]
    ref_species = list({t.species6 for t in ref})
    ref_index = TO.Reference(ref)

    def best_species(t) -> int:
        s = t.species6
        return max(len(s & r) for r in ref_species)

    lad_files = team_files(LADDER)
    if len(lad_files) != 22862:
        raise SystemExit(f"expected 22,862 hl_05_26 teams, found {len(lad_files)}")
    sp_hist: Counter = Counter()
    ms_hist: Counter = Counter()
    n_ok = 0
    for f in lad_files:
        t = TO.parse_team(f.read_text(errors="ignore"), f.name)
        if len(t.mons) != 6:
            continue
        n_ok += 1
        sp_hist[best_species(t)] += 1
        ms_hist[ref_index.max_shared(t)[0]] += 1
    weights = {name: round(sum(v for k, v in sp_hist.items() if pred(k)) / n_ok, 4) for name, pred in STRATA}

    panels = {}
    for pname, d in PANELS.items():
        if not d.is_dir():
            panels[pname] = {"status": "ABSENT on this box at run time", "path": shown(d)}
            continue
        files = team_files(d)
        members = {name: [] for name, _ in STRATA}
        hist: Counter = Counter()
        for f in files:
            t = TO.parse_team(f.read_text(errors="ignore"), f.name)
            k = best_species(t)
            hist[k] += 1
            members[stratum(k)].append(f.stem)
        panels[pname] = {"path": shown(d), "n": len(files), "digest": digest(files),
                         "best_species_overlap_hist": {str(k): hist[k] for k in sorted(hist)},
                         "strata_counts": {k: len(v) for k, v in members.items()},
                         "strata_members": members}

    # descriptive, post hoc: each model's heldout72 score split by the stratum of the team OUR side piloted
    # (the peer's per-game team is not recoverable from the banked rows: design doc F-DV-6)
    member_of = {stem: name for name, stems in panels["heldout72"]["strata_members"].items() for stem in stems}
    piloted: dict = {}
    for line in open(MEAS / "heldout_teams_kakuna_2026-10-18" / "rows.jsonl"):
        r = json.loads(line)
        model = r["read_cell"].split("_")[0]
        st = member_of[Path(r["our_team"]).stem]
        score = 1.0 if r["result"] == "win" else (0.5 if r["result"] == "tie" else 0.0)
        cell = piloted.setdefault(model, {}).setdefault(st, [0, 0.0])
        cell[0] += 1
        cell[1] += score
    piloted_by_stratum = {m: {st: {"games": n, "score": round(sc / n, 4)} for st, (n, sc) in v.items()}
                          for m, v in piloted.items()}

    doc = {
        "schema": "gen3_ladder_weights_v1",
        "status": "DRAFT — frozen only by X37a's registration commit; a change of any input is a NEW version",
        "matcher": TO.OVERLAP_VERSION,
        "ladder_proxy": {"path": shown(LADDER), "files": len(lad_files), "parsed_six_mon_teams": n_ok,
                         "digest": digest(lad_files)},
        "sample_reference": {"dirs": {k: shown(v) for k, v in REF_DIRS.items()}, "files": len(ref_files),
                             "six_mon_teams": len(ref), "distinct_species_cores": len(ref_species),
                             "digest": digest(ref_files)},
        "best_species_overlap_hist": {str(k): sp_hist[k] for k in range(7)},
        "best_species_overlap_share": {str(k): round(sp_hist[k] / n_ok, 4) for k in range(7)},
        "best_mon_set_overlap_share": {str(k): round(ms_hist[k] / n_ok, 4) for k in range(7)},
        "strata": {"CORE": "all 6 species of one sample team", "NEAR": "4-5 species", "FAR": "<= 3 species"},
        "weights": weights,
        "panels": panels,
        "heldout72_score_by_piloted_stratum_descriptive": piloted_by_stratum,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("ladder_proxy", "sample_reference", "best_species_overlap_share",
                                          "best_mon_set_overlap_share", "weights")}, indent=1))
    print("heldout72 score by the stratum OUR side piloted (descriptive):", piloted_by_stratum)
    for p, v in panels.items():
        print(p, {k: v.get(k) for k in ("n", "best_species_overlap_hist", "strata_counts", "status")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
