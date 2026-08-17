"""Team ARCHETYPE classifier — label every pool team by pace class + style tags.

The archetype-competence study measured a style gradient (turn-1 P(win): offense 0.69 / stall
0.55 / balance 0.46) over UNIFORMLY-sampled teams, and the league plan targets exploiters at
STYLES (trap, CM-stall, hyper-offense). Both need the team pool labeled. This derives
``data/teams/gen3_team_archetypes.json`` from the pool (the ``bot_elo_calibration`` /
``bot_elo_calibration`` committed-artifact pattern): rule-based composition features over the
parsed Showdown exports + the ``gen3_data`` facade (base stats, damaging-move classification),
with a k-means cross-tab printed as an unsupervised sanity check (agreement, not ground truth).

Each team gets:
  * ``archetype`` — one of ``ARCHETYPE_CLASSES`` (a coarse PACE spectrum). Deliberately coarse:
    canonical gen3 TSS classifies ``balance`` (it IS bulky attrition), and that's honest — the
    STYLE nuance lives in the tags, not the 5-way.
  * ``pace_score`` — the continuous offense(+)/stall(−) rubric the class thresholds cut.
  * ``tags`` — boolean style markers (``sand``/``spikes``/``spin``/``spinblock``/``phaze``/
    ``trap``/``toxic``/``wish``/``screens``/``boom``/``choice``/``setup_heavy``/``rain``) — the
    league-targeting axis (``trap`` = the Magneton/Dugtrio shortlist).

Teams are keyed by ``sha1(team_str)[:10]`` — the SAME fingerprint convention as the MatchupSpec
``pin_sha`` / fold-back provenance, so archetype labels join every other record.

CLI (writes the artifact + prints the distribution / cross-tab / trap shortlist):
    export PYTHONPATH=$PYTHONPATH:src
    python -m agents.training.team_archetypes [--out data/teams/gen3_team_archetypes.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field

from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

from agents.gen3_data import moves as g3moves
from agents.gen3_data import species as g3species

ARTIFACT_PATH = os.path.join("data", "teams", "gen3_team_archetypes.json")
ARCHETYPE_CLASSES = ("hyper_offense", "offense", "balance", "semi_stall", "stall")

# ── move/ability vocabularies (gen3-legal; ids via to_id_str) ────────────────────
RECOVERY_MOVES = {"softboiled", "recover", "rest", "wish", "moonlight", "morningsun",
                  "synthesis", "slackoff", "milkdrink"}
STATUS_MOVES = {"toxic", "willowisp", "thunderwave", "spore", "sleeppowder", "stunspore",
                "hypnosis", "lovelykiss", "sing", "glare", "yawn", "poisonpowder"}
SETUP_MOVES = {"swordsdance", "dragondance", "calmmind", "bellydrum", "curse", "agility",
               "tailglow", "growth", "meditate", "howl", "irondefense", "amnesia", "barrier",
               "acidarmor", "cosmicpower", "bulkup"}
PHAZE_MOVES = {"roar", "whirlwind"}
TRAP_MOVES = {"meanlook", "spiderweb", "block"}
TRAP_ABILITIES = {"arenatrap", "magnetpull", "shadowtag"}
BOOM_MOVES = {"explosion", "selfdestruct"}
SCREEN_MOVES = {"reflect", "lightscreen"}
WEATHER_SAND = {"sandstream"}          # ability (TTar); the move `sandstorm` also counts below
FAST_BASE_SPEED = 100                  # the gen3 "fast tier" line (Starmie/Gengar/Jolteon/Aero)

# The pace rubric: offense-positive minus stall-negative, in "mons" units. Transparent module
# constants (not fitted) — the classes are coarse on purpose; tune here if the cross-tab drifts.
_W = {"setup": 1.0, "attacker": 0.6, "fast": 0.4, "boom": 0.5, "choice": 0.4,
      "recovery": -1.1, "status": -0.4, "spikes": -0.5, "phaze": -0.5, "screen": -0.2}
# pace_score cut points (descending) → ARCHETYPE_CLASSES. Calibrated to the pool's observed
# pace distribution (p20/p40/p60/p80 ≈ 0.0/2.5/4.6/6.4) so the top class is the genuine ~20%
# fast-setup core the k-means cross-check isolates, not half the pool.
_CUTS = (6.5, 4.5, 1.5, -1.0)


@dataclass
class TeamFeatures:
    """Composition features of one parsed team (counts are MONS-with-≥1, not move totals,
    except the explicitly-named move counts)."""
    n_mons: int = 0
    n_recovery: int = 0        # mons with a reliable recovery move
    n_status: int = 0          # mons with a status-spreading move
    n_setup: int = 0           # mons with a stat-boosting move
    n_attackers: int = 0       # mons with ≥3 damaging moves
    n_fast: int = 0            # mons with base speed ≥ FAST_BASE_SPEED
    n_choice: int = 0          # Choice Band holders (gen3's only choice item)
    n_boom: int = 0            # mons with Explosion/Self-Destruct
    n_spikes: int = 0          # Spikes setters
    n_phaze: int = 0           # Roar/Whirlwind users
    n_wish: int = 0
    n_screens: int = 0
    n_trappers: int = 0        # trap ability (Arena Trap/Magnet Pull/Shadow Tag) or trap move
    has_spin: bool = False
    has_ghost: bool = False    # spin-blocker present
    has_sand: bool = False     # Sand Stream / sandstorm
    has_rain: bool = False     # Drizzle / rain dance
    mean_speed: float = 0.0
    mean_bulk: float = 0.0     # mean (hp+def+spd)/3
    species: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["mean_speed"] = round(self.mean_speed, 1)
        d["mean_bulk"] = round(self.mean_bulk, 1)
        return d


def team_sha(team_str: str) -> str:
    """The pool-wide team fingerprint: ``sha1(team_str.strip())[:10]``.

    STRIP-normalized because ``TeamLoader`` strips every file it loads — the artifact keys are
    shas of what the samplers actually draw. NOTE the convention gap vs MatchupSpec ``pin_sha``,
    which fingerprints the pin file's RAW bytes (recorded in live runs — immutable): a pin file
    with a trailing newline has ``pin_sha != team_sha``. To JOIN a pin to this artifact, hash the
    STRIPPED text (or use ``lookup_team``, which normalizes for you)."""
    return hashlib.sha1(team_str.strip().encode()).hexdigest()[:10]


def lookup_team(team_str: str, path: str = ARTIFACT_PATH) -> "dict | None":
    """The artifact record for one team export (normalizing join — see ``team_sha``)."""
    return load_team_archetypes(path)["teams"].get(team_sha(team_str))


def extract_features(team_str: str) -> TeamFeatures:
    """Parse one Showdown export and compute its composition features (pure; fail-loud on an
    unparseable team — a silent skip would mislabel the pool)."""
    mons = Teambuilder.parse_showdown_team(team_str)
    if not mons:
        raise ValueError("unparseable team (no mons)")
    f = TeamFeatures(n_mons=len(mons))
    speeds, bulks = [], []
    for mon in mons:
        sid = to_id_str(mon.species or mon.nickname or "")
        sp = g3species.get(sid)
        f.species.append(sid)
        if sp is not None:
            speeds.append(sp.base_stats["spe"])
            bulks.append((sp.base_stats["hp"] + sp.base_stats["def"] + sp.base_stats["spd"]) / 3)
            if sp.base_stats["spe"] >= FAST_BASE_SPEED:
                f.n_fast += 1
            if "GHOST" in sp.types:
                f.has_ghost = True
        mids = {to_id_str(m) for m in mon.moves}
        ability = to_id_str(mon.ability or "")
        item = to_id_str(mon.item or "")
        n_damaging = sum(1 for m in mids if g3moves.is_damaging(m))
        f.n_recovery += bool(mids & RECOVERY_MOVES)
        f.n_status += bool(mids & STATUS_MOVES)
        f.n_setup += bool(mids & SETUP_MOVES)
        f.n_attackers += n_damaging >= 3
        f.n_boom += bool(mids & BOOM_MOVES)
        f.n_spikes += "spikes" in mids
        f.n_phaze += bool(mids & PHAZE_MOVES)
        f.n_wish += "wish" in mids
        f.n_screens += bool(mids & SCREEN_MOVES)
        f.n_choice += item == "choiceband"
        f.n_trappers += (ability in TRAP_ABILITIES) or bool(mids & TRAP_MOVES)
        f.has_spin = f.has_spin or "rapidspin" in mids
        f.has_sand = f.has_sand or (ability in WEATHER_SAND) or ("sandstorm" in mids)
        f.has_rain = f.has_rain or (ability == "drizzle") or ("raindance" in mids)
    f.mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    f.mean_bulk = sum(bulks) / len(bulks) if bulks else 0.0
    return f


def pace_score(f: TeamFeatures) -> float:
    """The continuous offense(+) / stall(−) rubric (see ``_W``)."""
    return (_W["setup"] * f.n_setup + _W["attacker"] * f.n_attackers + _W["fast"] * f.n_fast
            + _W["boom"] * f.n_boom + _W["choice"] * f.n_choice
            + _W["recovery"] * f.n_recovery + _W["status"] * f.n_status
            + _W["spikes"] * f.n_spikes + _W["phaze"] * f.n_phaze
            + _W["screen"] * f.n_screens)


def classify(f: TeamFeatures) -> str:
    """pace_score → the coarse 5-way class (cut points in ``_CUTS``, descending)."""
    s = pace_score(f)
    for cut, cls in zip(_CUTS, ARCHETYPE_CLASSES):
        if s >= cut:
            return cls
    return ARCHETYPE_CLASSES[-1]


def style_tags(f: TeamFeatures) -> list:
    """Boolean style markers — the league-targeting axis (orthogonal to the pace class)."""
    tags = []
    if f.has_sand:
        tags.append("sand")
    if f.has_rain:
        tags.append("rain")
    if f.n_spikes:
        tags.append("spikes")
    if f.has_spin:
        tags.append("spin")
    if f.has_ghost and f.n_spikes:
        tags.append("spinblock")
    if f.n_phaze:
        tags.append("phaze")
    if f.n_trappers:
        tags.append("trap")
    if f.n_trappers >= 2:
        tags.append("trap_core")   # the double-trap (Mag+Dug-class) exploiter shortlist
    if f.n_status >= 2:
        tags.append("status_heavy")
    if f.n_wish:
        tags.append("wish")
    if f.n_screens:
        tags.append("screens")
    if f.n_boom >= 2:
        tags.append("boom")
    if f.n_choice:
        tags.append("choice")
    if f.n_setup >= 3:
        tags.append("setup_heavy")
    return tags


def classify_team(team_str: str) -> dict:
    """The full per-team record (the artifact's value shape)."""
    f = extract_features(team_str)
    return {
        "archetype": classify(f),
        "pace_score": round(pace_score(f), 2),
        "tags": style_tags(f),
        "species": f.species,
        "features": f.to_dict(),
    }


_ARCHETYPES_CACHE: "dict | None" = None


def load_team_archetypes(path: str = ARTIFACT_PATH) -> dict:
    """The committed artifact, lazy singleton (facade convention: raise if missing/empty)."""
    global _ARCHETYPES_CACHE
    if _ARCHETYPES_CACHE is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found — generate it with `python -m agents.training.team_archetypes`")
        with open(path) as fh:
            data = json.load(fh)
        if not data.get("teams"):
            raise ValueError(f"{path} has no teams")
        _ARCHETYPES_CACHE = data
    return _ARCHETYPES_CACHE


# ── the CLI: derive + validate + write ────────────────────────────────────────────

def _kmeans_crosstab(records: list, k: int = 5, seed: int = 0):
    """Unsupervised sanity check: k-means over the numeric feature vectors vs the rule classes.
    Agreement is a SMELL TEST (clusters have no names), printed as a cross-tab."""
    import numpy as np
    from sklearn.cluster import KMeans
    keys = ["n_recovery", "n_status", "n_setup", "n_attackers", "n_fast", "n_choice", "n_boom",
            "n_spikes", "n_phaze", "n_wish", "n_screens", "mean_speed", "mean_bulk"]
    X = np.array([[r["features"][key] for key in keys] for r in records], dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    tab: dict = {}
    for r, c in zip(records, labels):
        tab.setdefault(r["archetype"], [0] * k)[int(c)] += 1
    return tab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=ARTIFACT_PATH)
    ap.add_argument("--kmeans-k", type=int, default=5)
    args = ap.parse_args()

    from utils.git import get_git_hash
    from utils.team_loader import TeamLoader
    from datetime import datetime, timezone

    loader = TeamLoader()
    sample = loader.get_sample_teams()
    sample_shas = {team_sha(t) for t in sample}
    teams = loader.get_all_teams()
    out, dist = {}, {c: 0 for c in ARCHETYPE_CLASSES}
    for t in teams:
        rec = classify_team(t)
        rec["is_sample"] = team_sha(t) in sample_shas
        out[team_sha(t)] = rec
        dist[rec["archetype"]] += 1

    records = list(out.values())
    artifact = {
        "meta": {
            "n_teams": len(out),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_hash": get_git_hash(),
            "classes": list(ARCHETYPE_CLASSES),
            "distribution": dist,
        },
        "teams": out,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1)

    print(f"wrote {args.out}: {len(out)} teams")
    print("class distribution:", {c: dist[c] for c in ARCHETYPE_CLASSES})
    tags_count: dict = {}
    for r in records:
        for tg in r["tags"]:
            tags_count[tg] = tags_count.get(tg, 0) + 1
    print("tag counts:", dict(sorted(tags_count.items(), key=lambda kv: -kv[1])))
    print("\nk-means cross-tab (rows=rule class, cols=cluster):")
    for cls, row in _kmeans_crosstab(records, k=args.kmeans_k).items():
        print(f"  {cls:>14}: {row}")
    trap = [(sha, r) for sha, r in out.items() if "trap_core" in r["tags"]]
    print(f"\nTRAP_CORE teams (n_trappers >= 2; {len(trap)}) — the trap-exploiter shortlist:")
    for sha, r in trap[:15]:
        trappers = [s for s in r["species"] if s in ("magneton", "dugtrio", "wobbuffet", "trapinch")]
        print(f"  {sha}  {r['archetype']:>13}  {trappers or r['species'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
