"""X37a STAGE 1 setup — the frozen inputs of the evaluation matrix's core, written BEFORE any stage-1 game (2026-10-18).

It plays nothing. It:
  1. re-derives the stratum of every heldout72_hpfix team (best SPECIES overlap with one sample team, the
     `ladder_weights_v1` rule: CORE = 6, NEAR = 4-5, FAR <= 3) and REFUSES unless it equals the frozen
     `ladder_weights_v1.json` membership and the panel digest there;
  2. byte-copies the team sets our side pilots into the read's state dir (home72 from `n0_endofrun_2026-09-27`,
     heldout72_hpfix from its measurement dir), sha256-checked against their committed manifests;
  3. installs the three per-stratum Metamon sub-sets `gen3ai_x37a_hl72hp_{core,near,far}` as byte copies of
     `gen3ai_heldout72_hpfix` (sha256-checked against the hpfix manifest);
  4. writes the cell configs (the stock anchors config shape; `team_sets` / `our_team_sources` per cell);
  5. writes `x37a_stage1_registration.json`: every input's digest, the strata members, the cell list and seeds.

Usage (CPU only, rule 5b):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 \
    python designs/research_state/measurements/x37a_eval_matrix_2026-10-18/scripts/setup_stage1.py
Re-running it is idempotent: an existing file with the right bytes is kept; a wrong one is a refusal.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from main.anchors import team_overlap as TO
from utils.paths import repo_path

MEAS = repo_path("designs", "research_state", "measurements")
HERE = MEAS / "x37a_eval_matrix_2026-10-18"
STATE = Path(os.environ.get("X37A_STATE", os.path.expanduser("~/dev/gen3ai-reads/x37a_2026-10-18")))
METAMON_TEAMS = Path(os.path.expanduser("~/dev/metamon/cache/teams"))
LW = MEAS / "ladder_weights_2026-10-18" / "ladder_weights_v1.json"
HPFIX_DIR = MEAS / "heldout72_hpfix_kakuna_2026-10-18" / "teams_heldout72_hpfix"
HPFIX_MANIFEST = MEAS / "heldout72_hpfix_kakuna_2026-10-18" / "heldout72_hpfix_manifest.json"
HOME72_DIR = MEAS / "n0_endofrun_2026-09-27" / "teams_home72"
HOME72_MANIFEST = MEAS / "x35_breakers_stage_2026-10-17" / "home72_manifest.json"
REF_DIRS = {"home72": HOME72_DIR, "away": METAMON_TEAMS / "competitive" / "gen3ou"}
STRATA = ("CORE", "NEAR", "FAR")

METAMON = {
    "checkout": "/home/goodlad/dev/metamon",
    "python": "/home/goodlad/miniconda3/envs/metamon/bin/python",
    "cache_dir": "/home/goodlad/dev/metamon/cache",
    "agents": {"SmallRL": {"checkpoint": 40, "params": 13938567},
               "SyntheticRLV2": {"checkpoint": 48, "params": 200894911},
               "Kakuna": {"checkpoint": 34, "params": 142832563}},
    "reference_team_dirs": {
        "competitive": "/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou",
        "hl_05_26": "/home/goodlad/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou"},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def team_files(d: Path) -> list:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix in (".txt", ".gen3ou_team"))


def lw_digest(files: list) -> str:
    """`ladder_weights.py`'s panel digest (name + sha256 per file, in name order)."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def content_digest(files: list) -> str:
    """`gen3_team_set_digest_v1`: sha256 over the SORTED sha256(text.strip()) of every team file — equal for
    byte copies under different extensions (our `.txt`, Metamon's `.gen3ou_team`)."""
    hs = sorted(hashlib.sha256(f.read_text().strip().encode()).hexdigest() for f in files)
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def copy_checked(src: Path, dst: Path, want: str) -> None:
    if sha(src) != want:
        raise SystemExit(f"source sha256 mismatch: {src}")
    if dst.exists():
        if sha(dst) != want:
            raise SystemExit(f"existing copy differs: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha(dst) != want:
        raise SystemExit(f"copy sha256 mismatch: {dst}")


def stratum_of(k: int) -> str:
    return "CORE" if k == 6 else ("NEAR" if k >= 4 else "FAR")


def species_strata(files: list) -> dict:
    ref = [TO.parse_team(f.read_text(errors="ignore"), f.name)
           for d in REF_DIRS.values() for f in team_files(d)]
    ref_species = list({t.species6 for t in ref if len(t.mons) == 6})
    out = {s: [] for s in STRATA}
    for f in files:
        t = TO.parse_team(f.read_text(errors="ignore"), f.name)
        out[stratum_of(max(len(t.species6 & r) for r in ref_species))].append(f.stem)
    return out


def config(our_home: Path, metamon_home: str, readme: str, metamon_away: str = "competitive") -> dict:
    return {
        "_README": [readme],
        "showdown": {"node": "node", "port_range": [9500, 9599]},
        "opponents": {
            "metamon": {**METAMON, "team_sets": {"home": metamon_home, "away": metamon_away}},
            "foulplay": {"checkout": "/home/goodlad/dev/foul-play",
                         "python": "/home/goodlad/dev/foul-play/.conda/bin/python",
                         "team_dirs": {"home": str(our_home),
                                       "away": "/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou"}},
        },
        "our_team_sources": {
            "home": {"kind": "dir", "path": str(our_home)},
            "away": {"kind": "dir", "path": "/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou"},
        },
    }


def write_if_same(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise SystemExit(f"refusing to overwrite a different {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    lw = json.loads(LW.read_text())
    panel = lw["panels"]["heldout72_hpfix"]
    hp_files = team_files(HPFIX_DIR)
    if lw_digest(hp_files) != panel["digest"]:
        raise SystemExit("heldout72_hpfix's committed teams do not match ladder_weights_v1's panel digest")
    strata = species_strata(hp_files)
    if {s: sorted(v) for s, v in strata.items()} != {s: sorted(panel["strata_members"][s]) for s in STRATA}:
        raise SystemExit("re-derived strata differ from ladder_weights_v1's frozen membership")
    hist = lw["best_species_overlap_hist"]
    n_lad = sum(hist.values())
    counts = {"CORE": hist["6"], "NEAR": hist["4"] + hist["5"],
              "FAR": hist["0"] + hist["1"] + hist["2"] + hist["3"]}

    # (2) our side's copies
    hpm = json.loads(HPFIX_MANIFEST.read_text())["teams"]
    h72 = json.loads(HOME72_MANIFEST.read_text())["our_teams"]
    our_hp = STATE / "teams_heldout72_hpfix"
    our_h72 = STATE / "teams_home72"
    for stem, rec in hpm.items():
        copy_checked(HPFIX_DIR / f"{stem}.txt", our_hp / f"{stem}.txt", rec["sha256"])
    for stem, want in h72.items():
        copy_checked(HOME72_DIR / f"{stem}.txt", our_h72 / f"{stem}.txt", want)
    if len(team_files(our_hp)) != 72 or len(team_files(our_h72)) != 72:
        raise SystemExit("our-side copies are not 72 + 72 files")

    # (3) Metamon per-stratum sub-sets (byte copies of gen3ai_heldout72_hpfix)
    src_set = METAMON_TEAMS / "gen3ai_heldout72_hpfix" / "gen3ou"
    sub_sets = {}
    for s in STRATA:
        name = f"gen3ai_x37a_hl72hp_{s.lower()}"
        d = METAMON_TEAMS / name / "gen3ou"
        for stem in strata[s]:
            copy_checked(src_set / f"{stem}.gen3ou_team", d / f"{stem}.gen3ou_team", hpm[stem]["sha256"])
        present = sorted(p.name for p in d.iterdir())
        if present != sorted(f"{stem}.gen3ou_team" for stem in strata[s]):
            raise SystemExit(f"{d} holds files outside stratum {s}")
        sub_sets[s] = {"metamon_set": name, "n": len(strata[s]), "members": sorted(strata[s]),
                       "content_digest": content_digest(team_files(d))}

    # (4) configs
    cfgs = {}
    for s in STRATA:
        cfgs[f"deploy_{s.lower()}"] = config(
            our_h72, sub_sets[s]["metamon_set"],
            f"X37a stage 1 DEPLOYMENT cell, stratum {s}: our side pilots home72 (byte copy of "
            f"n0_endofrun_2026-09-27/teams_home72), Kakuna pilots the {sub_sets[s]['n']} heldout72_hpfix {s} teams "
            f"(Metamon set {sub_sets[s]['metamon_set']}). A DECLARED asymmetric cell (registration).")
    cfgs["reverse"] = config(
        our_hp, "gen3ai_home72",
        "X37a stage 1 REVERSE cell: our side pilots heldout72_hpfix (byte copy), Kakuna pilots home72 "
        "(Metamon set gen3ai_home72). A DECLARED asymmetric cell (registration).")
    cfgs["kcal"] = config(
        our_hp, "gen3ai_heldout72_hpfix",
        "X37a stage 1 strength calibration K(P, H): Kakuna (our side, --our-teamset away = gen3ai_home72) vs "
        "Kakuna (--teamset home = gen3ai_heldout72_hpfix). Needs B-D9. A DECLARED asymmetric cell.",
        metamon_away="gen3ai_home72")
    cfg_sha = {}
    for name, c in cfgs.items():
        p = HERE / "configs" / f"anchors_x37a_{name}.json"
        write_if_same(p, json.dumps(c, indent=1) + "\n")
        cfg_sha[name] = sha(p)

    home72_digest = content_digest(team_files(our_h72))
    hpfix_digest = content_digest(team_files(our_hp))
    reg = {
        "schema": "x37a_stage1_registration_v1",
        "ladder_weights": {"version": "ladder_weights_v1", "file": str(LW.relative_to(MEAS.parents[2])),
                           "counts": counts, "n_ladder_teams": n_lad,
                           "weights_exact": {s: f"{counts[s]}/{n_lad}" for s in STRATA},
                           "weights_float": {s: counts[s] / n_lad for s in STRATA},
                           "ladder_proxy_digest": lw["ladder_proxy"]["digest"],
                           "sample_reference_digest": lw["sample_reference"]["digest"],
                           "matcher": lw["matcher"]},
        "heldout72_hpfix": {"lw_panel_digest": panel["digest"], "content_digest": hpfix_digest,
                            "strata_counts": {s: len(strata[s]) for s in STRATA}},
        "home72": {"content_digest": home72_digest},
        "metamon_sub_sets": sub_sets,
        "configs": {n: {"path": f"configs/anchors_x37a_{n}.json", "sha256": h} for n, h in cfg_sha.items()},
        # (ours, theirs) content digests every DECLARED asymmetric cell must show (B-D6's unit_counts)
        "declared_digests": {
            **{f"deploy_{s.lower()}": [home72_digest, sub_sets[s]["content_digest"]] for s in STRATA},
            "reverse": [hpfix_digest, home72_digest],
            "kcal": [home72_digest, hpfix_digest],
        },
    }
    write_if_same(HERE / "x37a_stage1_registration.json", json.dumps(reg, indent=1) + "\n")
    print(json.dumps({"strata": reg["heldout72_hpfix"]["strata_counts"], "weights": reg["ladder_weights"]["weights_exact"],
                      "configs": cfg_sha}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
