"""X37a STAGE 2 panels — forum72 and P72, drawn by the rules PREREG.md §4 registered (commit 8a2e80de) BEFORE this ran.

It plays nothing. forum72: the smogon_forum v1 independent core (166), curator := `credited` else `[post_author]`,
cap 14 per curator, ids sorted then shuffled once by random.Random(3720), taken in order; a candidate is rejected
(the next taken) if the pinned validator rejects it or it is an exact team / shares >= 4 mon sets with ONE team of
the pool, `competitive`, home72, heldout72_hpfix or an already-drawn team. P72: `ou_random_teams.js --emit 200
--seed 3721`, each packed team exported by Showdown's `Teams.export` (trailing spaces trimmed), first 72 accepted in
emit order under the same rules (+ forum72 as a reference) plus every Hidden Power playing its named type at 70.

Writes teams_{forum72,p72}/ (committed), {forum72,p72}_manifest.json, the byte copies on our side
($X37A_STATE/teams_*) and in Metamon's cache (gen3ai_forum72 / gen3ai_p72), the two configs and
x37a_stage2_registration.json. Idempotent; refuses to overwrite different bytes.

  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 python <this>
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from main.anchors import team_overlap as TO
from main.anchors.session import load_team_texts, pool_team_texts
from utils.bridge.hp_set_check import check_teams
from utils.bridge.team_validator import validate_teams_locally
from utils.paths import repo_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_stage1 as S1  # noqa: E402

HERE = S1.HERE
FORUM = S1.MEAS / "smogon_forum_teams_2026-10-18"
CAP = 14
SEED_FORUM = 3720
SEED_P72 = 3721
N_EMIT = 200
GEN = repo_path("src", "rust_sim", "harness", "ou_random_teams.js")
SHOWDOWN = repo_path("deps", "pokemon-showdown", "dist", "sim")


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def reject_reason(team: TO.Team, refs: dict) -> str | None:
    if len(team.mons) != 6:
        return "not 6 mons"
    for name, ref in refs.items():
        if ref.keys[team.key]:
            return f"exact team in {name}"
        k, lab = ref.max_shared(team)
        if k >= 4:
            return f"shares {k} mon sets with {name}:{lab}"
    return None


def hp_bad(verdict: dict) -> list:
    """Hidden Power sets that do not play the type their line names at power 70, or whose IVs the validator rewrote."""
    return [s for s in verdict.get("sets", []) if s.get("hp_move") and (
        s.get("hp_type") != s["hp_move"].split()[-1] or s.get("power") != 70
        or s.get("ivs_after_validate") != s.get("ivs_in"))]


def references() -> dict:
    pool = [TO.parse_team(t, f"pool{i:03d}") for i, t in enumerate(pool_team_texts())]
    comp = TO.parse_items(load_team_texts(S1.REF_DIRS["away"]))
    h72 = TO.parse_items(load_team_texts(S1.HOME72_DIR))
    hp = TO.parse_items(load_team_texts(S1.HPFIX_DIR))
    return {"pool": TO.Reference(pool), "competitive": TO.Reference(comp), "home72": TO.Reference(h72),
            "heldout72_hpfix": TO.Reference(hp)}


def draw_forum(refs: dict) -> list:
    man = json.loads((FORUM / "manifest.json").read_text())
    core = sorted((r for r in man["teams"] if r.get("independent_core")), key=lambda r: r["id"])
    if len(core) != 166:
        raise SystemExit(f"expected the 166-team independent core, found {len(core)}")
    for r in core:
        if hashlib.sha256((FORUM / "teams" / f"{r['id']}.txt").read_bytes()).hexdigest() != r["sha256"]:
            raise SystemExit(f"forum file differs from its v1 manifest: {r['id']}")
    order = [r["id"] for r in core]
    random.Random(SEED_FORUM).shuffle(order)
    by_id = {r["id"]: r for r in core}
    texts = {i: (FORUM / "teams" / f"{i}.txt").read_text() for i in order}
    verdicts = dict(zip(order, validate_teams_locally("gen3ou", [texts[i] for i in order])))
    per_cur: collections.Counter = collections.Counter()
    drawn, log, drawn_teams = [], [], []
    for i in order:
        if len(drawn) == 72:
            break
        r = by_id[i]
        cur = r["credited"] or f"[{r['post_author']}]"
        if per_cur[cur] >= CAP:
            log.append({"id": i, "curator": cur, "skipped": "curator cap"})
            continue
        team = TO.parse_team(texts[i], i)
        why = None if verdicts[i].get("valid") else f"validator: {verdicts[i].get('errors')}"
        why = why or reject_reason(team, {**refs, "drawn": TO.Reference(drawn_teams)})
        if why:
            log.append({"id": i, "curator": cur, "rejected": why})
            continue
        per_cur[cur] += 1
        drawn.append({"id": i, "curator": cur, "text": texts[i], "category": r["category"], "tier": r["tier"],
                      "team_name": r["team_name"], "source_url": r["source_url"], "post_date": r["post_date"],
                      "forum_sha256": r["sha256"]})
        drawn_teams.append(team)
        log.append({"id": i, "curator": cur, "accepted": len(drawn)})
    if len(drawn) != 72:
        raise SystemExit(f"forum72 draw ended with {len(drawn)} teams")
    return drawn, log, dict(per_cur)


def export_packed(packed: list) -> list:
    js = ("const {Teams}=require(process.argv[1]);const ls=JSON.parse(require('fs').readFileSync(0,'utf8'));"
          "process.stdout.write(JSON.stringify(ls.map(p=>Teams.export(Teams.unpack(p)).split('\\n')"
          ".map(l=>l.trimEnd()).join('\\n').trim()+'\\n')));")
    p = subprocess.run(["node", "-e", js, str(SHOWDOWN)], input=json.dumps(packed), capture_output=True,
                       text=True, check=True)
    return json.loads(p.stdout)


def make_p72(refs: dict, forum_teams: list) -> tuple:
    p = subprocess.run(["node", str(GEN), "--emit", str(N_EMIT), "--seed", str(SEED_P72)],
                       capture_output=True, text=True, check=True)
    packed = [t for t in p.stdout.splitlines() if t.strip()]
    if len(packed) != N_EMIT:
        raise SystemExit(f"generator returned {len(packed)} teams")
    texts = export_packed(packed)
    verdicts = validate_teams_locally("gen3ou", texts)
    hp = check_teams(texts)
    refs = {**refs, "forum72": TO.Reference(forum_teams)}
    acc, log, acc_teams = [], [], []
    for k, (t, v, h) in enumerate(zip(texts, verdicts, hp)):
        if len(acc) == 72:
            break
        pid = f"p72_{k:03d}"
        team = TO.parse_team(t, pid)
        why = None if v.get("valid") else f"validator: {v.get('errors')}"
        if not why:
            bad = hp_bad(h)
            why = f"hidden power mismatch: {bad}" if bad else None
        why = why or reject_reason(team, {**refs, "p72": TO.Reference(acc_teams)})
        if why:
            log.append({"id": pid, "rejected": why})
            continue
        acc.append({"id": pid, "emit_index": k, "packed": packed[k], "text": t})
        acc_teams.append(team)
        log.append({"id": pid, "accepted": len(acc)})
    if len(acc) != 72:
        raise SystemExit(f"P72 ended with {len(acc)} teams")
    return acc, log, hp


def install(name: str, teams: list) -> dict:
    ours = S1.STATE / f"teams_{name}"
    mm = S1.METAMON_TEAMS / f"gen3ai_{name}" / "gen3ou"
    committed = HERE / f"teams_{name}"
    out = {}
    for t in teams:
        want = sha_text(t["text"])
        for d, ext in ((committed, ".txt"), (ours, ".txt"), (mm, ".gen3ou_team")):
            f = d / f"{t['id']}{ext}"
            if f.exists():
                if hashlib.sha256(f.read_bytes()).hexdigest() != want:
                    raise SystemExit(f"existing file differs: {f}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                f.write_text(t["text"])
        out[t["id"]] = want
    for d, ext in ((committed, ".txt"), (ours, ".txt"), (mm, ".gen3ou_team")):
        if sorted(p.name for p in d.iterdir()) != sorted(i + ext for i in out):
            raise SystemExit(f"{d} holds files outside the panel")
    return out


def main() -> int:
    refs = references()
    forum, flog, per_cur = draw_forum(refs)
    forum_teams = [TO.parse_team(t["text"], t["id"]) for t in forum]
    p72, plog, hp = make_p72(refs, forum_teams)
    forum_hp = check_teams([t["text"] for t in forum])
    forum_hp_bad = [(t["id"], s) for t, h in zip(forum, forum_hp) for s in hp_bad(h)]
    reg = {"schema": "x37a_stage2_registration_v1", "rules": "PREREG.md §4 (commit 8a2e80de)", "configs": {},
           "panels": {}}
    for name, teams, log, extra in (("forum72", forum, flog, {"cap": CAP, "seed": SEED_FORUM, "per_curator": per_cur,
                                                             "hp_check_failures": forum_hp_bad}),
                                    ("p72", p72, plog, {"seed": SEED_P72, "n_emit": N_EMIT,
                                                        "generator": "src/rust_sim/harness/ou_random_teams.js"})):
        shas = install(name, teams)
        files = S1.team_files(HERE / f"teams_{name}")
        strata = S1.species_strata(files)
        read = [TO.parse_team(t["text"], t["id"]) for t in teams]
        overlap = {k: TO.overlap(read, r) for k, r in refs.items()}
        if name == "p72":
            overlap["forum72"] = TO.overlap(read, TO.Reference(forum_teams))
        man = {"schema": f"x37a_{name}_manifest_v1", **extra, "draw_log": log,
               "teams": {t["id"]: {"sha256": shas[t["id"]], **{k: v for k, v in t.items() if k not in ("text", "id")}}
                         for t in teams},
               "strata": strata, "overlap": overlap}
        S1.write_if_same(HERE / f"{name}_manifest.json", json.dumps(man, indent=1) + "\n")
        cfg = S1.config(S1.STATE / f"teams_{name}", f"gen3ai_{name}",
                        f"X37a stage 2 symmetric cell {name}: both sides pilot {name} (our byte copy; Metamon set "
                        f"gen3ai_{name}), sha256-checked against {name}_manifest.json before every unit.")
        cp = HERE / "configs" / f"anchors_x37a_{name}.json"
        S1.write_if_same(cp, json.dumps(cfg, indent=1) + "\n")
        reg["configs"][name] = {"path": f"configs/anchors_x37a_{name}.json", "sha256": S1.sha(cp)}
        reg["panels"][name] = {"n": len(teams), "content_digest": S1.content_digest(files),
                               "strata_counts": {s: len(v) for s, v in strata.items()},
                               "manifest_sha256": S1.sha(HERE / f"{name}_manifest.json")}
    S1.write_if_same(HERE / "x37a_stage2_registration.json", json.dumps(reg, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "strata_counts")} for k, v in reg["panels"].items()}, indent=1))
    print("forum per curator", per_cur, "| forum HP failures", len(forum_hp_bad))
    print("forum rejections", [x for x in flog if "rejected" in x])
    print("p72 rejections", [x for x in plog if "rejected" in x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
