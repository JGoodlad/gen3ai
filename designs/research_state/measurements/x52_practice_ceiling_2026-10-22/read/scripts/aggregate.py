"""X52's REGISTERED read — pool the queue's units into rows.jsonl + cells.json and apply AM-X52-f / g.

  PYTHONPATH=/home/goodlad/.cache/gen3ai/x52/pin_f4b25238/src python aggregate.py [--integrity-only]

COUNTING (`main.anchors.results.unit_counts`, DECLARED mode: a reverse cell is 36-vs-72 by design): status OK,
n == 20, `regime_verified_decisions`, `peer_clean`, and the summary's own team digests EQUAL the declared pair
(`session.team_set_digest` of the half's sha-checked files, of Metamon's home72), AND X51's CAPTURE check: per game
our packed team in the START command is the row's `our_team` (a team of the half) and Kakuna's is exactly one home72
team. `--integrity-only` prints the counting verdicts and NO win rate.

THE STATISTIC (AM-X52-f): per half h, Δ_h = (s3501 @5M + s3502 @5M, pooled) win rate − P*'s win rate, ties are
non-wins; 95 % interval = a TEAM-CLUSTERED percentile bootstrap (resample the half's 36 teams with replacement,
10,000 resamples, random.Random(20261022)). THE CLASSES (AM-X52-g), in order, rule 8 at 0.0005 on every bound:
INCONCLUSIVE (any primary cell > 25 % of its registered games not counted) · NO FAMILIARITY (UB_pr < +0.040) ·
TRANSFERS (LB_pr > 0 and LB_un > 0) · NO TRANSFER (LB_pr > 0 and UB_un < +0.030) · FAMILIARITY ONLY (LB_pr > 0) ·
else UNDECIDED. A comparison whose bound lies within 0.0005 of its boundary decides nothing: the class it would
decide is UNDECIDED (rule 8).
DESCRIPTIVE: every cell's Wilson; per seed Δ; s3501 @1M − P* (practised, seeds 0-9) and C0 s3501 final − P*
(unpractised, seeds 0-19) on the SAME units, clustered the same way; twins (unit seed, half, index) checked to carry
the same our_team.
"""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
_spec = importlib.util.spec_from_file_location("x52q", HERE / "x52q.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

from main.anchors import team_overlap as TO  # noqa: E402
from main.anchors.results import unit_counts, wilson  # noqa: E402
from main.anchors.session import team_set_digest  # noqa: E402

GAMES = 20
MARGIN = 0.0005
B = 10_000
BOOT_SEED = 20261022
STAMP = ("opponent", "opponent_version", "opponent_commit", "our_regime", "their_regime", "regime_matched",
         "teamset", "our_team_count", "their_team_count", "battle_format", "model_zip", "model_step",
         "model_loader", "forfeit_turn_limit", "showdown_pin", "server_impl", "server_version", "our_side",
         "our_transport", "mirrored_pairs")
SHOWDOWN = q.PIN / "deps" / "pokemon-showdown" / "dist" / "sim"
HALVES = ("pr", "un")


def won(r) -> int:
    return 1 if r["result"] == "win" else 0


def set_index(d: Path) -> dict:
    idx = defaultdict(list)
    for f in sorted(d.iterdir()):
        if f.suffix in (".txt", ".gen3ou_team"):
            idx[TO.parse_team(f.read_text(errors="ignore"), f.stem).key].append(f.stem)
    return idx


def export(packed: list) -> list:
    js = ("const {Teams}=require(process.argv[1]);const ls=JSON.parse(require('fs').readFileSync(0,'utf8'));"
          "process.stdout.write(JSON.stringify(ls.map(p=>Teams.export(Teams.unpack(p)))));")
    p = subprocess.run(["node", "-e", js, str(SHOWDOWN)], input=json.dumps(packed), capture_output=True, text=True,
                       check=True)
    return json.loads(p.stdout)


def capture_check(cell: str, udir: Path, rows: list, idx: dict) -> tuple:
    iours, itheirs = idx[q.CELLS[cell]["config"]], idx["home72"]
    caps = {}
    for f in sorted((udir / "capture").glob("*.json")):
        c = json.loads(f.read_text())
        caps[c["tag"]] = json.loads(c["commands"][0].split(" ", 1)[1])
    reasons, their = [], {}
    if len(caps) != GAMES:
        reasons.append(f"capture holds {len(caps)} battles, not {GAMES}")
    packed, keys = [], []
    for tag, st in caps.items():
        for slot in ("p1", "p2"):
            packed.append(st[slot]["team"])
            keys.append((tag, st[slot]["name"]))
    texts = dict(zip(keys, export(packed))) if packed else {}
    for r in rows:
        st = caps.get(r.get("battle_tag"))
        if st is None:
            reasons.append(f"no capture for {r.get('battle_tag')}")
            continue
        names = {st["p1"]["name"], st["p2"]["name"]}
        our_name = next((n for n in names if n.lower().startswith("gen3aianchor")), None)
        their_name = next((n for n in names if n != our_name), None)
        k_ours = TO.parse_team(texts[(r["battle_tag"], our_name)]).key
        k_their = TO.parse_team(texts[(r["battle_tag"], their_name)]).key
        if Path(r["our_team"]).stem not in iours.get(k_ours, []):
            reasons.append(f"{r['battle_tag']}: our packed team is not {r['our_team']}")
        m = itheirs.get(k_their, [])
        if not m:
            reasons.append(f"{r['battle_tag']}: Kakuna's team is not a home72 team")
        else:
            their[r["battle_tag"]] = m[0] if len(m) == 1 else "|".join(sorted(m))
    return reasons, their


def collect() -> tuple:
    home = q.METAMON_HOME72
    halves = {h: q.STATE / f"teams_{h}" for h in ("practised", "unpractised")}
    declared = {h: (team_set_digest(d), team_set_digest(home)) for h, d in halves.items()}
    idx = {h: set_index(d) for h, d in halves.items()}
    idx["home72"] = set_index(home)
    rows, units, skipped = [], [], []
    for unit, cell, seed in q.units(list(q.CELLS)):
        udir = q.STATE / "cells" / unit
        sp = udir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        uc = unit_counts(s, games=GAMES, declared_digests=declared[q.CELLS[cell]["config"]])
        reasons = list(uc.reasons)
        urows = [json.loads(line) for line in open(udir / "games.jsonl")] if (udir / "games.jsonl").exists() else []
        cap_reasons, their = capture_check(cell, udir, urows, idx)
        reasons += cap_reasons
        rec = {"unit": unit, "cell": cell, "seed": seed, "counts": not reasons, "reasons": reasons,
               "status": s.get("status"), "n": s.get("n")}
        units.append(rec)
        if reasons:
            skipped.append(rec)
            continue
        for r in urows:
            rows.append({**r, "read_cell": cell, "unit": unit, "unit_seed": seed,
                         "their_team": their.get(r.get("battle_tag")), "our_team_stem": Path(r["our_team"]).stem})
    return rows, units, skipped, {h: list(d) for h, d in declared.items()}


def cell_stats(rs: list) -> dict:
    n = len(rs)
    w = sum(won(r) for r in rs)
    t = sum(r["result"] == "tie" for r in rs)
    p, lo, hi = wilson(w, n) if n else (None, None, None)
    by_half = {h: round(sum(won(r) for r in rs if r["half"] == h) / max(1, sum(r["half"] == h for r in rs)), 4)
               for h in ("ours_challenge", "peer_challenge")}
    turns = sorted(r["turns"] for r in rs if r.get("turns") is not None)
    return {"n": n, "wins": w, "losses": n - w - t, "ties": t, "win_rate": p, "wilson95": [lo, hi],
            "by_half": by_half, "hit_forfeit_limit": sum(bool(r.get("hit_forfeit_limit")) for r in rs),
            "median_turns": turns[len(turns) // 2] if turns else None,
            "distinct_our_teams": len({r.get("our_team") for r in rs}),
            "distinct_their_teams": len({r.get("their_team") for r in rs if r.get("their_team")}),
            "regime_stamps": dict(Counter(json.dumps({k: r.get(k) for k in STAMP}, sort_keys=True) for r in rs))}


def clustered_delta(treat: list, ctrl: list, teams: list) -> dict:
    """Δ = win rate(treat) − win rate(ctrl), pooled over games; 95 % team-clustered percentile bootstrap."""
    def per_team(rs):
        d = defaultdict(lambda: [0, 0])
        for r in rs:
            d[r["our_team_stem"]][0] += won(r)
            d[r["our_team_stem"]][1] += 1
        return d
    if not treat or not ctrl:
        return {"delta": None, "ci95": [None, None], "n_treat": len(treat), "n_ctrl": len(ctrl)}
    T, C = per_team(treat), per_team(ctrl)

    def stat(sample):
        tw = sum(T[t][0] for t in sample)
        tn = sum(T[t][1] for t in sample)
        cw = sum(C[t][0] for t in sample)
        cn = sum(C[t][1] for t in sample)
        return (tw / tn - cw / cn) if tn and cn else None
    point = stat(teams)
    rng = random.Random(BOOT_SEED)
    boots = sorted(v for v in (stat([rng.choice(teams) for _ in teams]) for _ in range(B)) if v is not None)
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    return {"delta": point, "ci95": [lo, hi], "n_treat": len(treat), "n_ctrl": len(ctrl), "n_boot": len(boots),
            "teams_seen": len(set(T) | set(C)), "treat_rate": sum(map(won, treat)) / max(1, len(treat)),
            "ctrl_rate": sum(map(won, ctrl)) / max(1, len(ctrl))}


def cmp(x: float, op: str, bar: float):
    """Tri-state: True / False, or None when |x − bar| < MARGIN (rule 8)."""
    if abs(x - bar) < MARGIN:
        return None
    return x > bar if op == ">" else x < bar


def classify(dpr: dict, dun: dict, uncounted: dict) -> dict:
    if any(v > 0.25 for v in uncounted.values()):
        return {"class": "INCONCLUSIVE", "why": f"uncounted share {uncounted}"}
    (lpr, upr), (lun, uun) = dpr["ci95"], dun["ci95"]
    steps = [("NO FAMILIARITY", [cmp(upr, "<", 0.040)]),
             ("TRANSFERS", [cmp(lpr, ">", 0.0), cmp(lun, ">", 0.0)]),
             ("NO TRANSFER", [cmp(lpr, ">", 0.0), cmp(uun, "<", 0.030)]),
             ("FAMILIARITY ONLY", [cmp(lpr, ">", 0.0)])]
    for name, conds in steps:
        if any(c is False for c in conds):
            continue
        if all(c is True for c in conds):
            return {"class": name, "why": "rule held"}
        return {"class": "UNDECIDED", "why": f"rule 8: a bound of {name} lies within {MARGIN} of its boundary"}
    return {"class": "UNDECIDED", "why": "no rule held"}


def twins_check(rows: list) -> dict:
    by = defaultdict(set)
    for r in rows:
        h = r["read_cell"].rsplit("_", 1)[1]
        by[(h, r["unit_seed"], r["half"], r["index"])].add(r["our_team"])
    return {"keys": len(by), "unequal": sum(len(v) > 1 for v in by.values())}


def main(argv) -> int:
    rows, units, skipped, declared = collect()
    if "--integrity-only" in argv:
        print(json.dumps({"units": len(units), "counted": sum(u["counts"] for u in units), "skipped": skipped,
                          "declared_digests": declared}, indent=1))
        return 0
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    cells = {c: cell_stats(by[c]) for c in q.CELLS}
    regs = {c: len(q.CELLS[c]["seeds"]) * GAMES for c in q.CELLS}
    uncounted = {c: round(1 - len(by[c]) / regs[c], 4) for c in q.CELLS if q.PRIORITY[c] == 0}
    halves_teams = {}
    man = q._manifests()
    for h, key in q.HALF_KEY.items():
        halves_teams[h] = sorted(man[key])
    a = {}
    for h in HALVES:
        teams = halves_teams[h]
        pooled = by[f"x52a_{h}"] + by[f"x52b_{h}"]
        a[f"delta_{h}"] = clustered_delta(pooled, by[f"pstar_{h}"], teams)
        a[f"delta_{h}_s3501"] = clustered_delta(by[f"x52a_{h}"], by[f"pstar_{h}"], teams)
        a[f"delta_{h}_s3502"] = clustered_delta(by[f"x52b_{h}"], by[f"pstar_{h}"], teams)
    seeds1m = set(q.CELLS["x52a1m_pr"]["seeds"])
    seedsc0 = set(q.CELLS["c0a_un"]["seeds"])
    a["desc_1m_minus_pstar_pr"] = clustered_delta(by["x52a1m_pr"], [r for r in by["pstar_pr"]
                                                                     if r["unit_seed"] in seeds1m],
                                                  halves_teams["pr"])
    a["desc_5m_s3501_minus_pstar_pr_same_units_as_1m"] = clustered_delta(
        [r for r in by["x52a_pr"] if r["unit_seed"] in seeds1m],
        [r for r in by["pstar_pr"] if r["unit_seed"] in seeds1m], halves_teams["pr"])
    a["desc_c0final_minus_pstar_un"] = clustered_delta(by["c0a_un"], [r for r in by["pstar_un"]
                                                                       if r["unit_seed"] in seedsc0],
                                                       halves_teams["un"])
    a["classification"] = classify(a["delta_pr"], a["delta_un"], uncounted)
    a["uncounted_share_primary"] = uncounted
    a["twins"] = twins_check(rows)
    out = {"schema": "x52_read_v1", "cells": cells, "analysis": a, "units": units, "skipped_units": skipped,
           "declared_digests": declared, "models": {k: [str(v[0]), v[1]] for k, v in q.MODELS.items()},
           "config_sha256": q.CONFIG_SHA, "pin": str(q.PIN)}
    (OUT / "cells.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in a.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
