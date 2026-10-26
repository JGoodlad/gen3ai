"""REVIEW E (adversarial review of the X59-S registration): the CPU checks behind REVIEW_E.md. Seeded, deterministic.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --max-s 600 8 \
        $PY designs/research_state/measurements/x59s_matched_students_2026-10-25/review_e_checks.py [pool|rows|ss|all]

* ``pool``: the plan §3.1 panel-exclusion rule (``main.human_data.textpass.PanelMatcher``, >= 3 fits of one team)
  applied to the 719 S-S pool teams against the SCORED panels only (H, F, P72, and the retired heldout72 whose sets
  stay excluded; ``forum216b`` is not read here), under three reveal profiles: every fact (4 moves, item, ability),
  two seeded moves a mon, one seeded move a mon.
* ``rows``: on 5 fixed TRAIN shards of ``hr_spec_v1.r1`` and their S-P labels (read-only): the consistent set's size,
  what the restriction removes and where P*'s mass moves, and P*'s agreement with the human by decisions-to-the-end
  for the quitter, the forfeit winner and both seats of KO games (the quitter-tail control).
* ``ss``: 2,100 seeded S-S rows re-scored by P* (CPU, 2 threads): per seat, P*'s entropy and its NLL of the recorded
  label (a seat-1 observation defect would split them).

Writes ``results/review_e_checks.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARCH = Path.home() / "gen3ai_archive" / "x59s"
ROWS = ARCH / "hr_spec_v1.r1" / "rows"
LABELS = ARCH / "hr_spec_v1.r1" / "labels_pstar"
SS = ARCH / "ss_rows"
SCORED = ("heldout72", "heldout72_hpfix", "forum72", "p72")
SHARDS = (0, 20, 40, 60, 80)
SW = np.arange(11) <= 5                        # actions 0-5 are switches (rowset._row_profile's rule)
OUT = Path(__file__).resolve().parent / "results" / "review_e_checks.json"


def _softmax(lg: np.ndarray, m: np.ndarray) -> np.ndarray:
    x = np.where(m, lg, -np.inf)
    x = x - x.max(1, keepdims=True)
    e = np.where(m, np.exp(x), 0.0)
    return e / e.sum(1, keepdims=True)


def pool() -> dict:
    from main.anchors.team_overlap import parse_team
    from main.human_data.textpass import NEAR, PanelMatcher
    from utils.team_sources import team_list

    pm = PanelMatcher(panels=SCORED)
    teams = [parse_team(x, f"pool{i}") for i, x in enumerate(team_list("pool"))]
    rng = np.random.default_rng(20261025)
    out = {"pool_teams": len(teams), "panels": list(SCORED), "near": NEAR}
    for mode in ("full", "two_moves", "one_move"):
        near_any, per = 0, {p: 0 for p in SCORED}
        for t in teams:
            side = {}
            for ms in t.mons:
                mv = list(ms.moves)
                if mode == "full":
                    side[ms.species] = (mv, ms.item or None, ms.ability or None)
                else:
                    k = 2 if mode == "two_moves" else 1
                    side[ms.species] = (sorted(rng.choice(mv, min(k, len(mv)), replace=False).tolist()), None, None)
            f = pm.side_max_fits(side)
            hit = [p for p, n in f.items() if n >= NEAR]
            near_any += bool(hit)
            for p in hit:
                per[p] += 1
        share = near_any / len(teams)
        out[mode] = {"teams_near_any": near_any, "share": round(share, 4),
                     "game_with_a_near_side": round(1 - (1 - share) ** 2, 4), "per_panel": per}
    return out


def rows() -> dict:
    man = json.loads((ROWS / "rows_manifest.json").read_text())
    F = ("mask", "act", "hidden", "cons", "quit_dist", "end", "bkey", "seat", "k", "rating")
    acc = {f: [] for f in F}
    lp, lg = [], []
    for i in SHARDS:
        name = man["train_shards"][i]["name"]
        z = np.load(ROWS / f"{name}.npz")
        for f in F:
            acc[f].append(z[f])
        lz = np.load(LABELS / f"{name}.npz")
        lp.append(lz["act_p"])
        lg.append(lz["logits"])
    H = {f: np.concatenate(v) for f, v in acc.items()}
    ap, LG = np.concatenate(lp).astype(int), np.concatenate(lg).astype(np.float64)
    lab = ~H["hidden"]
    li = np.nonzero(lab)[0]
    a = H["act"].astype(int)
    M, C = H["mask"].astype(bool), H["cons"].astype(bool)
    pf, pc = _softmax(LG, M)[li], _softmax(LG, C)[li]
    Ml, Cl = M[li], C[li]
    ncons = Cl.sum(1)
    res: dict = {"rows": int(len(a)), "labelled": int(len(li)), "shards": list(SHARDS)}
    res["consistent"] = {
        "legal_mean": round(float(Ml.sum(1).mean()), 3), "consistent_mean": round(float(ncons.mean()), 3),
        "consistent_equals_legal": round(float((ncons == Ml.sum(1)).mean()), 4),
        "size1_share": round(float((ncons == 1).mean()), 4),
        "legal_moves": round(float(Ml[:, ~SW].sum(1).mean()), 3),
        "consistent_moves": round(float(Cl[:, ~SW].sum(1).mean()), 3),
        "pstar_mass_removed_from_moves": round(float((pf * (Ml & ~Cl))[:, ~SW].sum(1).mean()), 4),
        "pstar_mass_removed_from_switches": round(float((pf * (Ml & ~Cl))[:, SW].sum(1).mean()), 4),
        "pstar_top1_outside_consistent": round(float((~Cl[np.arange(len(li)), pf.argmax(1)]).mean()), 4),
        "pstar_prob_of_human_full": round(float(pf[np.arange(len(li)), a[li]].mean()), 4),
        "pstar_prob_of_human_restricted": round(float(pc[np.arange(len(li)), a[li]].mean()), 4),
        "sp_label_equals_human": round(float((ap[li] == a[li]).mean()), 4),
        "switch_share": {"human": round(float(SW[a[li]].mean()), 4), "sp_label": round(float(SW[ap[li]].mean()), 4),
                         "pstar_full_mass": round(float(pf[:, SW].sum(1).mean()), 4),
                         "pstar_restricted_mass": round(float(pc[:, SW].sum(1).mean()), 4)},
        "uniform_nll": round(float(np.log(Ml.sum(1)).mean()), 4),
        "pstar_nll_of_human": round(float(-np.log(np.maximum(pf[np.arange(len(li)), a[li]], 1e-300)).mean()), 4)}
    # decisions to the seat's last KEPT row in these shards (a battle cut at a shard edge reads a little short)
    key = H["bkey"] * 2 + H["seat"]
    last: dict = {}
    for kk, k in zip(key.tolist(), H["k"].tolist()):
        last[kk] = max(last.get(kk, -1), k)
    dist = np.array([last[kk] - k for kk, k in zip(key.tolist(), H["k"].tolist())])
    p_full = _softmax(LG, M)
    q = H["quit_dist"] >= 0
    groups = {"quitter": q, "forfeit_winner": (H["end"] == 1) & ~q, "ko_both_seats": H["end"] == 0}
    bins = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6))
    tail = {}
    for g, sel in groups.items():
        s0 = np.nonzero(sel & lab)[0]
        tail[g] = {}
        for lo, hi in bins:
            s = s0[(dist[s0] >= lo) & (dist[s0] <= hi)]
            tail[g][f"{lo}-{hi}"] = {"n": int(len(s)),
                                     "top1": round(float((LG[s].argmax(1) == a[s]).mean()), 4),
                                     "nll": round(float(-np.log(np.maximum(p_full[s, a[s]], 1e-300)).mean()), 4)}
    res["tail_by_decisions_to_last_kept_row"] = tail
    res["quitter_rows_within_8_of_last_kept_share"] = round(float((q & (dist <= 7)).mean()), 4)
    r = H["rating"][li]
    nll = -np.log(np.maximum(pf[np.arange(len(li)), a[li]], 1e-300))
    res["by_rating"] = {f"{lo}-{hi}": {"share": round(float(((r >= lo) & (r <= hi)).mean()), 4),
                                       "pstar_nll_of_human": round(float(nll[(r >= lo) & (r <= hi)].mean()), 4)}
                        for lo, hi in ((1300, 1399), (1400, 1499), (1500, 1599), (1600, 9999))}
    return res


def ss() -> dict:
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits

    sman = json.loads((SS / "rows_manifest.json").read_text())
    rng = np.random.default_rng(7)
    parts = {f: [] for f in ("rows", "mask", "act", "seat", "turn")}
    for i in (3, 41, 77):
        z = np.load(SS / f"{sman['train_shards'][i]['name']}.npz")
        ix = np.sort(rng.choice(len(z["act"]), 700, replace=False))
        for f in parts:
            parts[f].append(z[f][ix])
    Z = {f: np.concatenate(v) for f, v in parts.items()}
    with inference_globals(2):
        m = load_checkpoint(Path(sman["pstar"]))
        lg = policy_logits(m, Z["rows"], Z["mask"].astype(np.uint8)).astype(np.float64)
    p = _softmax(lg, Z["mask"].astype(bool))
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(1)
    act = Z["act"].astype(int)
    nll = -np.log(p[np.arange(len(act)), act])
    out = {"rows": int(len(act)), "uniform_nll": round(float(np.log(Z["mask"].sum(1)).mean()), 4),
           "switch_label_share": round(float(SW[act].mean()), 4)}
    for s in (0, 1):
        k = Z["seat"] == s
        out[f"seat{s}"] = {"n": int(k.sum()), "pstar_entropy": round(float(ent[k].mean()), 4),
                           "pstar_nll_of_label": round(float(nll[k].mean()), 4),
                           "top1_equals_label": round(float((p[k].argmax(1) == act[k]).mean()), 4)}
    return out


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, fn in (("pool", pool), ("rows", rows), ("ss", ss)):
        if which in (name, "all"):
            res[name] = fn()
            print(json.dumps({name: res[name]}, indent=1), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
