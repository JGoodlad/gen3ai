"""OFFLINE PROBE SI-1 — is opponent SKILL/CLASS inferable from gameplay, and how fast?

First gate of the opponent-skill-conditioning candidate (ledger: "🎚️ DESIGN CANDIDATE:
opponent-SKILL conditioning", 8c1c2e8). From an in-progress battle's observable history alone
(what OUR side legally sees), predict (a) the opponent's CLASS (which scripted bot vs
pool-snapshot "model") and (b) the opponent's anchored ELO — as a function of TURNS OBSERVED.

Data: existing eval traces (models/<run>/eval_traces/step_*/<opponent>/*_summary.json),
READ-ONLY. Labels:
  - class    = the per-opponent directory name (9 scripted bots; sentinel_* pooled as "model").
  - bot ELO  = data/gen3_bot_elo_anchors.json (anchored Bradley-Terry, cross-run comparable).
  - sentinel ELO = <run>/eval_results.jsonl row at the eval step gives the ordered sentinel
    snapshot steps (sentinel_i = i-th entry); <run>/snapshot_ladder/ladder.json rates each
    snapshot step on the same bot-anchored scale.

FEATURES are built ONLY from opponent-observable behaviour — the opponent's realized action
each turn (`outcome.opp.action`), their public active HP / revealed bench, and the public HP
deltas of the exchange. Move identity is reduced to public move PROPERTIES (basePower,
accuracy, priority, status-ness via data/pokemon/gen3_moves.json) plus diversity/repetition
rates, never a move-id or species one-hot (a species/move one-hot would classify the TEAM
DRAW, not the behaviour). Explicitly excluded: our policy probs, our chosen action, the
recorded belief/opp_intent heads (our model's outputs), rewards, values, battle RESULT and
total battle length (a "short battle we won" feature would ride the trainee's win rate — a
label shortcut, not opponent behaviour).

A simple model is the point (HistGradientBoosting, default params, GroupKFold by
(run, eval-step) so no eval cycle straddles train/test): we measure the DATA's separability,
not a classifier ceiling.

Run (main checkout has models/; worktree needs PYTHONPATH + GEN3AI_MODELS_DIR resolution):
  export PYTHONPATH=$PYTHONPATH:src
  nice -n 15 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
      designs/research_state/measurements/opponent_skill_inferability_probe.py \
      [--models-dir /home/goodlad/dev/gen3ai/models] [--out-dir <this dir>]
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date

os.environ.setdefault("OMP_NUM_THREADS", "2")  # discipline: <=2 cores beside the live run
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402

BOT_NAMES = [
    "random", "heuristic", "heuristic2", "staller", "staller_v2",
    "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2",
]
DEFAULT_RUN_GLOBS = ["ai_v9_1[3-9]*", "ai_v9_2[01]*"]  # gen11..gen17 era + tdaux probes
PREFIXES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 0]  # 0 = full game
CACHE = "/tmp/si1_opponent_skill_cache.jsonl.gz"

_PCT = re.compile(r"(-?\d+(?:\.\d+)?)%")


def _pct(s):
    """'-6%' -> -0.06; None/'' -> None."""
    if not s:
        return None
    m = _PCT.search(s)
    return float(m.group(1)) / 100.0 if m else None


# ---------------------------------------------------------------------------
# Trace parsing -> per-battle behaviour sequence
# ---------------------------------------------------------------------------

def load_moves(repo_root: str) -> dict:
    with open(os.path.join(repo_root, "data", "pokemon", "gen3_moves.json")) as f:
        return json.load(f)


def parse_battle(path: str, moves: dict):
    """One summary.json -> a per-turn opponent-behaviour sequence (opponent-observable only)."""
    try:
        with open(path) as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None
    invs = s.get("invocations") or []
    seq = []
    seen_turns = set()
    prev_opp_dhp = None
    for inv in invs:
        if inv.get("phase") != "move_selection":
            continue
        t = inv.get("turn")
        if t in seen_turns:  # 2nd move_selection in one turn (rare) — keep the first
            continue
        out = inv.get("outcome") or {}
        opp_out = out.get("opp") or {}
        action = (opp_out.get("action") or "").strip()
        if not action or action == "none":
            continue
        seen_turns.add(t)
        opp_state = inv.get("opp") or {}
        bench = opp_state.get("bench") or ""
        n_revealed = 1 + (len([b for b in bench.split(",") if b.strip()]) if bench else 0)
        hp_before = _pct(opp_state.get("hp"))
        # action forms: "switched_to:X" | "move" | "move → X_sent_in"
        faint_repl = "_sent_in" in action
        base = action.split("→")[0].split("->")[0].strip()
        is_switch = base.startswith("switched_to:")
        move_id = None if is_switch else re.sub(r"[^a-z0-9]", "", base.lower()) or None
        mv = moves.get(move_id) if move_id else None
        rec = {
            "t": t,
            "switch": is_switch,
            "faint_repl": faint_repl,
            "move": move_id if not is_switch else None,
            "bp": (mv or {}).get("basePower", 0) if not is_switch else None,
            "acc": (mv or {}).get("accuracy", 100) if not is_switch else None,
            "prio": (mv or {}).get("priority", 0) if not is_switch else None,
            "known_move": mv is not None,
            "our_dhp": _pct((out.get("our") or {}).get("hp_delta")),
            "opp_dhp": _pct(opp_out.get("hp_delta")),
            "opp_hp": hp_before,
            "n_revealed": n_revealed,
            "prev_opp_dhp": prev_opp_dhp,
        }
        prev_opp_dhp = rec["opp_dhp"]
        seq.append(rec)
    if not seq:
        return None
    meta = s.get("meta") or {}
    return {"turns": meta.get("turns"), "result": meta.get("result"), "seq": seq}


def sentinel_elo_maps(run_dir: str):
    """(eval_step, sentinel_index) -> anchored ELO, via eval_results.jsonl + ladder.json."""
    out = {}
    ladder_p = os.path.join(run_dir, "snapshot_ladder", "ladder.json")
    er_p = os.path.join(run_dir, "eval_results.jsonl")
    if not (os.path.exists(ladder_p) and os.path.exists(er_p)):
        return out
    try:
        with open(ladder_p) as f:
            ratings = {int(k): float(v) for k, v in json.load(f)["ratings"].items()}
        with open(er_p) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError, KeyError):
        return out
    for row in rows:
        for i, sent in enumerate(row.get("sentinels") or []):
            st = sent.get("step")
            if st in ratings:
                out[(int(row["step"]), i)] = ratings[st]
    return out


def collect(models_dir: str, repo_root: str, run_globs, cache_path: str, refresh: bool):
    if os.path.exists(cache_path) and not refresh:
        battles = []
        with gzip.open(cache_path, "rt") as f:
            for line in f:
                battles.append(json.loads(line))
        print(f"[cache] {len(battles)} battles from {cache_path}")
        return battles
    moves = load_moves(repo_root)
    with open(os.path.join(repo_root, "data", "gen3_bot_elo_anchors.json")) as f:
        bot_elos = json.load(f)["ratings"]
    run_dirs = sorted({d for g in run_globs for d in glob.glob(os.path.join(models_dir, g))
                       if os.path.isdir(os.path.join(d, "eval_traces"))})
    print(f"[collect] {len(run_dirs)} runs: {[os.path.basename(r) for r in run_dirs]}")
    battles = []
    for rd in run_dirs:
        run = os.path.basename(rd)
        sent_elo = sentinel_elo_maps(rd)
        for step_dir in sorted(glob.glob(os.path.join(rd, "eval_traces", "step_*"))):
            try:
                step = int(os.path.basename(step_dir).split("_")[1])
            except ValueError:
                continue
            for opp_dir in sorted(glob.glob(os.path.join(step_dir, "*"))):
                opp = os.path.basename(opp_dir)
                if not os.path.isdir(opp_dir):
                    continue
                if opp in BOT_NAMES:
                    cls, elo = opp, bot_elos.get(opp)
                elif opp.startswith("sentinel_"):
                    cls = "model"
                    elo = sent_elo.get((step, int(opp.split("_")[1])))
                else:
                    continue  # ext_* external opponents etc. — out of scope
                for p in glob.glob(os.path.join(opp_dir, "*_summary.json")):
                    b = parse_battle(p, moves)
                    if b is None:
                        continue
                    b.update(run=run, step=step, opp=opp, cls=cls, elo=elo)
                    battles.append(b)
        print(f"  {run}: cumulative {len(battles)} battles")
    with gzip.open(cache_path, "wt") as f:
        for b in battles:
            f.write(json.dumps(b) + "\n")
    print(f"[collect] {len(battles)} battles -> {cache_path}")
    return battles


# ---------------------------------------------------------------------------
# Features over a turn prefix
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "vol_switch_rate",      # voluntary switches / decisions
    "n_species_revealed",   # opp species we have seen by t
    "status_move_rate",     # basePower==0 moves / move actions
    "heal_rate",            # own hp_delta >= +10% on a move turn
    "repeat_rate",          # same move as their previous move action
    "distinct_move_frac",   # distinct moves / move actions
    "mean_dmg_to_us",       # mean damage we took on their attacking turns
    "max_dmg_to_us",
    "zero_dmg_attack_rate", # attacking move that dealt us no damage (immune/miss/blocked)
    "stay_low_hp_rate",     # made a move while active <=30% HP
    "switch_low_hp_rate",   # voluntarily switched while active <=30% HP
    "tank_stay_rate",       # took <=-30% last turn and did NOT switch
    "priority_move_rate",   # attacking move with priority>0
    "mean_hp_at_vol_switch",# panic-vs-planned switching (1.0 default when none)
    "faint_repl_per_turn",
    "mean_move_bp",         # mean basePower of chosen attacks / 150
    "mean_move_acc",        # mean accuracy of chosen moves / 100
]


def features(seq, t):
    """Feature vector from the first t observed turns (t=0 -> full game)."""
    sub = seq if t == 0 else seq[:t]
    n = len(sub)
    mvs = [r for r in sub if not r["switch"]]
    atk = [r for r in mvs if (r["bp"] or 0) > 0]
    vol_sw = [r for r in sub if r["switch"]]
    dmg = [max(0.0, -(r["our_dhp"] or 0.0)) for r in atk]
    move_ids = [r["move"] for r in mvs if r["move"]]
    repeats = sum(1 for a, b in zip(move_ids, move_ids[1:]) if a == b)
    low = [r for r in sub if (r["opp_hp"] is not None and r["opp_hp"] <= 0.30)]
    tank = [r for r in sub if (r["prev_opp_dhp"] is not None and r["prev_opp_dhp"] <= -0.30)]
    heal = sum(1 for r in mvs if (r["opp_dhp"] or 0.0) >= 0.10)
    f = [
        len(vol_sw) / n,
        sub[-1]["n_revealed"],
        (len(mvs) - len(atk)) / max(1, len(mvs)),
        heal / max(1, len(mvs)),
        repeats / max(1, len(move_ids) - 1) if len(move_ids) > 1 else 0.0,
        len(set(move_ids)) / max(1, len(move_ids)),
        float(np.mean(dmg)) if dmg else 0.0,
        float(np.max(dmg)) if dmg else 0.0,
        sum(1 for d in dmg if d <= 0.005) / max(1, len(atk)),
        sum(1 for r in low if not r["switch"]) / max(1, len(low)) if low else 0.0,
        sum(1 for r in low if r["switch"]) / max(1, len(low)) if low else 0.0,
        sum(1 for r in tank if not r["switch"]) / max(1, len(tank)) if tank else 0.0,
        sum(1 for r in atk if (r["prio"] or 0) > 0) / max(1, len(atk)),
        float(np.mean([r["opp_hp"] for r in vol_sw if r["opp_hp"] is not None] or [1.0])),
        sum(1 for r in sub if r["faint_repl"]) / n,
        float(np.mean([min(r["bp"], 150) / 150.0 for r in atk] or [0.0])),
        float(np.mean([(r["acc"] if r["acc"] not in (None, True) else 100) / 100.0
                       for r in mvs] or [1.0])),
    ]
    return f


# ---------------------------------------------------------------------------
# Models + curves
# ---------------------------------------------------------------------------

def run_analysis(battles, out_json: str, out_md: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, r2_score,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr

    rng = np.random.RandomState(0)
    cls_names = sorted({b["cls"] for b in battles})
    groups_all = np.array([f'{b["run"]}|{b["step"]}' for b in battles])
    y_cls = np.array([b["cls"] for b in battles])
    y_bin = np.array([0 if b["cls"] == "model" else 1 for b in battles])  # 1 = scripted bot
    y_elo = np.array([b["elo"] if b["elo"] is not None else np.nan for b in battles])
    sent_mask = np.array([b["cls"] == "model" and b["elo"] is not None for b in battles])

    results = {"n_battles": len(battles),
               "n_by_class": dict(Counter(y_cls.tolist())),
               "n_sentinel_elo_labeled": int(sent_mask.sum()),
               "feature_names": FEATURE_NAMES,
               "prefixes": {},
               }
    conf_store = {}
    gkf = GroupKFold(n_splits=5)

    for t in PREFIXES:
        X = np.array([features(b["seq"], t) for b in battles], dtype=np.float64)
        key = "full" if t == 0 else f"t{t}"
        # ---- binary bot-vs-model
        accs, aucs = [], []
        bin_pred = np.zeros(len(battles), dtype=int)
        for tr, te in gkf.split(X, y_bin, groups_all):
            m = HistGradientBoostingClassifier(random_state=0)
            m.fit(X[tr], y_bin[tr])
            p = m.predict_proba(X[te])[:, 1]
            bin_pred[te] = (p >= 0.5).astype(int)
            accs.append(float(np.mean((p >= 0.5) == y_bin[te])))
            aucs.append(float(roc_auc_score(y_bin[te], p)))
        # ---- 10-way class
        cls_pred = np.empty(len(battles), dtype=object)
        for tr, te in gkf.split(X, y_cls, groups_all):
            m = HistGradientBoostingClassifier(random_state=0)
            m.fit(X[tr], y_cls[tr])
            cls_pred[te] = m.predict(X[te])
        bal = float(balanced_accuracy_score(y_cls, cls_pred.astype(str)))
        # ---- sentinel ELO regression (kin policies — the registered-weak leg)
        elo_stats = None
        if sent_mask.sum() > 100:
            Xs, ys, gs = X[sent_mask], y_elo[sent_mask], groups_all[sent_mask]
            pred = np.zeros(len(ys))
            for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
                r = HistGradientBoostingRegressor(random_state=0)
                r.fit(Xs[tr], ys[tr])
                pred[te] = r.predict(Xs[te])
            elo_stats = {"r2": float(r2_score(ys, pred)),
                         "spearman": float(spearmanr(ys, pred).statistic),
                         "mae": float(np.mean(np.abs(ys - pred))),
                         "n": int(len(ys)),
                         "label_sd": float(np.std(ys))}
        results["prefixes"][key] = {
            "binary_acc": float(np.mean(accs)), "binary_acc_sd": float(np.std(accs)),
            "binary_auc": float(np.mean(aucs)),
            "tenway_balanced_acc": bal,
            "sentinel_elo": elo_stats,
        }
        if t in (3, 10):
            cm = confusion_matrix(y_cls, cls_pred.astype(str), labels=cls_names)
            conf_store[key] = (cls_names, cm)
            results["prefixes"][key]["confusion_labels"] = cls_names
            results["prefixes"][key]["confusion"] = cm.tolist()
        print(f"[{key}] bot-vs-model acc={np.mean(accs):.3f} auc={np.mean(aucs):.3f} "
              f"10way-bal={bal:.3f} eloR2={elo_stats['r2'] if elo_stats else None}")

    # ---- STRICT cross-run robustness: GroupKFold by RUN. Under (run,step) grouping the
    # same sentinel policy recurs across its run's cycles, so the ELO leg can memorize a
    # policy fingerprint -> rating lookup; run-level grouping is the honest skill read.
    run_groups = np.array([b["run"] for b in battles])
    results["cross_run"] = {}
    for t in (3, 10, 0):
        X = np.array([features(b["seq"], t) for b in battles], dtype=np.float64)
        key = "full" if t == 0 else f"t{t}"
        accs, aucs = [], []
        for tr, te in GroupKFold(n_splits=5).split(X, y_bin, run_groups):
            m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y_bin[tr])
            p = m.predict_proba(X[te])[:, 1]
            accs.append(float(np.mean((p >= 0.5) == y_bin[te])))
            aucs.append(float(roc_auc_score(y_bin[te], p)))
        elo_stats = None
        if sent_mask.sum() > 100:
            Xs, ys, gs = X[sent_mask], y_elo[sent_mask], run_groups[sent_mask]
            pred = np.zeros(len(ys))
            for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
                r = HistGradientBoostingRegressor(random_state=0).fit(Xs[tr], ys[tr])
                pred[te] = r.predict(Xs[te])
            elo_stats = {"r2": float(r2_score(ys, pred)),
                         "spearman": float(spearmanr(ys, pred).statistic),
                         "mae": float(np.mean(np.abs(ys - pred)))}
        results["cross_run"][key] = {"binary_acc": float(np.mean(accs)),
                                     "binary_auc": float(np.mean(aucs)),
                                     "sentinel_elo": elo_stats}
        print(f"[cross-run {key}] binary acc={np.mean(accs):.3f} auc={np.mean(aucs):.3f} "
              f"eloR2={elo_stats['r2'] if elo_stats else None}")

    # ---- feature attribution: binary at t=3, 10-way full — one held-out fold, permutation
    attribution = {}
    for label, t, y in (("binary_t3", 3, y_bin), ("tenway_full", 0, y_cls)):
        X = np.array([features(b["seq"], t) for b in battles], dtype=np.float64)
        tr, te = next(gkf.split(X, y, groups_all))
        m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        pi = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0, n_jobs=2)
        order = np.argsort(-pi.importances_mean)
        attribution[label] = [
            {"feature": FEATURE_NAMES[i], "importance": float(pi.importances_mean[i]),
             "sd": float(pi.importances_std[i])} for i in order]
    results["attribution"] = attribution

    # ---- per-class one-vs-rest separability at full game (who is easy/hard)
    Xf = np.array([features(b["seq"], 0) for b in battles], dtype=np.float64)
    per_class_recall = {}
    cls_names_arr = np.array(cls_names)
    if "full" not in conf_store:  # recompute full-game confusion for per-class recall
        cls_pred = np.empty(len(battles), dtype=object)
        for tr, te in gkf.split(Xf, y_cls, groups_all):
            m = HistGradientBoostingClassifier(random_state=0).fit(Xf[tr], y_cls[tr])
            cls_pred[te] = m.predict(Xf[te])
        cmf = confusion_matrix(y_cls, cls_pred.astype(str), labels=cls_names)
        results["confusion_full"] = {"labels": cls_names, "matrix": cmf.tolist()}
        for i, c in enumerate(cls_names):
            per_class_recall[c] = float(cmf[i, i] / max(1, cmf[i].sum()))
    results["per_class_recall_full"] = per_class_recall

    # ---- class means of key features (the readable "horrible play" table)
    key_feats = ["vol_switch_rate", "zero_dmg_attack_rate", "status_move_rate",
                 "repeat_rate", "stay_low_hp_rate", "mean_dmg_to_us", "mean_move_acc"]
    idx = [FEATURE_NAMES.index(k) for k in key_feats]
    class_means = {}
    for c in cls_names:
        mask = y_cls == c
        class_means[c] = {k: float(np.mean(Xf[mask][:, i])) for k, i in zip(key_feats, idx)}
    results["class_feature_means_full"] = class_means

    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_json}")
    if os.path.exists(out_md):
        # The committed .md is the CURATED write-up (tables + interpretation + registered-
        # prediction scoring); never clobber it. A re-run refreshes the JSON and writes the
        # regenerated tables beside it for diffing.
        out_md = out_md.replace(".md", ".autogen.md")
    write_md(results, out_md)
    print(f"wrote {out_md}")
    return results


def write_md(res, out_md):
    L = []
    A = L.append
    A(f"# SI-1 — opponent skill/class inferability from observable gameplay ({date.today().isoformat()})")
    A("")
    A("Probe script: `opponent_skill_inferability_probe.py` (beside this file). "
      "Data: eval traces of the ai_v9 gen11–gen17-era runs, labels = trace directory name "
      "(class), bot ELO anchors + per-run snapshot ladder via `eval_results.jsonl` sentinel "
      "order (ELO). Features: opponent-observable behaviour only — see the script docstring "
      "for the exclusion list (no move/species identity, no our-policy internals, no result "
      "or battle-length features).")
    A("")
    A(f"**{res['n_battles']} battles**, classes: "
      + ", ".join(f"{k}={v}" for k, v in sorted(res["n_by_class"].items()))
      + f"; sentinel battles with resolvable anchored ELO: {res['n_sentinel_elo_labeled']}.")
    A("")
    A("## The inferability curve")
    A("")
    A("| turns observed | bot-vs-model acc | AUC | 10-way balanced acc | sentinel-ELO R² | Spearman | ELO MAE |")
    A("|---|---|---|---|---|---|---|")
    for key, p in res["prefixes"].items():
        e = p.get("sentinel_elo") or {}
        A(f"| {key} | {p['binary_acc']:.3f} ± {p['binary_acc_sd']:.3f} | {p['binary_auc']:.3f} "
          f"| {p['tenway_balanced_acc']:.3f} | "
          f"{e.get('r2', float('nan')):.3f} | {e.get('spearman', float('nan')):.3f} | "
          f"{e.get('mae', float('nan')):.0f} |")
    A("")
    for key in ("t3", "t10"):
        p = res["prefixes"].get(key) or {}
        if "confusion" not in p:
            continue
        labels = p["confusion_labels"]
        cm = np.array(p["confusion"])
        A(f"## 10-way confusion at {key} (rows = truth, recall on diagonal)")
        A("")
        A("| truth \\ pred | " + " | ".join(labels) + " | recall |")
        A("|---" * (len(labels) + 2) + "|")
        for i, lab in enumerate(labels):
            row = cm[i]
            rec = row[i] / max(1, row.sum())
            A(f"| **{lab}** | " + " | ".join(str(int(v)) for v in row) + f" | {rec:.2f} |")
        A("")
    A("## Feature attribution (permutation importance, held-out fold)")
    A("")
    for label, rows in res["attribution"].items():
        A(f"**{label}**: " + ", ".join(
            f"{r['feature']} {r['importance']:.3f}" for r in rows[:6]))
        A("")
    A("## Class means of the readable features (full game)")
    A("")
    feats = list(next(iter(res["class_feature_means_full"].values())).keys())
    A("| class | " + " | ".join(feats) + " |")
    A("|---" * (len(feats) + 1) + "|")
    for c, d in sorted(res["class_feature_means_full"].items()):
        A(f"| {c} | " + " | ".join(f"{d[k]:.3f}" for k in feats) + " |")
    A("")
    with open(out_md, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--models-dir", default="/home/goodlad/dev/gen3ai/models")
    ap.add_argument("--repo-root", default="/home/goodlad/dev/gen3ai")
    ap.add_argument("--out-dir", default=here)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUN_GLOBS)
    ap.add_argument("--refresh", action="store_true", help="ignore the /tmp parse cache")
    args = ap.parse_args()
    battles = collect(args.models_dir, args.repo_root, args.runs, CACHE, args.refresh)
    if not battles:
        print("no battles collected", file=sys.stderr)
        sys.exit(2)
    stamp = "2026-08-31"
    run_analysis(battles,
                 os.path.join(args.out_dir, f"opponent_skill_inferability_{stamp}.json"),
                 os.path.join(args.out_dir, f"opponent_skill_inferability_{stamp}.md"))


if __name__ == "__main__":
    main()
