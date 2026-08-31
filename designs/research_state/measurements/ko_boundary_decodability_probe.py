"""KO-BOUNDARY DECODABILITY — is the win-prob head's blindness to knockout-roll structure an
EXPRESSIVENESS problem, a SUPERVISION problem, or a REPRESENTATION-COVERAGE problem?

The finding under test: `starmie_ttar_risk_probe_v2_2026-08-31.md` measured the win-prob head
pricing the displayed HP bar smoothly with ZERO excess response at the KO-roll boundary, over a
~40x compressed range; `exploiter_fingerprint_truthcheck_2026-08-31.md` replicated the resolution
defect at population scale (per-state |error| 0.278 against an aggregate bias of +0.036).

This probe asks a different question of the same network: **is the information there?** We build a
population of real decisions with GROUND-TRUTH P(KO) measured by Monte-Carlo re-rolling the actual
turn, then decode that truth from FROZEN internal features at several taps with several probe
classes. The three-way verdict rule is registered in the write-up.

  target T_out : P(the opponent loses a mon during this turn)  -- our knockout
  target T_in  : P(we lose a mon during this turn)             -- their knockout
  ground truth : R fresh-dice re-rolls of the recorded turn with BOTH sides' recorded actions
                 (`utils.bridge.reconstruction.reroll_many`, rust driver). Only the dice change,
                 so the label is exactly the roll/accuracy/crit randomness the mission is about.

Taps (all read off ONE frozen forward per recorded observation):
  obs_raw       2501  the raw observation -- the representation-coverage FLOOR
  hp_only          2  opponent-active and our-active HP fraction -- the "prices the HP bar" control
  op_out_move      5  DamageOperator outgoing [low, high, crit, pko] for the CHOSEN move + outspeed
  op_pko           1  that pko channel ALONE -- the direct-read row
  op_flat        138  the whole DamageOperator flat block
  op_move_cell    62  the pointer MOVE cell for the chosen action (what the policy scores it from)
  pi_features    512  the policy trunk
  vf_features    512  the critic trunk
  value_pooled   128  THE tensor the win-prob head consumes (v96: vf_combined IS value_pooled)

Probe classes: ridge LINEAR / MLP(64) / GLU(64) gated-multiplicative -- grouped 5-fold CV over
BATTLES (a battle never appears in both train and test).

Run (repo root; needs deps/pokemon-showdown + the rust search_driver):
    python designs/research_state/measurements/ko_boundary_decodability_probe.py --phase all
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Phases: labels | features | probe | report | all. Resumable; CPU-only, <=2 cores.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
STAMP = "ko_boundary_decodability_2026-08-31"
RUN = "ai_v9_59_R2ACTION_0827"
CKPT_REL = f"{RUN}/final_model.zip"
R_ROLLS = 64          # re-rolls per state -> label sd <= 0.5/sqrt(64) = 0.0625
MIN_TURN = 2          # turn 1 is not openable by the re-roll layer (cf_audit's documented gap)

TAPS = ("obs_raw", "hp_only", "op_pko", "op_out_move", "op_flat", "op_move_cell",
        "pi_features", "vf_features", "value_pooled")


# ---------------------------------------------------------------- trace discovery

def models_dir() -> Path:
    from utils.paths import main_models_dir
    d = main_models_dir()
    assert d is not None, "no models/ archive found (set $GEN3AI_MODELS_DIR)"
    return d


def trace_prefixes() -> list:
    root = models_dir() / RUN / "eval_traces"
    recs = sorted(glob.glob(str(root / "*" / "*" / "*_reconstruction.json")))
    out = []
    for r in recs:
        pre = r[: -len("_reconstruction.json")]
        if os.path.exists(pre + "_summary.json") and os.path.exists(pre + "_states.npz"):
            out.append(pre)
    return out


def label_rows() -> list:
    """Every labelled decision, from the plain shards AND the committed `.gz` ones.

    The `.jsonl.gz` files are what ships in git (60 KB each vs 1.2 MB); a fresh checkout must be
    able to re-run `features`/`probe` off them without re-paying the 664k rollouts, and a resume
    of `labels` must not redo work that only exists in compressed form.
    """
    import gzip
    rows, seen = [], set()
    for shard in sorted(OUT_DIR.glob(f"{STAMP}_labels_w*.jsonl")) + \
            sorted(OUT_DIR.glob(f"{STAMP}_labels_w*.jsonl.gz")):
        op = gzip.open if shard.suffix == ".gz" else open
        with op(shard, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["prefix"], r["i"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    return rows


def _pct(s) -> float:
    try:
        return float(str(s).strip().rstrip("%")) / 100.0
    except Exception:
        return float("nan")


def _our_side(record, summary) -> str:
    """Which sim side the TRAINEE played, by matching the trace's own team roster."""
    ours = [m["species"] for m in summary["teams"]["ours"]]
    for side in ("p1", "p2"):
        if sorted(d.get("species", "") for d in record.team_details(side)) == sorted(ours):
            return side
    raise AssertionError("could not identify the trainee's side from the team roster")


# ---------------------------------------------------------------- phase: labels

def phase_labels(worker: int, workers: int, limit: int, deadline_min: float) -> None:
    from utils.bridge.reconstruction import ReconstructionRecord, reroll_many

    shard = OUT_DIR / f"{STAMP}_labels_w{worker}.jsonl"
    done = {(r["prefix"], r["i"]) for r in label_rows()}

    pres = trace_prefixes()
    if limit:
        pres = pres[:limit]
    mine = [p for k, p in enumerate(pres) if k % workers == worker]
    print(f"[w{worker}] {len(mine)}/{len(pres)} battles, {len(done)} rows already done")

    seeds = [f"{i + 1},{i * 7 + 3},{i * 13 + 5},{i * 29 + 11}" for i in range(R_ROLLS)]
    arms = [{"p1_action": "recorded", "p2_action": "recorded", "seed": s, "label": k}
            for k, s in enumerate(seeds)]
    t0 = time.time()
    n_new = n_err = 0
    with shard.open("a") as fh:
        for bi, pre in enumerate(mine):
            if deadline_min and (time.time() - t0) / 60.0 > deadline_min:
                print(f"[w{worker}] deadline hit at battle {bi}")
                break
            try:
                summary = json.loads(Path(pre + "_summary.json").read_text())
                record = ReconstructionRecord.load(pre + "_reconstruction.json")
                us = _our_side(record, summary)
            except Exception as e:                       # a malformed trace is skipped, counted
                n_err += 1
                print(f"[w{worker}] SKIP {os.path.basename(pre)}: {type(e).__name__} {e}"[:200])
                continue
            them = "p2" if us == "p1" else "p1"
            key = os.path.relpath(pre, str(models_dir()))
            for i, inv in enumerate(summary["invocations"]):
                if (key, i) in done:
                    continue
                if inv.get("phase") != "move_selection":
                    continue
                turn = int(inv.get("turn", 0))
                if turn < MIN_TURN:
                    continue
                try:
                    r = reroll_many(record, turn, arms, impl="rust", timeout=600)
                except Exception:            # the documented forced_switch / re-roll coverage gap
                    n_err += 1
                    continue
                pre_alive = {s: sum(1 for m in r.pre_state[s]["pokemon"] if not m["fainted"])
                             for s in ("p1", "p2")}
                n = len(r.arms)
                if n == 0:
                    n_err += 1
                    continue
                k_out = sum(1 for a in r.arms if a.outcome[them]["alive"] < pre_alive[them])
                k_in = sum(1 for a in r.arms if a.outcome[us]["alive"] < pre_alive[us])
                fh.write(json.dumps({
                    "prefix": key, "i": i, "turn": turn, "n": n,
                    "p_out": k_out / n, "p_in": k_in / n,
                    "our_hp": _pct(inv.get("our", {}).get("hp")),
                    "opp_hp": _pct(inv.get("opp", {}).get("hp")),
                    "our_alive": pre_alive[us], "opp_alive": pre_alive[them],
                    "result": summary["meta"].get("result"),
                }) + "\n")
                n_new += 1
            fh.flush()
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"[w{worker}] {bi}/{len(mine)} battles  {n_new} rows  "
                      f"{el:.0f}s  {n_new / max(el, 1e-9):.1f} rows/s  {n_err} err")
    print(f"[w{worker}] DONE {n_new} new rows, {n_err} errors, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- phase: features

def phase_features(batch: int = 128) -> None:
    import torch
    torch.set_num_threads(2)
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    rows = label_rows()
    print(f"{len(rows)} labelled decisions over {len({r['prefix'] for r in rows})} battles")

    mp = load_mappings()
    model, _ = load_foreign_opponent(str(models_dir() / CKPT_REL),
                                     current_version=current_model_version(mp), device="cpu")
    fe = model.policy.features_extractor
    op = fe.damage_op

    by_battle = {}
    for r in rows:
        by_battle.setdefault(r["prefix"], []).append(r)

    feats = {t: [] for t in TAPS}
    meta = {k: [] for k in ("p_out", "p_in", "n", "turn", "our_hp", "opp_hp",
                            "action", "battle", "win_prob_rec", "win_prob_now",
                            "value_now", "our_alive", "opp_alive")}
    bidx = {}
    t0 = time.time()
    for bi, (prefix, rs) in enumerate(sorted(by_battle.items())):
        z = np.load(str(models_dir() / (prefix + "_states.npz")))
        obs_all, act_all, mask_all = z["obs"], z["actions"], z["action_mask"]
        wp_all = z["win_probs"] if "win_probs" in z.files else None
        idx = [r["i"] for r in rs if r["i"] < len(obs_all)]
        keep = [r for r in rs if r["i"] < len(obs_all)]
        if not idx:
            continue
        bidx.setdefault(prefix, len(bidx))
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            ks = keep[s:s + batch]
            obs = torch.as_tensor(obs_all[sl].astype(np.float32))
            msk = torch.as_tensor(mask_all[sl].astype(np.float32))
            with torch.no_grad():
                pi, vf = fe({"observation": obs, "action_mask": msk})
                t = op.last_tensors
                out_pm = t.out_per_move.numpy()          # [B,4,4] low,high,crit,pko vs their ACTIVE
                out_spd = t.out_p_outspeed.numpy()       # [B,1]
                flat = t.flat.numpy()                    # [B,138]
                mcell = fe.last_pointer_inputs.move_cells.numpy()   # [B,4,62]
                vpool = fe.last_value_pooled.numpy()
                wp = torch.sigmoid(fe.last_win_prob_logits)[:, 0].numpy()
                val = model.policy.predict_values(
                    {"observation": obs, "action_mask": msk})[:, 0].numpy()
            acts = act_all[sl].astype(int)
            for b, (r, a) in enumerate(zip(ks, acts)):
                slot = int(a) - 6
                if not (0 <= slot < out_pm.shape[1]):
                    continue                              # switch decision: no move cell
                feats["obs_raw"].append(obs_all[sl[b]].astype(np.float32))
                feats["hp_only"].append(np.array([r["opp_hp"], r["our_hp"]], np.float32))
                feats["op_pko"].append(np.array([out_pm[b, slot, 3]], np.float32))
                feats["op_out_move"].append(
                    np.concatenate([out_pm[b, slot], out_spd[b]]).astype(np.float32))
                feats["op_flat"].append(flat[b].astype(np.float32))
                feats["op_move_cell"].append(mcell[b, slot].astype(np.float32))
                feats["pi_features"].append(pi[b].numpy().astype(np.float32))
                feats["vf_features"].append(vf[b].numpy().astype(np.float32))
                feats["value_pooled"].append(vpool[b].astype(np.float32))
                meta["p_out"].append(r["p_out"]); meta["p_in"].append(r["p_in"])
                meta["n"].append(r["n"]); meta["turn"].append(r["turn"])
                meta["our_hp"].append(r["our_hp"]); meta["opp_hp"].append(r["opp_hp"])
                meta["our_alive"].append(r["our_alive"]); meta["opp_alive"].append(r["opp_alive"])
                meta["action"].append(int(a)); meta["battle"].append(bidx[prefix])
                meta["win_prob_rec"].append(
                    float(wp_all[sl[b]]) if wp_all is not None else float("nan"))
                meta["win_prob_now"].append(float(wp[b]))
                meta["value_now"].append(float(val[b]))
        if bi % 25 == 0:
            print(f"  {bi}/{len(by_battle)} battles  {len(meta['p_out'])} rows  "
                  f"{time.time() - t0:.0f}s")
    arrays = {f"X_{k}": np.stack(v) for k, v in feats.items() if v}
    arrays.update({f"m_{k}": np.asarray(v, np.float64) for k, v in meta.items()})
    np.savez_compressed(str(OUT_DIR / f"{STAMP}_features.npz"), **arrays)
    print(f"saved {len(meta['p_out'])} rows x {len(TAPS)} taps "
          f"-> {STAMP}_features.npz ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- probes

_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


class _RidgePath:
    """One Gram + one eigendecomposition, then every alpha for free.

    The naive `solve(G + aI)` per alpha is O(p^3) each, and the widest tap here is p = 2501 —
    so the alpha sweep would dominate the whole probe. G is symmetric PSD, so eigh once and
    each alpha costs O(p^2). The intercept column is never penalised, which is why it is
    centred out rather than appended to the penalised block.
    """

    def __init__(self, X, y):
        self.mu = X.mean(0)
        self.ybar = float(y.mean())
        Xc = X - self.mu
        G = Xc.T @ Xc
        self.w, self.V = np.linalg.eigh(G)
        self.Vty = self.V.T @ (Xc.T @ (y - self.ybar))

    def coef(self, alpha):
        return self.V @ (self.Vty / (self.w + alpha))

    def predict(self, X, alpha):
        return (X - self.mu) @ self.coef(alpha) + self.ybar


def _fit_ridge(Xtr, ytr, Xte, alphas=_ALPHAS):
    """Ridge with an inner-split alpha choice. Closed form; no sklearn dependency."""
    n = len(Xtr)
    cut = max(2, int(0.8 * n))
    inner = _RidgePath(Xtr[:cut], ytr[:cut])
    B, yb = Xtr[cut:], ytr[cut:]
    best, best_a = None, alphas[0]
    for a in alphas:
        e = float(np.mean((inner.predict(B, a) - yb) ** 2)) if len(B) else 0.0
        if best is None or e < best:
            best, best_a = e, a
    return _RidgePath(Xtr, ytr).predict(Xte, best_a)


def _fit_torch(Xtr, ytr, Xte, kind: str, hidden: int = 64, epochs: int = 300, seed: int = 0):
    import torch
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, 1))
    elif kind == "glu":
        net = _GLUProbe(d, hidden)
    else:
        raise ValueError(kind)
    n = len(Xtr)
    cut = max(1, int(0.85 * n))
    perm = torch.randperm(n, generator=g).numpy()
    tr, va = perm[:cut], perm[cut:]
    Xa = torch.as_tensor(Xtr[tr], dtype=torch.float32)
    ya = torch.as_tensor(ytr[tr], dtype=torch.float32)[:, None]
    Xv = torch.as_tensor(Xtr[va], dtype=torch.float32)
    yv = torch.as_tensor(ytr[va], dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    bs = 256
    for ep in range(epochs):
        idx = torch.randperm(len(Xa), generator=g)
        net.train()
        for s in range(0, len(Xa), bs):
            j = idx[s:s + bs]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xa[j]), ya[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(net(Xv), yv))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(torch.as_tensor(Xte, dtype=torch.float32))[:, 0].numpy()


class _GLUProbe:
    """Deferred import wrapper so torch is only needed when a probe actually runs."""

    def __new__(cls, d, hidden):
        import torch

        class GLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(d, hidden)
                self.b = torch.nn.Linear(d, hidden)
                self.a2 = torch.nn.Linear(hidden, hidden)
                self.b2 = torch.nn.Linear(hidden, hidden)
                self.out = torch.nn.Linear(hidden, 1)

            def forward(self, x):
                h = self.a(x) * torch.sigmoid(self.b(x))      # gated multiplicative unit
                h = self.a2(h) * torch.sigmoid(self.b2(h))
                return self.out(h)

        return GLU()


def _auc(y_bin, score) -> float:
    y_bin = np.asarray(y_bin, bool)
    if y_bin.all() or not y_bin.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = np.asarray(score)[order]
    r = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):                                        # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    n1 = int(y_bin.sum())
    n0 = len(y_bin) - n1
    return float((ranks[y_bin].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def phase_probe(target: str = "p_out", folds: int = 5, subset: str = "all",
                max_rows: int = 0) -> dict:
    d = np.load(str(OUT_DIR / f"{STAMP}_features.npz"))
    y = d[f"m_{target}"].astype(np.float64)
    battle = d["m_battle"].astype(int)
    n_roll = d["m_n"].astype(np.float64)
    keep = np.ones(len(y), bool)
    if subset == "live":                    # the KO race is genuinely open at this state
        keep = (y > 0.02) & (y < 0.98)
    idx = np.nonzero(keep)[0]
    if max_rows and len(idx) > max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    y, battle, n_roll = y[idx], battle[idx], n_roll[idx]

    # binomial noise floor: the best R^2 any decoder could reach against these labels
    noise_var = float(np.mean(y * (1 - y) / n_roll))
    ceiling = 1.0 - noise_var / float(np.var(y)) if np.var(y) > 0 else float("nan")

    ub = np.unique(battle)
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(ub))
    fold_of_battle = {int(b): int(perm[k] % folds) for k, b in enumerate(ub)}
    fold = np.array([fold_of_battle[int(b)] for b in battle])

    res = {"target": target, "subset": subset, "n_rows": int(len(y)),
           "n_battles": int(len(ub)), "folds": folds,
           "label_mean": float(y.mean()), "label_var": float(np.var(y)),
           "label_rolls": int(np.median(n_roll)),
           "binomial_noise_var": noise_var, "r2_ceiling": ceiling,
           "frac_label_lt_0.02": float(np.mean(y < 0.02)),
           "frac_label_gt_0.98": float(np.mean(y > 0.98)),
           "taps": {}}

    for tap in TAPS:
        key = f"X_{tap}"
        if key not in d.files:
            continue
        X = d[key][idx].astype(np.float64)
        # drop constant columns; standardise per fold on the TRAIN half only
        res["taps"][tap] = {"dim": int(X.shape[1]), "probes": {}}
        for kind in ("linear", "mlp", "glu"):
            preds = np.zeros(len(y))
            t0 = time.time()
            for f in range(folds):
                te = fold == f
                tr = ~te
                mu, sd = X[tr].mean(0), X[tr].std(0)
                sd[sd < 1e-8] = 1.0
                Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
                if kind == "linear":
                    p = _fit_ridge(Xtr, y[tr], Xte)
                else:
                    p = _fit_torch(Xtr, y[tr], Xte, kind, seed=f)
                preds[te] = p
            pc = np.clip(preds, 0.0, 1.0)
            sse = float(np.sum((y - pc) ** 2))
            sst = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - sse / sst
            # per-fold r2 for a spread
            fr2 = []
            for f in range(folds):
                te = fold == f
                s1 = float(np.sum((y[te] - pc[te]) ** 2))
                s0 = float(np.sum((y[te] - y[tr := ~te].mean()) ** 2))
                fr2.append(1.0 - s1 / s0 if s0 > 0 else float("nan"))
            res["taps"][tap]["probes"][kind] = {
                "r2": r2, "r2_folds": [round(v, 4) for v in fr2],
                "r2_sd": float(np.std(fr2)),
                "r2_normalised": r2 / ceiling if ceiling and ceiling > 0 else float("nan"),
                "auc": _auc(y > 0.5, pc), "rmse": float(np.sqrt(sse / len(y))),
                "sec": round(time.time() - t0, 1),
            }
            print(f"  {tap:14s} {kind:6s} R2={r2:+.4f} (norm {r2 / ceiling:+.3f}) "
                  f"AUC={res['taps'][tap]['probes'][kind]['auc']:.3f} "
                  f"[{time.time() - t0:.0f}s]")

    # reference rows: what the model's own heads say about this target, unfitted
    for name, arr in (("win_prob_now", d["m_win_prob_now"][idx]),
                      ("value_now", d["m_value_now"][idx])):
        res.setdefault("head_reference", {})[name] = {
            "pearson_r": float(np.corrcoef(arr, y)[0, 1]),
            "auc": _auc(y > 0.5, arr),
        }
    return res


# ---------------------------------------------------------------- report

_TAP_NOTE = {
    "obs_raw": "the raw observation (representation floor)",
    "hp_only": "opp + our active HP fraction (the 'prices the HP bar' control)",
    "op_pko": "the op's outgoing pko for the CHOSEN move, ALONE",
    "op_out_move": "op outgoing [low, high, crit, pko] for the chosen move + p_outspeed",
    "op_flat": "the whole DamageOperator flat block",
    "op_move_cell": "the pointer MOVE cell the policy scores this action from",
    "pi_features": "the policy trunk",
    "vf_features": "the critic trunk",
    "value_pooled": "THE tensor the win-prob head consumes",
}


def render_report(results: dict) -> str:
    out = []
    for key in sorted(k for k in results if k.startswith("probe_")):
        r = results[key]
        out.append(f"\n### {key}  —  target `{r['target']}`, subset `{r['subset']}`")
        out.append(f"n = {r['n_rows']} decisions over {r['n_battles']} battles · "
                   f"label mean {r['label_mean']:.3f}, var {r['label_var']:.4f} · "
                   f"R = {r['label_rolls']} re-rolls · "
                   f"binomial-noise R2 ceiling **{r['r2_ceiling']:.3f}**")
        out.append("")
        out.append("| tap | dim | linear R2 | MLP R2 | GLU R2 | best/ceiling | linear AUC | GLU AUC |")
        out.append("|---|---|---|---|---|---|---|---|")
        for tap in TAPS:
            t = r["taps"].get(tap)
            if not t:
                continue
            p = t["probes"]
            best = max(p[k]["r2"] for k in p)
            out.append(
                f"| `{tap}` | {t['dim']} | {p['linear']['r2']:+.3f} | {p['mlp']['r2']:+.3f} | "
                f"{p['glu']['r2']:+.3f} | {best / r['r2_ceiling']:.3f} | "
                f"{p['linear']['auc']:.3f} | {p['glu']['auc']:.3f} |")
        hr = r.get("head_reference", {})
        if hr:
            out.append("")
            out.append("| unfitted head reference | pearson r vs truth | AUC |")
            out.append("|---|---|---|")
            for name, v in hr.items():
                out.append(f"| `{name}` | {v['pearson_r']:+.3f} | {v['auc']:.3f} |")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["labels", "features", "probe", "report", "all"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline-min", type=float, default=0.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    out_path = OUT_DIR / f"{STAMP}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("meta", {
        "date": "2026-08-31", "run": RUN, "checkpoint": CKPT_REL,
        "r_rolls": R_ROLLS, "impl": "rust", "taps": list(TAPS),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
    })

    if args.phase in ("labels", "all"):
        phase_labels(args.worker, args.workers, args.limit, args.deadline_min)
    if args.phase in ("features", "all"):
        phase_features()
    if args.phase in ("probe", "all"):
        for target in ("p_out", "p_in"):
            for subset in ("all", "live"):
                print(f"== probe target={target} subset={subset} ==")
                results[f"probe_{target}_{subset}"] = phase_probe(
                    target, folds=args.folds, subset=subset, max_rows=args.max_rows)
                out_path.write_text(json.dumps(results, indent=1))
                print(f"[saved {out_path}]")
    if args.phase in ("report", "all"):
        print(render_report(results))
    out_path.write_text(json.dumps(results, indent=1))
    print("done.")


if __name__ == "__main__":
    main()
