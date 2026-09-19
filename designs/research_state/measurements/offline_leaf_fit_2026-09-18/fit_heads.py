"""THE OFFLINE LEAF FIT — five value heads fitted on branched self-play, read on fresh contested forks.

    python3 fit_heads.py --train-forks <dir> --eval-forks <dir> --snapshot <zip> \
        --guard-tree <eval_traces/step_N> --out <dir>

WHAT MOVES AND WHAT DOES NOT. The TRUNK is frozen and is not in the graph: every feature is
forwarded once, offline, under `no_grad`, and every fit is a small MLP on a fixed matrix. Nothing
under `models/` is touched; CPU only.

THE TWO DATASETS ARE DIFFERENT POPULATIONS ON PURPOSE.

  TRAIN  `branch_forks.py`'s AlphaGo set — a uniformly-random legal move at a uniformly-random
         depth of a frozen-policy self-play game, the frozen policy to the terminal, plus the
         policy's own top-1 sibling from the same pre-fork state under common random numbers.
         Split 80/20 BY BATTLE (two forks of one battle share a prefix; splitting by fork leaks).
  READ   `paired_refit_discrimination_2026-09-14/forks.py`'s CONTESTED set, built VERBATIM from a
         SEPARATE tree of the same frozen policy — the instrument every published level on this
         metric is measured on (0.568-0.578, four heads). The fits never see it.

THE FIVE REGISTERED FITS, plus two init controls:

  (a)  WinProbHead, WARM-STARTED from the checkpoint's own head, BCE on the RAND-branch positions
       only -- the strict one-label-per-game AlphaGo dataset.
  (b)  the same, BCE on BOTH branches of the paired siblings.
  (c)  a WIDER head (LN -> 256 -> ReLU -> 256 -> ReLU -> 1), FRESH-INIT, on `value_pooled`.
  (d)  the same wider head on [PRE-POOL 12 team tokens (12x128, fainted slots zeroed, flattened)
       CONCATENATED WITH `value_pooled`] -- a strict SUPERSET of (c)'s input, so (d) > (c) means
       the value CLS pool discards ranking information.
  (e)  (c) plus a pairwise ranking term at coefficient 0.3 over the fork's own sibling pairs.
  (a0) WinProbHead FRESH-INIT on (a)'s data, and (c0) WinProbHead FRESH-INIT on (c)'s data -- the
       matched-init controls, because (a)/(b) are warm-started and (c)/(d)/(e) cannot be.

🚨 `vf_features` IS NOT THE PRE-POOL TENSOR. The extractor returns
`activation(value_projection(value_pre_norm(vf_combined)))`, and since the critic-route deletion
wave `vf_combined IS value_pooled` (`projection.py`), so `vf_features` is a DETERMINISTIC FUNCTION
of `value_pooled` and carries no information it lacks -- a fit on it could not answer the pooling
question. The genuine pre-pool tensor is the 12-token memory the value CLS query attends over,
captured here at the `cls_pool.value_cls_attn` seam (so it carries every value-route token
injection the pool itself sees) and concatenated with `value_pooled` (so the route contributions
that are added AFTER the pool are not silently dropped).

THE READ, on the contested set: PAIRWISE ACCURACY on non-tied sibling pairs, bootstrap over FORKS,
with ECE / Brier / separation beside it so ranking is not bought with calibration -- all of it
`refit.read_head` / `refit.paired_delta`, IMPORTED, never re-implemented. The branch-row ->
successor index is recomputed independently and ASSERTED (`score_forks.branch_index`), which is the
assertion the 2026-09-14 indexing artifact exists to make unrepeatable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_REC14 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                      "paired_refit_discrimination_2026-09-14"))
_REC16 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                      "fork_arm_read_2026-09-16"))
sys.path.insert(0, _REC14)
sys.path.insert(0, _REC16)

TRAIN_BRANCHES = ("top1", "rand")


# ----------------------------------------------------------------- the pre-pool tap
class PrePoolTap:
    """Capture the 12-token memory the VALUE CLS query attends over, at the attention seam.

    A hook on `cls_pool.value_cls_attn` rather than on `cls_pool` itself: the pool's key/value is
    `cat([our_for_value, their_team_out])` AFTER any value-token injection, so a pre-hook on the
    OUTER module would capture a tensor the pool does not actually read.
    """

    def __init__(self, model):
        fe = model.policy.features_extractor
        self.attn = fe.cls_pool.value_cls_attn
        self.keys = None
        self.mask = None
        self._h = None

    def __enter__(self):
        def hook(_mod, args, kwargs):
            self.keys = args[1].detach()
            m = kwargs.get("key_padding_mask")
            self.mask = None if m is None else m.detach()
            return None
        self._h = self.attn.register_forward_pre_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        return False

    def take(self):
        """[B, 12, D] with key-masked (fainted) slots ZEROED — a flat MLP has no key mask."""
        import torch
        k = self.keys
        if k is None:
            raise RuntimeError("the value CLS attention did not run in this forward")
        if self.mask is not None:
            k = k * (~self.mask.bool()).unsqueeze(-1).to(k.dtype)
        self.keys = self.mask = None
        return torch.as_tensor(k).numpy().copy()


def forward_feats(model, obs, masks, *, batch=256, want_prepool=True):
    """(value_pooled [N,D], prepool [N,12,D] or None, V0 [N]) from the FROZEN trunk."""
    import torch
    from forks import score_batch
    n = len(obs)
    pooled = np.empty((n, 128), np.float32)
    V0 = np.empty(n, np.float64)
    pre = np.empty((n, 12, 128), np.float32) if want_prepool else None
    tap = PrePoolTap(model) if want_prepool else None
    ctx = tap if want_prepool else _Null()
    with ctx:
        for i in range(0, n, batch):
            j = min(n, i + batch)
            pooled[i:j], V0[i:j] = score_batch(model, obs[i:j], masks[i:j])
            if want_prepool:
                pre[i:j] = tap.take()
    del torch
    return pooled, pre, V0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------- data
def load_rows(fork_dir, branches):
    """`refit.load_forks`'s reader, generalised to a row schema with a different branch set."""
    rows, succ, mask, off = [], [], [], 0
    for rp in sorted(glob.glob(os.path.join(fork_dir, "forks_s*.jsonl"))):
        tag = os.path.basename(rp)[len("forks_"):-len(".jsonl")]
        sp = os.path.join(fork_dir, f"succ_{tag}.npy")
        if not os.path.exists(sp):
            continue
        S = np.load(sp)
        M = np.load(os.path.join(fork_dir, f"succmask_{tag}.npy"))
        n_ok = 0
        for line in open(rp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break                                   # a torn last line on a killed shard
            if any((r["branches"][b].get("succ") or -1) >= len(S) for b in branches):
                continue                                # written after the last npy flush
            for b in branches:
                s = r["branches"][b].get("succ")
                r["branches"][b]["succ"] = None if s is None else s + off
            rows.append(r)
            n_ok += 1
        succ.append(S)
        mask.append(M)
        off += len(S)
        print(f"  {tag}: {n_ok} rows, {len(S)} successor states", flush=True)
    return rows, np.concatenate(succ, 0), np.concatenate(mask, 0)


def train_table(rows, branches):
    """Branch rows (the BCE data) + within-fork sibling pairs (the ranking data)."""
    b_fork, b_idx, b_y, b_battle, b_name = [], [], [], [], []
    for f, r in enumerate(rows):
        for nm in branches:
            br = r["branches"][nm]
            if br["succ"] is None or br["capped"] or br["outcome"] not in ("win", "loss"):
                continue
            b_fork.append(f)
            b_idx.append(br["succ"])
            b_y.append(1.0 if br["outcome"] == "win" else 0.0)
            b_battle.append(r["base"])
            b_name.append(nm)
    tab = dict(fork=np.asarray(b_fork), idx=np.asarray(b_idx),
               y=np.asarray(b_y, np.float32), battle=np.asarray(b_battle),
               name=np.asarray(b_name))
    by_fork = {}
    for k, f in enumerate(tab["fork"]):
        by_fork.setdefault(int(f), []).append(k)
    pa, pb, py, pf = [], [], [], []
    for f, ks in by_fork.items():
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                i, j = ks[x], ks[y]
                pa.append(i)
                pb.append(j)
                py.append(1.0 if tab["y"][i] > tab["y"][j]
                          else (0.0 if tab["y"][i] < tab["y"][j] else 0.5))
                pf.append(f)
    pairs = dict(a=np.asarray(pa), b=np.asarray(pb), y=np.asarray(py, np.float32),
                 fork=np.asarray(pf))
    return tab, pairs


# ----------------------------------------------------------------- heads
def wide_head(in_dim, hidden=256):
    import torch
    return torch.nn.Sequential(
        torch.nn.LayerNorm(in_dim),
        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))


def fit_head(head, X, Y, tr_rows, va_rows, pairs, p_tr, p_va, *, rank_coef=0.0,
             lr=1e-3, batch=1024, steps=20000, patience=40, seed=20260918, label=""):
    """`refit.fit2`'s recipe — Adam, val every 25 steps, early stop on the VALIDATION objective."""
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    r = np.random.default_rng(seed)
    bce = torch.nn.BCEWithLogitsLoss()
    ntr = p_tr[pairs["y"][p_tr] != 0.5] if len(p_tr) else p_tr
    nva = p_va[pairs["y"][p_va] != 0.5] if len(p_va) else p_va
    best, best_sd, since, step = float("inf"), None, 0, 0
    t0 = time.time()
    for step in range(1, steps + 1):
        sel = tr_rows[r.integers(0, len(tr_rows), size=min(batch, len(tr_rows)))]
        loss = bce(head(Xt[sel]).reshape(-1), Yt[sel])
        if rank_coef > 0 and len(ntr):
            ps = ntr[r.integers(0, len(ntr), size=min(batch, len(ntr)))]
            za = head(Xt[pairs["a"][ps]]).reshape(-1)
            zb = head(Xt[pairs["b"][ps]]).reshape(-1)
            lab = torch.as_tensor(pairs["y"][ps])
            d = torch.where(lab > 0.5, za - zb, zb - za)
            loss = loss + rank_coef * torch.nn.functional.softplus(-d).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                v = float(bce(head(Xt[va_rows]).reshape(-1), Yt[va_rows]))
                if rank_coef > 0 and len(nva):
                    za = head(Xt[pairs["a"][nva]]).reshape(-1)
                    zb = head(Xt[pairs["b"][nva]]).reshape(-1)
                    lab = torch.as_tensor(pairs["y"][nva])
                    dd = torch.where(lab > 0.5, za - zb, zb - za)
                    v += rank_coef * float(torch.nn.functional.softplus(-dd).mean())
            if v < best - 1e-6:
                best, since = v, 0
                best_sd = {k: t.detach().clone() for k, t in head.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    if best_sd is not None:
        head.load_state_dict(best_sd)
    head.eval()
    info = {"fit": label, "rank_coef": rank_coef, "steps_run": step, "best_val": best,
            "n_train_rows": int(len(tr_rows)), "n_val_rows": int(len(va_rows)),
            "n_train_nontied_pairs": int(len(ntr)), "wall_s": time.time() - t0}
    print(f"  [fit {label}] {step} steps, best val {best:.5f}, "
          f"{len(tr_rows)} train rows, {len(ntr)} train pairs ({info['wall_s']:.0f}s)", flush=True)
    return head, info


def head_V(head, X, batch=8192):
    import torch
    out = np.empty(len(X), np.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            j = min(len(X), i + batch)
            out[i:j] = torch.sigmoid(head(torch.as_tensor(X[i:j])).reshape(-1)).numpy()
    return out


# ----------------------------------------------------------------- the guard
def guard_frame(tree, model, n_per_opp=120, lo=4, hi=10, seed=3):
    """`refit.conditioning_frame`'s frame, with the PRE-POOL tensor captured alongside.

    Same files, same sampling, same LABEL ORIENTATION (1 = scripted bot, the published sign).
    The pooled half is cross-checked against `refit.conditioning_frame` itself in `main`.
    """
    import torch
    rng = np.random.default_rng(seed)
    plan = json.load(open(os.path.join(tree, "plan.json")))
    kinds = {it["key"]: it["kind"] for it in plan["items"]}
    P, PRE, cls = [], [], []
    tap = PrePoolTap(model)
    with tap:
        for opp, kind in kinds.items():
            d = os.path.join(tree, opp)
            if not os.path.isdir(d):
                continue
            files = [f for f in sorted(os.listdir(d)) if f.endswith("_summary.json")]
            rng.shuffle(files)
            got = 0
            for f in files:
                if got >= n_per_opp:
                    break
                base = os.path.join(d, f[: -len("_summary.json")])
                try:
                    summ = json.load(open(base + "_summary.json"))
                    npz = np.load(base + "_states.npz")
                except Exception:  # noqa: BLE001
                    continue
                obs = np.asarray(npz["obs"], np.float32)
                mk = np.asarray(npz["action_mask"]).astype(np.float32)[:, :11]
                keep = [k for k, iv in enumerate(summ.get("invocations", []))
                        if lo <= int(iv.get("turn", 0)) <= hi and k < len(obs)]
                if not keep:
                    continue
                k = int(rng.choice(keep))
                d_in = {"observation": torch.as_tensor(obs[k:k + 1]),
                        "action_mask": torch.as_tensor(mk[k:k + 1])}
                with torch.no_grad():
                    model.policy.get_distribution(d_in)
                    P.append(model.policy.features_extractor.stash.value_pooled.numpy()[0].copy())
                    PRE.append(tap.take()[0])
                cls.append(0 if kind == "sentinel" else 1)
                got += 1
    return np.asarray(P, np.float32), np.asarray(PRE, np.float32), np.asarray(cls)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-forks", required=True)
    ap.add_argument("--eval-forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--guard-tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-coef", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--cond-n", type=int, default=120)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    from forks import load_model
    import refit
    from refit import build_table, read_head, paired_delta, auc
    from score_forks import branch_index, dataset_stats
    from agents.model.aux_value_heads import WinProbHead

    model = load_model(args.snapshot)
    orig_sd = {k: v.detach().clone()
               for k, v in model.policy.features_extractor.win_head.state_dict().items()}
    rep = {"snapshot": args.snapshot, "train_forks": args.train_forks,
           "eval_forks": args.eval_forks, "guard_tree": args.guard_tree,
           "rank_coef": args.rank_coef, "seed": args.seed}

    # ---------------- TRAIN: the branched AlphaGo set ------------------------------------
    print("=== TRAIN set (branched, uniform depth)", flush=True)
    rows, succ, smask = load_rows(args.train_forks, TRAIN_BRANCHES)
    tab, pairs = train_table(rows, TRAIN_BRANCHES)
    print(f"[train] {len(rows)} forks, {len(succ)} successors, {len(tab['y'])} branch rows, "
          f"{len(pairs['y'])} pairs, {int((pairs['y']!=0.5).sum())} non-tied", flush=True)
    pooled, prepool, V0_tr = forward_feats(model, succ, smask)
    # VACUITY CHECK on the pre-pool tap. A hook that never fired, or that captured a tensor of the
    # wrong rank, would give fit (d) a silently degenerate input and its null would be unreadable.
    if prepool.shape[1:] != (12, 128):
        raise SystemExit(f"REFUSED: pre-pool tap shape {prepool.shape}, expected (N, 12, 128)")
    _zero_tok = float((np.abs(prepool).sum(-1) == 0).mean())
    _rowvar = float(prepool.reshape(len(prepool), -1).std(0).mean())
    if _rowvar == 0.0:
        raise SystemExit("REFUSED: the pre-pool tap is constant across states — it captured "
                         "nothing state-dependent, so fit (d) would be a fit on a bias term.")
    print(f"[train] forwarded ({time.time()-t0:.0f}s); pre-pool tap: "
          f"{_zero_tok:.3f} of tokens key-masked to zero, mean per-dim sd {_rowvar:.4f}",
          flush=True)

    X_pool = pooled[tab["idx"]]
    X_pre = np.concatenate([prepool[tab["idx"]].reshape(len(tab["idx"]), -1),
                            X_pool], axis=1).astype(np.float32)
    Y = tab["y"]

    # split BY BATTLE, 80/20
    rng = np.random.default_rng(args.seed)
    battles = np.unique(tab["battle"])
    rng.shuffle(battles)
    n_val = max(1, int(round(0.20 * len(battles))))
    val_b = set(battles[:n_val].tolist())
    is_val = np.array([b in val_b for b in tab["battle"]])
    tr_rows = np.flatnonzero(~is_val)
    va_rows = np.flatnonzero(is_val)
    tr_set, va_set = set(tr_rows.tolist()), set(va_rows.tolist())
    p_tr = np.flatnonzero([a in tr_set and b in tr_set
                           for a, b in zip(pairs["a"], pairs["b"])])
    p_va = np.flatnonzero([a in va_set and b in va_set
                           for a, b in zip(pairs["a"], pairs["b"])])
    rand_rows = np.flatnonzero(tab["name"] == "rand")
    tr_rand = np.intersect1d(tr_rows, rand_rows)
    va_rand = np.intersect1d(va_rows, rand_rows)
    per_branch_cap = {}
    for nm in TRAIN_BRANCHES:
        cap = sum(1 for r in rows if r["branches"][nm]["capped"])
        nos = sum(1 for r in rows if r["branches"][nm]["succ"] is None)
        per_branch_cap[nm] = {"capped": cap, "no_successor": nos, "n_forks": len(rows),
                              "cap_rate": cap / max(1, len(rows))}
    rep["train"] = {
        "n_forks": len(rows), "n_successors": int(len(succ)),
        "n_branch_rows": int(len(tab["y"])), "n_pairs": int(len(pairs["y"])),
        "n_nontied_pairs": int((pairs["y"] != 0.5).sum()),
        "tie_rate": float((pairs["y"] == 0.5).mean()) if len(pairs["y"]) else None,
        "base_rate": float(Y.mean()), "n_battles": int(len(battles)),
        "split": {"train_rows": int(len(tr_rows)), "val_rows": int(len(va_rows)),
                  "train_pairs": int(len(p_tr)), "val_pairs": int(len(p_va)),
                  "train_rand_rows": int(len(tr_rand)), "val_rand_rows": int(len(va_rand))},
        "per_branch": per_branch_cap,
        "branch_win_rate": {nm: float(Y[tab["name"] == nm].mean()) for nm in TRAIN_BRANCHES},
        "turn_median": float(np.median([r["turn"] for r in rows])),
        "depth_frac_mean": float(np.mean([r.get("depth_frac", 0.0) for r in rows])),
        "n_legal_mean": float(np.mean([r["n_legal"] for r in rows])),
        "gap_median": float(np.median([r["gap"] for r in rows])),
        "prepool_tap": {"shape": list(prepool.shape[1:]),
                        "fraction_tokens_key_masked": _zero_tok,
                        "mean_per_dim_sd": _rowvar},
    }

    # ---------------- the fits ------------------------------------------------------------
    print("=== FITS", flush=True)
    fits, fitlog = {}, []

    def warm():
        h = WinProbHead()
        h.load_state_dict(orig_sd)
        return h

    specs = [
        ("a_winprob_rand_warm", warm, X_pool, tr_rand, va_rand, 0.0),
        ("b_winprob_paired_warm", warm, X_pool, tr_rows, va_rows, 0.0),
        ("c_wide_pooled", lambda: wide_head(X_pool.shape[1]), X_pool, tr_rows, va_rows, 0.0),
        ("d_wide_prepool", lambda: wide_head(X_pre.shape[1]), X_pre, tr_rows, va_rows, 0.0),
        ("e_wide_pooled_rank", lambda: wide_head(X_pool.shape[1]), X_pool, tr_rows, va_rows,
         args.rank_coef),
        ("a0_winprob_rand_fresh", WinProbHead, X_pool, tr_rand, va_rand, 0.0),
        ("c0_winprob_paired_fresh", WinProbHead, X_pool, tr_rows, va_rows, 0.0),
    ]
    for name, mk, X, trr, var, rc in specs:
        h, info = fit_head(mk(), X, Y, trr, var, pairs, p_tr, p_va, rank_coef=rc,
                           lr=args.lr, batch=args.batch, steps=args.steps, seed=args.seed,
                           label=name)
        fits[name] = (h, X is X_pre)
        fitlog.append(info)
    rep["fitlog"] = fitlog

    # ---------------- READ: the contested fork set ---------------------------------------
    print("=== READ set (contested, forks.py verbatim)", flush=True)
    e_rows, e_succ, e_smask = refit.load_forks(args.eval_forks)
    e_pooled, e_prepool, e_V0 = forward_feats(model, e_succ, e_smask)
    e_tab, e_pairs = build_table(e_rows, e_pooled, e_V0)
    b_idx = branch_index(e_rows)
    if len(b_idx) != len(e_tab["y"]):
        raise SystemExit(f"REFUSED: branch table {len(e_tab['y'])} != {len(b_idx)}")
    if not np.allclose(e_tab["V0"], e_V0[b_idx]):
        raise SystemExit("REFUSED: branch-row / successor index disagreement")
    rep["eval_alignment"] = {
        "n_branch_rows": int(len(b_idx)), "n_successors": int(len(e_succ)),
        "rows_where_branch_index_differs_from_position":
            int((b_idx != np.arange(len(b_idx))).sum()),
        "max_offset": int((b_idx - np.arange(len(b_idx))).max())}
    rep["eval"] = dataset_stats(e_rows, e_tab)
    rep["eval"]["n_pairs"] = int(len(e_pairs["y"]))
    rep["eval"]["n_nontied_pairs"] = int((e_pairs["y"] != 0.5).sum())
    rep["eval"]["tie_rate"] = float((e_pairs["y"] == 0.5).mean())

    Ex_pool = e_pooled[b_idx]
    Ex_pre = np.concatenate([e_prepool[b_idx].reshape(len(b_idx), -1), Ex_pool],
                            axis=1).astype(np.float32)
    allp = np.arange(len(e_pairs["y"]))
    V = {"original": e_V0[b_idx]}
    for name, (h, uses_pre) in fits.items():
        V[name] = head_V(h, Ex_pre if uses_pre else Ex_pool)
    rep["reads"] = {}
    rep["deltas_vs_original"] = {}
    for name in ["original"] + list(fits):
        r = read_head(V[name], e_tab, e_pairs, allp, name)
        rep["reads"][name] = r
        print(f"  READ {name:24s} pairwise {r['pairwise_acc']:.4f} "
              f"[{r['pairwise_acc_ci'][0]:.4f}, {r['pairwise_acc_ci'][1]:.4f}] "
              f"n_nontied {r['n_nontied']} sep {r['sep_ratio']:.3f} "
              f"ECE {r['ece']:.4f} Brier {r['brier']:.4f}", flush=True)
        if name != "original":
            d = paired_delta(V[name], V["original"], e_pairs, allp)
            rep["deltas_vs_original"][name] = d
            print(f"       paired vs original {d['delta']:+.4f} "
                  f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] "
                  f"{'DETECTED' if d['detected'] else 'NOT DETECTED'}", flush=True)
    rep["deltas_d_minus_c"] = paired_delta(V["d_wide_prepool"], V["c_wide_pooled"],
                                           e_pairs, allp)
    rep["deltas_e_minus_c"] = paired_delta(V["e_wide_pooled_rank"], V["c_wide_pooled"],
                                           e_pairs, allp)

    # ---------------- the conditioning guard ---------------------------------------------
    print("=== GUARD (cond.opp_class_auc.t4_10)", flush=True)
    Pc, PREc, cls = guard_frame(args.guard_tree, model, n_per_opp=args.cond_n)
    Pc_ref, cls_ref = refit.conditioning_frame(args.guard_tree, model, n_per_opp=args.cond_n)
    agree = (Pc.shape == Pc_ref.shape and np.allclose(Pc, Pc_ref)
             and np.array_equal(cls, cls_ref))
    Xc_pre = np.concatenate([PREc.reshape(len(PREc), -1), Pc], axis=1).astype(np.float32)
    guard = {"frame_matches_refit_conditioning_frame": bool(agree),
             "n": int(len(cls)), "n_bot": int(np.sum(cls == 1)),
             "orientation": "label 1 = scripted bot (the published sign)"}
    orig_head = model.policy.features_extractor.win_head
    guard["original"] = float(auc(head_V(orig_head, Pc), cls))
    for name, (h, uses_pre) in fits.items():
        guard[name] = float(auc(head_V(h, Xc_pre if uses_pre else Pc), cls))
    for k, v in guard.items():
        if isinstance(v, float):
            print(f"  guard {k:24s} opp_class AUC t4-10 = {v:.4f}", flush=True)
    rep["conditioning_guard"] = guard

    # ---------------- save ----------------------------------------------------------------
    for name, (h, _p) in fits.items():
        torch.save({"win_head": h.state_dict(), "fit": name},
                   os.path.join(args.out, f"head_{name}.pt"))
        # `--leaf-head` loads a BARE state_dict whose keys match the checkpoint's win_head.
        if isinstance(h, WinProbHead):
            torch.save(h.state_dict(), os.path.join(args.out, f"leafhead_{name}.pt"))
    torch.save(orig_sd, os.path.join(args.out, "leafhead_original.pt"))
    rep["wall_s"] = time.time() - t0
    json.dump(rep, open(os.path.join(args.out, "fit_report.json"), "w"), indent=1)
    print(f"[fit] DONE in {rep['wall_s']:.0f}s -> {args.out}/fit_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
