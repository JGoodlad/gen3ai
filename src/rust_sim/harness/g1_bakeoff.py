"""G1 — the offline reducer bake-off (design_pair_reduction.md §7 G1, adopted §8.1).

Ranks the pair-reduction ladder by how well each rung's REDUCED per-defender features linearly
predict the better-line beam's preferred SWITCH at real loss decisions, on a frozen gen-5
checkpoint. Pre-registered reading: a rung that cannot predict the beam's switch from
ground-truth cells with a linear probe will not learn to once in the loop.

Arms (v1, honest about what each is):
  R0        — linear probe on the LEGACY reduced per-mon row (the 12 hard-max stats + CB tail 3).
  R1        — linear probe on Contract-W belief_mean output: [E_α[cell](6), E_α[cell²](6), Σα·w(1)]
              per defender, α = w/Σw over the K believed candidates.
  R1+R0     — both concatenated (does the coherent marginal ADD to the shipped stats?).
  SKYLINE   — linear probe on the RAW un-reduced cells (K×6 per defender + w) — the linear-readout
              upper bound ANY reducer can reach; brackets R2L without probing a random-init φ
              (which would under-read a learned rung — stated, not hidden).

Decision rule (pre-registered): R1 (or R1+R0) beating R0 beyond the bootstrap spread ⇒ the
coherence rungs carry real switch signal → delivery wiring + G7 proceed on the W-rungs;
SKYLINE ≫ R1 ⇒ the un-reduced grid holds MORE than any single-α mixture → R2L/deepsets justified.

Stages (separate processes; targets is hours of beam compute, probe is seconds):
  --stage targets : scan losses → better_line each → keep beam-prefers-a-switch decisions →
                    <out>/g1_targets.jsonl (full better_line dict kept per row).
  --stage probe   : re-load targets, capture the op's internal (w, cells) at each decision via a
                    capture-reducer monkeypatch, fit the four probes, report accuracy ± bootstrap.

Usage:
  python src/rust_sim/harness/g1_bakeoff.py --run <run_dir> --stage targets --limit 200 --depth 2
  python src/rust_sim/harness/g1_bakeoff.py --run <run_dir> --stage probe
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def _extract_alt_action(bl: dict):
    """The beam's divergence-ply action, defensively (schema may grow fields)."""
    ba = bl.get("best_alternative") or {}
    for k in ("action", "action_index", "a"):
        if isinstance(ba.get(k), int):
            return ba[k]
    line = bl.get("line") or []
    if line and isinstance(line[0], dict):
        for k in ("action", "our_action", "a"):
            if isinstance(line[0].get(k), int):
                return line[0][k]
    return None


def stage_targets(a):
    from main.prober.session import ProbeSession

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "g1_targets.jsonl")
    done = set()
    if os.path.exists(out_path):                       # resumable — beam hours are precious
        with open(out_path) as f:
            done = {(r["battle"], r["inv"]) for r in map(json.loads, f) if "battle" in r}

    with ProbeSession(a.run) as s:
        rows = s.scan(outcome="loss", limit=a.limit, metric="value_drop")
        print(f"[g1] scanned {len(rows)} losses; {len(done)} targets already done", flush=True)
        kept = skipped = failed = 0
        with open(out_path, "a") as f:
            for i, r in enumerate(rows):
                bid, inv = r["id"], r["worst"]["inv"]
                if (bid, inv) in done:
                    continue
                dv = r["worst"].get("delta_v")
                if dv is not None and abs(dv) < a.min_dv:
                    continue
                try:
                    bl = s.better_line(bid, inv, depth=a.depth, confirm_rollouts=0)
                except Exception as e:                 # noqa: BLE001 — one bad battle ≠ dead run
                    failed += 1
                    print(f"[g1] {i} {bid}@{inv} FAILED: {e}", flush=True)
                    continue
                alt = _extract_alt_action(bl)
                is_switch = alt is not None and 0 <= alt <= 5
                row = {"battle": bid, "inv": inv, "scan_dv": dv, "alt_action": alt,
                       "is_switch": bool(is_switch), "better_line": bl}
                f.write(json.dumps(row) + "\n")
                f.flush()
                kept += int(is_switch)
                skipped += int(not is_switch)
                print(f"[g1] {i + 1}/{len(rows)} {bid}@{inv} alt={alt} "
                      f"switch={is_switch} (kept {kept} / non-switch {skipped} / failed {failed})",
                      flush=True)
    print(f"[g1] DONE targets: {kept} switch targets, {skipped} non-switch, {failed} failed "
          f"-> {out_path}", flush=True)


class _CaptureReducer:
    """Monkeypatch stand-in for op.pair_reducer: stashes (w, cells), returns a 1-wide zero
    (satisfies the stash contract without touching physics — the block is untouched either way)."""

    def __init__(self):
        self.grabbed = None
        self.extra_dim = 1

    def __call__(self, w, cells):
        import torch
        self.grabbed = (w.detach().cpu(), cells.detach().cpu())
        return torch.zeros(cells.shape[0], cells.shape[1], 1)


def stage_probe(a):
    import torch
    from agents.model.snapshot import load_foreign_opponent
    from agents.model.pair_reduce import alpha_belief_mean, reduce_with_alpha

    tpath = os.path.join(a.out, "g1_targets.jsonl")
    targets = [r for r in map(json.loads, open(tpath)) if r.get("is_switch")]
    print(f"[g1] {len(targets)} switch targets", flush=True)
    if len(targets) < 40:
        print("[g1] too few targets for a stable probe — gather more first")
        return

    # Model: the same loader the split audit uses (arch-signature gate only — v61 ckpt loads
    # under v62 code since the signature is unchanged). Obs/mask: straight from each trace's
    # states.npz sibling (the trace-triple naming contract; mask = finite post-mask logits,
    # mirroring agents.model.audit_states).
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    ckpt = os.path.join(a.run, "final_model.zip")
    model, _ = load_foreign_opponent(ckpt, current_version=current_model_version(load_mappings()),
                                     device="cpu")
    op = model.policy.features_extractor.damage_op
    cap = _CaptureReducer()
    op.pair_reducer = cap                              # capture-monkeypatch (no physics change)

    feats_r0, feats_r1, feats_sky, labels, legal = [], [], [], [], []
    for r in targets:
        npz_path = r["battle"].replace("_summary.json", "_states.npz")
        if not os.path.exists(npz_path):
            continue
        with np.load(npz_path) as z:
            obs = z["obs"][r["inv"]]
            mask = z["logits"][r["inv"]] > -1e8        # the sampler's finite-threshold contract
        cap.grabbed = None
        ob = {"observation": torch.as_tensor(obs[None]).float(),
              "action_mask": torch.as_tensor(mask[None]).float()}
        with torch.no_grad():
            model.policy.get_distribution(ob)
        if cap.grabbed is None:
            continue
        w, cells = cap.grabbed                         # [1,C], [1,J,C,F]
        raw = op.last_raw_block[0].cpu()               # [660]
        pm = op.per_mon
        r0 = raw[: 6 * pm].view(6, pm)                 # legacy reduced rows [6,12]
        al = alpha_belief_mean(w)
        e1 = reduce_with_alpha(al, cells)[0]           # [J,F]
        e2 = reduce_with_alpha(al, cells.pow(2))[0]    # [J,F]
        prov = (al * w).sum(-1).expand(6, 1)           # [J,1]
        r1 = torch.cat([e1, e2, prov], -1)             # [6,13]
        sky = torch.cat([cells[0].flatten(1),          # [6, C*F]
                         w.expand(6, -1)], -1)         # + w per defender
        feats_r0.append(r0); feats_r1.append(r1); feats_sky.append(sky)
        labels.append(r["alt_action"]); legal.append(mask[:6])
    cap_n = len(labels)
    print(f"[g1] captured {cap_n} decisions", flush=True)

    y = torch.as_tensor(labels)
    lg = torch.as_tensor(np.array(legal))
    arms = {"R0": torch.stack(feats_r0), "R1": torch.stack(feats_r1),
            "R1+R0": torch.cat([torch.stack(feats_r0), torch.stack(feats_r1)], -1),
            "SKYLINE": torch.stack(feats_sky)}

    def fit_probe(X, y, lg, seed):
        g = torch.Generator().manual_seed(seed)
        n = len(y); perm = torch.randperm(n, generator=g)
        tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
        th = torch.zeros(X.shape[-1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([th, b], max_iter=150)
        Xtr, ytr, ltr = X[tr], y[tr], lg[tr]

        def score(Xs, ls):
            s = (Xs * th).sum(-1) + b
            return s.masked_fill(~ls, -1e9)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(score(Xtr, ltr), ytr) \
                + 1e-3 * th.pow(2).sum()
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            return float((score(X[te], lg[te]).argmax(-1) == y[te]).float().mean())

    print(f"\n[g1] RANKING (n={cap_n}, 5 seeds, test accuracy predicting the beam's switch):")
    results = {}
    for name, X in arms.items():
        accs = [fit_probe(X.float(), y, lg, s) for s in range(5)]
        results[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                         "accs": accs, "dim": int(X.shape[-1])}
        print(f"  {name:8s} dim={X.shape[-1]:3d}  acc {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    base = 1.0 / lg.float().sum(-1).mean()
    print(f"  chance ~= {float(base):.3f} (mean 1/#legal-switches)")
    out = os.path.join(a.out, "g1_probe_results.json")
    json.dump({"n": cap_n, "chance": float(base), "arms": results}, open(out, "w"), indent=2)
    print(f"[g1] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=("targets", "probe"), required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-dv", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.run, "g1_bakeoff")
    (stage_targets if a.stage == "targets" else stage_probe)(a)
