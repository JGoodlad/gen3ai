"""v8-era RANK arm of the capacity battery, run from a worktree pinned to the v8 commit.

Uses the CURRENT main checkout's estimators (copied verbatim into ./current_battery/):
capacity_probes.{capture_features,build_fresh_extractor,rank_probe,jsonable} and
audit_states.collect_states. Only probe (a) — effective rank — is run.
"""
import argparse, glob, inspect, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "current_battery"))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-states", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t0 = time.time()

    from agents.model.snapshot import current_model_version, load_model_snapshot
    from agents.observation.state_encoder import load_mappings
    from audit_states import collect_states
    from capacity_probes import (CAPACITY_BATTERY_VERSION, build_fresh_extractor,
                                 capture_features, jsonable, rank_probe)
    from utils.git import get_git_hash

    cfg_path = os.path.join(a.run_dir, "model_config.json")
    cfg = json.load(open(cfg_path))
    sig = inspect.signature(current_model_version).parameters
    toggles = {k: v for k, v in cfg.items() if k in sig and k != "mappings"}
    print(f"[v8cap] {len(toggles)} toggles forwarded to current_model_version", flush=True)
    version = current_model_version(load_mappings(), **toggles)
    model = load_model_snapshot(a.ckpt, env=None, current_version=version, device=a.device)
    policy = model.policy.eval()
    print(f"[v8cap] loaded {a.ckpt}  step {getattr(model, 'num_timesteps', 0):,}", flush=True)

    pattern = os.path.join(a.run_dir, "eval_traces", "**", "*_states.npz")
    obs, masks, coverage = collect_states([pattern], a.max_states, seed=a.seed)
    print(f"[v8cap] {len(obs)} states, obs dim {obs.shape[1]}", flush=True)

    fresh = build_fresh_extractor(policy, seed=a.seed)
    print("[v8cap] capturing trained taps", flush=True)
    feats = capture_features(policy.features_extractor, obs, masks,
                             batch=a.batch, device=a.device)
    print("[v8cap] capturing fresh taps", flush=True)
    feats_f = capture_features(fresh, obs, masks, batch=a.batch, device=a.device)

    report = {
        "meta": {
            "run_dir": a.run_dir, "run_name": os.path.basename(a.run_dir.rstrip("/")),
            "checkpoint": a.ckpt, "model_config": cfg_path,
            "arch_signature": cfg.get("arch_signature"),
            "config_version": cfg.get("config_version"),
            "obs_dim": int(obs.shape[1]),
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
            "n_states": int(len(obs)), "sampling": coverage, "seed": int(a.seed),
            "device": a.device, "worktree_git_hash": get_git_hash(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "probes_run": ["rank"],
            "estimator_source": "current main checkout capacity_probes.py (verbatim copy)",
        },
        "battery_version": CAPACITY_BATTERY_VERSION,
        "rank": {"trained": rank_probe(feats), "fresh": rank_probe(feats_f)},
    }
    report["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    report = jsonable(report)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, allow_nan=False)
    for arm in ("trained", "fresh"):
        print(f"\n-- {arm}")
        for tap, r in report["rank"][arm].items():
            print(f"  {tap:<14}dim={r['dim']:>4}  PR={r['pr']:8.3f}  effrank={r['effrank']:8.3f}"
                  f"  srank99={r['srank99']:6.0f}  n90={r['n90']:6.0f}  n_rows={r['n_rows']}")
    print(f"\nwrote {a.out}  ({report['meta']['runtime_sec']}s)")


if __name__ == "__main__":
    main()
