"""X34 staging: network identity of rb_toy_ent_s4002 at HEAD vs its pin.

  identity.py inputs OUT.npz                       (HEAD) real Rust-core obs rows, seeded
  identity.py run ARGV_FILE ZIP UNUSED INPUTS OUT  build from argv, compare to ZIP, strict-load, forward, save

Run at a checkout with PYTHONPATH=<checkout>/src, CPU, one torch thread (rule 5b: nice 19, taskset -c 5-7,13-15).
The pin side ran on a `git archive 3899277a src data designs` export.
"""
import inspect
import json
import sys
import zipfile

import numpy as np
import torch


def inputs(out):
    from utils.rust_env.fixture_battles import play_rows
    rows = play_rows(6, seed=34, n_teams=12, n_envs=2)
    obs = np.concatenate([o for o, _ in rows])[:256]
    mask = np.concatenate([m for _, m in rows])[:256]
    np.savez(out, obs=obs.astype(np.float32), mask=mask.astype(bool))
    print(f"inputs: {obs.shape} rows from {len(rows)} games -> {out}")


def _args(argv):
    import contextlib
    import io

    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def _version(args, layout, pk):
    from agents.model.model_version import ModelVersion
    sig = set(inspect.signature(ModelVersion.from_layout_and_policy_kwargs).parameters)
    names = ["vf_coef", "opp_belief_aux_coef", "move_belief_coef", "move_belief_latent_coef", "spread_belief_coef",
             "hp_type_belief_coef", "item_belief_coef", "arch_source", "policy_grad_coef", "intent_label_bot_weight",
             "policy_gae_lambda", "diagnostics_every", "opp_intent_coef", "fork_fraction", "fork_branches",
             "fork_contested_gap", "fork_contested_absv", "fork_max_per_battle", "fork_crn", "eval_sentinel_greedy",
             "promote_threshold", "eval_mirrored_pairs", "promotion_sprt", "trunk_precision", "canary_shift_max",
             "capacity_telemetry", "canary_reset_steps", "capacity_cosine_every", "capacity_velocity_every",
             "rank_tripwire", "rank_tripwire_drop"]
    kw = {n: getattr(args, n) for n in names if n in sig and hasattr(args, n)}
    if "opp_intent_coef" in kw:
        kw["opp_intent_coef"] = float(kw["opp_intent_coef"] or 0.0)
    return ModelVersion.from_layout_and_policy_kwargs(layout, pk, **kw)


def _forward(policy, obs, mask):
    policy.set_training_mode(False)
    with torch.no_grad():
        o = {"observation": torch.as_tensor(obs), "action_mask": torch.as_tensor(mask.astype(np.int8))}
        dist = policy.get_distribution(o, action_masks=mask)
        logits = dist.distribution.logits.detach().numpy().copy()
        values = policy.predict_values(o).detach().numpy().copy()
    return logits, values


def run(argv_file, zip_path, _unused, inputs_path, out):
    torch.set_num_threads(1)
    from main.fresh_checkpoint import build_fresh_model
    with open(argv_file) as fh:
        argv = fh.read().split()
    args = _args(argv)
    model, args, pk = build_fresh_model(0, args=args)
    built = model.policy.state_dict()
    with zipfile.ZipFile(zip_path) as z, z.open("policy.pth") as f:
        ck = torch.load(f, map_location="cpu", weights_only=True)
    kb, kc = set(built), set(ck)
    shape_diff = [k for k in kb & kc if tuple(built[k].shape) != tuple(ck[k].shape)]
    n_b = sum(v.numel() for v in built.values())
    n_c = sum(v.numel() for v in ck.values())
    rep = {"argv_file": argv_file, "n_keys_built": len(kb), "n_keys_ckpt": len(kc),
           "only_built": sorted(kb - kc), "only_ckpt": sorted(kc - kb), "shape_diff": shape_diff,
           "numel_built": n_b, "numel_ckpt": n_c,
           "n_params_built": sum(p.numel() for p in model.policy.parameters())}
    model.policy.load_state_dict(ck, strict=True)
    rep["strict_load"] = "OK"
    d = np.load(inputs_path)
    obs, mask = d["obs"], d["mask"]
    lg_a, v_a = _forward(model.policy, obs, mask)
    # the trainer's own resume/fork loader (load_model_snapshot: the version check + the sanitizer)
    from agents.model.snapshot import load_model_snapshot
    version = _version(args, pk["features_extractor_kwargs"]["layout"], pk)
    loaded = load_model_snapshot(zip_path, env=None, current_version=version, device="cpu")
    rep["load_model_snapshot"] = type(loaded).__name__
    lsd = loaded.policy.state_dict()
    rep["load_model_snapshot_bytes_equal_ckpt"] = all(torch.equal(lsd[k], ck[k]) for k in ck) and set(lsd) == kc
    lg_b, v_b = _forward(loaded.policy, obs, mask)
    rep["argvbuilt_vs_loader_logits_maxabs"] = float(np.nanmax(np.abs(np.where(np.isfinite(lg_a), lg_a - lg_b, 0))))
    rep["argvbuilt_vs_loader_values_maxabs"] = float(np.abs(v_a - v_b).max())
    np.savez(out, logits=lg_a, values=v_a, logits_loader=lg_b, values_loader=v_b)
    rep["torch"] = torch.__version__
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        inputs(sys.argv[2])
    else:
        run(*sys.argv[2:7])
