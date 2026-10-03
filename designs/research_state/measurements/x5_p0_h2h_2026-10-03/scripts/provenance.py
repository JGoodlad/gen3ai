"""Provenance of the four banked runs P0 reads (READ-ONLY on models/): which are true replicates, and exactly what
differs. Writes ../provenance.json and prints the table.

Reads, per run: metadata.json (original command, pin history, cli_args, matchup history, torch), model_config.json
(sha256 + comparison), the final snapshot's SB3 `data` block (the EFFECTIVE hyperparameters of the run's last
segment: a resume's cli_args can record a flag the checkpoint then ignores), and the final snapshot's sha256 (the
head-to-head's content hash). Run: PYTHONPATH=<checkout>/src python3 provenance.py [--models /path/to/models]."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

RUNS = {"A": "sizing_A_n48_e10_s1001", "A2": "sizing_A2_n48_e10_s1001", "Ap": "sizing_Ap_n48_e10_s1002",
        "B": "sizing_B_n256_e10_s1001"}
HYPER = ["batch_size", "n_epochs", "n_steps", "learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef",
         "grad_accum_steps", "target_kl", "seed", "num_timesteps", "_n_updates"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/home/goodlad/dev/gen3ai/models")
    a = ap.parse_args()
    from agents.training.fixed_opponent_pool import resolve_model_ref

    out = {}
    for tag, name in RUNS.items():
        d = Path(a.models) / name
        md = json.loads((d / "metadata.json").read_text())
        mc_path = d / "model_config.json"
        ref = resolve_model_ref(str(d))
        data = json.loads(zipfile.ZipFile(ref.zip_path).read("data"))
        pk_sha = hashlib.sha256(json.dumps(data.get("policy_kwargs"), sort_keys=True, default=str).encode()).hexdigest()
        ca = md["cli_args"]
        out[tag] = {
            "run": name, "final_snapshot": ref.zip_path, "rung": ref.rung, "num_timesteps": ref.num_timesteps,
            "final_sha256": sha256(ref.zip_path), "model_config_sha256": sha256(mc_path),
            "policy_kwargs_sha256": pk_sha,
            "original_command": md.get("original_command"),
            "pin": md["pin_history"][0]["git_hash"], "pin_history": md["pin_history"],
            "torch": md.get("torch_version"), "seed": data.get("seed"),
            "effective_hyperparameters": {k: data.get(k) for k in HYPER},
            "n_envs": ca.get("n_envs"), "n_steps_arg": ca.get("n_steps"),
            "resumed": bool(ca.get("model")), "resume_from": ca.get("model"),
            "last_segment_cli_batch_size": ca.get("batch_size"),
            "last_segment_unified_damage": ca.get("unified_damage"), "last_segment_unified_moves": ca.get("unified_moves"),
            "matchup_hash": (md.get("matchup_history") or [{}])[-1].get("hash"),
            "eval_trainee_teams": (((md.get("matchup_history") or [{}])[-1].get("spec") or {}).get("eval_trainee_teams")),
            "run_seed_recorded": (md.get("env_core") or {}).get("run_seed"),
            "core_stamp_commit": ((md.get("env_core") or {}).get("core_stamp") or "").split("commit=")[-1].split(";")[0],
        }
    cfgs = {t: json.loads((Path(a.models) / n / "model_config.json").read_text()) for t, n in RUNS.items()}
    out["_model_config_identical_across_all_four"] = all(json.dumps(c, sort_keys=True) == json.dumps(cfgs["A"], sort_keys=True)
                                                         for c in cfgs.values())
    out["_policy_kwargs_identical_across_all_four"] = len({out[t]["policy_kwargs_sha256"] for t in RUNS}) == 1
    hp = {t: out[t]["effective_hyperparameters"] for t in RUNS}
    keys = [k for k in HYPER if k not in ("num_timesteps", "seed", "_n_updates")]
    out["_effective_hyperparameters_differing"] = {k: {t: hp[t][k] for t in RUNS} for k in keys
                                                   if len({json.dumps(hp[t][k]) for t in RUNS}) > 1}
    Path(__file__).resolve().parent.parent.joinpath("provenance.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    for t in RUNS:
        r = out[t]
        print(f"{t:3s} {r['run']:28s} pin {r['pin'][:8]} seed {r['seed']} n_envs {r['n_envs']} n_steps {r['n_steps_arg']} "
              f"steps {r['num_timesteps']} resumed {r['resumed']} rung {r['rung']} sha {r['final_sha256'][:12]}")
    print("model_config.json identical across all four:", out["_model_config_identical_across_all_four"])
    print("final snapshots' pickled policy_kwargs (extractor / policy architecture) identical across all four:",
          out["_policy_kwargs_identical_across_all_four"])
    print("effective hyperparameters that differ:", out["_effective_hyperparameters_differing"] or "none (batch 2048, 10 epochs, lr 3e-4, accum 32, ...)")


if __name__ == "__main__":
    main()
