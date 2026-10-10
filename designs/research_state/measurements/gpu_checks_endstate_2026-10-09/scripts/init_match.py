"""Does the golden-learner build at the TRAINER's seed reproduce a launch's FRESH weights? (gpu_checks_endstate,
2026-10-09; CPU.)

    python init_match.py --cfg R --ckpt <run>/final_model_interrupted.zip [--seed 42]

Compares `compile_regions.param_sha256` of every policy parameter of `learner_golden.build_learner(args=<cfg>)`,
built UNPERTURBED at MODEL_SEED = ``--seed`` (the trainer's `--seed` default is 42), against the launch's own
init record (`param_init_sha256`, written by `model_build.construct_fresh_learner` before anything trained and
carried in every checkpoint). All equal => the reproduction builds the launch's starting weights bit for bit.
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cuda_parity_targets import CFG, resolved_args   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import agents.model.parity_probe as pp
    from agents.model.compile_regions import INIT_RECORD_ATTR, param_sha256
    from agents.training import learner_golden as LG

    data = json.loads(zipfile.ZipFile(a.ckpt).read("data"))
    rec = data.get(INIT_RECORD_ATTR)
    if isinstance(rec, dict) and ":serialized:" in rec:
        raise SystemExit("init record is pickled, not plain JSON — cannot compare")
    real, seed0 = pp.perturb_, LG.MODEL_SEED
    pp.perturb_ = lambda *x, **k: None
    LG.MODEL_SEED = a.seed
    try:
        model = LG.build_learner(args=resolved_args(CFG[a.cfg]))
    finally:
        pp.perturb_, LG.MODEL_SEED = real, seed0
    mine = {n: param_sha256(p) for n, p in model.policy.named_parameters()}
    same = [n for n in mine if rec.get(n) == mine[n]]
    diff = [n for n in mine if n in rec and rec[n] != mine[n]]
    only_mine = [n for n in mine if n not in rec]
    only_rec = [n for n in rec if n not in mine]
    print(json.dumps({"cfg": a.cfg, "seed": a.seed, "n_mine": len(mine), "n_record": len(rec), "equal": len(same),
                      "differ": len(diff), "differ_e.g.": diff[:8], "only_mine": only_mine[:8],
                      "only_record": only_rec[:8]}, indent=1))


if __name__ == "__main__":
    main()
