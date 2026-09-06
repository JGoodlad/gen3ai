"""CORRECTION 1, made auditable: what does the TRAINING PATH resolve each teacher to?

`main/train/model_build.py` loads a `--distill-teacher` by calling
`agents.training.fixed_opponent_pool._resolve_zip_and_config(teacher_path, None)`. Every fold in
both eras named a RUN DIRECTORY (verified from each fold's recorded `cli_args.distill_teacher`),
so the resolver's directory rung applies: `best_model/best_model.zip` -> `final_model.zip` ->
`best_model.zip`.

This script IMPORTS that resolver (never re-implements it) and prints, per teacher, the resolved
zip + config and the sha256 of both the resolved zip and the file content_locality scored, so the
correction is a measured file-identity difference rather than an assertion.

Run (gen tree; the era tree has a byte-identical resolver, checked separately):
  python resolve_teachers.py [out.json]
"""
import hashlib
import json
import os
import sys

from agents.training.fixed_opponent_pool import _resolve_zip_and_config

MD = "/home/goodlad/dev/gen3ai/models"

GEN_UNF = [f"ai_v9_{n}_R5F{p}_0831" for n, p in
           zip([92, 94, 96, 98, 100, 102, 104, 106],
               ["00", "02", "04", "06", "08", "10", "12", "14"])]
GEN_FUND = [f"ai_v9_{n}_R5FUND{p}_0901" for n, p in
            zip([120, 122, 124, 126, 128, 130, 132, 134],
                ["00", "02", "04", "06", "08", "10", "12", "14"])]
V8 = ["ai_v8_09_pool10_exploiter_0723",
      "ai_v8_06_semistall_3team_exploiter_0722",
      "ai_v8_13_defensive10_exploiter_0725"]
# The two PARENTS stay exactly as the folds loaded them; listed here only so the table is complete.
PARENTS = {"gen_parent": "ai_v9_59_R2ACTION_0827/final_model.zip",
           "v8_parent": "ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
           "gen_true_origin (rev-1 final)": "ai_v9_29_rev1_0823/final_model.zip"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out_path=None):
    rows = []
    for group, runs in (("gen_unfunded_R5F", GEN_UNF), ("gen_funded_R5FUND", GEN_FUND),
                        ("v8", V8)):
        for r in runs:
            d = os.path.join(MD, r)
            zip_path, cfg_path, base = _resolve_zip_and_config(d, None)
            old = None
            for cand in ("final_model.zip", "final_model_interrupted.zip"):
                if os.path.isfile(os.path.join(d, cand)):
                    old = os.path.join(d, cand)
                    break
            s_new = sha256(zip_path)
            s_old = sha256(old) if old else None
            rows.append({"group": group, "run": r,
                         "resolved_zip": os.path.relpath(zip_path, MD),
                         "resolved_config": os.path.relpath(cfg_path, MD),
                         "resolved_sha256": s_new,
                         "content_locality_scored": os.path.relpath(old, MD) if old else None,
                         "scored_sha256": s_old,
                         "identical": (s_new == s_old)})
            print(f"  {r:44s} -> {rows[-1]['resolved_zip']:34s} "
                  f"sha {s_new[:12]}   was {os.path.basename(old) if old else '-':30s} "
                  f"sha {(s_old or '')[:12]}  {'SAME' if s_new == s_old else 'DIFFERENT'}",
                  flush=True)
    n_diff = sum(1 for r in rows if not r["identical"])
    print(f"\n  {n_diff}/{len(rows)} teachers resolve to a DIFFERENT file than content_locality "
          f"scored", flush=True)
    print("\n  parents (unchanged, exactly as the folds loaded them):", flush=True)
    par = {}
    for k, rel in PARENTS.items():
        p = os.path.join(MD, rel)
        par[k] = {"path": rel, "sha256": sha256(p)}
        print(f"    {k:32s} {rel:60s} sha {par[k]['sha256'][:12]}", flush=True)
    out = {"resolver": "agents.training.fixed_opponent_pool._resolve_zip_and_config(run_dir, None)"
                       " — IMPORTED, the same call main/train/model_build.py makes",
           "teachers": rows, "parents": par, "n_different": n_diff}
    if out_path:
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
