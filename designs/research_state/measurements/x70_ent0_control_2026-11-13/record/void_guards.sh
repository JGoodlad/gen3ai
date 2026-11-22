#!/usr/bin/env bash
# X70 PREREG §8 VOID guards, re-executed per fork at the read (2026-11-22), read-only on models/ and the job dir.
# The window chain ran its own end checks at each fork's end (record/chain/); this is the reading unit's re-run.
# Adapted from CE-3d's record/void_guards.sh (the magnet rows are X70's: magnet_kl_coef PRESENT and 0.0, no sidecar).
# Usage: void_guards.sh <checkout at main> <scratch dir> > void_guards.txt   (CPU only: no lease, nice 19 on the agent cores)
set -u
WT=$1
SCRATCH=$2
export PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CPUS=$("$PY" -m utils.core_split agent-cpus)
LOW=(nice -n 19 taskset -c "$CPUS")
M=/home/goodlad/dev/gen3ai/models
J=/home/goodlad/.claude/jobs/x35
STAGE=$WT/designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0
cd "$WT" || exit 2
echo "# X70 §8 VOID guards; code: $(git -C "$WT" rev-parse HEAD) (main); agent cpus $CPUS; $(date -Is)"
for s in 7331 7332 7333 7334; do
  RUN=rb_x35_x70ent0_s$s; RD=$M/$RUN
  echo; echo "================================================================ $RUN"
  echo "--- [1] completeness (main.ops.arm_chain.armend.completeness on the launcher log + the run dir; rc from the wrapper's END line)"
  grep -h "END $RUN launcher_rc=" "$J/status.txt" | tail -2 | cut -c1-200
  "${LOW[@]}" "$PY" - "$RD" "$J/$RUN.launcher.log" "$J/status.txt" <<'PYE'
import json, re, sys
from pathlib import Path
from main.ops.arm_chain import armend
rd, log, st = sys.argv[1:4]
run = Path(rd).name
rc = int(re.findall(rf"END {run} launcher_rc=(-?\d+)", Path(st).read_text(errors="replace"))[-1])
summ = armend.parse_launcher_log(Path(log).read_text(errors="replace"))
print("launcher_rc", rc, "| last_step", summ.last_step, "| fps", summ.fps, "| restarts", summ.restarts, "| pins named", summ.pins)
print("metadata.json num_timesteps:", json.load(open(rd + "/metadata.json")).get("num_timesteps"))
print("checkpoints on disk:", armend.checkpoint_steps(rd))
print(armend.completeness(rd, 300_000_170, rc, summ))
PYE
  echo "rc=$?"
  echo "--- [2] K9(b) at the arm's end (python -m main.x35.kit k9b --run-dir; no record written)"
  "${LOW[@]}" "$PY" -m main.x35.kit k9b --run-dir "$RD"; echo "rc=$?"
  echo "--- [3] the launcher log's pin, and the fork role"
  grep -n -- "--pin-commit\|argv validated against PINNED" "$J/$RUN.launcher.log" | cut -c1-200
  echo "distinct --pin-commit shas in the launcher log: $(grep -o -- '--pin-commit [0-9a-f]\{40\}' "$J/$RUN.launcher.log" | sort -u | tr '\n' ' ')"
  [ "$(grep -o -- '--pin-commit [0-9a-f]\{40\}' "$J/$RUN.launcher.log" | sort -u)" = "--pin-commit $PIN" ] && echo "pin: ONLY the training pin $PIN" || echo "pin: FAIL"
  echo "the kit's GO-time dry run ($J/$RUN.go.dryrun.txt):"; grep -n "role  \|pin " "$J/$RUN.go.dryrun.txt" | cut -c1-200 | head -4
  "${LOW[@]}" "$PY" -m main.sidecar_audit "$RD" 2>&1 | head -12; echo "rc=$?"
  "${LOW[@]}" "$PY" - "$RD" <<'PYE'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); lin = m["lineage"]; fp = lin["fork_parent"]
print("lineage: role", lin["role"], "| parent run", fp["run_dir"], "| parent step", fp["num_timesteps"], "| parent sha256", fp["sha256"][:16], "| fork_step", lin["fork_step"])
ok = lin["role"] == "fork" and fp["run_dir"].endswith("/rb_x26_s2601") and fp["sha256"].startswith("814823c5") and lin["fork_step"] == 280000170
print("FORK of .../rb_x26_s2601 (P* sha256 814823c5...):", "yes" if ok else "FAIL")
PYE
  echo "--- [4] the post-spawn pool-seed check (python -m main.x35.kit pool-seed-check, re-run now on the finished run)"
  echo "at the launch: $(cat "$J/$RUN.pool_seed_check.txt")"
  "${LOW[@]}" "$PY" -m main.x35.kit pool-seed-check --arm x70 --run-dir "$RD" --call "$STAGE/reference_population_production.json" \
     --stage "$STAGE" --log "$J/$RUN.launcher.log" --log "$RD/launcher_child.full.log"; echo "rc=$?"
  echo "--- [5] the lever in the record (ent_coef 0, no magnet)"
  grep -h "$RUN early check 1/2" "$J/status.txt" | cut -c1-330
  echo "[MAGNET] banner lines in the full child log (X70 expects none): $(grep -c '\[MAGNET\]' "$RD/launcher_child.full.log")"
  echo "hyperparameter-table ent_coef rows in the full child log:"; grep -n "ent_coef   |" "$RD/launcher_child.full.log" | sort -t'|' -k3 -u | head
  echo "VOID strings (ent_coef 0.05 / 0.1 / 0.0125 rows, [ModelVersion] FATAL): $(grep -c 'ent_coef   | 0.05 \|ent_coef   | 0.1 \|ent_coef   | 0.0125 \|\[ModelVersion\] FATAL' "$RD/launcher_child.full.log") line(s)"
  "${LOW[@]}" "$PY" - "$RD" <<'PYE'
import glob, json, os, sys
rd = sys.argv[1]; m = json.load(open(rd + "/metadata.json")); c = json.load(open(rd + "/model_config.json"))
row = dict(metadata_ent_coef=m["ent_coef"], cli_ent_coef=m["cli_args"]["ent_coef"], cfg_has_magnet_kl_coef="magnet_kl_coef" in c,
           cfg_magnet_kl_coef=c.get("magnet_kl_coef"), cfg_magnet_refresh_steps=c.get("magnet_refresh_steps"),
           sidecar=os.path.isfile(rd + "/final_model.magnet.pt"), any_magnet_pt=sorted(glob.glob(rd + "/**/*.magnet.pt", recursive=True)),
           cli_seed=m["cli_args"]["seed"], n_epochs=m.get("n_epochs", m["cli_args"].get("n_epochs")))
print(row)
ok = (row["metadata_ent_coef"] == 0.0 and row["cli_ent_coef"] == 0.0 and row["cfg_has_magnet_kl_coef"]
      and row["cfg_magnet_kl_coef"] == 0.0 and not row["sidecar"] and not row["any_magnet_pt"])
print("lever in the record (ent_coef 0.0 twice; magnet_kl_coef present and 0.0; no magnet sidecar):", "yes" if ok else "FAIL")
PYE
  echo "--- [6] python -m main.dose $RUN"
  "${LOW[@]}" "$PY" -m main.dose "$M/$RUN"; echo "rc=$?"
  echo "the chain's own dose read at the fork's end ($J/x70/$RUN.dose.txt):"; cat "$J/x70/$RUN.dose.txt"
done
echo; echo "================================================================ the plan's guards (PREREG §5.2), re-run at the read on every run of the three crosses"
X=/home/goodlad/.cache/gen3ai/x70_cross/20261122T131756
D=$WT/designs/research_state/measurements/x70_ent0_control_2026-11-13/cross
for c in anchor c0 x67; do
  mkdir -p "$SCRATCH/plan_$c"
  "${LOW[@]}" "$PY" "$D/plan_cross.py" --cross "$c" --out "$SCRATCH/plan_$c/cells_plan.json" --shas "$SCRATCH/plan_$c/shas.json"; echo "plan_cross --cross $c rc=$?"
  cmp "$SCRATCH/plan_$c/cells_plan.json" "$X/$c/cells_plan.json" && echo "cells_plan.json ($c): byte-identical to the one the cross was played from"
  cmp "$SCRATCH/plan_$c/shas.json" "$X/$c/shas.json" && echo "shas.json ($c): byte-identical (every final's sha256 is the one the cross was played on)"
done
