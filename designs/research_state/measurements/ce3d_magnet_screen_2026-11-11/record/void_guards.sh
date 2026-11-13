#!/usr/bin/env bash
# CE-3d PREREG §8 VOID guards, executed per fork at the read (2026-11-12/13), read-only on models/ and the job dir.
# Usage: void_guards.sh <checkout at main> > void_guards.txt     (CPU only: no lease, nice 19 on the agent cores)
set -u
WT=$1
export PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CPUS=$("$PY" -m utils.core_split agent-cpus)
LOW=(nice -n 19 taskset -c "$CPUS")
M=/home/goodlad/dev/gen3ai/models
J=/home/goodlad/.claude/jobs/x35
STAGE=$WT/designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0
cd "$WT" || exit 2
echo "# CE-3d §8 VOID guards; code: $(git -C "$WT" rev-parse HEAD) (main); agent cpus $CPUS; $(date -Is)"
for s in 7321 7322 7323; do
  RUN=rb_x35_ce3dswap_s$s; RD=$M/$RUN
  echo; echo "================================================================ $RUN"
  echo "--- [1] completeness (main.ops.arm_chain.armend.completeness on the launcher log + the run dir; rc from the wrapper's END line)"
  grep -h "END $RUN launcher_rc=" "$J/status.txt" | tail -1
  "${LOW[@]}" "$PY" - "$RD" "$J/$RUN.launcher.log" "$J/status.txt" <<'PYE'
import re, sys
from pathlib import Path
from main.ops.arm_chain import armend
rd, log, st = sys.argv[1:4]
run = Path(rd).name
rc = int(re.findall(rf"END {run} launcher_rc=(-?\d+)", Path(st).read_text(errors="replace"))[-1])
summ = armend.parse_launcher_log(Path(log).read_text(errors="replace"))
print("launcher_rc", rc, "| last_step", summ.last_step, "| fps", summ.fps, "| restarts", summ.restarts, "| pins named", summ.pins)
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
  grep -h "arm:$RUN: PRECHECKED -> GOING\|arm:$RUN: PENDING -> PRECHECKED" "$J/status.txt" | cut -c1-260
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
  "${LOW[@]}" "$PY" -m main.x35.kit pool-seed-check --arm ce3d_swap --run-dir "$RD" --call "$STAGE/reference_population_production.json" \
     --stage "$STAGE" --log "$J/$RUN.launcher.log" --log "$RD/launcher_child.full.log"; echo "rc=$?"
  echo "--- [5] both halves of the lever in the record"
  grep -h "arm:$RUN: SPAWNED -> BANNER_OK" "$J/status.txt" | cut -c1-330
  grep -n "\[MAGNET\] --magnet-kl-coef" "$RD/launcher_child.full.log" | cut -c1-260
  echo "hyperparameter-table ent_coef rows in the full child log:"; grep -n "ent_coef   |" "$RD/launcher_child.full.log" | sort -t'|' -k3 -u | head
  echo "VOID strings (ent_coef 0.05 / 0.1 / 0.0125 rows, [ModelVersion] FATAL): $(grep -c 'ent_coef   | 0.05 \|ent_coef   | 0.1 \|ent_coef   | 0.0125 \|\[ModelVersion\] FATAL' "$RD/launcher_child.full.log") line(s)"
  "${LOW[@]}" "$PY" - "$RD" <<'PYE'
import json, os, sys
rd = sys.argv[1]; m = json.load(open(rd + "/metadata.json")); c = json.load(open(rd + "/model_config.json"))
row = dict(metadata_ent_coef=m["ent_coef"], cli_ent_coef=m["cli_args"]["ent_coef"], cfg_magnet_kl_coef=c.get("magnet_kl_coef"),
           cfg_magnet_refresh_steps=c.get("magnet_refresh_steps"), sidecar=os.path.isfile(rd + "/final_model.magnet.pt"),
           cli_seed=m["cli_args"]["seed"])
print(row)
ok = (row["metadata_ent_coef"] == 0.0 and row["cli_ent_coef"] == 0.0 and row["cfg_magnet_kl_coef"] == 0.05
      and row["cfg_magnet_refresh_steps"] == 2_000_000 and row["sidecar"])
print("lever in the record:", "yes" if ok else "FAIL")
PYE
  echo "--- [6] python -m main.dose $RUN"
  "${LOW[@]}" "$PY" -m main.dose "$M/$RUN"; echo "rc=$?"
  echo "--- [7] python -m main.ops.tb_read $RUN --tag magnet/refreshes"
  "${LOW[@]}" "$PY" -m main.ops.tb_read "$RD" --tag magnet/refreshes; echo "rc=$?"
done
