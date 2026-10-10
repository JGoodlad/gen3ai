#!/usr/bin/env bash
# ONE job at a time: the order_probe E control and the single-lever reversions, then the per-site MATH restriction
# (site_probe, "none" = its own control). Run from <checkout>/src under scripts/ops/gpu_lock.sh with the lease.
py=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
here=$(dirname "$0")
"$py" -c "import agents, utils, main; print('imports:', agents.__file__, utils.__file__, main.__file__)"
for v in E E-obs_facts E-move_resolution E-op_reduction E-speed_physics E+value_threat_inject; do
    echo "== variant $v $(date +%T)"
    timeout 600 "$py" "$here/order_probe.py" --cfg "$v" --steps cuda_eager,compile 2>&1 | grep -E "^compile worst|Error" | grep -v "Expected all" | cut -c1-400
done
for s in none trunk hopb extra; do
    echo "== site $s $(date +%T)"
    timeout 600 "$py" "$here/site_probe.py" --site "$s" 2>&1 | grep -E "^site=|Error" | grep -v "Expected all" | cut -c1-400
done
