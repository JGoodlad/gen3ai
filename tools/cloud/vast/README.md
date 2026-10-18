# `tools/cloud/vast/` — rent, bootstrap and destroy a vast.ai GPU box

The provisioning kit for the cloud-GPU benchmark (backlog T13). It is NOT part of the acquisition layer: it
writes nothing under `data/` and the runtime never imports it. **The runbook, with the measured step times, the
cost model, the cu126 / cu128 caveat, the offer criteria and the safety rules, is
[`designs/ops/cloud_gpu_benchmark.md`](../../../designs/ops/cloud_gpu_benchmark.md).**

| File | What it is |
|---|---|
| `api.py` | a stdlib REST client (Bearer key from `~/.config/gen3ai/vastai.key`, never printed) |
| `offers.py` | THE OFFER FILTER (pure): no China, reliability, CUDA 12.8+, CPU / RAM / disk / net / PCIe floors, the ranking |
| `cli.py` | `credit`, `search`, `ensure-key`, `create` (+ the detached DEADMAN; `--gpu`, `--min-cuda`, `--exclude-machines` for a host that misbehaved), `wait`, `ssh`, `scp-to/from`, `destroy`, `instances`, `verify-zero` |
| `requirements.py` | derives the remote's pip list from `environment_torch28.yml`: the same pins, torch on cu128 |
| `remote_bootstrap.sh` | runs ON the box: apt, uv + python, the cu128 (or, with `REQS=…cu126.txt`, cu126) stack, the torch sanity, the shallow clone + submodule, Node, the Showdown build, rustup + the env-core build, the CPU smoke, one GPU update sanity, and `bench` = THE FROZEN SUITE (`python -m agents.training.gpu_bench_suite run`, resumable); idempotent, every step timed |
| `remote_sanity.py` | the torch / CUDA sanity (a kernel the device RUNS in the arch list — its own sm or a same-major lower-minor cubin, `arch_runs`; fp32 matmul vs CPU, a tiny `torch.compile`, bf16) |
| `*_test.py` | the offer filter, the requirements derivation and the deadman's exits (no network) |

```bash
export PYTHONPATH=$PWD/src            # from the repo root; PY = the gen3ai_torch28 python
$PY -m tools.cloud.vast.cli credit && $PY -m tools.cloud.vast.cli search
$PY -m tools.cloud.vast.cli ensure-key
$PY -m tools.cloud.vast.cli create --offer auto --gpu "RTX 3090" --max-minutes 75 --min-credit 7.0 --disk 60   # --min-cuda 12.6 for a cu126 box
$PY -m tools.cloud.vast.cli wait --instance <id>
$PY -m tools.cloud.vast.requirements --lock-from $PY > /tmp/requirements_cu128.txt   # exact parity (or no flag: the env file)
for f in tools/cloud/vast/remote_bootstrap.sh tools/cloud/vast/remote_sanity.py /tmp/requirements_cu128.txt; do
  $PY -m tools.cloud.vast.cli scp-to --instance <id> "$f" /root/prov/; done
$PY -m tools.cloud.vast.cli ssh --instance <id> --detach /root/prov/run.out -- \
    'bash /root/prov/remote_bootstrap.sh all <FULL 40-hex pushed commit>'
$PY -m tools.cloud.vast.cli ssh --instance <id> -- 'cat /root/prov/times.tsv'     # poll
$PY -m tools.cloud.vast.cli ssh --instance <id> --detach /root/prov/bench.out -- \
    'BENCH_TAG="RTX 3090 / US" BENCH_META="--meta dph=0.26" bash /root/prov/remote_bootstrap.sh bench'
$PY -m tools.cloud.vast.cli scp-from --instance <id> /root/prov/results <local dir>
$PY -m tools.cloud.vast.cli destroy --instance <id> && $PY -m tools.cloud.vast.cli verify-zero
```

🚨 Every `create` arms a detached deadman (`~/.local/state/gen3ai/vast/deadman_<id>.log`) and logs to
`~/.local/state/gen3ai/vast/ledger.jsonl`; `create` refuses while ANY instance exists. Nothing private goes to the
box: the dedicated key `~/.ssh/gen3ai_vast` is the only identity offered (`-F /dev/null`, no agent).
