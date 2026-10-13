# The cloud-GPU benchmark — provisioning runbook (vast.ai)

**What this is.** How to rent a GPU box on vast.ai, put our stack on it, run something, and destroy it — for
the GPU-upgrade decision (backlog T13; plan in the memory note `project_cloud_gpu_benchmark.md`). The kit is
[`tools/cloud/vast/`](../../tools/cloud/vast/README.md). This document holds the MEASURED step times of the first
real rental, the cost model, the cu126 / cu128 caveat, the offer criteria and the safety rules. It is
always-current: a later rental that measures differently updates the tables here.

**Status (2026-10-12): provisioning DERISKED; the benchmark itself has NOT been run. The LOCAL cu128 env exists and is CPU-validated; its GPU window (the cu126-vs-cu128 baseline, §3) is PENDING.** One rental of an RTX 5090
went end to end (rent → bootstrap → torch sanity → CPU smoke → one compiled end-state update on sm_120 →
checkpoint copy → destroy → zero instances verified) in 23.8 minutes for $0.24.

---

## 1. The measured rental (2026-10-12)

**Box:** vast.ai offer 45669346 (machine 145154), South Korea, RTX 5090 32 GB, AMD Ryzen 7 7800X3D (16 threads,
all ours), 31 GB RAM, driver 580.95.05 (CUDA 13.0), reliability 0.998, listed $0.470/h. Image
`nvidia/cuda:12.8.1-base-ubuntu22.04` (0.1 GB), 60 GB disk. Code: `f96cbfb9` (public GitHub, shallow fetch).

Every time below is MEASURED wall time on that box (`remote_bootstrap.sh` writes `times.tsv`), one run each.

| step | wall | notes |
|---|---|---|
| `create` → `running` | 31 s | the image was pulled and started in that time |
| `create` → ssh answers | 40 s | `cli.py wait` (direct port) |
| `nvidia-smi` | <1 s | RTX 5090, 32,607 MiB. It showed 100 % util / 115 W at idle right after start, and 0 % / 8 W later (transient; see §6) |
| apt (git, curl, build-essential, …) | 22.3 s | |
| uv + python 3.11 + venv | 1.5 s | python-build-standalone |
| pip: the cu128 stack (~6.7 GB venv) | 81.8 s | `uv pip install`, `--index-strategy unsafe-best-match` |
| torch cu128 sanity | 7.5 s | **PASS**: arch list `sm_70 … sm_90, sm_100, sm_120`; fp32 4096² matmul vs CPU rel 1.0e-6; tiny `torch.compile` first call 0.98 s, diff vs eager 6e-8; bf16 matmul runs |
| shallow clone of the pinned commit | 10.1 s | |
| `git submodule update --init --depth 1` | 5.2 s | |
| Node v22.22.1 (tarball) | 1.5 s | |
| Showdown `npm ci` + build | 43.1 s | team validation's runtime need |
| rustup 1.96.0 (minimal) | 11.8 s | |
| env core + sim bridge `cargo build --release` | 11.9 s | |
| **setup total** | **3.3 min** | 197 s; plus 40 s to ssh |
| CPU smoke `--debug --steps 10000` | 156.8 s | exit 0, `Training complete` |
| GPU sanity: one `update_bench.py` call, `--arch endstate` + the ride-along flags, `--epochs 1 --warm 0 --reps 1` | 335.7 s | R1 startup parity gate **PASS** on sm_120; compile + gate + prewarm 305.1 s; one cold 1-epoch call 11.4 s; peak reserved 6,784 MiB. **SANITY, NOT A BENCHMARK** (one cold call; do not compare it with anything) |
| scp the 57.4 MB checkpoint up (KR) | 7.8 s | sha256 matched |
| scp the results dir down | 25 s | |
| `destroy` → verified gone | ~1 s | `verify-zero`: 0 instances |

**Money:** credit 10.00 → 9.7811 at the destroy, settling to **9.7590 five minutes later** (billing lags), so
**$0.241 for 23.8 min = $0.61/h effective** against the $0.47/h listing (the difference is the 60 GB of storage and
~7 GB of metered download; UNVERIFIED split — vast does not itemise it in the API we read). Read the credit again
a few minutes after a destroy before quoting a session's cost.

**What failed on the way, and the fix (each now in the repo):**
- the compile sentinel REFUSED `torch 2.8.0+cu128` (it keys on the exact version string). Verified on the box: the 20
  hashed torch internals are identical to the +cu126 row, both compile-config keys exist, and the compile contract
  tests pass (68 / 69; the red one used `2.8.0+cu128` as its "unknown version" example). `compile_control` /
  `compile_config` now carry a +cu128 row that copies the +cu126 one (`designs/training/compile_flags.md`, Lane K1);
- the training path died on `from scipy.stats import rankdata` (`ridealong_terms.rank_auroc`): **`environment_torch28.yml`
  does not declare scipy or scikit-learn**, which the code imports; the shared env has them from its clone of
  `gen3ai_stable`. `requirements.py` adds them (`UNDECLARED_RUNTIME`), and `--lock-from` pins the whole live env;
- the vast v0 instance list now answers HTTP 410 (`api.instances` uses `/api/v1/instances/`);
- a remote `nohup setsid … &` held the ssh session open until the job ended; `cli.py ssh --detach LOG` uses `setsid -f`.

## 2. The real benchmark — the procedure

```bash
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3   # repo root
$PY -m tools.cloud.vast.cli credit && $PY -m tools.cloud.vast.cli instances      # expect 0
$PY -m tools.cloud.vast.cli search                                                # read the ranked list
$PY -m tools.cloud.vast.cli create --offer auto --max-minutes <budget+15> --min-credit <credit - cap>
$PY -m tools.cloud.vast.cli wait --instance <id>
$PY -m tools.cloud.vast.requirements --lock-from $PY > /tmp/requirements_cu128.txt  # EXACT parity (§3)
for f in tools/cloud/vast/remote_bootstrap.sh tools/cloud/vast/remote_sanity.py /tmp/requirements_cu128.txt; do
  $PY -m tools.cloud.vast.cli scp-to --instance <id> "$f" /root/prov/; done
$PY -m tools.cloud.vast.cli ssh --instance <id> --detach /root/prov/setup.out -- \
    'bash /root/prov/remote_bootstrap.sh setup <pushed commit>'
$PY -m tools.cloud.vast.cli ssh --instance <id> -- 'cat /root/prov/times.tsv'   # poll (bounded waits, ≤ 55 min)
# … the benchmark jobs, each detached, each writing its own JSON under /root/prov/results …
$PY -m tools.cloud.vast.cli scp-from --instance <id> /root/prov <local dir>
$PY -m tools.cloud.vast.cli destroy --instance <id> && $PY -m tools.cloud.vast.cli verify-zero
```

- The commit must be PUSHED (the box clones the public repo). Pin it; record it with the results.
- Run every long job DETACHED on the box and poll; a dropped ssh must not kill a measurement. Results land
  incrementally (one JSON per job), so a cut-off rental loses only its tail.
- The benchmark jobs are the perf phase's `update_bench.py` configurations
  (`designs/research_state/measurements/perf_phase_2026-10-10/`), with `--epochs 1,3 --reps 2` as there, plus
  whatever short training run the benchmark plan names. No GPU lease applies on the rented box (it is not our
  GPU); the LOCAL baseline needs one, granted by the orchestrator.

## 3. cu126 vs cu128 — the comparability caveat, and how to remove it

The 5090 (Blackwell, **sm_120**) needs a CUDA 12.8 build: the +cu126 wheels' arch list stops at sm_90. So the box
runs **torch 2.8.0+cu128**, while the local 3080 Ti (sm_86) runs the shared env's **2.8.0+cu126**. The torch and
Triton versions are identical (2.8.0 / 3.4.0; the dynamo source is byte-identical, §1), and so is cuDNN (9.10.2.21).
What differs is the CUDA runtime under them: cuBLAS 12.8.4.1 vs 12.6.4.1, cuFFT, cuSPARSE, NVRTC, nvJitLink. A
cuBLAS heuristic change can move a GEMM-bound update by a few percent either way (**UNVERIFIED** for our graphs).

**To remove it:** measure the 3080 Ti baseline on cu128 too. sm_86 IS in the cu128 arch list (read on the box and
locally: `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`), so a LOCAL cu128 environment runs on the 3080 Ti. That env now
EXISTS (2026-10-12): the conda env **`gen3ai_torch28_cu128`**, built from `environment_torch28_cu128.yml` (the twin of
`environment_torch28.yml`; `./scripts/bootstrap.sh --variant cu128`). Its pip-visible package set equals the shared
env's except the CUDA family (torch / torchvision / torchaudio `+cu128`, the 14 `nvidia-*-cu12` wheels at their cu128
pins) and the deliberately absent `asyncio` backport; the conda layer differs by the CUDA header packages (12.8.1 label)
and by patch-level drift of the `defaults` base libraries (the exact diff is in the table below). It was built beside
`gen3ai_torch28`, which was never modified. The GPU measurement is ONE script, run under a lease:

```bash
GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/cuda_build_validate.sh run      # ~70 min, resumable; `plan` prints the cells
```

**The exact package diff, `gen3ai_torch28` (cu126) -> `gen3ai_torch28_cu128`** (measured 2026-10-12 from `pip list` /
`conda list` of both envs; `pip check` clean on the twin; both pass `python src/utils/env_file.py check` against their own
file, and the twin fails the main file on exactly 17 of its 82 pins: 3 conda headers, 11 `nvidia-*` wheels, torch / torchvision / torchaudio):

| layer | cu126 | cu128 |
|---|---|---|
| torch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 `+cu126` | the same releases `+cu128` (arch list `sm_70 … sm_90, sm_100, sm_120`; cu126's is `sm_50 … sm_90`) |
| `nvidia-*-cu12` wheels (pip) | cublas 12.6.4.1, cupti 12.6.80, nvrtc 12.6.77, cudart 12.6.77, cufft 11.3.0.4, cufile 1.11.1.6, curand 10.3.7.77, cusolver 11.7.1.2, cusparse 12.5.4.2, nvjitlink 12.6.85, nvtx 12.6.77 | cublas 12.8.4.1, cupti 12.8.90, nvrtc 12.8.93, cudart 12.8.90, cufft 11.3.3.83, cufile 1.13.1.3, curand 10.3.9.90, cusolver 11.7.3.90, cusparse 12.5.8.93, nvjitlink 12.8.93, nvtx 12.8.90 |
| the same in both | cudnn 9.10.2.21, cusparselt 0.7.1, nccl 2.27.3, triton 3.4.0 | (identical) |
| conda CUDA headers (AOTInductor) | `cuda-cccl` 12.6.77, `cuda-cudart(-dev/-static)` 12.6.77, `cuda-crt-dev_linux-64` 12.6.85, `cuda-version` 12.6 (10 packages, label `cuda-12.6.3`) | 12.8.90 / 12.8.90 / 12.8.93 / 12.8 (label `cuda-12.8.1`) |
| every other pip package (123 in cu126) | identical versions, **except** `asyncio` 4.0.0 (the deprecated PyPI backport the env file says never to add: present in the shared env from its clone history, inert; ABSENT from the twin) and the editable `gen3ai` (same `.pth`, the main checkout's `src/`) | |
| conda base libraries | not locked (patch-level drift of the `defaults` channel since the shared env was cloned; none touches a numeric path): ncurses 6.5 / sqlite 3.53.2 / xz 5.8.2 / tzdata 2026a / libexpat 2.8.4 / openssl 3.6.3 (conda-forge build) / ca-certificates 2026.8.13 / xorg-libx11 1.8.12, xorg-xorgproto 2024.1, libffi / libgcc / libstdcxx / pthread-stubs / `_openmp_mutex` build numbers, libgomp 15.2.0 | 6.6 / 3.53.4 / 5.8.4 / 2026c / 2.8.5 / 3.5.9 / 2026.9.25 / 1.8.13, 2025.1, newer build numbers (libgomp is folded into libgcc), the same python 3.11.15 and libstdc++ 15.2.0 |

(`packaging` 26.0 and `wheel` 0.46.3 are pinned in the twin's conda block: a fresh build would otherwise take 26.3 / 0.47.0
and the pip-visible diff would not be CUDA-only.)

It runs, on the 3080 Ti, per build (cu126, cu128): the numerics cells (one eager micro-batch of the K9 golden's real
rows from byte-identical weights: loss and gradient rel-L2 across builds against the same-build run-to-run noise;
fp32 and bf16) and the update cells (`update_bench.py`, the perf phase's real `train()` at the production shape with
the compile parity gate, for the end state `E` and production `P`, fp32 and bf16): `train_ms`, peak memory, compile
time, the gate's verdict. Results go to `designs/research_state/measurements/cuda_build_<date>/` with a `SUMMARY.md`.
**Local cu126 → cu128 delta (the caveat's size): PENDING the GPU window** (the GPU is leased to the training session's
bf16 A/B; the orchestrator schedules the window). When it lands, this section holds the table, and the clean
comparison for the cloud ladder is cu128 3080 Ti vs cu128 5090.

Two smaller differences: `uv` installs python **3.11.15** (pinned in `remote_bootstrap.sh`; the first rental got
3.11.17 from a bare `3.11`), and the rented CPU is a 7800X3D, not our 9800X3D (same Zen 4 / X3D class, lower
clocks), so the CPU-side compile and the env lane are slightly slower there — read the GPU-bound update, not the
compile time (305 s there vs 277.5 s locally for the end state, perf phase window A).

## 4. Offer criteria (encoded in `tools/cloud/vast/offers.py`, unit-tested)

| criterion | value | why |
|---|---|---|
| country | **not China** (code CN, or "China" in the name); an UNKNOWN country is refused too | owner, 2026-10-12: "don't use a 5090 from a china provider, all other countries are fine" |
| GPU | one RTX 5090, verified, rentable, on-demand (not interruptible) | the card under test; a bid instance can be pre-empted mid-measurement |
| reliability2 | ≥ 0.98 | |
| `cuda_max_good` | ≥ 12.8 | the driver must run a CUDA 12.8 build |
| CPU | ≥ 8 effective threads; RANKED: a desktop Zen 4 / Zen 5 Ryzen first, then more threads (to 16), then price | our trainer: one hot main thread + 8 Rust env workers (~5 physical cores); a high-clock desktop part is like our 9800X3D |
| RAM / disk / net | ≥ 30 GB / ≥ 60 GB / ≥ 200 Mbit/s down | the venv is 6.7 GB, the checkout + builds ~1.5 GB |
| PCIe | ≥ 10 GB/s measured (`pcie_bw`) | a 6.8 GB/s host was listed; the rollout → update copy would read the slot |
| price | ≤ $1.00/h (`dph_total`) | 35 of 40 offers passed on 2026-10-12, the best at $0.47–0.57/h |

Market notes (2026-10-12): 40 verified on-demand single-5090 offers; the cheapest CPU-like-ours offers were
Korean 7800X3D boxes at $0.47/h (several machines), then Brazil (9700X, $0.55), Kansas (9950X3D, $0.57, but
PCIe 6.8 GB/s), Ontario (9950X, $0.56). Offers CHURN: an offer id can vanish between a search and a create, and
the search returns one representative offer per machine that varies call to call — so create with
`--offer auto` (a fresh search, the ranked survivors tried in order).

## 5. Safety rules (hard; enforced by the tool where it can)

- **The API key** lives at `~/.config/gen3ai/vastai.key` (mode 600; `api.py` refuses a group/other-readable file).
  It is read into a header and never printed, logged, returned or sent to the box.
- **Nothing private on the box.** It clones the PUBLIC repo over https. No GitHub token, no other key: ssh / scp
  use ONLY the dedicated key `~/.ssh/gen3ai_vast` (ed25519, registered on the vast account), with
  `-F /dev/null` (no ~/.ssh/config), no agent, no forwarding, and a separate known_hosts. One checkpoint may go up
  to test the path; no `models/` run dir, no credentials, no proxy config.
- **ONE instance at a time:** `create` refuses while the account lists any instance.
- **THE DEADMAN:** every `create` spawns a detached process (its own session; it survives the agent) that
  destroys the instance at `--max-minutes` or as soon as the credit falls below `--min-credit`, retrying until the
  API no longer lists it (`~/.local/state/gen3ai/vast/deadman_<id>.log`; its three exits are unit-tested in
  `deadman_test.py` — the live rental exercised only the gone-already exit).
- **The ledger** `~/.local/state/gen3ai/vast/ledger.jsonl` records every create (offer, image, credit before,
  deadline) and destroy (verified, credit after).
- **The end of every session:** `destroy`, then `verify-zero` (exit 0 only with ZERO instances) and the credit.
- **Spend:** the owner's credit is prepaid and cannot auto-top-up; the per-session cap is set as
  `--min-credit = credit − cap`. The first spend of a benchmark session needs the owner's go (memory note).

## 6. Cost model for the real benchmark

Effective price $0.61/h (§1; a $0.47/h listing). **Fixed:** ~5 min to a ready box (40 s to ssh, 3.3 min setup,
plus the lock-mode install) ≈ **$0.05**. **Per `update_bench.py` configuration** (`--epochs 1,3 --reps 2`): the
compile + gate + prewarm (~5 min measured for the end state) + 1 warm + 4 timed calls of 1–3 epochs — on the 3080 Ti
that was ~6–8 min per config; budget **8 min ≈ $0.08** on the 5090 until measured. So the plan's matrix (end state +
production, fp32 + bf16 = 4 configs) is ~35 min ≈ **$0.36**; a short real training run of 30 min is **$0.31**; with
fetches and slack, **one session ≈ 1.25 GPU-h ≈ $0.80**, and a $2–3 per-session cap leaves 3× headroom.
Formula: `cost ≈ 0.61 × (0.1 + N_configs × 0.13 + T_train_h + 0.1)` dollars.

## 7. Risks left for the real benchmark

- **Not yet exercised live:** the deadman's destroying exits (unit-tested only); `--offer auto` falling through to
  a second offer; the `--lock-from` install from a COLD uv cache (it ran warm, 3.7 s); python 3.11.15 via uv;
  `scp-from` of a large run directory over a slow link (25 s for the 1 MB-scale results dir to Korea).
- **AOTInductor / the T2 inference service's AOT backend** needs the CUDA headers the conda env carries
  (`cuda-cccl` & co.); the base image has none. If the benchmark times T2's AOT path, use a CUDA 12.8 `devel`
  image or install the headers — untested.
- **The idle 100 % utilisation** read right after start: record `nvidia-smi` (util, power, clocks, temperature,
  other processes) before and after every timed job; `update_bench.py` already records `gpu_others` and the load.
- **Host variance:** a different machine of the same listing can differ (PCIe, clocks, thermals, a noisy neighbour
  CPU); for a buy decision, two boxes (or two sessions) per card, matched against the local cu128 baseline (§3).
- **The undeclared scipy / scikit-learn in `environment_torch28.yml`** (§1) was a defect of the env file, not of
  the cloud kit: a fresh `conda env create -f environment_torch28.yml` would have failed the same way. FIXED
  2026-10-12 (both files declare them, with joblib / threadpoolctl); `src/env_files_test.py` now fails when any
  third-party import of `src/` or `tools/` is declared by neither env file, and keeps the twin's pins equal to the
  main file's except the CUDA family. `tools/cloud/vast/requirements.py` still derives the remote's list from the
  env file (its `UNDECLARED_RUNTIME` belt is now redundant, harmless).


## Local cu126 vs cu128 on the RTX 3080 Ti (MEASURED 2026-10-12, a 42-min GPU window; `designs/research_state/measurements/cuda_build_2026-10-12/SUMMARY.md`)

The CUDA 12.8 build is **interchangeable** with 12.6 on our card, so the cloud ladder's cu126 / cu128 caveat is closed:
- Update time ratios (cu128 / cu126): end state fp32 **0.991**, end state bf16 **1.001**, production fp32 **1.001**, production bf16 **0.999**. Peak memory is identical, and every cell's compile parity gate PASSes.
- Numerics: same weights, same batch, eager. The loss is identical, and the gradient rel-L2 across builds (1.4e-9 fp32, 2.1e-9 bf16) sits at same-build noise (2.4e-9 / 2.2e-9).
- **The local baseline the cloud ladder compares against** (update s = train_ms / 1000): end state fp32 **53.0–53.5**, end state bf16 **41.2–41.3**, production fp32 **45.5–45.6**, production bf16 **37.7**. Peak memory: 6.7 / 6.4 / 6.6 / 6.1 GiB.
- Raw logs: `~/gen3ai_archive/cuda_build_2026-10-12/logs/`.
