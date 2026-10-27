# The cloud-GPU training plan: rented cards as a second lane

**What this is.** The plan for running queued TRAINING work on a rented vast.ai card beside our local RTX 3080 Ti.
It covers what may run there, the rule that keeps comparisons honest, the mechanics, the cost, security, and a
recommendation. It is always-current: update it when a rental measures differently or the queue changes. The rental
kit and its measured step times are [`cloud_gpu_benchmark.md`](cloud_gpu_benchmark.md) and
[`tools/cloud/vast/`](../../tools/cloud/vast/README.md). The card speeds are that runbook's §0.

**Status (2026-10-25): PLAN ONLY.** Nothing is built, rented or launched. Spending needs the owner's approval (§6).
The vast credit is $6.18, read 2026-10-25. It was approved for the benchmark ladder, not for training.

Abbreviations: GPU-h = hours of our 3080 Ti's work (the queue's unit); box-h = rented wall hours; P\* = the
production start node `rb_x26_s2601` @280,000,170; C0 = X35's plain-continuation control; H / P = X35's primary
(the Kakuna panels) and co-primary (S against R on `home72`); pp = percentage points.

---

## 1. What would run there, and the card rule

### 1.1 Why a different card is a confound

- **A run is not reproducible anyway.** Same pin + same `--seed` diverged at rollout 1 (measured 2026-09-09). The
  battle stream differs, so seeds are draws. A different card adds a variance source that could also be SYSTEMATIC.
- **What changes on a cloud 5090.** The arch is sm_120 against our sm_86. It needs the cu128 build (cuBLAS 12.8),
  so its GEMM and attention kernels, and the bf16 trunk's rounding, differ from ours. The cu126 / cu128 BUILD delta
  was measured on our own card and is nil: speed ratios 0.991–1.001, gradients equal to same-build noise. The CARD
  delta is not measured. The startup parity gate PASSed on every ladder card. That proves compiled equals eager on
  that card, not that two cards train alike.
- **What does NOT change.** Production runs serial collection (`overlap_collect: false`), so the card's speed
  changes no row's age. Checkpoints, snapshots and evals are on step cadences. A 20M-step production fork takes
  ~4 h locally and ~2.5 h on a 5090. Under `--restart-interval-hours 24` both finish with ZERO restarts, so no
  wall-clock event differs. `data/`, the Rust core, the Showdown validator and the team corpora come from the
  pinned commit. The pin's `data/` is read from the pin (`gen3_pinned_data_root_v1` is in `f4b25238` and in
  D3_PIN).
- **What is numerics-sensitive and UNVERIFIED on sm_120 over a whole run:**
  - K9(b), the behaviour-consistency check, has a 1e-4 bar. Its τ-class violations were a numerics bug (T62 /
    FP32_ALT).
  - The in-run compile canary.
  - The FP32_ALT τ solve's CUDA case.

  An X35 seed is VOID unless its arm-end K9(b) PASSes.
- **Why no cheap test can clear it.** X35's NULL band is ±3.0 pp. D3's predicted effect is +1.5 pp. The training-seed
  SD is about 2 pp (REVIEW D). One or two cloud seeds against five local ones cannot bound a card effect inside that
  band. An equivalence claim needs the delta's CI inside the bar, and this is nowhere near it.

### 1.2 THE CARD RULE (proposed)

1. **No comparison crosses card types.** An arm and its matched control run on the same card type and CUDA build.
   The banked local C0 rows are never the control of a cloud arm, and the reverse holds too. Hosts may differ; each
   host is recorded (§2.4).
2. **Whole arms only.** Every seed of an arm, at both looks, runs on one card type.
3. **The cloud control is its own set: C0v.** It is C0's argv token for token at C0's pin `f4b25238`, seeds
   3501–3503 (3504–3505 for look 2). The run names carry a tag (proposed `rb_x35_c0v_s<seed>`). Like C0, it is shared
   by EVERY cloud production arm, so the arm that pays for it is the first one.
4. **The card gate comes before any counted cloud arm.** The first C0v seed must pass the mechanical bars:
   - startup parity gate PASS;
   - no compile-canary UNCONFIRMED / FATAL;
   - `compile/recompiles_after_lock` 0;
   - T54's pool-seed check PASS;
   - zero restarts;
   - arm-end K9(b) PASS (AM-6's ceilings).

   It must also land inside the local C0 seeds' range on H and P. That range check is a gross screen, NOT an
   equivalence claim (§1.1). A failure means no cloud production arm, and it is reported as a finding.
5. **The card-boundary read is a by-product, never a license.** Local C0 against C0v on H and P (Welch, 3–5 seeds
   each) is REPORTED. It never licenses pooling the two. It is also the first data on whether a 5090 trains
   differently, which the buy decision needs: once a bought card replaces the 3080 Ti, every banked control is
   cross-card.

### 1.3 The queue candidates, judged

| candidate | GPU-h (source) | cloud? | why |
|---|---|---|---|
| **C0v**, the cloud control | 12.1 look 1 (3 × ≈ 4.04, D1's measured 3.87–4.31 per seed) | **YES, first** | rule 3–4; the gate seed s3501 first (≈ 4 GPU-h) |
| **D3** (X41, set mutation) seeds 3501–3503 (+3504–3505) | 12.1 look 1, 20.2 final (D3 PREREG §2) | **YES, after the gate, against C0v** | one pin (D3_PIN `9e289ba4`, a tag on origin), committed corpus, fixture-free fork of P\*. Cost of the rule: C0's banked rows (and their reuse condition, PREREG §3) are replaced by C0v's, about +12 CPU-h of local Kakuna + P play per look. Its GO is the orchestrator's fresh ruling |
| **SPRT**, **LR-F** (production side) | 12.9 per side each (REPLAN §1) | **YES, against C0v** | same frame as D3: the same side's C0 is the control, so C0v serves |
| LR-F / any arm on the **toy** side | toy fork +70M steps, ≈ 4.2 h at 4,607 fps | **NO** | the toy cycle is CPU-heavy (fps 3.5× production's), so a 5090's speedup there is unmeasured and likely small. The rented host's CPU becomes a second confound. Toy C0 is banked locally |
| **X59-S**, the matched-student contest | ≈ 2 (`human_data_architecture.md` §7) | **NO** | its rowset derives from the gitignored human replays and the salted-account scheme, so human data would leave the box (§4). 2 GPU-h fits a local gap. REVIEW D's majors are still open |
| **human-data A/B arms** (X60, H-TEAMS / H-BC) | unregistered | **NO, for now** | the switch is PROPOSED, NOT BUILT; X60 is REDESIGNED, pending review, and conditional on X59-S. Later only as a whole arm plus its own control, and only if the owner rules on human data leaving the box |

---

## 2. The mechanics

### 2.1 Image and bootstrap: the existing kit, unchanged

The image is `nvidia/cuda:12.8.1-base-ubuntu22.04`. `remote_bootstrap.sh setup <FULL pushed SHA>` installs the
following (measured 3.3 min, `cloud_gpu_benchmark.md` §1):
- apt, uv and python 3.11.15;
- the cu128 stack, locked from `gen3ai_torch28_cu128` (`requirements.py --lock-from`) for exact package parity;
- a shallow clone of the pin plus the submodule;
- Node and the Showdown build (the validator);
- rustup and the env-core `cargo build --release`.

Both pins carry the +cu128 compile-sentinel row (checked: `9201963d` is an ancestor of `f4b25238` and
`9e289ba4`).

**Path-identical layout.** `metadata.json` records ABSOLUTE paths (the parent run dir, the start checkpoint, the
snapshot dir, its own run dir under `/home/goodlad/dev/gen3ai/models/`). The box therefore clones into that same
path. The run then records the same lineage strings a local run would. No sidecar is edited on pull-back.

### 2.2 What goes up (per box, ≈ 170 MB)

The fork's parent skeleton, sha256-checked on the box:
- `rb_x26_s2601/{metadata.json, model_config.json}`;
- the start checkpoint `checkpoint_280000170_steps.zip` (57 MB);
- `snapshots/` (110 MB, 23 files), which `pool_seed.seed_pool` copies into the fork's pool.

The frozen reference population R (825 MB of members) does NOT go: R is read only by the local readouts. Whether
the fork reads any other parent file is **UNVERIFIED**. The box dry-run and the post-spawn pool-seed check (§2.3)
answer it.

### 2.3 Launch: program checks stay local; only the spawn is remote

`launch_x35.sh`'s pre-checks are about the PROGRAM's state, and every one reads the local `models/`: the side's
recorded pin across every rb_x35 run (2b), R's sha256s (4), R frozen before every run dir (7), stage records (9), the
pool record (10), and earlier arms' K9(b) (11). They also assume the local lease and `taskset` cores. So they run
LOCALLY, in a new `--remote` mode that stops before the spawn.

On the box, the wrapper does three things:
1. Accept the box: `remote_sanity.py`, `python -m agents.training.golden_obs_core --check`, and a launcher
   `--dry-run` that says "would launch", FORK of `rb_x26_s2601`, +20,000,000.
2. Spawn the kit's exact argv with `--restart-interval-hours 24 --pin-commit <pin>`, detached (`cli.py ssh
   --detach`).
3. Run `kit pool-seed-check`.

Whether a shallow clone of the pin is enough for the launcher's pinned worktree is **UNVERIFIED**; the box dry-run
answers it. Standing rule 3 applies to the ssh argv as well: no literal trainer script name, only
`python -m main.launcher`.

### 2.4 Monitoring, pull-back and teardown

- **One box per seed.** Setup is only ~5 min. `cli.py create` refuses while ANY instance exists, so seeds run one
  after another. Each box's deadman is `--max-minutes 240` (≈ 2.5 h run + setup + pull + margin), and its
  `--min-credit` floor is the program's cap.
- **The watcher.** The Training session runs the cloud lane: new runs go to it. Its single bounded watcher
  (`timeout 3300`) ssh-reads the remote launcher log (`END`, `FATAL`, the canary lines) and the last tb step. Every
  30 min it rsyncs `tb/`, the logs, `checkpoints/` and `metadata.json` into a staging dir OUTSIDE `models/`
  (`~/gen3ai_archive/cloud_inbox/<run>/`). A deadman destroy then loses at most 30 min and never the banked
  checkpoints.
- **The pull-back.** After `END`:
  - do a final full pull (D1's run dir is 1.8 GB; the measured pull speed was 25 s for the 2026-10-12 results);
  - build a sha256 manifest on both sides and compare them;
  - move the staged dir into `models/<run>` once, refusing an existing dir;
  - write a NEW sidecar `cloud_provenance.json` beside the run's files: card, VRAM, driver, torch build, host CPU,
    vast instance and machine ids, $/h, start / end, and the settled cost.

  The immutable blocks (`original_command`, `lineage`, `pin_history`) are never touched. **FINDING:** a run records
  no GPU model today. Only `torch_version` (`+cu128`) in `metadata.json` and `device_total_mib` in
  `cuda_ledger.json` tell a cloud run apart. So the provenance sidecar is the record, and `main.sidecar_audit` should
  learn to read it.
- **Teardown.** `destroy`, then `verify-zero`, then re-read the credit 5 min later (billing lags). Append the cost to
  `~/.local/state/gen3ai/vast/ledger.jsonl`.

### 2.5 The builds, and the setup cost

| build | agent-h (ESTIMATE) |
|---|---|
| B1 a `train` step in `remote_bootstrap.sh`: path-identical clone, parent upload + sha check, box acceptance, detached spawn | 2–3 |
| B2 `launch_x35.sh --remote`: every program pre-check locally, then hand the argv to B1; the remote pool-seed check | 1.5–2 |
| B3 incremental pull, the manifest, the one-way import into `models/`, `cloud_provenance.json` (+ tests, + the `sidecar_audit` reader) | 2.5–3.5 |
| B4 kit arm `c0v` (C0's argv with a tagged run name) and the readout taking C0v's rows as a cloud arm's control | 1.5–2 |
| B5 the watcher recipe in `TRAINING_RUN_SOP.md`, and this doc's measured numbers | 0.5–1 |
| B6 the dress rehearsal: C0's argv with `--steps 281000170` (+1M steps, ~8 min) into a scratch `GEN3AI_MODELS_DIR`, end to end, then destroy | 1 (≈ $0.20) |
| **total** | **≈ 9–12.5 agent-h**, plus ≈ 0.3 agent-h to operate each box |

---

## 3. Cost

**Prices** come from a read-only `cli.py search` on 2026-10-25, after the kit's filter (no China, reliability
≥ 0.98, CUDA ≥ 12.8). RTX 5090: 16 passing offers at $0.50–0.94/h. The cheapest is the South Korean 7800X3D host
used for the 2026-10-12 derisk ($0.497); the next desktop Zen 4 host is $0.67. RTX 5080: 10 passing at $0.34–0.41/h
(9950X hosts at $0.34). The first rental's effective rate was $0.61/h against a $0.47 listing over 24 min, which
included ~7 GB of downloads. On multi-hour boxes this plan assumes the listing + ~$0.05/h + ~$0.15 per box
(**UNVERIFIED**: vast does not itemise storage and bandwidth).

**Speed.** The 5090 runs the cycle at 1.63–1.70× (§0 of the runbook: 9900X / 9950X3D hosts). On the 7800X3D host
the cycle is **UNVERIFIED**. This plan uses 1.65×, so one production fork (≈ 4.04 GPU-h) is ≈ 2.45 box-h, and
≈ 2.6 box-h with setup and pull.

| workload | GPU-h | box-h | $ at $0.50 | $ at $0.94 ceiling |
|---|---|---|---|---|
| B6 rehearsal | — | 0.25 | 0.20 | 0.35 |
| C0v s3501 (the card gate) | 4.0 | 2.6 | 1.60 | 2.70 |
| C0v s3502–3503 | 8.1 | 5.2 | 3.20 | 5.40 |
| D3 look 1 (3 seeds) | 12.1 | 7.8 | 4.80 | 8.10 |
| **first program (the above)** | **24.2** | **15.9** | **≈ 9.80** | **≈ 16.60** |
| + SPRT and LR-F, look 1 (production) | 25.8 | 16.8 | 10.20 | 17.30 |
| + each look 2 (2 seeds) | 8.1 | 5.2 | 3.20 | 5.40 |

**The 5080 is not cheaper in practice.** At $0.34/h and 1.24×, one fork costs about $1.10–1.20, about 5 % less per
unit of work than the 5090. It takes 1.35× the wall time. The 5090 is also the card the runbook recommends buying,
so its training boundary is the one we need to learn. **The plan uses the 5090.**

**Wall clock saved.** Local look 1 of D3 + SPRT + LR-F is 37.9 GPU-h serial, ≈ 1.6 days.
- With the first program on the cloud (≈ 16 box-h) and SPRT + LR-F locally (25.8 h), the queue drains in ≈ 26 h
  instead of 38 h: **≈ 12 h saved for ≈ $10**.
- Moving SPRT to the cloud too gives ≈ 23 h against 38 h, about 15 h saved for ≈ $15.

C0v is the rule's price (≈ $5). After it, each cloud arm saves ≈ 12–13 local GPU-h for ≈ $5, about $0.38 per
local GPU-h.

**FINDING: the CPU readouts become the next bottleneck.** Every production look costs ≈ 12 CPU-h locally: Kakuna
3 × 420 games (≈ 3.5 lane-h) and P 3 × 3,500 pairs (≈ 8.6 CPU-h). C0v adds its own read, because C0's banked rows
cannot serve a cloud arm. A faster GPU queue doubles this load on the three agent cores of rule 5b.

---

## 4. Security and public-repo hygiene

- **The key.** The key stays in `~/.config/gen3ai/vastai.key` (mode 600). It is read only by `tools/cloud/vast/api.py`
  and never printed, logged, committed or sent to a box. The ssh identity is the dedicated `~/.ssh/gen3ai_vast` (no
  agent forwarding, `-F /dev/null`). The box holds NO credential to reach us: every transfer is a pull or push
  started locally.
- **What reaches a box:** the PUBLIC repo at a pushed full SHA (anonymous https fetch, no GitHub token), the cu128
  requirements lock, the kit scripts, and our checkpoints (P\* + its pool snapshots, ≈ 170 MB). **The checkpoints
  are a scope change.** The benchmark's rule was "no `models/` leaves the box", so they need the owner's yes (§6).
  Treat anything uploaded as readable by the host's operator: vast does not promise a disk wipe on destroy.
- **What never reaches a box:**
  - `replays/` and every human-data rowset;
  - `~/gen3ai_archive/human_snapshots/` (the salt and the name map);
  - the proxy config, `~/.claude/`, and any other key or token;
  - `designs/ops/anchors.json`'s box paths, which mean nothing there.
- **The repo is public.** This doc, the provenance sidecars and any ledger entry name card, country, CPU, price and
  the vast instance id. They never name an IP, an ssh port or the key path's contents. The rental ledger stays in
  `~/.local/state/`. Ship through `python -m utils.push_guard` as always.

---

## 5. Recommendation: GO, staged, on the owner's approval

1. **Now (CPU, no spend):** build B1–B5 (≈ 9–12 agent-h).
2. **The first spend:** the B6 rehearsal (≈ $0.20), then **C0v s3501 as the card gate** (≈ $1.60). This is the
   first workload. It is useful even if no cloud arm follows: it is the first evidence of whether a 5090 trains P\*'s
   fork differently, which the buy decision and every future cross-card comparison need.
3. **On a gate PASS:** C0v s3502–3503, then D3 look 1 on the cloud once the orchestrator gives D3 its GO. The local
   card meanwhile takes SPRT / LR-F against the local C0. **On a gate FAIL:** stop the cloud lane, report, and keep
   every production arm local.
4. **No-go conditions:**
   - the owner declines the checkpoint upload;
   - a 5090 purchase lands within ≈ 2 weeks: run the same gate on the bought card instead, and drop the rental
     program;
   - the GPU queue empties before B1–B5 are built.

---

## 6. The approval question (the owner's, verbatim)

> **Do you approve (a) topping up the vast.ai credit by $15 (no auto top-up) and spending up to $20 in total on this
> cloud training program, at most $0.70/h per RTX 5090, non-China hosts, one instance at a time, each with its
> deadman; (b) uploading our checkpoints (P\* and its opponent-pool snapshots, about 170 MB per box, never human
> replay data or any credential) to those rented hosts; and (c) the first workload: a rehearsal, then one cloud C0
> seed as the card gate, then, if it passes and D3 has its GO, C0 s3502–3503 and D3's three look-1 seeds on the
> cloud? Yes, no, or change which part?**

**STALE (2026-10-27):** the D3 look-1 workload in (c) is void; those seeds trained locally. Current candidates: `gpu_plan_2026-10-27.md`.
