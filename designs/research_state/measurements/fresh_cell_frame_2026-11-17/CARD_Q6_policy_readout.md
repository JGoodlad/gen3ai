# Lever card Q6: retire the flat policy tower, one fresh cell (DRAFT, 2026-11-17)

**Status: the arm Q6-N was REGISTERED 2026-11-20 by
[`../q6n_policy_readout_fresh_2026-11-20/PREREG.md`](../q6n_policy_readout_fresh_2026-11-20/PREREG.md), which adopts §1 to §5
and REPLACES §6 and §7 (the card check is the first item of the GPU window, at 4.5M steps and width 128 only; Q6 runs
before Q5). The arm Q6-W was NOT registered and stays a draft. The text below is the 2026-11-17 draft, kept as
written: nothing launched; not launch-ready until the card check is green.** A card
for [`FRAME.md`](FRAME.md). Queue proposal row Q6. Registering it is a FULL `PREREG.md` that adopts the frame.

## 1. The lever

- **`--policy-readout trunk --allow-nonproduction-arch`.** The move chooser reads the refined attention trunk
  through ONE learned query (`pools.PolicyStateQuery`, 4 heads, over every trunk token and the belief pool's
  outputs) instead of the flat path: the extractor's 512-wide `projection` and SB3's `mlp_extractor.policy_net`
  tanh tower, which are built and then retired before the optimizer. Each action is still scored from its own refined
  token. The value path is untouched. Built `38ed7d0da` (ledger 2026-10-06, config v138, audit F2; owner: "no need to
  keep the tower"); production stays `tower`, byte-identical.
- **`--allow-nonproduction-arch` is part of the lever**, as it is of B1's width lever: the consent a fresh run with
  an ARCH-surface key off production needs. At the pin the launcher only ADVISES on a pinned child's surface (its
  dry run printed the diff and refused nothing without it), but an unpinned check refuses it, and with it the run
  records `arch_source` as non-production. At width 256 the W control already types it, so there it adds nothing.
- **Why it might pay** (queue proposal; the 11-09 architecture diagnosis `a8265f36b`): the tower's first layer is
  0.897 saturated on and off the pool and cuts the policy gradient 15 to 18×, and it holds a large share of the
  parameters. No strength evidence either way: the flag has never been trained to a read.

## 2. The flag at the pin: the same-pin rule HOLDS

`--policy-readout` is defined at the pin (`src/main/train/parser/clean_world.py:202` at `15136b616`;
`resolve_config` resolves `policy_readout`, default `tower`, version-checked, fresh-only) and on main. Its code entered
at `38ed7d0da`, well before the pin. So the arm trains at B1's pin and reads against the banked cell (FRAME § 2 item 1).
No fix to the trunk path landed after the pin, as far as a diff can show: from the pin to main `81594609b`,
`src/agents/model/` changes 7 files (160 lines) and `src/main/train/config.py` 25 lines (the compile-trainer AUTO
decision on a resume), and no changed line under `src/agents/model/` or `src/agents/training/instrumented_ppo/`
names `policy_readout`, `policy_query` or `PolicyStateQuery`. If § 6's card check finds a defect whose fix lands
after the pin, FRAME § 7 applies: the arm buys its own control at the fixed pin.

## 3. Fit: the parameter count (CPU, measured at the pin)

Built with `main.fresh_checkpoint.build_fresh_model` from each argv's resolved config (`main.b1.arms.resolved_namespace`)
in a clean detached checkout of the pin, CPU, 2026-11-17 (`params.txt`):

| arm | learner parameters | change | the largest groups |
|---|---|---|---|
| B1 C (128, tower) | 2,609,183 (B1's `EXPECTED_PARAMS`) | — | `projection` 590,336 · `policy_net` 525,312 · `team_transformer` 401,536 |
| **Q6-N** (128, trunk) | **1,547,039** | −1,062,144 (−40.7 %) | `team_transformer` 401,536 · `hidden_opp_belief` 199,808 · `cls_pool` 199,296 |
| B1 W (256, tower) | 6,258,079 (B1's) | — | `team_transformer` 1,589,504 · `projection` 1,180,160 · `policy_net` 525,312 |
| **Q6-W** (256, trunk) | **4,894,879** | −1,363,200 (−21.8 %) | `team_transformer` 1,589,504 · `hidden_opp_belief` 792,832 · `cls_pool` 791,808; `policy_query` 198,144 |

Memory: the arm holds fewer parameters than its control at either width, so the optimizer state shrinks; activation
memory is not counted here. The one GPU record of this flag, at an older production (ledger 2026-10-06's GPU checks,
`blob`, width 128): `UpdateFit` demand 7,818 MiB, headroom 3,156 MiB, R1 compile parity PASS, the update-10 canary
PASS. B1's W pilot read headroom 1,398.5 MiB at 256 with the tower (374.5 above the 1,024 bar). **UNVERIFIED:** the fit
at 256 with the trunk readout at this pin; § 6 measures it.

## 4. Compile shape on CPU

`python -m main.compile_inventory fullgraph` compiles each candidate region with `fullgraph=True` (CPU, the eager
backend, fresh weights, the committed parity rows), but only for production's policy kwargs: the tool has no knob for
an arm. A scratch harness ran the tool's own `fullgraph_check.run(batch=16, backend="eager")` with
`main.fresh_checkpoint._production_policy_kwargs` bound to each arm's resolved config (no tool change):

**All 10 verdicts OK in all four builds** (B1 C, B1 W, Q6-N, Q6-W): the extractor's `forward`, the same through
`Module.__call__` and `extract_features`, each in train/grad, train/no-grad and eval/no-grad, and the rollout's
`policy(obs, action_masks=…)` in eval/no-grad, each compiles as ONE graph with `fullgraph=True`. Peak 1.05 GB.
(`fullgraph_cpu.txt`)

This is dynamo's graph capture on CPU only. The CUDA side (the Inductor compile of R1, its parity check, the T2
CUDA-graph capture) has no CPU stand-in; § 6 is its check.

## 5. The arms, the argvs, the CPU checks

| arm | banked control | lever typed | seeds (spare) | runs |
|---|---|---|---|---|
| **Q6-N** (128) | B1 C: `rb_b1c_s5101`, `rb_b1c_s5102`, `rb_b1c_s5110` | `--policy-readout trunk --allow-nonproduction-arch` | 5211, 5212, 5213 (5219) | `rb_q6n_s5211` … `_s5213` |
| **Q6-W** (256) | B1 W: `rb_b1w_s5103`, `rb_b1w_s5104`, `rb_b1w_s5111` | `--policy-readout trunk` | 5214, 5215, 5216 (5219) | `rb_q6w_s5214` … `_s5216` |

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --policy-readout trunk --allow-nonproduction-arch --checkpoint-every-steps 1000000 --steps 50000000 --seed 5211 --device cuda --log-level periodic --run-name rb_q6n_s5211
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch --policy-readout trunk --checkpoint-every-steps 1000000 --steps 50000000 --seed 5214 --device cuda --log-level periodic --run-name rb_q6w_s5214
```

- **checkargs** (the pin's parser): both exit 0, 0 unrecognized; the ARCH surface lists `policy_readout 'trunk'`
  against production's `'tower'` (with the consent noted), the RECIPE surface `ent_coef 0.0125` alone.
  (`checkargs_q6n.txt`, `checkargs_q6w.txt`)
- **The resolved-config diff** (FRAME § 2 item 2) at 50M and 100M: Q6-N against C differs in `policy_readout`
  (`tower` → `trunk`), `allow_nonproduction_arch` (False → True), `seed`, `run_name`; Q6-W against W in
  `policy_readout`, `seed`, `run_name`. PASS. (`diff_q6n.txt`, `diff_q6w.txt`)
- **The dry run**: both exit 3 with exactly B1's two refusals (the CUDA preflight, the inherited-nice gate); every
  other preflight passes. (`dryrun_q6n.txt`, `dryrun_q6w.txt`)
- **The seed scan**: no hit (`seed_scan.txt`, shared with Q5).
- **The pin's own CPU tests of the trunk readout:** `src/agents/model/policy_readout_test.py` (9 tests: the
  tower retired exactly and the count follows, the state-query read, key-order invariance and masking, equivariance
  under switch / move slot permutations, the T2 decision forward, one fresh-build CPU update under the K6 freeze,
  the recorded / migrated / refused config) and `d_model_width_test.py::test_a_wide_build_follows_the_width_and_trains`
  (6 cases, among them `trunk_readout` at width 256), run in the pin's checkout: **15 passed** in 23.7 s.
- **The CPU `--debug` smokes at the pin** (`--steps 10000`, the base and the lever typed; `smokes_cpu.txt`): both
  PASS. Q6-N: exit 0, 5 updates, round-trip PASSED with `pi shape (1, 128)` (the state query's width; the tower
  arms print 512), LEARNER FREEZE released (10 checks), Training complete, 2,330 s, peak 3.31 GB; the run records
  `policy_readout` `trunk`, `git_hash` `15136b616`. Q6-W: the same with `pi shape (1, 256)`, 2,406 s, peak 4.99 GB.
  **Throughput is not read from these**: one env on shared CPU, eager, `--debug` (no compile); § 6 measures ρ.

## 6. The ≤ 3 GPU-h card check: a command for a run boundary, NOT run

Written for the agent the orchestrator leases the card to, between two runs, from the MAIN checkout at a shell of
nice ≤ 10 with the card visible. It is B1 §13 item 4's pilot in the arm's launch shape, at the launch pin, one width
at a time, into a SCRATCH models dir (`GEN3AI_MODELS_DIR`; the pilot runs are never comparators and are deleted by
their owner's rule only). Seed 5299 (outside both cards' blocks; re-scan first).

```bash
export GEN3AI_MODELS_DIR=<an empty scratch dir> PIN=15136b61634b667bb9da5ccc6647ff9265238829
scripts/ops/gpu_lease.sh acquire --owner <name>          # the orchestrator names the agent; token into the env
python -m main.launcher --dry-run --restart-interval-hours 6 --pin-commit $PIN --arch production --ent-coef 0.0125 --behaviour-check warn --policy-readout trunk --allow-nonproduction-arch --checkpoint-every-steps 1000000 --steps 2500000 --seed 5299 --device cuda --log-level periodic --run-name q6n_pilot_s5299
GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/gpu_lock.sh python -m main.launcher --restart-interval-hours 6 --pin-commit $PIN --arch production --ent-coef 0.0125 --behaviour-check warn --policy-readout trunk --allow-nonproduction-arch --checkpoint-every-steps 1000000 --steps 2500000 --seed 5299 --device cuda --log-level periodic --run-name q6n_pilot_s5299
# then the same pair with --d-model 256 --trunk-ffn-dim 512 inserted after --allow-nonproduction-arch, --run-name q6w_pilot_s5299
scripts/ops/gpu_lease.sh release
```

Each pilot runs to 2,500,000 steps, past its first 1M checkpoint, its first evaluation cycle (2M) and its first pool
promotion, whichever is last (if no promotion by 2.5M, it is recorded as not reached and the check is still read).
**PASS iff every one of these holds, per width** (B1 §13 item 4's (b) to (e), plus the step time):

1. the bf16 R1 compile completes and its compile-parity check passes (the child log's parity line);
2. the inference T2 CUDA-graph capture succeeds (`T2 up … backend graph on cuda`);
3. the startup fit check passes and `update_fit.json`'s headroom is ≥ 1,024 MiB, recorded with the demand;
4. no parity-gate FATAL through the first promotion; the update-10 canary passes;
5. **ρ_Q6**, the step time over the steady updates after the compile against B1's pilot of the same width (B1 §18
   slots S1 to S7), is recorded. It decides nothing; it prices the cell (≈ 28.6 / 39.6 GPU-h × ρ_Q6 to 50M).

**Cost:** about 0.5 GPU-h of training a pilot (2.5M steps at B1's 9.55 h / 13.2 h per 50M: 0.48 and 0.66 h) plus a
cold compile each (B1 measured ≈ 16 min a cold start), ≈ 1.7 GPU-h for both widths, under 3. **UNVERIFIED:** whether
the shared compile cache misses (it will: the trunk readout is a new graph) and how long a miss takes at 256. A FAIL
holds the card and comes back as a build; a fix after the pin sends the arm to FRAME § 7.

## 7. Status

**Not launch-ready.** Ready: the argvs and their CPU checks, the fit count, the CPU graph capture, the smokes.
Owed: § 6 on the card (the one thing a CPU cannot show: the CUDA compile, the graph capture and the memory fit
at 256); then the FULL `PREREG.md` on main; the `b1_fresh_cell` rule and kit are built (FRAME F1; its arm file `fresh_arm_q6n.json` / `_q6w`).
Order (queue proposal § 3): Q6 ranks above Q5 if B1's width factor wins, since then trunk capacity is what pays.
