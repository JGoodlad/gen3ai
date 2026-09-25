# Population loop round 2 — launch kit + validation evidence (2026-09-24)

The registration (design, provenance, the bar, the branches, the queue) is
[`../../population_loop_round2_2026-09-24.md`](../../population_loop_round2_2026-09-24.md).
**Nothing here was executed for real by the session that wrote it: no arm was launched, and nothing
was written under `models/`.** Every launch script is **PREPARED BUT NEVER EXECUTED**, for the Training
Run session to run VERBATIM when the queue reaches it.

**The codes, each time:** **B2** = `ai_v13_27_popr2_loop`, the round-2 LOOP generalist (B +8M, stable
set {A, `ai_v13_13`, RB} at share 0.40). **C2** = `ai_v13_28_popr2_ctrl`, the round-2 NO-EXPLOITER
CONTROL (C +8M, share 0.0). **RB2** = `ai_v13_29_popr2_read_loop`, the fresh offense reader of B2.
**RC2** = `ai_v13_30_popr2_read_ctrl`, the fresh offense reader of C2. **RB+** =
`ai_v13_31_popr1_read_loop_ext`, round 1's reader of the loop (RB, `ai_v13_24_popr1_read_loop`)
forked +50 %. **RC+** = `ai_v13_32_popr1_read_ctrl_ext`, round 1's reader of the control (RC,
`ai_v13_25_popr1_read_ctrl`) forked +50 %. **A** = `ai_v13_18_teach5_offense_hidose`, the round-0
offense exploiter whose recipe every reader copies.

## Contents

| path | what |
|---|---|
| `generalists/` | **VERBATIM copies** of the Training Run session's B2 / C2 kit from `/home/goodlad/.claude/jobs/popr2_2026-09-24/`: `argv_B2.txt` (sha256 `9014d843…`), `argv_C2.txt` (sha256 `9dce840c…`), `launch_B2.sh`, `launch_C2.sh`, `chain.sh`, and their `--dry-run` outputs. B2 LAUNCHED 17:43 PT 09-24; C2 follows on the chain. Copied so the registration does not depend on a scratch path (finding P-5) |
| `launch_RB2.sh` | **RB2** (reader of B2). Refuses until B2 is finished at 111,280,128 |
| `launch_RC2.sh` | **RC2** (reader of C2). Refuses until C2 is finished at 111,280,128 |
| `launch_RBX.sh` | **RB+** (RB extended +50 %, the convergence side-check). Launchable NOW, since its parent RB and its target B exist |
| `launch_RCX.sh` | **RC+** (RC extended +50 %). Launchable NOW. Run it ADJACENT to RB+ |
| `argv_<arm>.txt` | the exact argv each script launches, built by `scripts/build_argvs.py` from RECORDED `original_command`s: A's for RB2 / RC2, RB's / RC's own for RB+ / RC+ |
| `scripts/_launch_common.sh` | the shared pre-flight: prerequisite FILES exist; the FORK PARENT is finished at exactly the registered step (and, for an extension, the exploiter TARGET too); checkargs + launcher `--dry-run` on every invocation; refuse to clobber a run dir; refuse to share the GPU with a live `train_rl_agent.py` (`ALLOW_CONCURRENT=1` overrides) |
| `scripts/build_argvs.py` | regenerates every argv and prints its diff against its template. It asserts that RB / RC are A's recipe with only the four registered values moved, and that each extension keeps its parent's `--exploiter` and `--fork-lr 2.5e-4 --fork-lr-freeze` |
| `scripts/validate_all.sh` | checkargs + `--dry-run` on all four (RB2 / RC2 on their STAND-IN argvs) + every launch script's `--dry-run` |
| `scripts/argv_R{B,C}2_STANDIN.txt` | the round-2 readers re-pointed at a stand-in target (validation only) |
| `scripts/brgap_ext_standin_sim.py` | builds weight-free stand-in RB+ / RC+ run dirs to exercise `main.best_response_gap` and the rule script on the §4.4 shape |
| `scripts/convergence_rule.py` | the §4.4 table, mechanised (Δ+, Δ_late, both rises, the S / B / C / I row) — a DESCRIPTOR |
| `validation/` | every output quoted below, verbatim (the worktree path is shortened to `<wt>`) |

Every script: `bash <script> --dry-run` resolves and creates nothing; no argument LAUNCHES. Each prints
its STOP-list after launching. The first two minutes are the only test of the preload layer.

## The argv diffs (printed by `scripts/build_argvs.py`)

```
== RB2  ai_v13_29_popr2_read_loop  (230 -> 230 tokens vs ai_v13_18_teach5_offense_hidose)
     --exploiter / --model: 'models/ai_v13_12_plateau/final_model.zip' -> 'models/ai_v13_27_popr2_loop/final_model.zip'
     --run-name -> 'ai_v13_29_popr2_read_loop'
     --steps: '103158272' -> '119280128'
== RC2  (the same four, target models/ai_v13_28_popr2_ctrl/final_model.zip, run ai_v13_30_popr2_read_ctrl)
== RBX (RB+)  ai_v13_31_popr1_read_loop_ext  (230 -> 230 tokens vs ai_v13_24_popr1_read_loop)
     --model: 'models/ai_v13_22_popr1_loop/final_model.zip' -> 'models/ai_v13_24_popr1_read_loop/final_model.zip'
     --run-name -> 'ai_v13_31_popr1_read_loop_ext'
     --steps: '111219200' -> '115280128'
== RCX (RC+)  (the same three vs ai_v13_25_popr1_read_ctrl, run ai_v13_32_popr1_read_ctrl_ext)
```

RB2 against round 1's `argv_RB.txt` differs in exactly `--steps`, `--run-name`, `--model` and
`--exploiter`. RB+ keeps `--exploiter models/ai_v13_22_popr1_loop/final_model.zip` (B), and RC+ keeps
`…/ai_v13_23_popr1_ctrl/…` (C). All four keep A's `--fork-lr 2.5e-4 --fork-lr-freeze`,
`--exploiter-keep-bots --exploiter-bot-fraction 0.5`, the five offense `--trainee-teams`,
`--eval-battles 100`, `--eval-sentinel-greedy`, `--seed 1001`, and `--pin-commit 6eb9c776…`.

## VALIDATED BY EXECUTING (2026-09-24 17:59–18:10 PT, this worktree's `src/`, cwd the main checkout)

| arm | checkargs: parser · accepted / launcher-owned / unrecognized | ForkLR | ARCH surface | dry-run role · steps | result |
|---|---|---|---|---|---|
| RB2 (stand-in) | PINNED `6eb9c776` · 129 / 2 / 0 | `✓ (--fork-lr 0.00025 --fork-lr-freeze)` | ✓ matches `production_config@360f8378dd90` | FORK of `ai_v13_24_popr1_read_loop` · `119,280,128 vs 111,280,128 (sidecar) → +8,000,000` | ✓ would launch |
| RC2 (stand-in) | same | same | ✓ | same | ✓ |
| **RB+ (REAL)** | PINNED `6eb9c776` · 129 / 2 / 0 · 56 inherited from RB's `model_config.json` | `✓` | ✓ | FORK of `ai_v13_24_popr1_read_loop` · `115,280,128 vs 111,280,128 (sidecar) → +4,000,000` | ✓ |
| **RC+ (REAL)** | same, from RC's | `✓` | ✓ | FORK of `ai_v13_25_popr1_read_ctrl` · `+4,000,000` | ✓ |

`launch_RBX.sh --dry-run` and `launch_RCX.sh --dry-run` ran end to end (parent finished at
111,280,128 with "Training complete"; target finished at 103,219,200) and stopped at `--dry-run
requested: stopping here. Nothing was created.` `launch_RB2.sh --dry-run` / `launch_RC2.sh --dry-run`
exit 4: `FATAL: prerequisite models/ai_v13_27_popr2_loop/final_model.zip does not exist`, as designed.

### `main.best_response_gap` at main, on the round-2 shapes

- **The round-1 primary reproduces EXACTLY at main:** RC round 2, RB round 3 → **−10.00 [−16.60,
  −3.27]**. **F3 is fixed** (`26015897`): A + A2 in round 1 POOL to +13.00 [+9.60, +16.28], spread −2.00
  [−8.65, +4.68] (`brgap_r1_with_A2_pooled.txt`).
- **Both chains' round-1 halves** (`brgap_loop_chain_r1half.txt`, `brgap_ctrl_chain_r1half.txt`):
  A → RB −7.00 [−13.68, −0.23]; A → RC +3.00 [−3.58, +9.54].
- **RB + RC `--check`**: 0 mismatches (`brgap_check_RB_RC.txt`).
- **Extension stand-ins** (RB's / RC's own files, forked at 111,280,128, landing 115,310,592, two
  cycles at 112M / 114M, SCENARIO counts): `--check` gives **0 mismatches, budget 4,030,464**, dose
  3.815e-08 on both. The unassisted listing put RB+ in round 1 and RC+ in round 2 (F2 again; the
  registered read passes `--rounds`). The pair read gives half-width ≈ 9.3 pp at n = 200. **RB + RB+
  `--check` REFUSED, `unmatched_budget`, 8,060,928 vs 4,030,464** (rc 2). `convergence_rule.py` on
  the stand-ins prints all four contrasts and a row (`convergence_rule_standin.txt`).

### Gates

`python3 -m pytest src/claude_md_freshness_gate_test.py src/mode_flag_doc_gate_test.py
src/ledger_index_gate_test.py src/file_size_gate_test.py -q` → **29 passed** (before the first commit;
re-run before the second).

### Not done here (findings, for the Training Run session / orchestrator)

- The TRAINING_RUN_SOP §1.3 60-second `--debug` CPU smoke of each reader argv (this session was barred
  from training). The readers are A's recipe, which has launched four times (A, A2, RB, RC) at this pin.
- The real RB2 / RC2 validation against B2's / C2's own `model_config.json` (K-2). The launch scripts
  re-run it at launch time.
- 🚨 **K-1:** `main.best_response_gap` writes `best_response_gap.json` into cwd unless `--json` is
  given. Run from the main checkout, it dirties main. This session did so once and moved the file out.

## Ready-to-append ledger paragraph

> ### 2026-09-24 · OPS · **POPULATION LOOP ROUND 2 IS REGISTERED (`3265ec83`, 17:54 PT, before B2's first eval). B2 (`ai_v13_27_popr2_loop`, the loop: B +8M with RB added to its stable set {A, `ai_v13_13`, RB} at share 0.40) against C2 (`ai_v13_28_popr2_ctrl`, the control: C +8M at share 0.0), each read by a FRESH offense reader at arm A's exact recipe (RB2 `ai_v13_29_popr2_read_loop`, RC2 `ai_v13_30_popr2_read_ctrl`). Primary Δ2 = gap(RB2) − gap(RC2), bar 5.0 (F = 2.00 reused), detection needs Δ2 ≲ −11.7 pp; round 1's own −10.00 would miss again. A convergence side-check (RB+ / RC+, round 1's readers forked +50 %) is a DESCRIPTOR, NOT a re-verdict of round 1.**
>
> The reader recipe stays A's token-exact (frozen `--fork-lr 2.5e-4`, 1.78×, 8,060,928 steps, 100 games a cycle), because anything else makes `main.best_response_gap` refuse the chain. So round 1's H-1 (RB still rising, 0.52 → 0.60) is answered by a separate pair. RB+ = `ai_v13_31_popr1_read_loop_ext` and RC+ = `ai_v13_32_popr1_read_ctrl_ext` are FORKS of each reader's final into new run dirs, +4,030,464 each (matched to each other, never to A), and are read by `scripts/convergence_rule.py` into S (the Δ survives) / B (both still climbing) / C (the Δ closes: every round-2 reading carries "at an 8M reader budget", and a T needs RB2+ / RC2+ before its replicate is spent) / I (inconclusive, the expected case at n = 200). Manipulation check M2 over all THREE specialists (n = 600); 🚨 the per-specialist exposure fell 0.18 → 0.12 at the carried share (P-1), which is branch M's round-3 lever (share 0.60). Guards on B2 − C2, CUMULATIVE over two rounds: untaught 8 on `6eb9c776` (reproduction `plateau_b1` 963/1600) at 3.69; SmallRL greedy on `7c511161` at 0.110, 24 new units. Branches V / K / R / T / N+ / M plus a new **E (EQUIVALENT)** row; stopping rule at 1 of 3, and any of R / N+ / M / E makes it 2, with round 3 the last. Queue: B2 → C2 → RB2 → RC2 → RB+ / RC+ (adjacent, anywhere), ≈ 21 new GPU-h; primary ETA ≈ 11:00 PT 09-25. Validated by executing: pinned parser 129/2/0 on all four reader argvs, the extensions against their REAL parents; the round-1 primary reproduces exactly at main; F3 is fixed at main (`26015897`). 🚨 **K-1: `main.best_response_gap` writes its JSON into cwd by default**, which dirtied main once this session (moved out within a minute). Tag: **OPS · POP LOOP r2 REGISTERED · Δ2 bar 5.0 · power ≲ −11.7 · convergence side-check descriptor · E row added · stopping 1 of 3 · NOTHING LAUNCHED by this session**.
