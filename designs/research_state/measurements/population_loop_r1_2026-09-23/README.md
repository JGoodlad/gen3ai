# Population loop round 1 — launch kit + validation evidence (2026-09-23)

> **THE READ (2026-09-24) is [`read/README.md`](read/README.md)** — manipulation check, both KILL guards,
> the primary Δ, and the §5 branch.

The registration (design, every number's provenance, the bars, the branches, the queue) is
[`../../population_loop_round1_2026-09-23.md`](../../population_loop_round1_2026-09-23.md). This
directory holds what the Training Run session needs to launch it, and the proof that each argv
launches and is the experiment. **Nothing here was executed for real: no arm was launched, nothing
was written under `models/`.**

## Contents

| file | what |
|---|---|
| `launch_B.sh` | **B (the loop)** `ai_v13_22_popr1_loop` — launchable now |
| `launch_C.sh` | **C (the control)** `ai_v13_23_popr1_ctrl` — launchable now |
| `launch_RB.sh` | **RB (reader of the loop)** `ai_v13_24_popr1_read_loop` — refuses until B has finished at 103,219,200 |
| `launch_RC.sh` | **RC (reader of the control)** `ai_v13_25_popr1_read_ctrl` — refuses until C has finished at 103,219,200 |
| `launch_A2.sh` | **A2 (arm A's seed replicate, the reader floor)** `ai_v13_26_popr0_exploit5_offense_s1002` — launchable now |
| `argv_<arm>.txt` | the exact argv each script launches, built by `scripts/build_argvs.py` from the RECORDED `original_command` of `ai_v13_12_plateau` (B, C) or `ai_v13_18_teach5_offense_hidose` (A2, RB, RC) |
| `scripts/_launch_common.sh` | the shared pre-flight: prerequisite FILES exist; a reader's target is finished at exactly 103,219,200; checkargs + launcher `--dry-run` on every invocation; refuse to clobber a run dir; refuse to share the GPU with a live `train_rl_agent.py` (`ALLOW_CONCURRENT=1` overrides) |
| `scripts/build_argvs.py` | regenerates every argv and prints its diff against its template |
| `scripts/validate_all.sh` | checkargs + `--dry-run` on all five (readers on their STAND-IN argvs) + opponent resolution |
| `scripts/resolve_opponents.py` | resolves `--stable-opponents` / `--exploiter` through `fixed_opponent_pool.resolve_stable_opponents` and prints file, step, rung, label and pinned teams |
| `scripts/argv_R{B,C}_STANDIN.txt` | the readers re-pointed at a stand-in target (validation only) |
| `scripts/brgap_standin_sim.py` | builds weight-free stand-in run dirs from A's own files to exercise `main.best_response_gap` on the round-1 shape |
| `validation/` | every output quoted below, verbatim |

Every script: `bash <script> --dry-run` resolves and creates nothing; no argument LAUNCHES. Each
prints its STOP-list after launching — the first two minutes are the only test of the preload layer.

## The argv diffs (printed by `scripts/build_argvs.py`)

```
== B  ai_v13_22_popr1_loop  (238 -> 241 tokens vs ai_v13_12_plateau)
     --model: 'models/ai_v13_09_wcont/final_model.zip' -> 'models/ai_v13_12_plateau/final_model.zip'
     --run-name: 'ai_v13_12_plateau' -> 'ai_v13_22_popr1_loop'
     --stable-opponent-mastered-wr: '0.8' -> '1.01'
     --stable-opponent-pfsp: None -> True
     --stable-opponent-selfplay-share: '0.2' -> '0.4'
     --stable-opponents: None -> 'models/ai_v13_18_teach5_offense_hidose/final_model.zip,models/ai_v13_13_exploit5_offense/final_model.zip'
     --steps: '95097344' -> '103158272'
== C  ai_v13_23_popr1_ctrl  (241 -> 241 tokens vs B)
     --run-name: 'ai_v13_22_popr1_loop' -> 'ai_v13_23_popr1_ctrl'
     --stable-opponent-selfplay-share: '0.4' -> '0.0'
== RB  ai_v13_24_popr1_read_loop  (230 -> 230 tokens vs ai_v13_18_teach5_offense_hidose)
     --exploiter / --model: 'models/ai_v13_12_plateau/final_model.zip' -> 'models/ai_v13_22_popr1_loop/final_model.zip'
     --run-name -> 'ai_v13_24_popr1_read_loop'
     --steps: '103158272' -> '111219200'
== RC  (the same four, target models/ai_v13_23_popr1_ctrl/final_model.zip)
== A2  ai_v13_26_popr0_exploit5_offense_s1002  (230 -> 230 tokens vs ai_v13_18_teach5_offense_hidose)
     --run-name -> 'ai_v13_26_popr0_exploit5_offense_s1002'
     --seed: '1001' -> '1002'
```

B and C keep G0's own `--fork-lr 2.8e-5 --fork-lr-freeze`, `--distill-coef 0.0`, `--seed 1001`,
`--team-block-episodes 1`, `--self-play`, `--pin-commit 6eb9c776…` unchanged from the template.
A2/RB/RC keep A's `--fork-lr 2.5e-4 --fork-lr-freeze`, `--exploiter-keep-bots`,
`--exploiter-bot-fraction 0.5`, `--trainee-teams <five offense>`, `--eval-battles 100`,
`--eval-sentinel-greedy`, pin `6eb9c776`.

## VALIDATED BY EXECUTING (2026-09-23 12:31–12:37 PT, this worktree's `src/`, cwd the main checkout)

### `python -m main.checkargs --argv "…"` — key lines (`validation/checkargs_*.txt`)

| arm | parser | accepted / launcher-owned / unrecognized | refused combos | ForkLR | ARCH SURFACE vs `production_config@360f8378dd90` |
|---|---|---|---|---|---|
| B | PINNED `6eb9c776` | 136 / 2 / 0 | none | `✓ this fork names its own dose (--fork-lr 2.8e-05 --fork-lr-freeze)` | ✓ every ARCH-surface key matches |
| C | PINNED `6eb9c776` | 136 / 2 / 0 | none | `✓ … (--fork-lr 2.8e-05 --fork-lr-freeze)` | ✓ every ARCH-surface key matches |
| A2 | PINNED `6eb9c776` | 129 / 2 / 0 | none | `✓ … (--fork-lr 0.00025 --fork-lr-freeze)` | ✓ every ARCH-surface key matches |
| RB (stand-in) | PINNED `6eb9c776` | 129 / 2 / 0 | none | `✓ … (--fork-lr 0.00025 --fork-lr-freeze)` | ✓ every ARCH-surface key matches |
| RC (stand-in) | PINNED `6eb9c776` | 129 / 2 / 0 | none | `✓ … (--fork-lr 0.00025 --fork-lr-freeze)` | ✓ every ARCH-surface key matches |

All five: `resolved against the FORK PARENT … 56 unset flag(s) INHERITED` and `✓ this command still
launches (its own commit's parser accepts every flag)`.

### `python -m main.launcher --dry-run …` — key lines (`validation/dryrun_*.txt`)

| arm | role | run dir | `--model` | steps | effective dose | result |
|---|---|---|---|---|---|---|
| B | FORK of `ai_v13_12_plateau` | `models/ai_v13_22_popr1_loop` [would be created] | `models/ai_v13_12_plateau/final_model.zip` | `--steps 103,158,272 vs checkpoint at 95,158,272 (sidecar) → +8,000,000` | `--fork-lr 2.8e-05`, `--fork-lr-freeze True` (from the argv); `--distill-coef 0.0`, `--distill-target 'kl'` (INHERITED) | `✓ DRY RUN — this command would launch` |
| C | FORK of `ai_v13_12_plateau` | `models/ai_v13_23_popr1_ctrl` | same | same | same | ✓ |
| A2 | FORK of `ai_v13_12_plateau` | `models/ai_v13_26_popr0_exploit5_offense_s1002` | same | same | `--fork-lr 0.00025`, `--fork-lr-freeze True` | ✓ |
| RB (stand-in) | FORK of `ai_v13_16_teach5_offense_dist` | `…_STANDIN` | `models/ai_v13_16_teach5_offense_dist/final_model.zip` | `--steps 111,219,200 vs checkpoint at 103,219,200 → +8,000,000` | `--fork-lr 0.00025`, freeze True | ✓ |
| RC (stand-in) | same | `…_STANDIN` | same | same | same | ✓ |

All: `pin 6eb9c776… (source: pin_commit)`, transport `in-process bridge [rust]`, restarts every 3.0 h,
nice 10. (`+8,000,000` is the requested delta; the run lands on the next rollout boundary,
+8,060,928.) The real `launch_B.sh --dry-run`, `launch_C.sh --dry-run`, `launch_A2.sh --dry-run` were
also run end to end and stop at `--dry-run requested: stopping here. Nothing was created.`;
`launch_RB.sh --dry-run` exits 4, `FATAL: prerequisite models/ai_v13_22_popr1_loop/final_model.zip
does not exist — this arm cannot launch yet.`

🚨 **The readers were validated against a STAND-IN**, because B's and C's finals do not exist:
`ai_v13_16_teach5_offense_dist/final_model.zip`, a frozen-lr self-play fork of G0 that landed at
**exactly 103,219,200**, the step B and C will land on — only the `--model` / `--exploiter` paths
and `--run-name` differ from the real reader argvs. The real reader scripts re-run checkargs and
dry-run at launch time against the real target.

### Resolved `--stable-opponents` / `--exploiter` files (`validation/resolve_opponents.txt`)

```
== B/C --stable-opponents
   ext_ai_v13_18_teach5_offense_hidose   /home/goodlad/dev/gen3ai/models/ai_v13_18_teach5_offense_hidose/final_model.zip
      step 103,219,200   rung explicit_zip   arch gen3_critic_route_wave_v1
      teams 9eb3abdc52876a63.txt, ac17a9dde5.txt, a185b2d193.txt, 9ba039ba8a.txt, b904dbe059.txt
   ext_ai_v13_13_exploit5_offense        /home/goodlad/dev/gen3ai/models/ai_v13_13_exploit5_offense/final_model.zip
      step 103,219,200   rung explicit_zip   arch gen3_critic_route_wave_v1
      teams (the same five)
== A2 --exploiter      ext_ai_v13_12_plateau  …/ai_v13_12_plateau/final_model.zip  step 95,158,272  explicit_zip  (pilots the shared POOL)
== stand-in --exploiter ext_ai_v13_16_teach5_offense_dist … step 103,219,200 explicit_zip (the five offense teams)
```

Both specialists resolve to their FINAL files by explicit zip (not a last-snapshot rung) and each
will pilot its own five pinned teams (`env_factory.py` builds a per-opponent `Gen3Teambuilder` from
`e.team_strs`; `wrappers._apply_opponent_team` switches it per episode — both present at `6eb9c776`).

### `main.best_response_gap` on the round-1 shape (`validation/brgap_*.txt`)

- **A vs `ai_v13_13`, `--check`: REFUSED, rc 2** — `DOSE MISMATCH … 3.815e-08 vs 8.392e-09 (4.55x
  apart, tolerance 10%)`. `ai_v13_13` cannot serve as A's replicate; hence A2.
- **Stand-in readers** (A's own `metadata.json`, sidecars and `eval_results.jsonl`, re-targeted at
  stand-in generalists at 103,219,200, win counts set to a SCENARIO — not a prediction):
  `--check` → `0 mismatch(es); 3 run(s) read`, every row `budget 8060928 · dose 3.814697265625e-08 ·
  archetype offense`. With `--rounds RC=2 RB=3` the `round 3 − round 2` row is gap(RB) − gap(RC):
  scenario −8.25 pp, Newcombe **[−14.93, −1.46]** (half-width ≈ 6.7 pp at n = 400 per reader), and
  the tool's VERDICT line reads `UNREADABLE — fewer than two archetypes` (finding F1).
- **Same-round collapse** (finding F3): A plus a stand-in A2 (scenario 0.70) in one invocation →
  both rows print in round 1, but the delta to round 2 used A2's +20.00 and silently dropped A's +14.00.

### Gates

`python3 -m pytest src/claude_md_freshness_gate_test.py src/mode_flag_doc_gate_test.py -q` →
**8 passed** in 0.94 s. `python -m main.baselines check` → `OK — 8 baseline(s), 1 list(s)` (G0 is
not among them — finding F9).

### Not done here (findings, for the Training Run session / orchestrator)

- The TRAINING_RUN_SOP §1.3 **60-second `--debug --steps 8000` CPU smoke** of each argv — this
  session was barred from training. **C matters most**: stable opponents loaded at share 0.0 is a
  configuration no banked run has used.
- `git submodule update --init` failed in this worktree (the clone could not be fetched); the
  `dist` / `node_modules` links were made, and every command above ran.

## Ready-to-append ledger paragraph

> ### 2026-09-23 · OPS · **POPULATION LOOP ROUND 1 IS REGISTERED, NOT LAUNCHED — B (the loop: G0 +8M at its own frozen 0.20×, the two offense exploiters of G0 as stable opponents at share 0.40, PFSP on, retirement off, no distillation) against C (the control: B's argv with the share at 0.0), each read by a fresh offense exploiter at arm A's exact recipe; arm A ALREADY EXISTS as `ai_v13_18_teach5_offense_hidose`, so D_e = 1.78× and round 0 costs nothing.**
>
> `designs/research_state/population_loop_round1_2026-09-23.md` registers it before any arm exists; the launch kit and its validation are in `measurements/population_loop_r1_2026-09-23/`. **A = `ai_v13_18_teach5_offense_hidose`**: it is `ai_v13_13`'s recorded command plus a named, frozen `--fork-lr 2.5e-4` — the design's definition of arm A — and it reads **gap +14.00 pp [+9.18, +18.55]** (pooled 256/400), its vs-target curve FLAT from the first cycle (0.65/0.58/0.67/0.66) where the 0.39× arm was still climbing, so a 1.78× reader measures a converged best response. **D_g = 0.20×** (`--fork-lr 2.8e-5 --fork-lr-freeze`, G0's own): C is then the plateau parent's next block (the last one was −1.50 pp [−3.75, +0.62] on the untaught 8), and the one loss-off precedent absorbed at this dose (`ai_v13_11_split_lossoff`, parent ~0.26 → 0.54/0.50 against its stable exploiters) — but it also carried `--distill-team-bias 0.4`, so absorption WITHOUT team bias is not established and a manipulation check is registered. Stable set **{A, `ai_v13_13`}**, not {A, balance, stall}: balance (0.525) and stall (0.455) carry no detected gap and would take ≈ 60 % of the slice. Mix, from the code and checked on `ai_v13_17_fold_k1`'s TB (0.72 / 0.18 / 0.90 at share 0.20): **B = bots 0.10 · pool 0.54 · stable 0.36 (≈ 0.18 per specialist); C = bots 0.10 · pool 0.90 · stable 0.00**. `--stable-opponent-mastered-wr 1.01` VERIFIED inert-safe (its only reader is `wr >= threshold`, `wr` ≤ 1.0). Queue after `ai_v13_21_wcont_b`: B → C → RB → RC → A2 (the reader floor, seed 1002), **≈ 21 GPU-h**, verdict ≈ 17:00 PT 09-24. Primary **Δ = gap(RB) − gap(RC)**, Newcombe, bar max(|gap(A) − gap(A2)|, 5.0 pp); 🚨 at n = 400 per reader the half-width is ≈ 6.7 pp, so a detection needs Δ ≲ −11.7 pp and EQUIVALENCE is unreachable — round 1 tests for a LARGE effect. Guards (B − C, KILL on OUTSIDE BELOW): untaught 8 at 3.69 pp with `plateau_b1` reproducing 60.19; SmallRL greedy 1,200 games at the 0.110 run floor. Six branches (void / kill / rose / turns-as-candidate / not-detected-absorbed / not-engaged) and the stopping rule (three rounds not below ⇒ the loop does not turn at this parent). 🚨 **FOUR METER/TOOL FINDINGS**: `main.best_response_gap` issues no verdict on a one-archetype round (F1), knows rounds not siblings so B and C need `--rounds RC=2 RB=3` (F2), and SILENTLY keeps only one of two same-archetype exploiters of one target (F3, demonstrated); `launcher --dry-run` printed ✓ on a stand-in argv that `checkargs` flagged `WOULD FAIL IN resolve_config` (F5). Validated by executing: checkargs 136/2/0 (B, C) and 129/2/0 (A2, readers on a stand-in at 103,219,200), ARCH surface clean on all five, dry-run ✓ on all five, pin `6eb9c776`; freshness + mode-flag gates 8 passed. Tag: **OPS · POPULATION LOOP ROUND 1 REGISTERED · A = hidose (exists, +14.00 pp) · D_e 1.78× · D_g 0.20× · {A, ai_v13_13} at 0.40 · C = share 0.0 · ≈ 21 GPU-h · Δ needs ≲ −11.7 pp to detect · 4 tool findings · NOTHING LAUNCHED**.
