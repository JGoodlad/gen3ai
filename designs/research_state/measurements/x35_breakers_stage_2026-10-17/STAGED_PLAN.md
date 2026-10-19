# X35, plateau-breakers: STAGED ON PAPER (2026-10-17)

**Status: STAGED, not launched; every launch NEEDS THE ORCHESTRATOR'S GO.** The spec is
[`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6. Its backlog rows are
`EXPERIMENT_BACKLOG.md` CE-3 to CE-3g, and its builds are `TASK_BACKLOG.md` T54–T59. X35 starts when X26 (`rb_x26_s2601`)
and X34 (`rb_toy_ent_s4002_deep`) have both reached their Tier-1 plateau and the GPU is free. **The stage kit (T55)
is BUILT (2026-10-17, §6):** the launch script, the R freeze, the readout and the Kakuna runner.

**D1 (X38, the pool-widening arm) is staged too (2026-10-19, §7):** its argv is `main.x35.kit` arm `d1`, dry-run
validated on the REAL P\* at pin `c1b0c1f0`; it NEEDS GO like every arm.

**R (production) is FROZEN (2026-10-19)** at P\* = X26 @ 280,000,170 (Amendment 5's `TIER1_PLATEAU` call, sha256
`814823c5…91ed`): [`reference_population_production.json`](reference_population_production.json), content sha256
`39bae18f…5bec0`, `refpop verify` OK. Six members: A-block 280 / 270 / 80 / 140 / 210M, and X-block `production`
(band score 0.6935 at 500 pairs, KEPT). **T\* is ABSENT** (X34 not launched): the X-block has one member, the toy
side's R is not frozen, and adding T\* needs a new freeze (F-R1). Write-up:
[`../x35_refpop_freeze_2026-10-19/README.md`](../x35_refpop_freeze_2026-10-19/README.md). Nothing was launched.

**How this was staged.** Everything ran at `9ebda190` with `CUDA_VISIBLE_DEVICES=""`, under `nice -n 19 taskset -c
5-7,13-15`, with no GPU lease. It used only `main.checkargs`, `main.launcher --dry-run` and one offline call to
`resolve_stable_opponents`. Nothing was created under `models/`, and nothing was written into X26's directory.

## 1. The stand-ins, and what replaces them at the plateau call

| stand-in here | at the call |
|---|---|
| P\* = `models/rb_x26_s2601/checkpoints/checkpoint_180000055_steps.zip` (the newest 10M node at staging) | the newest node of X26's second FLAT Tier-1 check, sha256 frozen |
| pin `9ebda1904da7a7782f22de2495b6e4c51368510b` | `X35_PIN_1`: a main commit carrying T54 (and T56 / T57 if they landed) |
| A1's archive = 20, 40, …, 160M (`a1_archive_spec_standin.txt`) | N_k = the 10M-multiple checkpoint nearest k·(P\* − 20M)/8, k = 1..8, ties down |
| `--steps 200000055` | P\* + 20,000,000 (production), T\* + 70,000,000 (toy) |
| toy C0 from `rb_toy_ent_s4002/final_model.zip` (100,081,428) | T\* in `rb_toy_ent_s4002_deep` |

## 2. The validated argvs (launcher flags: `--restart-interval-hours 24 --pin-commit <PIN>`)

| arm | argv file | `checkargs` | `launcher --dry-run` |
|---|---|---|---|
| **A1 s3501 (the FIRST arm)** | [`argv_a1_s3501_standin.txt`](argv_a1_s3501_standin.txt) | exit 0 ([`checkargs_a1_9ebda190.txt`](checkargs_a1_9ebda190.txt)) | "would launch", FORK of `rb_x26_s2601`, +20,000,000 ([`dryrun_a1_9ebda190.txt`](dryrun_a1_9ebda190.txt)) |
| C0 s3501 | [`argv_c0_s3501_standin.txt`](argv_c0_s3501_standin.txt) | exit 0 | "would launch", +20,000,000 |
| A2 s3501 | [`argv_a2_s3501_standin.txt`](argv_a2_s3501_standin.txt) | — | "would launch", +20,000,000 |
| A3 phase 1 (main exploiter) | [`argv_a3_main_exploiter_standin.txt`](argv_a3_main_exploiter_standin.txt) | — | "would launch", +5,000,000. **Without `--no-self-play` it is REFUSED** (F16) |
| A6 phase 1 (league exploiter) | [`argv_a6_league_exploiter_standin.txt`](argv_a6_league_exploiter_standin.txt) | — | "would launch", +5,000,000 |
| toy C0 | [`argv_toy_c0_standin.txt`](argv_toy_c0_standin.txt) | — | "would launch", FORK of `rb_toy_ent_s4002`, +70,000,000 |

**A1's archive resolves.** `resolve_stable_opponents` was called against X26's `ModelVersion`, offline
([`a1_archive_resolution_standin.txt`](a1_archive_resolution_standin.txt)). It returned 8 entries, rung
`explicit_zip`, with 8 unique labels (`ext_x26_020m` … `ext_x26_160m`). The arch gate passed. The dry-run cannot show
this, because the stable-opponent resolver runs in the child.

**What the dry-runs show for every production fork:**
- the 12 recipe knobs RESTORED from X26's record and passed typed (`n_envs=256 … n_epochs=10 … ent_coef=0.05
  self_play=True`);
- `--lr` / `--batch-size` / `--n-steps` INERT;
- the ARCH SURFACE matching the production mirror;
- a bf16 trunk;
- 20.68 GiB of disk REQUIRED against ≈ 318 GiB free;
- the Showdown deps present;
- no display process holding the GPU.

The first arm's block, with its placeholders, is in the design doc's §5.6.8. The literal stand-in line is:

```
# NEEDS GO. Stand-in values. Never run as written.
python -m main.launcher --restart-interval-hours 24 --pin-commit 9ebda1904da7a7782f22de2495b6e4c51368510b \
  $(cat designs/research_state/measurements/x35_breakers_stage_2026-10-17/argv_a1_s3501_standin.txt)
```

## 3. Pre-checks at the call (a failure is a HOLD; [`launch_x35.sh`](launch_x35.sh) runs them, §6)

1. X26 and X34 have stopped, and the GPU lease is the Training session's to grant.
2. `X35_PIN_1` contains T54, the pool seed's step filter (F18).
3. No `models/rb_x35_*` directory for the arm exists.
4. The sha256 of P\* and of every archive node is unchanged since the call.
5. `checkargs --pin X35_PIN_1` exits 0.
6. The dry-run says "would launch", FORK of `rb_x26_s2601`, +20,000,000.
7. The reference population R is frozen and its file is written before the first arm launches (§5.6.4). **Production: DONE 2026-10-19** (`reference_population_production.json`; toy: pending T\*).
8. The disk has room for the whole batch, or finished arms are being thinned (F22).

## 4. The batch-1 queue (production first, then the toy calibration)

| # | run | seed | GPU-h (est.) |
|---|---|---|---|
| 0 | R0: offense-archetype best responder of P\* | 3531 | ≈ 1.1 |
| 1–6 | A1 s3501 · C0 s3501 · C0 s3502 · A1 s3502 · A1 s3503 · C0 s3503 | 3501–3503 | 6 × 4.4 ≈ 26 |
| 7–9 | A2 s3501–s3503 | 3501–3503 | ≈ 13 |
| 10–15 | toy A1 / C0 in the same ABBA order | 3601–3603 | ≈ 26 |
| 16–18 | toy A2 | 3601–3603 | ≈ 13 |
| — | R2 readers (1 per arm final, seed 1) and the population reads, in the windows between arms | 3532+ | ≈ 1.4 per arm |

The first A1-vs-C0 read lands after ≈ 27 GPU-h. A3 (≈ 18 GPU-h) follows in the same batch if the orchestrator wants
it before the toy calibration. That is a tactical call (`feedback_gpu_scheduling_is_mine`).

## 5. Findings at staging

These are F13–F24 in the design doc. The four that bind a launch:
- **F14.** A fork defaults to `--behaviour-check fatal`, so the argv types `warn`.
- **F16.** An exploiter fork needs `--no-self-play`.
- **F18.** The fork pool seed leaks X26's post-P\* snapshots; T54 fixes it. **T54 shipped as `cdab9cfe`** (registered in
  [`x35_required_commits.txt`](x35_required_commits.txt)). Its seed runs in the child, so `launch_x35.sh` checks it
  AFTER the spawn (§6).
- **F31 (A2 only, a KNOWN BIAS, not fixed).** T54 filters the snapshots and prunes `pfsp_win_rates` to the KEPT
  steps, but the pool's scalar state is the parent's LATEST, not its value at P\*: `win_rate_vs_bots`, the
  `pfsp_win_rates` values of the kept snapshots (measured by X26 after P\*), and `pool_generation`. A2's early PFSP
  weights therefore start from post-P\* win rates. Every other arm reads the pool's scalars only through the
  self-play fraction, and b-ON and b-OFF share the bias. A2's read must declare it.
- **F21.** The home72 config lived outside the repo. **Closed by T55:** it is committed here (§6).

## 6. The stage kit (T55, built 2026-10-17)

Code in `src/main/x35/` (tests beside it); this directory holds the script, the committed config and the records.
Nothing below has been run on a real P\*: P\* is not called yet.

**(a) The launch: [`launch_x35.sh`](launch_x35.sh)** `<arm> <seed> --pin <X35_PIN_k> [--lease-owner NAME]
[--phase1 FILE] [--thinning] [--at-epoch EPOCH] [GO]`.
- ONE arm × seed per call. The default is a DRY RUN: every pre-check of §3, then the launch line printed. Only the
  literal `GO` launches, and a GO needs the orchestrator's go.
- The argv is BUILT by `python -m main.x35.kit describe` from the frozen R file of the arm's side, never typed by
  hand. Every production argv equals its stand-in in §2 flag for flag (`src/main/x35/kit_test.py`).
- The first failed pre-check is a HOLD (exit 7, written to `~/.claude/jobs/x35/status.txt`). Pre-check 2 reads
  [`x35_required_commits.txt`](x35_required_commits.txt): it must name T54's commit, so **until T54 lands and its
  commit is written there, every launch HOLDs.**
- `--at-epoch E` arms an ABSOLUTE epoch in [now, now + 24 h], runs the pre-checks at arming AND again at E. A
  launcher that dies within 180 s is a HOLD.
- **The POST-SPAWN pool-seed check (T54).** A dry run cannot preview the seed, because it runs in the child. So once
  the launcher is up, the script polls `python -m main.x35.kit pool-seed-check` for up to 30 min. It reads the fork's
  `snapshots/pool_seed.json`, which must be schema 2 with a NON-NULL `fork_step`. T54 only WARNS and seeds unfiltered
  when it cannot read the step, so a null is a HOLD here. `fork_step` must equal P\*'s `num_timesteps` and the call
  record's step, every seeded snapshot must be at or below it, and the startup line `[step filter <= N: DROPPED …]`
  must agree with the record. A FAIL, or no record in 30 min, SIGTERMs the launcher BY ITS PID (it saves and stops
  its child) and HOLDs. `a3_mx` (`--no-self-play`) seeds nothing and is exempt. Tested on constructed records
  (`kit_test.py`); never run on a real fork.
- **The two-phase kits.** Phase 1: `a3_mx` (seeds 3511–3513) or `a6_lx` (3521–3523), +5M each. When all three
  exploiters have ENDED, `python -m main.x35.kit phase1-record --arm a3|a6 --call <R> --out phase1_a3.json` writes a
  write-once record of their finals (path, step ≥ P\* + 5M, sha256). Phase 2 (`a3` / `a6`, seeds 3501–3505)
  reads it and verifies its sha256s before launching.
- **Validated** ([`kit_validation_acb99527.txt`](kit_validation_acb99527.txt)): on a stand-in R (P\* = 180,000,055)
  at pin `acb99527`, `a1`, `c0`, `a2`, `a3_mx` and `a6_lx` passed `checkargs` and "would launch", FORK of
  `rb_x26_s2601`, +20,000,000 / +5,000,000. That is the first `checkargs` of A2 and both exploiter argvs. The real
  script HOLDs at pre-check 1 while X26 is live, so checks 2–8 ran on a test copy without check 1. At today's disk
  pre-check 8 HOLDs batch 1 without `--thinning` (F28).

**(b) The R freeze: `python -m main.x35.refpop plan | freeze | verify`.**
- `freeze` takes `--side`, `--pstar`, every `--archive-dir` and the X candidates (`--x tstar=<zip>`,
  `--x production=baseline:production`), plus P\*'s band cells (`--band-ledger <root> --band-family F
  --band-request Q`, or `--band-json`). It writes `reference_population_<side>.json` here, ONCE, with every sha256
  and a content hash. `plan` prints the nodes and the `main.h2h play-many` command for the band cells.
- Tested on synthetic archives only (`refpop_test.py`), including P\* = 180M → archive 20…160M, A-block 180, 170, 50,
  90, 130M. A start node at or below 150M is REFUSED (F25): the toy side may need a rule first.

**(c) The readout: `python -m main.x35.readout decide <read.json> | self-check`.**
- R1, R2, G-F, G-E, G-C, G-K and the verdict, as pure functions; every declared choice is in the module's docstring
  and in F29.
- **The self-check ran BEFORE any real read: 14 / 14 PASS** ([`readout_selfcheck_t55.txt`](readout_selfcheck_t55.txt)).
  Welch was checked by hand (df 4 exactly) and against scipy's own interval (to 6e-14). Its 97.5 % coverage over
  4,000 draws per cell is 0.977–0.984 at 3 v 3 and 0.972–0.984 at 5 v 5, against a floor of 0.96. Newcombe was
  checked against the published 1998 method-10 table (to 5.1e-5, the 4-d.p. rounding). Every row of the class,
  guard and verdict tables was checked, with rule 8 at ±2.0, at 0 and at −0.05. The population score is exact on
  constructed cells.

**(d) The Kakuna runner: `python -m main.x35.kakuna plan | run | status | gk | smoke | manifest`.**
- It runs on the COMMITTED [`anchors_home72_kakuna.json`](anchors_home72_kakuna.json), byte-identical to the X26
  read's (sha256 `e8a064db…`), and [`home72_manifest.json`](home72_manifest.json), which pins the 72 team files on
  both sides. `metamon_team_dir` is Metamon's cache, outside the repo; no `data/` file is used.
- Before every unit it checks the T28 halt, the manifest and the model's sha256. It runs from the checkout root under
  rule 5b, on a free 95xx port it picks itself. A unit counts by §5.6.6's rule, and `gk` feeds G-K.
- **The smoke** ([`kakuna_smoke_2026-10-17.json`](kakuna_smoke_2026-10-17.json)): ONE 2-game unit, X26 @161,000,169's
  read-only copy against `metamon:SmallRL`, greedy vs greedy, on home72, port 9540, in 73 s. It came back `status OK`,
  `n 2`, `regime_verified_decisions`, `peer_clean`, no asymmetry: COUNTED. Its 0–2 is not a measurement.

**(e) The D1 / X38 readout kit (2026-10-19, B-D7 + the D1 rule; `design_diversity_teams_opponents.md` §4.2).**
- `python -m main.x35.kakuna plan --teamsets unfamiliar | panels | d1h --on X38 --off C0`: the Kakuna runner on the
  three unfamiliar panels (heldout72_hpfix, forum72, p72; 7 units × 20 games per panel per seed final), every panel
  checked against `heldout_panels.json` by content before each unit; panel units record per-game teams
  (`--capture-dir`). `home72` / `away` and G-K are unchanged.
- `python -m main.h2h play | play-many --panel home72 …`: mirrored pairs on a REGISTERED panel (B-D7), the panel's
  content digest in the regime's `team_set`; the input of D1's co-primary P (`python -m main.x35.d1 p-input`).
- `python -m main.x35.readout decide-d1 <read.json>` and `self-check --suite d1`: **10 / 10 PASS before any read**
  ([`readout_selfcheck_d1.txt`](readout_selfcheck_d1.txt)).

## 7. The D1 arm (X38, the pool-WIDENING fork), dry-run validated on the REAL P\*, NEEDS THE ORCHESTRATOR'S GO

**Status: LAUNCH-READY ON PAPER (2026-10-19), nothing launched.** D1 is X35 rank 2 (after A1, before A2;
`design_diversity_teams_opponents.md` §4.2, §6.1). Its ONE lever, b-ON vs C0: `--team-mix w_forum_v1@ABC:0.5
--team-mix-sides both`, i.e. half of each side's draws come from the pin-safe forum corpus
(`src/utils/team_corpora/w_forum_v1/`, 2,424 human-built teams since the 2026-10-19 crash-class lift (2,415 at `fc467f1f` / `c1b0c1f0`), minus every registered panel and its near neighbours);
the in-loop eval stays on the pool. Its verdict is D1's OWN rule (`main.x35.d1`), not R1.

| item | value |
|---|---|
| P\* | `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed` (R's frozen call) |
| pin | a main commit carrying `c1b0c1f0` (B-D2, `--team-mix`) and so `fc467f1f` (the corpus) and `cdab9cfe` (T54): both are in [`x35_required_commits.txt`](x35_required_commits.txt), so pre-check 2 HOLDs a pin without them. **Validated at pin `c1b0c1f0`** |
| seeds | 3501–3503 (look 1), 3504–3505 (look 2); shared with C0 |
| argv (s3501) | [`argv_d1_s3501.txt`](argv_d1_s3501.txt), built by `python -m main.x35.kit describe --arm d1 --seed 3501 --call reference_population_production.json` |
| `checkargs --pin c1b0c1f0` | exit 0, 9 flags accepted by the PINNED parser ([`checkargs_d1_c1b0c1f0.txt`](checkargs_d1_c1b0c1f0.txt)) |
| `launcher --dry-run` | "would launch", FORK of `rb_x26_s2601`, +20,000,000 steps (280,000,170 → 300,000,170), 13.53 GiB required vs 296.59 GiB free, Showdown deps present ([`dryrun_d1_c1b0c1f0.txt`](dryrun_d1_c1b0c1f0.txt)) |
| batch-1 order | `main.x35.kit` `BATCH1`: A1 / C0 ABBA (6), then **d1 s3501, s3502, s3503**, then A2 × 3 |

```
# NEEDS GO. A design block is not a launch command. Pin: any main commit carrying c1b0c1f0 (validated AT c1b0c1f0).
launch_x35.sh d1 3501 --pin <X35_PIN_k>          # dry run: every pre-check, then the line below; add GO only on the orchestrator's go
python -m main.launcher --restart-interval-hours 24 --pin-commit <X35_PIN_k> \
  --model models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn \
  --steps 300000170 --checkpoint-every-steps 1000000 --device cuda \
  --team-mix w_forum_v1@ABC:0.5 --team-mix-sides both --seed 3501 --run-name rb_x35_d1wide_s3501
```

The control is C0 s3501 (the same line without the two `--team-mix` flags, `--run-name rb_x35_c0_s3501`), at the SAME
pin. Startup banners to expect beyond §5.6.8's: `🧬 [TEAM-MIX] w_forum_v1@ABC (2424 teams, tier sha 7bcdc835c107)
share 0.50 on both sides; eval stays on the pool` with **0 panel hits** and 0 pool copies, and a `🧭 [MATCHUP]` hash
that differs from C0's while the eval line still reads "full pool + 10% sample-team bias". (A pin that predates the lift commit reads the old corpus: `2415 teams, tier sha 65465d25252e`; the banner decides which corpus the run trained on, and the C tier is a DIFFERENT 2,415 vs 2,424 sample, not a superset.) A dry run cannot show either
(they are child-side): read them in the first minute. A corpus that fails verification, a held-out panel hit or a
validator drop is a startup `FATAL_CONFIG`.

**The readout (§6 (e)), and what it still needs at read time:** H = `kakuna plan --teamsets unfamiliar` + `run` on
each D1 and C0 seed final (≈ 2,520 games per arm, ≈ 6.3 lane-hours); P = an ab family registered, then `main.h2h
play-many --panel home72` each final vs R's 6 members (500 pairs each: ≈ 18k games per arm, ≈ 10 GPU-minutes or
≈ 8.6 CPU-hours), then `main.x35.d1 p-input`; the guards G-F / G-E / G-C from X35's tools, assembled into the read
JSON by hand (no assembler exists). L / L_c1, GEN-M and the D0e transfer read are reported-only and not built.
