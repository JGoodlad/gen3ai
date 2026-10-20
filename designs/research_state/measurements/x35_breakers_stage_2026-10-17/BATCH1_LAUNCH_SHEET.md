# X35 batch 1 + X47: the LAUNCH SHEET (staged 2026-10-19, nothing launched)

**Status: STAGED and AMENDED (§0, 2026-10-19, before any X35 launch); T\* RESOLVED (X34 plateaued, 2026-10-19: "Words used here"); the
toy R is FROZEN on the real T\*; every argv of the amended order (32 forks, the toy on the real T\*) validated at main
HEAD `03b983be` (§4.3); nothing launched; every launch NEEDS THE ORCHESTRATOR'S GO.** X47 is a **kit arm**
(`toy_x47`, F-LS-1 closed). The orchestrator has ruled **C1** (a), **C2** (X47 interleaved with its toy C0), **C3** (the
toy arms run K9(b) `warn` under X34's ceilings), **C4** (batch 1 launches without R0) and every amendment of the
[adversarial review](X35_BATCH1_REVIEW.md) (AM-1 to AM-7, the POOL check, the G-E readers); C5 and C6 are still open
(§5). This sheet is
what the Training session runs, in order, once X34 has stopped and the GPU window (the battery, then the X14 A/B) is
over. The design is [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md)
§5.6 (X35) and [`design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §2.1 (X47); the kit is
[`STAGED_PLAN.md`](STAGED_PLAN.md) §6. A design block is not a launch command: the lines below are the launch commands
only after GO, with `<PIN>` and (toy side) the T\* values filled in. `<PIN>` = main HEAD at launch, a descendant of the
kit commit that carries `toy_x47` and the toy `warn` (a pin without it HOLDs `launch_x35.sh` at `describe` for X47 and
launches the toy arms under `fatal`: take no earlier pin).

**Words used here.**
- **P\*** = X26's plateau checkpoint, `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256
  `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed` (R's frozen call).
- **T\*** = X34's plateau checkpoint, **RESOLVED 2026-10-19**: X34 (`rb_toy_ent_s4002_deep`) PLATEAUED and is stopped (the
  orchestrator's news of 2026-10-19; the run dir holds `final_model_interrupted.zip` from 23:00 PDT and a last
  checkpoint at 292,000,216; no launcher is live, `kit live` exits 0; the Tier-1 record behind the call is the
  plateau call's, not this sheet's). T\* is the **290M node**,
  `models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip`, step **290,000,087**, sha256
  `cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7` (`cfa7f7be…a22965e7`; re-hashed here on the archive
  file AND on the verified copy in `~/.cache/gen3ai/x34_plateau/`, whose `SHA256SUMS` lists it). Every toy line below
  is built from the toy R file frozen on it (`reference_population_toy.json`, §4.2). The earlier 200M and 210M
  guesses (a stand-in `checkpoint_200000244_steps.zip`, sha256 `bb6271a3…0de057`; the 210M node of a hypothetical
  second FLAT check) are RETIRED: no sheet line, argv or validation below uses them, and the old stand-in validations
  are kept as history only (`batch1_validation_803a13e3/`, `batch1_validation_9c3bf3c4/`, `batch1_validation_d825c6a1/`).
- **C0** = the plain continuation control (the same fork with no lever). An arm is read only against the C0 of its own
  start checkpoint AND its own pin.
- **b-ON / b-OFF** = the arm with its lever / its C0, at the same seed.

---

## 0. AMENDMENT 2026-10-19: the orchestrator's rulings on the review, made BEFORE any X35 launch

The [adversarial review](X35_BATCH1_REVIEW.md) (`6a6e6db4`) read GO-WITH-AMENDMENTS. The orchestrator ruled every
amendment the same day; each is BUILT into the kit (`src/main/x35/batch.py`, `kit.py`, `x47.py`, `d1.py`) and
enforced by `launch_x35.sh` where it gates a launch. A HOLD is exit 7, as before.

| ruling | what it says | where it is enforced |
|---|---|---|
| **AM-1 ACCEPTED (blocking): ONE pin** | X35_PIN_1 is ONE commit, written ONCE to `x35_pin_batch1.txt` (this directory) (sha, date, the T62 verdict) by the FIRST launch's GO, after its pre-checks. Every later launch's `--pin` must EQUAL it, and every existing batch-1 run's RECORDED pin (`metadata.json` `git_hash` + every `pin_history` span) must equal it; a batch-1 run whose pin cannot be read HOLDs too | pre-check 2b (`kit pin-check`); the GO writes the file (`--t62 <verdict>` is required on the first launch). A later batch's `rb_x35*` runs carry their own pin and are listed, not judged |
| **AM-2 ACCEPTED: the variance gate G1** after row 6 | σ̂ = the pooled within-arm SD of S over the six finals A1 / C0 × s3501–s3503 (4 df). **PASS iff σ̂ ≤ 1.2 pp**; above, or within 1e-9 of 1.2 (rule 8), is **HOLD**: D1, A2 and the toy calibration wait for the orchestrator's re-plan, C0 s3504–s3505 run next and σ̂ is re-estimated on 5 C0 + 3 A1 (6 df, reported). It reads variance only, never an arm's Δ̂. A1's own look-1 R1 is read as registered either way | `python -m main.x35.kit gate <s.json> --stage <stage> --write` (exit 0 PASS / 1 HOLD; written once to `x35_gate_g1.json`); pre-check 9 reads it. The zero-training pre-read on X26 290–380M is a separate agent's (review F-RV-2) |
| **AM-3 ACCEPTED** | toy A1 (resp. toy A2) launches only after production A1 (resp. A2) has a FINAL class GAIN, NULL or LOSS; an UNDECIDED production read ends that arm's calibration as NOT SHOWN, with no toy fork. The toy C0 still runs (X47 needs it) | pre-check 9 (`kit stage-check`), from the saved read `reads/<arm>_look<k>.json` |
| **AM-4 ACCEPTED** | X47 seeds 3601–3605: look 2 = X47 s3604–s3605 against toy C0 s3604–s3605 (the toy C0 seeds were already registered; the pairing is seed for seed), launched only on X47's look-1 CONTINUE. **B-X47-READ BUILT:** `main.x35.x47` (s_eq at equal effective learn time, the matched-step secondary, the PER-STEP ONLY class) | `kit.X47_SEEDS`; pre-check 9; `python -m main.x35.x47 seq / decide` |
| **AM-5 ACCEPTED** | D1's H interval widened by √max(1, φ̂), the seed overdispersion | `main.x35.d1` (self-check 11 / 11) |
| **AM-6 ACCEPTED** | K9(b) is read at EACH arm's end over the fork's OWN rows (steps > its fork step; the inherited rows are X26's), deterministically: judged max ≥ 1e-3, p99 ≥ 1e-4, any non-finite value or the child log's NON-FINITE K9(b) line ⇒ the arm is **VOID** for every read and is re-run at the same pin and seed after the cause is fixed; within 1e-9 relative of a ceiling is recorded, no fire; no post-fork row is INCONCLUSIVE, never a pass | `python -m main.x35.kit k9b --run-dir models/<run> --stage <stage>` (exit 0 PASS / 1 VOID / 3 INCONCLUSIVE), run by the Training session at arm end; pre-check 11 (`kit k9b-batch`) reads (or records) it for every FINISHED batch-1 arm before the next GO |
| **AM-7 ACCEPTED: the order** | the first 6 arms → G1 → C0's look-2 seeds and A1's look 2 → D1, A2 → toy C0 + X47 → X47's look 2 → the conditional toy calibration (§1) | `kit.BATCH1` (= `batch.STAGES`, pinned by `batch_test.py`); pre-check 9 for every conditional stage |
| **POOL** | every arm of a side seeds the SAME snapshot set: the first launch of the side records it (`x35_pool_batch1_production.json` (this directory; the toy side's `x35_pool_batch1_toy.json`): names, sha256, digest), every later launch recomputes it and HOLDs on any difference, and the post-spawn check compares the fork's `pool_seed.json` with it. **C0 = P\* continued with X26's pool as of today: 14 snapshots, 164–268M** (T54 drops the 6 post-P\* ones; the newest pre-P\* selves, 270–280M, were evicted before the call), under T62. Not "X26 unchanged" (review F-RV-8) | pre-check 10 (`kit pool-check`); the post-spawn `kit pool-seed-check --stage` |
| **G-E readers** | the CHEAPER honest option: **not launched in batch 1.** An arm's R1 (D1: H) GAIN reads **INCONCLUSIVE, never ADOPT-CANDIDATE,** until G-E is read (`readout.decide`, as registered). The readers (one fresh best responder per (arm, seed-3501 final), C0 included: ≈ 1.1 GPU-h each, ≈ 2.2 per arm) are registered, dry-run validated and launched ONLY when an arm's primary reads GAIN, at X35_PIN_1. G-E decides only between ADOPT-CANDIDATE and TRADE, so a reader for a non-GAIN arm buys nothing. Their argv is still unregistered (F27) | `readout.decide` (a GAIN with G-E absent is INCONCLUSIVE) |
| **C1 RULED: (a)** | production launches at main HEAD with T62, pending the GPU battery's read; its measured cost is recorded beside the batch. b-ON and C0 pay it alike, so it biases no A/B; T62 changes the τ inputs relative to the code that trained P\* (`9c033489`), a small input shift every fork shares (review F-RV-12). The pin file records the T62 verdict known at the first GO (`PENDING` until the battery reads) | `--t62` on the first launch |

The read records the stage rules read are the saved JSON of each read: `reads/<arm>_look<k>.json` =
`python -m main.x35.readout decide` (A1, A2, toy A1 / A2), `decide-d1` (D1) or `python -m main.x35.x47 decide`
(X47), naming its arm and look (a record for another arm or look is a refusal).

---

## 1. The arms, in GPU order

One arm × seed holds the GPU at a time. The order is AM-7's (§0), the kit's `BATCH1` = `batch.STAGES`, pinned by
`batch_test.py`. Stages P1, P2 and T1 are unconditional; P3, P4, T1b and T2 launch only when pre-check 9 says ALLOWED.
Inside T1 and T1b, X47 sits beside its own toy C0 in counterbalanced pairs (ruling C2: s3601 C0 then X47, s3602 X47
then C0, s3603 C0 then X47; s3604 X47 then C0, s3605 C0 then X47). Toy A1 left the interleave: it is now
conditional (AM-3). Every production arm and toy A1 / A2 stops at a FIXED step, so their order buys information, not
balance (review F-RV-16); X47's primary is time-based, which is why its pairs stay interleaved.

**GPU-h basis (MEASURED from checkpoint mtimes, the SOP's meter):**
- **Production:** X26 250M → 280M took 22,501 s (1,333 steps/s); 270M → 280M took 7,365 s (1,358 steps/s). A 20M fork
  is therefore ≈ 4.1 h, plus ≈ 0.2 h for start and compile, ≈ **4.3 GPU-h**. A1 (+8 evaluated archive opponents) and
  A2 (+5 sentinels) pay ≈ 2 % more at equal steps, declared and not corrected (§5.6.1): ≈ 4.4. D1's per-step cost on
  forum teams is not measured: ≈ 4.3, **UNVERIFIED**.
- **Toy:** X34 150M → 200M took 10,595 s (4,719 steps/s). A 70M fork is ≈ 4.1 h + 0.2 ≈ **4.3 GPU-h**. X47's ρ (its
  wall time per step over C0's) is ESTIMATED at 1.3–1.4 from FLOPs (`design_bot_ideas_screen.md` F-BI-7): ≈ 5.6–6.0
  GPU-h per seed, measured on its first 1M.

| stage | # | arm × seed | in plain words | condition | GPU-h | matched control |
|---|---|---|---|---|---|---|
| **P1** | 1 | **A1 s3501** | the ARCHIVE POOL: half the self-play games go against 8 frozen X26 nodes spread over 30–260M | none | 4.4 | C0 s3501 |
| | 2 | C0 s3501 | production control: P\* continued with X26's pool as of today (14 snapshots, 164–268M), under T62 | none | 4.3 | — |
| | 3 | C0 s3502 | control | none | 4.3 | — |
| | 4 | A1 s3502 | archive pool | none | 4.4 | C0 s3502 |
| | 5 | A1 s3503 | archive pool | none | 4.4 | C0 s3503 |
| | 6 | C0 s3503 | control | none | 4.3 | — |
| **G1** | — | the S read of the six finals (a GPU window: 36k games, ≈ 5 min at 118 games/s, **UNVERIFIED** for the end state; or ≈ 9 h of CPU) | AM-2's variance gate: `kit gate` | — | ≈ 0.1 | — |
| **P2** | 7–8 | C0 s3504, s3505 | the registered look-2 controls, moved forward: every arm's look 2 and the 6-df σ̂ | G1 read (either verdict) | 2 × 4.3 | — |
| **P3** | 9–10 | A1 s3504, s3505 | A1's look 2: the top-ranked arm decided first | A1 look 1 = CONTINUE | 2 × 4.4 | C0 s3504–s3505 |
| **P4** | 11–13 | **D1 s3501–s3503** (X38) | WIDEN THE TEAM POOL: half of each side's teams from 2,424 Smogon forum teams; eval stays on the pool | G1 PASS | 3 × 4.3 (**UNVERIFIED**) | C0 s3501–s3503 (`main.x35.d1`) |
| | 14–16 | **A2 s3501–s3503** | PFSP: recent selves the trainee loses to are drawn more often (+5 sentinels feed it win rates) | G1 PASS | 3 × 4.4 | C0 s3501–s3503 |
| **T1** | 17 | toy C0 s3601 | toy control | none | 4.3 | — |
| | 18 | **X47 s3601** | DEPTH GROWTH: the toy's 2 attention rounds become 4, the 2 new rounds exact identities, so the network starts as T\*'s function bit for bit | none | 5.6–6.0 | toy C0 s3601 (PRIMARY at equal effective learn time, node s_eq, `main.x35.x47`) |
| | 19 | X47 s3602 | depth growth | none | 5.6–6.0 | toy C0 s3602 |
| | 20 | toy C0 s3602 | toy control | none | 4.3 | — |
| | 21 | toy C0 s3603 | toy control | none | 4.3 | — |
| | 22 | X47 s3603 | depth growth | none | 5.6–6.0 | toy C0 s3603 |
| **T1b** | 23–26 | X47 s3604, toy C0 s3604, toy C0 s3605, X47 s3605 | X47's look 2 (AM-4) | X47 look 1 = CONTINUE | 2 × 5.6–6.0 + 2 × 4.3 | toy C0 s3604–s3605 |
| **T2** | 27–29 | toy A1 s3601–s3603 | the archive pool on the toy (8 nodes of T\*'s own lineage) | G1 PASS and production A1 FINAL GAIN / NULL / LOSS (AM-3) | 3 × 4.4 | toy C0 s3601–s3603 |
| | 30–32 | toy A2 s3601–s3603 | PFSP on the toy | G1 PASS and production A2 FINAL GAIN / NULL / LOSS (AM-3) | 3 × 4.4 | toy C0 s3601–s3603 |

**The GPU-h, by condition** (the basis below; the review's §3 has the same stages):

| what runs | GPU-h |
|---|---|
| unconditional: P1 + G1 + P2 + T1 | 26.1 + 0.1 + 8.6 + 29.7–30.9 ≈ **65** |
| + P4 (G1 PASSes; the expected path) | + 26.1 ≈ **91** |
| + P3 (A1 look 1 CONTINUE) | + 8.8 |
| + T1b (X47 look 1 CONTINUE) | + 19.8–20.6 |
| + T2 (AM-3: production A1 and A2 both FINAL) | + 0–26.4 |
| every conditional stage | ≈ **146–148** |
| not in BATCH1, at X35_PIN_1 when they happen | D1 / A2 look 2 (2 × 4.3 / 2 × 4.4, on their own look-1 CONTINUE; the kit allows them, AM-7 does not schedule them), toy A1 / A2 look 2, the G-E readers (≈ 2.2 per GAIN arm, §0), R1 reads (≈ 0.3 GPU-h each in a window) |

The old queue's "≈ 108 GPU-h ≈ 4.5 days" was look 1 only (review F-RV-13); a fully decided batch was ≈ 190.

**The stop rule, every arm.** Each arm runs to its fixed `--steps` (P\* + 20M, T\* + 70M) and ends with the launcher's
`Training complete`. No arm stops on a score: the length is what makes it comparable with C0 (§5.6.1; X47's PRIMARY
reads the node s_eq inside its run, its SECONDARY the final, so X47 also runs to +70M). An arm ends early only on:
- **K9(b)** (the behaviour-policy check): EVERY arm, production and toy, runs `warn` with the ceilings read by hand via
  `main.ops.tb_read` at every restart and daily: STOP on any update with judged max ≥ 1e-3, p99 ≥ 1e-4, or a non-finite
  |Δ| (rule 8 at 1e-9 relative). **AM-6:** at EACH arm's end the Training session runs
  `python -m main.x35.kit k9b --run-dir models/<run> --stage $S` (the fork's own rows only); a VOID arm enters no
  read and is re-run at the same pin and seed after the cause is fixed, and pre-check 11 refuses the next GO until
  every finished arm reads PASS. Production's are X26's Amendment 3. The toy's are X34's (ruling **C3**; its
  `STAGED_LAUNCH.md` §7, `c9516bac` / `96c663d4`), and `main.x35.kit describe` carries them for the toy arms. **The
  mechanism is the flag plus a human read: no code enforces a ceiling** (F-LS-9).
- a `FATAL_*` exit (not restarted), or a 4th crash (the launcher restarts 3).
- the post-spawn pool-seed check of `launch_x35.sh` (T54), which SIGTERMs a fork whose seeded pool leaks post-P\*
  snapshots.

**Not on the GPU list.** R0 (the best responder of P\*, ≈ 1.1 GPU-h) has no registered argv (F27); ruling **C4**: batch 1
launches WITHOUT it (A1 / C0 / D1 / A2 do not read it; stage its argv before the A3 / A6 batch). The R1
reads (≈ 0.3 GPU-h each in a lease window, or ≈ 12 h of CPU) and D1's Kakuna panels (CPU) go in the gaps between
arms, as §5.6.4 plans. Look 2 (seeds 3504–3505 / 3604–3605) is a later batch at the SAME pin.

---

## 2. The pre-launch checks (in this order; any failure is a HOLD)

`launch_x35.sh` runs checks 4–9 itself for every kit arm (exit 7 = HOLD, reason in `~/.claude/jobs/x35/status.txt`),
X47 (`toy_x47`) included (F-LS-1 is closed), and since the amendment (§0) four more: **2b** the ONE pin (AM-1), **9**
the stage is ALLOWED (AM-7 / AM-3 / AM-4), **10** the seeded pool set equals the side's record (POOL), **11** every
finished batch-1 arm PASSed K9(b) at its end (AM-6). The first GO also needs `--t62 <verdict>` (it writes
`x35_pin_batch1.txt`).

1. **X34 has STOPPED** (it plateaued: "Words used here"), its launcher PID is gone (`ps -p`; `python -m main.x35.kit live`
   exits 0), and T\* is the 290M node: path, step and sha256 are in the toy R file (`reference_population_toy.json`,
   `pstar`) and re-hashed by `refpop verify` at every launch. **DONE 2026-10-19** (the checks above).
2. **The GPU is clear**: the battery and the X14 A/B have finished, `nvidia-smi --query-compute-apps=pid --format=csv`
   lists nothing, and `scripts/ops/gpu_lease.sh status` reads FREE or the Training session's lease.
   `python -m main.x35.kit live` exits 0 (no training run live).
3. **The battery's T62 verdict is read** (`read_battery.py`): ADOPT, FP32_ALT, INCONCLUSIVE or INCOMPLETE. It picks the
   pin variant (§3).
4. **The pin** carries every commit in [`x35_required_commits.txt`](x35_required_commits.txt): T54 `cdab9cfe`, B-D2
   `c1b0c1f0`, the forum-corpus lift `f9360870`, F-BI-2 `4df10e65`, B-GROW `24616919`, B-GROW × F-BI-2 `aa5feaf5`, T62
   `bd3126e0`, and the canary YARDSTICK fix `03b983be` (its code is `ee49f1f9`, `gen3_r1_yardstick_margin_v1`; `03b983be`
   is the ship tip; all added 2026-10-19 except the first two; `launch_x35.sh` pre-check 2 enforces the file).
5. **P\*'s sha256** is `814823c5…91ed`, and `python -m main.x35.refpop verify reference_population_production.json`
   prints OK (it re-hashes P\*, the 8 archive nodes and R's 6 members, all in X26's run dir: never thin those 14 files,
   F-R3).
6. **The toy side's R is frozen** before the first toy arm: **DONE 2026-10-19** on the real T\*
   (`reference_population_toy.json`, §4.2; `python -m main.x35.refpop verify` it prints OK). Not needed for the
   production block.
7. **No `models/rb_x35*` directory exists** for the arm (a restart RESUMES through the launcher, never this sheet).
8. **`data/` is FROZEN** from the first arm's launch until the last X35 arm ends: every X35 arm is a pinned run, and a
   pin isolates code, not data. The forum corpus D1 reads lives in `src/utils/team_corpora/` (pin-safe), not `data/`.
9. **The disk.** Today 442.9 GiB are free (`kit disk`, 2026-10-19 23:10; it was 289 GiB at 18:10). The dry-run asks
   10.97 GiB per production fork and 13.05 GiB per toy fork (each with a 25 % margin and a 4 GiB reserve, so the sum
   over-counts). The kit's pre-check 8 counts every `BATCH1` fork not yet created (**32** since the amendment, X47's 5
   included; a conditional fork its records REFUSED is not counted) at the REQUIRED GiB of the launch in hand: at the
   FIRST production launch 32 × 10.97 = 351.0 GiB against 442.9 free, a PASS with ≈ 92 GiB to spare (F-LS-5); the same
   count at the toy figure would be 32 × 13.05 = 417.6 GiB, also a PASS today. The count falls as run directories
   appear, but the free space falls with it: thin finished arms to their 10M nodes + final as they end (F22), and
   re-read `df` before each toy launch. `--thinning` is the orchestrator's word, never the script's default.

---

## 3. The pin: two variants

The battery's T62 gate ([`../gpu_window_battery_2026-10-19/README.md`](../gpu_window_battery_2026-10-19/README.md)):
adopt the float64 τ solve iff it costs ≤ 3 % of a training cycle (`cost_hi ≤ 3 %`); **FP32_ALT** iff `cost_lo > 3 %`;
INCONCLUSIVE in between (the orchestrator decides).

- **Variant T62 (ADOPT): `<PIN>` = main HEAD at launch**, a descendant of `03b983be` (the commit every line here was
  last validated at, §4.3; it carries T62 `bd3126e0` and the canary yardstick fix). Main HEAD carries T62 by default.
- **Variant FP32_ALT: NO SUCH COMMIT EXISTS (F-LS-2).** FP32_ALT is not a flag and not a revert. The battery README
  defines it as "the fp32 complementary-constraint alternative" from the T62 decision row in
  [`design_x5_belief_tokens.md`](../../../endstate/design_x5_belief_tokens.md): solve the COMPLEMENTARY constraint
  when the mass saturates, "a second code path and a second case split". That is a build: the code, tests that fail
  on revert, and re-recording the learner golden (`learner_golden.json`) and the h2h reveal file's `OFF_BITS` that
  `bd3126e0` / `803a13e3` re-recorded under fp64. A plain revert of `bd3126e0` is NOT FP32_ALT: it brings back the
  K9(b) τ-class divergence (X26's A3 / A6 stops) that T62 fixed, and it would also need `803a13e3` undone.
  - When such a commit lands on main, it is a descendant of `bd3126e0`, so `x35_required_commits.txt` holds in both
    variants. The argvs are unchanged: only `--pin-commit <FP32_ALT_PIN>` differs, and each line must be dry-run again
    at that pin (`validate_batch1.sh --pin <FP32_ALT_PIN>`).
  - **The toy is NOT affected by T62.** The toy has no belief tokens (`move_belief_mode off`, `opp_belief_cls_k 0`,
    no species prior). MEASURED here: the toy network's forward on its 64 recorded rows (`toy_network_identity_test`)
    calls `hypothesis_set.fixed_size_tau` **0 times**, against 20 calls in the positive control (the
    `hypothesis_set_test` presence cases), with a counting patch. Only the CPU forward was checked; the learner path
    has no belief loss on the toy (its belief coefficients are 0), which was not executed here.

**C1 RULED (orchestrator, 2026-10-19): (a).** Production launches at main HEAD with T62 whatever the battery reads
(FP32_ALT does not exist, and T62's cost is paid by b-ON and C0 alike); the measured cost is recorded beside the
batch, and `x35_pin_batch1.txt` records the T62 verdict known at the first GO (`PENDING` until the battery reads).

---

## 4. The launch lines

**Launcher flags on every arm:** `--restart-interval-hours 24 --pin-commit <PIN>` (24 h as X26 / X34). Every line is
run from the main checkout (`/home/goodlad/dev/gen3ai`) with `export PYTHONPATH=$PWD/src`, by the Training session that
holds the GPU lease. `launch_x35.sh` without `GO` is a dry run of every pre-check; `GO` launches and needs the
orchestrator's go.

### 4.1 Production (validated at `03b983be` on the REAL P\*)

For each row of §1 (arms `a1`, `c0`, `d1`, `a2`, seeds 3501–3503), in order:

```bash
# NEEDS GO. <PIN> = variant T62: main HEAD (≥ 03b983be); variant FP32_ALT: the FP32_ALT commit (does not exist yet, §3).
S=designs/research_state/measurements/x35_breakers_stage_2026-10-17
$S/launch_x35.sh a1 3501 --pin <PIN> --lease-owner <owner>          # dry run: every pre-check, prints the line
$S/launch_x35.sh a1 3501 --pin <PIN> --lease-owner <owner> GO       # only on the orchestrator's GO
```

The argv the kit builds is, flag for flag, the validated file `batch1_validation_03b983be/argv_<arm>_s<seed>.txt`.
The launcher line it runs is:

```bash
python -m main.launcher --restart-interval-hours 24 --pin-commit <PIN> \
  $(cat $S/batch1_validation_03b983be/argv_<arm>_s<seed>.txt)
```

| arm | the argv after `--model P* --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda` | run name |
|---|---|---|
| C0 | (nothing) | `rb_x35_c0_s<seed>` |
| A1 | `--stable-opponents <N_8 … N_1 as zip:label, 260M … 30M> --stable-opponent-selfplay-share 0.5` | `rb_x35_a1arch_s<seed>` |
| D1 | `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both` | `rb_x35_d1wide_s<seed>` |
| A2 | `--pfsp-scale 2.5 --n-sentinels 10` | `rb_x35_a2pfsp_s<seed>` |

Each is followed by `--seed <seed> --run-name <run name>`.

### 4.2 Toy (validated at `03b983be` on the REAL T\*, 290,000,087)

**The toy R is FROZEN (2026-10-19, CPU, rule 5b, no GPU lease):** [`reference_population_toy.json`](reference_population_toy.json),
content sha256 `856c04cb87e9415eda9289845d66a770582c30e773bb7c23df7fdb6221a55a4d`, `refpop verify` OK. The full write-up
(nodes, the band cell, cost, findings) is
[`../x35_toy_refpop_freeze_2026-10-19/README.md`](../x35_toy_refpop_freeze_2026-10-19/README.md). In short:
1. `refpop plan --side toy` on T\* = `models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip` (sha256
   `cfa7f7be…a22965e7`) with the archive dirs `models/rb_toy_ent_s4002/checkpoints` and
   `models/rb_toy_ent_s4002_deep/checkpoints`: the archive N_1…N_8 = **30, 70, 100, 130, 170, 200, 240, 270M** (hand-computed
   first: 67.5 → 70, 135 a tie → 130, 202.5 → 200) and the A-block **290, 280, 80, 150, 220M** (mid(N_2, N_3) = 85 a tie → 80).
   `refpop_test.py` pins both lists.
2. The band cell T\* vs P\* (500 mirrored pairs, greedy vs greedy, reveal off, CPU eager, a cross-architecture cell on
   one engine, the production band cell's regime id `adfdbee824c9eb0a`): **0.4160, 95 % [0.3863, 0.4457]**, W/L/D
   413 / 581 / 6, 14.1 min on one lane. P\* is **KEPT** in the toy's X-block (0.216 inside the lower edge, far from rule
   8's 1e-9). R has **6 members: A-block 290 / 280 / 80 / 150 / 220M + X `pstar_x26`**.
3. **The toy pool, PRE-READ (read-only, no record written).** `python -m main.x35.kit pool-check --arm toy_c0 --call
   reference_population_toy.json --stage $S` (no `--write`) reads X34's `snapshots/` at the fork step 290,000,087: **20
   snapshots (92M … 270M; all at or below the fork step, so T54's filter drops none), digest `2e86c4f9d7a1…`** (full
   digest `2e86c4f9d7a10da2ef5e0516516fc31c219dc966842ccf74567bafdc0c41bbd5`; every name and sha256 in
   [`toy_pool_preread.json`](../x35_toy_refpop_freeze_2026-10-19/toy_pool_preread.json)). That file is NOT the batch record:
   the first toy GO writes `x35_pool_batch1_toy.json` (and `x35_pin_batch1.txt` is written by the FIRST GO of the whole
   batch, production's, with `--t62`). X34 is stopped, so the first toy GO must recompute this same digest; a different
   one is a HOLD worth reading (something moved in X34's `snapshots/`). Neither file was written here.
4. Every toy argv was built from that file and validated at the pin, in the kit's order (§4.3): 16 toy lines (toy C0 ×5,
   X47 ×5, toy A1 ×3, toy A2 ×3), every one `verdict=PASS`, none printing `STANDIN`.

The command for a re-validation at another pin (X47 is the kit arm `toy_x47`; `x47` is accepted as its alias) is the
whole of `BATCH1` in one call, production and toy together:
`$S/validate_batch1.sh --pin <PIN> --toy-call $S/reference_population_toy.json --out /tmp/x35_val_<PIN8> $(python -c 'from main.x35.kit import BATCH1; print(" ".join(f"{a}:{s}" for a, s in BATCH1))')`.
The script gives each line a VERDICT from `main.x35.kit validate-run` (a deterministic check, not a read of a grep):
both tools exit 0, "would launch", `role=[FORK of …rb_toy_ent_s4002_deep]`, `delta=[+70,000,000 steps]`, `k9b=[warn]`,
and `grow=[DECLARED trunk_layers 2 -> 4]` on the X47 lines and `grow=[none]` on the rest. Every line must end
`verdict=PASS`, and the last line must read `SUMMARY 32 PASS, 0 FAIL`.

**All four toy arms are kit arms** (`toy_a1`, `toy_c0`, `toy_a2`, `toy_x47`) and launch exactly as §4.1:
`$S/launch_x35.sh toy_x47 3601 --pin <PIN> --lease-owner <owner>` (dry run), then the same with `GO` on the
orchestrator's go. `launch_x35.sh` runs the §2 pre-checks, spawns the launcher, runs T54's post-spawn pool-seed check
and starts the watcher for X47 exactly as for the others; the SOP's remaining layers (the Monitor, the 55-min cron)
are the Training session's, as for every kit arm.

**X47** is the toy C0 argv plus ONE lever, `--grow trunk_layers=4`, and its own run name `rb_x35t_x47d4_s<seed>`
(seeds 3601–3605 since AM-4: look 2 = 3604–3605). The kit builds it from the toy R file, so it cannot drift from the
call. On the real T\* it reads (`argv_toy_x47_s3601.txt` of the validation dir):

```bash
# NEEDS GO. T* = 290,000,087, --steps = T*.step + 70,000,000 = 360,000,087, both from the toy R file.
python -m main.launcher --restart-interval-hours 24 --pin-commit <PIN> \
  --model /home/goodlad/dev/gen3ai/models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip \
  --behaviour-check warn --steps 360000087 --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 \
  --supply-starve-cycles self_play_pool=0 --device cuda --grow trunk_layers=4 --seed 3601 --run-name rb_x35t_x47d4_s3601
```

Its hand reads before GO are the dry-run's `grow        : DECLARED trunk_layers 2 -> 4` line (the validation's verdict
checks it) and §2's checks. In the first minutes of the child log read: the fork-time growth gate (the grown network's
logits and values bit-identical to T\*'s on real rows; a mismatch is `FATAL_CONFIG`) and the seeded pool GROWN the same
way. T54's pool-seed check runs inside `launch_x35.sh`; by hand it is
`python -m main.x35.kit pool-seed-check --arm toy_x47 --run-dir models/rb_x35t_x47d4_s<seed> --call $S/reference_population_toy.json --log models/rb_x35t_x47d4_s<seed>/launcher_child.full.log`
(F-LS-9: the growth adds a `grown` block to `pool_seed.json`; the check reads it as before, which `kit_test.py` pins
on a constructed record, and has not run on a real grown child).

### 4.3 What the dry-runs showed (32 of 32 PASS at `03b983be`, the toy on the REAL T\*)

**Validated 2026-10-19 23:22–23:27 PDT on the real P\* and the real T\*: `SUMMARY 32 PASS, 0 FAIL (pin 03b983be)`**,
[`batch1_validation_03b983be/`](batch1_validation_03b983be/) ([`summary.txt`](batch1_validation_03b983be/summary.txt), all 32
argvs, the `checkargs` / `--dry-run` transcripts of seed 1 of each of the 8 arms and of the look-2 lines C0 / A1 s3504,
X47 / toy C0 s3604). `03b983be` is main HEAD (`03b983be6`, the canary yardstick fix's ship tip) with this unit's
docs / frozen-record changes in the working tree; none of them touches a trainer, launcher or `checkargs` code, and the
kit's argv builder is unchanged. Built from the kit's own `BATCH1` (the launch order) with
`--toy-call reference_population_toy.json`: no line prints `STANDIN`.

| arms | `checkargs` | `--dry-run` |
|---|---|---|
| A1, C0, D1, A2 × s3501–s3503 and C0 / A1 × s3504–s3505 (16) | exit 0 | "would launch", FORK of `rb_x26_s2601`, 280,000,170 → 300,000,170 (+20,000,000); K9(b) `warn`; ✓ fork arch; 10.97 GiB REQUIRED; `grow=[none]`. All 16 argvs are **byte-identical** to the `d825c6a1` validation's (`cmp`) |
| toy C0 ×5, A1 ×3, A2 ×3 (11), s3601–s3605 | exit 0 | "would launch", FORK of `rb_toy_ent_s4002_deep`, **290,000,087 → 360,000,087 (+70,000,000)**; K9(b) `warn`; 13.05 GiB REQUIRED; `grow=[none]`. Toy A1's archive is the 8 nodes of §4.2 (`toy_270m` … `toy_030m`) |
| X47 ×5, s3601–s3605 | exit 0, `declared growth (--grow) : trunk_layers 2 -> 4` | as the toy rows, plus **`grow=[DECLARED trunk_layers 2 -> 4]`** (the dry-run's full line: "parent from its model_config.json — the parent loads strictly, identity-init parameters are appended, its AdamW state carries, new parameters start fresh", the seeded pool grown the same way) |

CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`, one job at a time, no GPU lease), with
`GEN3AI_MODELS_DIR` on a scratch dir that stayed empty; nothing was written under `models/` (X34's and X26's run dirs
were only READ). The shared compile cache's line (`cap 20.00 GiB - 18.15 GiB in use` against a 1.85 GiB growth) is the
same in every validation since `d825c6a1`.

**The earlier validations, on the 200M STAND-IN for T\* (history; the production lines are unchanged).** At
`d825c6a1`, the amended order: `SUMMARY 32 PASS, 0 FAIL`
([`batch1_validation_d825c6a1/`](batch1_validation_d825c6a1/)), the toy lines on `reference_population_toy_STANDIN_200m.json`
(printed as `call=…STANDIN…`: never a launch value). The 24 argvs of the pre-amendment order were byte-identical to
`9c3bf3c4`'s; the 8 look-2 lines read as their arms do.

**The first validation, at `9c3bf3c4` (24 forks, the toy on the stand-in):**

[`batch1_validation_9c3bf3c4/`](batch1_validation_9c3bf3c4/) holds all 24 argvs, the verdict line of each
([`summary.txt`](batch1_validation_9c3bf3c4/summary.txt), ending `SUMMARY 24 PASS, 0 FAIL`) and the full `checkargs` and
`--dry-run` transcripts for seed 1 of each of the 8 arms. `9c3bf3c4` is main HEAD (`9c3bf3c44`) with this unit's kit
change in the working tree, so the parser `checkargs --pin` reads and the fork-arch check are the checked tree's own
(F-BI-2); the kit change touches only `src/main/x35/`. All ran on CPU (`CUDA_VISIBLE_DEVICES=""`,
`nice -n 19 taskset -c 5-7,13-15`), one job at a time, with `GEN3AI_MODELS_DIR` on a scratch dir that stayed empty;
nothing was written under `models/` (X34's run dir was only READ). The list was built from the kit's own `BATCH1`, so
the validation order is the launch order.

| arms | `checkargs` | `--dry-run` |
|---|---|---|
| A1, C0, D1, A2 × s3501–s3503 (12) | exit 0 | "would launch", FORK of `rb_x26_s2601`, 280,000,170 → 300,000,170 (+20,000,000); K9(b) `warn`; ✓ fork arch; 10.97 GiB REQUIRED; `grow=[none]`. The 12 argvs are byte-identical to the `803a13e3` validation's |
| toy A1, C0, A2 × s3601–s3603 (9) | exit 0 | "would launch", FORK of `rb_toy_ent_s4002_deep`, 200,000,244 → 270,000,244 (+70,000,000, STAND-IN); K9(b) **`warn`** (ruling C3); 13.05 GiB REQUIRED; `grow=[none]`. Each differs from the `803a13e3` argv in ONE token, `fatal` → `warn` |
| X47 × s3601–s3603 (3) | exit 0, `declared growth (--grow) : trunk_layers 2 -> 4` | as the toy rows, plus **`grow=[DECLARED trunk_layers 2 -> 4]`** (the dry-run's full line: "parent from its model_config.json — the parent loads strictly, identity-init parameters are appended, its AdamW state carries, new parameters start fresh", the seeded pool grown the same way). Built by `build_argv("toy_x47", …)`: the `803a13e3` hand-patched line with `fatal` → `warn`, token for token (`kit_test.py` pins it) |

"24 of 24" counts the three X47 lines inside the 24 batch-1 forks (12 production + 9 kit toy + 3 X47). The first
validation (`batch1_validation_803a13e3/`, superseded for the toy lines; keep the directory, `kit_test.py` reads its
argv files) showed a `grow=[…]` column that for every non-X47 arm printed the dry-run's always-present
"(child-only: a declared growth's …)" note; the verdict now checks the DECLARED line itself, in the dry-run AND in
`checkargs`, and demands its absence on every other arm.

The toy stand-in R the three EARLIER validations built their toy argvs from (history; replaced by `reference_population_toy.json`) is
[`batch1_validation_803a13e3/reference_population_toy_STANDIN_200m.json`](batch1_validation_803a13e3/reference_population_toy_STANDIN_200m.json),
made by `refpop.freeze` with no X candidate into a scratch path. Its file name keeps `launch_x35.sh` from ever reading it,
and `validate_batch1.sh` prints `call=…STANDIN…` on every toy line it builds from it.

---

## 5. The choices (the orchestrator's): C1, C2, C3 and C4 RULED; C5 and C6 still open

- **C1. The battery does not read ADOPT. RULED (orchestrator, 2026-10-19): (a)**, launch production at main HEAD with
  T62 pending the battery read (§0, §3). The registered rule had said FP32_ALT, which does not exist (§3).
  - (a) Launch production at main HEAD anyway (T62), recording the measured cost. T62 is a correctness fix (one τ for the
    compiled rollout and the eager learner), and its cost is paid by b-ON and C0 alike, so it biases no A/B; it only
    raises the GPU-h.
  - (b) HOLD production until an FP32_ALT build lands and is validated (days), and use the GPU meanwhile for the toy
    block (τ-free, §3).
  - (c) INCONCLUSIVE only: a collect-phase A/B first (the README's suggestion).
  - **Recommendation: (a) for FP32_ALT and INCONCLUSIVE alike.** It changes the registered rule, so it needs the
    orchestrator's amendment, recorded before the first launch.
- **C2. Where X47 sits in the toy block. RULED (orchestrator, 2026-10-19): (b), interleaved.** X47 runs beside its own
  toy C0, toy A1 on the other side, toy A2 LAST: toy A1 s3601, toy C0 s3601, **X47 s3601, X47 s3602**, toy C0 s3602,
  toy A1 s3602, toy A1 s3603, toy C0 s3603, **X47 s3603**, toy A2 s3601–s3603. The alternative was (a) X47 after toy A2,
  un-interleaved with its C0. The reason: X47's read is against C0, so box drift should not land on one side, and A2
  (the calibration pair's plausible null, the cheapest to defer) still finishes at the same pin. BUILT: the kit's
  `BATCH1` is this order and `kit_test.py` pins it (the whole list, each X47 adjacent to its C0 with C0 first on s3601
  and s3603, A2 last). **Superseded in part by AM-7 (§0):** the X47 / toy C0 pairs keep C2's interleave (look 2 too),
  toy A1 moved to the conditional toy calibration, last.
- **C3. The toy arms' K9(b) mode. RULED (orchestrator, 2026-10-19): (b), `warn` with X34's ceilings.** The kit had typed
  `fatal` "as X34 does" (§5.6.1, written 2026-10-17), but X34 itself has run `warn` since its §7 amendment (2026-10-19;
  `STAGED_LAUNCH.md` §7, `c9516bac` / `96c663d4`). X34's mechanism is `--behaviour-check warn` on the argv plus ceilings
  read by hand (`main.ops.tb_read`, every restart and daily): STOP on judged max ≥ 1e-3, p99 ≥ 1e-4, or any non-finite
  value; within 1e-9 relative of a ceiling is recorded, no fire. BUILT: the four toy `Arm`s' `behaviour` is `warn`,
  `kit_test.py` pins it (the argv, `describe`'s `behaviour` and `k9b_ceilings`; production unchanged), and the old
  `fatal` stand-in `argv_toy_c0_standin.txt` is compared except for that one flag. **Evidence it is safe, READ here:**
  X34's own `behaviour/*` TensorBoard scalars (`models/rb_toy_ent_s4002_deep/tb`, read-only, 2026-10-19 ≈ 18:25) hold
  2,160 updates up to 212.4M with judged max 7.7e-5, p99 max 1.5e-5, no non-finite value, 0 updates at the 1e-4 bar
  (which is also the `fatal` bar): no ceiling has fired, and `fatal` would not have stopped it either. T\* is
  X34's own network and the toy has no belief tokens, so the production τ class cannot occur.
- **C4. R0** (the best responder of P\*, queue #0 of STAGED_PLAN §4) has no argv (F27). **RULED (orchestrator,
  2026-10-19): launch batch 1 without it**: A1 / C0 / D1 / A2 do not read it. It is A6's trigger and the R2 baseline, so
  stage its argv before the A3 / A6 batch.
- **C5. The production R's X-block** (refpop README F-R1) **STILL OPEN, its facts updated 2026-10-19**: it has one
  other-lineage member (`production`), because T\* did not exist when it was frozen. **T\* exists now** (290M, sha256
  `cfa7f7be…a22965e7`). (a) Keep R as frozen. (b) Re-freeze with T\* as a second X member BEFORE the first production arm:
  the old file moves aside (R is write-once; `launch_x35.sh` reads the canonical name) and a P\*-vs-T\* band cell is
  played on CPU. **The toy freeze's cell cannot be reused as it stands:** it has T\* as the PLAYER, and `refpop freeze
  --band-ledger` matches `player == the freeze's P*` (checked: `_band_from_ledger` returns no cell for the production
  orientation), so (b) needs either a second cell with P\* as the player (≈ 14 min of one CPU lane; the toy cell took
  14.1 min) or a `--band-json` cell that DECLARES score = 1 − 0.4160 = 0.5840 (an assumption of exact symmetry, not a
  measurement). Either way T\* would be inside the band (the toy cell reads 0.416). **Recommendation: (b) if a P\*-player
  cell can be played before the launch window; otherwise (a)**, because R cannot change once an arm exists (pre-check 7),
  and F-R2's own-lineage lean (5 of 6 members are X26's own nodes) is the read's main weakness. The pin / pool / toy
  record do not depend on this choice.
- **C6. X44 between the blocks.** The bot screen ranks X44 (≈ 10.4 GPU-h) above the toy calibration and X47. It is its
  own unit. Running it between the production and toy blocks delays the toy block, but changes no batch-1 pin (the toy
  block has its own C0). Not decided here.

---

## 6. Failure handling

- **A `launch_x35.sh` HOLD (exit 7).** Read `~/.claude/jobs/x35/status.txt` and the named transcript. Fix the cause and
  re-run the dry run. Never edit a pre-check to pass, and never launch by hand around a kit HOLD.
- **The launcher dies within 180 s** (a HOLD): read `~/.claude/jobs/x35/<run>.launcher.log`. A `FATAL_CONFIG` at start
  (for X47: the growth gate's bit-identity refusal) is NOT retried in another form: report it.
- **The post-spawn pool-seed check fails** (the script SIGTERMs the launcher by its PID and HOLDs): verify the PID is gone
  and the GPU is free (SOP §4.2), move the run dir aside as `models/<run>.POOLSEED_FAIL` (never delete it), bank it, and
  relaunch under the same name only after the cause is fixed.
- **A crash mid-run:** the launcher restarts it (≤ 3) from the latest checkpoint at the same pin. A 4th crash, or any
  `FATAL_*`, ends the arm: report to the orchestrator before anything else runs. A relaunch under the same name and pin
  is allowed for an operational fix; a scientific change is a new arm with a new registration (SOP §4.4).
- **A K9(b) ceiling crossed (production, warn):** stop the arm at its next safe point (SIGTERM by PID through the
  identity gate `kill_if.sh`), keep the GPU idle, and page the orchestrator. Under T62 the τ class should not recur;
  X26's Amendment 6 (the τ class STOP at 1e-2) is NOT carried into X35.
- **A `--sync-to-main` line, or a `Pinned to` sha that is not `<PIN>`:** the arm is VOID; stop it.
- **A HOLD between arms:** the GPU stays idle until the orchestrator decides (TRAINING_RUN_SOP §3, rule 3); the next arm
  never starts over a failed one.

---

## 7. FINDINGS (standing rule 7)

- **F-LS-1 (gap) CLOSED 2026-10-19: X47 is a kit arm.** `main.x35.kit` has `Arm("toy_x47", "toy", L_TOY, X47_SEEDS,
  "rb_x35t_x47d4_s{seed}", 0, "warn")` (seeds 3601–3603) whose lever is `--grow trunk_layers=4`, three places in
  `BATCH1`, and `kit_test.py` cases that fail on revert: so `launch_x35.sh` dry-runs and launches it, `kit disk` counts
  its 3 forks, and `validate_batch1.sh` gives it a verdict.
- **F-LS-2 (blocker for the FP32_ALT variant): no FP32_ALT commit exists** (§3). The registered T62 rule cannot be
  executed as written if the battery does not read ADOPT. Choice C1, RULED (a) on 2026-10-19.
- **F-LS-3 CLOSED 2026-10-19: the toy K9(b) mode** is `warn` (ruling C3, §5). The kit's old `fatal` for the toy was the
  stricter check and a stop would have left an arm short; X34's own scalars never reached its bar (C3).
- **F-LS-4 CLOSED 2026-10-19: the toy lines are on the REAL T\*.** X34 plateaued; T\* = the 290M node (sha256
  `cfa7f7be…a22965e7`), the toy R is frozen on it (§4.2) and all 16 toy lines of `batch1_validation_03b983be/` PASS with
  no `STANDIN` call. The stand-in R, its argvs and the three older validations are history (`kit_test.py` still reads
  `batch1_validation_803a13e3/` for the argv shapes it pins; never delete that directory).
- **F-LS-5: the disk margin at the FIRST launch is ≈ 92 GiB** (re-read 2026-10-19 23:10; was ≈ 26): §2 check 9, 32 forks ×
  10.97 = 351.0 GiB against 442.9 free. It falls with every run directory and every read output. A HOLD at pre-check 8
  is the F22 check working. Thin finished arms as they end; `--thinning` is the orchestrator's word.
- **F-LS-6: the dry-runs ran at `03b983be`, but the launch pin is main HEAD at launch time.** Any commit between them is
  unvalidated: `launch_x35.sh` re-runs `checkargs` and the dry-run at `<PIN>` before every arm, and X47's lines must be
  re-run with `validate_batch1.sh --pin <PIN>`.
- **F-LS-7: D1's GPU-h on forum teams is not measured**, and neither is X47's ρ (ESTIMATED 1.3–1.4). Both are read on the
  first 1M.
- **F-LS-8: "T62 does not touch the toy" was verified on the CPU eager forward only** (§3); the compiled and learner
  paths were not run.
- **F-LS-9: nothing ENFORCES the K9(b) ceilings.** `--behaviour-check warn` only LOGS a mismatch; the STOP at judged max
  ≥ 1e-3 / p99 ≥ 1e-4 / a non-finite value rests on the Training session's hand reads (every restart, daily; 24 h
  restart interval). A crossing between reads runs on until the next read. This is X26's and X34's regime and the
  orchestrator's ruling (C3); it is now also the toy block's. Separately, the post-spawn pool-seed check has not run
  against a real GROWN child (a constructed record only, §4.2).
- **F-LS-10: the validation is CPU and at ONE pin.** It proves each argv parses and "would launch" (and X47's growth is
  DECLARED); the growth's bit-identity gate, the grown pool and the pool-seed record are child-side and are first seen
  at the real launch (read the first minutes of the log, §4.2).
