# REVIEW I: an adversarial review of X60-T's draft PREREG (2026-11-06)

**Under review:**
- the PREREG draft at `agent-acda06c11bf64be1a` `2589a7fa7` (`PREREG.md`, `analysis.py`, `results/analysis.json`);
- the switch (main `fbd1afa38`; main has since moved to `636bdcc1e`, which landed the builder);
- the builder `src/main/human_data/hteams.py` (worktree `agent-a85f687a4c26954cc`, `eb7e67407` plus a working copy
  identical to main's `636bdcc1e`);
- the corpus `~/gen3ai_archive/human_teams/hr_spec_v1.r3/` (manifest, statistics and the work dir; no team text or
  account name is reproduced here);
- the read rule `src/main/reads/rules_x60.py` (`agent-ac44d9089f3848aca` `1eafff4c5`) and `read_spec_look1.DRAFT.json`.

**Method:** two lenses, as the owner asks for human data: data and selection bias, and experimental and causal
validity. I checked each claim against code or banked data. Two opus-high sub-reviewers took the read rule / statistics
and the training-side switch; I re-checked their load-bearing claims in the code. `forum216b` was never read.

## Verdict: READY-WITH-FIXES

The core design holds. The look-1 rule is exactly power.py's, σ̂ and the power numbers reproduce from the banked data,
and ON − C0 and ON − D1 each differ by one lever in code. The corpus is panel-clean, and the side pin's soak and
pack-identity checks were run. None of the fixes needs a redesign.

**Registration and the first ON launch are gated on B1, M1, M2 and M3.** B1 is a VOID rule nobody can evaluate. M1 is
a kit check that can pass only if human data enters git. M2 is a resume command that silently changes the seed. M3 is
a read that does not fix its seeds. M4 and M5 are text fixes that must also land before registration. The MINORs can
land with the registration.

## What holds (checked, not read)

- **Routing.** X59-S read Δ_labels OWN-LABELS-BETTER, and the closure is blocked by Δ_restriction. Design §6.1 row 3
  names OWN-LABELS-BETTER ⇒ X60-T.
- **The freeze hashes.** The `e1512c361` file is sha256 `ad34aee9…`, and its §6 slice is `f288b331…`, both as quoted.
  (§6 has since changed on main: see m1.)
- **σ̂ (subA).** From X61P look 2's cells (C0 607/625/598/610/583, N1-P 619/622/616/622/577 of 1,200), σ̂ is 0.209 pp
  and its one-sided 95 % UB 2.035 pp.
  - The UB is power.py's `sigma_upper95`.
  - REVIEW F's 8-seed rule (`e41beafe1`, 10-26) predates X61P look 2 (`62e46874a`, 10-30).
  - The decision holds without pooling N1-P (C0 alone gives 2.70) and on all 10 finals as one sample (1.81).
- **The rule and the power (subA).**
  - `rules_x60.rule_x60` against power.py's `rule_x60`: 0 mismatches in class or stop over 3,600 simulated cells.
  - An independent 20k re-run at σ 1.0 gives GAIN at +3 pp 0.888, false GAIN 0.020, a look-1 stop 0.508 at the null
    and 0.018 at +3.
  - `analysis.py` imports power.py by sha and does not re-implement it.
  - `rules_x60_test.py` passes, 45 tests.
- **The look-1 import (subA).** The spec sha `1266daad…`, the three C0 sha256s, E3 and regime `2118812c27885de3` all
  match. `imports.check` returns 180 rows (607 / 625 / 598 of 1,200), and `check-spec --pin-pending` exits 0.
- **One lever: ON vs D1 (subB, re-checked).**
  - `--human-data teams:` builds the same `TeamMix` → `Gen3Teambuilder._install_mix` path that `--team-mix` does, on
    both sides, with the same per-episode, per-side Bernoulli draw (`teambuilder.py:298`).
  - `440b61c32` (D1's pin) is `f4b25238` plus `src/utils/team_corpora/` only (the throw-only AM-D1-FIX label guard).
- **Off and the null dose (subB, code only).** With no mix installed, no RNG is consumed (`teambuilder.py:191`). The
  §14.2–3 played-bits checks at the pin are still required.
- **Resume streams (subB).** `segment_seed(seed, num_timesteps)` exists at `f4b25238`, so the corpus, env and team
  streams key on seed plus step: a resume does not replay step 0's draws.
  - The self-play pool is re-seeded only on a fork.
  - A pin change on a restart is FATAL_CONFIG.
  - The read side (`void_pin_guard`, `main.dose`) assumes no uninterrupted run (subA).
- **The stamp and clause hash.** The `[HUMAN-DATA]` line is emitted on every launch, resumes included
  (`matchup_setup.py:250`). The hash for `teams:hr_spec_v1.r3@0.5` is **`544c9bc00ea95030`** (subB), which fills that
  TBD.
- **Smogon-only priors (subB).** No prior, usage or set-inference consumer reads team corpora. C-GAME's fill is the
  log's reveals plus Smogon draws (`x50_r1/run.py`: "no Smogon prior" exclusion).
- **No egress so far.** Git holds only the stub (`human_corpora.json`: shas and counts). `hteams._guard_root` refuses a
  root inside a checkout. The PREREG, `analysis.py` and `analysis.json` carry no team id, account or log path.
- **Panel contamination.**
  - §3.1 runs on revealed facts: 0 drops.
  - The exact-set guard (≥ 4) dropped 0, and the post-fill ≥ 3 net (m4) dropped 92: forum72 9, H 4 (+1 retired
    heldout72), home72 57, away 15.
  - The side pin's own registry already holds forum72, H and p72, so §15's contamination VOID is not vacuous there.
  - `f4b25238` carries the pinned data root.
- **The 10-26 parity exposure is not asymmetric** (F-M3 fix 2). By my count, 30.9 % of corpus teams touch a 10-26
  parity class. That is below the pool's own: `data/teams/others` 39.2 %, `sample` 33.3 %, forum72 62.5 %, H 27.8 %.
  ON meets no more mis-simulated mechanics at E1 than C0 does.
- **The pack identity at the pin.** The 58 happiness-0 Frustration teams and 7 engine-unplayable teams were dropped
  BEFORE finalize. So the 2,295-team stub IS the at-pin set, and `human.load` refuses anything else.

## BLOCKER

### B1. The realised-share VOID rule (§15) and smoke check (§14.1) cannot be evaluated

- **Evidence.**
  - `Gen3Teambuilder._draw_team` (`src/utils/teambuilder.py:298-301`) draws a corpus team with `self._rng.random() <
    self.mix_prob`, sets `_last_pool_idx = None` and counts nothing.
  - `grep` over `src/` finds no corpus-draw counter, TensorBoard tag or sidecar field.
  - The trainee-side per-team win-rate counts merge corpus draws with sample-team bias draws, and the opponent side is
    not tracked at all. D1 never measured its share either.
  - So the rule "realised human share outside ±10 % of 0.5 ⇒ VOID" has no input, and the smoke cannot show "the
    realised human share within ±10 %".
- **Fix (either).**
  - (a) Before the side pin is cut, count corpus draws per side in the team stager and record them in each
    checkpoint's sidecar and in TensorBoard (counting consumes no RNG; off-means-off covers it).
  - (b) Rewrite both checks to what is checkable: the `[HUMAN-DATA]` stamp's share 0.5 and sides `both` in every
    launch log, plus a smoke-level count from a unit test that drives `_draw_team` 10⁵ times at the pin.
  - (b) costs no pin change. Over ~10⁶ draws a run, a Bernoulli(0.5) cannot miss ±10 %, so the rule's only job is to
    catch a mis-wired share, and the stamp does that.

## MAJOR

### M1. Data egress: `kit corpus-check` can pass only if the human corpus is committed into the pin

- **Evidence.**
  - `main.x35.kit.corpus_check` (`src/main/x35/kit.py:727-770`) runs `git show <pin>:src/utils/team_corpora/<corpus>/
    manifest.json` and `…/soak_selfcheck.json`.
  - `launch_x35.sh` runs `corpus-check` (§14).
  - PREREG §3 / X8 say the side pin carries "the corpus files".
  - The switch, by contrast, reads the corpus from `~/gen3ai_archive/human_teams/` (`utils/paths.py:404`), and git
    holds only the stub.
  - As written, the launch HOLDs, or someone makes it pass by committing the corpus directory into a pin commit, which
    can be pushed.
- **Fix.**
  - Give `corpus_check` a human-corpus branch: check the pin's `human_corpora.json` stub and run `python -m
    utils.team_corpora.human verify hr_spec_v1.r3` with the pin's code.
  - Reword §3 / X8 to "the committed stub; the corpus never enters git".
  - Add a test that `corpus_check` never reads a human corpus path from git.

### M2. §13's resume command silently changes the seed when run through the side pin's launcher

- **Evidence (subB).**
  - At `f4b25238` there is no `launch_flags.py`. Restoring `--seed`, `--behaviour-check` and
    `--checkpoint-every-steps` from the run's record (the 10-24 fix, `recipe_surface.materialize_recorded_recipe`,
    `launcher/run.py:362`) happens only in main's launcher.
  - §13's command types none of the three. Through the pin's own launcher, a resume would run seed 42 (sibling seeds
    share streams) and the default `--behaviour-check`.
  - §13's checks 1–6 test the dose, the pin and the stamp, but not the seed. So a mis-seeded resume passes §13 and
    enters the read.
  - Rule 8 (deterministic checks) does not cover this; it is a silent change of input.
- **Fix.**
  - Type `--seed <s> --behaviour-check warn --checkpoint-every-steps 1000000` in §13's command.
  - Add check 7: the resumed launch's resolved seed equals the run's recorded seed (`checkargs` or `--dry-run` output,
    kept in the record).
  - Keep `--run-name` out (with it the launch becomes a fork, `checkpoint.resolve_launch_run_dir`), and say why.

### M3. The read code does not fix which seeds it reads

- **Evidence (subA; re-read at `rules_x60.py:323-331`, 352-360).**
  - `check` requires only 3 or 8 distinct-seed labels per side, and a `look1_seeds` list of any 3 seeds both sides
    hold.
  - The contrast need only be a 5-subset of ON's finals, and D1's seeds are unchecked.
  - On the test file's look-2 fixture, `look1_seeds = [3506, 3507, 3508]` or `[3501, 3503, 3508]` are accepted.
  - The look-2 spec is written after look 1 is read, so its author can re-pick the look-1 triplet (to manufacture a
    futility stop) or the contrast's 5 ON finals (after seeing per-final F).
  - The PREREG text fixes the seeds; the code does not.
- **Fix.**
  - Hard-code `LOOK1 = (3501, 3502, 3503)`, `LOOK2 = range(3501, 3509)` for ON and C0, and `CONTRAST = range(3501,
    3506)` for ON and D1.
  - Tie each label's seed to its run (`_s<seed>` in the ref or path, and the run's recorded seed).
  - Add a refusal test for each.
  - This lands in the play-pin commit, before look 1 is read.

### M4. Selection bias in the corpus: the six-seen rule keeps losing teams and long games, and the PREREG never says so

- **Evidence.**
  - Re-reading every ≥ 1500, unflagged seat of r2's kept battles (9,193 seats; script
    `scratchpad/me/winshare.py`, logs through `snapshots.read_verified`, `|win|` against the seat's player):

    | seats | n | won | median turns |
    |---|---:|---:|---:|
    | all ≥ 1500 candidates | 9,193 | **54.3 %** | — |
    | six seen (kept) | 4,925 | **41.1 %** | 31 |
    | not six seen (dropped, 46 %) | 4,268 | 69.6 % | 18 |
    | **final corpus** | 2,295 | **39.0 %** | — |

  - "Six mons seen" mostly selects the side whose mons were all revealed, which in practice is the loser, and the
    longer games.
  - So H-TEAMS is a 61 %-loser, long-game slice of ≥ 1500 ladder teams, not "human-built ladder teams" at large.
  - This does not bias Δ_F as an A/B of THIS corpus. It does bias what a result can be attributed to:
    - §1's "is the gain the LADDER part";
    - §10's LADDER-BETTER / FORUM-BETTER, where D1's forum teams are curated (quality-selected), while H-TEAMS is
      anti-selected on outcome;
    - a NO-USEFUL-GAIN read as "ladder teams do not help".
- **Fix (text, no rebuild).**
  - State the win share and length skew in §1 and §3.
  - Add to §10's m3 attribution: "LADDER-BETTER / FORUM-BETTER also confound source with outcome selection (39 %
    winners vs curated forum teams)".
  - Add to §11's NO-TRANSFER / NO-USEFUL-GAIN license: "for this six-seen, loser-enriched slice".
  - Report the corpus's winner share in §12.
  - A win-balanced rebuild is a separate registration, not required here.

### M5. R72's team ids are reversible battle pointers, and the PREREG plans to commit them

- **Evidence.**
  - `hteams.team_id` is `"hr3_" + sha256(f"{rel}|{seat}")[:16]`, unsalted, where `rel` is
    `<date>/battle-gen3ou-<number>.log`.
  - Battle numbers are a public, roughly sequential counter, so for a known date range they can be enumerated: about
    10⁷ candidates × 2 seats is seconds of hashing.
  - PREREG §9 and the TBD table plan to put "R72 seed, digest, **team ids**" into the PREREG, which is pushed to
    GitHub. That would publish which spectated battles, and therefore which accounts, our human data came from.
  - The r72 manifest's `proposed_registration` itself carries no ids (good).
- **Fix.**
  - Commit R72's `content_digest` (`gen3_team_set_digest_v1`, irreversible), seed and rule only. Keep the ids local.
  - Or re-key the ids with a local secret salt, as `rowset.key64` does for accounts.
  - Add a push-guard pattern for `hr3_[0-9a-f]{16}`.

## MINOR

- **m1. The freeze list is incomplete.** §2 says §6 is byte-identical at `0b9e6f354`. It is, but main's §6 now
  differs: `fbd1afa38` records that the switch keeps provenance in `metadata.json` only (not `model_config.json`) and
  has **no consumer snapshot check**, both departures from §6.9 item 1.
  - Also (subB), `human_data_setup.decide:92` refuses only an UNTYPED resume. A resume typing `off` or another clause
    passes with a warning, and a test asserts that. So §13's "the switch's refusal holds a resume that drops or
    changes it" overstates it; §13.4's stamp check is the real guard.
  - Fix: list these as X14, and reword §13.
- **m2. The look-1 stop's other outcomes (subA).** A stop can end NOT-RESOLVED (0.065 of nulls at σ 1, 0.21 at 2.04)
  or LOSS (0.019 / 0.040). The overall false-LOSS rate is 0.019–0.041, above the nominal 0.025, and LOSS licenses
  "drop H-TEAMS". §16's split assumes a stop means NO-USEFUL-GAIN. Fix: add both rates to §5.2.
- **m3. The Welch guard has no tie band (rule 8; subA).** `guard_lb > SIGMA_F**2` (`rules_x60.py:132`) picks the
  interval. Fix: a registered direction within 1e-12 (for example, the wider interval).
- **m4. `guards` and `void` are optional at look 2 (subA).** Without them a GAIN can only reach INCONCLUSIVE, and the
  VOID gate can be omitted. Also, one `pin` covers every run, so C0 s3506–s3508's pin and K9(b) are not checked by the
  read. Fix: add both to `LOOK2_REQUIRED`, with a pin per arm.
- **m5. PREREG text vs code (subA).**
  - The code gives NO-USEFUL-GAIN for UB in [−1e-9, 0); §5 says "0 ≤ UB".
  - §15's "any short cell" contradicts the floor's 20-game tolerance.
  - INCONCLUSIVE (a missing meter, a guard not read, a failed determinism guard) is absent from §11's table.
  - Fix: align the text with the code.
- **m6. FINDING 3 and §4 / §17 are stale (subA).** A second group importing C0 s3504–s3505 from X61P look 2's state
  (`f28654cc…`) returns 120 rows (610 / 583 of 1,200, E3, same regime), and `pool_anchor` pools groups. So F needs no
  two-state import and no 3,240-game re-play. Fix: correct §4, FINDING 3 and T1's cost, and check S group `p` the same
  way.
- **m7. The play-pin commit needs more than §4 lists (subA).** `de15dfc7b`'s `rules._void_check` accepts only
  `{k9b_dir, runs}`, so the draft's `void.pin` / `void.dose` would refuse there (this fails closed). Its
  `imports.py` lacks the engine, sides and peer-pack checks, which apply only if assemble runs from main. Fix: list
  both in §4, and state where assemble runs.
- **m8. Effective N and the dedup rule.**
  - The "duplicate" rule (same six completed sets) dropped 0, because C-GAME's fill is seeded per (battle, seat,
    attempt), so a repeated team never matches.
  - 791 of 2,295 teams repeat their own account's species-six, leaving about 1,504 distinct (account, species-six)
    teams.
  - So §3's dose per human team is about 190 draws a side, not 124. Fix: report both counts in §3.
- **m9. Provenance nits.**
  - The manifest's `commit` (`0b9e6f354`) does not contain `hteams.py`; only `source_sha256` (`e60b9268…` =
    `eb7e67407`'s file) ties it. Main's landed builder (`636bdcc1e`, sha `bd527453…`) differs from it by the
    `human_data_ok` arguments alone.
  - The stub's `"snapshot": null` drops the `hr_spec_v1 → r3` chain §3 cites.
  - Fix: record both in §3.
- **m10. "Built once, as `n1_hrep_v1` too" (§3) is not true.** No `n1_hrep_v1` exists, and X53-N1's doc still names
  `--team-mix n1_hrep_v1`, which the F-M7 guard now refuses. F-M7 fix 2 (reconcile X61 §10 with §6.4) is not disposed.
  Fix: reword §3, and add F-M7 fix 2 to §18 as open, bearing on N1H only.
- **m11. Backport hazards (subB).**
  - `utils.paths.gen3ai_archive_dir` is absent at `f4b25238` and must be backported.
  - `f4b25238`'s `_install_mix` has no run-time `check_pack_identity`. It is safe today only because the offline
    pin-check already dropped the 58.
  - The backport must not bring main's `TeamLoader` (its label guard would refuse a level-≠100 pool team that C0
    trained on).
  - Fix: list all three in `kit pin-check`'s expected-path list.
- **m12. The stamp's `rows_sha` field name.** §13.4 / §15 match on "rows sha", but the corpus is a teams directory
  (data / tier / manifest shas). Name exactly which sha the registered stamp carries.

## FINDINGS (standing rule 7)

1. **Not verified:**
   - played-bits equality of off and `@0` at the side pin (the pin does not exist);
   - that an ON checkpoint loads at the play pin;
   - that registering R72 leaves forum72 / home72's ids unchanged;
   - S group `p`'s import from look 2's state;
   - `main.dose` on a run that was actually resumed;
   - fm5 / fm6 / fm8 and §7.1's goal-(a) and Δ_P legs (only §5's power was re-run);
   - whether H (`hl_05_26`) carries the same six-seen skew as the corpus;
   - that the F-M7 tests fail on revert;
   - that no training-time writer puts corpus team text under `models/`.
2. **Hazard during this review.** SubB's pytest run rewrote main's tracked `designs/ops/slow_tier_status.json` (one
   slow-test row) and restored it with `git checkout --` on that file. Main is clean now (`git status` empty). The
   slow-tier recorder writes into whichever checkout runs it, so a read-only reviewer must pass `-m "not slow"`, or
   the recorder should skip writes outside a worktree. The builder worktree also carries that file modified.
3. **`<<TBD>>` filling rules.** Each is a fact-to-be-built with a mechanical rule, except "the play pin's two-state
   import (built or not)", which m6 makes moot. None leaves a choice that could be made after seeing X60 data, once
   M3 fixes the seeds in code.
4. **Process.**
   - CPU only, as `GEN3AI_BLOCKING` at nice 0 on 5-7,13-15, `OMP_NUM_THREADS=1`, under `mem_cap.sh` (≈ 1 min of log
     reads).
   - Read-only on `models/` and the archive. No GPU, game or launch.
   - Team texts and account names were parsed in memory and never printed.
