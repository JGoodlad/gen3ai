# X65 ExIt-0: the adversarial review (2026-11-05)

The review the [PREREG](PREREG.md) requires before the pin amendment. It covers builds B1–B5 at `c9d86ef53`
(`e0dc4e466..c9d86ef53`), the PREREG and the
[design doc](../../../endstate/design_x65_exit0_search_distillation.md). The reviewer had no GPU, changed no code
and registered nothing. Every check below was run, not just read, unless it is marked **UNVERIFIED**.

**Verdict: TRAIN. No blocker.** Nothing found requires rebuilding the bank (built from `c9d86ef53`), changing the
recipe, or moving a bar. Three MAJORs and twelve MINORs need a disposition in the pin amendment, which the PREREG
already places before the first student trains:
- R-M1: test the delivery scorer before the delivery gate runs;
- R-M2: make ExIt-1's licence wait for the placebo's attribution;
- R-M3: add tests for the label rows' alignment guards and the bank's VOID test.

Severities:
- **BLOCKER**: fix before any student trains.
- **MAJOR**: dispose in the pin amendment; it could make a verdict wrong or unaudited.
- **MINOR**: a note, a test gap that cannot bias this read, or something the record should say.

## What was checked, and how

| question | how | result |
|---|---|---|
| (1a) is the `frame_sha256` refusal real? | mutation M1 (the guard turned off); two other mis-orders built by hand | real; its revert-failing test is the digest-corruption test (R-m1) |
| (1b) is π′ the live piKL target at λ 0.015? | read `rules.pikl_scores` / `search.py:178-192` / `search_side.py` `_search`; recomputed by hand on 6 random searched cells | exact (below) |
| (1c) does the legal-move index map to the action? | read `bank.game_rows`; the same 6 cells; the partial bank; mutations M4, M5 | aligned on real data; two guards untested (R-M3) |
| (2) can anything leak across folds? | team-file overlap over the three panels; the world prior's source; `student.split`; mutations M8, M8b, M9, M9b | no team, deal or state leaks; sets overlap (R-m8) |
| (3) is the control fair? | read the rule's VOID list and `anchors.results.unit_counts`; the regime id's inputs | same engine, bytes, greedy regime, peer and regime id per panel, all enforced; the dose control decides nothing (R-M2) |
| (4) does the rule implement the classes and bounds exactly? | 12 mutations of `rules_distill_play`; read against PREREG §1, §4, §6, §10 | all 12 KILLED; one false sentence in the PREREG (R-m3) |
| (5) are seeds 600–674 disjoint? | every `designs/**/*.json` spec scanned; the archive scanned for played units; `derive_seed_from_base` | disjoint (R-m11) |
| (6) does each guard fail on revert? | [`review_mutations.py`](review_mutations.py) → [`review_mutations.txt`](review_mutations.txt): 35 mutations | 25 KILLED, 10 SURVIVED (R-M1, R-M3, R-m2, R-m10) |

**Clean, with evidence.**

- **π′ is the live target.** The live pick is argmax of Q(a) + λ·log π(a) (`rules.pikl_scores`). That is λ times
  (log π + Q/λ), so the bank's `pi_prime` (`bank.py:177-186`), softmax(log π + Q̂/λ) over the legal actions, has the
  live pick as its mode.
  - Its `q` is the same uniform world mean that picked (`search.py:176-187`).
  - Its `log_pi` is the live forward's log-softmax over the legal actions only (`search_side.py`, `lsm`).
  - Recomputed independently on 6 random searched cells (126 label rows): every label row kept, mode = `played` on
    126 / 126, 0 drops, 0 `log_sha256` or `at` mismatches.
  - Rounding `q` to 6 decimal places moves the exponent by at most 3.3e-5 nats.
  - These 126 rows deviate from P\*'s greedy choice on 55 (44 %). So a(P\*) in the delivery gate is about 0.56.
- **The index mapping holds on real data.** The bank requires three things of a label row:
  - its legal set equals the keys of `q`, and also the rebuilt mask;
  - `played` comes from the rebuilt tokens and equals the recorded index;
  - P\*'s offline greedy action on the rebuilt row equals `pi_argmax`.
  The evidence:
  - the builder's 18-shard sample: 6,007 / 6,007 label rows;
  - the first 12 raw shards of the live build (forum72 300–311): 0 drops, 0 cross-check failures, 0 P\* mismatches,
    1 tie excluded.
- **The fold split.**
  - No row of the read panel is loaded. `split` refuses a read-panel row or deal; both mutations are KILLED.
  - Inner validation is keyed by `deal_key`, which both arms of a deal share.
  - The trainable set (242 of P\*'s 428 parameters) is the same on 8 different 64-row probes taken from the bank, so
    the probe does not decide which parameters train.
- **No leak through S′'s worlds.** The world sampler's prior is Smogon usage (`world_draws.Prior`), never a panel's
  teams.
- **The control.** The rule VOIDs on any of these:
  - a counted unit off E4, off its label's pinned sha256, or not greedy;
  - a panel whose counted units (both arms and G-BANK) span more than one regime id, peer code or policy mode;
  - a unit that `unit_counts` rejects (peer not clean, a non-battle ending, the forced-end guard).
  Every unit is a lone one-game unit, and a deal's battle seed and team draw are functions of the unit seed and the
  game alone (`--team-seed = --seed-base =` the unit seed). So the student and the banked raw game face the same
  opponent settings and the same dice.

## BLOCKER

None.

## MAJOR

**R-M1. The delivery scorer is executed by no test** (`src/main/x65/delivery.py:62-89`).
- `score()` computes a(student) and a(P\*) per label row. Nothing runs it: the delivery tests feed `bootstrap_lb` and
  `gate` with constructed arrays.
- Mutation M15 SURVIVED. It replaced a(P\*) = 1[`pi_argmax` = `played`] with the student's agreement with P\*.
- The NOT-DELIVERED class rides on this file. The rule (`read_delivery`) and `by_hand.py` both TRUST its `delivered`.
  They check only that it agrees with the file's own LB, so neither would catch a wrong Δagree.
- **Failure scenario.** A slip in the `keep` / `lab_k` re-indexing scores the student on the wrong rows, or on the
  pre-turn-10 rows. Δagree is then wrong in either direction:
  - a false NOT-DELIVERED cancels look 1;
  - a false DELIVERED spends ≈ 23 CPU-h on students that learned nothing.
- Read by hand at `c9d86ef53`: correct.
- **Disposition (before the delivery gate runs, not before training):**
  - a constructed-bank test of `score()` with `_forward` monkeypatched, asserting each row's `a_student`, `a_pstar`
    and `deal`;
  - `by_hand.py`, or a small sibling, recomputing Δagree_X and the LB from the three score files.

**R-M2. GO licenses ExIt-1 before anything separates search information from fine-tuning itself** (PREREG §1, line 36;
§7).
- The banked P\* control is matched on deals, engine, opponent and regime, but not on training dose. The student had
  ≈ 1M row-passes of fine-tuning and P\* had none.
- The dose-matched control is the PLACEBO: the same recipe, rows and seeds, with Q̂ permuted. It runs only after GO,
  and it is "reported, never decided".
- **Failure scenario.** Fine-tuning toward sharp π′ targets moves P\*'s greedy play in a way Kakuna happens to punish
  less. Nothing about S′'s choices causes this.
  - GO fires, and its wording ("a network distilled from S′'s choices beats P\*") stays literally true.
  - ExIt-1 (≈ 35 GPU-h) is registered on it.
  - The placebo then reads NOT-SEPARATED.
- **Disposition (an amendment to the licence; it moves no bar, fold, deal, z, β or class rule):**
  - on GO, ExIt-1 is funded only if the placebo attribution reads SEARCH-INFO;
  - GO + NOT-SEPARATED licenses only the descriptive claim.

**R-M3. The label rows' alignment guards and the bank's VOID test have no test that fails on revert.** Each was
mutated off and every bank test still passed:
- `bank.py:415`: the legal set must equal the keys of `q` and the rebuilt mask (M4).
- `bank.py:389`: the rebuilt `played` index must equal the recorded one (M5).
- `bank.py:604`: more than 0.5 % of a panel's label rows dropped VOIDs the bank (M25). This is PREREG §10's first
  VOID.

**Failure scenario.** A later edit to `bank.py`, for a re-build after an amendment or for ExIt-0H's honest labels,
breaks one of these lines. Then either a π′ is trained against shifted action indices, or a bank that dropped 2 % of
a panel ships as un-VOID. No gate notices.

The current code is correct, and on real data it drops nothing. So this bank is not affected.

**Disposition:** three constructed-record tests:
- a `q` whose keys differ from the mask;
- a recorded `played` that differs from the rebuilt one;
- a `summarize` over shards with 0.6 % of label rows lost.
They change no bank behaviour, so the running bank stays valid.

## MINOR

**R-m1. B1's PREREG-named test never reaches the identity guard** (`src/main/x65/bank_test.py:113`).
- The test builds the mis-ordered log from the frame probe's naive prefix. The row it should drop is dropped as
  `rebuild:BankError` by `rebuild_row`'s decision-index check (`bank.py:314`), not as `frame_sha256`.
- With the guard off (M1), that test still passes.
- The guard does fail on revert: the digest-corruption test (`bank_test.py:131`) catches M1.
- Two other mis-orders, built by hand on the fixture, gave the SAME frame, 15 / 15 kept:
  - the opponent's choice for the current update leaked into the log;
  - the filtered set left in arrival order.
- That is correct. The row is the state before both choices resolve, so the guard protects ROWS, not logs, and no
  search runs offline. The log is checked by `log_sha256`, which is reported and not dropped.
- **Disposition:**
  - the test asserts its drop reason;
  - the amendment says which check removes the probe's miss;
  - optionally, mutate a rebuilt row's bytes in a test, which reaches the guard through the real path.

**R-m2. Two trainer guards are untested** (`src/main/x65/student.py:92`, `:184`).
- M9 SURVIVED: inner-validation label rows were left in `train_label`. The pick would then be contaminated; the read
  panel stays clean.
- M26 SURVIVED: the β anchor term was dropped from the loss.
- **Disposition:** two assertions in `student_test.py`, cheap, before the first student trains:
  - `train_label` ∩ inner-validation deals = ∅;
  - L on a constructed batch equals the label KL plus β times the anchor KL.

**R-m3. PREREG §1 (line 43) says GO and NULL "cannot both hold". They can.**
- Both hold when 0 < LB_go and UB_null < 2 pp. That needs SE < 2 / (z_go + 1.96) pp:
  - at look 1, SE < 0.44 pp: a discordance d ≲ 0.05, so practically never;
  - at look 2, SE < 0.505 pp: d ≲ 0.18. Possible for a student held close to P\* by β = 1.
- The rule checks GO first. The test `test_go_is_checked_before_null_and_the_boundary_never_crosses` covers that
  order, so the class is deterministic and the rule needs no change.
- **Disposition:** the amendment corrects the sentence. The record says "NULL's test also held" whenever both hold;
  the rule's `tests` block already carries both.

**R-m4. The leak-exposure split is degenerate** (`bank.py:475`; PREREG §5, reported only).
- `residual` contains `counters:seen` in 10,956 of 10,956 searched records (the first 300 cells). So `leak_resid` is
  True on every row, and the "none" bucket is always empty.
- **Disposition:** the amendment names the residual counter that was meant, or the report notes the split is
  uninformative.

**R-m5. No per-pair START check** (`src/main/reads/rules_distill_play.py:420`).
- A student unit and its raw unit are paired by `(panel, seed, game)` alone.
- G-BANK compares the START of 135 banked deals rebuilt at the play pin, and a fresh arm is built by the same code.
  So a deal that moved would surface there.
- **Failure scenario:** a team directory re-listed at the new pin. G-BANK's mismatches send look 1 to the fresh
  control, and the pairing stays valid.
- **Disposition (optional, cheap):** compare every pair's START document and VOID a mismatch.

**R-m6. G-BANK proves reproduction only at the time it runs.**
- **UNVERIFIED:** the cause of Kakuna's measured 9 / 2,100 lone-replay divergence, and so whether it depends on
  machine load. The students play beside a live run at concurrency 2. This review did not check S′-LIVE's load.
- **Disposition:** queue the 135 G-BANK units interleaved through the student units, not all first.

**R-m7. The play pin.** The design doc's status records that the read cannot be planned at `0c76b521e`.
- **Disposition:** the amendment names the successor pin, and runs `export write` with `--pin-root` at THAT pin, so
  B3's bit-identity holds where the students actually play.

**R-m8. Sets overlap across panels; teams do not.**
- No pair of panels shares a team or a six-species core, and no team pair shares more than 2 exact sets.
- Exact sets (species + item + moves) are shared: H–F 51, H–p72 63, F–p72 44, out of 343 / 255 / 373 distinct sets.
- "Teams it never saw" holds. "Sets it never saw" does not.
- **Disposition:** GO's wording keeps "teams".

**R-m9. The drop budget.**
- The cap is 0.5 % of 900, so 4 drops a panel at look 1.
- S′-LIVE's two banked raw drops (OP-7) count against it. If both are on one panel, two student drops there VOID
  look 1.
- **Disposition:** a note in the amendment.

**R-m10. Four more bank guards are untested.** Each was mutated off and every bank test still passed:
- the turn match (`bank.py:385`; M21);
- the key and deal match (`:400`; M7);
- P\*'s tie exclusion (`:444`; M20);
- the P\*-mismatch drop (`:532`; M24).
All pass on real data. **Disposition:** test them alongside R-M3.

**R-m11. Seeds 600–674.**
- Every spec under `designs/` was scanned. Only X65's look-2 groups name these seeds on U, and no unit has been
  played at them on any U panel in `~/gen3ai_archive`.
- Battle seeds are a hash of the base (`derive_seed_from_base`), so they are disjoint from 300–344, 200–244 and < 120.
- FINDING X65-F3 has no guard. S′-HONEST is not registered yet, and its registration must avoid these seeds.
- §7's home72 floor (600–614) shares dice streams with look 2's 600–614 on other teams. That is harmless, because it
  is reported only.

**R-m12. Two hazards, for the record.**
- The bank build runs under `timeout 18000` (5 h). It is at 21 / 270 shards after 680 s, so the ETA is ≈ 2.5–3 h,
  and `build` is resumable.
- `bank_test.py::test_the_bank_builds_checks_and_never_opens_a_result_file` fails in a worktree without a cargo
  build: `env/Cargo.lock` is in E4's env stamp (129 files against 128). The mutation runs deselected it.

## Readings the builders took (design doc §7), accepted

- Inner validation keyed by `deal_key`: accepted.
- A fresh anchor permutation each label epoch: accepted.
- Every decision is an anchor row, one-legal rows and both arms' shared pre-turn-10 rows included: accepted. Two
  side effects: the one-legal rows (KL 0) dilute β's effective weight, and the early game weighs twice.
- π_P\* computed once from a frozen copy: accepted.
- The inner-validation L as float64 means: accepted.
- The gate's a(P\*) from the recorded `pi_argmax`: accepted, since `pstar_check` ties the offline forward to it on
  every label row apart from the counted ties.
