# REVIEW D: adversarial RE-REVIEW of the revised human-data drafts (431b1fdf, 2026-10-25)

**Under review (commit `431b1fdf`, "REVISED DRAFT, pending re-review"):**
- [`human_data_architecture.md`](../../../endstate/human_data_architecture.md), "the arch doc";
- [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md), "the plan";
- [`RESPONSE.md`](RESPONSE.md), the dispositions of REVIEWS [A](REVIEW_A_data.md), [B](REVIEW_B_experiments.md) and
  [C](REVIEW_C_reasoning.md);
- [`src/human_data_doc_gate_test.py`](../../../../src/human_data_doc_gate_test.py), "the gate";
- [`plan_power.py`](plan_power.py) and [`hrq.py`](hrq.py).

**Lens.** Two jobs:
1. Check every "FIXED" in RESPONSE against the docs and the code, and re-run the cheap checks.
2. Attack the new content: X59-S, X60 on F + P72, the first-reads order, and the new panel exclusion.

**Method.**
- CPU only, on rule 5b's cores at nice 19. The heavy step ran under `mem_cap.sh` (peak 3.4 GB).
- Read-only on the corpus, the stamped `hr_spec_v1` cache, the durable copy and the panels. The sealed `forum216b`
  was never read.
- Two scripts are committed beside this file. Their outputs are in `results/review_d_*`.
  - [`review_d_tamper.py`](review_d_tamper.py) runs the real `hrq.py extract` CLI on COPIES of 40 real logs, in a temp
    dir, against their real manifest rows.
  - [`review_d_checks.py`](review_d_checks.py) has two parts. `power` re-derives X60 and X59-S under what the plan
    registers. `data` measures what the ≥ 3-fit exclusion keeps, and whether a C-GAME-like fill re-creates panel
    cores.
- The docs under review were not edited.

**Since `431b1fdf`.** Commit `57fd657b` landed during this review. It registered FR-1 and FR-3, pinned `LargeRL` and
`LargeIL`, and made `hr_arch_2026-10-24`'s durable copy. That closes B-M5's deferred downloads and RESPONSE FINDING 1
at HEAD. This review judges `431b1fdf` and notes HEAD where it differs.

**Abbreviations.**
- P\* = X26's plateau. HOM = the human opponent model. H, F, P72 = `heldout72_hpfix`, `forum72`, the procedural panel.
- U = the unfamiliar panels against Kakuna. C-GAME = rung 1b's completion of a seat's hidden facts from the Smogon prior.
- φ = AM-5's overdispersion ratio (variance between seeds relative to binomial noise).
- pp = percentage points; lane-h = one CPU lane for an hour.

## Verdict: READY-WITH-FIXES

The revision is a real improvement, and most dispositions hold:
- The status box, the inventory and the rowset chain are in place.
- The doc gate works on the cases it claims.
- `plan_power.py` reproduces byte for byte.
- No flagged account name remains in any human-data file at HEAD.
- The new exclusion turns out better than feared: what it keeps is ENRICHED in pool-unfamiliar teams (D-M7).

It is not ready to build as a whole, though:
- **X60 is NOT-READY.** B-B2's fix holds only for a φ rule the plan does not register, and its seed rule fires even
  when seeds do not vary (D-B1).
- **X59-S is ready once D-M2..D-M5 are fixed.** They are text, plus one `plan_power` update and one more student.
- **FR-1 to FR-3 are ready** and already playing at HEAD.

**Cold read: 5.5 / 10 on page one** (C scored 4.5); 10 / 10 for the whole doc.

**"Fixed" that wasn't** (it was reworded or done only in part):
- **A-M5:** the extract HANGS on a tampered log instead of refusing (D-M1).
- **B-B2:** see D-B1.
- **B-M6:** the drawn-move "match" has no mechanism (D-M2), and the non-human meter is missing from X60 (D-M6).
- **B-m7:** four falsifiers are still point statements (D-m4).
- **C-F7:** page one still answers "is there evidence it helps?" wrongly (D-M8).
- **One decline is not justified:** A-M2's HOM re-fit. The un-refit HOM still routes H-INIT (D-M5).

## 1. Findings at a glance

| # | severity | finding | fix |
|---|---|---|---|
| D-B1 | **BLOCKER** (X60 only) | Under the registered upper-quantile φ, X60's NULL power is 0.16 (χ²) or 0.30 (t), not 0.58, and GAIN at +2.5 pp is 0.39 / 0.63, not 0.77. The seed rule fires 67 % of the time at a TRUE seed SD of 0. Its fallback path (+3 seeds a arm, or "NULL unreachable") has no cost line and no power | pick ONE φ rule, put it in `main.x35.d1`, re-power it (t needs ≈ 1,000 games a cell); set a seed bar that 10 finals can resolve; simulate and cost the 8-seed path |
| D-M1 | MAJOR | `hrq.py extract` HANGS forever on a changed log or a wrong header: `SnapshotError` is a `SystemExit` raised inside a `Pool` worker. Only the digest check (main process) refuses | `SnapshotError(Exception)`; a CLI test under `timeout` on a tampered copy |
| D-M2 | MAJOR | X59-S is not label-matched. S-P's P\* sample can be a drawn move, a drawn bench mon, or (on a hidden turn) a switch the log rules out; S-H's label never is. "Matched across arms" has no mechanism | sample S-P from P\* restricted to the log-consistent actions; give hidden turns the SAME set label in both arms; a VOID assertion |
| D-M3 | MAJOR | X59-S's interval covers game noise only. With one student per arm, a 2 pp training-seed SD makes P(HUMAN-LABELS-BETTER \| Δ = 0) 0.20, not 0.025. The 0.30 base has no source, and the rowset is ≈ 2M rows, not 6.9M | ≥ 2 seed pairs in the primary, with a t interval across pairs; a 200-game floor pilot; restate the volume |
| D-M4 | MAJOR | X59-S answers "human LABELS vs P\*'s labels on human states". The owner asked "human DATA vs self-play DATA", which is S-H − S-S, and S-S is optional, with no rule and no power | register S-S; make Δ_data = S-H − S-S the owner's Q-H9 read; keep Δ_S as the labels part of the split |
| D-M5 | MAJOR | The declined HOM re-fit still steers decisions. "H-INIT at no extra build" forks the existing HOM, whose rows kept 3-set cores of H / F / P72 (≥ 4-fit, H-only exclusion). Its descriptive Δ_U on F + P72 routes H-INIT | bar the existing HOM from every lever and every routing rule; H-INIT forks a `hom_<rowset>` refit on a §3 rowset |
| D-M6 | MAJOR | B-M6's non-human meter is not in X60. The U primary is against human-pretrained Kakuna only, and the in-house co-primary is home72 | an in-house U read (ON − OFF on F + P72, `main.h2h`) as a registered GAIN guard |
| D-M7 | MAJOR | What the ≥ 3-fit exclusion KEEPS is a different population: 66 % forfeits or timer losses (vs 48 % among the excluded), 19 % of games ≤ 10 turns (vs 1.7 %), and half the sides not fully seen. It does keep 42 % of the pool-unfamiliar sides, 3,417 of them | state it in §3.1; record the profile in each rowset manifest; register an information floor for H-TEAMS teams |
| D-M8 | MAJOR | Page one still answers cold-read Q10 ("is there evidence it helps?") wrongly. The box pins two positive-looking numbers and leaves out fact 4. Page one scores 5.5 / 10 | add a box line "evidence human labels help goal (b) : none either way"; say in words what the `open` line means |
| D-m1..m10 | MINOR | the gate's blind spots; a misleading `extraction_commit`; the durable copy's threat model; falsifiers; X59-S's volume and cost; the H-p exception; the audit's HOM bias; the contrasts' seed variance; wording | §3 |

**Checks that PASS** (what holds):
- **The fill.** Post-fill exact-set panel cores among the KEPT sides are 0.32 % (P72 0.02 %), and 0 % among the
  pool-unfamiliar ones. A-B1's exclusion on revealed facts plus the fill does NOT re-create panel cores (§2.7).
- **The doc gate.** 5 / 5 planted mismatches of the kinds it claims are caught (§4).
- **The power script.** `plan_power.py` reproduces `results/plan_power.json` exactly.
- **The durable copies.** `ARCHIVE.json` matches the registry, the files are read-only, and every member is
  verified (376,410).
- **The registry digest.** A changed digest is refused, exit 1.
- **Privacy.** 0 of the 77 mapped account names appear in the 54 human-data files at HEAD.

---

## 2. BLOCKER and MAJORs

### D-B1 (BLOCKER for X60) — B-B2 is reworded, not fixed

**Evidence.**
- **Two incompatible fixes.**
  - The plan's X60 row registers "AM-5's widened Newcombe (`main.x35.d1`, with an upper-quantile φ, REVIEW B m6)",
    and RESPONSE marks B-m6 FIXED.
  - But `plan_power.py`'s `x60_fixed` uses the POINT φ̂ (`sqrt(max(1, φ̂))`). So does `main.x35.d1` today
    (`d1.py:68-69`: "no t / F quantile is applied").
  - The quoted NULL 0.58 and GAIN 0.77 are for a classifier the plan does not register.
- **Under the registered rule** (`review_d_checks.py power`, 100k reps, the same two-look procedure, the sign rule
  and the heterogeneity guard):

| AM-5's multiplier | seed SD | NULL at Δ = 0 | GAIN at +2.5 pp | half-width at look 2 |
|---|---:|---:|---:|---:|
| point φ̂ (`plan_power`, the code today) | 0 | 0.58 | 0.77 | 1.74 pp |
| t at df (2, then 8) | 0 | **0.30** | **0.63** | 2.05 pp |
| χ² upper 95 % φ | 0 | **0.16** | **0.39** | 2.86 pp |
| point φ̂ | 1.5 pp | 0.24 | 0.56 | 2.45 pp |
| t at 1,000 games a cell | 0 | 0.59 | 0.84 | 1.63 pp |
| χ² at 1,260 games a cell | 0 | 0.48 | 0.72 | 2.02 pp |

- **The seed rule fires when seeds do not vary.** The rule: "if the topped-up φ̂'s one-sided upper bound implies a seed
  SD > 1.5 pp, add 3 seeds per arm, or declare NULL unreachable."
  - On C0 + D1's 10 finals at 630 games × 2 panels (df 8), it fires with **P = 0.67 at a TRUE seed SD of 0**, 0.81 at
    0.75 pp and 0.95 at 1.5 pp.
  - The median implied upper bound at zero seed variance is 1.83 pp.
  - Ten finals cannot resolve a 1.5 pp bar. (The formula reproduces REVIEW B's 2.7 pp from D1's φ̂ 0.77 at 140
    games.)
- **What follows.** The fallback is now the expected path:
  - +3 seeds per arm is ≈ +60 % of X60's ≈ 16 + 11 GPU-h per arm. It is not in §6.1.
  - The 8-seed classifier (look boundaries, sign rule) is not registered and not powered.
  - Or the plan declares "NULL unreachable" up front, which is B-B2's original BLOCKER.
- **The contrasts carry the same gap.** H-TEAMS − D1 and H-BC − P-BC quote a 1.74 pp half-width with no seed term.
  At a seed SD of 1.5 / 2.7 pp, their false-positive rate at Δ = 0 is 0.09 / 0.18, not 0.025.

**Fix.**
1. Register ONE AM-5 multiplier and implement it in `main.x35.d1` before X60 is registered. t at df is the milder
   choice.
2. Re-run `plan_power` with that rule, and size the games from it: t needs ≈ 1,000 games per (seed final, panel) for
   NULL 0.59 and GAIN 0.84. That is ≈ 1,000 games instead of 630 per cell, ≈ +75 % on the F + P72 top-ups.
3. Re-set the seed rule to a bar that 10 finals can resolve. Two options:
   - the point estimate of the seed SD > 1.5 pp; or
   - the one-sided 80 % bound > 2.5 pp.
   Simulate its firing rate at seed SD 0 and 1.5 pp, and register and cost the 8-seed classifier.
4. Widen the two contrasts by the same AM-5 multiplier.

**Scope.** X60 runs after X59-S, so this does not block the first reads or X59-S.

### D-M1 (MAJOR) — A-M5's "REFUSES on any mismatch" is a hang for the per-log checks

**Evidence** (`results/review_d_tamper.json`; 40 real `2026-07-01` logs copied with their real manifest rows; the real
CLI under `timeout 60`):

| case | exit | result |
|---|---|---|
| clean | 0 | 40 / 40 records |
| one byte changed, fresh cache | **none: HUNG** | 0 records, no message |
| the registry's manifest digest changed | 1 | `hr_t: manifest sha256 … != registry …` |
| spectator logs declared `archive` (wrong header) | **none: HUNG** | — |
| one byte changed AFTER a clean extract, cache reused | 0 | `2026-07-01 cached` (the cached day is not re-hashed) |

- **The cause.** `class SnapshotError(SystemExit)`.
  - `extract_day` runs in a `multiprocessing.Pool`, whose worker catches only `Exception`.
  - The `SystemExit` kills the worker, and the pool replaces it. The task's result never arrives, so
    `imap_unordered` waits forever.
- **No wrong number can reach a result.** `load_snapshots` refuses a cache whose record count is not the manifest's.
- **But the claim is false, and the failure is an unbounded wait.** The rows marked HUNG in the table never exit.
  - Root rule 2 forbids unbounded waits.
  - On the full corpus a hung job is a stuck background extraction that nobody is told about.
- **Why the tests pass.** `hrq_test.py` calls `verify_bytes` directly, never the CLI.

**Fix.**
1. Make `SnapshotError` an `Exception` subclass. The alternative is to return an error marker from `extract_day` and
   raise in the parent.
2. Add a CLI-level test: a tampered copy in `tmp_path`, under `timeout`, asserting a non-zero exit with the path in the
   message.
3. Document that a cached day is not re-verified, or re-hash it on reuse.

### D-M2 (MAJOR) — X59-S's "one difference" is not the only difference

**Evidence.**
- **The drawn-move signature.** The plan says "X59-S matches it across arms" (§3.5) and "X59-S's contrast survives (both
  students see identical completed states)" (§2.2).
  - Identical STATES do not match a LABEL artefact. S-H's label is never a drawn move or a drawn bench mon, because the
    chosen action is what revealed it.
  - S-P's label is a sample from P\*'s full masked policy, so it IS a drawn option whenever P\* puts mass there.
    (Our observation has no "revealed" bit, so P\* cannot avoid them.)
  - So S-H learns "an unused, Smogon-typical option is never chosen" and S-P does not. That is a systematic difference
    between the arms besides "human vs P\*".
  - FR-2 measures whether models learn this signature. It does not remove it from X59-S.
- **Hidden turns.**
  - On the ≈ 12 % of rows where the log proves a MOVE was chosen (`cant`, fainted before acting), S-H gets
    −log Σ_moves p, while S-P gets one P\* sample.
  - That sample is a switch whenever P\* would switch: a label the log refutes.
  - The plan declares the set-vs-point asymmetry, but not the refuted labels.

**Fix.**
1. Sample S-P's label from P\*'s policy RESTRICTED and renormalised to the actions the log leaves possible for the
   human: no drawn move, no drawn bench target, and only moves on a hidden turn.
2. On hidden turns, give BOTH arms the identical set label. It carries no teacher content, so the arms match exactly
   there.
3. Add a VOID: no S-P label is an action the log rules out.
4. Then §2.2's "the contrast survives" becomes true, and FR-2 leaves X59-S's critical path.

### D-M3 (MAJOR) — X59-S's power ignores training-seed variance; its base rate and volume have no source

**Evidence.**
- **One seed per arm.** "Δ_S's LB > 0" with a binomial interval tests these two networks, not the two label sources.
  Two from-scratch students differ by their init and batch-order seed even with identical labels.
- **What a seed SD does** (`review_d_checks.py power`, normal approximation; per-student SD of the pooled U win rate):

| student seed SD | P(HLB) at Δ = 0, one pair | the same, replicated | power at +4 pp, one pair | replicated |
|---:|---:|---:|---:|---:|
| 0 (the plan) | 0.025 | 0.001 | 0.86 | 0.73 |
| 1 pp | 0.09 | 0.008 | 0.77 | 0.59 |
| 2 pp | **0.20** | 0.04 | 0.67 | 0.45 |
| 3 pp | **0.28** | 0.08 | 0.62 | 0.39 |

  (HLB = HUMAN-LABELS-BETTER.)
- **The replication guard covers only part of the routing.** It protects a GPU build. But the first-pair class routes
  the plan's next CPU and agent work (UNCLEAR goes to H-TEAMS first), and OWN-LABELS-BETTER has no replication at all.
- **The base rate has no source.** "Students at ≈ 0.30 on U" is unsourced. Nothing in the card or the X50 docs measures
  a from-scratch student against Kakuna.
- **The volume is pre-exclusion.** The ≥ 1300 rowset holds 6.94M informative rows before §3.1 (`results.json`
  `labels_x50_train`). If 1300–1499 battles are excluded at the ≥ 1500 rate, that becomes ≈ 2.0M (ESTIMATE).
  - The cost line ("≈ 5–10 lane-h each at one epoch") is REVIEW B's 6.9M-row figure.
  - A one-epoch student on ≈ 2M rows is more likely to sit at the 0.15 floor.

**Fix.**
1. Run ≥ 2 seed pairs in the PRIMARY: Δ_S averaged over pairs, with a t interval across pairs or the pair spread
   added. Report the measured student seed SD.
2. Re-run `plan_power` with the seed term.
3. Add a floor pilot: 200 games per student against Kakuna before the 2,400-game read. If both students are under
   0.15, go straight to the in-house rule and save ≈ 14 lane-h.
4. Restate the rows and the cost after the exclusion.

### D-M4 (MAJOR) — X59-S does not answer the owner's question as asked

**Evidence.**
- **The owner's question.** It is "is human data better than self-play data". The arch doc §7.1 restates it as
  "on the same states, do human labels teach better than ours?", which holds STATES fixed by construction.
- **What Δ_S = S-H − S-P measures.** "Is a ≥ 1300 human a better teacher than P\* on human states?" That is the LABEL
  half.
- **What the owner asked.** Human states plus human labels against self-play states plus self-play labels at matched
  volume. That is S-H − S-S.
- **S-S is "optional"**, with no rule and no power. A HUMAN-LABELS-BETTER verdict could be reported as "human data is
  better" while the data question was never read.

**Fix.**
1. Register S-S. That is P\*'s self-play states with P\*'s sampled labels, the same row count, and the same matched
   recipe and seeds.
2. Make **Δ_data = S-H − S-S** the read that answers the owner.
3. Report the split:
   - Δ_S = S-H − S-P, the labels part;
   - S-P − S-S, the states part, which is the breadth mechanism of H-TEAMS and D1.
4. Add one student and 2,800 games (≈ +13 lane-h at the plan's rates).

### D-M5 (MAJOR) — the declined A-M2 re-fit still steers decisions, through H-INIT

**Evidence.**
- **What the decline rests on.** RESPONSE declines re-fitting the HOM on an account-disjoint split because
  "X59 no longer decides anything".
- **But the existing HOM still enters the routing:**
  - §4 rank 2: "H-INIT (a fork of the HOM itself, plain PPO) … at no extra build".
  - X59-S's OWN-LABELS-BETTER branch: "H-INIT remains only if the HOM's descriptive Δ_U ≥ 0".
  - H-KL's `kl:<model>`.
- **That HOM was fitted on rows the revised plan forbids.** The arch doc §1 says its exclusion was "heldout72 ≥ 4-fit".
  - So its training battles kept the 3-set cores of H, F and P72. That is 84 % of ≥ 1500 battles carrying a ≥ 3-fit
    neighbour (REVIEW A B2).
  - Its early stopping used a battle-id split (plan FINDING 4).
- **The consequences.**
  - Its Δ_U on F + P72 may read panel memory as skill, and that read routes a lever.
  - A fork of it into X60 would put the exact contamination §3.1 forbids into the F + P72 primary.

**Fix.**
1. Bar the existing HOM, by name, from every lever and every routing rule. It stays a measurement artefact.
2. Define H-INIT and H-KL on `hom_<rowset>`: a refit on a registered §3 rowset, with account-disjoint early stopping.
   Cost it: H-INIT is not "no extra build".
3. Either re-read the descriptive Δ_U on that model, or label it contaminated and let it route nothing.

### D-M6 (MAJOR) — B-M6's non-human meter was adopted in X59-S only, where it rarely decides

**Evidence.**
- **What B-M6 fix 3 asked.** "Add one non-human-trained meter to every human-lever read": in-house head-to-head on U,
  and/or Foul Play. The Foul Play half was declined.
- **Where the accepted half went.**
  - In X59-S the head-to-head decides only under the floor fallback.
  - X60's U primary is against Kakuna alone. Kakuna is human-pretrained, so it is not a neutral meter (arch doc §2).
  - X60's in-house co-primary P is home72, not U.
- **Why it matters.** An X60 GAIN on F + P72 cannot be told apart from "learned to beat a policy pretrained on the same
  human population". That is the exact confound B-M6 named.

**Fix.** Register an in-house U read in X60:
- ON − OFF on F + P72 against the frozen reference population or P\*, through `main.h2h`;
- CPU only, thousands of games, half-width ≈ 1.5 pp.
GAIN then also needs that read's point estimate > 0. A GAIN that only Kakuna sees is reported as "Kakuna-specific".

### D-M7 (MAJOR) — the ≥ 3-fit exclusion keeps the unfamiliar region, but in a different population

**The question.** Does what the exclusion leaves still cover the unfamiliar region, which was the point of goal (b)?
**Yes, and better than proportionally.** But the kept battles are a different population.

**Evidence** (`review_d_checks.py data`, `hr_spec_v1` TRAIN dates, rated battles with a ≥ 1500 side, 56,073 battles):
- The rule over four panels (H, `heldout72`, F, P72; `forum216b` not read) excludes **70.70 %**. That reproduces A's
  70.7 %; `heldout72` adds nothing.

| | KEPT (16,429 battles) | EXCLUDED (39,644) |
|---|---:|---:|
| forfeit or timer loss | **66.3 %** | 47.6 % |
| games of ≤ 10 turns | **19.5 %** | 1.7 % |
| mean turns | 27.4 | 30.9 |
| mons seen per side | 4.82 | 5.68 |
| full-six sides | **50.1 %** | 76.7 % |
| moves seen per seen mon | 1.89 | 1.91 |
| ≥ 1500 full-six non-bot sides | 10,853 | 39,974 |
| … of which pool-UNFAMILIAR (≤ 2 set fits to any pool team) | **31.5 %** (3,417) | 11.6 % (4,649) |

**What it means.**
- **Coverage survives.**
  - 42.4 % of all pool-unfamiliar ≥ 1500 full-six sides are kept, against 29 % of battles overall.
  - Teams far from every panel are also far from the pool, so the exclusion removes the mainstream.
  - Goal (b)'s training material is not destroyed: 3,417 sides over four months, before a per-account cap.
- **But the rule selects on how much a game revealed.** It counts only mons that revealed a move, so a short or
  abandoned game cannot reach 3 fits.
  - The kept rows are dominated by forfeits and short games, and half their sides were never fully seen.
  - So H-TEAMS teams would often carry whole drawn mons.
- **Who pays.**
  - X59-S's contrast is unharmed: both arms see the same states.
  - Its LEVELS, H-TEAMS's corpus and any rowset-level claim are about a forfeit-heavy slice, not "the ≥ 1500
    ladder".
- **What §3.1 states.** It gives only the share removed.

**Fix.**
1. Add the kept / excluded profile to §3.1 and to every rowset manifest.
2. For H-TEAMS, register an information floor: for example, ≥ 5 mons seen or ≥ 10 turns. Report its cost in
   pool-unfamiliar sides.
3. Read X59-S's absolute levels as this slice's.

### D-M8 (MAJOR) — the cold read: page one still misleads on the action-relevant question

The full table is §5. C-F7 is fixed in part: page one moved from 4.5 to 5.5 / 10. Q10 ("is there evidence human data
will help goal (b)?") still scores 0, for three reasons:
- The box pins `human sides in the panels' region : 15.9 %` and `HOM - P* NLL … : -0.503 nats`. A page-one reader
  takes them as "coverage, and learnable, so it helps".
- The doc's own conclusion (fact 4: no evidence either way; the proxy is uninformative) is on page two.
- The `open` line ("Q-H10 first read; the trace twin; X59-S") is jargon a cold reader cannot decode.

**Fix.**
1. Add one box line: `evidence human labels help goal (b) : none either way (the familiarity proxy is uninformative)`.
2. Say in words what `open` means: "does a human-only model beat ours; does a model learn the fill; human vs
   self-play data".
3. Add one line per switch value: `bc` = a loss term, `teams` = the pool, `kl` = a pull toward a model. These cost
   three lines and lift page one to ≈ 8 / 10.

---

## 3. MINORs

- **D-m1. The gate's blind spots** (in-memory mutations of the real doc, §4). Seven of twelve plausible edits pass:
  - the box lines `human decisions … : none / none / none` and `replay-derived teams in eval : yes`;
  - a stale duplicate box line placed BEFORE the real one (`status_box` keeps the last);
  - Q-H7's 0.503 [0.467, 0.542], fact 3's 15.9 % and Q-H6's ≈ 71 %;
  - the plan's 70.7 %.

  **Fix:**
  - refuse duplicate box keys;
  - pin the qualitative lines where a source exists (`heldout_panels.json` for the replay-derived panel,
    `anchors.json` for the human-trained models);
  - replace each number repeated outside the box with a pointer to it, or pin it.
- **D-m2. `extraction_commit` names code that did not extract.** The stamp says `742decb2`, but that commit's
  `hrq.py` hashes `233bbce3…`. The stamp's `hrq_sha256`, `bf9ba7c9…`, is the file as committed at `431b1fdf`.
  The content hash pins the code; the commit field misleads. **Fix:** refuse to stamp from a dirty `hrq.py`, or
  record a dirty flag.
- **D-m3. The durable copy's threat model.**
  - `extract` reads only `HRQ_CORPUS`, so nothing reads `logs.tar.zst` back. A restore is manual.
  - The copy sits on the same NVMe partition as the original (`/dev/nvme0n1p6`). It protects against a checkout
    deletion (the incident class), not a disk loss.
  - The account SALT exists once. Losing it breaks every future account-disjoint split against `hr_spec_v1`.

  **Fix:** add `extract --from-durable`, state the threat model, and keep a second copy of the salt wherever the owner
  chooses.
- **D-m4. B-m7 is fixed in part.** These falsifiers are still point statements: A-2 (FR-2's "gap > 0.05 nats"),
  A-4 ("> 20 pp across terciles"), A-5 (the smoke) and A-6 (a sign flip). FR-2 also lacks rule 8's boundary
  exclusion. **Fix:** an interval and a boundary rule for each.
- **D-m5. X59-S's volume and cost.** See D-M3: ≈ 2.0M rows after the exclusion (ESTIMATE), not 6.9M.
- **D-m6. The H-p exception is ambiguous.** "ON's rows INCLUDE the ≥ 3-fit battles near H-p" does not say what happens
  to a battle that also fits an F or P72 team. If such battles are included, the F + P72 primary is contaminated.
  **Fix:** include a battle only if every one of its ≥ 3-fit matches is to an H-p team.
- **D-m7. The Kakuna-family audit's score is biased toward missing a fleet.** The score is LL_Kakuna − LL_HOM.
  - The HOM trained on these very accounts' TRAIN rows, and ≈ 17 % of its gain is account memory.
  - So high-volume accounts get a higher LL_HOM and are LESS likely to be flagged, and a fleet is high-volume.

  **Fix:** cross-fit (score each account under a HOM that never trained on it), or use P\* as the reference.
- **D-m8. The contrasts' seed term.** See D-B1, point 4: at a 1.5 pp seed SD the registered half-width gives a 9 %
  false positive rate.
- **D-m9. RESPONSE wording.**
  - "630, more than the ≥ 420 asked": it is the same 1,260 games per final as B's frame, on two panels instead of
    three.
  - "A-B2 … removes 70.7 %" is for H, F and P72; REVIEW A's headline 84 % was the all-mons count.
- **D-m10. The box drops a qualifier and misnames the measure.** "15.9 %" is a lower bound ("≥", since pool fits are
  upper bounds). It is measured against the POOL, not the panels. **Fix:** relabel it
  `>= 1500 full-six sides unfamiliar to the pool (<= 2 fits; a lower bound)`.

---

## 4. Re-running the cheap checks

| check | how | result |
|---|---|---|
| the doc gate | `pytest src/human_data_doc_gate_test.py` in this worktree | 9 passed |
| planted mismatches the gate claims | the real doc mutated in memory, the gate's own check functions | **caught 5 / 5**: §5 file count; the §0 durable copy; the switch state; the HOM number; a renamed key. An unregistered `hr_spec_v2` in the plan is caught too |
| planted mismatches it does not claim | as above | **missed 7 / 7** (D-m1) |
| `plan_power.py` | re-run, diffed | identical to `results/plan_power.json`. It took 7 s, not "about 2 min" |
| the extract's refusal | `review_d_tamper.py` | digest: refused. Per-log hash and header: **hang** (D-M1) |
| the durable copy | `ARCHIVE.json`, the registry, the file modes | consistent; `-r--r--r--`; 376,410 members verified |
| privacy | the 77 mapped names, whole-word, over the 54 human-data files at HEAD | 0 hits |

## 5. The cold read (REVIEW C's method: the same 10 questions; page one = lines 1–48)

| # | question | page one (`431b1fdf`) | score | whole doc | score |
|---|---|---|---:|---|---:|
| 1 | What human data do we have? | the box: both snapshots, their file counts, 1.80M rows | 1 | §5 | 1 |
| 2 | Is any human data in training today? | the box and its words: no decisions; human-built teams yes | 1 | §2 | 1 |
| 3 | Which results depend on which snapshot? | §1's table | 1 | §1 | 1 |
| 4 | What does each toggle value do? | only "NOT BUILT"; no values | 0 | §6 | 1 |
| 5 | Can I type `--human-data` now? | "NOT BUILT" | 1 | §6 | 1 |
| 6 | What is unknown? | the `open` line, in undefined jargon | 0.5 | §7 | 1 |
| 7 | Can an eval result be contaminated by human data? | the box: a replay-derived panel; human-trained anchors | 1 | Q-H13 | 1 |
| 8 | How much of a replay row is real? | not on page one (fact 1 is in §3) | 0 | fact 1 | 1 |
| 9 | Can the archive train a lever? | not on page one | 0 | §4: DROP | 1 |
| 10 | Is there evidence human data will help goal (b)? | two positive-looking pinned numbers; fact 4 absent (misleading) | **0** | fact 4 | 1 |
| | **total** | | **5.5 / 10** | | **10 / 10** |

The whole doc now reads correctly end to end. Page one is the gap, and D-M8's three lines close most of it.

## 6. The four declined sub-fixes

| sub-fix | declined because | verdict |
|---|---|---|
| A-M2: re-fit the HOM on an account-disjoint split | "X59 no longer decides anything" | **NOT justified as stated.** True for X59, false for H-INIT, H-KL and the Δ_U routing rule (D-M5). Justified only once the existing HOM is barred from levers and routing |
| B-B1 fix 3: the λ logit mix | needs a two-network agent that `main.anchors` is UNVERIFIED to host; it would be read only after X59-S | **justified.** B offered it as optional, and X59-S answers the same question better |
| B-M6 fix 3: Foul Play as a meter | the head-to-head is faster at higher n, and Foul Play answers no question of its own | **justified for Foul Play.** But the accepted head-to-head half is missing from X60 (D-M6) |
| B-m10: the tier-A re-cut | the proxy is withdrawn and steers nothing | **justified** |

## 7. The first-reads order: is it the cheapest path to the owner's two questions?

1. **"Does a human-only model beat ours?"** Yes, FR-1 then FR-3 is the cheapest path. Both use pinned Metamon models,
   need zero build, and are registered and playing at HEAD (`57fd657b`). X59-S's S-H − P\* adds little: a one-epoch
   from-scratch student is not a fair "human-only model".
2. **The trace twin (FR-2) is not on X59-S's critical path once D-M2's restricted sampling is adopted.** It gates only
   HOM-based reads: Δ_U, H-INIT and H-KL. Run it in parallel, not in series.
3. **"Is human data better than self-play data?"** X59-S, with three changes:
   - S-S added (D-M4);
   - ≥ 2 seed pairs (D-M3);
   - a 200-game floor pilot first, which saves ≈ 14 lane-h if both students sit under 0.15.

   The cheapest decisive version is about **60 lane-h** (ESTIMATE at the plan's rates of ≈ 10 s a Kakuna game and
   373 rows / s): three students × two seed pairs at ≈ 2M rows, plus 2,800 games each. A smaller one cannot separate
   teacher from seed. The floor pilot can cut most of the game cost.

## 8. Reproduce

```
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES=""
D=designs/research_state/measurements/human_replay_quality_2026-10-25
nice -n 19 taskset -c 5-7,13-15 $PY $D/review_d_tamper.py                                   # < 3 min, a temp dir
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --max-s 300 4 $PY $D/review_d_checks.py power   # seconds
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 14 $PY $D/review_d_checks.py data               # ~5 min, 3.4 GB
```

Seed 20261027. The outputs are `results/review_d_tamper.json`, `results/review_d_power.txt` and
`results/review_d_data.txt`. The doc-gate mutations are §4's list, run through the gate's own `check_*` functions on
the doc text in memory.

## 9. FINDINGS (standing rule 7)

1. **The post-fill check is an approximation.**
   - It uses one move draw per side through `world_draws.draw_moves`, as C-GAME does, but omits C-GAME's legality
     redraws and its item, ability and spread draws.
   - It compares exact sorted move sets, the guard's key.
   - Its "does not re-create cores" conclusion is about move sets only.
2. **`forum216b` was not read** (sealed). The exclusion share and the kept profile are for the other four held-out
   panels.
3. **The student seed SD is unmeasured.** The table in D-M3 is a sensitivity range, not an estimate.
4. **The pool-unfamiliar share uses pool fits that are upper bounds**, so 31.5 % / 11.6 % are lower bounds, like the
   card's 15.9 %.
5. **HEAD moved during this review** (`57fd657b`: FR-1 / FR-3 registered, `hr_arch` copied). Its new PREREG and
   `hrrq.py` were not reviewed.
6. **Process.** CPU only; no GPU, game, training, download or launch. `models/`, `data/`, the corpus, the cache and
   the archive were only read; the tamper test ran on copies in a temp dir. The documents under review were not
   edited.
