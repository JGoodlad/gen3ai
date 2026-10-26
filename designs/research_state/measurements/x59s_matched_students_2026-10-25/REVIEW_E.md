# REVIEW E: adversarial review of the X59-S registration (before any GPU time, 2026-10-25)

**Under review:** [`PREREG.md`](PREREG.md) with Amendment 1 (`5cbf46dc` … `e0d76a51`);
[`results/registration.json`](results/registration.json); `src/main/human_data/` (`student.py`, `rowset.py`,
`teacher.py`, `textpass.py`, `checks.py`, `x59s_read.py`, `snapshots.py`); the rowset and self-play manifests under
`~/gen3ai_archive/x59s/` (read-only). Context: [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md),
[`human_data_architecture.md`](../../../endstate/human_data_architecture.md), REVIEWS A–D and
[`RESPONSE.md`](../human_replay_quality_2026-10-25/RESPONSE.md).

**Method.** I read the code and checked it against the prereg. I re-measured what the brief's seven questions turn
on, using the built rows, the S-P labels and the S-S rows:
- [`review_e_checks.py`](review_e_checks.py) is seeded and CPU-only. It ran at nice 19 on the agent cores under
  `mem_cap.sh` (peak 0.8 GB). Its output is [`results/review_e_checks.json`](results/review_e_checks.json).
- The `rows` part reads 5 fixed TRAIN shards: 81,920 rows, 75,096 of them labelled.
- The prereg was not edited. Nothing touched `models/` or the archives except reads.

Abbreviations: as in the PREREG. **Fit** = the plan §3.1 rule (`textpass.PanelMatcher`: a mon fits a panel set when
its species, every revealed move, and its revealed item and ability match). **Near** = ≥ 3 fits to one panel team.

**Verdict: NOT-READY.** Every fix below runs on the CPU or is a dated amendment. None of them needs the GPU. Two
BLOCKERs bias or hollow out the reads the routing depends on, and the data cannot repair either one after training:
- **B1** biases the owner's own contrast (Δ_data) toward self-play.
- **B2** registers a sensitivity read that cannot be run without retraining.

Once B2 and M1 are settled, S-H and S-P can train. S-S has to wait for B1's regenerated rows. That is about 2 to 3 h
of CPU at the measured 200 rows a second.

---

## BLOCKERs

### B1. S-S's states are held to a much looser panel-proximity standard than S-H's: Δ_data is biased toward self-play

**Evidence.** The human rowset drops any battle in which either side's REVEALED facts fit ≥ 3 mons of one held-out
team (§2.1 rule 2). That rule removes 83.2 % of candidate battles, and REVIEW D found it "removes the mainstream".
The S-S rows play the 719-team pool with no exclusion at all.

The U panels were drawn to reject only ≥ 4 exact sets shared with the pool (`heldout_panels.json` draw rules). So by
construction the pool sits at or just under the threshold the human data is cut at.

I applied the same matcher to the pool, scored panels only (H, F, P72 and the retired heldout72; `forum216b` was not
used), under three reveal profiles:

| reveal profile per pool mon | pool teams near a scored panel | a self-play game with a near side | near F (forum72) |
|---|---:|---:|---:|
| every fact (4 moves, item, ability) | 27 (3.8 %) | 7.4 % | 26 |
| two seeded moves, no item | 376 (52.3 %) | 77.2 % | 270 |
| one seeded move, no item | 665 (92.5 %) | 99.4 % | 566 |

A human side reveals 5.14 mons on average and only some of each mon's moves. Under a realistic reveal profile, then,
**about three quarters of S-S games would fail the rule that S-H's battles had to pass**. That matches the 71–83 %
the human corpus lost to the same rule.

S-S therefore trains on mainstream, panel-adjacent states: 26 pool teams share 3 exact sets with an F team. S-H trains
on the panel-distant remainder of the ladder. This is the "familiarity" advantage the exclusion was built to remove,
and it now sits in one arm only. Δ_data and Δ_states inherit it in S-S's favour. Δ_labels does not, because S-H and
S-P share their states.

A SELF-PLAY-DATA-BETTER read would close human data as a lever (§5) partly because of this asymmetry.

**Fix (before S-S trains; CPU).** Draw S-S's games only from pool teams that pass the human rule under a reveal profile
matched to the kept human sides:
1. Measure per-mon revealed-move counts from the rowset's `reveals`.
2. Simulate them seeded, and include `forum216b`, exactly as the human rule does.
3. Regenerate S-S from the surviving teams. Record each game's team pair; `play_part` records none today. The
   generation costs the same as before, because no game is wasted.

Also register Δ_data per panel as descriptive. If the regeneration is refused, the minimum is:
- drop the 7.4 % of games with a fully-revealed near side (the team pairs can be rebuilt from `[5900, part, 0]`'s
  draw order); and
- register the residual asymmetry in §8 as a limit that can only favour S-S.

### B2. The Kakuna-family sensitivity cannot be read as registered, and the audit misses 78 % of the rows

**Evidence.** §8.4 says "the primary is read without [the audit]; its sensitivity re-read follows the audit". The
plan's A-6 falsifier is "the primary re-read without the flagged accounts".

For X59-S that re-read is not a re-read. The students are trained once. Removing flagged accounts means retraining
S-H and S-P on a filtered rowset (10 more students), and nothing registers that cost, a threshold or a trigger.

The audit is also scoped (plan §3.4) to **≥ 1500 accounts with ≥ 50 TRAIN games**. In this rowset, 78.0 % of rows
come from 1300–1499 actors, so most of S-H's teacher is outside the audit by its own definition.

The name rule (`textpass.BOT_NAME` catches `kakuna`, `metamon`, `pac-`) and the volume and hours rules cannot see a
fleet that runs at under 100 games a day under ordinary names. That is the plan's own stated reason for the audit.

The primary meter is Kakuna. If S-H imitated Kakuna-family decisions, it would score as skill against the same
policy and produce a false HUMAN-DATA-BETTER or HUMAN-LABELS-BETTER, and the latter routes H-BC.

**Fix (amendment, before S-H/S-P train).** Preferred: run the audit first. It costs about 6 lane-h and an agent-day,
all on the CPU. Extend it to every ≥ 1300 account with ≥ 20 TRAIN games, and build the rowset without the flagged
accounts.

Otherwise, register:
- a threshold (for example, flagged rows > 1 % of TRAIN rows);
- that crossing it triggers a retrain of S-H and S-P on the filtered rowset;
- that the retrained read REPLACES the primary for both contrasts;
- the GPU cost (about 2 GPU-h).

Without one of the two, A-6 cannot be falsified.

---

## MAJORs

### M1. The quitter's degraded play runs well past the 2 dropped decisions

**Evidence.** I measured P\*'s agreement with the human against the number of decisions to the seat's last KEPT row
(`results/review_e_checks.json` `tail_by_decisions_to_last_kept_row`):

| decisions to the last kept row | 0 | 1 | 2–3 | 4–7 | 8–15 | ≥ 16 |
|---|---:|---:|---:|---:|---:|---:|
| quitter: top-1 agreement / NLL | 0.220 / 2.02 | 0.261 / 2.00 | 0.303 / 2.02 | 0.316 / 2.08 | 0.348 / 2.03 | 0.352 / 1.97 |
| forfeit winner | 0.571 / 1.24 | 0.501 / 1.36 | 0.431 / 1.61 | 0.385 / 1.81 | 0.387 / 1.81 | 0.364 / 1.92 |
| KO games, both seats (incl. the loser) | 0.464 / 1.13 | 0.426 / 1.22 | 0.359 / 1.41 | 0.337 / 1.60 | 0.345 / 1.79 | 0.354 / 1.97 |

Near the end of a game, agreement RISES for everyone (endgames are simpler) except the quitter, whose agreement falls.
This holds for the KO-game loser too, who is also in a lost position. The anomaly spans about 8 decisions beyond the
2 already dropped.

Those rows are **7.5 % of rows** (the quitter within 8 decisions of the last kept row). Changing the cut is a data
fix that only works before training.

**Fix.** Drop the quitter's rows with `quit_dist ≤ 9` (the field is in every shard). Then recut S-S's merge to the new
counts: the parts hold the rows, and the merge is minutes. Register the cut with this table as its evidence, which
is forced by the data and not by an outcome.

The `≤ 10`-turn filter is sound. It removes walk-overs and rage-quits symmetrically for both seats, and no agreement
evidence argues against it.

### M2. S-P is P\* restricted to the human's whole-game support: not an independent teacher

**Evidence.** `consistent_mask` takes its `reveals` from `textpass.record`, which reads the WHOLE log. At decision k,
the consistent set is every move this mon used at any point in the game, plus every mon the human ever brought in.
That is the human's realised support, future included.

What the restriction does (`rows.consistent`):

| quantity | value |
|---|---:|
| P\*'s expected probability of the human's action: unrestricted → restricted | 0.364 → 0.450 |
| S-P's sampled label equals the human's | 45.1 % |
| P\*'s top-1 is outside the consistent set (S-P cannot follow P\*'s choice) | 16.8 % of labelled rows |
| P\* mass removed: from moves / from switches | 14.4 % / 3.9 % |
| switch share: human / S-P labels / P\* unrestricted / S-S labels | 31.4 % / 37.2 % / 34.0 % / 29.8 % |
| rows whose consistent set is the whole legal set | 25.5 % |

On top of this, the 8.2 % hidden rows give S-P the human's own set label.

So S-P is a hybrid: P\*'s preferences inside the human's chosen support. Because unverified moves are removed, mass
shifts toward switching, and S-P ends up switching more than either the human or P\* does. This pulls in two
directions:
- S-P is pulled toward the human, which shrinks Δ_labels.
- S-P is distorted away from P\*, so it is not "P\* as a teacher".

The restriction is the correct fix for the drawn-option signature (REVIEW D M2), but the prereg describes S-P as "P\*'s
labels" and lets Δ_labels route H-BC. Δ_states (S-P − S-S) also mixes three things: state coverage, this distortion,
and M3's signature.

**Fix (amendment).**
- Register S-P's definition with these numbers in §8.
- Rule that OWN-LABELS-BETTER is conservative (it holds against a handicapped P\*), while HUMAN-LABELS-BETTER is
  reported "against restricted P\*". It routes H-BC only if the S-P student's yardstick on S-S validation (P\*'s labels
  on P\*'s states) is within a registered margin of the S-S student's.
- Optionally add S-P-full (P\* unrestricted, 5 students, about 1 GPU-h) to measure what the restriction costs.

### M3. The drawn-move signature is matched across S-H/S-P but not against S-S

**Evidence.** Plan §3.5 says X59-S "matches it across the label arms by construction". That holds for S-H vs S-P only.

In S-H and S-P, a C-GAME-drawn move or bench mon is never a target. Per row, legal moves average 3.54 and consistent
moves 2.42, so about 1.1 listed moves per active mon are never chosen. The students learn this pattern from inputs in
which drawn and real options look the same. S-S's own side is true, so S-S has no such signature.

Δ_data and Δ_states carry it against S-H. FR-2, which would say whether a model learns the signature, is registered
and has not been run (plan §2.2).

**Fix.** Register this in §8 as a confound that can only cost S-H, and read FR-2 (about 1 lane-h) before Δ_data is
reported. If FR-2 reads ARTEFACT, a SELF-PLAY-DATA-BETTER does not close the lever.

### M4. No UNINFORMATIVE outcome, and the floor fallback answers a different question

**Evidence.** The prereg predicts arm means of 0.10–0.25 and says "the floor pilot may fire". At 3 epochs of about
2,600 steps (about 7,800 steps), a from-scratch network trains on 27,586 battles, far below the corpora behind
Metamon's IL agents (hundreds of thousands of battles; **UNVERIFIED** here).

The S-H target is hard: P\* itself predicts the human labels at 1.87 nats against 1.95 for uniform over the legal
actions (top-1 36 %). The S-S target is easy: P\*'s own labels sit at about 1.0 nat and top-1 60 % (`ss`). So matched
steps are not matched fit: at a fixed budget the arm with the cleaner, single-policy target fits better. That is
part of what the data is, but it pushes S-H toward the floor.

Under the floor, an in-house head-to-head of near-random students decides Δ_data. SELF-PLAY-DATA-BETTER would then
"close human data as a goal (b) lever" on the evidence of which beginner beats which.

The prereg has no floor reference: random-legal and max-damage against Kakuna on the same twins. It has no
"learned nothing" guard either.

**Fix (amendment).**
- Register a floor reference: random-legal and a max-damage heuristic, 240 games each on the pilot twins.
- Register an **UNINFORMATIVE** class: every arm's U rate is within 5 pp of the better floor reference, or the floor
  fallback fires. Under it, the in-house classes are reported but close nothing and route nothing beyond "H-TEAMS
  first".
- Add a VOID guard: a student whose final validation NLL on its own labels is not at least 0.1 nat below the uniform
  NLL (1.95 human, 1.86 S-S) did not learn (a bug or a recipe failure).

### M5. The teacher is mostly 1300–1499, but the closing rule speaks for human data in general

**Evidence.**
- Ratings: 49.2 % of rows come from 1300–1399 actors, 30.9 % from 1400–1499, and 292,412 rows from ≥ 1500.
- §8.2 says the verdict speaks only for the kept slice. §5 nonetheless lets SELF-PLAY-DATA-BETTER "close human data as
  a goal (b) lever at this volume", and that would also close the ≥ 1500 data the plan ranks as the real teacher.
- P\*'s agreement improves only mildly with rating (NLL 1.895 → 1.808 from 1300–1399 to ≥ 1600). That is weak
  evidence that the strata differ little by P\*'s standard, but it is no test of what they teach.

**Fix (amendment).**
- Scope the closure to "1300+ ladder data, at this mix and volume".
- Register a CONDITIONAL follow-up that fires when Δ_data is not HUMAN-DATA-BETTER. It has three arms of about
  292k rows each, 5 seeds each:
  - S-H≥1500;
  - S-H-mix, a seeded account-level subsample of the whole rowset to the same rows;
  - S-S cut to the same rows.
- Each student costs about 0.22 of a full student (about 0.7 GPU-h for all 15), and the arms are read in-house, so
  rating is separated from volume.
- Without that follow-up, "is human data better?" is answered only for mediocre humans.

### M6. NOT-RESOLVED merges "no difference" and "no information", and it is the predicted outcome

**Evidence.** The prereg predicts Δ_data ≈ −1 pp (−6 to +4), and the 80 % MDE is about 4 pp. The most likely class
is therefore NOT-RESOLVED. As registered, that class says the same thing whether the interval is [−2, +2] pp or
[−9, +7] pp.

Under the measurement rule ("equivalence needs the delta's CI inside the bar"), the narrow case IS an answer to the
owner: "human data is not better by more than X".

**Fix (amendment).** Split NOT-RESOLVED (and UNCLEAR) into two classes:
- **NO-DIFFERENCE-AT-4PP**: the 95 % interval lies inside ±4 pp;
- **NOT-RESOLVED**: otherwise.

Each gets its action. The registered bar is the MDE.

### M7. The CUDA path has no registered pass criteria

**Evidence.** The prereg says "the first GPU student's first chunk is the CUDA smoke" and gives no criterion.

The code builds the network on the CPU and moves it with `.to(device)`. `inference_globals(threads, "cuda")` turns
TF32 off for both the matmul and cuDNN paths. The policy stays fp32, because `build_fresh_model` never calls
`apply_run_precision`.

The CPU path is tested. The CUDA-specific parts are not:
- `th.load(map_location=device)` on resume;
- the optimiser state on the device;
- that the export of a CUDA-trained state on the CPU agrees with the trained network;
- throughput;
- memory: rows are 2,845 float32, so an 8-shard buffer is 1.5 GB of host RAM, with about 3 GB transient during the
  concatenate. Three concurrent students take about 10–13 GB, and §9's command runs them without `mem_cap`.

The window's "repeated until it exits 0" loop would also retry a REFUSED exit 1 forever.

**Fix: register the checks for the first GPU minutes, each pass or fail.**
1. `init_sha256` on CUDA equals the CPU-built digest of that seed.
2. Run S-H 5901 for 200 steps on CPU and on CUDA from the same init, into scratch dirs:
   - the step-0 validation NLL agrees within 1e-5;
   - the step-100 and step-200 losses agree within 1e-3;
   - the parameter relative L2 distance at step 200 is < 1e-3.
3. Print `allow_tf32` (matmul and cuDNN) and `get_float32_matmul_precision()` inside the loop: they read
   False, False and `highest`.
4. Resume parity: run 100 steps, then resume to 200. The step counter and the lr match an uninterrupted run, and the
   loss agrees within GPU nondeterminism. The optimiser state is on the device.
5. Export: the bare-load check passes. The CPU logits of the exported student agree with its CUDA logits within 1e-4
   on 256 validation rows.
6. ms/step and peak host and GPU memory, extrapolated to 15 students. The window runs each student under
   `mem_cap.sh`.
7. The retry loop continues only on exit 3, and stops and reports on any other exit.
8. The training loss is below the uniform NLL by step 1,000.

### M8. No rule for when Kakuna and the in-house meter disagree

**Evidence.** §8.3 grants that Kakuna is human-pretrained and "not a neutral meter of human-likeness". The in-house
head-to-heads are "always reported". Yet no rule says what happens when a Kakuna-scored HUMAN-*-BETTER meets an
in-house contrast at or below 0. X60 has such a rule (D-M6: a gain only Kakuna sees is "Kakuna-specific"); X59-S does
not. Together with B2, that leaves the route to H-BC open to a Kakuna-specific artefact.

**Fix (amendment).** A HUMAN-DATA-BETTER or HUMAN-LABELS-BETTER read on Kakuna whose in-house point estimate is ≤ 0 is
reported as "Kakuna-specific". It opens no lever and routes no H-BC.

---

## MINORs

- **m1. Matched rows are not matched label information.** S-H and S-P carry 8.2 % set labels on hidden rows; S-S
  carries none. Report the effective labelled rows (1,222,236 against 1,330,809) beside the arms.
- **m2. The false-positive control is slightly anti-conservative.** `power.json` gives 0.028 one-sided at a seed SD of
  2–3 pp against a nominal 0.025. The pooled W also assumes equal seed SD across arms. Report each arm's 5-student SD
  beside σ̂.
- **m3. Blind the floor pilot.** Release only the floor boolean until the full read. State which teams units 0–3
  cover, so the pilot rate is representative. The stop rule is a symmetric function of the arm means and the pilot
  units are counted in the final read, so leakage into the contrasts is small.
- **m4. The trainer does not refuse unregistered inputs.** `train` records the digests but never checks them against
  `results/registration.json`, so VOID 1 is post hoc. Check them in `train` and `export`.
- **m5. Report overfitting.** The validation curves are descriptive. Report any arm whose validation NLL rises over
  its last epoch, because a fixed budget can overfit one arm and underfit another.
- **m6. The cleaner S-S target (M4's 1.0 vs 1.87 nats) is part of the data, not a bug.** Report the yardsticks
  beside every read, so a reader can see fit and play together.
- **m7. Turn-length asymmetry.** S-S rows are longer-game (median turn 19 against 15) and have fewer legal actions
  (9 legal actions on 24 % of rows against 38 %). This is state-distribution content of the data; report it.
  Self-play has no `≤ 10`-turn filter. P\* games that short are rare, but report their share.

## What I checked that passed

- **S-S seat 1 is sound.** P\* re-scores its recorded labels at the policy's own entropy for both seats: seat 0 NLL
  1.032 against entropy 1.008; seat 1 1.012 against 0.996. A seat-1 observation defect would split them.
- **Registration.** The S-S manifest's P\* sha256 equals the registered `814823c5…91ed`, and the shard sizes are
  equal.
- **Consistency.** The S-P writer and the trainer both refuse a label outside the consistent set. Hidden rows carry
  −1 in S-P, and the trainer checks that its hidden rows equal the rowset's.
- **Matched batches.** Same seed gives the same init and the same batch plan, so S-H and S-P see identical batches.
- **The interval code.** `x59s_read.py` implements §4's interval as written: the per-student binomial sum / k² equals
  b̄ / k.
- **Leakage into S-H.** Leakage of the panels into S-H is controlled conservatively. The human rule is stricter than
  the panels' own pool rule (B1 is the mirror of this). Player overlap with the panels' source replays does not
  matter: those battles fail the ≥ 3-fit rule. What remains is player style, which I judge negligible.

## FINDINGS (standing rule 7)

1. **`forum216b`: one exploratory pass read its texts.** My first, uncommitted run of the pool check used
   `PanelMatcher()`'s default panel set, which includes the sealed `forum216b`, through the sanctioned exclusion API
   (`panel_team_texts`, counts only). It printed only how many pool teams are near it. The committed check and every
   number above use the scored panels only. No team content was printed or stored.
2. **Not verified.** I did not run any GPU or CUDA path, train any student, or play any game. M7's numbers (buffer
   RAM, throughput) are computed from the shard shapes and are not measured.
3. **Sampling.** M1's distance-to-end is measured within 5 shards. A battle cut at a shard edge reads a little short.
   The quitter-vs-control contrast is far larger than that error.
4. **A live training run was on the box** (`rb_x35_d3mut_s3502`). Every job here ran at nice 19 on the agent cores
   under `mem_cap.sh`.
