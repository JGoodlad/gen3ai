# REVIEW H (causal): adversarial review of X59-S's experimental design and inference (2026-10-27)

**Under review:** [`PREREG.md`](PREREG.md) through Amendment 4c (`e652875c`); [`power.py`](power.py);
[`read_spec_pilot.json`](read_spec_pilot.json) and [`read_spec_full.json`](read_spec_full.json);
`src/main/human_data/x59s_read.py`, `x59s_ops.py`; `src/main/reads/rules_x59s.py`; [`designs/tools/reads.md`](../../../tools/reads.md)
§4 (one engine per read, the Kakuna team pack, replay determinism, `e28318c8`'s choice-level guard). Context: the Q-H
table of [`human_data_architecture.md`](../../../endstate/human_data_architecture.md) and [`REVIEW_E.md`](REVIEW_E.md).
Another reviewer covers data and bias; this one covers the experiment and the inference.

**Method.** I read the code against the PREREG. I simulated the registered rule where `power.py` does not look:
[`review_h_sim.py`](review_h_sim.py) (seeded, CPU, nice 19 on the agent cores, about a minute) writes
[`results/review_h_sim.json`](results/review_h_sim.json). I read the released artefacts under
`~/gen3ai_archive/x59s/read/`: `floor_refs.json`, `identity.json`, `fr2/fr2.json`, the floor units' argv and summaries,
and the input manifests (read-only). **I read no student cell.** The pilot was running (11 of 181 units settled at the
time of this review). The PREREG was not edited.

Abbreviations: as in the PREREG. **σ_arm** = an arm's student seed SD. **FPR** = the rate of a false directional class
on one side (nominal 2.5 %).

**Verdict: the design answers a narrower question than Q-H9, and one registered rule needs a decision before the
pilot is decided.**
- **B1 is time-critical.** The floor bar of 0.15 was set before the floor references existed. They now read random
  0.000 and max-damage 0.046, so the bar declares UNINFORMATIVE reads that Kakuna resolves easily.
  - A blind amendment is possible only until `main.reads decide` writes the pilot's `inputs.json`, which carries
    every student's wins (m6).
  - The alternative is to accept the rule as registered and tell the owner that UNINFORMATIVE is a likely answer.
- **Six MAJORs** concern the inference:
  - one arm's larger seed noise doubles the false-class rate (M1);
  - the equivalence class is on the wrong scale at the predicted bases (M2);
  - the arms' training rows straddle an observation-semantics change (M3);
  - two of the three U panels tilt toward S-H (M4);
  - the meter-agreement guard protects only one direction (M5);
  - the closure overreaches what a 7,041-step behaviour-cloning comparison can show (M6).
- Every fix runs on the CPU or is a dated amendment.

---

## BLOCKER

### B1. The 0.15 floor bar makes informative reads UNINFORMATIVE; the floor-reference band is dead code

**Evidence.**
- **The floor references** (`floor_refs.json`, engine `bdc5d2be…/94`, units 0–3, 240 games each): random 0 / 240;
  max-damage 11 / 240 = 0.046 (H 6 / 80, F 4 / 80, P72 1 / 80). A2.6's band is therefore "every arm ≤ 0.096".
- **The two rules overlap, and the bar always wins.** `x59s_read.verdict` sets UNINFORMATIVE if the PILOT floor fired
  (every arm's units-0–3 mean < 0.15), or if every arm's full-read rate is ≤ 0.096.
  - The band can only matter when the floor did NOT fire, which means some arm scored ≥ 0.15 on the pilot.
  - That arm would then have to fall by ≥ 5.4 pp between its pilot units and its full read. That essentially never
    happens.
  - In effect, UNINFORMATIVE ⇔ "every arm < 0.15 on the pilot". The measured floors enter nothing.
- **Under the floor, the read decides nothing.** `verdict` returns UNINFORMATIVE whenever `read_["floor"]` is true, so
  the in-house classes are reported and route nothing.
  - §4 still calls the in-house head-to-heads "the decider under the floor", and §5 still says the classes are "read
    on the in-house contrasts".
  - A2.6 superseded both (m2).
- **The PREREG's own prediction sits in this band:** arm means of 0.10–0.25. REVIEW E M4's case for the floor was
  "near-random students". With the references measured, an arm at 0.10–0.15 is two to three times the max-damage
  heuristic, which is not near-random.
- **Simulated** (`results/review_h_sim.json`, 40,000 reps, σ_arm 1.5 pp):

| true arm rates (S-H, S-P, S-S) | P(floor fires ⇒ UNINFORMATIVE) | what Kakuna's full read would have said |
|---|---:|---|
| 0.04, 0.04, 0.13 (a 3× gap) | **0.95** | Δ_data SELF-PLAY-DATA-BETTER with probability 1.00 |
| 0.13, 0.13, 0.08 | **0.91** | Δ_data HUMAN-DATA-BETTER 0.99 |
| 0.12, 0.09, 0.16 | 0.21 | Δ_data SELF-PLAY-DATA-BETTER 0.90; Δ_labels HUMAN-LABELS-BETTER 0.72 |

**What if some arms are above the bar and some below?** As registered, the floor fires only when EVERY arm is below,
so a mixed outcome is read normally on Kakuna. That is right for a contrast with one arm well above the floor (S-S at
0.18 against S-H at 0.05 is a real SELF-PLAY-DATA-BETTER). But a contrast whose BOTH arms sit at the floor is then
classified as if it were informative, and M2 shows it lands in NO-DIFFERENCE-AT-4PP. The registered "every arm" test
is the wrong unit: floor-ness belongs to a contrast, not to the read.

**Fix (a blind amendment, before `main.reads decide` runs on the pilot state; the orchestrator's call).**
- Replace the pilot's bar and A2.6's band with ONE per-contrast rule, both read from the pilot's units 0–3 on the
  same twins as the references: **a contrast is UNINFORMATIVE iff both of its arms' pilot means are ≤ the better
  floor reference + 5 pp (0.096)** (a tie within 1e-9 counts as within).
- The full read plays if any registered contrast is informative.
- An UNINFORMATIVE contrast is reported and routes nothing. The others are read as registered.
- Cost: the full read plays in more outcomes (≈ 720 units, ≈ 67 lane-h at the floor references' measured ≈ 5.6 min
  a unit, ESTIMATE).
- If the orchestrator keeps the 0.15 bar instead, record that the band is inert, delete §4's "decider under the
  floor" wording in that amendment, and tell the owner up front that a floor outcome answers nothing.

---

## MAJORs

### M1. σ̂ is pooled over all three arms, so one noisier arm doubles the false-class rate

**Evidence.** `x59s_read.seed_sd` pools W over the three arms (df 12), and `contrast` uses `2 max(1.5 pp, σ̂)² / k` for
every contrast. This assumes the arms share one seed variance.
- That assumption is untested. S-H fits a 1.87-nat human target, and S-S fits P\*'s labels at about 1.0 nat (REVIEW E
  M4), so S-H's seed spread may well be the larger one.
- **Simulated** at Δ = 0, base 0.20, σ_S-H = 4 or 5 pp and σ_S-P = σ_S-S = 1 pp: P(HUMAN-DATA-BETTER) = **0.048 /
  0.046** and P(SELF-PLAY-DATA-BETTER) = 0.049 / 0.044, against 0.026 at an equal σ of 3 pp. The same happens at a
  base of 0.30 (0.045).
- The guard does not save it. Pooling dilutes the noisy arm's variance (W ≈ (25 + 1 + 1) / 3), and the guard then
  hands only 52–71 % of these reads to Welch.

**Fix (blind, any time before the full record).** Take the wider of the registered z interval and the Welch interval
on the contrast's own 10 students. Equivalently, estimate σ̂ from the two arms in the contrast (df 8). Then report
each arm's SD beside σ̂ (A2.11 m2 already asks for it). Re-run `review_h_sim.py` against the amended rule to confirm
the FPR is back near 0.025.

### M2. NO-DIFFERENCE-AT-4PP is an absolute win-rate band sized at p = 0.5, and the read lives at 0.03–0.25

**Evidence.**
- **What the band means depends on the base.** A2.8 sends the owner "neither better nor worse by more than 4 pp". At
  a base of 0.10, 4 pp is a 40 % relative change (logit 0.41, against 0.16 at 0.50).
- **Simulated** (σ_arm 1 pp, S-S at 0.20 so the floor never fires): with S-H at 0.05 and S-P at 0.03 (a 1.7× label
  effect), Δ_labels reads **NO-DIFFERENCE-AT-4PP with probability 0.49**. That routes "H-TEAMS first, a ≤ 4 pp label
  effect is below X60's MDE", which is a statement about a different scale.
- At an equal base of 0.10 (σ 1.5 pp), NO-DIFFERENCE is the modal class (0.85).
- The opposite failure appears at σ_arm ≥ 3 pp: NO-DIFFERENCE is almost unreachable (≤ 0.10), so the class works only
  in the regime where it misleads.

**Fix (blind).**
- NO-DIFFERENCE-AT-4PP also requires the matching in-house interval to lie inside ±4 pp. The head-to-heads sit near
  0.50, where 4 pp means what it says.
- The owner's message carries both arms' base rates beside any Kakuna class.
- A logit-scale margin would also do; the in-house conjunction needs no new number.

### M3. The arms' training rows straddle an observation-semantics change; only S-S trained on the play-time encoder

**Evidence** (manifests, read-only):
- **S-H / S-P rows:** `hr_spec_v1.r1` parts were extracted at `da36173c9`, with obs golden `9a6f947b`. This
  predates `d8d674b99` (`gen3_volatile_silent_ends_v1`, golden 36 / 991 decisions) and `f9978586b` (lock-in and
  recharge counters). r2 carries these rows unchanged.
- **S-S rows:** `ss_rows_v2` was built at `d02550c0d`, with golden `95e24c32`.
- **The play tree:** the pin `0534e90a6` also has golden `95e24c32` (`git show <c>:src/agents/training/golden_obs_fixture.json | sha256sum`).

So S-S trains on exactly the encoder it plays with, and S-H and S-P train on the stale one: an active-mon volatile
left at 1 after Showdown silently ended it. The bias points against S-H and S-P, that is, toward SELF-PLAY-DATA-BETTER
and STATES-HURT, which is the direction that closes a lever.

Nothing in the PREREG mentions this:
- VOID 8's byte-equality was against the HOM extraction of `2bbdffa7`, also pre-change.
- A4c registered the drift for FR-2 only.

**Magnitude:** probably small. A4c's G-ENC on X26's games found 27 differing rows in 11 of 2,106 battles (about
0.03 % of rows), one cell each. **UNVERIFIED on human logs**, where Destiny Bond, Endure and trapping rates may differ.

**Fix (CPU, before the full verdict).**
1. Re-extract VOID 8's 200 seeded battles at the pin and count the rows whose bytes differ from r2's.
2. Record that count as a registered limit.
3. If more than 1 % of rows differ, the closure (SELF-PLAY-DATA-BETTER) and STATES-HURT close nothing until the S-H
   students' NLL on stale vs re-encoded twins is measured. That is `read_fr2`'s twin-gap machinery: under 0.05 nats
   stands.

The same check belongs in the ≥ 1500 follow-up, whose S-S cut and S-H≥1500 rows inherit the asymmetry.

### M4. Two of the three U panels tilt toward S-H; HUMAN-DATA-BETTER on U can be "learned the ladder"

**Evidence.**
- **What Q-H13 says about the panels:**
  - **H** (`heldout72_hpfix`) is the human levers' own population: ladder sides with usage fills, so "a gain there
    alone is 'learned the ladder'".
  - **P72** is fill-adjacent: its generator draws from the same Smogon tables C-GAME used to fill S-H's own sides.
  - **F** is the cleanest open panel. That is why X60's primary is F alone.
- **X59-S does the opposite:** its primary is the equal-weight U over H, F and P72, and the per-panel Δ is
  descriptive. A Δ_data carried by H and P72, with F flat, classifies HUMAN-DATA-BETTER and keeps the levers open.

**Fix (blind).**
- Add a POPULATION-SPECIFIC demotion, mirroring KAKUNA-SPECIFIC: a HUMAN-DATA-BETTER or HUMAN-LABELS-BETTER on U
  whose F-only point estimate is ≤ 0 opens no lever.
- Report the per-panel Δ rows with intervals (the §4 interval on one panel's 400 games a student). Those rows are not
  findings on their own.

### M5. The meter-agreement guard is one-sided: the closure needs no in-house agreement

**Evidence.** `verdict` demotes a "better" class to KAKUNA-SPECIFIC when the in-house point estimate is ≤ 0 (A2.10).
Nothing checks SELF-PLAY-DATA-BETTER (or OWN-LABELS-BETTER) against the in-house meter, yet these are the classes that
close a lever or drop H-BC.
- A2.10's reason, "Kakuna is human-pretrained, so it is not a neutral meter", predicts no direction. A
  human-pretrained opponent may punish human-like play as easily as reward it.

**Fix (blind, a one-line change to `verdict`).** SELF-PLAY-DATA-BETTER closes nothing, and OWN-LABELS-BETTER drops
nothing, when the matching in-house point estimate is ≥ 0 (a tie within 1e-9 counts as ≥ 0).

### M6. The closure claims more than a 7,041-step behaviour-cloning comparison can show (external validity)

**Evidence.** Every student is a fresh 2.6M-parameter network trained for 3 epochs (7,041 steps at batch 512, about
3.6M samples) by an untuned behaviour-cloning recipe. P\* is 280M steps of PPO.
- **Fitting speed is confounded with data value.** At matched steps, the arm with the cleaner single-policy target
  fits faster: S-S's labels sit at about 1.0 nat, S-H's at 1.87 (REVIEW E M4). An underfit comparison can rank the
  data by how fast it can be fitted, not by what it is worth.
- **The closure's scope** (A2.7) is "1300+ ladder data, at this mix and volume". It omits "at this budget" and "as a
  behaviour-cloning target for a fresh network". Neither appears in the owner's message.

**Fix (blind).**
- Add "at 3 epochs of behaviour cloning from scratch" to the closure's wording and to the owner's message.
- Gate the closure on S-H being fitted. Reuse m5's flag in reverse: S-H's validation NLL must have stopped falling
  over the last epoch, for example a last-quarter-epoch drop of < 0.01 nat. The threshold is registered before the
  record is read.
- Otherwise SELF-PLAY-DATA-BETTER reads "at this budget" and closes nothing.

---

## MINORs

- **m1. The pilot's early stop inflates the false-class rate slightly near the bar.** The pilot units enter the final
  estimate, and the read continues only when some arm's pilot mean is ≥ 0.15.
  - At a null base of 0.14 (σ 2 pp), the FPR conditional on continuing is 0.039 / 0.041, against 0.029 / 0.029
    unconditionally. At 0.15 (σ 3 pp) it is 0.029 against 0.027.
  - Fix: report P(continue) beside the verdict, or drop the pilot units from the contrasts (this costs 20 % of the
    games). B1's per-contrast rule changes the conditioning; re-simulate it.
- **m2. The PREREG text and the verdict string contradict A2.6.** §4 says "the decider under the floor" and §5 says
  "read on the in-house contrasts". `rules_x59s._full_decide` prints "FLOOR (in-house decides)" and then
  "UNINFORMATIVE". Fix: one wording in the next amendment, and the string changed to "FLOOR (UNINFORMATIVE; in-house
  reported)".
- **m3. The seed pairing is unused.** S-H_k and S-P_k share an init and batch order, and S-S_k shares the init. The
  registered V treats the arms as independent, which is conservative. Fix: report the paired per-seed contrast (t,
  df 4) as descriptive.
- **m4. The pin predates `e28318c8`.** `main.reads` loads the rule in the playing checkout (`0534e90a6`), so:
  - VOID 6 compares only the 5-tuples, which miss about a quarter of divergences (reads.md §4: 11 vs 15);
  - H21's "any peer fault voids the unit" is absent. H5's mid-half fault scores the next game as a 1-turn Kakuna
    FORCELOSE (a win for us), and turn-250 forfeits depend on the student, so its rate can differ by arm. The archive
    shows 0 of 950 counted units with the fault line.
  - Fix (CPU, before the full verdict): scan every counted X59-S unit's peer log for `session.PEER_FAULT_LINE`, uncount
    and re-run any hit, and re-run the repeat check with `assemble.compare_choices` from HEAD on the two captures.
- **m5. The in-house engine is not machine-checked.** `_full_decide` checks one engine across the Kakuna groups and
  the floor references only. Fix: `x59s_ops inhouse` records the engine stamp, and the rule refuses on a second one.
- **m6. The pilot's blinding is partial.** A2.11 m3 keeps arm rates out of `decision.json`, but `inputs.json` and the
  record's `inputs` carry every student's per-panel wins. Fix: the pilot's record stays in the archive, uncommitted,
  until the full record exists. Any blind amendment (B1, M1, M2, M4, M5, M6) lands before the pilot's `decide`.
- **m7. S-P0's Δ_restriction has no registered code.** `x59s_read.read` takes exactly three arms. Fix: register which
  σ̂ pool and which interval it uses before its games are read.
- **m8. The H-BC gate barely falsifies.** "S-P's yardstick on P\*'s labels is below S-H's" holds almost by
  construction, since S-P trained on P\*'s labels. Fix: a midpoint test, which needs no guessed number: S-P's mean
  yardstick must lie closer to the S-S students' mean than to the S-H students' mean.
- **m9. "Q-H10 in our architecture" cannot answer Q-H10.** A 7,041-step student against a 280M-step P\* is OURS-BEATS
  by construction, across engines too. FR-1 and FR-3 already answered Q-H10. Fix: rename it "the student-to-P\* gap"
  in the record and the Q-H table.
- **m10. The teams are a fixed effect.** The interval is conditional on the 216 panel teams, while the claim is "plays
  unfamiliar teams better". Fix: report a team-cluster bootstrap of Δ_data as descriptive. If it crosses 0 where the
  registered interval does not, the owner's message says "on these 216 teams".
- **m11. FR-2 gates the closure through a proxy.** Its NO-ARTEFACT is read on the HOM, not on the students. Its
  direction is reassuring: D = −0.149 [−0.163, −0.134], with the HOM's twin gap at 0.069 nats against P\*'s 0.218.
  Fix: run `read_fr2`'s twin gap on S-H_k against S-S_k, as descriptive (about 9 min a pair at 2 threads, measured
  on FR-2's own run).
- **m12. The band compares unlike team sets.** It compares full-read arm rates (units 0–19) with floor references
  from units 0–3. This is moot under B1's fix, which reads both on the pilot twins.
- **m13. X59-S's forum72 rows are the legacy-peer-pack regime.** They predate `b0441623e`: Kakuna's `sft431` Gyarados
  plays Frustration at base power 1. This is consistent within the read, and X51 shares it. Fix: the record names the
  regime, and no later F read (X60's primary) is set against X59-S's F numbers.
- **m14. Multiplicity is acceptable as registered.** The two registered contrasts each run at 95 % without
  correction. The "better" routes are conjunctive (the in-house sign, the yardstick gate), so their family-wise rate
  under the global null stays below 5 %. The risk lies in the descriptive rows: 9 per-panel Δs, 3 Q-H10 lines,
  Δ_restriction, the follow-up's 2 contrasts and the 2 in-house reads, all without intervals in places. Fix: every
  descriptive row carries its interval and a "descriptive, no claim" stamp. No per-panel or Q-H10 line enters
  `UNDERSTANDING.md` or the ledger without its own registration.
- **m15. Power at a realistic seed SD.** The 80 % MDE of about 4 pp holds only for σ_arm ≤ 1.5 pp. Simulated at base
  0.20: at σ = 3 pp, +4 pp gives P(LB > 0) = 0.40 and +6 pp gives 0.71; at σ = 5 pp, 0.18 and 0.34. PREREG FINDING 5
  already calls the table a sensitivity range. Fix: none before the data. The record quotes the MDE at the measured
  σ̂, not the registered one.

---

## Checked, no finding

- **A4c's timing could not have steered FR-2.** The exclusion is defined by protocol tokens and was registered before
  FR-2 ran. It removes 2,860 rows against 62,415 paired decisions kept, and D's upper bound is −0.134 against the
  0.05 bar. To reach UNCLEAR, the excluded rows would have needed a D of about +4 nats, against gaps of 0.07–0.22
  nats.
- **A4 and A4b were made before any student game.** The only outcomes seen were the identity PASSes (the gate itself)
  and the students' final validation NLLs, which no changed rule reads. `floor_b712_aborted` holds one start event and
  no result.
- **The floor references played the right panels.** The argv's `--teamset home` is the configuration's slot, set by
  `unit_env(config=…)`. Each summary's `our_team_source` is the panel's own directory (H, F and P72), on the read's one
  engine.
- **Q-H10's cross-engine demotion is correct and sufficient.** E0 → E1 crossed no observation change (1,360 / 1,360
  equal), and E1 → E3 differs on 11 / 1,260 games (reads.md §4). Both X51 and X59-S play the legacy forum72 pack.
  - Nothing else in the registered contrasts crosses an engine or a peer-pack regime.
  - What does cross a regime is the TRAINING encoder (M3) and the in-house meter's unchecked engine (m5).

## What the read cannot conclude, even with every fix

1. **Whether adding human data helps P\* or any production run.** The students are fresh, small and underfit; the
   question for a plateau fork is X60's.
2. **Whether human data is worth more than more self-play.** The comparison matches rows, but self-play rows are free
   and human rows are capped (3,497 games a day while the collector ran; it is deleted). HUMAN-DATA-BETTER per row is
   not "worth collecting".
3. **Anything about ≥ 1500 play, or human data in general.** The slice is 77.5 % 1300–1499 actors, forfeit-heavy and
   panel-excluded (§8.2).
4. **Anything about the RL value of the data.** S-S distils P\*'s labels; production self-play feeds PPO's
   advantages, not cloning targets.
5. **Anything about Q-H10 at scale** (m9).
6. **That two arms are equivalent at a low base** without the in-house interval (M2).
7. **Anything beyond the 216 panel teams** without m10's team-level check.
