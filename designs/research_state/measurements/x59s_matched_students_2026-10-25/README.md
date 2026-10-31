# X59-S: is human data better than our own self-play data? (read 2026-10-30; closure blocked the same day)

Registered in [`PREREG.md`](PREREG.md) (`5cbf46dc0`) and amended before every read it governs (Amendments 1–6; the
blind ones, 5 and 6, answer [REVIEW H](REVIEW_H_causal.md) and [REVIEW G](REVIEW_G_data_bias.md)). Abbreviations
are the PREREG's. U = the equal-weight mean of H (`heldout72_hpfix`), F (`forum72`) and P72; pp = percentage points.

## Answer

**SELF-PLAY-DATA-BETTER.** Matched in volume, recipe and initial network, a student taught by our plateau policy's
own games and choices (S-S) beats one taught by 1300+ ladder players' games and choices (S-H) against Kakuna on
unfamiliar teams by **7.28 pp [4.28, 10.29]** (S-H 0.320, S-S 0.393), and the in-house meter, which never trained on
human data, agrees: **−9.06 pp [−10.98, −7.14]**. **It closes nothing:** the registered guard Δ_restriction (S-P0 −
S-P) read **+2.85 pp [+0.14, +5.56]**, so part of the gap is what restricting a label to the human's whole-game support
(the drawn-move signature) costs, and human data is NOT closed as a lever. Human LABELS do lose to P\*'s (H-BC is
dropped).

| contrast | Kakuna, U (5 students an arm) | class | in-house (5 pairings) | what follows |
|---|---|---|---|---|
| **Δ_data = S-H − S-S** (the owner's question) | **−7.28 pp [−10.29, −4.28]** | **SELF-PLAY-DATA-BETTER** | −9.06 [−10.98, −7.14] | **closes nothing** (Δ_restriction's LB > 0); the ≥ 1500 follow-up fires |
| **Δ_labels = S-H − S-P** (which teacher, on human states) | **−4.50 pp [−7.42, −1.58]** | **OWN-LABELS-BETTER** | −5.23 [−7.30, −3.15] | **H-BC dropped** (conservative: against P\* restricted to the human's support) |
| Δ_states = S-P − S-S (descriptive) | −2.78 pp [−5.68, +0.12] | UNCLEAR | — | routes nothing |
| **Δ_restriction = S-P0 − S-P** (the closure's guard, A6.1) | **+2.85 pp [+0.14, +5.56]** (S-P0 0.393) | RESTRICTION-COSTS | — | **blocks the closure** |

- **The closure is blocked.** Amendment 6 A6.1 made Δ_restriction (S-P0 − S-P: the same human rows and the same
  teacher, P\*, with and without the labels restricted to what the log allows the human) the guard: a lower bound > 0
  closes nothing. It read +2.85 pp [+0.14, +5.56] (300 units, all counted; σ̂ pooled over the two arms, widened to
  their Welch). Every other gate had passed (S-H fitted, re-encode 0.60 %, Δ_states' LB ≤ 0, Δ_labels not
  HUMAN-LABELS-BETTER, both meters and F alone agreeing). **Descriptive:** S-P0, P\*'s unrestricted labels on HUMAN
  states, reads 0.393, the same as S-S (0.393): on these students the human states cost nothing once the labels carry
  no signature, so S-H's 7.28 pp is human labels (S-H − S-P −4.50) plus the restriction (S-P − S-P0 −2.85), not human
  states.
- **The arms' rates:** S-H 0.320 (SD 2.07 pp over 5 students), S-P 0.365 (1.93), S-S 0.393 (2.04). σ̂ = 1.47 pp
  (df 12; the guard did not fire); every interval is A5.2's, the registered z widened to the contrast's own Welch.
  The 80 % MDE at the measured intervals is about 4.2 pp.
- **What this says about the data, read narrowly:** human labels lose to P\*'s labels on the same human states
  (Δ_labels), and human states lose a little more to P\*'s own states (Δ_states, UNCLEAR). A human teacher of this
  slice is worse than our plateau policy as a cloning target, at this volume.

## Descriptive (no claim; each its own registration before it enters the view)

| row | value |
|---|---|
| Δ_data per panel, H / F / P72 | −8.75 [−13.16, −4.34] / **−8.80 [−12.25, −5.35]** / −4.30 [−8.87, +0.27] |
| Δ_labels per panel, H / F / P72 | −6.70 [−11.11, −2.29] / −3.05 [−6.45, +0.35] / −3.75 [−7.78, +0.28] |
| Δ_states per panel, H / F / P72 | −2.05 [−5.61, +1.51] / −5.75 [−9.23, −2.27] / −0.55 [−5.36, +4.26] |
| paired per seed (t, df 4), Δ_data / Δ_labels / Δ_states | −7.28 [−9.83, −4.74] / −4.50 [−8.02, −0.98] / −2.78 [−7.32, +1.75] |
| game length, terciles cut at 30 / 46 turns: Δ_data short / mid / long | −7.0 [−10.0, −4.1] / −10.2 [−13.2, −7.3] / −4.6 [−7.6, −1.6] (not concentrated in long games, A6.2) |
| the student-to-P\* gap (secondary, cross-engine: X51 on E0) | S-H −16.7, S-P −12.2, S-S −9.4 pp (every arm far below P\*, as the PREREG guessed) |
| yardstick: NLL on the S-S validation rows' P\* labels (5-seed means) | S-H 1.506, S-P 1.360, S-S 1.281 (S-P is closer to S-S: it learned its teacher, A5.8) |
| Δ_restriction's record | `results/restriction_record_decision.json` (full record `~/gen3ai_archive/x59s/read/restriction/record.json`) |
| FR-2 (reported only, A6.1) | NO-ARTEFACT, D = −0.149 nats [−0.163, −0.134] |
| floor references (units 0–3) | random 0.000, max-damage 0.046; the gate found every contrast informative |

**Predictions (§6, GUESS) against the read:** arm means 0.10–0.25 guessed, 0.32–0.39 read; Δ_labels −3 (−8 to +2)
guessed, −4.50 read; Δ_data −1 (−6 to +4) guessed, −7.28 read (outside the guessed range); Δ_states +2 guessed,
−2.78 read.

## Guards and identity

- VOID 3: one init per seed, 7,041 steps each, every student learned (`results/students.json`). VOID 5: the identity
  unit replayed X51's `pstar_home72_00` 20 / 20 on the read's engine (`results/identity.json`). VOID 6: the seeded
  repeat equals its original game for game and choice for choice (`results/gate_record.json`, `results/m4_checks.json`).
  VOID 7: no cell lost a unit; all 901 primary and 300 S-P0 Kakuna units counted.
- A5.9 m4 (the pin predates `e28318c8`): 0 peer faults in counted units (1 in an uncounted partial attempt, whose
  retry counted).
- **One engine:** every Kakuna unit, the floor references and the in-house games ran on
  `port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129`, at the play pin `89c2c2501` (tag `x59s-play-89c2c250`;
  `0534e90a6` + the Amendment 5 and 6 rule code). forum72's regime is the legacy Kakuna peer pack (before
  `b0441623e`).
- **Records:** the full `main.reads` record `~/gen3ai_archive/x59s/read/full/record.json` (COMPLETE; its decision
  without the 746 KB inputs block is [`results/full_record_decision.json`](results/full_record_decision.json)); the
  gate's [`results/gate_record.json`](results/gate_record.json); the in-house [`results/inhouse.json`](results/inhouse.json).

## What it cannot say

From-scratch students at 3 epochs are not a plateau fork (X60's question); the slice is the panel-distant, mostly
1300–1499, forfeit-heavy one, so nothing here speaks for ≥ 1500 play (the registered follow-up isolates rating at fixed
slice and volume) or for human data in general; S-S is P\* distillation (our plateau policy's own games and choices),
not the RL rollout data that trains P\*; each arm is one dataset, so dataset variance is outside the interval; the
read is conditional on the 216 panel teams.

## FINDINGS (standing rule 7)

1. **Throughput, not lane-hours, bound the read:** units played in about 6 min each but waited up to 30 min for a
   heavy slot; the read took 4 days at one, then two, slots.
2. **Three registration fixes were needed before any student game** (Amendments 4, 4b): the anchors config paths,
   the playing checkout loading the read's own rule (the pin moved to a side commit carrying it, tagged), and the engine
   (main's, not X51's E0).
3. **R1's trace extraction read only the trainee's seat**, so FR-2 needed an opponent-seat extraction; its G-ENC
   failed on 11 battles from `d8d674b99`'s silent volatile ends (drift toward correctness), handled by a registered
   exclusion (Amendment 4c). REVIEW G then showed FR-2's class cannot detect the signature it was meant to gate (A6.1).
4. **The in-house engine check read no rows** until the ledger's gzipped count rows were read (`8f1a141d9`).
5. **The gate's first `record` from the pin tree refused**: its repo-relative artefacts postdate the pin; every
   record was taken from main HEAD (an operator verb).
</content>
</invoke>
