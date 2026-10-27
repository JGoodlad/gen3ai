# Views audit, 2026-10-26: UNDERSTANDING, STATUS and the beat-Metamon design against the evidence

**What was checked.** Every quantitative claim in `designs/STATUS.md`, in `designs/endstate/design_beat_metamon_path.md`
(all sections) and in `designs/research_state/UNDERSTANDING.md` (the TL;DR, §1, §3.1, §4.4–§4.11, §4.x, §5, §6), for:
(a) the number, interval, n and verdict class against the cited ledger entry or record; (b) the cited commit exists and
lands that record; (c) a later result supersedes it; (d) the three documents contradict each other; (e) an
always-current claim is phrased as current but is history. Then the ledger's entries of 2026-10-19 → 10-26 for a
record, valid ids and registrations that precede their reads. The checks ran at `main` `6fca2c061`, CPU only, no games.
Three read-only sub-audits (UNDERSTANDING §3.1 + §4.x; UNDERSTANDING §4.4–§6; beat-Metamon §1–§8) fed the tables.

**Severity.** MAJOR: the claim changed a stated conclusion, a verdict or a headline number, or named the wrong thing as
current. MINOR: a wrong detail, a stale phrase, or a missing caveat. NIT: wording or a path.

**Totals.** 72 discrepancies: MAJOR 13, MINOR 47, NIT 12.
- 62 are fixed in place in the three views: M1–M13, m1–m41, n1–n8.
- 3 are in the ledger (L1–L3). L1 is corrected by an appended entry.
- 7 are not fixed; they are listed at the end.

No ledger number is wrong. One correction entry was appended to the ledger (2026-10-26 · CORRECTION · *THE VIEWS
AUDIT*).

**What checked out.** Every headline number matches its record:
- X51, X52, D1 looks 1–2, X26c, D3 look 1, PastaMittens, FR-1 / FR-3, the belief prior mix (λ = 0.25 and λ = 1), H3
  on calibration;
- R0-toy / R0-prod, A1 / G1, SPRT, X47, L1 and its replication, E1-0, the plateau step, the collapse battery, X55,
  X37a / X37b, TSR;
- the 75M-era search, critic and Foul Play figures.

Every cited commit exists. The ledger also passes its own checks:
- Every MEASUREMENT entry names a record directory that exists.
- Every registration commit is an ancestor of its read.

## The discrepancies that changed a stated conclusion (MAJOR)

| # | where | was | evidence | now |
|---|---|---|---|---|
| M1 | STATUS §1, goal 2b | "Best arm so far: D3 at 0.491 (look 1)" | the belief prior mix λ = 0.25 scored U 0.502 [0.473, 0.530] (ledger 2026-10-26, `f299a7243`); D1's final 0.492 [0.471, 0.513] (`eb99ab362`) | no lever's interval clears 0.50; highest points: prior mix 0.502, D1 0.492, D3 0.491, each against its own control, none detected |
| M2 | UNDERSTANDING TL;DR 15 | "Live now: `ai_v12_02_winprob_critic`", "the standing kill is G7" | the arm is long dead; the critic gate was retired 2026-10-25 | the bullet states what is current (the win-prob head IS the critic) and keeps the adoption read |
| M3 | UNDERSTANDING §1 | "it is what is running now"; ai_v14 "LIVE"; "The live ARM was ai_v12_02"; runs table "ai_v12_01 … the live arm"; no Rustboro era | `utils/era.py` (Rustboro from 2026-10-03); STATUS | ai_v14 CLOSED (pre-era); a Rustboro row added; the live-arm paragraph and row removed |
| M4 | UNDERSTANDING TL;DR 22 | whether production's machinery raises the ceiling "is R-T2's question … UNREAD" | X51: T\* @290M vs Kakuna −10.5 pp (home72), −7.3 pp (forum72) below P\*, both detected (ledger 2026-10-21) | at the plateaus the ordering reverses; R-T2's head-to-head itself stays unread |
| M5 | UNDERSTANDING §3.1, Foul Play | "the two anchors bracket the policy: a 200M search-free model below it" | the matched regime (same section; ledger 2026-09-14 · *THE METAMON BASELINE AT A MATCHED REGIME*): `SyntheticRLV2` 0.420 [0.328, 0.518], point on the losing side | they do not bracket it |
| M6 | UNDERSTANDING §4.x | the playoff arm "fails the determinization prefix gate on 72 of 73 decisions … must not be run" | ledger 2026-09-22 · *THE PLAYOFF ARM ACTS*: the P1 was wrong twice; zero prefix-gate failures at HEAD | the blocker is withdrawn; the arm acts, no strength claim |
| M7 | UNDERSTANDING §4.x | "it is the LABEL" (the leaf ceiling's bound) | ledger 2026-10-06 · *CRN LABEL REFIT*: a de-noised training label does not lift the head (−0.014 [−0.032, +0.002]); the bound is representation / coverage | scoped to the METRIC's label, with the refit read |
| M8 | UNDERSTANDING §6 | "Does the belief heads' off-pool overconfidence cost WIN RATE?" listed as open | λ = 0.25 +1.50 [−2.74, +5.74] and λ = 1 −1.50 [−5.47, +2.47], both NOT DETECTED (ledger 2026-10-26) | the row removed; the answer lives in §3.1 |
| M9 | beat-Metamon §5, D1 row | "+2 to +6 (GUESS)", rank 3 | D1 final UNDECIDED +2.10 [−0.92, +5.11], in-pool −1.61 (ledger 2026-10-24) | MEASURED, DONE |
| M10 | beat-Metamon §5, the arithmetic | "the plausible stack is D1 (+2–6) … X52 decides whether this family is the right one" | X52 UNDECIDED, D1 UNDECIDED, X26c FLAT, D3 +1.27, prior mix ±1.5 | every measured lever reads below the ≈ 4 pp bar; the open routes are D3 look 2, N1-P, X59-S |
| M11 | beat-Metamon §8, FINDING 6 | "Every expected-pp entry in §5 except search's is a GUESS … X52 is the first" | six Kakuna reads since | MEASURED where a read exists, GUESS elsewhere |
| M12 | beat-Metamon §7 | every trigger (X51 outcomes, X52 NO FAMILIARITY / NO TRANSFER) | all resolved: X51 read, X52 UNDECIDED | the live triggers: D3 look 2 GAIN, N1-P GAIN, all three null ⇒ the DRAFT fallback |
| M13 | beat-Metamon §6.5, X53's trigger | "X52 = TRANSFERS, and D1 reads NULL or GAIN-short" | neither fired | the trigger is N1-P's GAIN (as X53-N1) |

## MINOR and NIT, fixed

| # | where | was → now | evidence |
|---|---|---|---|
| m1 | STATUS goal 3 | "P\* is equal to Metamon's ensemble" → level on unfamiliar teams, ahead on home72 | PastaMittens home72 0.550 [0.501, 0.598] (`82f273eff`) |
| m2 | STATUS goal 3 | PILOT_c +0.123 without its checkpoint or record → "(X26 @161M)", `8b43eb44a` added | ledger 2026-10-18 · *X37a STAGE 1* |
| m3 | STATUS terms, §4 | "ours has 2.03M", "about 70 times smaller" → 2.61M (2.03M in the extractor), about 55 times | `ARCHITECTURE.md`: the production learner is 2,609,183 parameters |
| m4 | STATUS goal 5 | "Graded by a real bot, the gains shrink" → marked descriptive; the registered read is UNDECIDED | E1-0 (`015f00e20`) |
| m5 | STATUS §3, L1 row | "REPLICATES thinly, only at 25M" → pooled over both anchors; the 25M split is post hoc | `f756019d7` |
| m6 | STATUS §3, D3 row; beat-Metamon §0 and §6.4; UNDERSTANDING §3.1 | the engine-mix caveat added (below, L1) | REVIEW F F-M2; `d8d674b99`, `f9978586b`, `2cc864f2f` |
| m7 | beat-Metamon status line | "D1 proceeds as staged", D3 "look 1, no read yet", N1-P "not launched" → rewritten to the current state | D3 look 1 READ (`58287c525`); `rb_x35_n1p_s3501–3` trained (`77315ac46`, `b054fd8e6`, `716eac7c3`) |
| m8 | beat-Metamon §0 items 5, 6 | N1-P "BUILT"; X53 "≈ 21 GPU-h, if D1 reads NULL…" → TRAINED, being read; X53 conditional on N1-P as X53-N1 (≈ 39 GPU-h), a REVISED DRAFT gated on a zero-GPU seed-spread read (`7574743d3`) | `strategy_review_2026-10-26/REVIEW.md` |
| m9 | beat-Metamon §0 | the belief reads absent → a bullet: the belief channel moves no Kakuna games detectably | ledger 2026-10-26 (both λ) |
| m10 | beat-Metamon §9, panel-validity row | H's 4.2 % "matches a random hl team's" → below it (8.9 %) | `goal_b_panel_validity_2026-10-26/README.md` (the 4.2 % match is forum216b's) |
| m11 | beat-Metamon §5, D3 row | GUESS, "runs on GO", ≈ 12.1 GPU-h → MEASURED look 1, 12.54 GPU-h, look 2 queued | ledger 2026-10-26 · *X41 D3 LOOK 1* |
| m12 | beat-Metamon §5, N1-P row; §6.5b heading; §6.8 | "BUILT; the orchestrator's GO" → TRAINED, look 1 being read | as m7 |
| m13 | beat-Metamon §5, X35 breakers row | "runs anyway; report U per final" → A1 NULL, SPRT NULL (FINAL), LR-F queued | ledger 2026-10-22, 2026-10-24 |
| m14 | beat-Metamon §5, capacity row | "the toy X47 reads first" → read UNDECIDED, Δ̂ ≤ 0, FINAL | ledger 2026-10-21 · *X47 LOOK 1* |
| m15 | beat-Metamon §5, human-replay row | "FR-1, FR-3, playing"; "X59-S built … then the GPU window" → READ (0.666 / 0.631 / 0.790); students training | ledger 2026-10-25 / 26 |
| m16 | beat-Metamon §5, search row | "stage 1c registered" → READ: nearer, not paying | ledger 2026-10-21 · *Stage 1c's PHASE B* |
| m17 | beat-Metamon §5 | no plain-training row, no belief row → both added, MEASURED | X26c; the prior mix |
| m18 | beat-Metamon §6.3 | heading "as staged, with a scheduling ask"; "neither now — the X26 continuation fork runs instead" → FINAL READ UNDECIDED; X26c ran FLAT, D3 ran on a fresh GO | ledger 2026-10-24, 10-25 |
| m19 | beat-Metamon §6.4 | "Look 2 … is licensed" → queued next on the GPU (≈ 8.1 GPU-h) | STATUS §2 |
| m20 | beat-Metamon §6.5 | X53-N1 "DRAFT" with no review state → the concurrent revision's text kept on rebase (REVISED DRAFT after review `6fca2c06`) | `7574743d3` |
| m21 | beat-Metamon §6.6 | "have now steered four reads and will steer D1" → every goal-(b) read so far | — |
| m22 | beat-Metamon §6.7 | "Every X35 FINAL reports U against Kakuna" → "is to report"; A1 and SPRT did not (new FINDING 12) | their records hold no Kakuna U |
| m23 | beat-Metamon §6.8 | X52 CPU ≈ 11.8 → 8.4; N1-P look 2 CPU ≈ 74 → ≈ 29 (AM-N1P-1); D3 and N1-P rows to their state | X52 read README; `9b9f72474` |
| m24 | beat-Metamon §8 | FINDINGS 3, 4, 11 still read as open → RESOLVED / RULED | — |
| m25 | beat-Metamon §1, goal (b) row | no panel-validity caveat → "without H 0.481 [0.447, 0.516]; the panels stand" | `4f555e577` |
| m26 | beat-Metamon §3 | "every Kakuna read" lacked D1, D3 and the prior mix → U rows added, and a "no lever moves U" bullet | the three records |
| m27 | UNDERSTANDING TL;DR | no current-state headline → bullet 0 (P\*, the goals, the deficit) and a note that bullets 1–18 are pre-era | STATUS; X51; X26c |
| m28 | UNDERSTANDING §1, runs table | `production` named `ai_v9_21_gen17_pfspoff_0820` → `rb_es_end2_s2005`; gen-17 is `production_pre_endstate` | `designs/baselines.json` |
| m29 | UNDERSTANDING §3.1 | "(held-out teams: LOST, below)" → not met; LOST at 161M, level at the plateau | X51 |
| m30 | UNDERSTANDING §3.1 | "one checkpoint of a live run", "161M of a 500M cap" → one checkpoint at 161M, re-read at the plateau | X26 ended; X51 |
| m31 | UNDERSTANDING §3.1 | "Zero protocol failures", the poke-env integration block, "`production` … does not load at HEAD" → the matched read's hazards; the poke-env client deleted (P6); the dead registry sentence removed; the matched read cited | ledger 2026-09-14 (matched regime); `EXTERNAL_ANCHORS_SOP.md` |
| m32 | UNDERSTANDING §3.1, D3 | "2,876 of our pool teams" → 2,876 set-mutated variants of our 719 pool teams | `x41_d3_setmut_2026-10-25/PREREG.md` |
| m33 | UNDERSTANDING §3.1, X52 | "D1 proceeds as staged" → removed | D1 FINAL 2026-10-24 |
| m34 | UNDERSTANDING §3.1 | the panel-validity read absent → a paragraph after X51 | ledger 2026-10-26 · *GOAL (b)'S PANELS* |
| m35 | UNDERSTANDING §3.1, X55 | two hypotheses stated as open → both since read (A1 NULL; R0-prod +11.1 pp SATURATED) | ledger 2026-10-22 |
| m36 | UNDERSTANDING §3.1, N1-P | "[BUILT · no measurement yet]" → TRAINED, look 1 being read | as m7 |
| m37 | UNDERSTANDING §4.4 | LR-F "a batch-2 CANDIDATE, toy first" → REGISTERED (toy and production), production queued | X35 `STAGED_PLAN.md` §4.1.2 |
| m38 | UNDERSTANDING §4.7, the belief bullet | "p ≈ 0.01 … against the ≈ 0.2–0.3" (contradicts the red-team bullet) → 0.001 / 0.014 on the 8-rollout leaf, its 1/8 resolution named | maturity README; ledger 2026-10-25 |
| m39 | UNDERSTANDING §4.7, team identification | "against bots and own snapshots alike" → marked SELF-REFEREE, with E1-0's real-bot read | ledger 2026-10-25 · *E1-0* |
| m40 | UNDERSTANDING §4.8 | X59-S "built and pre-registered (not trained)" → TRAINING | STATUS §2 |
| m41 | UNDERSTANDING §4.8 | the X50 reads' and the HOM's pre-fix intent masks unstated → stated | ledger 2026-10-26 · FIX (D-1) |
| n1 | UNDERSTANDING §4.10 | "That matches the X55 LG-0 prediction" → the point is inside it, the modal class was wrong | ledger 2026-10-20 · *X55 LG-0* |
| n2 | UNDERSTANDING §5 | the teacher ceiling "invariant to budget and to target strength"; the budget law "dissolved" → scoped to meter teams | ledger 2026-08-31 · *EXPLOITABILITY DECOMPOSED* |
| n3 | UNDERSTANDING §6 | the R0 row "Still to come …" → ANSWERED for the policy; search cells DEFERRED | R0-prod |
| n4 | UNDERSTANDING §6 | three ai_v12 rows ("live arm", "in-flight") → removed | the gate retired 2026-10-25 |
| n5 | UNDERSTANDING §6 | eight pre-era rows with no backlog entry → marked dormant pre-era directions | `EXPERIMENT_BACKLOG.md` |
| n6 | UNDERSTANDING §3.1, X26c | "the D3-revisit trigger … does not fire", then D3 runs → D3 ran on a separate GO | D3 PREREG |
| n7 | UNDERSTANDING §4.x | "the only read that can run today" (the P1 withdrawn) → removed | M6 |
| n8 | beat-Metamon §6.8 totals | "≈ 88" unlabelled → labelled (D3 at look 1; N1-P outside) | — |

Severity in the totals: n2–n5 are MINOR; n1 and n6–n8 are NIT. MINOR 47 = m1–m41, n2–n5, L1 and the survey
FINDING. NIT 12 = n1, n6–n8, L2, L3 and the six wording items not fixed.

## The ledger (2026-10-19 → 10-26)

| # | finding | severity | action |
|---|---|---|---|
| L1 | The D3 look-1 entry (`58287c525`) reuses C0's rows played at `5e389ddac`, before the obs / label fix `d8d674b99` and the engine fix `f9978586b`; D3 played at `2cc864f2f`, which carries both. The regime id cannot see the engine; the guard re-played one cell (140 games) | MINOR | CORRECTION entry appended; the caveat now in all three views |
| L2 | The ten 2026-10-26 RUST SIM entries use `##`, not `### YYYY-MM-DD · TAG · title`, and cite no record path (evidence: commit bodies and tests) | NIT | noted in the correction entry (the index lists them) |
| L3 | "H" names both D1's pooled statistic and the `heldout72_hpfix` panel in the D1 / D3 entries | NIT | noted in the correction entry |
| — | Three backticked hex strings are content hashes, not commits (`76e29a01`, `ac8e88e7`, `cef0960e`) | — | no action: correct as written |

## Not fixed (FINDINGS)

1. `research_state/literature/sota_survey_2026-10-26.md` (a dated survey, not an always-current view) says "2.03M
   parameters" and "~70 times smaller". The production learner is 2.61M (`ARCHITECTURE.md`).
2. UNDERSTANDING §3.1's cross-era path NIT (`arch_transfer_2026-09-05/cross_era_head_to_head/`), the Foul Play axes date
   (09-16 directory, 09-17 entry), §4.5's floor provenance and §4.6's "legacy goes" NITs, and FR-1's "H18 favours P\*"
   caveat. Each is wording only. "H18" is an undefined id in the views, so quoting it would not help a reader.
3. beat-Metamon §4 and §8 FINDING 2 say D1's corpus is 1,982 teams. D1 trained on 1,981 after AM-D1-FIX. FINDING 2 is
   history of the carve, so it is left.
4. Not verified by the sub-audits:
   - X52's "≈ 3,300 pilots a team";
   - X55's "164–268M" bound;
   - "naive search loses 74–80 %" (the ledger says three games in four);
   - Kakuna's per-game CPU figures;
   - the GXE preset values;
   - FINDING 7's bootstrap widths;
   - whether X35 A2 / A3 have been read.
5. Most of UNDERSTANDING's TL;DR (bullets 1–18) and §2 hold pre-era findings, written in the present tense. A note
   now marks them as pre-era directions. A full rewrite into the Rustboro era is a larger unit than this audit.

## Evidence

- Ledger entries: the landing commits by `git blame` of each heading line; ids by `git cat-file -e`; registrations by
  `git merge-base --is-ancestor <prereg> <read>`; records by path existence.
- Running state (STATUS §2): the N1-P look-1 read supervisors (`main.reads run --state …/x61p_n1p_read/look1`) and
  the X59-S student window, live at the audit.
