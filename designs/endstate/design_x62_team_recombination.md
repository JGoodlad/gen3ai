# X62 — team recombination: train P\*-forks on new sixes made of familiar sets

**Status: DRAFT (2026-10-26), explicit-only, GATED ON H-CTX. Nothing is built, registered or launched.** It designs
the training lever that H-CTX implies, if the H-CTX read supports it. H-CTX comes from the piloting-gap forensics
([`piloting_gap_forensics_2026-10-26/README.md`](../research_state/measurements/piloting_gap_forensics_2026-10-26/README.md),
`c758fcb82`) and is registered as the rc72 read
([`hctx_rc72_2026-10-26/PREREG.md`](../research_state/measurements/hctx_rc72_2026-10-26/PREREG.md), Amendment 1 at
`caec2bb2c`, rule `premium_recovery`). Goal (b), its meter and every read cited here belong to
[`design_beat_metamon_path.md`](design_beat_metamon_path.md). The ranking rules are
[`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md)'s: P(GAIN) per full cost, CPU
included, and one lever per A/B. The two scripts behind this doc's new numbers are in
[`x62_design_2026-10-26/`](../research_state/measurements/x62_design_2026-10-26/): `set_familiarity.py` and
`power_pgain.py`, each with its `.log`. Both are CPU-only and run in under a minute.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau); C0 = X35's plain-continuation control (P\* + 20M, no lever); the POOL = the 719 teams
P\* trained on; home72 = 72 pool teams; F = `forum72`; H = `heldout72_hpfix`; P72 = the procedural panel; U = H, F and
P72 at equal weight (goal (b)'s meter); S = in-house strength against the frozen reference population on home72; a
SET = one mon's species plus its sorted moves, Hidden Power typed (`gen3_team_overlap_v1`, `main.anchors.team_overlap`);
**RECOMBINED-LIKE** = a team with ≥ 5 of its six sets in the pool that shares ≤ 2 sets with any ONE pool team (familiar
sets in a new six); σ_s = the between-seed SD of a fork final's true F score; **lane-h** = one wall-hour of the Kakuna
anchor lane at `--concurrency 1` on rule 5b's agent cores, at the MEASURED 160–230 games an hour (strategy doc's
definition); GUESS = an unmeasured effect; ESTIMATE = a cost from measured rates; MY JUDGEMENT = an inference no read
states.

---

## 0. One screen

- **The lever.** Fork P\* for 20M steps, with half of each side's teams drawn from **`x62_recomb_v1`**: 12,000 NEW
  teams. Every mon block in them is copied verbatim from a pool team, and no team shares more than 2 sets with any pool
  team. The network keeps every set it knows and loses every six it has memorised, so the only new thing to learn is
  how to play a team. It is the complement of D3 (new sets inside old teams).
- **N1-P is NOT this lever** (measured here, §1.2). Only **28 %** of N1-P's mon slots carry a pool set, and 73 of its
  12,000 teams (0.6 %) are recombined-like. N1-P changes set novelty, coherence and context all at once.
  **The closest existing arm is D1.** 78 % of D1's corpus slots are pool sets, and **624 of its 1,981 teams (31 %)
  are recombined-like**. D1 has therefore already delivered a partial, human-sourced dose of X62 (≈ 16 % of draws), and
  read UNDECIDED. X62 is D1 purified and scaled. It adds information beyond N1-P and D1 in three ways:
  - it is the one corpus that adds no set to any panel's coverage, so its F, H, P72 and U are all clean;
  - it is the one arm that isolates team context from set novelty;
  - when N1-P reads NULL, it separates "new contexts do not help at the plateau" (plasticity) from "N1-P's contexts
    were the wrong kind" (incoherent, unfamiliar sets).
- **The gate (§6).**
  - **H-CTX SUPPORTED** ⇒ build the corpus and register X62, unless N1-P has read GAIN. In that case X62 is shelved as
    a standalone lever and its corpus is offered to X53-N1.
  - **KILLED** ⇒ shelve. The premium is set-level, and X62 adds no set.
  - **UNDECIDED** ⇒ build the corpus and hold. The fork runs as a D1-rule screen only if N1-P reads NULL or LOSS.
  - **INCONCLUSIVE or VOID** ⇒ hold, and extend H-CTX's controls first.
  - H-MID never blocks X62. It read CRITIC-MISCALIBRATED-ON-U (`f200258f`), so M3 is a registered secondary. Whether
    search on V reopens on U is S-U's question ([`design_search_u_headroom.md`](design_search_u_headroom.md)).
- **The read (§5).** Goal (b) on **F** against Kakuna, against C0 on ONE engine. Two read options:
  - **The PR rule** (17,400 F games an arm, 5 seeds): P(GAIN) **0.80 / 0.67 / 0.44 at +1.5 pp** for σ_s = 0 / 0.5 /
    1 pp, and 1.00 / 1.00 / 0.95 at +3 pp. Its false GAIN is 0.025 with a correct widening, and **0.10 at σ_s = 1 pp
    if the interval ignores seed noise**.
  - **D1's rule**, as a screen: 0.37 at +1.5 pp and 0.89 at +3 pp (σ_s 0); false GAIN 0.026 to 0.047.
- **Cost (§7).**
  - **GPU:** **20.5 GPU-h** for 5 seeds (12.3 GPU-h at look 1).
  - **CPU:** **88–127 lane-h** by D1's rule (C0 re-played on the read's engine), or **126–181 lane-h** by the PR rule
    if PR-1 has already banked C0 on that engine (**187–269** if not).
  - **Build:** ≈ 2 agent-days, plus ≈ 1–3 CPU-h for the corpus build and soak.
- **The odds.** P(GAIN) given H-CTX SUPPORTED and N1-P not GAIN is **0.52 by D1's rule and 0.65 by the PR rule**
  (σ_s 0; GUESS +2.0 pp, 80 % (−1.0, +5.0)). That is ≈ 0.09–0.10 per resource-day, level with the strategy's breadth
  stack (0.12) and H-TEAMS (0.09).

---

## 1. Why this lever, and why it is not one we already have

### 1.1 The evidence that points here

- **The premium goes with the TEAM.** P\*'s mons out-kill Kakuna's by +0.081 KOs per entry on home72 and −0.028 on U.
  Restricted to sets in our pool on both sides, the drop is **+0.093 [+0.004, +0.176]**: a familiar set loses its edge
  inside an unfamiliar six. Set novelty costs about a third as much again (forensics §4; DESCRIPTIVE, post hoc).
- **The gap forms in the mid-game, from level positions.** From level material at turn 10, P\* wins 0.609 on home72
  and 0.501 on U (forensics §1). That is where a team plan matters (which mon to preserve, the win condition).
- **The policy is compositional at the set level.** Re-drawing one familiar bench set moves P\* only 1.19×
  [1.07, 1.33] as much as re-drawing an unfamiliar one, so N2 was dropped
  ([`n2_identity_test_2026-10-26/README.md`](../research_state/measurements/n2_identity_test_2026-10-26/README.md),
  `fd4160d76`). That read also leaves open "plan memorisation keyed on the battle's course or on revealed mons", which
  is H-CTX's mechanism.
- **The lever H-CTX names.** If P\* memorised its 719 sixes, the cure is many sixes built from parts it already knows,
  so that training pays only for the team-level skill. The forensics wrote this down as "recombine pool sets into new
  teams, the complement of D3".

### 1.2 Where each corpus sits on set novelty and team context (measured, `set_familiarity.log`)

| corpus / panel | teams | slots carrying a pool set | max sets shared with ONE pool team | recombined-like | what it varies |
|---|---:|---:|---|---:|---|
| `d3_setmut_v1` (D3) | 2,876 | 0.813 | 4: 1,381 · 5: 1,495 | 0 | **sets**, inside familiar sixes |
| `w_forum_v1@ABC` (D1) | 1,981 | 0.776 | ≤ 2: 1,301 · 3: 579 · ≥ 4: 101 | **624** (≥ 4 pool sets: 997) | contexts, mostly familiar sets, human-built; the forum population |
| `n1_proc_v1` (N1-P) | 12,000 | **0.280** | ≤ 2: 11,960 · 3: 40 | **73** | contexts **and** sets (72 % new), procedural |
| **`x62_recomb_v1` (X62, proposed)** | 12,000 | **1.000** | **≤ 2 by construction** | **12,000** | **contexts only** |
| home72 | 72 | 1.000 | 6 (pool teams) | 0 | — |
| rc72 (H-CTX's panel) | 72 | 1.000 | 1: 1 · 2: 71 | 72 | — |
| F | 72 | 0.701 | 1: 14 · 2: 42 · 3: 16 | 19 | — |
| H | 72 | 0.380 | ≤ 2: 71 · 3: 1 | 3 | — |
| P72 | 72 | 0.308 | ≤ 2: 72 | 0 | — |

The pool holds 658 distinct sets of 69 species, and 54 of those species have ≥ 2 sets. Every corpus team's species
are 97–100 % pool species.

**What this says, honestly.**
- **N1-P is not X62.** Most of N1-P's slots are sets P\* never piloted, drawn independently of each other
  (incoherent by construction, its PREREG FINDING 5). A NULL from N1-P cannot tell "new contexts do not help" from
  "new sets plus incoherence cost what contexts buy".
- **D1 is partly X62 already.** Under H-CTX, D1's U +2.10 [−0.92, +5.11] (UNDECIDED, final) came from a corpus whose
  draws were ≈ 0.5 × 31 % ≈ 16 % recombined-like, and ≈ 25 % at the looser (≥ 4, ≤ 2). Its F is hazarded: F is D1's
  own forum population. X62 is the version with no set novelty, no shared population with any panel, and 6× the team
  count.
- **F is itself 26 % recombined-like** (19 of 72 teams). Of all the unfamiliar panels, F is where a context lever
  should show first, and it is goal (b)'s clean panel for the whole breadth family.

### 1.3 The counterweight: the per-panel gradient fits the set-level account too

Across X51's four panels, P\*'s score against Kakuna falls with the share of pool sets:

| panel | share of pool sets | P\* vs Kakuna |
|---|---:|---:|
| home72 | 1.00 | 0.598 |
| F | 0.70 | 0.530 |
| H | 0.38 | 0.497 |
| P72 | 0.31 | 0.433 |

Each score is ±5 pp (n 400; X51 README). A purely set-level account predicts exactly this ordering. So does H-CTX,
because familiar contexts and familiar sets move together across these panels. Only rc72 separates the two, which is
why X62 waits for it.

---

## 2. The corpus `x62_recomb_v1` (a builder sketch; NOT BUILT)

A builder beside `build_n1_proc` in `utils.team_corpora` (proposed name `build_x62_recomb`), deterministic with
`--check` as its siblings are. It generalises rc72's `build_rc72.py` from a 72-team panel to a 12,000-team corpus.
**It reads the POOL for every set, the pool's own species counts for the profile, and Smogon only for teammate
coherence.** It ships no prior: it is a training corpus, like `n1_proc_v1`.

**The draw of candidate i** (one `random.Random` per (stream, i), as `build_n1_proc`):
1. **Species, usage-matched to what P\* trained on.**
   - Slot 1 is drawn with weight c_pool(s), the species' occurrence count over the pool's 4,314 mons.
   - Each later slot is drawn among species not yet on the team (Species Clause by construction), with weight
     c_pool(s) × geomean over the team's species t of lift(t, s). Here lift(t, s) = P(s | t) / share(s) from Smogon's
     teammate and usage tables, clipped to [¼, 4]. These are rc72's rules, with the pool's counts in place of home72's.
   - So per-species exposure stays near the pool's: the arm changes contexts, not which species P\* pilots. That
     deliberately leaves H-RARE (rare-species piloting) out of this lever: one lever per A/B.
   - The manifest reports the species profile's total variation distance from the pool and from Smogon usage.
2. **The set: one pool OCCURRENCE of the drawn species**, uniform over occurrences, so a set is drawn as often as the
   pool repeats it. Item, ability, EVs, nature, IVs and moves are copied verbatim. **The nickname line is dropped**: the
   network does not observe it (N2 S3), and Showdown's Nickname Clause would refuse two copied blocks with the same
   nickname. Because the item and spread are verbatim, the sets are pool sets at the strict tier too, not just the mon
   tier.
3. **Distance refusals at draw time** (the slot is redrawn, up to rc72's `SLOT_TRIES`; then the candidate fails):
   - **≤ 2 shared sets with ONE pool team.** This is the lever's definition, rc72's `MAX_SHARED_TRAIN`.
   - **< 3 shared with ONE team** of every held-out panel (`heldout72`, H, F, P72, the sealed `forum216b`), of
     `n1p72`, and of **rc72**. This is REVIEW A's revealed-fit radius, as in N1-P's and rc72's builds. rc72 is
     refused because it must stay H-CTX's and X62's held-out-context panel: its sets are all in the corpus (as they
     are all in the pool), but none of its sixes may be.
   - **≤ 4 shared with any corpus team already accepted**, and no duplicate six. That keeps near-copies from spending
     dose. Whether 12,000 teams fit under this cap is UNVERIFIED (FINDING 3). The fallback is ≤ 5, recorded in the
     manifest.
4. **Whole-team checks** (a failure moves to the next candidate; the manifest counts each rule's refusals):
   - the pinned `TeamValidator('gen3ou')` (`ladder_corpus_filter.js`);
   - the pack oracle (level 100, `build_w_forum.level_defects`);
   - every Hidden Power playing its named type at 70 once validated (rc72's `hp_set_check.js` check);
   - the engine scan (`scan_move_probe`, THROW on an unplayable id);
   - `check_label_encodable`.

**Legality, verified at source.** Showdown's `[Gen 3] OU` entry is `Standard` + One Boost Passer Clause + Accuracy
Trap Clause + Freeze Clause Mod + Speed Pass Clause, and the banlist is all species, ability, move and item bans
(`deps/pokemon-showdown`, its config formats file). **There is no Item Clause in gen3 OU**, so two mons may hold the
same item. A pool block passes every SET-level rule already, because it passed inside its pool team. Recombination can
break only the TEAM-level rules, which the validator runs:
- Species Clause: enforced at draw time;
- Nickname Clause: void, because nicknames are dropped;
- **One Boost Passer Clause**: at most one Baton Passer that can boost. Two pool Baton Passers can collide, so this is
  the one refusal we expect.

Sleep and Freeze Clause are battle mods, not team rules.

**Size, tiers and refresh.**
- **12,000 teams, N1-P's count**, so X62 − N1-P is a corpus contrast at one team count and one share. Prefix tiers
  K1 ⊂ K3 ⊂ ALL support a later dose read.
- **A static corpus, not re-drawn.** At share 0.5 over 20M (≈ 570k games; D1 doc §4.1's rule), each corpus team is
  piloted ≈ 24 times per side, the same as N1-P. Memorising one team took thousands of pilots: X52 practised 36 teams
  for 5M steps, ≈ 4,000 pilots each, for +3.2 pp [−0.2, +6.8]. So at 24 pilots a team, a static corpus is already
  effectively fresh, and a re-draw every N steps would change nothing measurable.
- What memorisation still acts on is **the pool's own 719 teams**: half the draws, ≈ 395 pilots each per 20M. No
  refresh of the corpus touches that. The variant that does is share 1.0 on the trainee side, which is a second lever
  (§8, item 6).
- Re-drawing matters only for a from-scratch run at the X53 scale (300M steps ⇒ ≈ 360 pilots per team at 12k). If
  such a run is ever registered, the answer is a larger static build (the builder takes N), not an in-loop generator.
- **The startup cost is measured, not guessed.** N1-P's 12,000-team table had a team phase of 18.5 s
  (`startup_team_phase_at_pin.txt`) and trained at **1,433 steps/s, 4.0 GPU-h** for its 20M (dose record
  `x35_breakers_stage_2026-10-17/dose/rb_x35_n1p_s3503.json`), level with C0's 1,384–1,424 steps/s.

**The soak** (`utils.team_corpora.soak`, `gen3_team_corpus_soak_v2`): 120,000 battles, N1-P's size. D3's re-soak of
24,240 battles took 382.5 s at nice 19 on 3 cores, so this one takes ≈ 30–35 min (ESTIMATE).

**Separation, restated.**
- X62 adds no set to any panel's coverage, so the M4 hazard (N1-P PREREG §8) does not apply. Its U is as readable as
  C0's.
- Every held-out panel and rc72 are refused at the radius-3 fit.
- The startup guard at the pin (`team_mix_guard`, radius 4) passes by construction.

---

## 3. The arm and its control

| item | value |
|---|---|
| start | P\* (`checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`), as C0 |
| **the ONE lever** | `--team-mix x62_recomb_v1@ALL:0.5 --team-mix-sides both`: N1-P's share and sides, so X62 − N1-P and X62 − D1 are corpus contrasts at one share (descriptive) |
| everything else | C0's argv token for token (kit arm `x62`, proposed, with a kit test of the `test_n1p_is_c0_plus_…` kind): `--behaviour-check warn`, +20M to 300,000,170, the recipe inherited from X26's record |
| **pin** | the production pin `f4b25238` + ONE commit holding only the corpus's three files and its one `CORPORA` line (AM-N1P's pattern, `batch.N1P_PIN`). C0's code byte for byte, so the 10-26 GIGO fixes are absent from both arms (N1-P PREREG §2's argument) |
| "the then-current recipe" | only if C0 has been re-run at a newer pin for some other reason. Otherwise moving the recipe costs a new C0 (5 × ≈ 4.1 = 20.5 GPU-h) and buys no information about this lever. Not recommended |
| seeds | 3501–3503 at look 1, plus 3504–3505 (D1's rule, on CONTINUE); all five at once under the PR rule |
| **control** | **C0 s3501–s3505**, banked runs at `f4b25238` (each K9(b) PASS): 0 GPU-h. **Their Kakuna games are played on the X62 read's engine** (the engine rule, `20875e962`, `designs/tools/reads.md` §4): either re-played, or imported sha-checked from a read whose play path is byte-identical (PR-1's C0 if it ran at that engine). Never D1's or X51's banked rows from another engine |

---

## 4. How it differs from D3, D1 and N1-P, and what N1-P's read implies

| | D3 | D1 | N1-P | **X62** |
|---|---|---|---|---|
| new sets per team | 1–2 (by construction) | ≈ 1.3 (22 % of slots) | ≈ 4.3 (72 %) | **0** |
| new six (≤ 2 shared with any pool team) | never (≥ 4 kept) | 66 % of teams | 99.7 % | **100 %** |
| coherence | a pool core | human-built | procedural, incoherent | pool sets, teammate-lift species |
| teams | 2,876 | 1,981 | 12,000 | 12,000 |
| share | 0.3 | 0.5 | 0.5 | 0.5 |
| panel hazard | none | F is its population | H, P72 (62 % / 56.5 % of full sets) | **none** |
| the H-CTX part it trains | set novelty (≈ ¼ of the premium) | mostly context | context + sets | **context only** |

**N1-P look 1 (in flight), and what each outcome implies for X62:**

| N1-P outcome | implication for X62 |
|---|---|
| **GAIN** (F, ⇒ X53-N1 GO) | New contexts at 10⁴ move F even with mostly new, incoherent sets. X62's only extra information would be the corpus choice for X53. A ±3 pp read cannot separate two positive arms (X62 − N1-P needs ≈ 4× the games). **X62 is REDUNDANT as a standalone plateau lever.** Its corpus is offered to X53-N1 as a later corpus arm, or to the breadth stack |
| **NULL** (F inside ±3 pp) | NOT a kill of X62. N1-P's NULL is confounded by set novelty and incoherence (its FINDING 5) and by plasticity. X62 is the arm that splits them. X62 NULL too ⇒ plasticity is the live explanation, and the route goes to N5-S, or X53 behind its σ gate. X62 GAIN ⇒ coherent, familiar-set contexts are the form that works at the plateau. **X62 adds the most information here.** Its GUESS falls to +1.5 (−1.5, +4.5), because a plateau that ignored 12,000 new contexts makes plasticity likelier |
| **UNDECIDED, Δ̂ > 0** (CONTINUE) | wait for N1-P's look 2 (≈ 8.2 GPU-h + its read) |
| **LOSS** | incoherence (or set novelty) costs. X62, which has neither, rises in rank, as in the NULL row |
| **UNDECIDED, final** | as NULL |

**D3 (look 2 queued).** D3 is the other half of the same decomposition. Under H-CTX its ceiling is ≈ a quarter of the
premium (forensics, "How these bear"). D3 and X62 are complements, not rivals: a D3 GAIN and an X62 GAIN would sum
under H-CTX, which is the case for a later stack.

---

## 5. The read

**The primary: goal (b) on F against Kakuna.** Kakuna checkpoint 34, greedy against greedy, CPU, role-balanced 20-game
units, F's committed config, every unit on ONE engine with each arm's C0 at that engine.
- **Why F rather than U.** X62's U is unhazarded (§2), but F is where a context lever should show (70 % pool sets,
  26 % recombined-like) and it is the breadth family's shared panel (D1, D3, N1-P, PR-1). U is REPORTED, and goal
  (b)'s own number may be read from it with no coverage caveat.

**The rule: two options, chosen at registration by one fact, whether C0 is banked on the read's engine.**

| | **PR rule** (strategy §3.2) | **D1's rule** (N1-P's read, F alone) |
|---|---|---|
| games | 17,400 F per arm, 5 seeds; C0 topped up to 17,400 | 1,200 F per seed final; looks of 3 then 5 seeds |
| when | PR-1 has run and banked ≈ 14,000 C0 F games on an engine X62 can still play on | otherwise: a screen |
| P(GAIN) at +1.5 pp, σ_s 0 / 0.5 / 1 | **0.80 / 0.67 / 0.44** (widened correctly) | 0.37 / — / 0.31 |
| at +3 pp | 1.00 / 1.00 / 0.95 | 0.89 / — / 0.77 |
| false GAIN (true 0) | **0.025** if the interval carries σ_s; **0.046 / 0.10** at σ_s 0.5 / 1 pp if it does not (binomial SE only) | 0.026 (σ_s 0), 0.047 (σ_s 1) |
| P(GAIN) given H-CTX SUPPORTED, N1-P not GAIN (GUESS +2.0, 80 % (−1.0, +5.0)) | **0.65 / 0.56** (σ_s 0 / 1) | **0.52 / 0.49** |

(`power_pgain.log`. The PR rule's figures are closed form with AM-5 not modelled. The strategy doc's ≈ 0.74 at +1.5
includes its own allowance. D1's rule is the Monte Carlo of `main.x35.d1.d1_primary` in `pgain_rows.log`.) **The
honest false-GAIN line is the PR rule's unwidened one.** σ_s ≤ 0.5 pp is unestablished (strategy FINDING 5), so the
PR-rule registration must carry AM-5's √φ̂ widening, or its false GAIN is up to 0.10.

**The co-primary P** (D1's, verbatim): S against the frozen reference population on home72, Welch 97.5 %,
NON-INFERIOR iff LB > −2.0 pp. C0 is re-played, not imported. X62 halves each pool team's exact draws, as D1 and N1-P
do. D1 cost −1.61 pp in-house. X62 keeps every set's exposure, so it is predicted to cost less (§5, predictions).

**REPORTED, never decided** (mechanism reads; each names what it would show under H-CTX):
- **M1, rc72 recovery:** X62 − C0 on rc72, 20 units per final. rc72 is held out by six (§2), so this is Procgen's
  held-out-level read of the corpus's own generator. If X62 lifts rc72 but not F, it learned "recombined-team play",
  not human team play.
- **M2, the seen-set per-mon KO edge on F** (the forensics' statistic, from the captures; minutes of CPU). H-CTX
  predicts X62 raises it toward home72's +0.081.
- **M3, H-MID's Δbias** on X62's and C0's F and home72 captures, re-encoded and read through H-MID's own scripts
  (forward passes, CPU, ≈ a few CPU-h; H-MID PREREG §3's fidelity gate applies). H-MID read CRITIC-MISCALIBRATED-ON-U
  (`f200258f`), so M3 is a registered SECONDARY (§6).
- **M4, H-TGT's switch-punish DiD** on F (forensics §2): X62 trains both sides, so it should move toward 0 if
  opponent-team breadth carries switch anticipation.
- **H, P72, home72 against Kakuna** at 7 units per final (D1's REPORTED group), and **U**. home72 is goal (a)'s
  guard: X62 must not cost goal (a) silently.

**Predictions (GUESS; conditional on H-CTX SUPPORTED and N1-P not GAIN).**

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F (primary)** | **+2.0 pp** | (−1.0, +5.0) |
| Δ_rc72 (M1) | +4.0 | (0.0, +8.0) |
| Δ_H / Δ_P72 | +1.0 / +0.5 | (−2.0, +4.0) / (−2.5, +3.5) |
| Δ_home72 vs Kakuna | −1.0 | (−4.5, +2.5) |
| **Δ_P (co-primary)** | **−1.0 pp** | (−2.5, +0.5) |

The basis (MY JUDGEMENT):
- F's home72 − F premium is 6.8 pp, and the forensics put ≈ ¾ of the per-mon premium with context, so ≈ 5 pp is
  addressable on F.
- A 20M plateau fork at share 0.5 recovering ≈ 40 % of it gives ≈ +2.
- D1, with ≈ 16 % recombined-like draws and its own population on F, read F +4.0 and H +2.1.

---

## 6. The decision tree

```
G0  The H-CTX read (rc72, premium_recovery on ONE engine; R = rc72 − F, P = home72 − F)
 ├─ KILLED (R ≥ 0.75 P) ────────► SHELVE X62. The premium is set-level and X62 adds no set; set levers (D3, N1-P's
 │                                 new sets) keep the hope. Record it here; nothing is built.
 ├─ INCONCLUSIVE (P's LB ≤ 0) or VOID ► HOLD, nothing built. The cheap remedy is more home72 / F control units on
 │                                 the same engine (CPU only); the orchestrator decides. The cross-engine bands are
 │                                 descriptive and never route this tree.
 ├─ UNDECIDED (0.25 P < R < 0.75 P) ► BUILD-AND-HOLD the corpus (≈ 1 agent-day, ≈ 1–3 CPU-h). Then G1:
 │                                 N1-P NULL / LOSS / UNDECIDED-final ⇒ X62 by D1's rule as a SCREEN (GUESS +1.0,
 │                                 P(GAIN) 0.35); a screen GAIN goes to the PR rule;
 │                                 N1-P GAIN ⇒ no fork; the corpus is a candidate member of the breadth stack (strategy
 │                                 rank 3, only on PR-1 BREADTH-GAIN).
 └─ SUPPORTED (R ≤ 0.25 P) ─────► BUILD the corpus, then G1:
       G1  N1-P look 1 / final
        ├─ GAIN ──────────────► X62 REDUNDANT as a standalone plateau lever (§4). The corpus is offered to X53-N1
        │                        (a later corpus arm, its own registration) and to the stack.
        ├─ CONTINUE ──────────► wait for N1-P look 2.
        └─ NULL / LOSS / UNDECIDED-final ► REGISTER X62: the PR rule if PR-1's C0 is banked on a playable engine,
                                 else D1's rule. GUESS +1.5 (−1.5, +4.5) after a NULL; P(GAIN) 0.44 (D1) / 0.57 (PR).
G2  H-MID (registered 2dd522785) routes the MECHANISM, never the go
 ├─ POLICY-CANNOT-ACT ──► the critic already sees the U deficit, so the policy carries the memorisation, which is
 │                         X62's target. The prior rises slightly (MY JUDGEMENT). Mechanism reads M1 + M2. Search on
 │                         V has headroom on U (design_search_next.md), a separate CPU-lane lever that competes with
 │                         X62's read for the anchor lane: the orchestrator orders them.
 ├─ CRITIC-MISCALIBRATED-ON-U ◄ READ 2026-10-26 (f200258f): Δbias +0.076 [+0.016, +0.139]; V's ordering intact
 │                         (AUC 0.667 U vs 0.663 home72). The critic shares the memorisation; breadth trains both
 │                         heads, so X62 still applies. M3 (Δbias on X62 vs C0 captures) IS a registered SECONDARY:
 │                         does X62 shrink Δbias, the part of the U deficit V does not see? (Not "V's optimism":
 │                         V is about calibrated on U and under-confident on home72, H-MID FINDING 2.) Search: V's
 │                         error is a between-set LEVEL shift, which a one-ply comparison within a node cancels, so
 │                         this branch alone does not keep search paused (H-MID FINDING 1). The direct test is S-U
 │                         (design_search_u_headroom.md), a CPU read (≈ 1,000 CPU-h, a rented-box read) that would
 │                         compete with X62's for the anchor lane: the orchestrator orders them.
 └─ INCONCLUSIVE ───────► no change; M1–M3 all reported.
G3  X62's own outcome (by its registered rule)
 ├─ GAIN, P NON-INFERIOR ► team-context breadth works at the plateau. X53-N1's corpus question reopens (X62 or the
 │                         union), and X62 joins the breadth stack. Nothing joins the production recipe on one read.
 ├─ GAIN, P not non-inferior ► TRADE: the exchange rate goes to the owner, as D1's.
 ├─ NULL ───────────────► with N1-P NULL as well: plasticity is the leading explanation; strategy T3 (the plasticity
 │                         probe, N5-S). Context breadth at the plateau is retired for this era.
 └─ UNDECIDED ──────────► feeds PR-1's family as a reported line (never pooled after the fact).
```

**Why UNDECIDED H-CTX does not fund a standalone fork when N1-P reads GAIN.** Under a partial context premium, X62's
expected effect (+1.0 pp) is below what its read can see without PR-1's C0. N1-P's GAIN already funds the larger
breadth test (X53). The corpus build is cheap and serves the stack either way.

---

## 7. Costs

| item | GPU-h | lane-h (160–230 games / h) | agent-days | note |
|---|---:|---:|---:|---|
| corpus build + soak + tests | 0 | 0 | ≈ 1 | ≈ 1–3 CPU-h for the build (UNVERIFIED: the validator over ≈ 15k candidates; FINDING 4) + ≈ 30–35 min of soak, at nice 19 under `mem_cap.sh` |
| pin commit, kit arm, specs, PREREG | 0 | 0 | ≈ 1 | AM-N1P's pattern: an own-pin check, a kit test, two read specs |
| training, look 1 (3 seeds) | **≈ 12.3** | — | — | 3 × ≈ 4.1 (11 sibling forks at this pin and length, 3.84–4.43; N1-P s3503 4.0) |
| training, 5 seeds | **≈ 20.5** | — | — | C0 shared: 0 GPU-h |
| **read, D1's rule** | 0 | **≈ 88–127** | — | look 1: 6 finals × (1,200 F + 820 reported: H / P72 / home72 7 units each, rc72 20) = 12,120 games, 53–76 lane-h; look 2 with the sha-checked cross-state import: 4 new finals × 2,020 = 8,080 games, 35–51 lane-h. C0 re-played on the read's engine |
| **read, PR rule** (PR-1's C0 banked on the engine) | 0 | **≈ 126–181** | — | X62: 5 × (3,480 F + 820) = 21,500 games; C0 top-up 3,400 F + 5 × 820 = 7,500 |
| read, PR rule, C0 fresh | 0 | ≈ 187–269 | — | 43,000 games |
| M2 / M3 mechanism reads | 0 | 0 | ≈ 0.5 | captures parsed (minutes); M3 ≈ a few CPU-h of forward passes |
| **total, D1's rule** | **20.5** | **88–127** | **≈ 2.5** | ≈ 4.5–6.1 resource-days; P(GAIN) 0.52 ⇒ ≈ 0.10 per resource-day |
| **total, PR rule with PR-1's C0** | **20.5** | **126–181** | **≈ 2.5** | ≈ 6.1–8.4 resource-days; P(GAIN) 0.65 ⇒ ≈ 0.09 per resource-day |

The CPU lane is the binding resource, as it is for the whole strategy tree (strategy §6). X62's read competes with the
H-CTX read itself (35–70 h), N1-P's look 1 (≈ 60 h), D3 look 2's read and PR-1 (165–238 lane-h). A rented CPU box
(strategy §3.1, an owner question) would serve X62's read on the same approval.

---

## 8. Adversarial self-check

1. **"X62 is D1 again."** Partly true (§1.2). D1 was 31 % recombined-like, and its UNDECIDED +2.10 is the nearest
   evidence. What X62 adds is purity (0 new sets), clean panels (no population hazard on F) and 6× the teams. If
   D1's effect came from its forum population, X62 shows less on F than D1 did. If it came from context, X62 shows
   more. Both outcomes are informative. **The incremental information is real but modest, and this doc does not
   claim more.**
2. **H-CTX is about P\* PLAYING new contexts, not about TRAINING fixing them.** A SUPPORTED read says the premium is
   tied to sixes. It says nothing about whether 20M steps at the plateau can learn team-level play. N1-P is the best
   evidence on that, which is why G1 sits under G0. The plasticity confound is shared with every plateau fork, and X62
   does not escape it.
3. **The per-panel gradient (§1.3) fits the set-level account just as well,** so KILLED is a live outcome (GUESS: H-CTX
   SUPPORTED 0.30, UNDECIDED 0.30, KILLED 0.20, INCONCLUSIVE 0.20). The rc72 author predicted R / P ≈ 0.25–0.4,
   SUPPORTED or a low UNDECIDED. X62 is designed before the verdict so that the tree is written before it lands, not
   because the verdict is expected.
4. **Recombination at ≤ 2 shared breaks the pool's cores** (309 set pairs co-occur in ≥ 5 pool teams). The new teams
   may be less coherent than human-built ones, and the network could learn "recombined-team play", which helps rc72
   and not F. Teammate-lift weighting mitigates this. M1 against F is the read that separates the two. A looser
   distance (≤ 3) keeps more cores but is a different lever; it is not registered here.
5. **Kakuna's own response to recombined teams (assumption A-K)** enters H-CTX's gate, not X62's primary: F is the
   meter, and rc72 appears only as M1. A gate verdict driven by Kakuna piloting rc72 badly would mis-route this tree.
   The H-CTX PREREG names this confound (its §7.2), and no Kakuna-vs-Kakuna strength cell was played.
6. **Share 0.5 keeps half the draws on the 719 memorised sixes.** If memorising the pool is the failure, a stronger
   form is the trainee side at share 1.0. That is a second lever, with a likely larger in-pool cost. It is listed, not
   designed: one lever per A/B, and N1-P's share keeps the corpus contrast clean.
7. **Family-wise false GAIN.** X62 is a fourth breadth arm beside D1, D3 and N1-P. At 0.025–0.05 each, the chance that
   at least one of four reads a false GAIN is ≈ 0.10–0.19. PR-1's pooled estimand is the family's guard. X62's own
   line enters PR-1 only if PR-1 is (re)registered with it before any of its games.
8. **The post-hoc origin.** The forensics are descriptive and post hoc, and X62 was designed from them. The registered
   H-CTX read is what turns this into a pre-registered chain. X62 must be registered after G0 and G1 and before any
   X62 game.
9. **Costs could be low-balled.** The lane rate is the measured 160–230 games an hour, not the quiet-box 360. The
   validator build time and the within-corpus cap are UNVERIFIED (FINDINGS 3–4). The PR-rule cost assumes PR-1's C0
   is on a still-playable engine. The engine has moved weekly (parity batches B6–B10, BF), so assume the fresh-C0 row
   (187–269 lane-h) unless the engine is frozen.
10. **Is any of this needed if N1-P reads GAIN?** No (§4): X62 is then redundant as a standalone lever, and the doc
    says so rather than finding a reason to run it.

---

## 9. FINDINGS (standing rule 7)

1. **D1's corpus is already 31 % recombined-like** (624 / 1,981; 997 at ≥ 4 pool sets and ≤ 2 shared). Every past
   reading of D1 as "human-team breadth" is partly a reading of team-context breadth with familiar sets.
2. **F holds 19 recombined-like teams.** A post-hoc split of X51's banked F games (P\* on those 19 teams against the
   other 53) is a free, banked mini-H-CTX. At ≈ 100 games it would be ±10 pp, so it was NOT computed. It must never
   stand in for the registered read.
3. **The within-corpus cap (≤ 4 shared with an accepted team) is unverified at 12,000 teams.** The pool's profile is
   concentrated: Tyranitar is on 63 % of pool teams and Swampert on 59 %. The builder's manifest must report the
   within-corpus histogram, with a fallback cap of 5 declared.
4. **The build's wall time is unverified.** Node validation of ≈ 15k candidates has no measured rate here. N1-P's and
   D3's build times are not recorded in their manifests.
5. **The engine-identity rule makes the cheap PR-rule cost conditional.** PR-1 has not run. If it runs, its C0 is
   reusable only on a byte-identical play path (`repin`).
6. **H-CTX's Amendment-1 read is not listed in STATUS's "Running now"** (as of `fd4160d76`). Its timing, and so this
   gate's, is unknown here. At 2–4 units an hour it needs 35–70 h once started.
7. **"Familiar set" is the mon tier** (species + moves), as in the forensics. Because X62 copies blocks verbatim, its
   sets are also pool sets at the strict tier (item and spread). The forensics' and rc72's familiarity measures are
   mon tier only.
8. **This unit** ran two CPU scripts (`set_familiarity.py`, `power_pgain.py`; < 1 min each, nice 19 on the agent
   cores, no lease, `CUDA_VISIBLE_DEVICES=""`). It played no game, built no corpus and changed nothing under `data/`.

---

## 10. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT, gated on H-CTX)** (research-design worker, orchestrator brief; docs + two CPU scripts) | X62: a P\*-fork at the X35 production shape (+20M, the `f4b25238` pin + a corpus-only commit), with `--team-mix x62_recomb_v1@ALL:0.5 --team-mix-sides both`: 12,000 teams of verbatim pool blocks, ≤ 2 shared with any pool team, a pool-matched species profile with Smogon teammate lift, static; read on F vs Kakuna against C0 on one engine (the PR rule if PR-1's C0 is banked, else D1's); the gate tree keyed on H-CTX, then N1-P, with H-MID routing the mechanism reads | treating N1-P as this lever (28 % pool-set slots, 0.6 % recombined-like); an in-loop re-draw (≈ 24 pilots a team makes it a no-op); share 1.0 (a second lever); the then-current recipe (a new C0 for no information); a standalone fork on UNDECIDED H-CTX with N1-P GAIN | `piloting_gap_forensics_2026-10-26/` (`c758fcb82`); `hctx_rc72_2026-10-26/PREREG.md` (`caec2bb2c`); `n2_identity_test_2026-10-26/` (`fd4160d76`); `x61p_n1_production_2026-10-26/PREREG.md`; [`x62_design_2026-10-26/`](../research_state/measurements/x62_design_2026-10-26/) (`set_familiarity.log`, `power_pgain.log`) |
| 2026-10-27 | **G2 resolved: H-MID read CRITIC-MISCALIBRATED-ON-U** (search-research design worker, orchestrator brief; docs only) | G2's branch marked READ; M3 is a registered SECONDARY, its question re-worded to Δbias (H-MID FINDING 2: "optimism" depends on an absolute baseline); "search stays paused until some arm fixes V on U" replaced by H-MID FINDING 1 (a between-set level error cancels within a node; V's ordering is intact) and a pointer to S-U | keeping "search stays paused" as an X62 consequence (it is the search ledger's call, and S-U is its test) | [`h_mid_value_test_2026-10-26/README.md`](../research_state/measurements/h_mid_value_test_2026-10-26/README.md) (`f200258f`); [`design_search_u_headroom.md`](design_search_u_headroom.md) |
