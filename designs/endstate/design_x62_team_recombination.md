# X62 — team recombination: train P\*-forks on new sixes made of familiar sets

**Status: REVISED DRAFT (2026-10-27), explicit-only, GATED ON rc72b. Nothing is built, registered or launched.** It
designs the training lever that H-CTX implies, if the H-CTX read supports it. This revision answers the adversarial
review [`reviews/x62_review_2026-10-27.md`](reviews/x62_review_2026-10-27.md) (`d997a146`); §10 disposes of every
finding. H-CTX comes from the piloting-gap forensics
([`piloting_gap_forensics_2026-10-26/README.md`](../research_state/measurements/piloting_gap_forensics_2026-10-26/README.md),
`c758fcb82`) and is registered as the rc72b read
([`hctx_rc72_2026-10-26/PREREG.md`](../research_state/measurements/hctx_rc72_2026-10-26/PREREG.md), Amendment 2 at
pin `905c29f5b`, rule `premium_recovery`; rc72 is RETIRED, no game of it counted). The rc72b read is expected to
finish ≈ 15:00 PT on 2026-10-27. This doc does not wait for it: §6 writes every branch. Goal (b), its meter and every
read cited here belong to [`design_beat_metamon_path.md`](design_beat_metamon_path.md). The ranking rules are
[`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md)'s: P(GAIN) per full cost, CPU
included, and one lever per A/B. The scripts behind this doc's numbers are in
[`x62_design_2026-10-26/`](../research_state/measurements/x62_design_2026-10-26/): `set_familiarity.py` and
`power_pgain.py` (the first draft), the review's `review_2026-10-27/` pair, and this revision's
`revision_2026-10-27/strict_draw_sim.py` and `revised_odds.py`, each with its `.log`. All are CPU-only and run in
under a minute.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau); C0 = X35's plain-continuation control (P\* + 20M, no lever); the POOL = the 719 teams
P\* trained on; home72 = 72 pool teams; F = `forum72`; H = `heldout72_hpfix`; P72 = the procedural panel; U = H, F and
P72 at equal weight (goal (b)'s meter); S = in-house strength against the frozen reference population on home72; a
SET = one mon's species plus its sorted moves, Hidden Power typed (`gen3_team_overlap_v1`, `main.anchors.team_overlap`);
**RECOMBINED-LIKE** = a team with ≥ 5 of its six sets in the pool that shares ≤ 2 sets with any ONE pool team (familiar
sets in a new six); **the STRICT class** = a Hidden Power that plays its named type at 70 as written (Amendment 2's
rule; a Hidden Power with no IV line and a type other than Dark fails it); σ_s = the between-seed SD of a fork
final's true score; **PR-1′** = the strategy's precision read as re-scoped by its second review (D3 + N1-P against
C0 on F, D1 reported; [`REVIEW_2.md`](../research_state/measurements/strategy_review_2026-10-26/REVIEW_2.md) §6);
**the tree pin** = the one post-`b0441623` playing commit REVIEW_2 M1 asks every X35-frame read to share (its option
C preferred, A the fallback); **lane-h** = one wall-hour of the Kakuna anchor lane at `--concurrency 1` on rule 5b's
agent cores, at the MEASURED 160–230 games an hour; **resource-day** = (GPU-h + lane-h) / 24; **r** = the exchange
rate of §5.2; GUESS = an unmeasured effect; ESTIMATE = a cost from measured rates; MY JUDGEMENT = an inference no
read states.

---

## 0. One screen

- **The lever.** Fork P\* for 20M steps, with half of each side's teams drawn from **`x62_recomb_v1`**: 12,000 NEW
  teams. Every mon block is copied from a pool team (nickname stripped), no team shares more than 2 sets with any
  pool team, and every Hidden Power is of the STRICT class. The network keeps the sets it knows and loses every six
  it has memorised, so what is new is how to play a team. It is the complement of D3 (new sets inside old teams).
- **The STRICT rule costs the corpus 86 sets** (§2.2, measured on the prototype draw). 739 of the pool's 4,314 mons
  (17 %) carry a Hidden Power with no IV line; under the first draft's rule 69 % of corpus teams (8,334 of 12,000)
  would have carried one. Under STRICT the corpus draws none: **572 of the pool's 658 sets** and 63 of its 69
  species. The lever is therefore "contexts on familiar sets, minus 86 sets that keep only their pool-half exposure".
  home72 holds none of the 86.
- **N1-P is NOT this lever** (§1.2): 28 % of its slots carry a pool set, and 0.6 % of its teams are recombined-like.
  **The closest existing arm is D1**: 31 % of its teams are recombined-like. It read UNDECIDED.
- **The gate (§6), every branch written before rc72b lands.**
  - **G0, rc72b.** KILLED ⇒ SHELVE the fork (the premium is set-level). INCONCLUSIVE or VOID ⇒ HOLD. **SUPPORTED and
    UNDECIDED route the same way** (build the corpus, then G1), and differ only in the prior (+1.5 vs +1.0 pp on F),
    because the read cannot tell them apart at the R / P ≈ 0.25–0.40 its author predicts (review M4).
  - **G1, PR-1′ (when it has read).** KILL ⇒ **X62's 20M fork is retired**. The corpus still serves X63 (a ≥ 100M
    dose, which a 20M KILL does not reach) and X64. BREADTH-GAIN ⇒ no standalone fork; the corpus goes to the union
    (X63, or the stack). UNDECIDED, or not yet read ⇒ G2.
  - **G2, N1-P.** GAIN ⇒ X62 is redundant as a standalone fork. CONTINUE ⇒ wait. NULL, LOSS or UNDECIDED-final ⇒
    REGISTER X62.
- **The read (§5).**
  - **The primary:** F against Kakuna, by the PR rule with AM-5. That is 5 seeds, 3,480 F games a final, and C0
    re-played at the tree pin.
  - **The in-pool cost is a PRICE, not a harm bar.** An in-pool (home72) loss is acceptable up to the size of the U
    gain, both against Kakuna. That is r = 1, the orchestrator's default pending the owner, who can change r at any
    time before registration.
  - **The FLOOR (goal (a)):** the arm's home72 against Kakuna keeps LB > 0.50.
  - S, D1's old co-primary, is REPORTED and decides nothing.
- **The revised odds** (`revised_odds.log`; σ_s 0 / 1 pp):
  - **P(GAIN) 0.56 / 0.49** on the SUPPORTED run branch (GUESS +1.5, 80 % (−1.5, +4.5)), and 0.47 / 0.39 on
    UNDECIDED (+1.0, (−1.5, +3.5)).
  - **P(ADOPT-CANDIDATE) = GAIN, within the price and above the floor: 0.31 / 0.28** on SUPPORTED, 0.24 / 0.21 on
    UNDECIDED.
  - The design's own GUESS has the net (in-pool loss −1.5 against a U gain of ≈ +0.9) slightly over the default
    rate. At r = 2 the joint odds rise to 0.40.
- **Cost (§7).**
  - **GPU:** 20.5 GPU-h.
  - **CPU, C0 fresh:** 238–342 lane-h, 10.8–15.1 resource-days. Per resource-day that is **0.032–0.052 for GAIN and
    0.018–0.028 for ADOPT-CANDIDATE**.
  - **CPU, C0's F banked at the tree pin by PR-1′:** 173–249 lane-h. Per resource-day, 0.043–0.069 (GAIN) and
    0.025–0.038 (ADOPT-CANDIDATE).
  - The first draft quoted 0.09–0.10 per resource-day. Measured the same way, X62 sits below the strategy's rows
    4–9 (0.06–0.09, P(GAIN) alone).

---

## 1. Why this lever, and why it is not one we already have

### 1.1 The evidence that points here

- **Whether the premium goes with the TEAM or the set is OPEN until rc72b.** P\*'s mons out-kill Kakuna's by +0.081
  KOs per entry on home72, and by −0.028 on U. Restricted to sets in our pool on both sides, the drop is **+0.093
  [+0.004, +0.176]**. But that measure restates the result, because the winner KOs all six. Within won games the drop
  is −0.011, and within lost games −0.027. The comparison that holds the result fixed (seen against unseen sets on
  the same U team) is −0.007 [−0.093, +0.070], which both H-CTX and the set-level account survive. This is context
  for H-CTX, not support (forensics §4; DESCRIPTIVE, post hoc; review
  `../research_state/measurements/h_mid_value_test_2026-10-26/REVIEW.md` B1, M3).
- **The gap forms in the mid-game, from level positions.** From level material at turn 10, P\* wins 0.609 on home72
  and 0.501 on U (forensics §1). That is where a team plan matters: which mon to preserve, what the win condition is.
- **The policy is compositional at the set level.** Re-drawing one familiar bench set moves P\* only 1.19×
  [1.07, 1.33] as much as re-drawing an unfamiliar one, so N2 was dropped
  ([`n2_identity_test_2026-10-26/README.md`](../research_state/measurements/n2_identity_test_2026-10-26/README.md),
  `fd4160d76`). That read leaves open "plan memorisation keyed on the battle's course or on revealed mons", which is
  H-CTX's mechanism.
- **The lever H-CTX names.** If P\* memorised its 719 sixes, the cure is many sixes built from parts it already
  knows, so that training pays only for the team-level skill. The forensics wrote this down as "recombine pool sets
  into new teams, the complement of D3". The context-generalisation survey
  ([`context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md),
  `5d554c23`) ranks it first, conditional on rc72b, and asks for its in-pool cost to be registered as a price (§5.2).

### 1.2 Where each corpus sits on set novelty and team context (measured, `set_familiarity.log`)

| corpus / panel | teams | slots carrying a pool set | max sets shared with ONE pool team | recombined-like | what it varies |
|---|---:|---:|---|---:|---|
| `d3_setmut_v1` (D3) | 2,876 | 0.813 | 4: 1,381 · 5: 1,495 | 0 | **sets**, inside familiar sixes |
| `w_forum_v1@ABC` (D1) | 1,981 | 0.776 | ≤ 2: 1,301 · 3: 579 · ≥ 4: 101 | **624** (≥ 4 pool sets: 997) | contexts, mostly familiar sets, human-built; the forum population |
| `n1_proc_v1` (N1-P) | 12,000 | **0.280** | ≤ 2: 11,960 · 3: 40 | **73** | contexts **and** sets (72 % new), procedural |
| **`x62_recomb_v1` (X62, proposed)** | 12,000 | **1.000** | **≤ 2 by construction** | **12,000** | **contexts**, on 572 of the 658 pool sets (§2.2) |
| home72 | 72 | 1.000 | 6 (pool teams) | 0 | — |
| rc72b (H-CTX's panel) | 72 | 1.000 | 2: 72 (Amendment 2) | 72 | — |
| F | 72 | 0.701 | 1: 14 · 2: 42 · 3: 16 | 19 | — |
| H | 72 | 0.380 | ≤ 2: 71 · 3: 1 | 3 | — |
| P72 | 72 | 0.308 | ≤ 2: 72 | 0 | — |

The pool holds 658 distinct sets of 69 species, and 54 of those species have ≥ 2 sets. Every corpus team's species
are 97–100 % pool species.

**What this says, honestly.**
- **N1-P is not X62.** Most of N1-P's slots are sets P\* never piloted, drawn independently of each other (incoherent
  by construction, its PREREG FINDING 5). So an N1-P NULL cannot tell "new contexts do not help" from "new sets plus
  incoherence cost what contexts buy".
- **D1 is partly X62 already.** Under H-CTX, D1's U +2.10 [−0.92, +5.11] (UNDECIDED, final) came from a corpus whose
  draws were ≈ 0.5 × 31 % ≈ 16 % recombined-like. Its F is hazarded, because F is D1's own forum population. X62 is
  the version with no new set, no shared population with any panel, and 6× the team count.
- **F is itself 26 % recombined-like** (19 of 72 teams). Of the unfamiliar panels, F is where a context lever should
  show first.

### 1.3 The counterweight: the per-panel gradient fits the set-level account too

Across X51's four panels, P\*'s score against Kakuna falls with the share of pool sets:

| panel | share of pool sets | P\* vs Kakuna |
|---|---:|---:|
| home72 | 1.00 | 0.598 |
| F | 0.70 | 0.530 |
| H | 0.38 | 0.497 |
| P72 | 0.31 | 0.433 |

Each score is ±5 pp (n 400; X51 README). A purely set-level account predicts exactly this ordering. So does H-CTX,
because familiar contexts and familiar sets move together across these panels. Only rc72b separates the two, which is
why X62 waits for it.

---

## 2. The corpus `x62_recomb_v1` (a builder sketch; NOT BUILT)

A builder beside `build_n1_proc` in `utils.team_corpora` (proposed name `build_x62_recomb`), deterministic, with
`--check` as its siblings have. It generalises rc72b's `build_rc72.py` from a 72-team panel to a 12,000-team corpus.
**It reads the POOL for every set, the pool's own species counts for the profile, and Smogon only for teammate
coherence.** It ships no prior: it is a training corpus, like `n1_proc_v1`.

### 2.1 The draw of candidate i

One `random.Random` per (stream, i), as in `build_n1_proc`.

1. **Species, usage-matched to what P\* trained on.**
   - Slot 1 is drawn with weight c_pool(s), the species' occurrence count over the pool's STRICT-class mons.
   - Each later slot is drawn among species not yet on the team, which enforces the Species Clause by construction.
     Its weight is c_pool(s) × the geomean, over the team's species t, of lift(t, s). Here lift(t, s) = P(s | t) /
     share(s), from Smogon's teammate and usage tables, clipped to [¼, 4]. These are rc72b's rules, with the pool's
     counts in place of home72's.
   - So per-species exposure stays near the pool's: the arm changes contexts, not which species P\* pilots. That
     deliberately leaves H-RARE (rare-species piloting) out of this lever: one lever per A/B.
2. **The set: one STRICT-class pool OCCURRENCE of the drawn species**, uniform over those occurrences, so a set is
   drawn about as often as the pool repeats it. Item, ability, EVs, nature, IVs and moves are copied verbatim.
   **The nickname is stripped** (review m9): a head line `Nick (Species) (G) @ Item` becomes `Species (G) @ Item`,
   keeping the gender and item. A head line with no nickname is left alone. The builder's tests pin the rewrite on
   both forms. The network does not observe the nickname (N2 S3), and Showdown's Nickname Clause would refuse two
   copied blocks with the same nickname. Because the item and spread are verbatim, the sets are pool sets at the
   strict tier as well as the mon tier.
3. **Distance refusals at draw time.** A refused slot is redrawn, up to rc72b's `SLOT_TRIES`; after that the
   candidate fails.
   - **≤ 2 shared sets with ONE pool team.** This is the lever's definition (rc72b's `MAX_SHARED_TRAIN`).
   - **< 3 shared with ONE team** of every held-out panel (`heldout72`, H, F, P72, the sealed `forum216b`), of
     `n1p72`, and of **rc72b** (review M1). This is REVIEW A's revealed-fit radius, as in N1-P's and rc72b's builds.
     rc72b must stay H-CTX's and X62's held-out-context panel: its sets are all in the corpus's source, but none of
     its sixes may be. Without this refusal, 491 of 12,000 prototype teams sit at ≥ 3 shared sets with an rc72b
     team (`recomb_draw_sim_no_rc72b.log`). With it, all are at ≤ 2.
   - **≤ 4 shared with any corpus team already accepted**, and no duplicate six, so that near-copies do not spend
     dose. It fits (below), so the fallback cap of 5 is not needed.
   - **Not an exact pool species six** (review m2). A team with a pool team's six species is refused, so the "new
     six" holds at the species level too. This cost 298 refusals in the prototype and left 0 such teams.
4. **Whole-team checks.** A failure moves to the next candidate, and the manifest counts each rule's refusals.
   - the pinned `TeamValidator('gen3ou')` (`ladder_corpus_filter.js`);
   - the pack oracle (level 100, `build_w_forum.level_defects`);
   - **the STRICT Hidden Power rule**: rc72b's `hp_bad` in its `strict` form, through `hp_set_check`. It refuses a
     Hidden Power that does not play its named type at 70 as written, or whose IVs the validator rewrites. It is the
     authority; step 2's text filter only keeps the draw from wasting attempts;
   - the engine scan (`scan_move_probe`, which THROWS on an unplayable id);
   - `check_label_encodable`.

### 2.2 The Hidden Power rule: STRICT, and what it costs (`strict_draw_sim.log`)

**Why STRICT, not `validated`.**
- STRICT is the class the live server accepts, and the only class in rc72b (X62's gate and its M1 panel), home72 and
  F.
- A corpus shared with X63 and X64 should not carry a set class that every meter excludes.
- `validated` (the first draft's rule, and what the pool holds) accepts a Hidden Power with no IV line, because
  Showdown's batch validator rewrites its IVs. The live server rejects that team. Whether the rewritten set plays its
  named type in TRAINING is Amendment 2's open question (FINDING 9).

**What STRICT does to the corpus** (the review's prototype re-run under the revised rules; text emulation, no Node
validator):

| quantity | first draft (`validated`; review prototype) | **revised (STRICT, rc72b refused, no pool species six)** |
|---|---|---|
| pool mons the draw can use | 4,314 | **3,575** (739 removed, 17.1 %; Amendment 2 counts 737 HP sets by the validator) |
| distinct pool sets drawn | 658 of 658 | **572 of 658** (86 sets have no STRICT occurrence in the pool) |
| species | 69 | **63** (arcanine, blaziken, dragonite, glalie, linoone and weezing occur only in the no-IV-line form) |
| corpus teams carrying a no-IV-line Hidden Power | **8,334 of 12,000 (69 %)**, 17.2 % of slots | 0 |
| slots carrying any Hidden Power (pool 0.443) | — | 0.367 |
| species TVD from the pool | 0.046 | 0.048 |
| **set-exposure TVD from the pool's occurrences** | 0.073 | **0.173** |
| Tyranitar / Swampert share of slots (pool 0.106 / 0.098) | 0.091 / 0.081 | 0.090 / 0.080 |
| within-corpus max shared with an earlier team | {≤ 2: 3,089, 3: 8,137, 4: 774} | {≤ 2: 2,373, 3: 8,307, 4: 1,320}; 29 refusals at the ≤ 4 cap |
| max shared with ONE rc72b team | 491 at ≥ 3 | **all ≤ 2** |
| accepted | 12,000 of 12,000 indices | 12,000 of 12,000 indices (25 s, Python only) |

**What this means for the lever.**
- **It is no longer "contexts only".** The 86 withdrawn sets keep their pool-half exposure (share 0.5), so P\*
  pilots them about half as often as C0 does. That is a set-level change inside a context lever.
- **Under H-CTX it should be small**, because the premium is in the sixes, not the sets. The cost side is benign:
  home72 holds none of the 86, since every home72 mon is a pool occurrence and home72 carries no no-IV-line Hidden
  Power (Amendment 2).
- **Under the set-level account** it would show as a small F or H loss on any panel that holds those sets. The
  manifest therefore lists the 86 sets, and §5.4 reports F's and H's exposure to them.
- **No reweighting.** Re-weighting the remaining sets to the pool's profile would be a second lever, and the review's
  m1 left it optional. Both TVDs and the exposure-ratio quantiles go into the manifest.

**Rejected: completing the IV line.** The alternative is to write into each no-IV-line block the IVs the validator
would write. That keeps all 658 sets and passes STRICT. It is not verbatim, though, and it is right only if training
plays these sets as the validator rewrites them, which is FINDING 9's open question. It can be revisited after that
question is answered, as its own lever.

### 2.3 Legality, verified at source

Showdown's `[Gen 3] OU` entry is `Standard` + One Boost Passer Clause + Accuracy Trap Clause + Freeze Clause Mod +
Speed Pass Clause, and the banlist is all species, ability, move and item bans (`deps/pokemon-showdown`, its config
formats file). **There is no Item Clause in gen3 OU**, so two mons may hold the same item. A pool block passes every
SET-level rule already, because it passed inside its pool team. Recombination can break only the TEAM-level rules,
which the validator runs:
- the Species Clause, enforced at draw time;
- the Nickname Clause, which is void because nicknames are stripped;
- **the One Boost Passer Clause**: at most one Baton Passer that can boost. Two pool Baton Passers can collide, so
  this is the one refusal we expect (4 in the review's prototype).

The Sleep and Freeze Clauses are battle mods, not team rules.

### 2.4 Size, tiers, refresh, manifest

- **12,000 teams, N1-P's count**, so X62 − N1-P is a corpus contrast at one team count and one share. The prefix
  tiers K1 ⊂ K3 ⊂ ALL serve X64's curve and a later dose read.
- **A static corpus, not re-drawn.** At share 0.5 over 20M (≈ 570k games; D1 doc §4.1's rule), each corpus team is
  piloted ≈ 24 times a side, as in N1-P. Memorising one team took thousands of pilots: X52 practised 36 teams for 5M
  steps, ≈ 4,000 pilots each, for +3.2 pp [−0.2, +6.8]. At 24 pilots a team, a static corpus is already effectively
  fresh.
- **What memorisation still acts on is the pool's own 719 teams.** On the opponent side they get half the draws,
  ≈ 395 pilots each per 20M. On the trainee side X26's draw is `default_biased` with `bias_prob` 0.1 toward the
  curated sample teams (`models/rb_x26_s2601/metadata.json`). At share 0.5 the trainee side is therefore 0.5 corpus,
  0.45 pool and 0.05 sample teams, ≈ 355 pilots a pool team (review m7). Whether the sample teams are home72 is
  UNVERIFIED (FINDING 10).
- A share of 1.0 on the trainee side is a second lever (§8 item 6).
- **The startup cost is measured, not guessed.** N1-P's 12,000-team table had a team phase of 18.5 s
  (`startup_team_phase_at_pin.txt`). It trained at **1,433 steps/s, 4.0 GPU-h** for its 20M (dose record
  `x35_breakers_stage_2026-10-17/dose/rb_x35_n1p_s3503.json`), level with C0's 1,384–1,424 steps/s.
- **The manifest reports:**
  - each rule's refusals;
  - the species and set-exposure TVDs, and the exposure-ratio quantiles;
  - the 86 withdrawn sets;
  - the within-corpus histogram and the max shared with rc72b;
  - **the near-duplicates against the other breadth corpora** (review m8; reported, not refused).

  The prototype's near-duplicates: no team shares ≥ 5 sets with any D1, D3 or N1-P team. At 4 there are 2 teams
  against D3, 9 against N1-P and 50 against D1. The union corpora (X63's and the stack's) re-check this on their own
  builds.

**The soak** (`utils.team_corpora.soak`, `gen3_team_corpus_soak_v2`): 120,000 battles, N1-P's size. D3's re-soak of
24,240 battles took 382.5 s at nice 19 on 3 cores, so this one takes ≈ 30–35 min (ESTIMATE).

**Separation.**
- X62 adds no set to any panel's coverage, so the M4 hazard (N1-P PREREG §8) does not apply. Its U is as readable as
  C0's.
- Every held-out panel, `n1p72` and rc72b are refused at the radius-3 fit.
- The startup guard at the pin (`team_mix_guard`, radius 4) passes by construction.

---

## 3. The arm and its control

| item | value |
|---|---|
| start | P\* (`checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`), as C0 |
| **the ONE lever** | `--team-mix x62_recomb_v1@ALL:0.5 --team-mix-sides both`: N1-P's share and sides, so X62 − N1-P and X62 − D1 are corpus contrasts at one share (descriptive) |
| everything else | C0's argv token for token (kit arm `x62`, proposed, with a kit test of the `test_n1p_is_c0_plus_…` kind): `--behaviour-check warn`, +20M to 300,000,170, the recipe inherited from X26's record |
| **pin** | the production pin `f4b25238` + ONE commit holding only the corpus's three files and its one `CORPORA` line (AM-N1P's pattern, `batch.N1P_PIN`). This is C0's code byte for byte, so the 10-26 GIGO fixes are absent from both arms (N1-P PREREG §2's argument) |
| seeds | 3501–3505, all five at once (the PR rule has no screen look) |
| **control** | **C0 s3501–s3505**, banked runs at `f4b25238` (each K9(b) PASS): 0 GPU-h. **Every Kakuna game, X62's and C0's, is played at the tree pin** (REVIEW_2 M1: option C, E1's game paths under the post-`b0441623` harness with the H22 pack, if built; else option A). C0's F rows are imported only from a read at that same pin, sha-checked (PR-1′'s C0, if it ran there). Never D1's, X51's or any legacy-pack rows |

---

## 4. How it differs from D3, D1 and N1-P, and what N1-P's read implies

| | D3 | D1 | N1-P | **X62** |
|---|---|---|---|---|
| new sets per team | 1–2 (by construction) | ≈ 1.3 (22 % of slots) | ≈ 4.3 (72 %) | **0** (86 pool sets withdrawn, §2.2) |
| new six (≤ 2 shared with any pool team) | never (≥ 4 kept) | 66 % of teams | 99.7 % | **100 %** |
| coherence | a pool core | human-built | procedural, incoherent | pool sets, teammate-lift species |
| teams | 2,876 | 1,981 | 12,000 | 12,000 |
| share | 0.3 | 0.5 | 0.5 | 0.5 |
| panel hazard | none | F is its population | H, P72 (62 % / 56.5 % of full sets) | **none** |
| the H-CTX part it trains | set novelty (≈ ¼ of the premium) | mostly context | context + sets | **context** |

**N1-P (G2), and what each outcome implies for X62:**

| N1-P outcome | implication for X62 |
|---|---|
| **GAIN** (F; X53-N1 GO) | New contexts at 10⁴ move F even with mostly new, incoherent sets. X62's only extra information would be the corpus choice for X53. A ±3 pp read cannot separate two positive arms (X62 − N1-P needs ≈ 4× the games). **X62 is REDUNDANT as a standalone plateau fork.** Its corpus is offered to X53-N1 as a later corpus arm, and to X63's union |
| **NULL** | NOT a kill of X62. N1-P's NULL is confounded by set novelty and incoherence (its FINDING 5), and at 20M by dose: ≈ 24 pilots a team. **Plasticity is not the explanation:** the plasticity check read NO-LOSS (`8bebdfe31`; P\*'s refit AUC 0.986× X26 @25M's). X62 is the arm that splits "new sets and incoherence cost" from "contexts at 20M are too small a dose". GUESS +1.5 (−1.5, +4.5): a plateau that ignored 12,000 new contexts makes a dose shortfall likelier |
| **UNDECIDED, Δ̂ > 0** (CONTINUE) | wait for N1-P's look 2 (≈ 8.2 GPU-h plus its read) |
| **LOSS** | Incoherence or set novelty costs. X62, which has neither, rises in rank, as in the NULL row |
| **UNDECIDED, final** | as NULL |

**D3 (look 2 queued)** is the other half of the same decomposition. Under H-CTX its ceiling is ≈ a quarter of the
premium (forensics, "How these bear"). D3 and X62 are complements, not rivals: a D3 GAIN and an X62 GAIN would sum
under H-CTX, which is the case for a union.

**X63 and X64 inherit this corpus.** X63 ([`design_x63_long_breadth_continuation.md`](design_x63_long_breadth_continuation.md))
puts it in a ≥ 100M union continuation (the survey's "dose before more 20M forks"). X64
([`design_x64_toy_context_curve.md`](design_x64_toy_context_curve.md)) builds a toy context-count curve from its K1 ⊂
K3 ⊂ ALL tiers. Both take §2's rules, including the STRICT rule's 86 withdrawn sets.

---

## 5. The read

### 5.1 The primary: F against Kakuna, by the PR rule

Kakuna checkpoint 34, greedy against greedy, CPU, role-balanced 20-game units, F's committed config. Every unit plays
at the tree pin, X62's and C0's alike.
- **The rule** is strategy §3.2's PR rule, with AM-5:
  - 5 seeds a side, **3,480 F games a final** (17,400 an arm);
  - the normal interval, widened by √max(1, φ̂), where φ̂ is the mean of the two arms' seed-dispersion ratios
    (`main.x35.d1`'s estimator);
  - **GAIN iff LB > 0**; **NULL iff UB < +1.5 pp** (PR-1's KILL bar; LOSS iff UB < 0, a sub-case); otherwise
    **UNDECIDED**. By rule 8, a bound within 1e-9 of a threshold reads UNDECIDED.
- **D1's rule is dropped** as a read option. It was a screen at 0.37 at +1.5 pp (σ_s 0). The strategy reads every
  T2 fork by the PR rule, and the price below needs every final on the same panels anyway.
- **Why F rather than U.** F is where a context lever should show (70 % pool sets, 26 % recombined-like), and it is
  the breadth family's shared panel. U enters the price (§5.2), and goal (b)'s own number is read from U with no
  coverage caveat (review m6).
- **Power** (`power_checks.log` A, with AM-5): at +1.5 pp P(GAIN) is 0.74 / 0.66 / 0.46 at σ_s 0 / 0.5 / 1 pp, and
  at +3 pp it is 1.00 / 1.00 / 0.94. **False GAIN is 0.018 / 0.028 / 0.040** with φ̂ estimated (review m3; the first
  draft's 0.025 assumed σ_s known).

### 5.2 The in-pool cost: a registered PRICE, with an exchange rate (review B1)

**Why a price.**
- Under H-CTX the in-pool edge IS the memorised pool sixes. X62 halves their exposure exactly as D1 did, so under its
  own premise it should cost about what D1 cost: −1.61 pp [−2.90, −0.32] in S.
- The literature makes an in-pool drop the expected sign of breadth working: CoinRun's train score, Hill's train
  accuracy, VGC-Bench's seen teams (survey recommendation 1(a)).
- A non-inferiority bar at −2.0 pp holds only 0.07 of the time at D1's cost (`power_checks.log` C). It would kill the
  lever for its predicted side effect.

**The rule.** Every final also plays **home72, H and P72 against Kakuna, at 400 games each** (20 units), and so does
C0. All the Δ below are X62 − C0.
- **net = Δ_home72 + r · Δ_U**, where Δ_U = (Δ_F + Δ_H + Δ_P72) / 3, all against Kakuna.
- **PRICE WITHIN iff net's point estimate ≥ 0.** By rule 8, within 1e-9 of 0 reads not WITHIN.
- **PRICE EXCEEDED iff the Welch 95 % UB of net < 0**, over the per-seed values, 5 against 5.
- **The exchange rate r = 1** is the orchestrator's default, pending the owner. In words: **an in-pool loss is
  acceptable up to the size of the U gain.** The owner can change r at any time before registration. The PREREG
  records the r in force and who set it. After the read, the owner can still rule on a TRADE, as with D1, but that is
  recorded as a post-hoc owner decision, never as the registered verdict.

**The FLOOR (goal (a)).** The arm's pooled home72 against Kakuna (2,000 games), widened by its own √max(1, φ̂), must
keep **LB > 0.50**. Goal (a) stays MET (P\* 0.598 [0.549, 0.644]). It is a floor on the arm, not a contrast.

**What decides.**

| class | condition | what it means |
|---|---|---|
| **ADOPT-CANDIDATE** | GAIN and PRICE WITHIN and FLOOR | context breadth on familiar sets moves F at an in-pool price inside the rate. A candidate, never a recipe change on one read (§6, G3) |
| **TRADE** | GAIN and FLOOR, neither WITHIN nor EXCEEDED | the best estimate is over the rate, but not clearly. Goes to the owner with net's interval; not adopted at the registered rate |
| **OVER-RATE** | GAIN and (EXCEEDED or FLOOR fails) | the in-pool cost is clearly over the rate, or goal (a) is at risk. Not adopted |
| **NULL / LOSS / UNDECIDED** | by §5.1 | §6, G3 |

**S, D1's co-primary, is REPORTED.** D1's NON-INFERIOR bar (Welch 97.5 %, LB > −2.0 pp) is printed beside it and
decides nothing. It costs ≈ 15 heavy-slot-h (review m5, ESTIMATE). It runs only when the heavy slots have room, and
never delays the Kakuna read.

### 5.3 The revised odds (`revised_odds.log`)

The model behind these numbers:
- Truths (GUESS, MY JUDGEMENT): Δ_F from the branch's GUESS; Δ_H = 0.5 Δ_F + N(0, 1 pp); Δ_P72 = 0.25 Δ_F + N(0,
  1 pp); Δ_home72 ~ −1.5, 80 % (−4.0, +1.0), independent of Δ_F.
- 5 against 5 seeds, every panel with the same σ_s.

| branch (GUESS on F) | σ_s | P(GAIN) | P(PRICE WITHIN) | P(FLOOR) | **P(ADOPT-CANDIDATE)** | P(GAIN, not EXCEEDED, FLOOR) |
|---|---:|---:|---:|---:|---:|---:|
| **SUPPORTED** (+1.5, (−1.5, +4.5)) | 0 | 0.558 | 0.417 | 0.994 | **0.305** | 0.519 |
| | 0.5 | 0.541 | 0.418 | 0.995 | 0.298 | 0.505 |
| | 1 | 0.487 | 0.419 | 0.993 | **0.276** | 0.458 |
| **UNDECIDED** (+1.0, (−1.5, +3.5)) | 0 | 0.474 | 0.376 | 0.995 | 0.240 | 0.434 |
| | 1 | 0.393 | 0.378 | 0.992 | 0.206 | 0.366 |

- **The exchange rate moves the joint** (SUPPORTED, σ_s 0.5): r = 0.5 gives 0.225, r = 1 0.299, **r = 2 0.400**,
  r = 3 0.442.
- **The design's own GUESS has net ≈ −0.6 at r = 1**: a U gain of ≈ +0.9 against an in-pool loss of −1.5. So the
  default rate is slightly stricter than the trade this doc predicts. That is the owner's to weigh.
- **At fixed truths** (σ_s 0.5):
  - Δ_F +1.5 with Δ_home72 −1.6 reads ADOPT-CANDIDATE 0.24 and EXCEEDED 0.05;
  - with Δ_home72 −3.0 it reads ADOPT-CANDIDATE 0.08 and EXCEEDED 0.17;
  - under the null it reads ADOPT-CANDIDATE 0.017.
- These replace the first draft's "P(GAIN) 0.52 / 0.65". That figure rested on the wrong branch's GUESS (+2.0), a
  closed form without AM-5, and a banked C0 (review M2).

### 5.4 REPORTED, never decided

Mechanism reads; each names what it would show under H-CTX.
- **M1, rc72b recovery:** X62 − C0 on rc72b, 20 units per final. rc72b is held out by six (§2), so this is Procgen's
  held-out-level read of the corpus's own generator. If X62 lifts rc72b but not F, it learned "recombined-team play",
  not human team play.
- **M2, the seen-set per-mon KO edge on F** (the forensics' statistic, from the captures; minutes of CPU). H-CTX
  predicts X62 raises it toward home72's +0.081.
- **M3, H-MID's Δbias** on X62's and C0's F and home72 captures, re-encoded and read through H-MID's own scripts. This
  is forward passes on CPU, ≈ a few CPU-h, and H-MID PREREG §3's fidelity gate applies. H-MID read
  CRITIC-MISCALIBRATED-ON-U (`f200258f`), so M3 is a registered SECONDARY (§6).
- **M4, H-TGT's switch-punish DiD** on F (forensics §2). X62 trains both sides, so it should move toward 0 if
  opponent-team breadth carries switch anticipation.
- **The dose line (review M3):** F at each final's 10M checkpoint (290M), 400 games a model, X62 and C0. Δ_F at 10M
  beside 20M is the cheapest dose read this fork can give. If C0's 290M checkpoints are not on disk, the line is
  X62 @10M against P\*, descriptive (FINDING 11).
- **The 86 withdrawn sets** (§2.2): each panel's share of slots carrying one, and P\*'s and X62's per-mon KO edge on
  those slots.
- **U, and goal (b)'s number**, with each panel's line.

**Predictions** (GUESS, written before rc72b; on the SUPPORTED branch after an N1-P NULL; UNDECIDED shifts Δ_F to
+1.0):

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F (primary)** | **+1.5 pp** | (−1.5, +4.5) |
| Δ_H / Δ_P72 | +0.75 / +0.4 | (−1.2, +2.7) / (−1.1, +1.9) |
| Δ_U | ≈ +0.9 | (−1.0, +2.7) |
| **Δ_home72 against Kakuna (the price)** | **−1.5** | (−4.0, +1.0) |
| Δ_S (REPORTED) | −1.6 | (−3.0, −0.2) |
| Δ_rc72b (M1) | +3.0 | (−1.0, +7.0) |

The basis (MY JUDGEMENT):
- F's home72 − F premium is 6.8 pp, and the forensics put ≈ ¾ of the per-mon premium with context, so ≈ 5 pp is
  addressable on F.
- A 20M plateau fork at share 0.5 that recovers ≈ 30 % of it gives ≈ +1.5. The first draft's 40 % had no stated
  basis. A 20M fork is ≈ 7 % of P\*'s 280M steps, and the survey's breadth results all trained broad for 10⁸ steps
  (review M3).
- D1, with ≈ 16 % recombined-like draws and its own population on F, read F +4.0 and H +2.1.
- The in-pool price is re-derived under H-CTX as ≈ D1's cost, because X62 halves the pool sixes' exposure as D1 did
  (review B1). The first draft's −1.0 used the set-level account's reasoning.

---

## 6. The decision tree

```
G0  The H-CTX read (rc72b, premium_recovery on ONE engine; R = rc72b − F, P = home72 − F)   ◄ expected ≈ 15:00 PT 10-27
 ├─ KILLED (R ≥ 0.75 P) ────────► SHELVE the X62 fork. The premium is set-level and X62 adds no set. Nothing is
 │                                 built for X62; X63's union drops the recombined third (its U2), and X64 does
 │                                 not run as designed (its gate 1). Record it here.
 ├─ INCONCLUSIVE (P's LB ≤ 0) or VOID ► HOLD, nothing built. The cheap remedy is more home72 / F control units at
 │                                 the same engine (CPU only); the orchestrator decides.
 └─ SUPPORTED (R ≤ 0.25 P) or UNDECIDED ► BUILD the corpus (≈ 1 agent-day, ≈ 1–3 CPU-h). One route; the prior
                                   differs: SUPPORTED GUESS +1.5, UNDECIDED +1.0 (§5.3). Then G1.
G1  PR-1′ (the strategy's precision read, re-scoped by REVIEW_2), when its class exists
 ├─ KILL (UB < +1.5) ───────────► X62's 20M FORK IS RETIRED: "breadth at the plateau, ≤ 20M" is KILL's scope, and
 │                                 X62 is a 20M plateau breadth fork. No exception is claimed. The corpus still
 │                                 serves X63 (≥ 100M, a dose KILL does not reach) and X64.
 ├─ BREADTH-GAIN ───────────────► no standalone X62 fork. The corpus joins the union (X63; the stack if the
 │                                 orchestrator still runs it). PR-1′'s GAIN raises X63's prior.
 └─ UNDECIDED, or NOT YET READ ─► G2. If X62 registers before PR-1′ reads, a later KILL does not void a registered
                                   read in flight (KILL retires FUTURE 20M forks), and X62's line is never pooled
                                   into PR-1′'s estimand after the fact.
G2  N1-P look 1 / final (§4)
 ├─ GAIN ──────────────► X62 REDUNDANT as a standalone fork. The corpus goes to X53-N1 (a later corpus arm) and
 │                        to X63's union.
 ├─ CONTINUE ──────────► wait for N1-P look 2.
 └─ NULL / LOSS / UNDECIDED-final ► REGISTER X62 by the PR rule at the tree pin (§5), C0 re-played there.
G3  X62's own outcome (§5.1, §5.2)
 ├─ ADOPT-CANDIDATE ───► team-context breadth works at the plateau at an in-pool price inside the rate. X53-N1's
 │                        corpus question reopens (X62 or the union), X63's GUESS rises, and X62 is a candidate for
 │                        the baseline proposal. Nothing joins the production recipe on one read.
 ├─ TRADE ─────────────► the owner rules, with net's interval and goal (b)'s U.
 ├─ OVER-RATE ─────────► not adopted. Reported with the rate; the owner may set a different r for LATER reads.
 ├─ NULL ──────────────► "context breadth at 20M on familiar sets does not move F by ≥ 1.5 pp". NOT retired for the
 │                        era: the leading explanation is DOSE (≈ 24 pilots a team, ≈ 7 % of P\*'s steps), and
 │                        plasticity read NO-LOSS (8bebdfe31). Route: X63 (≥ 100M on the union, pool ≤ 0.2), which
 │                        replaces the strategy's row 6 (fork extension) as the dose read; the dose line (§5.4) is
 │                        its first hint.
 └─ UNDECIDED ─────────► reported beside PR-1′'s family (never pooled after the fact); the union at dose (X63) is
                          the next read.
G4  H-MID (registered 2dd522785) routes the MECHANISM reads, never the go
 ├─ POLICY-CANNOT-ACT ──► the critic already sees the U deficit, so the policy carries the memorisation: M1 + M2.
 ├─ CRITIC-MISCALIBRATED-ON-U ◄ READ 2026-10-26 (f200258f): Δbias +0.076 [+0.016, +0.139]; V's ordering intact
 │                         (AUC 0.667 U vs 0.663 home72). This does NOT show the critic shares the memorisation:
 │                         a self-play critic had no signal to learn an opponent-relative premium (review M1).
 │                         M3 (Δbias on X62 vs C0 captures) IS a registered SECONDARY: does X62 shrink Δbias, the
 │                         part of the U deficit V does not see? Search: V's error is a between-set LEVEL shift,
 │                         which a one-ply comparison within a node cancels (H-MID FINDING 1); the direct test is
 │                         S-U (design_search_u_headroom.md), which competes with X62's read for the anchor lane.
 └─ INCONCLUSIVE ───────► no change; M1–M3 all reported.
```

**Why SUPPORTED and UNDECIDED route alike (review M4).** At X51's home72 0.598 and F 0.530, the registered rule
gives these verdicts (`power_checks.log` E; `revised_odds.log` C):
- at R / P = 0.25: SUPPORTED 0.36 and UNDECIDED 0.41;
- at R / P = 0.40 (the PREREG author's upper estimate): SUPPORTED 0.21 and UNDECIDED 0.51.

The split is a coin flip exactly where the truth is predicted to lie, so X62 does not hang a different action on it.
The cost is KILLED's false shelving: 0.07 at R / P = 0.4 and 0.21 at 0.6 (a premium mostly team-level, shelved one
time in five).

The alternative was to SHELVE only when R / P's lower bound is ≥ 0.5. It was rejected: at this read's size it
shelves 0.16 of the time even at R / P = 1 (`revised_odds.log` C), so the gate would have no teeth. A KILLED read
with R / P's interval reaching below 0.5 is recorded here as a FINDING for the orchestrator. The gate stays the
registered verdict.

**Why PR-1′'s KILL retires X62 and not X63.** KILL's registered scope is "breadth at the plateau, ≤ 20M, at these
corpora", and the strategy adds "no further plateau breadth fork". X62 is exactly that, so arguing that KILL tested
set-novel corpora would be an exception made after seeing the result. X63 is ≥ 100M: "KILL at 20M, GAIN at 100M" is
what the dose explanation predicts. X63 §9 makes the same distinction. REVIEW_2 M3 reads KILL as not reaching X62
either, because PR-1′ tested set-novel corpora and X62 is context recombination. This doc takes the stricter reading
for its own 20M fork, as the orchestrator directed for this revision: KILL's text covers every further 20M plateau breadth fork, and an
exception argued for one corpus would be made after the fact.

---

## 7. Costs

| item | GPU-h | lane-h (160–230 games / h) | agent-days | note |
|---|---:|---:|---:|---|
| corpus build + soak + tests | 0 | 0 | ≈ 1 | ≈ 1–3 CPU-h for the build (UNVERIFIED: the validator over ≈ 15k candidates; FINDING 4) + ≈ 30–35 min of soak, at nice 19 under `mem_cap.sh` |
| pin commit, kit arm, specs, PREREG | 0 | 0 | ≈ 1 | AM-N1P's pattern: an own-pin check, a kit test, two read specs |
| training, 5 seeds | **≈ 20.5** | — | — | 5 × ≈ 4.1 (11 sibling forks at this pin and length, 3.84–4.43; N1-P s3503 4.0). C0 shared: 0 GPU-h |
| **read, C0 fresh at the tree pin** | 0 | **238–342** | — | X62 5 × (3,480 F + 400 each of home72, H, P72, rc72b) = 25,400; C0 the same 25,400; the dose line 10 × 400 = 4,000; 54,800 games |
| read, C0's F banked at the tree pin by PR-1′ | 0 | 173–249 | — | PR-1′ at 30,000 games with C0 = 2 × an arm banks 3,000 F a C0 seed: C0 top-up 5 × 480 F + 5 × 1,600 = 10,400; 39,800 games |
| S, REPORTED | 0 | 0 (≈ 15 heavy-slot-h) | — | review m5, ESTIMATE; not in the rate; dropped first if the slots are short |
| M2 / M3 mechanism reads | 0 | 0 | ≈ 0.5 | captures parsed (minutes); M3 ≈ a few CPU-h of forward passes |
| **total, C0 fresh** | **20.5** | **238–342** | **≈ 2.5** | **10.8–15.1 resource-days; P(GAIN) 0.49–0.56 ⇒ 0.032–0.052 per resource-day; ADOPT-CANDIDATE 0.28–0.31 ⇒ 0.018–0.028** |
| total, C0 banked | 20.5 | 173–249 | ≈ 2.5 | 8.1–11.2 resource-days; GAIN 0.043–0.069 per resource-day; ADOPT-CANDIDATE 0.025–0.038 |

**The CPU lane is the binding resource**, as it is for the whole strategy tree. REVIEW_2 M2 puts ≈ 84,000–122,000
queued Kakuna games ahead of PR-1′, and X62's read counts in them. X62's read competes with the H-CTX read itself,
N1-P's look 1 remainder, D3 look 2, X48's K, X63's Stage A and X64.

**On the strategy's metric, X62 now sits below rows 4–9 (0.06–0.09)**, which are quoted on P(GAIN) alone at
≈ 107 lane-h. REVIEW_2 F3 notes that the same correction (the in-pool co-primary, and the real lane cost) applies to
those rows too. A rented CPU box (strategy §3.1, an owner question) would serve X62's read on the same approval.

---

## 8. Adversarial self-check

1. **"X62 is D1 again."** Partly true (§1.2). D1 was 31 % recombined-like, and its UNDECIDED +2.10 is the nearest
   evidence. What X62 adds is purity (no new set), clean panels (no population hazard on F) and 6× the teams. If D1's
   effect came from its forum population, X62 shows less on F than D1 did. If it came from context, X62 shows more.
   **The incremental information is real but modest.**
2. **H-CTX is about P\* PLAYING new contexts, not about TRAINING fixing them.** A SUPPORTED read says the premium is
   tied to sixes. It says nothing about whether 20M steps at the plateau can learn team-level play. **Plasticity is
   not the obstacle** (NO-LOSS, `8bebdfe31`). Dose is: ≈ 24 pilots a team at 20M. That is why NULL routes to X63
   and is not read as a retirement, and why the dose line is reported.
3. **The per-panel gradient (§1.3) fits the set-level account just as well**, so KILLED is a live outcome (GUESS:
   H-CTX SUPPORTED 0.30, UNDECIDED 0.30, KILLED 0.20, INCONCLUSIVE 0.20). The rc72b author predicted R / P ≈ 0.25–0.4.
   G0's power at that range is in §6.
4. **Recombination at ≤ 2 shared breaks the pool's cores** (309 set pairs co-occur in ≥ 5 pool teams). The new teams
   may be less coherent than human-built ones, and the network could learn "recombined-team play", which helps rc72b
   and not F. Teammate-lift weighting mitigates this, and M1 against F separates the two. A looser distance (≤ 3)
   keeps more cores, but it is a different lever and is not registered here.
5. **STRICT is a set-level change inside a context lever** (§2.2): 86 sets at half exposure, and set TVD 0.17. Under
   H-CTX it is small, and its cost side is clean (home72 holds none of the 86). It is reported per panel. The
   alternative (`validated`) would train on a class that every meter and the live server exclude.
6. **Share 0.5 keeps half the draws on the 719 memorised sixes.** If memorising the pool is the failure, a stronger
   form is the trainee side at share 1.0. That is a second lever, with a likely larger in-pool cost. It is listed,
   not designed: one lever per A/B.
7. **Family-wise false GAIN, and the shared control** (review m4). X62 is a fourth breadth arm against the same C0.
   At its own branch (N1-P not GAIN) its power barely moves (0.465 → 0.428 at σ_s 1). But false GAINs cluster under
   one control: X62's false GAIN given an N1-P GAIN rises from 0.038 to 0.104 at σ_s 1 (`power_checks.log` B). A
   GAIN read in two arms against one C0 is therefore not two independent confirmations, and the record says so.
8. **The post-hoc origin.** The forensics are descriptive and post hoc, and X62 was designed from them. The
   registered rc72b read turns this into a pre-registered chain. X62 is registered after G0, G1 (or PR-1′'s absence)
   and G2, and before any X62 game, with r written into the PREREG.
9. **The price could be gamed after the fact.** r is fixed at registration. A post-read change is an owner decision
   recorded as post hoc (§5.2), never a re-reading of the verdict.
10. **Costs could be low-balled.** The lane rate is the measured 160–230 games an hour, not the quiet-box 360. The
    build time is UNVERIFIED (FINDING 4). The fresh-C0 row is the headline, because no C0 bank exists at the tree
    pin yet.
11. **Is any of this needed if N1-P reads GAIN, or PR-1′ reads KILL?** No (§6): the fork is then redundant or
    retired, and the doc says so rather than finding a reason to run it. The corpus still serves X63 and X64.

---

## 9. FINDINGS (standing rule 7)

1. **D1's corpus is already 31 % recombined-like** (624 / 1,981; 997 at ≥ 4 pool sets and ≤ 2 shared). Every past
   reading of D1 as "human-team breadth" is partly a reading of team-context breadth with familiar sets.
2. **F holds 19 recombined-like teams.** A post-hoc split of X51's banked F games (P\* on those 19 teams against the
   other 53) would be a free, banked mini-H-CTX. At ≈ 100 games it would be ±10 pp, so it was NOT computed. It must
   never stand in for the registered read.
3. **The prototype is not the builder** (review F1). It skips the Node validator, the level pack, the engine scan and
   the label check. It emulates the One Boost Passer Clause and the STRICT class from the set text. Real acceptance
   rates come from the build.
4. **The build's wall time is unverified.** Node validation of ≈ 15k candidates has no measured rate here. The Python
   draw takes 25 s under the revised rules.
5. **The tree pin does not exist yet** (REVIEW_2 M1). Its option C is ≈ 0.5 agent-day plus 1–2 lane-h of identity
   units. Until it exists, every C0 cost here is the fresh row.
6. **The rc72b read is running.** It is expected ≈ 15:00 PT on 2026-10-27. This revision was written before it.
7. **"Familiar set" is the mon tier** (species + moves), as in the forensics. Because X62 copies blocks verbatim, its
   sets are also pool sets at the strict tier (item and spread).
8. **STRICT withdraws 86 of 658 pool sets and 6 of 69 species** from the corpus (§2.2), and set TVD rises from 0.073
   to 0.173. X63's recombined third and X64's corpus tiers inherit this. For X64 it breaks "the same 658 sets in every
   arm": the 86 sets' exposure falls with the arm's pool share. That doc carries the note.
9. **Amendment 2's open question is open here too**: does a no-IV-line Hidden Power play its named type in TRAINING
   (the validator's rewrite), or its IV-implied type? The answer decides whether §2.2's IV-completion alternative is
   the faithful one.
10. **The trainee side's sample-team bias** (review m7): 0.5 corpus / 0.45 pool / 0.05 sample. Whether the sample
    teams are home72 is UNVERIFIED. If they are, the home72 price is read on the most-practised teams.
11. **Whether C0's 10M checkpoints (290M) are on disk is UNVERIFIED.** The dose line's form depends on it (§5.4).
12. **The strategy's ranking table has no X62 row**, and its rows 4–9 carry the same overstatement (P(GAIN) alone at
    ≈ 107 lane-h; review F3, REVIEW_2 M4). This doc does not edit the strategy: the re-rank is the orchestrator's.
13. **The exchange rate is the orchestrator's default, not the owner's.** The owner has not ruled. The PREREG must
    record whose r it carries.
14. **This revision** ran two CPU scripts (`strict_draw_sim.py` 25 s, `revised_odds.py` < 1 min; nice 19 on the agent
    cores, no lease, `CUDA_VISIBLE_DEVICES=""`). It played no game, built no corpus and changed nothing under `data/`.

---

## 10. RESPONSE to the review (`d997a146`)

| finding | disposition | where / what changed |
|---|---|---|
| **B1** (BLOCKER): the co-primary is set up to fail under H-CTX (P(NI) 0.07 at D1's cost) | **FIXED** | §5.2. The in-pool cost is a registered PRICE: net = Δ_home72 + r · Δ_U against Kakuna, with r = 1 (an in-pool loss up to the U gain) as the orchestrator's default, changeable by the owner before registration. The FLOOR is home72 LB > 0.50. S is REPORTED with D1's NI bar beside it. Δ_home72 is re-derived under H-CTX as −1.5 (−4.0, +1.0). P(ADOPT-CANDIDATE) is quoted beside P(GAIN) everywhere (§0, §5.3, §7) |
| **B2** (BLOCKER): the tree ignores PR-1, whose KILL forbids this fork | **FIXED** | §6 G1, above N1-P (G2). KILL retires X62's 20M fork, with no exception claimed. BREADTH-GAIN sends the corpus to the union. UNDECIDED or not yet read goes to G2. Reconciled with X63: a 20M KILL does not reach a ≥ 100M dose, and the corpus serves X63 in every branch but KILLED H-CTX. The strategy row is FINDING 12 (the orchestrator's re-rank) |
| **M1**: Amendment 2 (rc72b) never reached the doc | **FIXED** | rc72b throughout. Radius-3 refusal of rc72b (491 → 0 teams at ≥ 3). The STRICT Hidden Power rule replaces `validated` (§2.2): 69 % of the first draft's teams carried a no-IV-line set, and the cost is 86 of 658 sets and 6 species, measured. Near-duplicates against D1, D3 and N1-P are reported (none ≥ 5). The open question is FINDING 9. X63 and X64 carry a minimal note |
| **M2**: the headline odds are 1.5–2.5× high | **FIXED** | §0, §5.3, §7: the run branch's GUESS, the PR rule with AM-5, and fresh C0 as the headline. P(GAIN) 0.56 / 0.49 and ADOPT-CANDIDATE 0.31 / 0.28. Per resource-day 0.032–0.052 (GAIN) and 0.018–0.028 (ADOPT-CANDIDATE); banked C0 is the conditional. D1's rule is dropped |
| **M3**: NULL blames plasticity, measured NO-LOSS | **FIXED** | G3 NULL routes to dose (X63), cites `8bebdfe31`, and strikes "retired for this era". N1-P's NULL row (§4) and self-check 2 say the same. The dose line (F at 10M) is REPORTED. The 40 % recovery GUESS is lowered to 30 % with its basis stated |
| **M4**: G0 cannot split SUPPORTED from UNDECIDED at R / P 0.25–0.40 | **FIXED (option 1)** | §6. SUPPORTED and UNDECIDED route identically; only the GUESS differs (+1.5 / +1.0), with both odds rows in §5.3. Option 2 (shelve on R / P's LB ≥ 0.5) was measured and rejected: 0.16 at R / P = 1 (`revised_odds.log` C) |
| m1: set exposure close to the pool's, not equal | **ADOPTED (report)** | §2.2, §2.4: both TVDs and the quantiles go into the manifest. No reweight (a second lever). Under STRICT the set TVD is 0.173 |
| m2: species-level distance unconstrained | **ADOPTED** | §2.1 step 3: an exact pool species six is refused (298 refusals, 0 left) |
| m3: false GAIN assumes σ_s known | **ADOPTED** | §5.1: 0.018 / 0.028 / 0.040 with φ̂ estimated |
| m4: the shared control correlates false GAINs | **ADOPTED** | §8 item 7 |
| m5: the S read is missing from the cost | **ADOPTED** | §5.2, §7: ≈ 15 heavy-slot-h, REPORTED, never delays the read |
| m6: an F GAIN is not goal (b) | **ADOPTED** | Δ_U enters the price (§5.2), U is goal (b)'s number (§5.1), and the predictions give Δ_U ≈ +0.9 (§5.4) |
| m7: the trainee side is not uniform | **ADOPTED (arithmetic) / FINDING** | §2.4 (0.5 / 0.45 / 0.05, ≈ 355 pilots); FINDING 10 |
| m8: the training corpora are not refused | **ADOPTED (report, not refuse)** | §2.4: none ≥ 5 shared with a D1, D3 or N1-P team. The unions re-check on their own builds |
| m9: the nickname is not a line | **ADOPTED** | §2.1 step 2: the head-line rewrite, gender and item kept, with tests on both forms |
| F1–F2: the prototype and the Monte Carlos are approximations | **CARRIED** | FINDING 3. `revised_odds.py` inherits F2's simplifications (one binomial panel per meter, normal seed noise) and adds its own GUESS model (§5.3) |
| F3: the strategy's per-resource-day column flatters breadth rows | **CARRIED** | FINDING 12 |
| F4: M4's table uses X51's cross-engine truths | **CARRIED** | §6 quotes it as such; rc72b's own home72 and F replace it once read |
| F5: the review changed nothing | — | — |

---

## 11. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-26 | **This doc CREATED (DRAFT, gated on H-CTX)** (research-design worker, orchestrator brief; docs + two CPU scripts) | X62: a P\*-fork at the X35 production shape (+20M, the `f4b25238` pin + a corpus-only commit), with `--team-mix x62_recomb_v1@ALL:0.5 --team-mix-sides both`: 12,000 teams of verbatim pool blocks, ≤ 2 shared with any pool team, a pool-matched species profile with Smogon teammate lift, static; read on F vs Kakuna against C0 on one engine; the gate tree keyed on H-CTX, then N1-P, with H-MID routing the mechanism reads | treating N1-P as this lever (28 % pool-set slots, 0.6 % recombined-like); an in-loop re-draw (≈ 24 pilots a team makes it a no-op); share 1.0 (a second lever); the then-current recipe (a new C0 for no information); a standalone fork on UNDECIDED H-CTX with N1-P GAIN | `piloting_gap_forensics_2026-10-26/` (`c758fcb82`); `hctx_rc72_2026-10-26/PREREG.md` (`caec2bb2c`); `n2_identity_test_2026-10-26/` (`fd4160d76`); `x61p_n1_production_2026-10-26/PREREG.md`; [`x62_design_2026-10-26/`](../research_state/measurements/x62_design_2026-10-26/) (`set_familiarity.log`, `power_pgain.log`) |
| 2026-10-27 | **G2 (now G4) resolved: H-MID read CRITIC-MISCALIBRATED-ON-U** (search-research design worker; docs only) | M3 a registered SECONDARY, worded as Δbias; search's pause replaced by H-MID FINDING 1 and a pointer to S-U | keeping "search stays paused" as an X62 consequence | [`h_mid_value_test_2026-10-26/README.md`](../research_state/measurements/h_mid_value_test_2026-10-26/README.md) (`f200258f`) |
| 2026-10-27 | **Two overstatements softened after the H-MID review** (doc editor; docs only) | "the premium goes with the TEAM" became OPEN until rc72b; "the critic shares the memorisation" became not shown | keeping both as the rationale | `../research_state/measurements/h_mid_value_test_2026-10-26/REVIEW.md` B1, M1, M3 |
| 2026-10-27 | **REVISED DRAFT after the adversarial review** (research-design worker, orchestrator brief; docs + two CPU scripts) | **B1:** the in-pool cost becomes a PRICE (net = Δ_home72 + r · Δ_U vs Kakuna, r = 1 the orchestrator's default, the owner's to change before registration) with a home72 FLOOR (LB > 0.50), S REPORTED. **B2:** a PR-1′ node above N1-P; its KILL retires the 20M fork, not the corpus (X63 / X64). **M1:** rc72b refused at radius 3, the STRICT Hidden Power rule (86 of 658 sets and 6 species withdrawn; the first draft's 69 % no-IV-line teams go to 0), near-duplicates reported. **M2:** P(GAIN) 0.56 / 0.49 and ADOPT-CANDIDATE 0.31 / 0.28 on the run branch; 0.032–0.052 and 0.018–0.028 per resource-day at fresh C0. **M3:** NULL routes to dose (X63), with a 10M dose line. **M4:** SUPPORTED and UNDECIDED route identically. Plus the minors (exact species six refused, the nickname rewrite, the trainee bias arithmetic); D1's read rule dropped; Kakuna at the tree pin | `validated` HP (trains a class every meter and the live server exclude); IV-line completion (not verbatim; depends on FINDING 9); NI at −2.0 pp as a decider; shelving on R / P's LB ≥ 0.5 (no teeth: 0.16 at R / P = 1); an argued exception to PR-1′'s KILL; D1's rule as a screen | [`reviews/x62_review_2026-10-27.md`](reviews/x62_review_2026-10-27.md) (`d997a146`) and its `review_2026-10-27/` logs; [`revision_2026-10-27/`](../research_state/measurements/x62_design_2026-10-26/revision_2026-10-27/) (`strict_draw_sim.log`, `revised_odds.log`); `REVIEW_2.md` (`69c7cd34d`) M1–M4, §6; the plasticity check (`8bebdfe31`); the survey (`5d554c23`) recs 1–2; X63 §9 |
