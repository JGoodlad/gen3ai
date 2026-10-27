# H-CTX A-K: does Kakuna, piloting both sides, find rc72b's recombined sixes weaker than coherent ones? (PREREG, 2026-10-27)

Registered before any game of this read. Parent read: H-CTX `hctx_rc72_am2_2026-10-26`
([`../hctx_rc72_2026-10-26/README.md`](../hctx_rc72_2026-10-26/README.md), SUPPORTED) and its adversarial review
([`REVIEW.md`](../hctx_rc72_2026-10-26/REVIEW.md), MAJORs M1 and M3). This read is the review's "cheapest
discriminating check".

Abbreviations: pp = percentage points; P\* = `rb_x26_s2601` @280,000,170; Kakuna = Metamon's Kakuna, checkpoint 34;
F = `forum72` (human-built forum teams); home72 = P\*'s own pool sample; rc72b = new sixes of verbatim pool sets
(≤ 2 sets shared with any pool team); COHERENT = home72 ∪ F; A-K = the open assumption that Kakuna's own play does not
change on rc72b; Elo = 400·log10 odds; −35 Elo = an expected score of 0.450.

## 1. The question and the three accounts

H-CTX found that P\*'s edge over Kakuna on its home teams (0.615) does not survive new sixes of its familiar sets
(rc72b 0.514; F 0.540). The review (M1) measured rc72b's sixes as less coherent than both controls (Smogon teammate
lift −0.005 against +0.20 / +0.14; pool species-pair PMI −0.22 against +0.08 / +0.05), so three accounts fit:

- **(a) memorised sixes:** P\* learnt its 719 teams as wholes.
- **(b) core knowledge:** P\* plays common species pairs (cores) well; F keeps them, rc72b breaks them.
- **(c) no plan:** a recombined six has no plan to play, so it is a weaker team in anyone's hands.

And (M3): both H-CTX seats piloted rc72b, so the drop mixes P\*'s loss with any change in Kakuna's own play on
recombined sixes (A-K). No neutral pilot has rated rc72b.

**This read:** Kakuna on BOTH sides, Kakuna-on-rc72b against Kakuna-on-COHERENT. Neither seat has trained on any of
these teams, so P\*'s familiarity is absent from the cell.

## 2. Predictions, per account (written before any game)

| account | what it says about this cell | predicted class (§4) |
|---|---|---|
| **(c) no plan** | a planless six is weaker for any competent pilot | rc72b **MATERIALLY WEAKER** (S well below 0.50) |
| **(a) memorised sixes** | the loss is P\*'s alone; Kakuna never saw the pool | **NOT MATERIALLY WEAKER** (S ≈ 0.50) |
| **(b) core knowledge, P\*'s** | the loss is P\*'s alone | **NOT MATERIALLY WEAKER** (S ≈ 0.50) |

**What each outcome licenses:**

- **NOT MATERIALLY WEAKER.** (c) loses its basis: the sixes play as well as human-built ones for a pilot that knows
  neither. A-K is bounded at the set level: Kakuna's own response to rc72b is under 5 pp. H-CTX's drop is then
  P\*'s, and (a) and (b) remain.
- **MATERIALLY WEAKER.** This supports (c)'s premise, and it is equally what a Kakuna that has ITSELF learnt human
  cores would show: (b) acting on Kakuna, which is A-K. This read does not separate those two.
  - It does not by itself make (c) the cause of P\*'s drop. H-CTX's cells are symmetric, so team strength cancels to
    first order; (c) explains the drop only if incoherence costs P\* more than it costs Kakuna (REVIEW M1).
  - The H-CTX panel was then partly a weak-team panel. X62's corpus uses the same ≤ 2 rule and would inherit the
    weakness.
- **UNDECIDED.** A-K stays open; the interval is reported as its bound.

**What separates (a) from (b): not this read.** Both say the loss is P\*'s; neither predicts a Kakuna effect. The
separating design is a NEW panel (REVIEW M3), registered here only as a prediction so it cannot be fitted later:
- **rc72c:** new sixes at the same ≤ 2-set radius from every pool team, whose species pairs are drawn with pool
  PMI > 0 to home72's level.
- **Predictions:** (a) says P\* plays rc72c like rc72b, with a recovered share ≤ 0.25 of the premium. (b) says P\*
  recovers most of it (share ≥ 0.75).

This read adds only a descriptive within-rc72b row (§5 D4).

**The author's own prediction:** S ≈ 0.47 (UNDECIDED, leaning weaker). This is a guess, from two facts:
- the review's lowest-lift third of rc72b is weaker as a team (its holder scores 0.44);
- Kakuna rates procedural Smogon-usage teams (P72) about 128 Elo below forum teams (TSR, 2026-10-18).

## 3. The cell, the tool, the plan

**Why this tool.**
- `main.reads` has no Kakuna-vs-Kakuna group shape: an anchor group needs a model of ours.
- The team-strength meter's Kakuna lane (`python -m main.team_strength`, pilot B) IS that cell: the anchors' declared
  asymmetric pair cell (`--opponent-a metamon:Kakuna --opponent-b metamon:Kakuna --our-teamset away --teamset home
  --declare-asymmetric-teams`), greedy against greedy, CPU.
- It attributes every game's two teams from the front end's capture (the START line's packed teams, matched by
  strict team identity), so Metamon's CSV off-by-one "Team File" column (REVIEW F1) cannot enter.
- **Rejected alternative.** Rating rc72b against the banked TSR reference (`tsr_ref24`), where home72 and F were
  rated on 2026-10-18. That crosses engines and pack regimes, and its per-set intervals were about ±80 Elo.

**The registration** ([`tsr_spec.json`](tsr_spec.json) → [`tsr_registration.json`](tsr_registration.json), frozen by
digest):
- **Side A** (the reported side): Kakuna on rc72b, all 72 teams.
- **Side B:** Kakuna on the REFERENCE = home72's 71 distinct teams + F's 72, 143 teams, one file each.
- **Panels:** the same files (by content digest) H-CTX played: rc72b `4feda9ef…`, home72 `e7f90019…`, F `efa72e52…`.
- **Units:** 60 units of 20 role-balanced games (1,200 games). Each unit is one `main.anchors` invocation at
  `--team-seed = --seed-base =` the plan's seed (registration seed 7277). Metamon draws each side's team uniformly
  from its set, so about 600 games meet each control.
  - A unit is 10 matchups played twice with the seats swapped (EXTERNAL_ANCHORS_SOP), so 600 distinct matchups.
  - 60 units, not the review's 20, because a matchup's two games can correlate: the per-unit SD may reach √2 times
    the binomial SD.
- **Guards:** `rederived_guards: count`. A forced end in any capture, a front-end mid-battle disconnect or a peer
  long-tail fault line voids the unit (SOP G6 / G2 re-derived, G14).

**Play.**
- **Pin.** A clean detached worktree at the first main commit that contains BOTH the F1 fix for both-external-peer
  cells (the coordinator's SHA, pending at registration) AND this registration's lane changes (port claims, the short
  lane, Kakuna-only registrations, re-derived guards). That is post-`b0441623e`, the new peer pack.
- **One engine:** every counted unit's `engine_stamp`, `peer_code` and `peer_pack_changed` must be identical
  (G7 / G14). The stamps are recorded.
- **How it runs.** One unit at a time (concurrency 1) on a claimed 95xx port, under `nice -n 19 taskset -c
  5-7,13-15 mem_cap.sh`, on the heavy slots' short lane (600 s; an overrun is retried once as a long job). The N1-P
  look-1 Kakuna lane keeps priority: this lane never runs a second unit and never bypasses the slots.
- **Attempts:** each unit gets at most 2 failed attempts.

## 4. The rule (one primary, one verdict)

**S** = side A's mean score over all counted games (win 1, tie 0.5, loss 0; the capture's `|win|` / `|tie|` decides).
Its 95 % interval is a percentile bootstrap over counted UNITS (each a 20-game cluster that holds both seat halves of
its team draws), B = 10,000, seed 7274. The 72 + 143 teams are fixed: the claim is about these panels.

**The bar is 0.450** (−35 Elo, TSR's registered materiality margin; ≈ 5 pp).

| class | condition |
|---|---|
| **MATERIALLY WEAKER** | upper bound < 0.450 |
| **NOT MATERIALLY WEAKER** | lower bound > 0.450 |
| **UNDECIDED** | otherwise |

**Rules that override the classes:**
- A bound within 0.001 of 0.450 is UNDECIDED (rule 8).
- Descriptive class, beside the verdict: **DETECTED WEAKER** iff the upper bound < 0.500.
- **INCONCLUSIVE** if fewer than 54 of 60 units count, or if the counted units span more than one engine, peer code
  or pack regime.
- **Registered stop:** 72 h after the first unit starts, the read is recorded with what has counted, under the same
  54-unit floor.

**Power** (1,200 games; SE 1.4 pp if binomial, 2.0 pp if the pairs fully correlate):
- if the truth is 0.50, NOT MATERIALLY WEAKER ≈ 90–98 %;
- if 0.40, MATERIALLY WEAKER ≈ 90–98 %;
- 0.45–0.48 mostly reads UNDECIDED.

Verdict line: `hctx_kk_2026-10-27: rc72b <CLASS> under Kakuna (S = x [lo, hi])`.

## 5. Descriptive rows (no verdict; written now so none is chosen later)

- **D1. By control.** S against home72's teams and against F's (each about 600 games, the same unit bootstrap), and
  their difference. Does Kakuna find the two coherent panels equally hard to beat? TSR: F − home72 = +7 Elo
  [−98, +110].
- **D2. Team sampling.** S with a two-stage bootstrap: side A's teams resampled, then units. It reads S as a claim
  about "recombined sixes" in general, not these 72.
- **D3. Elo form:** 400·log10(S / (1 − S)) with its interval.
- **D4. Coherence within rc72b** (REVIEW M1's metrics, recomputed by the record script with the review's formulas:
  Smogon teammate lift and pool species-pair PMI per team):
  - S by tercile of each metric;
  - the correlation of each rc72b team's Kakuna score with each metric.

  (c) predicts S rises with coherence. (a) and (b) predict nothing here.
- **D5. Bridge to H-CTX.** Each rc72b team's Kakuna-vs-Kakuna score (team strength under a neutral pilot), set against
  P\*'s banked per-team edge e(T) on rc72b (REVIEW C4, where the team's own strength cancels).
  - (c) predicts e(T) falls on the weakest, least coherent teams.
  - (b) predicts e(T) rises with species-pair PMI after controlling for that strength.
  - (a) predicts no gradient.
  - The review measured e(T)'s reliability at about 0.33, so this row is read as direction only.
- **D6. Integrity:** units counted and refused (with reasons), re-derived guard hits, the capture-to-record agreement,
  ties, game length, seat halves.

## 6. What this read does not do

- It does not test (a) against (b) (§2: rc72c).
- It plays no P\* game.
- It does not re-decide H-CTX; that verdict stands.
- One Kakuna checkpoint, one draw of rc72b (seed 7272).
