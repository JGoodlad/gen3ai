# H-CTX (rc72b): adversarial review (2026-10-27)

**Under review:** the read `hctx_rc72_am2_2026-10-26` (record `79ef3e169`, [`README.md`](README.md)): **H-CTX SUPPORTED**.
rc72b recovers −0.34 [−2.05, +0.20] of P\*'s home premium against Kakuna (rc72b 0.514, home72 0.615, F 0.540 on one
engine; premium +7.5 pp [+2.9, +12.1]; 2,800 games). Also read: [`PREREG.md`](PREREG.md) and its Amendments 1–2,
`read_spec_am2.json`, the piloting-gap forensics (`c758fcb8`) and its review (`90708bff`), the X62 design (`58cb02dc`)
and the pre-commitment memo (`82f55d35`).

**Checks:** [`scripts/review_checks.py`](scripts/review_checks.py) → [`review_results.json`](review_results.json).
They use the banked games only and play nothing. The read state `~/gen3ai_archive/hctx_rc72_am2_read` is read-only
input. Cost: about 2 min of CPU at nice 19 on the agent cores.

Abbreviations: pp = percentage points; P = premium (home72 − F); R = recovery (rc72b − F); U = H, F and P72 with equal
weight (X51's unfamiliar set); A-K = the open assumption about Kakuna's own response to a panel; lift = Smogon
teammate lift (the builder's own coherence weight, as a log, 0 = a random pairing); PMI = how much more often two
species share a pool team than chance (as a log; below 0 = less often than chance).

## Verdict of the review

**The registered verdict stands. The numbers reproduce exactly, and no integrity defect was found.** The rule's
SUPPORTED also survives every numeric attack in the brief:

- the interval method;
- the legacy pack on F;
- the species mix;
- lopsided matchups;
- game length;
- the strict Hidden Power rule.

**What it licenses is narrower than the docs said.**
- **Licensed:** P\*'s home edge does not carry over to new combinations of its familiar sets. At most about a fifth
  of the premium goes with the sets: the upper bound is 0.20 against F. Against X51's U, read across engines, the
  share is 0.25.
- **Not licensed:**
  - that the premium is memorised per-team plans (win conditions);
  - that the levers "must" add team contexts.

**Why not.** rc72b's teams are measurably less coherent than both controls (M1). One account predicts the observed
rc72b < F and pure H-CTX does not: P\* plays common CORES (species pairs) well, and human teams keep those cores while
rc72b breaks them. The docs are corrected in place (§ Docs changed).

**BLOCKERs: none. MAJORs: 3. MINORs: 8. FINDINGs: 2.**

## MAJOR

### M1. rc72b's teams are less coherent than home72's and F's, so "team context" and "a broken core" move together

The README argues against weakness: "rc72b is drawn by home72's species profile from the pool's most repeated sets,
so its teams are not weak or strange in the way P72's are". By the builder's own coherence metric, and by the pool's
co-occurrence, that is wrong.

| per-team mean (C3) | rc72b | home72 | F |
|---|---:|---:|---:|
| Smogon teammate lift, log (0 = random pairing) | **−0.005** | +0.204 | +0.142 |
| pool species-pair PMI, log | **−0.219** | +0.076 | +0.055 |
| set pairs that share a pool team (of 15) | 4.2 | 15 | 3.1 |
| teams with both Spikes and Rapid Spin | 0.24 | 0.33 | 0.33 |

**Why.** The ≤ 2-shared refusal undoes the lift weighting. Any pair of sets that the pool's teams carry together
tends to push a third shared set over the limit. The draw is therefore steered toward pairings that the pool and
Smogon players avoid, which is X62 §9 item 4's "recombination breaks the pool's cores", now measured on the played
panel. rc72b's sixes are as coherent as a random pairing; F's human-built sixes keep their builders' cores.

**Three accounts fit SUPPORTED:**
- **(a) H-CTX proper:** P\* memorised its 719 sixes.
- **(b) Core-level knowledge:** P\* plays common pairs and cores well. F keeps them (PMI +0.055) and rc72b breaks them
  (−0.219).
- **(c) No plan exists:** a recombined six has no plan to play, the README's confound 4.

Account (b) predicts the observed sign, rc72b < F (R = −2.6 pp). Pure H-CTX predicts rc72b ≈ F, and so does (c)
unless incoherence costs P\* more than it costs Kakuna.

**The banked games do not support (c) or (b) as the whole story, but they are underpowered (C4).** P\*'s edge on a
team is its score over all games the team appears in, on either side; the team's own strength cancels in that score.
Within rc72b that edge barely moves with coherence:
- **by lift tercile:** 0.503 / 0.510 / 0.531;
- **by PMI tercile:** 0.49 / 0.54 / 0.52.

The most coherent third of rc72b (lift 0.06–0.47, about F's mean) sits at F's level (0.531 against 0.534), about
8 pp below home72. Two limits: the per-team edge reliability is only about 0.33, and lift predicts a team's strength
only weakly (r = 0.15 within rc72b). The lowest-lift third IS weaker as a team (its holder scores 0.44), and P\*'s
edge on it is no lower than on the rest.

**Consequence.**
- "The premium does not go with the sets" is measured.
- "It goes with the particular sixes (their plans)" is one of (a)–(b)–(c) and is NOT measured.
- **For X62**, whose corpus uses the same ≤ 2 rule and so trains on core-breaking sixes: an X62 gain on rc72b (its
  M1) is in-distribution for its own generator and is not evidence. Only F reads it. X62 §9 item 4 already says
  this; keep it binding.

### M2. The docs overstate the result; the registered rule bounds the sets' share, it does not set it to zero

**What was written:**
- the README's "goes with the TEAM, not the set";
- UNDERSTANDING's "the premium belongs to the particular pool teams (their plans)";
- design_beat_metamon_path's "the per-team win-condition account is now measured, not a hypothesis" and "the levers
  that reach goal (b) must add TEAM contexts";
- STATUS's "It goes with the TEAM".

**What the read supports (C1):**
- **Against F, on one engine:** R / P = −0.34, upper bound **+0.20** (percentile and Fieller intervals agree:
  [−2.06, +0.20]). The linear form of the band test, R − 0.25 P, is **−4.5 pp [−8.5, −0.5]**, so SUPPORTED holds
  without the ratio's instability.
- **Against X51's U, read across engines** (the forensics' own definition): the share is **(0.514 − 0.487) /
  (0.598 − 0.487) = 0.25**, right at the band edge.
- **The choice of F as the sole unfamiliar control moves the point estimate from +0.25 to −0.34.** F is the strongest
  unfamiliar panel against Kakuna, and P72's weak procedural teams pull U down.

So the sets keep "at most about a fifth (against F) to a quarter (against U)". That is close to the forensics' prior
of about 0.03 of 0.11 KOs per mon, and to the PREREG author's registered prediction (R / P ≈ 0.25–0.4, "SUPPORTED or
a low UNDECIDED"). The mechanism is M1's open question.

**Fixed in place** (§ Docs changed). The lever reading is "set breadth alone (D3) cannot reach more than about a
quarter of the premium", not "only team contexts can".

### M3. Kakuna's own response (A-K) is unmeasured, and the one check that would bound it was not played

**The confound.** Both seats pilot rc72b, so R mixes P\*'s loss of team context with any change in how Kakuna plays
recombined sixes. The README says it has no strength rating for rc72b. The forensics proposed exactly that cell (a
Kakuna-vs-Kakuna rc72b-vs-home72 pair cell), and the PREREG §7.2 dropped it.

**The likely direction is conservative, but that is a guess.** Kakuna learned from human replays, and rc72b's
pairings are non-human (lift ≈ 0). If Kakuna pilots them worse, rc72b's score is pushed UP, against SUPPORTED. That
is unverified.

**The cheapest discriminating check (≈ 3–6 Kakuna-lane hours, no GPU)** is the forensics' unplayed cell.

- **Design:** Kakuna on both sides, rc72b against home72, 20 role-balanced units (400 games). X37a's `kcal` cell is
  the precedent: it ran Kakuna on held-out teams against Kakuna on home72. Alternatively, rate rc72b with the TSR
  Kakuna lane (`python -m main.team_strength`) against `tsr_ref24`, where home72 (−17 Elo) and F (−10 Elo) are
  already rated.
- **If rc72b is NOT materially weaker** under a neutral pilot, account (c) loses its basis and (a) / (b) remain.
- **If it is materially weaker,** the read was partly a weak-team read, and X62's corpus inherits the weakness.

**Telling (a) from (b) needs a new panel** that keeps pool cores while breaking the sixes, for example new sixes
whose species pairs are drawn with pool PMI > 0 at the same ≤ 2-set radius. Within-rc72b gradients cannot do it at
this power (M1).

**Identify teams from the captures' START lines, never from Metamon's CSV** (FINDING F1).

## MINOR

1. **The interval method is sound, and the ratio is not the weak point.**
   - P ≤ 0 in 0.06 % of draws, so the ratio's upper tail is well defined; Fieller [−2.06, +0.20] matches the
     percentile interval.
   - **The bootstrap treats the 72 teams as fixed.** P\*'s edge varies across rc72b's teams (excess SD 0.066,
     against 0.014 on home72 and 0.030 on F).
   - **Adding a team-sampling term** (≈ 0.8 pp in quadrature) moves R − 0.25 P's upper bound from −0.5 to about
     −0.2 pp. That is still SUPPORTED for "recombined teams" in general, but only about 2 SE inside the band.
   - **Units are not independent draws.** The teams Kakuna draws in unit s are the teams P\* draws in unit s + 1, and
     both halves of a unit replay the same ten team pairings. φ ≈ 1 in every arm (0.81–1.08), so the effect is small.
2. **F's legacy pack (`b0441623e`) biases toward SUPPORTED, negligibly (C6).**
   - Kakuna held `sft431` (Frustration at base power 1) in 6 of F's 800 games; P\* won 5.
   - If all 5 should have been losses, F falls by 0.6 pp, and R / P rises from −0.34 to at most −0.24.
3. **The species mix explains at most about 1.3 pp of the 10.1 pp drop (C5).** A ridge logistic on both sides'
   species, fitted on home72 alone, predicts rc72b at 0.602–0.617, about home72's level; observed 0.514.
   - **Absent from rc72b:** seven of home72's species (Donphan, Misdreavus, Moltres, Raikou, Regice, Regirock,
     Weezing).
   - **Under-represented:** Metagross −9, Salamence −9, Dugtrio −7.
   - **Over-represented:** Swampert +8, Gengar +8.
4. **The sets are NOT a skewed subset in any way that matters (C3, C4).**
   - **Pool repeats are slightly lower:** log mean 2.85 against 3.01, median 18 against 21.
   - **No dose response:** within rc72b, P\*'s edge does not rise with them (r = −0.14).
   - **Sleep moves are three times as common:** 0.18 against 0.06 per team.
   - **The strict Hidden Power rule matches rc72b to home72 and F,** which hold none of the 86 withdrawn
     no-IV-line sets. It is not a confound of this verdict; FINDING 1 of the README stands on its own.
5. **No compression from lopsided matchups (C2, C4).**
   - **Team-strength spread is ordinary:** the excess SD of holder advantage is 0.079 on rc72b, against 0.069 on
     home72 and 0.088 on F.
   - **Game lengths are ordinary:** mean turns 42.4 / 42.2 / 47.3.
   - **Descriptive only:** rc72b's deficit is already present in short games (≤ 30 turns: 0.529, against 0.628 on
     home72 and 0.596 on F). That is unlike the forensics' "lost after turn 10" profile on U, but n is small.
6. **The README's FINDING 4 is wrong for home72.** "Every unit's capture is kept" holds only for rc72b and F. The 40
   home72 units ran without `--capture-dir` and have 0 captures. A post-hoc per-mon read cannot use this read's
   home72; it must use X51's, across engines. Corrected in the README.
7. **Integrity is clean.**
   - 140 / 140 units are counted, with 0 defaults, 0 re-decides, a 100 % argmax match and every peer clean.
   - The sixes of all 2,000 captured games match their panel files on both sides.
   - `decision.json` is reproduced to the last digit from the unit rows.
8. **No forking path.** One registered verdict. Amendments 1 and 2 changed the controls and the panel before any
   rc72 / rc72b game counted. Amendment 1's counted control units were discarded and the rule did not change. The
   secondary (cross-engine) bars agree.

## FINDINGS

- **F1 (tooling, out of this unit's scope).** In Metamon's per-battle CSV (`battle_log_*.csv`), the "Team File"
  column names the NEXT drawn team, not the one played.
  - **Checked here:** 692 of 800 games (`rc72b_a` and `forum72_a`) name the next draw. The rest are draws repeated
    back to back and the last game.
  - **The played team** is the peer report's `team_draws[i]`, checked against all 2,000 START lines.
  - **Where it bites:** `main.anchors.session.read_peer_battles` stores that column as `our_team` for peer-vs-peer
    cells (the `kcal` kind), so a per-team attribution from such a cell's `games.jsonl` is off by one.
  - **Not affected:** the TSR Kakuna lane and the forensics both match teams from captures.
  - It needs an owner (a test and a fix in `read_peer_battles`, or a refusal).
- **F2.** The rc72b read has no home72 captures (MINOR 6). Any post-hoc mechanism read pairs rc72b's captures with
  X51's home72 captures, which crosses engines.

## Docs changed (overstatement fixes, in place)

- **`README.md` (this folder):**
  - the verdict reading names the bound and the open mechanism;
  - the "not weak or strange" sentence is replaced by the measured coherence gap;
  - FINDING 4 now says home72 has no captures;
  - a review pointer is added.
- **`designs/research_state/UNDERSTANDING.md`:** item 4 and the §4 "Team against set" bullet. Sets carry at most
  about a fifth to a quarter; the mechanism is open (memorised sixes, broken cores, or no plan); this review is
  cited.
- **`designs/STATUS.md`,** row 3: the same correction in one sentence.
- **`designs/endstate/design_beat_metamon_path.md`:**
  - the status line and the §4 bullet: "must add TEAM contexts" becomes "set breadth alone reaches at most about a
    quarter";
  - a Decision-record row is added.
