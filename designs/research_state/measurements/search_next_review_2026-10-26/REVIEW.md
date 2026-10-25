# WHAT SEARCH NEEDS NEXT: ADVERSARIAL REVIEW (2026-10-26)

This reviews [`design_search_next.md`](../../../endstate/design_search_next.md) (DRAFT, `1d792221`), its pointer in
[`search_ledger.md`](../../../endstate/search_ledger.md), and the code it cites (`src/main/value_search/playout.py`,
`search.py`, `rules.py`, `src/rust_env/src/search/playout.rs`, `src/rust_env/src/bots/mod.rs`). The design was not
edited. Besides reading, I ran two CPU checks: one recomputes numbers from banked rows (§3, no new playouts), and one
is arithmetic (§4). Both ran at nice 19 on the agent cores.

## Verdict: SOUND-WITH-FIXES

**The diagnosis holds.**
- D-1 is true as stated: no IMP has ever been scored with a non-self opponent in the referee.
- D-2 holds in direction: decision banks of n ≈ 300 cannot rule out game-sized gains, even if N or the first-order
  step is off by 2×.

**The plan does not hold as written.**
- Two decision rules cannot reach their verdicts: E2(b)'s ratio (F6) and E4's power (F11).
- E2(a) tests a mathematical identity, not the bridge the rest of the doc relies on (F5).
- §0.2 credits the world finding with an opponent independence the instrument cannot show (F2).
- §4's identification target mixes up mass with presence (F3).

**Run first: E1-0**, a sub-cell of E1, which tests the doc's own headline (D-1) at almost no cost (F8):
- take the picks already banked by stage 1 / 1c / the maturity battery on the bot-stratum states;
- re-score them with the bot in the referee (the `bot` key that the core already has);
- set that beside the self-referee on the same picks.

The cost is about one build-hour and 1–3 CPU-h. Second comes E3(a), the zero-playout read of human team
identification. E2 and E4 should not register until F5, F6 and F11 are fixed.

| # | severity | finding (one line) |
|---|---|---|
| F1 | MINOR | Claim 1 verified in code and in the ledger. The only game-level search plays are pre-era MIRROR games (N6, the search-dividend battery, N1's void live read), all against self, and they should be cited |
| F2 | MAJOR | "The worlds bind, whatever the opponent" overclaims. Under the self-referee, W is the only factor still mismatched against the referee, and the full-length leaf rewards exactness by construction |
| F3 | MAJOR | §4.4's target treats average exact MASS as the dose. The dose is concave in the per-state weight, so the target must be E[f(p_state)], and E3(a)'s 0.04 bar inherits the error |
| F4 | MAJOR | The PDL arithmetic is right, but the ×N conversion predicts only the MIRROR game, and on states the mirror never visits. N is 21 against bots and 31 against snapshots, the bank weights are not natural, and the D-2 direction survives |
| F5 | MAJOR | E2(a) tests a theorem: on B-BR the PDL is exact, so D ≠ 0 can only mean referee ≠ environment. D's interval is ±6.9 pp from the game term alone, and "BRIDGE-HOLDS iff the CI contains 0" is equivalence by absence of evidence |
| F6 | **BLOCKER (E2(b))** | R = Ī_arm / Ā_BR. The denominator is ≈ 0.004 against a ±0.0097 half-width at n = 300, so R's interval is unbounded and E2(b) reads PARTIAL by construction for F. POOL ≈ O on pool draws |
| F7 | MAJOR | E1's S-BINDS / S-SMALL cannot be reached at 200 states (the point must be ≳ +0.017 or ≲ −0.007), and the doc's own predictions sit in UNDECIDED |
| F8 | MAJOR | The instrument term can be read on BANKED picks first (E1-0). A zero-playout check finds the self-referee ranks real BOT outcomes well at the state level (AUC 0.90), so the open risk is the action contrast |
| F9 | MINOR | L3-EXACT is close to a tautology for scripted bots, and its result says nothing about the ladder |
| F10 | MAJOR | E1's HOM stratum draws the truth's team from the corpus that C_human searches, so C_human is closed-population identification (a POOL analogue), not an open-ladder read |
| F11 | **BLOCKER (E4)** | E4's sizing is against 0 while its rule needs lower > +4.57 pp. The discordance of 0.2–0.3 is unsupported (N4: CRN decorrelates), and D-8's leaks make a PAYS uncreditable |
| F12 | MINOR | §4.3's "4,000–9,000 states" gives 50 % power at best. 80 % at +0.0018 needs ≈ 18,000 states (≈ 300 CPU-h), so the decision route is no cheaper than games |
| F13 | MINOR | D-10's reading is defensible, and the owner's 2026-10-20 "exploiter + pi search" is its precedent. But "each experiment is one ply" is loose wording |
| F14 | MINOR | E2's G-REPRO voids the bank on any forward or limit mismatch, and R0 counts a tie as a non-win where the doc counts it as ½ |
| F15 | MINOR | In E3(a), the opponent's own past games are a legitimate scouting source, not an "upper arm". Deduplicate battles across spectator logs |

---

## 1. Claim 1: "we have never measured search against a non-self opponent"

### F1 (MINOR): verified, with exceptions to cite

**The code agrees.**
- `LeafEngine` serves both sides from ONE `forward`. The `who` array is read only to find the searched side's
  decisions for the depth capture (`playout.py` `_lockstep`).
- `LeafEngine` never sends the `bot` key.
- The Rust core supports `bot: {side, name, seed}` (`playout.rs` header and `bot_from_json`), refuses a bot on the
  searched side, and gives branch b its own bot streams `stream_seed(seed, b, ·)`.
- `bots::Bot` holds only two RNG streams, so an in-core bot is memoryless apart from its dice. The real σ can
  therefore be re-created at the root exactly, which is good news for E1.
- **One caveat for E1's power:** the bot's dice are not common across actions, because each branch is an action and
  draws its own streams.

**The ledger agrees.** Every IMP is the §6.2 referee: 24 truth playouts, with π on both sides. Q16 and R0's F-R0-9
already named this for R0. D-1 is new only as a ledger-wide statement.

**The exceptions** (game-level search plays, all PRE-ERA, all MIRROR):
- N6: depth-1 critic search lost 17–22 points of mirror win rate.
- The 2026-09-11 search-dividend battery (`UNDERSTANDING.md`): 5,556 mirror battles; "naive search loses 74–80 % of
  games to its own unsearched self".
- N1's live read of 0.4719 is VOID, per the ledger's 2026-09-20 correction.
- W10 is Foul Play's search against our policy. That is a third party's search, not ours.

None of these is a non-self opponent, so claim 1 stands. They are, however, the only game-level evidence about the
sign that §1.6 infers for the argmax (F4), and the doc should cite them there in place of N6 alone.

## 2. Claim 3: the worlds bind

### F2 (MAJOR): "opponent-independent" is what this instrument cannot show

**The structure.** Under the self-referee, the leaf and the referee share σ̂ = σ = π, the same V and the same
continuation. Only W and the dice differ. An exact world therefore makes the leaf an unbiased estimate of the
referee's own target, and every inexact world estimates a different function. W is the one factor the instrument
leaves free, so W is the one it can find binding.

**Partly right worlds score near zero by construction.**
- LEAF-K8 scores each world through greedy playouts to the end of the game, and these decorrelate fast: N4 measures a
  correlation of 0.44–0.53 after one turn and 0.06 at the terminal.
- So "EXACT-ONLY at LEAF-K8" (L2) is the expected behaviour of the leaf, not a fact about opponents.
- The continuous TRUNC-4 dose (§5.1, rising from p = 0.01) is the evidence that partial correctness pays at all.

**POOL on pool-team banks is identification by lookup.** Its exact mass is 0.82, which makes it nearly the oracle.

**What survives:**
- the open samplers carry ≈ 0 exact mass;
- under the self-referee, only identification paid.

"Bots and snapshots alike" names where the STATES came from. The referee's opponent was π in both.

**Fix:**
- §0.2: change "does not depend on the opponent" to "binds under the self-referee, on states from either stratum".
- Make the TRUNC-4 dose the headline evidence on partial worlds.
- List W as unmeasured against a real referee, beside S and V.

### F3 (MAJOR): §4.4's identification target confuses mass with presence

**The two estimates measure different things.** The two §4.4 estimates (q ≈ 0.04–0.11 from a line through the origin
and POOL; p ≈ 0.01–0.04 from the TRUNC-4 dose) disagree by about 3×. They are not two estimates of one quantity:
- p in §5.1 is the weight on the truth in EVERY state;
- q_SM is the exact mass AVERAGED over states, and for the open samplers that is mostly zero states.

**The curve is concave.** On the TRUNC-4 dose:
- the slope from 0 to p = 0.05 is (0.009 − 0.002) / 0.05 ≈ 0.14;
- the slope from 0 to p = 1 is 0.036.

So mass spread thinly over many states is worth about 4× more per unit than mass concentrated in a few, and a target
written as a mean mass is not the right object.

**The ledger already says the threshold depends on the leaf** (§5.1: "that threshold is the 8-rollout leaf's
resolution"). The line through POOL is fitted on LEAF-K8, which carries the tie-break step (+0.019 at p = 0.001).

**Fix:**
- State the target as E_states[f_TRUNC4(p_state)] ≥ bar, and report the presence rate (the share of states whose
  proposal set holds the truth) next to the mean mass.
- Derive E3(a)'s LADDER-CLOSED bar from that, or make E3(a) descriptive until it is derived.
- Drop the "40–100× short" figure. It mixes the two units.

## 3. Claim 2: the PDL bridge

### F4 (MAJOR): the arithmetic is right, and it predicts only the mirror game

**The arithmetic checks.** 4.6 / 26 = 0.00177, 4.3 / 26 = 0.00165, 11.1 / 26 = 0.00427, and 0.010 × 26 = 0.26. N:
8,047 eligible decisions / 312 traces = 25.8 (`states_offpool.json`).

**The assumptions do not all hold.**
- **Which game.**
  - The PDL needs A^π for the opponent of the target game.
  - A self-referee IMP is A^{π vs π}, so N·Ī predicts π + search against π: the MIRROR game.
  - The banks are P\*-vs-bot and P\*-vs-snapshot states. Those are not the mirror's states, so the first-order step
    substitutes a distribution that neither game visits.
  - "≈ ±26 pp a game" is therefore a statement about a hypothetical mirror game read on foreign states. §4 is careful
    here ("against the target opponent's own referee"); §0.3 and §1.6 are not.
- **N depends on the opponent.** Per stratum, the off-pool pool gives 3,464 / 162 = 21.4 eligible decisions per game
  against bots and 4,583 / 150 = 30.6 against snapshots. R0 §4 cites 29–40 turns for P\* games. For the mirror,
  N ≈ 31, so the bar is ≈ 0.0015, not 0.0018. Turn 1, forced switches and turns past 40 are outside N, and search
  would act on them too.
- **Natural weights.**
  - The banks are not loss-enriched only: they are stratified (k bin × stratum cells, at most 3 states a battle).
  - The off-pool eligible pool's cell shares run 0.064–0.146, where the bank's are 0.10 each.
  - So Ī must be re-weighted by cell from the eligible pool's counts, not only by the "10–20 %" outcome correction.
- **Correlated decisions** leave the PDL's linearity intact. They only widen its variance (F5).
- **The argmax's "≈ −30 pp".**
  - The doc's own saturation argument applies in reverse: 26 × −0.012 is not a small policy change.
  - The pre-era mirror battery (−24 to −30 pp; F1) supports "a large loss" qualitatively.
  - "May be a large loss" is sound; the point value is not.
  - It is moot anyway: no one proposes the plain argmax.

**The direction of D-2 survives every one of these.** Even at N = 15, ±0.010 a decision is ±15 pp a game. What does
not survive is the precision of the bars (MATERIAL 0.005, game 0.0018): they are first-order MIRROR bars, and should
be labelled so.

**The zero-playout check (new; `~/gen3ai_archive/hidden_info_maturity`, rows `O` at X26 225M and 25M).**
- The referee's T for the action actually played (the bank's `rec_action`; seeds 16–39), against the real game's
  outcome:

  | checkpoint | bot states: AUC | n | snapshot states: AUC | n |
  |---|---|---|---|---|
  | 225M | **0.901** | 155 | 0.826 | 157 |
  | 25M | 0.844 | 387 | 0.805 | 392 |

- Selection on outcome does not bias an AUC. A level comparison would need the natural outcome rate of the eval
  cycles, which I did not read.
- At 225M, mean T is 0.84 in bot wins and 0.32 in bot losses. At the natural ≈ 96 % win rate against bots, the
  self-referee would under-predict P\*'s win LEVEL by ≈ 0.1 (an estimate).
- **Meaning.**
  - At the STATE level, the self-referee is not blind to how bot games go.
  - D-1's risk is the ACTION contrast: whether a move that is good against P\*-on-the-bot's-team is good against the
    bot.
  - That is exactly E1's instrument term, and it is what E1-0 reads first (F8).
- The method: join `bank/decisions.jsonl.gz` (`rec_action`, `outcome`, `opp_class`) to
  `rows/<cp>/O/shard_0_of_1.jsonl` by `id`, set T = mean(`out[rec_action][16:40]` == '+'), drop draws, and take the
  AUC by `opp_class`.

## 4. The experiments

### F5 (MAJOR): E2(a) tests an identity, not the bridge

On B-BR the BR's opponent is P\*, and the referee plays P\* on both sides. The doc itself says the PDL "then holds
exactly". A theorem cannot be refuted by data. D ≠ 0 can only mean the referee is not the environment:
- `max_turns` 999 in the playouts against the eval core's limit;
- the stall rule;
- ties scored ½ by the referee, while R0 counts a tie as a non-win (F14);
- dice handling, or the forward's device and precision.

That makes E2(a) a GIGO gate on the referee, which is worth having. It does not measure the two parts of the bridge
that are uncertain:
- **the first-order shift for a SEARCH policy.** B-0's secondary gives the shift for the BR, a policy learned over
  20M steps, not for one-ply search;
- **the self-to-non-self mapping**, which is E1's.

**The power** (two-sided 95 %, recomputed here):
- The game term alone, 400 deals at 0.605 and 0.475, has a 95 % half-width of ≈ **±6.9 pp** if outcomes are
  near-independent once the actions diverge (N4).
- The prediction's own half-width is ≈ ±4.4 pp, from n·SE² ≈ 0.0074 a decision × ≈ 28 decisions a game.
- So D's interval is ≈ ±8 pp, and BRIDGE-BROKEN can see only an error of more than ≈ 60 %.
- "BRIDGE-HOLDS iff D's interval contains 0" is equivalence by absence of evidence. The standing rule is that the
  delta's CI must lie inside the bar.

**Fix:**
- Re-label E2(a) as **G-REFEREE**: PASS iff D's CI lies inside ±6.5 pp, as a validity gate for any live player.
- The same calibration can be read more cheaply on B-0's own states: P\* vs P\*, T(a_P\*) against the realized
  outcome, with no deviations needed.
- Measure the first-order bridge for search where it applies: the searched player's own mirror games against P\*
  (S0b's shape). That needs the live player, so it belongs with E4, not ahead of it.

### F6 (BLOCKER for E2(b) as written): the ratio cannot be read

**The denominator is inside its own noise.**
- Ā_BR(B-0) is the BR's per-decision advantage on P\*'s states: ≈ 0.13 / 30 ≈ **0.004**, if its game gain spreads
  over its decisions.
- At n = 300, the per-decision 95 % half-width is ≈ **±0.0097** (stage 1c's n·SE² = 285 × (0.010 / 1.96)² = 0.0074).
- So the denominator's interval contains 0, and R's interval (Fieller) is unbounded.
- SEARCH-CLOSES-GAP (lower(R) > 0.5) and ONE-PLY-CEILING-LOW (upper(R_O) < 0.5) are both unreachable for any arm
  whose numerator is not large. E2(b) reads PARTIAL by construction for F, which is the arm that decides anything.

**The doc contradicts itself.** D-2 says n ≈ 300 banks are ≈ 5× too coarse, and E2(b) then registers a 300-state
ratio.

**Two of its arms are already known.**
- POOL holds the opponent's pool team (exact mass 0.82), so R_POOL ≈ R_O by construction.
- R_F ≈ 0 is what stage 1 and L1 already predict on pool banks.

**What is new here** is only whether the ORACLE one ply recovers a known piloting hole (the H question).

**Fix:**
- Drop the ratio.
- Read the oracle's Ī_O − Ā_BR as a paired DIFFERENCE, at LEAF-K8, where the numerator is large.
- Leave F and POOL out of E2, or size them for the game bar (≈ 18,000 states, F12).

### F7 (MAJOR): E1's per-stratum rules are unreachable at 200 states

**The rules need large points.** At n·SE² ≈ 0.0074, n = 200 gives a 95 % half-width of ≈ ±0.012. Then:
- S-BINDS (lower > +0.005) needs a point of about +0.017 or more;
- S-SMALL (upper < +0.005) needs a point of about −0.007 or less.

**The doc's own predictions are UNDECIDED.** For BOT it predicts +0.004 [−0.004, +0.012], and for BR and HOM +0.001
and +0.003. The paired "±0.008" is UNVERIFIED.

**Fix:**
- Make E1 two-stage. Measure the paired S-term's variance on its first 40 states, then fix n before any outcome is
  read: cost is not an outcome, the same re-size rule R0 §4 uses.
- Or register S-term as DESCRIPTIVE.

### F8 (MAJOR): read the instrument term first, on banked picks (E1-0)

`IMP_real − IMP_self` needs no new search:
- The maturity battery, stage 1 and 1c already hold every arm's pick on about 140–160 bot-stratum states a bank. The
  picks can be recomputed from the banked leaf rows, which are deterministic.
- The bot that played each state is named in its id.
- Re-score those picks (π's top choice plus each distinct pick) with 24 referee playouts, with the in-core bot on the
  opponent's side.
- The only build is passing `bot` through on the referee's path, about one build-hour.
- At about 0.3 CPU-s a playout, the cost is ≈ 1–3 CPU-h.

**What it decides.** Whether the ledger's bot-stratum IMPs must be relabelled (SELF-REFEREE-MISREADS), and whether
E1's full build is worth its day.

**Its resolution** is about ±0.015: it sees a gross misreading, not the game bar. That is the right first question,
and §3's state-level AUC makes a gross misreading less likely than D-1's framing implies.

### F9 (MINOR): L3-EXACT is nearly a tautology

A scripted bot judged by its own in-core policy makes the true world reproduce its choices by construction. In honest
worlds the drawn moves the bot would have preferred mark the gap. So EVIDENCE-EXISTS is close to certain for the
scripted bots, except RandomPlayer. The read separates J-a from J-b only formally, and no ladder judge has the
opponent's policy. Keep it as a 1–2 CPU-h sanity check, and do not let it steer.

### F10 (MAJOR): the HOM stratum's C_human leaks the truth's team

**The leak.** The truth's team is drawn from `hr_spec_v1`'s train-date corpus, and C_human is the posterior over the
same corpus. So the truth's species and revealed sets are in the proposal set exactly, and C_human can identify the
team by lookup, as POOL does. Filling the unrevealed parts from Smogon does not change that: the identifying facts are
the corpus's.

**What it measures** is "a calibrated prior over a closed population", L4's question. It is not the ladder's
open-population case, which E3(a) owns.

**Fix:** hold the truth's team (and its account) out of C_human's index for that state, or label the arm POOL-like
and never quote it as a ladder number.

### F11 (BLOCKER for E4 as written): sized for the wrong rule, and its positive is uncreditable

**The power is computed against the wrong rule.**
- The rule is PAYS iff the paired lower bound is above +4.57 pp. The sizing is 80 % power to detect 4.6 pp against
  0. At a true +4.6 pp, the power of the registered rule is ≤ 50 % at any n.
- Recomputed (two-sided 95 %, 80 % power):

  | discordance | against 0, 4.6 pp | against the bar, true 9 pp | against the bar, true 11 pp |
  |---|---|---|---|
  | 0.2 | 742 | 800 | 380 |
  | 0.3 | 1,113 | 1,200 | 570 |
  | 0.5 | 1,855 | 2,000 | 949 |

- The doc says 600–900 pairs and "100–150 for 11 pp".

**The discordance of 0.2–0.3 is unsupported.** Shared seeds misalign at the first differing action (N4: correlation
0.06 at the terminal). A searched game deviates on 10–20 % of decisions (R0's guess), so most pairs decorrelate and
the discordance approaches 2p(1 − p) ≈ 0.5. At 32 CPU-min a game, 950–2,000 searched games come to ≈ 500–1,070
CPU-h.

**The leaks (D-8) push the read upward,** by an unknown amount:
- the true seed's speed-tie dice in 20 % of decisions;
- the hidden counters.

So a PAYS cannot be credited to honest search. Only a negative is informative.

**The gate admits arms E4 cannot play.** σ̂ = σ needs Kakuna inside the core, and Metamon cannot run there.

**Fix:**
- Add DOES-NOT-PAY iff the upper bound is below +4.57 pp. Given the optimism, that is the informative class.
- Size against the bar, from the discordance measured on the first 50 pairs (fixed before any outcome is read).
- Treat PAYS as "leak-inclusive", and require the fresh-dice replay (§6.1's cure) before it counts.
- Restrict E4's gate to arms with σ̂ ∈ {π, HOM}.

### F12 (MINOR): §4.3's state counts

- SE 0.0009 needs 0.0074 / 0.0009² ≈ **9,200** states, and that puts +0.0018 two SEs out: 50 % power.
- 80 % power at +0.0018 needs SE ≈ 0.00064, ≈ **18,000** states. At 60 CPU-s a state that is ≈ 300 CPU-h.
- A decision-level read at the game bar therefore costs as much as E4. The doc's implied cost advantage of
  decisions over games does not exist at that bar. It exists only for MATERIAL-sized effects.

### F13 (MINOR): the scope, D-10

**The rule.** The owner's 2026-09-30 rule (memory `project_search_scope_one_ply`) caps IN-LOOP search and label
additions at one ply, and defers playouts to the end.

**Offline measurement is outside it, with precedent.**
- Every ledger read used TRUNC-4 or LEAF-K8 and a terminal referee offline.
- On 2026-10-20 the owner asked for "exploiter + pi search just to know" (memory `project_exploiter_headroom_r0`).
  That produced R0's registered LIVE piKL cells with a TRUNC-4-K8 leaf.

So a live, measurement-only player of the E4 kind has the owner's request behind it, and the doc should cite it.

**But the wording is loose.**
- "Each experiment is one ply" (§0.5) and "leaves are TRUNC-4-K8" mean root-only branching, a 4-turn rollout leaf
  and a referee that plays to the end.
- The memory's literal in-scope item is the one-ply V-bootstrapped Q̂.
- Any use of the TRUNC-4 leaf outside measurement (in the loop, or in public play) is a new decision for the owner.

D-10's question is the right one to ask. It should describe the leaf accurately.

### F14 (MINOR): E2's G-REPRO and the tie convention

- **G-REPRO.** Replaying the curve's 242 and 190 bit for bit needs the curve's own forward path and the eval core's
  limits. The device is not recorded in `curve_rb_x35_r0_s3531.json`. On a mismatch, re-play both files on the
  bank's path and use those counts (the gain is re-measured, not voided).
- **Ties.** R0 counts ties as non-wins (1 tie at k = 0), and the doc's statistics count a tie as ½. Use one
  convention in D.

### F15 (MINOR): E3(a)'s arms and hygiene

- **The opponent's own past games** are legitimate on the ladder (replays are public) and are the commonest human
  scouting route. Read them as a primary arm, labelled with that source, not as an "upper" arm.
- **Duplicates.** Deduplicate the same battle across spectator logs before splitting train from held-out.
- **The truth proxy.** D-9's caveat (partial, selected by game length) stands.

## 5. A plan order that survives these findings

| order | unit | why | cost |
|---|---|---|---|
| 1 | **E1-0**: the bot referee on banked picks (F8) | tests D-1 directly; decides whether the ledger's bot-stratum rows are relabelled | ≈ 1 build-hour + 1–3 CPU-h |
| 2 | **E3(a)**, with F3's target and F15's arms | the ladder's one open route (identification), zero playouts | as the doc |
| 3 | E1, two-stage (F7), C_human held out (F10) | the S and V terms, if E1-0 shows the instrument matters | as the doc |
| 4 | the live player + G-REFEREE (F5) + the mirror bridge | needs the player that E4 and R0's cells already need | ≈ 1–1.5 agent-days |
| 5 | E4, re-sized against the bar, with DOES-NOT-PAY (F11) | only after the leak cure, or read as an upper bound | ≈ 500–1,000 CPU-h |
| — | E2(b) | drop, or keep only the oracle difference (F6) | — |

Nothing here was registered, and this review changes no verdict in the ledger.
