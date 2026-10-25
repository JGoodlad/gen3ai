# WHAT SEARCH NEEDS NEXT: a diagnosis of the off-self failure, what human data and Metamon change, and a ranked plan

**Status: DRAFT (2026-10-25), explicit-only, for adversarial review.** The owner's goal 5, "what search needs". Nothing
here is built or registered, and search stays PAUSED (`search_ledger.md`). Every number is cited to the doc or folder that
owns it. The derivations in §1.6 and §4 (the decision-to-game bridge, the identification target) belong to this doc and
are **UNVERIFIED** until experiment E2 runs. The only work done for it was reading committed artefacts; no CPU job ran.

**Abbreviations.**
- P\*: X26 @280M, the plateau policy. π: the searched policy. σ: the real opponent's policy. σ̂: the opponent model a search uses.
- IMP: the win probability gained per decision by playing search's pick instead of π's top choice (`search_ledger.md` §6.2).
- Worlds: O is the oracle (the true hidden facts); F is the 8 honest Smogon-drawn worlds; POOL is the training-pool teams
  consistent with what was seen.
- Leaves: LEAF-K8 is 8 greedy playouts to the end; TRUNC-4-K8 is 8 playouts of 4 turns, then the critic V.
- piKL: L1's anchored acting rule, argmax Q + λ·log π.
- HOM: the human opponent model. BR: R0-prod's best responder against P\*. PDL: the performance-difference lemma.

---

## 0. In one screen

1. **We have not measured search against a non-self opponent.** Every IMP in the ledger is scored by a referee whose
   playouts put the searched network (P\* at the plateau) on BOTH sides (`src/main/value_search/playout.py`: "Both sides play the forward's greedy action").
   So a "bot-stratum IMP" is a gain against our own network's continuation from a state that a bot game reached.
   - The one off-self failure actually measured is the JUDGE. The opponent's past choices identify its team only when
     the opponent model nearly is the opponent: off-pool, snapshots 0.794, bots 0.472, the HOM on bots 0.459.
   - Two of the four candidate factors, the opponent model inside the playouts and V's calibration against the real
     opponent, have never been varied (§1.2).
2. **What is measured, and does not depend on the opponent, is that the worlds bind.** Only identifying the opponent's
   TEAM pays: POOL +0.038 over F, on bots and snapshots alike. Every open-population sampler reads NOT DETECTED. The
   exact-world mass stays ≤ 0.001 (§1.3).
3. **The decision-level instrument may be far too coarse for the question that matters.** Search's game-level gain
   equals the sum of its per-decision advantages along its own games (the PDL, §1.6). With ≈ 26 eligible decisions a
   game:
   - a **+4.6 pp** game gain (S0b's bar) needs a mean IMP of only ≈ **+0.0018** per decision;
   - R0's **+11 pp** hole needs ≈ **+0.004**;
   - the ledger's honest reads have intervals of ≈ **±0.010** per decision.

   So the NOT-DETECTED reads (stage 1's open samplers, stage 1c's phase B) do not exclude game-relevant gains. The
   plain argmax's −0.012 at X26 may be a large game LOSS. The bridge is first-order and UNMEASURED; checking it is the
   first experiment.
4. **The R0 exploiter is the known-headroom testbed.** Its gain is general piloting of 5 offense teams, against an
   opponent that IS our network. It is the one setting in which the referee is the real environment, so it can do two
   things:
   - calibrate the bridge against a game gain already measured (+13.0 pp on the same deals);
   - ask how much of the exploiter's per-decision advantage oracle, POOL and honest one-ply search recover (E2).
5. **The ranked plan** (§3). Each experiment is one ply, offline, default OFF:

   | rank | experiment | cost |
   |---|---|---|
   | E2 | the bridge, and the exploiter testbed | ≈ 0.5 agent-day + 25–35 CPU-h |
   | E1 | the opponent-exact referee: worlds × leaf opponent model × leaf, on three opponents other than P\* that we can run in-core (the nine bots, the BR, the HOM) | ≈ 1 agent-day + 30–50 CPU-h |
   | E3 | human identification and the human judge, zero playouts, on held-out human replays and on P\* vs Metamon's LargeIL | ≈ 2 agent-days + 10–15 CPU-h |
   | E4 | a live one-ply search player against Kakuna; GATED on E1–E3 predicting ≥ +4.6 pp | ≈ 1–1.5 agent-days + 300–470 CPU-h |

6. **What "search pays" needs, quantitatively** (§4). Through the bridge:
   - ≥ ≈ +0.002 per eligible decision against the target opponent's own referee;
   - a few per cent of weight on the exact true world (≈ 4–11 % exact-team mass, or p ≈ 0.01–0.04 on the continuous
     leaf), not the old 20–35 %;
   - a resolution of about ±0.001: ≈ 4,000–9,000 bank states, or ≈ 600–900 paired games.

   Today's open samplers hold ≤ 0.001 of exact mass, still 40–100× short. On the ladder, the one route that reaches
   the target is team identification, and how often a human corpus identifies a human's team is the number nobody has
   measured (E3).

---

## 1. The diagnosis

### 1.1 What a one-ply search estimates, and where each error enters

For our action a at state s, against an opponent playing σ, with true hidden world w\*, the quantity that decides
strength is Q\*(s, a): the win probability after a, with our continuation π and the opponent playing σ, in w\*. Search
estimates it as

  Q̂(s, a) = Σ_w ρ(w) · L(w, a; σ̂, V, d)

where ρ is the weight on world w (proposals × the dice weight × the opponent-choice JUDGE), and L is the leaf: K greedy
playouts of a in w, the opponent side played by σ̂ (today σ̂ = π), cut at depth d and scored by V, or played to the end.
The pick is argmax Q̂, or piKL's anchored pick. Five things can make it wrong:

| factor | what it is | today |
|---|---|---|
| **W**, the worlds | ρ puts too little weight on w\* or on near-w\* worlds | F: Smogon draws; POOL on the pool bank only |
| **J**, the judge | the opponent-choice likelihood inside ρ, computed under a model of σ | π (P\*); the HOM in R-b2 only |
| **S**, the leaf opponent | σ̂ plays the opponent inside L | always π |
| **V**, the leaf value | the critic's error at the cut, including being trained on π vs our own pool | P\*'s critic |
| **H**, the horizon | one ply over a greedy π continuation cannot plan a line that exploits σ later | one ply, by the owner's scope rule |

The INSTRUMENT is a sixth thing. IMP's referee scores the pick with 24 playouts in w\*, with **π on both sides**. It
measures Q\* only when σ = π.

### 1.2 Which read varied which factor, and against whom it was scored

| read (owner) | W | J | S | V / leaf | referee's opponent | bank's opponents |
|---|---|---|---|---|---|---|
| Tier 1, E2, the maturity battery (`design_hidden_info_binding.md` §16.8, §16.12, §12.13.7) | O, F, FN, F+X, POOL | — | π | critic / TRUNC-4 / LEAF | **π** | bots + snapshots |
| stage 1 (`design_world_sampler.md` §6.2) | F, POOL, four open samplers | — | π | LEAF-K8 / TRUNC-4 | **π** | bots + snapshots (pool teams) |
| L1 (`../research_state/measurements/search_l1_l5_2026-10-20/`) | F, O | — | π | acting rules | **π** | Tier 1, E2, X26 |
| L3, off-pool, R-b2 (`../research_state/measurements/search_offpool_bank_2026-10-20/`, `../research_state/measurements/x50_r1b_hom_2026-10-23/`) | scored only | π, HOM | — | — | no playout (an AUC) | bots + snapshots (off-pool teams) |
| stage 1c (`../research_state/measurements/search_1c_2026-10-20/`) | F128L, C128L, M256L | π | π | piKL, LEAF / TRUNC-4 | **π** | the pool bank |

**FINDING D-1.** Against a non-self opponent, only J has ever been measured. S and V were never varied: σ̂ = σ = π in
every leaf and every referee. So "search fails against non-self opponents" is established for the judge alone.
- The world and acting-rule nulls are failures against everyone, including our own snapshots.
- No read has a referee in which a bot, a human or Metamon plays its own side.
- The R0 registration names the same gap for its live cells (F-R0-9: "neither models the opponent it faces").

### 1.3 Factor W, the worlds: BINDS (measured, opponent-independent under the self referee)

**The evidence.**
- The oracle minus honest gap: G +0.070 at 15M (Tier 1), +0.053 at X26 225M (maturity, LEAF-K8).
- Only team identification pays: POOL +0.038 [+0.022, +0.055] over F, r 0.72, "on bots and own snapshots alike". Every
  open sampler is NOT DETECTED (−0.008 … +0.002, ±0.015).
- The exact-world mass (`design_world_sampler.md` §6.1, §6.3) is ≤ 0.001 for every open arm (POOL 0.82) and
  ≤ 0.0003 for stage 1c's.
- The network's species belief is 0.31 exact at 225M; the SETS are what is missing.

**What the evidence does not say.**
- Whether partial correctness pays at a finer leaf. L2 read EXACT-ONLY at LEAF-K8, while the continuous TRUNC-4 dose
  rises from p = 0.05.
- How often an open corpus identifies a HUMAN opponent's team. Stage 1's corpus arms ran on pool opponents, whose teams
  the corpora rarely hold (λ ≈ 0.14–0.29).

**The cheapest separating measurement:** E3(a), a zero-playout identification screen on held-out human games.

### 1.4 Factors J and S, the opponent model: J BINDS off-self, S is UNMEASURED

**J (measured).** The truth outranks the honest worlds by the choice likelihood:
- snapshots: 0.794 off-pool (P\* as the judge);
- bots: 0.472, NO-EVIDENCE;
- the HOM on bots: 0.459, with ΔAUC against P\* of −0.013 [−0.068, +0.045]; on snapshots it falls to 0.706.

Two readings fit these numbers, and nothing has separated them:
- **(J-a) mis-specification.** Our models are wrong for scripted bots. Bots switch on 6 % of labels against humans'
  30 % (X50 R1), so a model of humans is the wrong judge of a bot, as F-1b-3 says.
- **(J-b) no evidence.** A scripted bot's choices carry little information about its own team.

The cheapest separating read is **L3-EXACT** (E1(c)): judge the bots' choices with the bot's own in-core policy. A
high AUC means J-a; an AUC near 0.5 means bots are a poor proxy for any opponent.

For humans, the open read is L3-H (E3(b)), with the IL-agent version E3(c). The HOM predicts held-out humans 0.50
nats better than P\* (unseen accounts), and R1 found X26's intent head worse than uniform on human switches. So J may well work
against humans with the HOM. R-b2 could not test that.

**S (never varied).** Every leaf plays the opponent's side with π, so we do not know how much a wrong σ̂ costs once the
world is right. Against a strong, near-self opponent it may cost little; against bots or humans it could cost a lot,
in either direction. The cheapest separating measurement is E1, which crosses W ∈ {O, F} with σ̂ ∈ {π, σ}. The tools
exist:
- the Rust playout core already plays an in-core scripted bot on the non-searched side (the `bot` key,
  `src/rust_env/src/search/playout.rs`, exercised in `src/utils/rust_env/successors_integration_test.py`). The search
  `LeafEngine` never passes it.
- a network opponent needs the lockstep's per-side dispatch (the `who` array already exists in `LeafEngine._lockstep`).

### 1.5 Factors V and H: V is UNMEASURED off-pool, H does not bind against self

**V.** The critic is a weak leaf even in-pool:
- oracle-world critic one-ply +0.026 against LEAF-K8's +0.081 (W1);
- CRITIC-INFO-LIMITED: the whole truth buys the critic search only +0.014;
- one or two turns of playout buy nothing over the critic (W3).

Against a non-self opponent V is also biased: it predicts π-vs-π outcomes, while P\* wins 96 % of the off-pool bot
stratum's states. An argmax is hurt only by ACTION-DEPENDENT bias, and that has not been measured. Neither has V's
calibration on our own unfamiliar teams, although P\* pilots home72 ≈ 11 pp better than unfamiliar teams against
Kakuna (X51). The cheapest measurement is E1's TRUNC-4 vs critic one-ply contrast against the real referee, plus V's
calibration and within-state ranking against that referee.

**H.** One ply over a greedy π continuation has large headroom against self:
- oracle LEAF-K8 +0.081, and K16 +0.098, still climbing (W2);
- at the plateau the oracle's edge falls (ΔIMP_O −0.021), but it stays large;
- L5 places Gen 3 OU near Skat, where one-ply determinization loses least.

So H does not explain the honest null. Two things are untested. Can one ply recover a KNOWN piloting hole? A greedy
continuation cannot plan a multi-turn line, only take the one-step best reply under π's continuation; E2(b) measures
that on R0's hole. And does H bind harder against a non-self σ, whose weaknesses the continuation cannot plan to
exploit? That waits on E1's exact referee.

### 1.6 The instrument: a self-play referee, and an unmeasured bridge to games

**The bridge (a derivation, UNVERIFIED).** Take a fixed opponent and a terminal win reward. The PDL (Kakade and Langford,
2002) gives

  J(π′) − J(π) = E over π′'s games of Σ_t A^π(s_t, π′(s_t)),

where π′ is π plus search and J is the win rate. Because π is greedy, IMP(s) = Q^π(s, a′) − Q^π(s, a_π) = A^π(s, a′),
as long as the referee is the real environment. So a game's gain is the sum of IMP over its decisions, read on π′'s
own states. To first order, with π's states standing in for π′'s, ΔWR ≈ N · Ī.

- **N**, the eligible decisions per game: the off-pool bank counted 8,047 eligible decisions (free move round, turns
  2–40, ≥ 3 legal) in 312 battles, so **N ≈ 26** (`states_offpool.json`). Ineligible decisions add an unknown share.
- **Ī** is the mean IMP at natural weights. The banks are loss-enriched, and re-weighting lowers IMP by ≈ 10–20 %
  (`search_ledger.md` §6.2).

**What follows if the bridge holds.**
- **The game-relevant per-decision effects are tiny.** +4.6 pp needs Ī ≈ +0.0018, and +11 pp needs ≈ +0.004. The
  ledger's honest intervals are ≈ ±0.010 (1c phase B, n 285), ≈ ±26 pp a game to first order.
- **NOT DETECTED excludes nothing that matters.** C128L's +0.007 [−0.003, +0.017] spans "useless" to "large" at the
  game level.
- **The first order cannot hold at large IMP.** The oracle's +0.081 × 26 > 1 must saturate: once search is winning,
  its later states carry little advantage.
- **The plain argmax's −0.012 at X26 is, to first order, ≈ −30 pp.** Its sign and scale agree with the pre-era read
  that depth-1 critic search cost 17–22 points of mirror win rate (N6; PRE-ERA, a different instrument, descriptive).

**FINDING D-2.** The bridge has never been measured ("a per-decision improvement is not a per-game win rate (never
measured)", `search_ledger.md` §6.2). Until it is:
- no decision-level NOT DETECTED can be quoted as "search does not pay";
- the decision-level bar for any search verdict should be stated in game units.

E2(a) measures the bridge against a game gain that is already known.

### 1.7 The R0 exploiter as a known-headroom testbed

**What we know** (`../research_state/measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md`, `r0/transfer/`):
- the BR (+20M from P\*, piloting a 5-team offense pin against P\* on pool draws) gains H +11.1 pp [+8.3, +13.8]; the
  learned part is +13.6, and the matched point is +13.0 (242 against 190 of 400 on the same deals);
- the gain is GENERAL: +14.5 pp [+7.6, +21.2] against C0, a P\* + 20M fork. So it is better piloting, within P\*'s family;
- it sits mostly on two teams, `ac17a9dd` and `b904dbe0`.

**Why it is the right testbed.**
- **The referee is exact.** The BR's start is P\* and its opponent is P\*. So the IMP referee (π = P\* on both sides) is
  exactly the environment of the BR-start vs P\* game. On the BR's own states the PDL then holds exactly, not to first
  order.
- **The headroom is known, and it is a piloting hole, not an information hole.** POOL is legitimate here, because the
  opponent draws a pool team.
- **The PDL lets the hole be written in decision units.** The BR's game gain is the sum of A^{P\*}(s, a_BR) along its
  games. So "does search close the gap" becomes a ratio of two per-decision advantages measured on the same states with
  the same referee.

**What the theory predicts.**
- With exact Q (the oracle, infinite rollouts), one-ply greedy improvement picks argmax_a Q^{P\*}(s, a), which is
  ≥ Q^{P\*}(s, a_BR) at every state. So the oracle one ply should recover ≥ 100 % of the BR's per-decision advantage on
  P\*'s states.
- Honest search, by everything in §1.3, should recover ≈ 0.

The informative cases:
- the oracle recovering much LESS than 100 %, which would mean that K8 noise or the referee's dice, not H, binds;
- POOL recovering a material share, which would show identification turning information into piloting gain;
- any honest share above 0.

---

## 2. What changed: human data and Metamon

**Human data.** The owner allows it as a toggleable lever (`human_data_architecture.md` §0, §6). Offline measurement may
read any snapshot, rowset or model. A live use needs a switch clause: `judge` left the grammar when R-b2 read
HOM-NONE, so reopening it is a new registration.
- The HOM exists: X26's network fine-tuned on 1.76M human choices, −0.582 nats against P\* on held-out humans and
  −0.503 on unseen accounts.
- The spectator snapshot `hr_spec_v1` (376,410 logs; train ≤ 09-20, held out 09-21..23) holds human TEAMS (partially
  revealed) and human CHOICES.

**Metamon.** Kakuna (P\* vs Kakuna: U 0.487 [0.459, 0.515], home72 0.598) is a strong non-self opponent at parity,
outside our lineage. SmallRL, LargeRL and LargeIL are human-only agents (FR-1 and FR-3 are playing). LargeIL imitates
humans, so it is the closest stand-in for a human opponent whose team we know exactly. None of them can play inside the
Rust playout core.

| ledger question (`search_ledger.md` §7) | answerable now? | how |
|---|---|---|
| Q5 / S0b: does one-ply search pay in GAMES? | yes, against a non-self opponent, after the live-player build | E4 against Kakuna (the goal-(b) gap ≈ 4.3 pp is a concrete game target) |
| Q15 / L3-H: do a HUMAN's choices identify a HUMAN's team, with the HOM as the judge? | yes, zero playouts | E3(b) on held-out human replays, with a future-reveal proxy truth; E3(c) on P\* vs LargeIL with full truth |
| Q12 on the ladder: which source puts weight on a human's true team? | yes, zero playouts | E3(a): the `hr_spec_v1` corpus as a team-identification source for held-out humans |
| Q14 / L4: does a calibrated but diffuse prior suffice? | yes, for a SYNTHETIC human population | E1's HOM stratum with the corpus worlds: the HOM pilots corpus teams in-core, so the prior is right by construction and the referee is exact |
| Q16 / R0's search cells: does search add to the exploiter or defend against it? | yes, at decision level, cheaper than the ≈ 213 CPU-h live cells | E2(b) (adds) and E1's BR stratum (defends, with σ̂ = BR as F-R0-9's opponent-aware leaf) |
| never on the list: does the leaf's opponent model matter? | yes | E1 (σ̂ ∈ {π, σ}) |
| any human-population IMP with an exact referee | **no** | a human cannot play inside the core; the HOM and LargeIL are stand-ins (FINDING D-6) |
| full hidden truth on human teams | **no** | the (c1) / (c2) intakes are not built (`human_data_architecture.md` §4) |

---

## 3. The ranked plan (preregistration-ready, NOT registered)

**The shared rules.**
- **Scope** (owner, 2026-09-30, one ply). Every arm branches only OUR action at the root. The opponent's simultaneous
  reply and every later move come from a policy (σ̂ or π), with no tree. Leaves are TRUNC-4-K8 or the critic.
  LEAF-K8 appears only as the oracle-ceiling instrument, as in every ledger read. Nothing runs in the training loop and
  nothing trains (FINDING D-10 asks the owner to confirm this reading).
- **Toggles, all default OFF** (PROPOSED). The leaf's opponent goes in the search config, alongside today's worlds and
  weights fields: `self` (today), an in-core bot, or a named checkpoint. The referee's opponent is a parameter of the
  measurement harness only. With every toggle at its default, a G-REPRO gate (as R-b2's) must reproduce banked rows
  bit for bit.
- **Statistics.** Battle-clustered bootstrap, 95 %, ties as half. A bound within 0.0005 of a bar takes the undecided
  class (root rule 8).
- **The bars** are in decision units, set from the bridge. MATERIAL = **+0.005** per decision (≈ +13 pp to first
  order). The game bar = **+0.0018** (+4.6 pp). After E2 reads the bridge, the bars are restated in E2's measured units
  before any later experiment registers.

### E2 (rank 1): the bridge, and the exploiter testbed

**What it decides:**
- whether IMP can be read in game units at all;
- whether one-ply search (oracle, POOL or honest) recovers a known piloting hole.

**The banks.** R0-prod's deals: the offense pin × pool draws, seed 20261020, the curve's procedure, greedy against greedy.
- **B-BR:** every eligible decision of the BR (k = 20M) side in the matched read's 400 games against P\*.
- **B-0:** every eligible decision of the BR's start (P\*'s own weights) on the same 400 deals.

Both are made by a patched trace generator, as the off-pool bank's `gen_cycle.py` was. **G-REPRO:** greedy play on
the Rust eval core is deterministic per (model, deal), so the re-played games must reproduce 242 and 190 wins exactly,
or the bank is void.

**(a) The bridge.**
- On B-BR, at every decision where a_BR ≠ a_P\*, the referee plays 24 + 24 playouts with P\* on both sides.
  Â = T(a_BR) − T(a_P\*).
- The prediction is the per-game sum of Â, averaged over games, with the ineligible decisions included by the same rule
  (their share reported).
- The contrast is D = prediction − measured, where the measured gain is +13.0 pp (242 − 190 of 400). Its interval comes
  from a bootstrap over deals that resamples both terms together.
- **BRIDGE-HOLDS** iff D's interval contains 0 AND the prediction's half-width is ≤ 6.5 pp. **BRIDGE-BROKEN** iff D's
  interval excludes 0. Otherwise **UNDERPOWERED**, and the referee's playouts double once.
- **First-order (secondary):** the same Σ Â read on B-0's states, which stands in for the first-order use. The ratio of
  the two is the state-shift correction for an improvement of this size.

**(b) Does search close the gap?**
- On 300 states of B-0, the arms are O, POOL and F, each TRUNC-4-K8 under piKL λ 0.015, with the argmax also reported,
  plus O at LEAF-K8. The referee plays P\* on both sides, which is exact here.
- **The statistics:** R_arm = Ī_arm / Ā_BR(B-0), the share of the BR's per-decision advantage the arm recovers, and the
  agreement P(pick = a_BR | a_BR ≠ a_P\*), against its chance rate.
- **SEARCH-CLOSES-GAP(arm)** iff lower(R_arm) > 0.5. Through the bridge that predicts ≥ half of the BR's game gain, and
  qualifies the arm for E4 against P\* on the pin.
- **ONE-PLY-CEILING-LOW** iff upper(R_O) < 0.5: even the oracle recovers less than half, so H or the leaf's noise binds
  for piloting holes.
- Otherwise PARTIAL.

**Predictions (GUESSES):**
- BRIDGE-HOLDS, with ineligible decisions 10–25 % of the sum;
- R_O ≈ 2–4, since the oracle's one-step gains exceed the BR's;
- R_POOL ≈ 1–2;
- R_F ≈ 0 [−1, +1];
- agreement O ≈ 0.35, F at chance.

**Cost.**
- ≈ 0.5 agent-day: the pinned-side trace generator and the bridge read.
- CPU: the bank battles ≈ 1–2 CPU-h; (a) ≈ 2,000 deviating decisions × 48 playouts ≈ 6–10 CPU-h, from ≈ 0.3 CPU-s a
  playout (the pilot's LEAF-K8 cost); (b) ≈ 15–25 CPU-h. All UNVERIFIED; the deviation rate of BR against P\* is
  unknown.

**Toggle.** None in play: these are measurement harness arms.

### E1 (rank 2): the opponent-exact referee, the decomposition

**What it decides:** for non-self opponents, how much each of S, V and the instrument costs, on top of W.

**The strata.** About 200 states each, drawn by the maturity battery's rule.
- **BOT:** P\* on pool teams against the nine in-core bots on `heldout72_hpfix` teams, LOSS-ENRICHED, so that ≥ 40 % of
  the states come from P\* losses. The off-pool bank was 96 % wins (its FINDING 1).
- **BR:** P\* against the BR on the offense pin, P\*'s decisions (R0's S-T setting).
- **HOM:** P\* against the HOM playing its side greedily, piloting completed human teams from `hr_spec_v1`'s train
  dates (a synthetic human).

**The arms.** W ∈ {O, F} × σ̂ ∈ {π, σ} × leaf ∈ {TRUNC-4-K8, critic one-ply}, under piKL λ 0.015, with the argmax also
reported. On the HOM stratum, W also takes C_human, the `hr_spec_v1` corpus posterior. The truth's team is drawn from
that corpus, and its unrevealed parts are C-GAME's Smogon fills, which the arm draws the same way. So the prior is right
in distribution but not exact: L4's calibrated-by-construction bank.

**The referee.** The real σ in-core, with 24 playouts per action, and P\*'s old self-referee on the same picks.

**The readouts.**

| quantity | contrast |
|---|---|
| S-term | IMP(·, σ̂ = σ) − IMP(·, σ̂ = π), in O and in F |
| W-term | IMP(O, ·) − IMP(F, ·) |
| V-term | TRUNC-4 − critic one-ply; plus V's calibration (mean V − T_real) and the within-state Spearman between Q̂ and T_real |
| instrument | IMP_real − IMP_self for the same picks |

**(c) L3-EXACT.** On the BOT stratum, the AUC of the truth against honest worlds, judged by the bot's own policy. Each
past choice gets the bot's in-core answer in world w, by frequency over 32 bot streams with ε = 0.02 smoothing. P\* and
the HOM are judged on the same states.

**The rules (per stratum).**
- **S-BINDS** iff lower(S-term in O) > +0.005: the leaf's opponent model is a lever, and the HOM as σ̂ against human-like
  opponents becomes an E4 arm. **S-SMALL** iff upper(S-term in O) < +0.005. Otherwise UNDECIDED.
- **SELF-REFEREE-MISREADS** iff the interval of IMP_real − IMP_self for F excludes 0 with |point| > 0.0018. The ledger's
  bot-stratum IMPs in `search_ledger.md` are then relabelled as self-referee reads.
- **L3-EXACT:** **EVIDENCE-EXISTS** iff lower(AUC) > 0.555 (R-b2's bar), which supports J-a; **NO-EVIDENCE** iff
  upper(AUC) < 0.555, which supports J-b.
- **Power.** The contrasts are paired: the same states and the same referee, differing only where the picks differ. At
  200 states a stratum they resolve MATERIAL effects (≈ ±0.008, UNVERIFIED), not the game bar; that needs §4's sizes.

**Predictions (GUESSES):**

| stratum | S-term in O | V-term |
|---|---|---|
| BOT | +0.004 [−0.004, +0.012], UNDECIDED | negative (the critic mis-ranks against bots) |
| BR | +0.001 | — |
| HOM | +0.003 | — |

- The instrument term on BOT is non-zero, with a correlation between IMP_real and IMP_self of ≈ 0.6.
- L3-EXACT AUC ≈ 0.65 (EVIDENCE-EXISTS).

**Cost.**
- ≈ 1 agent-day (PROPOSED build B-S1): the bot passthrough and the per-side forward in `LeafEngine`, a bot-answer query
  for L3-EXACT, the bank patches, and the HOM loaded as a player. The HOM has never played (FINDING D-5).
- CPU: the banks ≈ 3–8 CPU-h (bot losses are rare); the arms ≈ 3–4 CPU-min a state × 600 ≈ 30–40 CPU-h; L3-EXACT ≈ 1–2
  CPU-h. All UNVERIFIED.

### E3 (rank 3): the human screen, zero playouts

**What it decides:**
- whether a human team corpus identifies human opponents (the ladder's POOL analogue);
- whether the HOM judges a human-like opponent's choices (L3-H; it reopens or closes the judge for humans).

**(a) Identification.** On ≈ 1,000 sampled decisions of held-out games (09-21..23, actor ≥ 1500, unseen accounts as a
stratum):
- the corpus posterior over train-date teams consistent with the reveals so far (C_ladder's backoff method), with the
  same account's own games excluded and reported as a separate upper arm;
- the truth proxy: the facts revealed by the END of the game;
- the readouts: the exact never-seen species set mass; FCM, the mass on teams consistent with every end-of-game reveal;
  the hit rate on future-revealed moves; all against F and C_ladder (`hl_05_26`).

**(b) L3-H.** On the same states, the AUC of a future-consistent world against honest worlds, by the human's past
choices, under the HOM and under P\*.

**(c) The IL judge.** The same AUC on P\* vs LargeIL games, where LargeIL's team is known exactly, if the anchors
harness can record a replayable core log (UNVERIFIED, FINDING D-7).

**The rules.**
- FCM is an UPPER bound on the exact-team mass, because end-of-game reveals are partial. So the rule is asymmetric.
  **LADDER-CLOSED** iff upper(FCM_corpus) < 0.04, the bridge's lower identification target (§4): no human corpus can
  identify ladder teams well enough. **LADDER-OPEN** iff lower(FCM_corpus) ≥ 0.04 AND lower(FCM_corpus − FCM_F) > 0.
  That is necessary, not sufficient, and sends C_human to a playout read on E1's HOM stratum. Otherwise UNDECIDED.
- **HOM-JUDGES-HUMANS** iff lower(AUC_HOM) > 0.555 AND lower(AUC_HOM − AUC_P\*) > 0, on (b), or on (c) if (b) is
  INCONCLUSIVE. Then a new judge registration, with the switch clause the owner decides.

**Predictions (GUESSES):** exact species-set mass 0.15–0.30; FCM 0.02–0.06 (so the identification read is borderline);
AUC_HOM ≈ 0.60, AUC_P\* ≈ 0.53, so HOM-JUDGES-HUMANS.

**Cost.** ≈ 2 agent-days: the world builder on C-GAME-completed spectator logs, the future-reveal truth, and the corpus
index. CPU ≈ 10–15 h (the honest builds dominate).

### E4 (rank 4, GATED): a live one-ply search player against Kakuna

**The gate.** It runs only if an arm from E1 or E2 has an IMP lower bound, against the right referee, that through the
bridge predicts ≥ +4.6 pp.

**The player.** The anchors core slot (`src/main/anchors/core_side.py`) searches each decision with that arm: TRUNC-4-K8,
piKL, 8 worlds.

**The read.** Against Kakuna ckpt 34 on U and home72, paired with raw P\* on the same deals (X51's twins), greedy. The
rule is S0b's (`design_value_discrimination.md` §13.4): PAYS iff the paired lower bound > +4.57 pp.

**Cost** (UNVERIFIED). The player is ≈ 1–1.5 agent-days. A game costs ≈ 32 CPU-min (R0 §4). For 80 % power at
4.6 pp, with a paired discordance of 0.2–0.3, it needs ≈ 600–900 pairs, ≈ 300–470 CPU-h (≈ 4–7 days on the 3 agent
cores). An R0-sized 11 pp needs ≈ 100–150 pairs, ≈ 55–80 CPU-h.

**Its limit.** Today's honest builder takes the TRUE log in process (FINDING D-8). That is acceptable on our own server
for measurement, never for any public play.

---

## 4. What "search pays" would need to be true, quantitatively

All of this is UNVERIFIED until E2(a).

1. **The bridge holds within a factor of about 2** (E2(a)).
2. **A per-decision gain, measured against the target opponent's own referee.** With N ≈ 26 eligible decisions a
   game:

   | game target | Ī per eligible decision (natural weights) |
   |---|---|
   | +4.6 pp (the replicate floor, S0b's bar) | ≈ +0.0018 |
   | +4.3 pp (goal (b)'s gap to Kakuna) | ≈ +0.0017 |
   | +11.1 pp (R0's hole) | ≈ +0.0043 |

   The information headroom is G ≈ +0.053 at X26. So search needs to capture only **≈ 3–8 % of G**, without the
   argmax's harm. L1's anchor already removes that harm: −0.012 becomes ≈ −0.003 … 0.
3. **The resolution to see it.** 1c phase B gives n·SE² ≈ 0.007. Reaching SE ≈ 0.0009, so that a +0.0018 effect sits
   two standard errors from 0, needs ≈ **4,000–9,000** states: ≈ 70–150 CPU-h at ≈ 60 CPU-s a state (the referee only
   on deviations). A game read needs ≈ 600–900 pairs (E4). Today's banks hold 285–1,477 states.
4. **Weight on the truth.** Two estimates:
   - **Linear in the exact mass**, through the origin and POOL (+0.038 over F at q ≈ 0.82; stage 1's L2 read
     EXACT-ONLY): a lift of +0.002 to +0.005 over anchored F needs **q ≈ 0.04–0.11** exact-team mass.
   - **The continuous TRUNC-4 dose** (Tier 1, post hoc): +0.002 → +0.004 → +0.009 at p = 0 → 0.01 → 0.05. The same
     lift needs **p ≈ 0.01–0.04**.

   Today: ≤ 0.001 for open samplers, ≤ 0.0003 for stage 1c. So the requirement falls several-fold from the ledger's
   "0.2–0.35" (≈ 2–10× in exact-mass units), but the gap stays 40–100× (more for stage 1c). Only identification has
   reached it so far (POOL, 0.82, on our closed pool).
5. **Against a non-self opponent,** two further conditions:
   - the S-term must not cost more than ≈ 0.002, or must be repaired by a σ̂ (E1);
   - V's action-dependent bias against σ must not undo the gain (E1's V-term).
6. **The cost,** in the order the evidence needs it:
   - ≤ 60 CPU-s a decision is enough for measurement;
   - a cheaper world build amortised across a game (the honest screen dominates, ≈ 28–96 s a state) is what would make
     E4-sized reads routine. It is not designed here.

**The verdict this doc expects** (a judgement). The cheapest true statement may turn out to be "honest one-ply search
already pays a few points, and we could not see it". E2 decides that within a few days (≈ 25–35 CPU-h). If the bridge holds
and every honest R is ≈ 0, the only open route is identification. On the ladder that means E3(a)'s human corpus.

---

## 5. Hazards and FINDINGS

- **D-1.** No ledger read scores search against a non-self opponent: the referee plays the searched network on both
  sides. The bot-stratum IMPs are self-referee numbers (§1.2).
- **D-2.** The decision-to-game bridge is unmeasured. To first order, n ≈ 300 banks are ≈ 5× too coarse for
  game-relevant effects, so decision-level NOT DETECTED must not be quoted as "does not pay" (§1.6). The arithmetic is
  this doc's, UNVERIFIED.
- **D-3.** The leaf's opponent model has never been varied. The core supports a bot opponent in playouts; `LeafEngine`
  never passes it (§1.4). The build sizes in §3 are UNVERIFIED.
- **D-4.** Bot banks need loss enrichment (off-pool FINDING 1: 154 wins / 6 losses). Bots are weak, so a bot result says
  little about strength.
- **D-5.** The HOM has never played a game. Loading it as an in-core player is UNVERIFIED. It is not in the
  snapshot → rowset → model registry (rowsets and models: 0 / 0), so any read citing it must pin its archive path and
  checkpoint sha256.
- **D-6.** No human-population IMP can have an exact referee. The HOM and LargeIL are stand-ins, and E3's human reads
  are zero-playout only.
- **D-7.** Whether the anchors harness keeps a replayable core log of a Metamon game is UNVERIFIED. Metamon's agents
  cannot run inside the playout core.
- **D-8.** A live search player on today's builder needs the true log in process. Its residual leaks (seed-reuse speed
  ties in 20 % of decisions, hidden counters; `search_ledger.md` §6.1) make any live read optimistic.
- **D-9.** E3's truth proxy (end-of-game reveals) is partial and selected. About half of all games end in a forfeit,
  and long games reveal more.
- **D-10.** The scope reading (a 4-turn greedy-playout leaf as part of a one-ply evaluation, offline only; live only in
  the gated E4) is an interpretation of the owner's 2026-09-30 rule. The owner should confirm it.
- **D-11.** E2's POOL arm is legitimate only because R0's opponents draw from the pool, and the R0 transfer is within
  P\*'s lineage only. Neither transfers to the ladder.
- **D-12.** Unread here: the BR's deviation rate against P\*, the paired discordance for E4's power, and every cost in
  §3. R0's per-game ledgers live outside the repo (`~/gen3ai_archive/x35_r0_transfer/`).
- **D-13.** R0's registered live search cells (S-BR, S-T) stand as registered. E1 and E2 answer their questions at the
  decision level for a fraction of their ≈ 213 CPU-h, and change nothing in that registration.

## 6. Questions for the owner

1. Is §3's scope reading right (D-10)?
2. May the HOM PLAY in offline measurement (E1's HOM stratum)? If a later live arm uses it as σ̂ or as the judge, which
   switch clause should it take?
3. E4 needs ≈ 300–470 CPU-h. Should it crawl on the agent cores once its gate opens?

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **This doc CREATED (DRAFT), the owner's goal 5** | Four findings: the off-self failure is measured for the JUDGE only, since the referee plays the searched network on both sides; the worlds bind, whoever the opponent; the leaf's opponent model and V against a non-self opponent were never varied; the decision-level instrument may be ≈ 5× too coarse through the PDL bridge. The plan ranks E2 (the bridge + R0's known hole), E1 (the opponent-exact referee), E3 (human identification + judge, zero playouts) and E4 (a live player against Kakuna, gated) | re-running a world sampler at the decision level (it cannot resolve the game-relevant bar); R0's live cells first (≈ 213 CPU-h, ±8–10 pp); multi-ply or equilibrium search (out of scope, and ranked after a sampler); reopening the HOM judge on bots (wrong population) | `search_ledger.md` §1, §6.2, §7; `R0_REGISTRATION.md` §4 and its transfer read; the off-pool and R-b2 READMEs; `human_data_architecture.md`; `src/main/value_search/playout.py`; `src/rust_env/src/search/playout.rs` |
