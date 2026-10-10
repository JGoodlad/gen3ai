# Hand-computed features: what we compute for the network instead of letting it learn

**ALWAYS-CURRENT** ([`README.md`](README.md)). Created 2026-10-09 at the owner's request: *"put an end state doc where
we list what we are hand computing instead, so we know what we would add next or attempt to remove."* It is the ledger
of the project's INDUCTIVE BIASES: every quantity the model READS that our code DERIVES and hands it, instead of the
network learning it from the raw state through attention or its own layers. For each one it says why it exists, what
measured evidence it has, and what we would do with it next.

[`../ARCHITECTURE.md`](../ARCHITECTURE.md) states what the model IS and wins any disagreement with this doc; this doc
does not restate it, it classifies it. Verified against the code at `2e357971` (`arch_constants.py`, the extractor's
phase modules, `damage_op*.py`, `board_tokens.py`, `static_tokens.py`, `belief_tables.py`, `damage_tables.py`, the
Rust encoder `src/rust_sim/src/encoder/` and trackers `src/rust_sim/src/trackers/`); the FACT / JUDGMENT re-sort of
2026-10-09 (§1's classification test) re-read `intent_threshold.py`, `switch_branch.py`, `pair_outcome.py`,
`damage_op_blocks.pair_outcome_coords`, `damage_op._rolls` and `trackers/clock.rs` at `f0869884`. Tags: **MEASURED** (a number with
its source), **DESCRIPTIVE** (a measured number that is not a registered test), **UNVERIFIED** (never ablated, or not
checked here).

---

## 1. Principle

**The owner's architecture direction** (2026-10-05 / 2026-10-06, [`design_arch_audit.md`](design_arch_audit.md)
Decision record): STATIC per-mon tokens (who the Pokémon is) plus a small dynamic part (what is happening to it), with
*"the attention and the damage op doing the heavy lifting."* The network learns judgment; we hand it facts.

**We hand-compute a quantity when it is one of three kinds** (the letters are the inventory's "why" column):

| | kind | examples | why not learn it |
|---|---|---|---|
| **(a)** | **exact game physics** that is expensive or hard to learn | damage rolls, P(KO), speed order, accuracy, status-landing rules, sleep-wake odds, the Protect stall rule | the network would need ~10^5+ games to approximate arithmetic we can write exactly (audit F28); a learned approximation is also wrong in the tails, where the decisions are |
| **(b)** | **a fact that would otherwise reach the deciding token only through a narrow channel** | counts on the SIDE tokens, the op's per-mon amounts as token content, Spikes on our side | attention is a softmax AVERAGE (it cannot count), an edge bias is a RATIO (it cannot carry an amount), and with two trunk rounds a fact that arrives at the last round cannot be combined with anything (the static diagnostic's lesson, `measurements/static_diag_2026-10-09/`) |
| **(c)** | **a prior from Smogon** | move / item / ability / species / spread / speed priors | the opponent's hidden set is not in the observation; a prior is the starting belief. **It must be Smogon-derived, never pool-derived** (owner rule 2026-08-15): the 719-team pool may MEASURE structure but never ships as a prior |

**The classification test (owner-approved 2026-10-09).** Whether a quantity is a FACT or a JUDGMENT is decided by what
it IS, never by what it is for or how it is named:

| class | the test | consequence |
|---|---|---|
| **FACT** | the probability or magnitude of a GAME EVENT, in its own natural unit (an HP fraction, a probability, turns of a mechanic, hazard layers), determined by the rules + the board + beliefs (including the opponent's intent α). It is verifiable by simulating the game | hand-computed; kept |
| **JUDGMENT** | it (1) COMBINES different units with a weight we chose (an exchange rate), or (2) ASSUMES what WE will do on later turns (our own plan), or (3) applies a THRESHOLD that grades good / bad rather than describing an event | not hand-computed; the network's job. A judgment still in production is marked **J** and removed or replaced by facts |
| **APPROXIMATE FACT** (**AF**) | a fact computed with a shortcut (the KO ramp smooths the 16 damage rolls and omits the crit KO) | the fix is to make it EXACT, never to remove it |

**What we deliberately do NOT hand-compute: judgment and strategy.** A feature that says what the RIGHT play is (a
valuation, a threshold on when a move "pays") belongs to the network. **Facts** (what will actually happen if I press
this) are kept. The factual record: on 2026-10-06 the owner DECLINED, as judgments, `tempo_cost`, `wasted_ko` ("don't
click the KO into the obvious switch"), `neutralization`, `spin_value_lost`, the hazard stake of `spin_denied` and every
hand threshold (Focus Punch, Substitute, Endure, Endeavor) (audit Decision record, F11), and move resolution (§2.6) was
built on that list. **On 2026-10-09 the owner re-examined it** ("several rows marked J are actually FACTS, hard-to-compute
quantities handed to the model so it doesn't spend capacity on them") and adopted the test above; it re-sorts the list:

| quantity | 2026-10-06 | 2026-10-09 by the test |
|---|---|---|
| P8 `intent_threshold`: Focus Punch survives, Destiny Bond × P(KO), Substitute, Endure × P(KO), Endeavor × (1 − P(KO)) | "hand thresholds" | **FACT** (an event probability each; the Substitute ramp is an **AF**). KEEP |
| P6 `wasted_ko` = P(KO \| they stay) × α_SWITCH | judgment | **FACT** (an event probability), REDUNDANT with its two input columns (an interaction term). Remove only as a redundancy bisect, not as bias |
| P6 `spin_value_lost`, P3 `spin_denied`'s hazard stake | judgment | **FACT** (P(spin blocked) × hazard layers = the expected layers preserved). KEEP |
| P2 `neutralization` | judgment | **JUDGMENT as implemented** (a base-stat proxy for burn; paralysis 0.25 + 0.75·Δ outspeed is a weighted sum across units), though its intent ("the net effect of a status") is a fact. REPLACE with separate exact facts (§4 rank 4) |
| P2 `tempo_cost` | judgment | **MOSTLY JUDGMENT** (it assumes our cure plan). REPLACE with FACT flags (§4 rank 4) |
| O10 `turns_since_progress` | not classed | **JUDGMENT** (thresholds that grade progress). KEEP as removal rank 1 |
| D9 the KO ramp | not classed | **AF**. Make it exact (§4 rank 3) |

**Game cliffs stay probabilities**: a KO, a speed tie or a forfeit deadline is delivered
as a probability or a margin, never as a hand threshold (Destiny Bond's feature is P(the opponent KOs us this turn),
"no threshold", owner 2026-10-06; the clock gives the deadline as two REMAINING scalars, not a "near the end" bit; a
tie is a coin flip, never a rounding rule).

**A hand-computed feature is not free.** Each is a place a GIGO bug can live (the speed-stat, Beat Up, fixed-damage,
prior-denominator and ability-known bugs were all found in or under the operator, audit F28), a toggle, a version
field and tests. So every row below carries its evidence, and a row with none is marked **UNVERIFIED: never
ablated**. Dependence is not coverage (UNDERSTANDING P9): a low ablation reading says the model does not LEAN on a
feature, not that its fact has another home.

---

## 2. The inventory

**Columns.** *why* = (a) / (b) / (c) of §1, or **J** where the feature is a JUDGMENT by §1's classification test (it
should not be hand-computed); **AF** marks an APPROXIMATE FACT (a fact computed with a shortcut: fix it, do not remove it). *status* = **ON** (production builds and reads it) · **OFF** (built, production off) · **ARM** (built,
OFF in production, inside a registered screen) · **INERT** (built, consumes nothing). *action* = **KEEP** · **TRY
REMOVING** (one-lever ablation) · **TRY ADDING** (built or specified, not in production) · **REPLACE WITH LEARNED** (a
learned path exists or is proposed) · **REPLACE** (a better hand form exists).

**"The bundle"** below is the screen the owner set on 2026-10-08 / 2026-10-09: `--token-encoding static`,
`--move-resolution on`, `--speed-physics on` and `--op-reduction principled` against production as ONE arm,
NON-INFERIORITY, split on failure (audit Decision record 2026-10-08; `design_static_tokens.md` Decision record
2026-10-09, whose ruling also names "the critic route" — that this means `--value-threat-inject off` is
**UNVERIFIED** here). The static encoder alone is in its own registered screen (`design_static_tokens.md` §8.2),
at look 3 after look 2's CONTINUE (Δ̂ −2.70 pp static vs legacy).

Evidence pointers: **ARCH** = [`../ARCHITECTURE.md`](../ARCHITECTURE.md); **§4.1 op table** and **§5.4 edge table** are
ARCH's gen-3 audits (`run_20260807_135637_gen3` at 9.6M of 40M steps, 6,000 states, 2026-08-07: an EARLIER
generation, before the Baton Pass, prior-denominator, Beat Up and ability-known fixes, so STALE per audit F22);
**diag** = `designs/research_state/measurements/static_diag_2026-10-09/` (DESCRIPTIVE).

### 2.1 Observation facts (the Rust encoder derives them; every row is in the 2845-dim obs)

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| O1 | sleep-wake belief (3 / mon) | `[sleep_is_deterministic (Rest), p_wake, counter_reliable]`: P(wake on the next attempt) from the verified gen-3 tables (opp sleep 2–5, Rest 3, Early Bird halves), Early Bird marginalised over the Smogon ability prior when unrevealed | (a) (c) | `encoder/slot.rs::sleep_belief` · ON | correctness MEASURED (fuzz-calibrated against the sim RNG, `src/agents/observation/CLAUDE.md`); strength **UNVERIFIED: never ablated** | the gen-3 sleep RNG from a noisy counter, and Rest vs a random sleep at the same counter | KEEP |
| O2 | protect-success odds (1 / mon) | P(a Protect / Detect / Endure by this mon succeeds now): 1/2^k, floored | (a) | `encoder/slot.rs::protect_success_probability` · ON (also read by `c4`, `intent_conditional`) | **UNVERIFIED: never ablated** | counting consecutive Protects from history | KEEP |
| O3 | opponent Hidden-Power type block (17 / mon) | per species: the Smogon HP-type row, every type eliminated that could not produce an observed effectiveness bucket; restart from flat when the row is refuted | (a) (c) | `trackers/hp_belief.rs` · ON | correctness MEASURED (0 of the pool's 1,912 HP users carry a type their row gives 0; ARCH §1.2); strength **UNVERIFIED: never ablated** | inferring the HP type from effectiveness messages | KEEP |
| O4 | unrevealed-ability prior (in the ability block) | an unrevealed opponent's top-2 Smogon abilities + the top one's share, `known = 0` | (c) | `encoder/slot.rs::ability` · ON | **UNVERIFIED: never ablated** | species → ability | KEEP |
| O5 | recency (3 / mon) | turns since seen / acted / hit, turn-anchored, log-saturated at 10 | (b) | `encoder/slot.rs::recency` · ON | **UNVERIFIED: never ablated** | counting turns from the 32-event window, which forgets beyond 32 events | KEEP |
| O6 | last-action block (6, active only) | the side's last executed action: move id + was_switch / hit / miss / fail / crit | (b) | `encoder/slot.rs::last_action` · ON | **UNVERIFIED: never ablated** | reading the newest event rows (the event window carries the same action) | TRY REMOVING (low: a likely duplicate of the event window; check coverage first) |
| O7 | pair-history tendencies h[i,j] (6 × 6 × 5) | per (their mon i, our mon j): switch-ins, attacks, status clicks by i while j was active, shared-field turns, pairing recency; log-saturated at 10 | (b) | `encoder/mod.rs::pair_history`; read ONLY by the `h` edge · ON | **UNVERIFIED: never ablated** (not in the §5.4 edge table, which predates it) | opponent tendencies from history (the event window is 32 events) | TRY REMOVING (one-lever: drop `h` from the families string) |
| O8 | event-record derived columns | per event row: the attributed `hp_delta`, faint CAUSE, item TRANSITION, ACTION DENIAL rows (the gen-3 turn cut), entry reason, REL mon, Spikes layers at the event, Pursuit-on-switch | (a) (b) | `trackers/history.rs` · ON (via the event seats) | owner requirement E12 (`obs_enrichment_backlog.md` §1a); the derivations are correctness-gated (`window_record_test.rs`); strength **UNVERIFIED: never ablated** | parsing raw protocol lines; residual damage is not an event at all | KEEP |
| O9 | deadline clock (3) | `log(1+turn)/log(251)`, `(250 − turn)/250`, `log(1+250−turn)/log(251)` | (b) | `encoder/mod.rs::global_env` · ON | motivation MEASURED pre-fix (`ai_v9_09` @16M: a positive V before a forfeit in 13 of 14 timeout losses; ARCH §1.4); the fix is **UNVERIFIED: never ablated** | resolution at the forfeit cliff from a single log-elapsed scalar | KEEP |
| O10 | `turns_since_progress` (1) | a no-progress counter, log-saturated at 10, reset by a HAND definition of "progress" (eight reset conditions): our move's own hit ≥ 3 % and the target's net fall ≥ 3 %, a status on them, a new Spikes layer, they switched, a winning residual, a boost, a new Substitute, a successful Wish; with Rest-loop, wasted self-cure, heal-freeze and other carve-outs | **J** (thresholds that grade progress) | `trackers/clock.rs` · ON (global token / FIELD token) | **UNVERIFIED: never ablated.** It was the deleted shaped reward's tax input; it survived as an obs scalar | judging whether a turn achieved anything | **TRY REMOVING** (§5 rank 1). If a clock signal then proves needed, the threshold-free FACT alternative is "turns since either side lost HP" |
| O11 | Wish pending (1 / side) | a flat 0.5 (gen-3 Wish heals the recipient's maxhp/2) when a Wish resolves this turn | (a) | `encoder/mod.rs::board` · ON | **UNVERIFIED: never ablated** | remembering a Wish across two turns | KEEP |
| O12 | weather turns-left + permanent bit | `max(0, 5 − turns_active)/…`; Drizzle / Drought / Sand Stream permanent | (a) | `encoder/mod.rs::global_env` · ON | **UNVERIFIED: never ablated** | counting weather turns | KEEP |
| O13 | the OBS-FACTS block (84) | `seen` (what the opponent has seen of each of our mons), `choice` (evidence AGAINST their Choice lock), `vol` (Encore / Taunt / Disable / Uproar / partial trap: elapsed, min left, max left), `screens` (turns left) | (a) (b) | `encoder/facts.rs`; read only under `--obs-facts v1` · OFF | **UNVERIFIED: never trained** | the opponent's knowledge of us, the lock, volatile and screen timing (no other source exists: entity audit §4 A5) | TRY ADDING (§4 rank 6) |
| O14 | move-slot dex features (in the per-mon slot) | `power/200`, has_secondary, has_recoil, category, accuracy, never-miss bit, max PP | (a) | `encoder/slot.rs::moves` · ON | **UNVERIFIED: never ablated** | move semantics from the id embedding alone | KEEP |

### 2.2 The per-mon token (legacy production, and the static arm's S + D)

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| T1 | `MOVE_ATTR` into the move latent | per move: BP/200, physical / special / status, accuracy, never-miss, priority, drain, recoil, per-status secondary chances, utility flags, then a learned MLP with the move and type embeddings | (a) | `damage_tables.build_move_attr`, `encoders.MoveLatentEncoder` · ON (`--move-latent`) | **UNVERIFIED: never ablated** | which moves are mechanically alike (Rock Slide ≈ Hidden Power Rock) | KEEP |
| T2 | legacy: board facts broadcast into every mon and move encoder | clock, weather, faint counts, Spikes and screens fed to all 12 role encoders and 48 move encodings | (b) as a ROUTE | `PokemonEncoder` · ON (`--token-encoding legacy`) | the static screen removes it: look 2 Δ̂ −2.70 pp static vs legacy (CONTINUE; not isolated from stage 2); diag: our-side Spikes after the trunk R² 0.33 static vs 0.46 legacy, the one gap that grows with training (DESCRIPTIVE) | attention delivering board context to each token | REPLACE WITH LEARNED (the static screen, `design_static_tokens.md` §8.2) |
| T3 | E2 active-context injection | each side's boosts + volatiles (60) scattered onto its ACTIVE mon's row | (b) as a ROUTE | `PokemonEncoder` (legacy); D under static · ON | **UNVERIFIED: never ablated** (the v76 dedup removed the head concat on measurement, ARCH §3.1) | attention from the board token to the active | KEEP (it is per-mon state, D's home under static) |
| T4 | static S: actual L100 stats | ours: the gen-3 stat formula on the observed spread; theirs: the Smogon usage-weighted mean ± std of each realised stat; ÷ `STATIC_STAT_SCALE` 500 | (a) (c) | `static_tokens.py`, `belief_tables.build_static_stat_prior` · ARM | part of the static arm only (look 2 above); not isolated | stat arithmetic from base stats + a raw spread | KEEP |
| T5 | static D: per-move PP / legality, matched by move-num identity | `Σ_k ReLU(W[m_k; pp_k; legal_k])` with legality crossed from request order to sorted order by identity | (a) | `static_tokens.py`, `extractor_ctx.active_move_legality_sorted` · ARM (the identity rule is ON in both encodings) | the identity rule fixed a wrong legality input on 6.8 % of real move-bearing decisions (F-ST-1, MEASURED) | which request slot is which move | KEEP |
| T6 | `prefuse_proj`: the op's incoming rows as our tokens' content | each of our mons' incoming per-mon row (12) added to its token, zero-init | (b) | `extractor_forward.py` · ON | **UNVERIFIED: never ablated in isolation** (the §4.1 probe zeroed the concat only) | the amount of damage each of our mons takes, as content rather than an edge ratio | KEEP |
| T7 | static OPC: the op's per-mon amounts on BOTH sides | `amount_proj`: each mon's `x` ⊕ `g` cells (entry chip, Pursuit exposure, grounded; the end-of-turn ledger); on each of THEIR mons the `d1` cells of our active's moves as a SET (a shared bias-free per-move `Linear(6, 32)` + ReLU summed, then the zero-init `outgoing_proj`; since `gen3_static_port_v1` — the request-ordered `Linear(24, 128)` before it gave each request slot its own weights) | (b) | `board_tokens.OpContent` · ARM | diag: the entry-chip column of `amount_proj` grew to weight norm 2.7–3.6 across the five static seeds vs ~1.0 for the weather chip (DESCRIPTIVE: a weight norm, not an ablation) | amounts the trunk can only see as ratios (entity audit §4 A3) | KEEP |
| T8 | `op_worst_proj`: the noisy-OR KO worst case on our tokens | per our mon `[P(some physical move of theirs KOs it), P(some special move KOs it)]`, zero-init `Linear(2, 128)` | (a) | `op_reduction.py` · ARM (`--op-reduction principled`) | strength **UNVERIFIED** (in the bundle) | the worst case the hard max used to stand in for | KEEP pending the bundle |

### 2.3 Board / side / field tokens

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| B1 | legacy global token + the `non_matchup_rest` head concat | `[our_ctx 60, opp_ctx 60, non_matchup_rest 25] → global_proj`; the same 25 scalars concatenated straight into the policy projection | (b) as a ROUTE | `team_transformer.py`, `ProjectionAssembler` · ON (legacy) | **UNVERIFIED: never ablated** (audit F4: four routes for one fact) | — | REPLACE (the static arm's three board tokens delete the bypass, pi 1177 → 1152) |
| B2 | SIDE tokens: alive / fainted / revealed counts | counts ÷ 6 per side (ours alive = HP > 0; theirs = `opp_addressable`) | (b) | `board_tokens.side_features` · ARM | structural: a fainted mon is a masked key and a softmax cannot count (entity audit §4 A1); diag: after layer 2 the faint counts read within 0.04 of legacy (DESCRIPTIVE) | counting by attention (impossible) | KEEP |
| B3 | SIDE tokens: Spikes layers, Reflect / Light Screen / Safeguard / Mist presence, Wish pending, Sleep Clause used | side-relative, ONE `side_proj` for both sides; Sleep Clause used = a non-Rest sleeper on that side (the op's own read) | (b) | `board_tokens.side_features` · ARM | diag: our-side Spikes is the board fact static decodes worst (R² 0.33 vs 0.46 after the trunk; their side within 0.02) (DESCRIPTIVE) | — | KEEP; see §3 for the per-mon copy |
| B4 | FIELD token | weather (one-hot 5 + permanent + turns left), the 3 clock scalars, turns since progress | (b) | `board_tokens.field_features` · ARM | diag: weather and clock reach our tokens within ~0.03–0.04 of legacy after layer 2 (DESCRIPTIVE) | — | KEEP |

### 2.4 The damage operator: rows, rules and constants

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| D1 | incoming per-mon row (6 × 12) | per our mon, per physical / special channel: low / high / crit roll ÷ max HP, P(KO), accuracy, over their believed moves; P(outspeed); provenance | (a) | `damage_op.py` · ON | §4.1 op table: KL 0.0160, 6.0 % flips, shuffle control 0.0130 (barely above shuffle; STALE) | the gen-3 damage formula × the belief | KEEP |
| D2 | Choice-Band tail (13) | `phys_high_cb`, `pko_cb` per mon, `p_cb` from the item belief | (a) (c) | `damage_op.py` · ON | §4.1: KL 0.0013, 1.4 % flips, shuffle 0.0014 — no dependence the probe separates from noise (STALE) | Band's ×1.5 on the believed holder | TRY REMOVING (§5 rank 6) |
| D3 | outgoing single-active block (45) | per our move: low / high / crit / P(KO) vs their active, P(outspeed), 7 secondary chances; also the pointer move cell's base 13 | (a) | `damage_op_blocks.py` · ON | §4.1: KL 0.0176 vs shuffle 0.0254 at the CONCAT (no dependence there, STALE); its pointer-cell route never isolated | what our move does | KEEP |
| D4 | status landing (8) | per our status move: P(lands) folding accuracy, type and ability immunity (known or Smogon prior), already statused, Sleep Clause, their Substitute, their Safeguard, Yawn on a drowsy target; `known` | (a) | `damage_op_blocks._outgoing_status_land`, `status_rules.py` · ON | §4.1: KL 0.0006 (STALE); correctness MEASURED (old op: 12 of 199 certain-zero claims landed; new op 0 of 181, `measurements/op_gigo_2026-10-07/`) | the status rules | KEEP |
| D5 | incoming status rules | our Safeguard, incoming Sleep Clause, Freeze Clause, our active's Substitute, Yawn on our drowsy active | (a) | `status_rules.incoming_status_mask` · ON | correctness (same GIGO fix); strength **UNVERIFIED: never ablated** | — | KEEP |
| D6 | opponent ability: known or Smogon marginal | every op read of their ability: revealed → exact row; unrevealed → the species' Smogon marginal (damage multipliers, status blocks, Early Bird, traps) | (a) (c) | `damage_op_blocks.opp_ability_view` · ON | GIGO fix (Toxic "never landed" on every unrevealed Snorlax before, ARCH §4) | — | KEEP |
| D7 | non-formula damage models | fixed (Seismic Toss 100, Dragon Rage, Psywave's mean), target-HP fraction (Super Fang, OHKO moves), Endeavor, Flail / Reversal / Eruption by HP, Return / Frustration 102, Magnitude 71, Present 52, Beat Up exact; Counter / Mirror Coat / Bide / Low Kick / Spit Up 0 by declaration | (a) | `damage_tables.DAMAGE_MODELS`, `damage_kinds.py` · ON | Beat Up MEASURED against the sim (507.2 HP sim mean vs 511.5 op, `beatup_sim_parity_test.py`); strength **UNVERIFIED: never ablated** | — | KEEP |
| D8 | field base-power modifiers | weather ×1.5 / ×0.5; Mud / Water Sport ×0.5 on Electric / Fire | (a) | `DamageOperator._sport_mult` · ON | **UNVERIFIED: never ablated** | — | KEEP |
| D9 | the KO ramp | `acc · clamp((dmg − cur_hp) / (0.15 · dmg), 0, 1)`: a continuous ramp across the roll range; also inlined for the Choice-Band tail (`ko_cb`) and re-thresholded at the sub's HP in `intent_threshold` (P8) | (a) **AF** | `DamageOperator._rolls` · ON | **UNVERIFIED** as physics: it smooths the 16 discrete rolls and ignores the 1/16 crit KO (FINDING, §7); `_rolls`' docstring wrongly called it "the exact realized KO probability" (docstring corrected 2026-10-09) | — | **FIX: make it EXACT** over the 16 rolls + the crit chance (§4 rank 3). KEEP the feature |
| D10 | hand constants | the gen-3 RULES (roll 0.85–1.0, crit 1/16 ×2, paralysis speed ×0.25, Band ×1.5) and the hand CLAMPS (`_DMG_CHIP_CAP` 1.5, `_DMG_CRIT_CAP` 3.0) | (a) | `damage_op_layout.py` · ON | audit F7c: the rules are exact physics (KEEP); the clamps are documented saturations | — | KEEP |
| D11 | Pursuit | `p_pur_vs_us` (P(some mon of theirs holds Pursuit)); the `x` cell's `pursuit_p` / `pursuit_eff`; `intent_conditional`'s ×2 never-miss strike on a departing target | (a) | `damage_op_pairwise.py`, `intent_conditional.py` · ON | edge `x` 0.3 % flips (§5.4, STALE); the rest **UNVERIFIED: never ablated** | — | KEEP |

**The pairwise kernels as attention-edge cells** (all 17 families ON, each a zero-init `Linear(cell, 2 · heads)`;
696 parameters in all, audit F12). An edge writes a RATIO within a softmax row, never an amount (ARCH §5.3). Evidence
is the §5.4 edge table (zero the family → masked KL / argmax flips / |dV|; gen-3, 9.6M, STALE per audit F22).

| # | family | cell | why | evidence (§5.4: KL · flips · \|dV\|) | action |
|---|---|---|---|---|---|
| D12 | `d1` our move × their mon | low, high, crit, P(KO), type mult, revealed | (a) | 0.0345 · 6.0 % · 0.274 | KEEP |
| D13 | `d2` our mon × their active | best high, best P(KO), P(outspeed), alive (our bench's offense) | (a) | 0.0426 · 7.6 % · 1.308 (the largest) | KEEP |
| D14 | `d3` their threat seat × our mon | high, P(KO), effectiveness, physical, presence | (a) | 0.0013 · 1.9 % · 0.141 | KEEP |
| D15 | `d4` our mon × their bench (C1b-style) | physical / special high and P(KO) of their BENCH's believed moves | (a) | 0.0015 · 1.1 % · 0.492 | KEEP |
| D16 | `s1` / `s3` status landing | P(lands), P(lands)·immobilising (+ presence on `s3`) | (a) | 0.0017 · 0.8 % / 0.0003 · 0.9 % | KEEP |
| D17 | `v` speed | P(outspeed), both alive, revealed | (a) | 0.0035 · 2.9 % · 0.651 | KEEP |
| D18 | `t` trapping | P(i traps j), P(j traps i) from the Shadow Tag / Arena Trap / Magnet Pull priors | (a) (c) | 0.0003 · 0.7 % · 0.121 | KEEP |
| D19 | `c1` setup consequence | is_boost, Δ best high / P(KO), Δ P(outspeed), HP cost, Δ incoming high / P(KO) after the boost | (a) | 0.0002 · 0.4 % · 0.015 | TRY REMOVING after a current re-read (§5 rank 4) |
| D20 | `c2` status consequence | is_status, lands, Δ their P(outspeed), Δ incoming physical high, Δ schedule, sleep free turns | (a) | 0.0007 · 0.4 % · 0.017 | TRY REMOVING after a re-read |
| D21 | `c3` recovery | is_recovery, Δ incoming P(KO), Rest sleep turns | (a) | 0.0002 · 0.2 % · 0.018 | TRY REMOVING after a re-read |
| D22 | `c4` Protect | is_protect, P(success), net ours / theirs | (a) | 0.0000 · 0.1 % · 0.001 | TRY REMOVING after a re-read |
| D23 | `c5` Baton Pass receiver | is_bp, Δ best high / P(KO), Δ P(outspeed) per receiver | (a) | 0.0000 · 0.2 % · 0.018 | TRY REMOVING after a re-read |
| D24 | `x` entry / exit | entry chip (Spikes 1/8 · 1/6 · 1/4, grounded), Pursuit P and effect, grounded | (a) | 0.0000 · 0.3 % · 0.036 | KEEP (the static diag says Spikes is where static is weakest; do not cut its routes now) |
| D25 | `g` end-of-turn ledger | Leftovers, weather chip, status tick (Toxic at its next ramp), Leech Seed | (a) | 0.0000 · 0.1 % · 0.016 | KEEP (the stall-team residual race; see §4) |
| D26 | `h` pair history | O7's tendencies | (b) | not measured (post-dates §5.4) — **UNVERIFIED: never ablated** | TRY REMOVING (§5 rank 7) |
| D27 | `r` event references | is_actor / is_target / is_rel: event e names mon m (species equality, side-gated) | (b) structural | not measured — **UNVERIFIED: never ablated** | KEEP (structural, no physics) |

Every family off together: KL 0.1011, **13.9 %** flips, |dV| 1.857 (§5.4, STALE).

### 2.5 Pair outcome and the seven per-action blocks (production), F11's judgments re-sorted by §1's test

All read the opponent-intent α / β (detached) and widen the pointer cells; production carries all seven
(move cell 13 → 62 wide, switch cell 15 → 34). Audit F11: **no end-of-run read of any block's dependence exists —
every row is UNVERIFIED: never ablated.** The move-resolution family (§2.6) RETIRES all seven when on, and
it was built on the 2026-10-06 list, so it also DROPS several quantities the 2026-10-09 test calls FACTS (`spin_value_lost`,
`spin_denied`'s stake, four of P8's six); restoring them is §4 rank 5.

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| P1 | `pair_outcome_cell` — facts (12) | at our active: the α-reduced incoming low / high / crit / KO ramp / acc / physical, and P(par / brn / frz / slp / psn / tox lands) by status identity | (a) | `pair_outcome.py` · ON | **UNVERIFIED: never ablated** | trading damage against status in one currency | REPLACE (move resolution keeps them) |
| P2 | `pair_outcome_cell` — `neutralization`, `tempo_cost` | `neutralization` = Σ_s p_s · sev_s: burn = `0.5 · base_atk/(base_atk+base_spa)` (a base-stat proxy), paralysis = 0.25 + 0.75·Δ P(outspeed), freeze / sleep = 1.0, poison / toxic = the first residual tick (an HP fraction): units combined with chosen weights. `tempo_cost` = P(any major status) × `undo_turns`, the cheapest of {self-cure 1, Natural Cure 1, Rest 2, bench cleric 2}: the turn counts assume OUR cure plan, and items (Lum / Chesto) are not read | **J** as implemented (`neutralization`: weighted sum across units; `tempo_cost`: MOSTLY J, it assumes our plan; the existence of a cure path is a fact) | `damage_op_blocks.pair_outcome_coords` (feeds `pair_outcome.py`) · ON | **UNVERIFIED: never ablated**; owner DECLINED `tempo_cost` (2026-10-06), re-classed 2026-10-09 | the value of a status | **REPLACE with facts** (§4 rank 4): burn → the exact damage lost on the mon's believed PHYSICAL moves; paralysis → P(full para) 0.25 and Δ P(outspeed) as separate columns; cure → FACT flags. The cell leaves production with the bundle (move resolution drops it) before the facts exist |
| P3 | `pair_outcome_switch` (15) | P1's 12 at EVERY defender + the 2 judgments + `spin_denied` (= is_ghost(j) · α_spin · their hazards) | (a) + **J** (the two P2 coordinates) | `pair_outcome.py` · ON | **UNVERIFIED: never ablated** | — | REPLACE (move resolution keeps the facts and only the probability half of `spin_denied`). `spin_denied` itself, probability × hazard stake, is a FACT (expected layers preserved): KEEP it, and restore the stake (§4 rank 5) |
| P4 | `conditional_threat_cell` (4) | `e_pko_acc` (P(this mon dies)), `e_type_mult`, margin high, margin crit | (a) | `conditional_threat.py` · ON | **UNVERIFIED: never ablated** | — | REPLACE (move resolution keeps all four) |
| P5 | `switch_branch_cell` — facts (7) | our move's outcome on the β-weighted arrival (high, P(KO), mult), P(they switch), P(spin blocked), Protect attack / blocked mass | (a) | `switch_branch.py` · ON | **UNVERIFIED: never ablated** | — | REPLACE (move resolution keeps them) |
| P6 | `switch_branch_cell` — `wasted_ko`, `spin_value_lost` | `pko_stay · α_SWITCH` = P(KO \| they stay) × P(they switch), an event probability and an interaction term of two delivered columns; `p_spin_blocked` × the Spikes fraction on OUR side = the expected hazard layers a blocked spin leaves | (a) both FACTS (`wasted_ko` REDUNDANT with its inputs) | `switch_branch.py` · ON | **UNVERIFIED: never ablated**; owner DECLINED both (2026-10-06), re-classed FACT 2026-10-09 | — | `wasted_ko`: **TRY REMOVING as a redundancy bisect** (§5 rank 11), not as bias; `spin_value_lost`: **KEEP** (move resolution drops it: restore, §4 rank 5) |
| P7 | `intent_move_cell` (7) | what a landed status does (Δ their outspeed, burn, schedule, sleep, sleep-free turns) + 2 redundant | (a) | `intent_move_cell.py` · ON | **UNVERIFIED: never ablated**; G2 usage baseline before the build | — | REPLACE |
| P8 | `intent_threshold` (6) | P(KO) context; P(Focus Punch survives) = 1 − Σ α·acc·1[high > 0]; Destiny Bond gate × P(KO); Substitute = P(their move breaks a 25 % sub), the KO ramp re-thresholded at the sub's HP; Endure gate × P(KO); Endeavor gate × (1 − P(KO)) (P(not KO'd this turn); it does not condition on move order) | (a) all FACTS; the Substitute ramp is an **AF** (via the KO-ramp window) | `intent_threshold.py` · ON | **UNVERIFIED: never ablated**; usage before the build: Endure 0.0 %, Substitute 0.9 % (`gen12_mechanic_usage_baseline.json`); owner DECLINED "the hand thresholds" (2026-10-06), re-classed FACT 2026-10-09 | — | KEEP (the Substitute ramp becomes exact with the KO-ramp fix, §4 rank 3). Move resolution drops four of the six: restore (§4 rank 5) |
| P9 | `intent_conditional` (13) | Counter / Mirror Coat returns, flinch usefulness, Explosion's trade and blockers (Protect / Detect only), Pursuit's ×2 trigger, Protect's avoided damage / status and its odds × P(an action follows), Magic Coat's `reflectable` bounce | (a) | `intent_conditional.py` · ON | **UNVERIFIED: never ablated** | — | REPLACE |

### 2.6 Move resolution (audit F11 as built)

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| R1 | move cell (38) | per legal move: `p_resolve` = P(it resolves as stated: lands, not blocked, not immune, not a no-op), intent-weighted over their seats + OTHER (priced per priority level), with `p_lands_stay`, `p_lands_switch`, `p_ko_first`, `p_act`; `p_ko_us`; `dbond_p_ko` (no threshold); the seven blocks' 31 kept facts | (a) | `move_resolution*.py` · ARM (`--move-resolution on`) | correctness MEASURED: Lane S, 31,942 played moves, 1,515 exact-zero claims met an executed move, ZERO resolved; the certain direction 41 / 11,738 failed (0.35 %, the named residuals); strength **UNVERIFIED** (in the bundle) | "will this actually work?" across ~19 gen-3 rules | KEEP (adopt via the bundle) |
| R2 | switch cell (18) | mon j's α-reduced incoming damage + status, `p_spin_denied` (fact half), `e_pko` (accuracy once), `e_type_mult`, the two margins, `p_switch_resolves` (their Pursuit can KO the departing mon) | (a) | `move_resolution*.py` · ARM | as R1 | — | KEEP (adopt via the bundle) |

### 2.7 Speed physics

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| S1 | the outspeed logistic | `sigmoid((our_spe − their_spe) / 15)` at every op site that prices who moves first; 15 is hand-chosen ("about one stage") | (a) with a hand constant | `damage_op_layout._DMG_SPEED_SCALE` · ON | Lane S, 23,598 equal-priority rows: Brier 0.0425, log loss 0.1341, ECE 0.0157 (`measurements/speed_physics_f7b_2026-10-07/`) | — | REPLACE (S2, in the bundle) |
| S2 | exact speed physics + the Smogon spread mixture | our speed exact (stat, stage floor, paralysis rounding); theirs the species' DISCRETE Smogon distribution over the Speed stat (every chaos spread); a tie a coin flip; the priority bracket `p_seat_first`; Quick Claw format-gated off (banned) | (a) (c) | `move_order.py`, `damage_op_speed.py`, `belief_tables.build_species_speed_mix` · ARM (`--speed-physics on`) | same rows: log loss 0.1308 (better), 0 rows certain and wrong, but Brier 0.0440 and ECE 0.0184 (worse) — NOT better calibrated overall; strength **UNVERIFIED** (in the bundle) | — | KEEP; condition the mixture on observed move order (E8) when it is built |

### 2.8 Belief, hypothesis tokens and their priors

The oracle ceiling bounds this whole group: an arm that SAW the opponent's species (or full sets) from turn 1 gained
only +2.44 [+0.31, +4.57] pp (species) / +2.18 [+0.53, +3.84] pp (full) at 15M, below the replicate floor (4.57 pp)
— "belief representation has little strength leverage at this budget" (X5 look 1, `measurements/x5ab_look1_2026-10-06/`).

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| H1 | the Smogon move prior | `P(m in set \| s)` = chaos `Moves[m]` over the rating-weighted set total; learnset-gated (an unlearnable move `logit(1e-6)`); a legal move absent from the data gets `move_candidate_floor` 0.02; revealed moves pinned | (c) | `belief_tables.build_move_prior_logits` · ON | **UNVERIFIED: never ablated** (the learned correction on top earns +5.2 pp on the pool and nothing on ladder teams, ledger 2026-09-24; that zeroed the correction, not the prior) | the opponent's moveset from scratch | KEEP |
| H2 | the T0 species prior | Smogon teammate naive Bayes over the revealed team, lift clipped to ±4 nats, marginal floor 1e-4, Species Clause | (c) | `t0_species.py`, `belief_tables.build_species_cooccur_prior` · ON | **UNVERIFIED: never ablated** (audit F32: KEEP, parameter-free so it cannot memorise the pool) | the hidden team from scratch | KEEP |
| H3 | the hidden-slot move mixture (E10) | `P(m \| hidden) = Σ_s P_T0(s \| revealed) · P(m \| s)` | (c) | `MoveBelief` · ON | MEASURED for belief, not strength: recall@4 0.28 vs 0.10 for the constant it replaced (pool) | — | KEEP |
| H4 | typed Hidden Power composition | HP exists only as the 16 typed moves, each `logit(presence · P(type))`; bare HP driven to −30 | (a) | `HPTypeBelief.compose_typed_hp` · ON | **UNVERIFIED: never ablated** | — | KEEP |
| H5 | item prior and the Choice-Band prior | the Smogon item row (floor 1e-5); `SPECIES_CB_PRIOR` where the item belief is off | (c) | `belief_tables.build_item_prior`, `damage_tables.build_species_cb_prior` · ON | **UNVERIFIED: never ablated** | — | KEEP |
| H6 | spread, nature and EV priors | the Smogon spread prior (mean ± std per stat; nature and EV priors) behind the learned spread belief | (c) | `belief_tables.build_opp_spread_prior` & co · ON | **UNVERIFIED: never ablated** | — | KEEP |
| H7 | fixed-mass presence | `π_s = σ(a_s + τ)` with τ solved so Σπ = 6 − revealed (64-step bisection); moves the same at k = 4 − revealed; π DETACHED at every reduction | (a) | `hypothesis_set.py` · ON | X5 look 3: NON-INFERIOR at matched steps (Δ̂ −0.99 pp [−2.92, +0.94]); intent log loss 1.753 vs 1.904 nats (t 10.05) (`measurements/x5ab_look3_2026-10-07/`) | the team-size constraint | KEEP |
| H8 | presence semantics: detached `log π` on every opponent key | added to the trunk's attention, `their_cls`, `value_cls`, the belief pool, the critic pool, the flat pointer | (a) | `hypothesis_tokens.py` · ON | part of X5 (above); not isolated | — | KEEP |
| H9 | OTHER_species and OTHER_move averaged construction | OTHER's defender = `P_tail @ tables`, attacker = E_tail[stats / STAB / speed], moves = the E10 mixture over the tail; presence 1 − Π(1 − π) at a max site | (c) | `hypothesis_tokens.other_roster` · ON | part of X5; OTHER carries 57–74 % of the hidden mass at a 1:1 budget, so pricing it is required (X5 §3.3, MEASURED) | — | KEEP |
| H10 | hypothesis dex rows | a hidden slot's token = its hypothesis species' dex row (stats, types, the ability prior, full HP, no status) | (c) | `hypothesis_dex_rows.json`, `hypothesis_encode.py` · ON | part of X5 | — | KEEP |
| H11 | belief-head logit pins | `_REVEAL_LOGIT` 10, `_HP_PRESENCE_OFF_LOGIT` −30 | (a) | `belief_heads.py` · ON | audit F7: documented saturations, KEEP | — | KEEP |
| H12 | belief TRUNK SHAPING | six belief heads' labels reshape the shared trunk at 0.05 each (`belief_grad_mode shaping`); the intent head is detached | — (a training choice, not a feature) | `--belief-grad-mode` · ON | **UNVERIFIED** (audit F9: only the OUTPUTS were ever zeroed) | — | TRY REMOVING (§5 rank 8; the `detached` arm is built) |

### 2.9 The opponent-intent pointer's inputs

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| I1 | the flat pointer's prior term | each candidate (their K move seats, OTHER_move, a switch to each slot, OTHER_species) scored by one shared scorer PLUS its detached log π | (c) | `flat_intent.py` · ON | X5 look 3's intent log loss above (1.753 vs 1.904) | the opponent's action distribution without a prior | KEEP |
| I2 | the intent label rules | masked when their switch-in was dragged by our Roar, was a free replacement, or Encore overrode it; a called move labelled as its caller; a denied action masked | (a) — training labels, not a forward input | `src/rust_env/src/labels/intent.rs` over the TurnDelta fold (`trackers/delta.rs`) · ON | correctness (`gen3_intent_label_semantics_fixes_v1`) | — | KEEP |

### 2.10 Format-spec priors

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| F1 | banned → prior 0 | a format-banned species / item / move / ability gets no mass in every Smogon prior and the ILLEGAL value in every model table (Quick Claw, Sand Veil, Soundproof, Swagger, the OHKO and evasion moves, Smeargle's Ingrain, the ability-locked species) | (c) + the format's rules | `format_spec.py`, `compute_priors.py`, the four table builders · ON | correctness: 148 of 991 golden decisions moved in exactly 2 columns (`design_format_spec.md` §5.1) | — | KEEP |
| F2 | board clauses read from the spec | Sleep Clause / Freeze Clause in force, read by `status_rules`, the op and move resolution | (a) | `format_spec.sleep_clause_mod()` / `freeze_clause_mod()` · ON | rules, verified at the source | — | KEEP |

### 2.11 Critic inputs

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| C1 | `value_entity_pool`'s op-row source | the op's 6 per-our-mon incoming rows as extra keys of the critic's entity pool (with the 12 team tokens, + the board / global and belief rows under `full`) | (b) | `value_readouts.py` · ON | the WHOLE pool: dV 5.490 = 97 % of all routes off (gen-14, ARCH §3.2); the op-row source alone **UNVERIFIED: never ablated** | the critic reading damage through the trunk only | KEEP |
| C2 | `value_threat_inject` | each of our mons' α-reduced incoming row (R1 presence belief) projected onto the VALUE pool's copy of its token | (b) | `value_threat_inject.py` · ON | dV 1.0686 (gen-14, dependence, not coverage: audit F10) | — | TRY REMOVING (§5 rank 3; the OFF arm is built) |

### 2.12 Reductions over the opponent's believed moves

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| L1 | per-channel hard maxima (`--op-reduction max`) | `max_c(w_c · v_c)` separately per channel and defender; `acc` and provenance at the argmax of another channel — one row can describe several moves (defect D2) | (a) with an INCOHERENT reduction | `damage_op.py`, `damage_op_pairwise.py` · ON | none for the reduction itself; the NaN class it caused is fixed (`max_by_index`) | — | REPLACE (L2, in the bundle) |
| L2 | principled reductions (`--op-reduction principled`) | the α-weighted EXPECTATION (α = presence ÷ total presence, one per attacker) + the noisy-OR KO worst case (T8) | (a) | `op_reduction.py` · ARM | K9(b) excluded share 10.4 % → 7.9 % (a gate statistic, not strength); strength **UNVERIFIED** (in the bundle) | — | KEEP (adopt via the bundle). Its JUDGMENT half moves to the network: the trunk's attention over the E4 threat tokens is the learned reduction (owner: "compute the FACTS exactly; the trunk's attention does the judgment") |
| L3 | the E5 tail score | per their mon, the believed moves beyond the top-K: `p_tail` and `worst_phys` / `worst_spec` = max of `π · BP/150 · acc` (defender-free; the expectation under `principled`) | (a) with hand normalisers | `pointer_head.py` · ON | **UNVERIFIED: never ablated** | — | KEEP; note the BP/150 here vs the move slot's power/200 (audit F7) |
| L4 | the pair-reduction rungs R1 / R2W / R2L / R3 | Contract-W/L reducers beside the hard max | — | `pair_reduce.py` · INERT (constructor-only; production builds nothing) | none | — | TRY REMOVING (dead code: backlog T21, audit F8) |

**Size: 88 rows** (observation 14 · per-mon token 8 · board 4 · operator 11 + edge families 16 · pair outcome and the seven blocks 9 · move resolution 2 · speed 2 · belief 12 · intent 2 · format spec 2 · critic 2 · reductions 4), plus 2 in flight (§3).

---

## 3. BUILT behind their own flags, OFF (the static port, `gen3_static_port_v1`, config v147, 2026-10-09)

Both come from the static diagnostic (`measurements/static_diag_2026-10-09/`, H2 / H3: static is less expressive for
board facts that arrive only through attention, mostly at the last of two trunk rounds). Both are BUILT
(`agents/model/static_facts.py`, `design_static_tokens.md` §12.3–12.4), each a STRUCTURAL `cli` flag that requires
`--token-encoding static`, OFF in production, through a zero-init bias-free `IsolatedLinear` built LAST (no RNG draw:
the ON build is the OFF build plus one zero matrix). Neither is trained; the bundle screen after the perf phase reads
them.

| # | feature | what it computes | why | where · status | evidence it helps | learning it replaces | action |
|---|---|---|---|---|---|---|---|
| N1 | the entry-hazard cost per mon, BOTH sides (`--mon-hazard-cost on`) | for every mon: its OWN side's Spikes layers / 3, and the HP fraction it would lose on switching in (1/8, 1/6, 1/4; 0 for a Flying type or Levitate — our ability exact, an opponent's revealed-exact else its species' Smogon P(Levitate), 0 or 1 per species in gen 3); an X5 hidden slot priced as its hypothesis; 2 → 128, added to the mon token after the op content, before the trunk (D content, not D's MLP: a hidden slot's static token is a dex-table gather) | (b) | `static_facts.mon_hazard_features` reading `DamageOperator.spikes_entry`, the ONE rule the `x` cell reads too · BUILT, OFF | diag: our-side Spikes after the trunk R² 0.33 static vs 0.46 legacy (bench mon 0.31 vs 0.50), the one gap that GROWS with training; static holds stall teams 10.7 pp worse than legacy (DESCRIPTIVE). Strength **UNVERIFIED** (the bundle screen) | attention carrying "Spikes are on my side" into each token | TRY ADDING (bundle screen) |
| N2 | our active's HP and status on its move seats (`--move-actor-state on`) | our active's `[HP fraction, status one-hot (7)]`, 8 → 128, added to its four VALID E3 seats (≡ zero-init input columns of `move_seat_proj`); the opponent's E4 seats get nothing (neither encoding ever gave them their active's state, so the diagnostic shows no gap there) | (b) | `static_facts.move_actor_features`, root `move_actor_proj` · BUILT, OFF | diag: our active's HP in its move token R² 0.59 static vs 0.80 legacy (DESCRIPTIVE). Strength **UNVERIFIED** (the bundle screen) | attention from the move seat to its owner | TRY ADDING (bundle screen) |

**The N1 hazard, RESOLVED as built (option (a), ONE rule):** the rule is factored into
`DamageOperator.spikes_entry` and BOTH the `x` cell and N1 call it, so there is no second implementation to drift.
N1 is added after the op runs (beside OPC, not inside D's MLP), so it reads the op's function on the op's context
within the tier contract. `static_port_test.py` plants a value into `spikes_entry` and sees both move, and checks the
two agree on constructed boards (Flying, Levitate known, an opponent's Levitate unknown under its prior, a top-1
prior id that is not a reveal, 0–3 layers). Overlap that stays: N1's fraction column restates the `x` cell's
`entry_chip`, which OPC (T7) already carries onto every mon, alive-gated; N1's new fact is the layers column. **The typechange finding is RESOLVED (v148,
`gen3_spikes_entry_base_types_v1`, 2026-10-09).** The observation's type columns are a mon's CURRENT types (the Rust
`PMon::types` returns the temporary types: Color Change, Transform, Conversion, Castform's Forecast), while a
switch-in follows `clearVolatile` -> `setSpecies(baseSpecies)` (verified in `deps/pokemon-showdown`
`sim/pokemon.ts` / `sim/battle-actions.ts`), so an active Kecleon turned Flying read "immune on re-entry" in both the
`x` cell and N1. `spikes_entry` now reads the species' BASE types (`SPECIES_TYPE`, from `ctx.species_ids`; an X5
hidden slot's species is its hypothesis'). A BENCHED mon's columns were already base (`PMon::switch_out` clears the
temporary types), so only the ACTIVE mon's own cell was wrong. This changed the production `x` cell for an active
mon whose current types differ from its species'. **The ability half is RESOLVED too (v149,
`gen3_spikes_entry_species_levitate_v1`):** Levitate was read from the ability column (the CURRENT ability: Trace,
Role Play, Skill Swap, Transform), while a switch-in resets it to the base ability. In gen 3 Levitate is the SOLE
ability of its 17 species (Showdown's gen-3 pokedex and `data/` agree), so the species' Smogon P(Levitate) is exactly
0 or 1 and `spikes_entry` reads it for every mon on both sides, revealed or not: a Gardevoir that Traced Levitate pays
on its next entry; no base-ability column is needed.

---

## 4. Next candidates to ADD, ranked

Each is a fact (never a judgment), Smogon-derived where it is a prior, and would be ONE lever with its own screen.
The diagnostic's archetype table drives the ranking: holding stall teams static is 10.7 pp worse than legacy,
semi-stall 8.2 pp, hyper-offense 1.2 pp, and Wish / spin / spinblock / phaze / Spikes teams hold at 45–46 % against
49 % for choice / setup teams (DESCRIPTIVE). The game static loses is the long, switch-heavy, residual game.

| rank | candidate | the fact | why now | home | cost |
|---|---|---|---|---|---|
| 1 | **the entry chip in the SWITCH pointer cell** (entity audit B3) | the HP fraction mon j loses on switching in (Spikes × grounded) | the switch logit has NO entry-chip coordinate today (incoming row 12 + CB 3 + pair outcome 15 + threat 4; ARCH §3.3); Spikes is the central gen-3 hazard and the diag's sharpest gap; the cell is the absolute route the switch decision reads | the switch cell (a widening, the `pair_outcome_switch` precedent); under move resolution, beside `p_switch_resolves` | S (the `x` cell computes it) |
| 2 | **the residual ledger per mon in D / on the switch cell** | the end-of-turn net per mon: Leftovers, weather chip, the Toxic tick at its next ramp, Leech Seed, the Wish heal arriving to whoever is in the slot | stall's residual race; under static the `g` amounts reach tokens only via OPC (T7), and the switch cell has none | D (ours), the switch cell | S (the `g` cell computes most of it) |
| 3 | **the EXACT KO ramp** (D9, an **AF**) | P(KO \| hit) over the 16 discrete damage rolls (85 … 100 %) plus the crit chance (1/16, ×2 damage, screens ignored), instead of the continuous 15 %-window ramp that smooths the rolls and omits the crit KO | an APPROXIMATE FACT feeds every P(KO) the model reads (D1 · D3 · D12-D15, P1 · P4 · P5 · P8, the Substitute break, `ko_cb`), wrong in the tails where decisions are; the fix is exactness, not removal | `DamageOperator._rolls` (+ the inlined `ko_cb`, `intent_threshold`'s sub ramp) | S–M (one rule, three sites; a behaviour change: its own version and screen) |
| 4 | **the status FACTS that replace `neutralization` / `tempo_cost`** (P2) | **burn**: the exact damage lost on the mon's believed PHYSICAL moves (the operator can compute it); **paralysis**: P(full para) 0.25 and Δ P(outspeed) as SEPARATE columns; **cure**: FACT flags, a Heal Bell / Aromatherapy user alive on that side, Natural Cure, Rest available, believed Lum / Chesto (items are read nowhere today) | the P2 judgments leave production with the bundle; the facts they approximated (what a status costs, whether a cure exists) have no other home. The existence of a cure path is a fact, only its turn price (our plan) was a judgment | the pair-outcome / move-resolution status block (`damage_op_blocks.pair_outcome_coords`) | S–M |
| 5 | **restore the FACTS move resolution dropped** (P3 · P6 · P8) | `spin_value_lost` and `spin_denied`'s hazard stake (P(spin blocked) × layers, expected layers preserved); P8's Focus Punch survives, Substitute break (exact via rank 3), Endure × P(KO), Endeavor × (1 − P(KO)) | the 2026-10-09 test calls them FACTS, and the bundle's move-resolution arm drops all of them as judgments; a bundle screen on the old list would measure its absence | `move_resolution*.py` (the switch and move cells); its dropped-judgment test (`move_resolution_test.py`) and ARCHITECTURE's "Dropped (judgments)" paragraph change with it | S |
| 6 | **the OBS-FACTS block** (`--obs-facts v1`, O13) | what the opponent has seen of us; evidence against their Choice lock; Encore / Taunt / Disable / Uproar / partial-trap elapsed and bounds; screen turns left | built and gated, never trained; facts with no other source (entity audit §4 A5) | as built (`obs_facts_inject.py`) | XS to enable; one screen |
| 7 | **phazing's entry damage** | for Roar / Whirlwind: the expected entry chip of the mon gen 3 drags in at random (the mean over their alive bench of `x`'s chip) | phaze teams are among those static loses most; Roar into Spikes is a core gen-3 plan (`obs_enrichment_backlog.md` E12) and no cell prices it | the move cell (move resolution's family) | S |
| 8 | **Freeze Clause used, per side** | any frozen mon on that side (the op already reads it) | the side token carries Sleep Clause but not Freeze Clause (`design_format_spec.md` §8) | SIDE | XS |
| 9 | **Future Sight / Doom Desire pending on the TARGET side** (entity audit B6) | turns left and the damage priced at cast (typeless) | today a bit on the USER, lost when the user switches (WRONG entity) | SIDE + the op | S |
| 10 | **speed-order evidence** (backlog E8) | whether an observed move order contradicts the believed speed | conditions S2's Smogon mixture on the battle (its open lever) | the belief / S2 | S |
| 11 | **per-cause HP accounting** (backlog E2 / E3) | HP lost and gained by cause, per mon per turn; the opponent's PP estimate | the stall resources legibility question | D / event rows | S–M |

**Not on the list, on purpose:** any "is this my last answer" scarcity feature (`design_pair_reduction.md` §11, a judgment);
anything from the pool or the omniscient board.

**The learned alternative to all of these** is a third trunk round under static (diag "Next lever"; the owner's
direction allows "add a trunk round if needed"). It is not a hand feature and is not ranked here, but every row above
should be screened knowing it is the competitor: if depth closes the Spikes gap, rows 1–2 may not be needed.

---

## 5. Candidates to REMOVE / ABLATE, ranked

**What an ablation costs.** A one-lever removal is screened on the static screen's recipe
(`design_static_tokens.md` §8.1 / §8.2): one commit for every seed, 15M steps per seed, 3 / 5 / 8 seeds per arm at
group-sequential looks (O'Brien–Fleming boundaries 5.761 / 2.683 / 1.874), each look a mirrored head-to-head cross at
1,000 pairs per cell, NON-INFERIORITY at δ = 3.5 pp. That is 6–16 training runs of 15M steps and up to 64 cells of
2,000 games; its GPU hours are **UNVERIFIED** here. A removal that passes is adopted ("not worse" is the bar for
deleting a hand feature: simpler at equal strength). Several of these can be bundled and split on failure, as the
owner did for the current bundle. A cheap CPU READ comes first where the evidence is stale: the edge-family and
op-block re-read on current checkpoints (audit F22, ~0.5 agent-day, CPU forwards).

| rank | candidate | why remove | what replaces it | cost |
|---|---|---|---|---|
| 1 | **`turns_since_progress`** (O10) | a hand definition of "progress" (3 % thresholds, eight reset conditions, carve-outs) is a JUDGMENT (thresholds that grade progress), the kind §1 says the network should learn; it survived the shaped reward it was built for; never ablated | the event window, the deadline clock and the trunk; if a clock signal proves needed, the threshold-free FACT "turns since either side lost HP" | one screen; an in-model mask of its column avoids an obs change |
| 2 | **the judgment-as-implemented P2 coordinates** (`neutralization`, `tempo_cost`) | by §1's test: `neutralization` sums units with chosen weights (a base-stat burn proxy, 0.25 + 0.75·Δ outspeed), `tempo_cost` assumes our cure plan; the owner DECLINED `tempo_cost` (2026-10-06) and re-sorted the rest on 2026-10-09 | SEPARATE exact facts (§4 rank 4) | already IN the bundle (move resolution drops the cell); the facts are a later add |
| 3 | **`value_threat_inject`** (C2) | a third critic route for one fact (audit F10); its α is a deliberately crude presence belief | the trunk (`prefuse_proj`) and the entity pool's op rows | the OFF arm is built; meter `main.ops.critic_read` (`gate.resolution.*`) |
| 4 | **the consequence edges `c1`–`c5`** (D19–D23) | ≤ 0.4 % argmax flips each (§5.4, gen-3, STALE) | the trunk and the pointer cells | re-read first (F22, CPU); then one bundle of five, split on failure |
| 5 | **the hard maxima** (L1) | incoherent (defect D2) | L2 + the trunk's attention over the E4 tokens | already IN the bundle |
| 6 | **the Choice-Band tail** (D2) | no dependence beyond its shuffle control (§4.1, STALE) | the item belief through the damage kernel's own `p_cb` read | re-read first, then one screen |
| 7 | **pair history `h`** (O7 / D26) | the one obs-fed edge; never audited; the event window covers the recent history | the event seats + `r` | one screen (drop `h` from the families string) |
| 8 | **belief trunk shaping** (H12) | six aux losses reshape the trunk at an unchosen 0.05; the oracle ceiling says belief has little strength leverage at this budget | the belief heads' outputs alone (`belief_grad_mode detached`, built) | one screen |
| 9 | **the last-action block** (O6) | likely a duplicate of the newest event rows | the event seats | coverage check first (P9), then one screen |
| 10 | **the inert pair-reduction rungs** (L4) | dead code, no consumer | nothing | tech debt (T21), no screen |
| 11 | **`wasted_ko`** (P6), only as a REDUNDANCY bisect | a FACT (P(KO \| stay) × α_SWITCH), but the product of two columns the cell already carries: an interaction term the network may form itself. NOT a bias removal | its two input columns | already IN the bundle; as a lone screen, drop the one coordinate |

**Removed by the static arm if it is adopted:** the legacy board broadcast (T2) and the `non_matchup_rest` bypass (B1).

---

## 6. The rule for changing this doc

**ALWAYS-CURRENT.** Any build that ADDS, REMOVES or CHANGES a hand-computed feature (a quantity our code derives
and the model reads: an observation column the Rust encoder computes, an operator row or kernel, a pointer-cell
block, a prior table, a board-token fact, a reduction, a critic route) updates its row here IN THE SAME COMMIT:
its status, its evidence (the measured result and its pointer, or **UNVERIFIED: never ablated**) and its action. A
screen that reads a verdict on a row updates that row's evidence and action. A new row says which kind (a / b / c)
it is; a feature that is a JUDGMENT by §1's classification test is marked **J** and needs the owner; an approximate fact is marked **AF** and is fixed, not removed. `src/agents/model/CLAUDE.md` points here.
ARCHITECTURE.md stays the statement of what the model IS; where the two disagree, ARCHITECTURE wins and this doc is
the bug.

---

## 7. FINDINGS (from writing this doc)

1. **`turns_since_progress` is a JUDGMENT in production's observation.** Its "progress" definition
   (`trackers/clock.rs::is_progress`: a 3 % damage floor, eight reset conditions, Rest-loop / wasted-self-cure /
   heal-freeze and other carve-outs) is a hand opinion of what a good turn is, the class the owner dropped from the pointer
   cells on 2026-10-06. No doc classified it as one; ARCHITECTURE calls it a "no-progress clock" and the entity
   audit moves it to FIELD unchanged (K8 keeps the deadline clock, not this). Never ablated. The threshold-free
   fact alternative, "turns since either side lost HP", is the fallback if a clock signal proves needed (2026-10-09).
2. **The KO ramp is not exact.** `DamageOperator._rolls` calls `ko_ramp` "the exact realized KO probability", but it
   is a continuous ramp over `[0.85, 1.0] · dmg` (the 16 discrete rolls smoothed) and it ignores the 1/16 crit KO.
   F6b names "a crit-KO noisy-OR" as not built; no doc names the non-crit ramp's own approximation. By the 2026-10-09
   test it is an APPROXIMATE FACT (fix: exactness, §4 rank 3); the "exact realized KO probability" claim was in
   `DamageOperator._rolls`' docstring only (corrected 2026-10-09, comment-only), and `_damage_rolls`' docstring already said
   "modal no-crit". Three sites share the shape: `_rolls`, the inlined `ko_cb`, and `intent_threshold`'s sub-break ramp.
3. **Production carries the quantities the owner declined on 2026-10-06; on 2026-10-09 the owner re-examined them.** The
   factual record stands: `tempo_cost`, `neutralization`, `wasted_ko`, `spin_value_lost`, the hazard stake of
   `spin_denied` and the hand thresholds ride every production forward until the bundle adopts move resolution
   (P2, P3, P6, P8), by design (one-lever screening). The 2026-10-09 classification test (§1) re-sorts them: only
   `neutralization` (as implemented) and `tempo_cost` (mostly) are JUDGMENTS; `wasted_ko`, `spin_value_lost`, `spin_denied`'s
   stake and P8's thresholds are FACTS. So production carries two judgments, not the declined list, and the bundle's
   move-resolution arm DROPS FACTS (§4 rank 5; `move_resolution_test.py::test_spin_denied_keeps_the_fact_and_drops_the_stake`
   and ARCHITECTURE's "Dropped (judgments)" paragraph still encode the 2026-10-06 list).
4. ~~The in-flight entry-hazard input (N1) duplicates the `x` cell's rule~~ RESOLVED as built (2026-10-09): one
   rule, `DamageOperator.spikes_entry`, read by both (§3). ~~A typechange (Color Change, Transform) is read from the
   current types~~ RESOLVED (v148, 2026-10-09): the rule reads base types. The current-ABILITY read of Levitate
   (Trace, Role Play, Skill Swap) is RESOLVED too (v149): the species decides.
5. **The evidence base for KEEP is mostly stale or absent.** Of the 88 rows, the ones with a strength reading tied to
   the feature itself are the gen-3 9.6M audits (§4.1, §5.4: STALE per F22), two critic-route dV reads (gen-14), and
   the X5 / static screens (which test bundles, not single rows). Every other row is a correctness gate or
   UNVERIFIED (the list is in the report).
6. **"The critic route" in the owner's 2026-10-09 bundle list is ambiguous** here: whether it means F10's
   `--value-threat-inject off` was not verifiable from the docs.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-09 | **The doc is created** (owner request: "put an end state doc where we list what we are hand computing instead, so we know what we would add next or attempt to remove") | One always-current ledger of every hand-computed feature the model reads, grouped by where it lives, each with its kind (a exact physics / b narrow-channel fact / c Smogon prior, or J a judgment), status, evidence and candidate action; in-flight work marked; ranked add / remove lists; `src/agents/model/CLAUDE.md` points here | Folding it into ARCHITECTURE.md (which states what IS, not why or what next); one row per tensor (too fine to rank) | owner 2026-10-09; this doc §2 verified against the code at `2e357971` |
| 2026-10-09 | **N1 / N2 BUILT, OFF; the N1 hazard resolved by ONE rule** (the static port, `gen3_static_port_v1`) | N1 (`--mon-hazard-cost`) and N2 (`--move-actor-state`) built behind their own flags; the Spikes entry rule factored into `DamageOperator.spikes_entry`, read by both the `x` cell and N1 (option (a) of the orchestrator's hazard note); N1 covers BOTH sides (the brief) and is token content after the op; T7's outgoing route is now a set function | a test-only agreement check with two implementations (option (b)); N1 inside D's MLP (ends the X5 dex-table gather) | `design_static_tokens.md` §12; `static_port_test.py` |
| 2026-10-09 | **The Spikes entry rule reads BASE types** (`gen3_spikes_entry_base_types_v1`, v148; GIGO fix) | `spikes_entry` takes Flying immunity from `SPECIES_TYPE[species_ids]`, not the obs type columns (current types after Color Change / Transform / Conversion / Forecast); changes the production `x` cell for an active mon with changed types | keeping the current-type read (wrong: a switch-in reverts to base); a new base-type obs column (a layout change for a derivable fact) | `deps/pokemon-showdown` `sim/pokemon.ts` `clearVolatile` / `setSpecies`; `static_port_test.py::test_the_spikes_entry_reads_base_types_not_the_current_ones` |
| 2026-10-09 | **The Spikes entry rule reads the SPECIES' Levitate** (`gen3_spikes_entry_species_levitate_v1`, v149; GIGO fix) | Levitate from `SPECIES_TRAP_PRIOR[species, 3]` (exactly 0 or 1 in gen 3: Levitate is the sole ability of its 17 species), not the current-ability column or the revealed-ability view; changes the production `x` cell for a Traced / Role-Played / Skill-Swapped / Transformed active mon | a base-ability obs column (unneeded: the premise holds); a known-ability fallback for revealed mons (needed only if a species held Levitate beside another ability) | Showdown gen-3 pokedex (node `Dex.mod('gen3')`, num <= 386); `static_port_test.py::test_the_spikes_entry_reads_the_species_levitate_not_the_current_ability`, `::test_every_gen3_species_levitate_prior_is_zero_or_one` |
| 2026-10-09 | **FACT / JUDGMENT re-classification** (owner: "several rows marked J are actually FACTS, hard-to-compute quantities handed to the model so it doesn't spend capacity on them"; test owner-approved via the orchestrator) | §1 adopts the classification test (FACT: an event's probability or magnitude in its natural unit; JUDGMENT: mixes units with a chosen weight, assumes our later plan, or grades with a threshold; APPROXIMATE FACT: a fact with a shortcut, fix it). Re-sorted: P8 (all six) FACT, Substitute an AF; P6 `wasted_ko` FACT but redundant (a bisect candidate, §5 rank 11); `spin_value_lost` / `spin_denied` FACT, KEEP; P2 `neutralization` JUDGMENT as implemented and `tempo_cost` MOSTLY JUDGMENT, REPLACE with facts (§4 rank 4); O10 JUDGMENT, §5 rank 1; the KO ramp an AF, fix (§4 rank 3). §4 gains ranks 3-5, §5 rank 2 is rewritten and rank 11 added. The 2026-10-06 decline is kept as the factual record. `DamageOperator._rolls`' docstring ("exact realized KO probability") corrected, comment-only | keeping the old list as "judgments" (it hid facts from the model); removing the KO ramp instead of making it exact; replacing `neutralization` by a re-weighted sum (still a judgment) | owner 2026-10-09; code read at `f0869884`: `intent_threshold.py`, `switch_branch.py`, `pair_outcome.py`, `damage_op_blocks.pair_outcome_coords`, `damage_op._rolls`, `trackers/clock.rs::is_progress` |
