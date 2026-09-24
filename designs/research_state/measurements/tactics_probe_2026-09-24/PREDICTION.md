# Tactics probe battery: ACTION DENIAL and LURES (pre-registration, 2026-09-24)

**Status: REGISTERED before any endpoint number was read.** What had been seen at commit time is listed
in §8. The verdict is read at the registered n (§4) and nowhere else.

## 1. The question

Do our policy and its win-prob critic understand two gen-3 tactical concepts?

* **Family D, ACTION DENIAL** (`designs/endstate/obs_enrichment_backlog.md` §1a, E12). In gen-3 singles,
  ANY faint mid-turn cancels every remaining queued action (Showdown `sim/battle.ts` `faintMessages`,
  `gen <= 3 && singles` → `queue.cancelAction` for every active mon). So a FASTER Gengar's Explosion
  denies the slower Tyranitar's Dragon Dance **whether or not Explosion KOs**, because Gengar's own faint
  ends the turn. When Tyranitar is faster, the Dragon Dance resolves first and Explosion comes after.
  **The denial contrast is SPEED ORDER**, not Tyranitar's HP (the orchestrator's first framing, an
  HP-threshold contrast, was corrected by the owner the same day; HP is kept as a SECONDARY factor that
  changes the sack's value). The Blissey case is the same rule: a faster attacker killed by its own
  Double-Edge recoil denies Blissey's Soft-Boiled.
* **Family L, LURES.** How much belief the model puts on a lure tech before the reveal, whether the
  belief snaps after it, and whether the policy hedges.

## 2. Verified before registration (design inputs, not endpoints)

* **The rule, in both engines.** `scripts/verify_rule.py`, four constructed battles, run on `impl=rust`
  (the pin's `sim_bridge`) AND `impl=node` from the same seeds: the normalised per-side protocol is
  IDENTICAL across engines in all four, and (A) fast Gengar Explosion: Tyranitar's Dragon Dance never
  appears, no boost; (B) Tyranitar at +1 (319 > 308): Dragon Dance resolves first (+2), then Explosion;
  (C) Tyranitar's Double-Edge recoil KO: Blissey's Soft-Boiled never appears; (D) no recoil KO:
  Soft-Boiled resolves. **The port agrees with Showdown; no port bug.**
* **Damage in the real (Rust) sim** (`scripts/damage_ranges.py`, 120 seeds): STD Gengar (Timid, 0 Atk)
  Explosion into the family-D Tyranitar (387 HP): **104–123 non-crit (26.9–31.8%)**, 211–243 crit.
  Explosion does NOT KO from full HP; it KOs only at Tyranitar HP ≤ 103 for certain (≤ 123 possibly).
  The family-B Tyranitar's Double-Edge (at +0) into Blissey: 264–311 (crit 534–609); recoil **88–103**
  (crit 178–203). The family-B states carry Intimidate (−1 Atk), so their recoil is smaller; each
  B state's P(recoil KO) is measured from its own rollouts (§5 E7), never assumed.
* **Speeds** (from the sim's own request stats): STD Gengar 308, FAST Gengar 335 (the STD set with its
  100 SpD EVs moved to Spe; HP and Def unchanged), Tyranitar 213 at +0 and 319 at +1.

## 3. Models, tree, engine

* G0 = `ai_v13_12_plateau`, B = `ai_v13_22_popr1_loop`, C = `ai_v13_23_popr1_ctrl`, each
  `final_model.zip` (sha256 in `out/states.json`), loaded through `main.capacity.load_policy`
  (`check_compatible`, no arch drift allowed) by **tree `6eb9c776`** (all three trained on that pin;
  their own pin, not the 7c511161 tooling tree: nothing newer was needed). Detached worktree
  `gen3ai-wt/tactics-pin-6eb9c776`, its own Rust `sim_bridge` (`POKESIM_SIM_BRIDGE_BIN`), cwd = the pin
  tree (its `data/` equals main's for every file read here), CPU, `torch` 1 thread, nice 15.
* Critic: all three run `critic=winprob`; the value head IS the win-probability head (V is read from
  `predict_values`; the separate `win_prob` stash is recorded beside it).

## 4. The states and the oracle

**States** (`scripts/states.py`, `scripts/lure.py`): fixed Smogon-sample teams (every set is named by its
`data/teams/sample/` file; the few modifications are stated in `scripts/teams.py`), a SCRIPTED prefix
(both sides' choices turn by turn), a fixed sim seed, the probe turn T. The seed is the first
`k = 0, 1, …` of `mint_seed(seed_key:k)` whose prefix meets every predicate of the state (HP band,
boosts, status, reveals, speed order, side conditions); `scripts/find_seeds.py` registered all 39
(`out/seeds.json`). `scripts/build_states.py` then replays each prefix with the continuation players,
captures the obs at T and RE-CHECKS every predicate; a state that fails is REFUSED, not measured.

| Family | States | Model plays | What varies |
|---|---|---|---|
| DG core (GL) | `DG_GL_{fast0,slow1,fast1}_{high,mid,low}` (9) | Gengar (p1) | speed × Tyranitar HP (96% / 64–67% / 26%, where 26% = 99–100 HP < 104, so Explosion KOs) |
| DG secondary (SL) | `DG_SL_K1..K5b` (6) | Gengar | at the fast, full-HP corner: Pursuit absent / unrevealed / revealed; Spikes (their side) up / not; Swampert weakened (55%) / healthy |
| DT mirror | `DT_GL_{fast0_low,fast0_high,fast1_low,slow1_low}` (4) | Tyranitar (p2) | the same boards as DG, other side |
| B | `B_low` (TTar 13%), `B_high` (TTar 41%); Blissey 74–75% | Double-Edge Tyranitar (p1) | our HP across the recoil-KO threshold |
| L | `L1..L6 × {pre_lure, pre_std, post}` (18) | the side facing the lure user | before / after the reveal; the two PRE worlds differ ONLY in the lure move slot |

Speed arms: **fast0** = TTar +0, STD Gengar (308 > 213); **slow1** = TTar +1 after one Dragon Dance
(319 > 308); **fast1** = TTar +1, FAST Gengar (335 > 319). **fast1 vs slow1 is the CLEAN pair**: the same
prefix and the same +1 Tyranitar; only our Gengar's Spe/SpD EVs differ (visible to us in our own spread
block; Tyranitar's speed stat is hidden, so reading the order requires the spread belief). **fast0 vs
slow1 is the NATURAL pair** (the boost itself flips the order, and also raises the threat).

Lures (Smogon prior p = `data/pokemon/gen3_move_priors.json`): L1 Tyranitar HP Grass → Swampert
(p 0.089); L2 Metagross Thunder Punch → Skarmory (0.012); L3 Gengar Focus Punch → Tyranitar (0.023);
L4 Celebi HP Fire → Skarmory (0.239, a COMMON tech as a contrast); L5 Blissey Fire Blast → Metagross
(0.040); L6 Salamence Hydro Pump → Tyranitar (0.021, the brief's example). L1, L2, L4 and L5's lure worlds
are the sample sets verbatim (so they are ON-POOL sets); L3 and L6 lure worlds are modified sets (OFF-pool).

**The ORACLE** (`scripts/rollouts.py`). For every state, regime and legal action a: rollouts to a
terminal from the real state. Prefix on the state's seed → at T our side plays a → the sim PRNG is
reseeded at the start of T (`resumeReseed`; verified to vary the dice on the Rust bridge) → both sides
play on as the FIXED CONTINUATION POLICY, **G0 stochastic at T = 1** (each player its own seeded
sampler). Common random numbers: rollout r uses the same reseed and the same two policy seeds for every
action. Score: win 1 / loss 0 / tie 0.5; a line that reached the stall cap is 0.5 whatever it says.
**Regimes:** LIVE (the opponent's turn-T move is chosen by the continuation policy: the true-world value
against that opponent; this is THE ground truth for correctness) and PREMISE (the opponent's turn-T move
is scripted: DG Dragon Dance, DT Explosion, B Soft-Boiled; the owner's scenario taken literally). DG_SL
and L run LIVE only.

**REGISTERED n:** R = **48** rollouts per (state, regime, action) for DG_GL, DT and B; R = **24** for
DG_SL and L. 9 legal actions per state. ~18,100 rollouts. A state-regime whose non-ok rollouts
(timeout / error) exceed 25% is **INCONCLUSIVE**, never a result.

**Readouts per model M ∈ {G0, B, C} per state** (`build_states.py`, obs captured at T):
π_M(a) (masked softmax, T = 1), V_M(s), win-prob; for L, the opp-active move-belief P(lure move)
(typed id for Hidden Power) and the model's own Smogon prior buffer. **Critic after the action:** in
every rollout, V_M and win-prob of all three models at OUR first decision after a (mean over rollouts
= V̄post_M(a)).

**Statistics.** Q(a) = mean score over rollouts; 95% CIs by a paired bootstrap over rollout index r
(2,000 resamples, seed 20260924; all actions of a state resampled together). Contrasts across two
DIFFERENT states are independent bootstraps. **Tied-best set** = {a : the paired CI of Q(a*) − Q(a)
contains 0}, a* = argmax Q.

## 5. Endpoints and decision rules

Let A_X(s) = Q(X) − max_{a ≠ X} Q(a) (the advantage of action X), Expl = Explosion.

* **E1 (primary; policy denial).** Δπ_M = π_M(Expl | fast) − π_M(Expl | slow) at each HP level, for the
  CLEAN pair (fast1 − slow1) and the NATURAL pair (fast0 − slow1).
  **DENIAL-AWARE** iff Δπ ≥ +0.05 at ≥ 2 of 3 HP levels and ≤ −0.05 at none; **INVERTED** iff
  Δπ ≤ −0.05 at ≥ 2 of 3; **BLIND** iff |Δπ| < 0.05 at ≥ 2 of 3; else **MIXED**. Per model, per pair.
* **E2 (primary; does denial move the true value?).** ΔA = A_Expl(fast) − A_Expl(slow) per HP level
  and pair, PREMISE and LIVE. **DETECTED** iff the CI excludes 0 (positive) at ≥ 2 of 3 HP levels on
  the clean pair in PREMISE. If NOT DETECTED, a BLIND policy is not an error at these states and is
  reported that way.
* **E3 (correctness).** Per state and model: greedy action ∈ tied-best set (LIVE); π_M(tied-best);
  expected regret Q(a*) − Σ_a π_M(a) Q(a); greedy regret. Summaries: agreement rate over the 39 states
  and per family.
* **E4 (critic).** Per state: Spearman ρ across actions between V̄post_M(a) and Q(a) (LIVE).
  Critic denial: ΔC_M = [V̄post_M(Expl) − V̄post_M(best non-Expl by Q)]_fast − [same]_slow;
  **CRITIC DENIAL-AWARE** iff ΔC > 0 with CI excluding 0 at ≥ 2 of 3 HP levels (clean pair, PREMISE).
  Sack bias: V̄post_M(Expl) − Q(Expl) averaged over the DG states.
* **E5 (secondary; the KO threshold).** π_M(Expl | fast0_low) − π_M(Expl | fast0_high), and the same
  in the oracle (A_Expl). **KO-AWARE** iff ≥ +0.05.
* **E6 (mirror).** π_M(Dragon Dance) and A_DD at the four DT states. **RESPECTS DENIAL** iff
  π(DD | fast0_low) and π(DD | fast1_low) are both ≤ π(DD | slow1_low) − 0.05.
* **E7 (Blissey).** P(recoil KO | Double-Edge vs Soft-Boiled) per B state from its PREMISE rollouts;
  Δπ_M(DE) = π(DE | B_low) − π(DE | B_high); A_DE per state. **RECOIL-DENIAL-AWARE** iff Δπ ≥ +0.05.
* **E8 (lures).** (a) b_pre = P_M(lure) at PRE (identical obs in both worlds is ASSERTED); class
  OVER-BELIEVES if b_pre > 2p + 0.05, UNDER-BELIEVES if b_pre < p/2 (p ≥ 0.02), else PRIOR-LIKE.
  (b) b_post at POST; **SNAPS** iff ≥ 0.9 (non-HP lures; for the two HP lures the protocol never names
  the type, so the typed belief is reported, not scored). (c) hedging: Δπ_victim = π(switch to victim |
  post) − π(switch to victim | pre); Bayes value Q_bayes(a) = p·Q_lure(a) + (1 − p)·Q_std(a) (p = the
  Smogon marginal; the model's own b_pre as a secondary weight); is "switch to victim" in the Bayes
  tied-best set, and the lure's cost Q_std(victim) − Q_lure(victim).
* **E9 (M3 note).** For every KO-line state, does the event log / event window show the denial?

## 6. Predictions

1. **E2:** denial DETECTED in PREMISE on the clean pair (a denied +1 → +2 Dragon Dance is worth a lot);
   in LIVE, smaller.
2. **E1 clean pair:** all three models **BLIND** (the order is decided by a 335-vs-308 spread
   difference against a hidden Tyranitar speed; the network has no denial concept or signal).
3. **E1 natural pair:** **INVERTED or MIXED**: π(Expl | slow1) > π(Expl | fast0) at ≥ 2 of 3 HP levels
   (a +1 Tyranitar looks more threatening, which pulls toward the trade even though the trade is worse).
4. **E5:** **KO-AWARE** in all three models (the damage operator sees a 26% Tyranitar).
5. **Oracle best (LIVE):** at `DG_GL_fast0_high` a switch (Swampert or Starmie); at `DG_GL_fast0_low`
   Thunderbolt or Explosion (the KO).
6. **E4:** mean Spearman ρ ≥ 0.3 across the DG states; the critic is **sack-averse**: V̄post(Expl) is
   below Q(Expl) by ≥ 0.05 on average; CRITIC DENIAL-AWARE: no, for all three.
7. **E6:** the Tyranitar side does NOT respect denial: π(DD | fast0_low) ≥ 0.2 and not below
   π(DD | slow1_low).
8. **E7:** recoil KO certain in B_low (P ≥ 0.95) and absent in B_high (≤ 0.05); **not
   RECOIL-DENIAL-AWARE** (|Δπ(DE)| < 0.1); A_DE(B_low) > A_DE(B_high).
9. **Models:** G0, B and C agree with the oracle at rates within 0.05 of each other; none is
   DENIAL-AWARE.
10. **E8(a):** the ON-POOL lure sets (L1, L2, L4, L5) are OVER-BELIEVED pre-reveal (pool memorisation,
    as the 2026-09-24 belief-calibration read found for revealed-slot moves); the OFF-pool ones (L3, L6)
    are PRIOR-LIKE or UNDER.
11. **E8(b):** the four non-HP lures SNAP (revealed moves are pinned in the posterior); the two HP lures'
    typed belief stays below 0.9.
12. **E8(c):** Δπ_victim ≤ −0.05 in ≥ 4 of 6 lures (the policy reacts to the reveal); pre-reveal,
    "switch to victim" is in the Bayes tied-best set for the rare lures (L2, L3, L6).

## 7. Procedure

`find_seeds.py` → `build_states.py` (refuses a state that fails a predicate) → `rollouts.py` on 3
workers at nice 15, detached (`nohup … < /dev/null &`), units of 4 rollouts × every legal action per
state-regime (≈ 2–3 min), rows appended durably to
`~/gen3ai_archive/tactics_probe_2026-09-24/rows/w<k>.jsonl` with an fsync per rollout; a restart skips
every rollout on disk (one kill + resume is exercised on the real run and reported). Interim reads are
for progress and health only. `analyze.py` runs once, at the registered n, and writes
`out/results.json` + the README.

## 8. Seen before this commit (disclosed)

* The rule verification, damage ranges and speeds (§2), and every state's snapshot at T (HP, boosts,
  reveals: `out/seeds.json`).
* The full build: 39 of 39 states passed every predicate at T and none was refused; the PRE-lure and
  PRE-std obs are byte-identical for all six lures (asserted). Model readouts were written to the
  manifest, not printed or opened.
* A 3-state build pilot printed G0's V at `DG_GL_fast0_high` (0.828) and `L1_pre_lure` = `L1_pre_std`
  (0.854). No other model readout was printed or opened (the full build writes readouts to the manifest
  without printing them).
* A 36-rollout cost pilot on `DG_GL_fast0_high` LIVE (r = 0–3, separate directory, EXCLUDED from the
  verdict): all ok, 3.8 s per rollout; its turn-T lines showed the G0 Tyranitar choosing Dragon Dance in
  all four draws and Explosion denying it. No rollout score was read.

---

## Amendment 1 (2026-09-24, owner addition: family R, ROLES). Registered BEFORE any number of any family was read

**What changed and why.** The owner added a third question mid-run: does the policy / critic PRICE a
Pokémon's role, and know that a role's value depends on the opponent's roster? Thirteen states were
added (`scripts/r_family.py`, and one board in `scripts/states.py`), found and proven exactly like the
others (`out/seeds.json`; all 13 pass every predicate at T, none refused). The original 39 states, their
endpoints and their predictions are UNCHANGED. The rollouts of the original 39 were already running
when this was written; none of their rows has been read (only row counts, for progress).

**States.**

| State | T | Model plays | Board |
|---|---|---|---|
| `R1_entry` | 1 | our Swampert (SkarmBliss) | vs their Choice Band Aerodactyl lead (the `9283210847f806ee` sample set verbatim: Jolly 224 Atk / 32 SpD / 252 Spe, EQ / Rock Slide / Double-Edge / HP Bug; 393 Spe in-sim). The BELIEF read |
| `R1_{alive,fainted}_sk{healthy,weak}` | 4 | our Skarmory vs their Zapdos (Zapdos took one Hydro Pump in every arm: 75–80%) | Aerodactyl ALIVE (retreated turn 1, full HP) or FAINTED (Hydro Pump KO turn 1) × our Skarmory HEALTHY (100%) or nearly GONE (21–22%, a Thunderbolt). Residual asymmetry: in the FAINTED arms our Swampert took a Hidden Power Ice (76%) |
| `DG_GL_slow2_high` | 3 | our Gengar | the DG high-HP board at **+2** (426 Spe; two Dragon Dances). With `DG_GL_fast0_high` (+0) and `DG_GL_slow1_high` (+1) it is the **R2** series. It joins the DG core (LIVE + PREMISE, R = 48) |
| `R3_pursuit` | 1 | our special Pursuit Tyranitar (`e541f7be8713393c`: Crunch / Pursuit / Fire Blast / Brick Break) | vs their Gengar lead (SkarmBliss) |
| `R3_{alive,fainted}` | 3 | our fresh Tyranitar vs their Swampert | their Gengar retreated from our Choice Band Metagross on turn 1 (ALIVE, full HP; their Swampert took the Meteor Mash: 83%) or stayed and was KO'd (FAINTED; Swampert 100%, our Metagross took a Thunderbolt: 63%) |
| `R3m_lead_{gengar,blissey}` | 1 | the SkarmBliss side (p2) | our Pursuit Tyranitar leads vs their Gengar / their Blissey: P(Pursuit) on OUR Tyranitar |
| `R3m_swin_{gengar,blissey}` | 2 | the SkarmBliss side (p2) | our Tyranitar SWITCHES IN on their Gengar / their Blissey |

**Oracle.** LIVE only, R = **24** per (state, action), except `DG_GL_slow2_high` (DG core: LIVE + PREMISE,
R = 48). ~3,500 more rollouts. Same continuation policy (G0, T = 1), CRN, scoring and INCONCLUSIVE rule.
**Extra readout (R states only):** the opp-active belief stashes (move belief top-12 and named moves,
item belief top-5 and named items, believed derived stats).

Let V̄ = Σ_a π_G0(a)·Q(a) (the state's oracle value under the continuation policy's own first move) and
Q* = max_a Q(a). "Stay" = any move of the active mon (as opposed to a switch).

**Endpoints and rules.**
* **R1a (belief):** P_M(Choice Band) (Smogon prior 0.761) and the top-4 believed moves.
  **OFFENSIVE LEAN** iff P(Choice Band) ≥ 0.5 AND the top-4 believed moves are all attacking moves.
* **R1b (critic prices the check):** the interaction I_V = [V_M(alive, healthy) − V_M(alive, weak)] −
  [V_M(fainted, healthy) − V_M(fainted, weak)]; the same on the oracle (I_Q on V̄ and on Q*, independent
  bootstrap over the four states). **CRITIC ROLE-PRICED** iff I_V ≥ +0.03. The oracle's I_Q says whether
  the role really is worth something on these boards (DETECTED iff its CI excludes 0).
* **R1c (policy keeps the check):** Δπ_stay = π(stay | alive, healthy) − π(stay | fainted, healthy).
  **PRESERVES THE CHECK** iff Δπ_stay ≤ −0.05; compared with the oracle's A_stay = max_stay Q − max_switch Q.
* **R2 (threat priced):** V_M at +0 / +1 / +2. **THREAT PRICED** iff V decreases monotonically and
  V(+0) − V(+2) ≥ 0.05; the oracle V̄ and Q* series beside it; the policy's mass on {switch to Swampert
  or Skarmory}, {Explosion}, {attacks} per boost (the denial is available only at +0).
* **R3 (role depends on the roster):** V_M(R3_alive) vs V_M(R3_fainted); Δπ_stay = π(stay | alive) −
  π(stay | fainted): **ROLE-AWARE** iff ≤ −0.05 (Tyranitar preserved while the Gengar it exists to trap
  is alive), read against the oracle's A_stay in each. `R3_pursuit`: π(Pursuit) and A_Pursuit.
* **R3m (belief mirror):** b = P_M(Pursuit on our Tyranitar), Smogon prior 0.134. **RAISES** iff
  b(front = Gengar) − b(front = Blissey) ≥ 0.05, at the LEAD pair and at the SWITCH-IN pair. Normative
  note, registered here: gen 3 has NO team preview, so a set cannot depend on the opponent's roster; at
  the LEAD pair the right answer is NO rise. After a Tyranitar switches INTO a Gengar, a rise is licensed
  by the player's behaviour, not by the roster.

**Predictions (amendment 1).**
13. **R1a:** OFFENSIVE LEAN in all three models (the item prior alone is 0.76 Choice Band).
14. **R1b:** the critic is NOT role-priced: |I_V| < 0.03 in all three; V(alive) < V(fainted) in both
    Skarmory arms (material). The oracle's I_Q is positive but NOT DETECTED at R = 24.
15. **R1c:** NO preservation (|Δπ_stay| < 0.05).
16. **R2:** THREAT PRICED in all three (V falls by ≥ 0.05 from +0 to +2); at +2 the policy puts ≥ 0.5 on
    switching to Swampert or Skarmory.
17. **R3:** V(alive) < V(fainted) in all three; NOT ROLE-AWARE (|Δπ_stay| < 0.05); at `R3_pursuit`
    π(Pursuit) < π(Crunch).
18. **R3m:** NO rise at the LEAD pair (correct) and NO rise at the SWITCH-IN pair (the behavioural
    evidence is not read), |Δb| < 0.05 in both.
