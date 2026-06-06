# design_pathologies.md — Model Pathology Register

A **living register** of observed model pathologies, the fixes applied, and what each fix should
change. Unlike the `ai_vN/impl_step*.md` records (what was built) and `design_*.md` (forward
plans), this is the **diagnostic thread**: *what's wrong → what we changed → what we expect to be
different next time*. **Review this before every retrain**, and after each eval add a row noting
whether the predicted change actually showed up (so we learn whether a fix worked, not just that it
shipped).

How to use it:
1. Read the **Findings** table and the **Open questions** — these are what a fresh review should
   re-check against the latest eval traces.
2. For each fix marked *implemented*, check the **"expect to see"** prediction against the new run.
   Confirm it, or record that it didn't land (and why).
3. Promote anything still open into the retrain plan.

---

## Source of these findings

- **Run:** `models/run_20260531_182804` (live, 300M-step target), arch **ai_v4**
  (`gen3_trapping_signals_v1`, obs **3321**), fixed-bot eval pool — the "pathology-hunting phase
  before self-play."
- **Evidence:** forensic eval traces at steps **17M / 18M / 19M** (per-decision obs/logits/value +
  reward breakdown + action distribution), 9 opponent classes. Full report:
  **`models/run_20260531_182804/LOSS_ANALYSIS.md`**.
- **Headline eval:** ~**63% vs bots** at 19M; worst matchups SetupSweepV2 55%, Heuristic2 57%,
  StallerV2 59%. (Single-eval win rates are ±noise over 100 battles/opp.)

---

## 1. Findings (what we found)

| ID | Pathology | Root cause | Status |
|----|-----------|------------|--------|
| **HARNESS** | Did the model click what it intended? | **No bug.** 0 mismatches in 10,352 decisions (`chosen == argmax(masked logits)`); obs re-runs faithfully; ordering-integrity guards never fired. The "clicked Thunderbolt, got Ice Beam" class does not exist. | ✅ ruled out |
| **P-0** | **All-or-nothing policy.** Every game ends 6-0 — 135/135 captured wins sweep clean, 247/248 losses are full wipes. No close games. A brittle momentum/snowball policy with no risk management. | Offense-heavy dense reward + γ=0.9999 + training only vs 8 fixed bots (nothing teaches defensive/positional play). | ⏳ needs **self-play (P0)** + reward rebalance |
| **P-1** | **Move-EFFECT blindness at the policy head.** For status/utility moves the only action-aligned signals were base power (0) + type multiplier (1.0) — *identical across protect/heal/setup/status* — so the head literally could not tell them apart. Symptom: flat ~20-30% action smears; Toxic into a Poison-type (immune) clicked 8 straight turns; Will-O-Wisp into already-burned; Leech Seed into Grass; Toxic into Immunity-Snorlax 16 turns. | INFO gap: the rich per-move features live in the per-mon block (sorted, pooled into role tokens), never action-aligned to the move logits. | ✅ **FIXED this pass** (see §2) |
| **P-2** | **Chronic under-switching / no pivoting.** Voluntary-switch rate 8-10%; in dead matchups switch prob stays 2-9% while the mon does nothing and dies. | Reward: switching is locally dominated (eats a hit for only +0.5) and the staying penalties (matchup −0.15 / dead-matchup −0.10) are too weak; `matchup_penalty` fires equally in wins and losses → it isn't teaching *when* to pivot. | ⏳ planned (P2) |
| **P-3** | **Can't close / stall loops.** Vs walls (Blissey/Skarmory/Suicune) and Random the model loops a chip move that's out-healed, never switches to a breaker, drags to 80-250 turns, sometimes PP-stalls into Struggle and loses. | `futile_attack` is only −0.05 (weaker than the repetition tax); no aggregated "opp is out-healing me" signal. | ⏳ planned (P2) |
| **P-4** | **Setup/utility misplays.** Sets up (Calm Mind/DD) at low HP, while paralysed, on the last mon, or into a faster lethal threat. | `setup_low_hp` (−0.1) and `futile_setup` (−0.3) too weak; value head's boost-illusion (below) reinforces it. | ⏳ planned (P2) |
| **VALUE** | Value head is **mostly well-calibrated** (corr with material +0.42; any deficit → ≈−22). Over-optimism while behind is rare (49/10,352 ≈ 0.5%), isolated to the **setup-boost illusion** (Calm Mind/DD stacking inflates V while down material). | — | note only |

**Correction to watch for in future reviews:** the huge `repetition_tax` totals (−721 vs Random,
−246 vs SetupSweep) are **inflated by a few long stall-out games, not pervasive** — normalize per
turn before concluding (wins ≈ losses except in stall games). Repetition is a *symptom* of P-3,
not an independent driver.

---

## 2. Fix implemented this pass — P-1: action-aligned move-effect block (`gen3_move_effects_v1`)

**What changed.** Each of the 4 **request-order** move slots (so feature slot *k* lines up with
action logit 6+*k*) now carries **9 effect features** in the reactive block, before the matchups
(so the extractor picks them up in `non_matchup_rest` → both policy and value heads):

`is_boost · is_heal · is_protect · is_phaze · is_hazard · inflicts_status · status_will_land ·
pp_fraction · status_will_land_known`

- **Static flags** (`is_boost/heal/protect/phaze/hazard`, `status_inflicted`) are derived in the
  acquisition tool (`tools/pokemon_data_extractor/sync.py:build_moves`) from the field **Showdown
  itself keys each mechanic on** — `flags.heal`, `volatileStatus∈{protect,endure}`, `forceSwitch`,
  `sideCondition`, primary `status`, declarative self-positive `boosts` — **never** guessed from
  the move name. The one callback-only boost (**Belly Drum**, `onHit: this.boost`) is a curated
  override; **Memento** is correctly excluded (foe-target debuff + self-faint); **Curse** is
  resolved live (self-boost only for a non-Ghost user).
- **`status_will_land`** is a **prior-weighted probability** in [0,1]
  (`gen3_mechanics.status_land_estimate`), built "priors first, then confirmation" exactly like the
  matchup-cell ability expectation: 0 on a certain block (type immunity / already-statused /
  Substitute), else `1 − P(ability blocks it)` over the opponent's Smogon ability prior — so an
  unrevealed Snorlax reads ≈**0.14** for Toxic, collapsing to 0/1 once the ability is revealed.
- **`status_will_land_known`** is the prior-vs-confirmed bit, routed with the **same predicate as
  the per-mon ability block's `known` flag** (`_ability_revealed`: a set ability ≠
  `"unknownability"`) **or** a type-certain hard block — so the head can tell a confirmed outcome
  from a prior estimate (closes a real discrepancy: abilities had a known bit, status didn't).

**Dims / version.** obs **3321 → 3357**, `REACTIVE_DIM` 302 → 338 (matchups shifted to
reactive-offset 50/194), `ARCH_SIGNATURE` `gen3_item_num_fix_v1` → **`gen3_move_effects_v1`**.
Retrain-class (old checkpoints fail loudly).

**Data quality — how we know it's not garbage-in** (the make-or-break of this change):
- Each static flag is sourced from Showdown's own keying field (above).
- **Reproducibility:** committed `gen3_moves.json` == builder output (`extractor_parity_test`).
- **Source faithfulness:** re-derive status/heal/protect/phaze/hazard straight from the Showdown
  `.ts` for every gen3 move → **0 mismatches** (`extractor_parity_test`, integration-gated).
  *(is_boost is gen-sensitive — gen3 Charge/Stockpile lost their gen9 boosts — so it's validated
  via the gen3-aware curated cross-check instead, not raw `.ts`.)*
- **Cross-check vs the project's independently-curated move sets** (`gen3_mechanics`,
  `reward_manager`): **0 misses**; our set is *more complete* (swallow, grasswhistle, poisongas,
  howl, tailglow) and gen3-correct (stockpile excluded — no Def/SpD boost in gen3).
- **0-power invariant:** no damaging move ever carries a utility flag (Dream Eater / Leech Life /
  Rage / Skull Bash correctly excluded).
- **Wiring (bridge fuzz, `action/move_effects_fuzz_test.py`):** ~17k move-slots over 40 real
  battles, **0 violations**, every category covered (incl. prior-fractional status, and both
  `known` states).
- Full non-e2e suite (1596) + new unit/fuzz tests pass; obs-build benchmark no regression; model
  build/save/reload round-trips at 3357.

**Key files:** `agents/observation/reactive.py`, `constants.py`, `gen3_data/moves.py`,
`gen3_mechanics.py` (`status_land_estimate`, `ABILITY_STATUS_IMMUNITY`, `_ability_revealed`),
`tools/pokemon_data_extractor/sync.py`, `model/model_version.py`.

---

## 3. What we expect to be different next time (verify against the next eval)

P-1 targets the **status/utility misplay class specifically**. Concretely, in the next run's
traces we expect:

1. **No type-immune status loops** — Toxic into Steel/Poison, Thunder Wave into Ground,
   Will-O-Wisp into Fire should ~vanish (`status_will_land` reads 0, `known`=1).
2. **Fewer wasted status into already-statused / Substitute** targets.
3. **Immunity-ability loops shrink** — Toxic into Immunity-Snorlax should stop after the first
   reveal (turn-1 prior ≈0.14 already discourages it; post-reveal `status_will_land`=0, `known`=1).
4. **Sharper utility-move distributions** — the flat ~20-30% smear across protect/spikes/toxic/
   recover should concentrate, because the head can finally tell them apart.
5. **`status_wasted` and the type-immune `futile_attack`/`dead_matchup_tax` reward penalties fire
   far less** (they were the symptom of this blindness).

**Caveat — set expectations:** P-1 alone is **unlikely to move the headline win rate much**. The
dominant pathology is **P-0 (all-or-nothing) + P-2 (under-switching)**, which need the **reward
rebalance (P2)** and **self-play (P0)** — not yet implemented. Judge P-1 by the *behavioral*
signals above (utility/status misplays), not primarily by win rate. The win-rate move should come
when P2/P0 land.

---

## 4. Open design question (parked for review) — matchup-cell prior-vs-confirmed parity

The fix above gave `status_will_land` a `known` bit so the model can tell prior from confirmed —
**matching how the per-mon ability block is routed** (`[id1, id2, dominance, known]`; ability1 is
always the higher-dominance ability, so `dominance ≥ 0.5`, and a single dominance is lossless in
gen3 because species have ≤2 abilities whose probs sum to 1). **The 144+144 matchup matrices —
and the active-mon `move_multiplier` scalar — have the *same* ambiguity and *no* known bit.**

- **The ambiguity:** each matchup cell is an *expected* multiplier that marginalizes over two
  uncertainty sources — the **opponent's ability** (Smogon prior when unrevealed → `our_matchups` +
  `move_multiplier`) and the **opponent's Hidden Power type** (tracker distribution →
  `their_matchups`). A mean hides variance: Surf into a maybe-Water-Absorb mon reads ≈0.5× (a
  coin-flip between a hard wall and free damage), *indistinguishable from a confirmed 0.5×*. HP-type
  blends can mix 0× and 4× into a meaningless middle.
- **Why it's lower-value than the status bit was:** the confirmed-ness info already exists in the
  obs — the per-mon ability block's `known`/`dominance` and the per-mon HP-candidate block's
  `hp_revealed` + distribution — it's just **not co-located** with the matchup cells (per-mon
  blocks feed the pooled role tokens; matchup cells feed a separate attention path). And the
  high-variance cells are a minority: single-ability species (260) and revealed mons have
  `dominance=1.0` → already effectively confirmed.

**Options if we close it:**

| Opt | What | ~Dims | Notes |
|-----|------|-------|-------|
| A | per-**cell** confirmed bit | +288 | doubles the matchup block; hugely redundant (confirmed-ness is per-mon/per-move, not per-cell) |
| B | per-**opp-mon** "ability-confirmed" bit (+ HP-type-known already in the HP block) | ~6–12 | matches where the uncertainty lives; cheap |
| C | per-mon ability **entropy/variance** scalar (not a binary) | ~6–12 | richer — distinguishes a coin-flip 0.5 from a confirmed 0.5 |
| D | do nothing; rely on the per-mon `known`/HP blocks the net already has | 0 | status quo |

**Recommendation: measure before building.** Unlike P-1 (motivated by a concrete observed failure),
we have **no evidence yet** the model misvalues matchup variance. Before spending obs dims + a
retrain, **probe the traces** for it — e.g. does the model over-commit into a *possible*-immunity
(Surf into a maybe-Water-Absorb, EQ into a split-Levitate mon, or an HP move whose type is still a
coin-flip)? If the probe shows it, prefer **option C per-mon** (entropy is more informative than a
bit, only ~6–12 dims). If not, leave at D.

→ **Action for next review:** add a matchup-variance probe to the loss-analysis toolkit and decide
B/C vs D from evidence.

---

## 5. Retrain roadmap (priority order; only P-1 is implemented)

- **P0 — self-play / league** *(biggest lever on P-0; code landed, not yet run — see ai_v5)*.
- **P1 — action-aligned move-effect obs** — ✅ **implemented this pass** (§2).
- **P2 — reward rebalance** *(planned)*: strengthen `futile_attack` (−0.05 → ≈−0.25); escalate the
  staying penalties *and* make the pivot positively worth it; dense clipped material-advantage
  shaping `+k·(our_alive − opp_alive)`; setup discipline (penalize setup while paralysed / last-mon
  / into a revealed faster lethal threat).
- **P3 — hyperparameters** *(planned, A/B)*: anneal `ent_coef` (0.058) down late (the ~50% top-1
  flat distributions suggest over-regularization); try γ 0.9999 → 0.999 for near-term credit
  assignment on pivots.
- **P4 — closing-out** *(planned)*: a finish incentive / earlier `stall_tax` ramp vs a last opp
  mon; `pp_fraction` (added in P-1) already lets the model see Struggle approaching.

**Success signal for the whole program:** close games start appearing (3-2 / 2-1 outcomes), the
voluntary-switch rate rises, and the V2-setup/staller win rates close the gap to their V1 versions.

---

## 6. Fix implemented (2026-06-06) — critic tail-blindness: incoming-damage belief (`gen3_incoming_damage_v1`)

**New source.** This entry is from a *later* run than §1's: forensics on `run_20260601_193826`
(122M, self-play; `models/run_20260601_193826/LOSS_ANALYSIS_2026-06-05.md`). It makes the **value-side
of P-0/P-4 concrete**: the policy loses by **one or two mispriced catastrophic decisions per game** —
the critic sits at strongly positive V, the actor commits, then an opponent answer it never priced
lands and V crashes −10..−59. Three faces of one root: (1) the obs had *effectiveness* (`2×`), never
the **OHKO threshold** (`power·A/Def·eff·STAB·roll ≥ HP`); (2) `their_matchups` is built from
*revealed* moves only → **blank exactly on a just-switched-in mon** (the Claydol case, 89% coverage
blank); (3) the incoming lane was **barely attended** (saliency ≈0.0018 vs own-offense ≈0.47).

**Gating sequence (so we didn't over-build).** The cheap critic-first lever was tried first — a
**no-vf-clip** run (122M→158M). It did **not** dissolve the tail-blindness: within-run on the
zero-retrain falsifier the unpriced-incoming-KO cliffs **persisted** (cliff rate 10.0%→8.6%, −14%
only; decisive-turning-point structure essentially unchanged). That residual was the clean GO.

**What changed.** A 33-dim **incoming-KO belief block** (per our mon: phys/spec expected-chip +
mode-max P(KO) + P(outspeed); then opp recovery scalars), routed via `non_matchup_rest` to both heads.
A calibrated *belief* under hidden info (revealed ∪ Smogon-usage candidates; offensive-stat
distribution → tail P(KO) + mean chip; closed-form 16-roll P(KO); P(outspeed) over the Speed
distribution), gen3-faithful for KO-relevant hits (incl. the review-found Explosion/Self-Destruct
Def-halve, which is gen≤4; the review's Sandstorm-SpD claim was a gen4 mechanic and was dropped).
obs **3357 → 3390**, `ARCH_SIGNATURE` `gen3_move_effects_v1` →
**`gen3_incoming_damage_v1`**. Full as-built: `designs/ai_v5/impl_step4_incoming_damage_obs.md`;
design + go/no-go: `designs/ai_v5/design_incoming_damage_obs.md`.

**What we expect to be different next time (verify against the next eval — this is Gate 2, NOT yet
confirmed).** This is a **necessary-but-not-sufficient** fix — the binding gap is policy-side
under-switching, so a critic-side obs feature only helps if the retrained model both *attends* to it
and the policy *acts*:
1. The incoming block's **saliency rises ≥10–50×** from ~0.002 (else the lane is still dead).
2. The **CRITIC_BLINDSPOT turning-point count drops** on a fresh forensics pass (fewer
   optimistic-V → unpriced-incoming-KO cliffs).
3. **Secondary:** voluntary-switch rate into a flagged-lethal matchup rises (the policy *acting* on
   the new valuation) — the true test, and the one most at risk from chronic under-switching.

If saliency rises but switching doesn't, the residual is policy-side (P-2) and needs the reward
rebalance / self-play, not more obs. If the cliffs collapse and switching rises, the fix worked.

**Roadmap slot.** This is the **value-side** complement to the P0/P-2/P-4 levers in §5 — it fixes the
*valuation* (the info gap), while P2 (reward rebalance) + P0 (self-play) must fix the *action*
(under-switching). Judge it by the three signals above, not headline win rate.
