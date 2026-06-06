# Design — Incoming-Damage / OHKO Belief Observation (ai_v5)

> **BUILT & SHIPPED 2026-06-06 (commit `34b4724`).** The as-built record — file structure, the
> as-built-vs-this-plan scope deltas, gates, and what's deferred — is
> **`impl_step4_incoming_damage_obs.md`**; this doc is the forward design (the reasoning, the
> go/no-go, the rejected alternatives) and is retained as the design thread. What landed: priors
> (`gen3_{move,spread,item}_priors.json` + `gen3_data.priors` accessors + `stat_distribution` +
> public `gen3_stat`), the **pure belief math** (`observation/incoming_damage.py`), and the
> battle→belief glue behind a single `encode_block` in a **separate deep module**
> (`observation/incoming_damage_encoder.py`; `reactive.py` just calls it). 33-dim block at reactive
> offset 50 → both heads via `non_matchup_rest`; `ARCH_SIGNATURE = gen3_incoming_damage_v1`; obs
> **3357 → 3390**. Gates green: full unit suite (1834), math + new bridge **invariants** fuzz
> (`incoming_damage_fuzz_test.py`), golden-obs fixture regenerated + byte-exact parity, obs-build
> benchmark +7.7% vs pre-feature (under the 10% gate, after per-species `lru_cache`), model roundtrip
> + `--debug` smoke. Docs updated (root/observation/model/prober CLAUDE.md).
>
> **As-built scope (tighter than §5 — see impl_step4 for the table):** built the closed-form
> `gen3_damage_max`/`p_ko` (a fresh poke-env-free core, *not* the §5.2 `_estimate_damage_fraction`
> reuse), the constant fixed-damage branch (Seismic Toss/Night Shade/Dragon Rage/Sonic Boom), the
> full modifier set incl. the review-found **Explosion/Self-Destruct Def-halve** (gen≤4 — confirmed
> in the gen3 sim), Substitute zeroing, the §5.3b **speed-as-probability** scalar, and the
> recovery `cures_status` scalar from `design_stall_recovery_obs.md`
> ([[project_stall_recovery_analysis]]). **Deferred to v2:** the §5.3 log-space stats + immune-bit
> gating, §5.2c turn-shifted/residual-into-HP, HP-relative/reflective/OHKO candidates, the **CB
> worst-case channel** (item-prior data is built but *not yet consumed* — CB attackers are
> under-priced ~1.5× on the physical channel), per-mon-token placement, and **Gate 1** (the
> model-free calibration oracle; the shipped fuzz checks invariants, not calibration).
>
> **Gate 2 still pending a retrain.** The CONDITIONAL-GO verdict was *necessary-but-not-sufficient*:
> the binding gap is policy-side under-switching, so the retrain must move the *policy*, not just the
> critic. After a training run on the new arch, re-run the falsifier + a forensics pass — saliency
> must rise ≥10–50× from ~0.002 **and** the CRITIC_BLINDSPOT turning-point count must drop (§6 Gate 2).
>
> **Why we built it (the gating sequence that led here).** The cheap critic-first lever (a no-vf-clip
> run, 122M→158M) did **not** dissolve the tail-blindness — within-run on the falsifier the
> unpriced-incoming-KO cliffs PERSISTED (cliff rate 10.0%→8.6%, −14% only; decisive-TP structure
> essentially unchanged). That residual was the clean GO to build this scoped feature. Original
> verdict (2026-06-05): **CONDITIONAL-GO (medium confidence)** from a zero-retrain falsifier +
> 5-agent adversarial verification — the critic is provably tail-blind to unpriced incoming hits at
> ~half of decisive loss turning points (necessary), the sufficient conditions await Gate 2.
>
> Artifacts: this doc · `impl_step4_incoming_damage_obs.md` (as-built) · the falsifier
> `designs/ai_v5/falsifier_cliff_attribution.py` · the prober `threats` decode (commit `70dbbcd`) ·
> forensics `models/run_20260601_193826/LOSS_ANALYSIS_2026-06-05.md`.

---

## 1. Problem — what loses games

Forensics on `run_20260601_193826` (122M, self-play) found the policy loses by **one or two mispriced
catastrophic decisions per game**, not gradual disadvantage: the critic sits at strongly positive V,
the actor commits (attack / setup / stay / pivot), then an opponent answer it never priced lands and V
crashes by **−10 to −59** (turning-point td-residual, almost always an *unanticipated* negative
surprise; wins have worst-drop −2 to −10). The prober pinned three mechanisms:

1. **Effectiveness, not damage.** The obs `their_matchups` block is a *type-multiplier* matrix —
   "2× effective" ≠ OHKO. The KO threshold (`power · A/Def · eff · STAB · roll ≥ HP`) is never computed.
2. **Blank exactly when it matters.** `their_matchups` is built from *revealed* moves only, so a
   just-switched-in mon contributes **zeros** (the Claydol switch-in: `revealed_frac = 0.11` — 89% of
   the opponent's coverage was blank in the obs).
3. **Barely attended.** Saliency: own-offense ≈ 0.47 vs incoming ≈ 0.0018 (~260×).

This is one root with three faces — **critic tail-blindness → greedy self-setup + under-switching** —
and it reproduces identically vs the self-play pool, so self-play is not eroding it. Critically, the
average critic is *fine* (explained_variance 0.83); the failure is in the **tail**, and the live
`clip_fraction_vf ≈ 0.70` means the large value errors the OHKOs produce are being clipped away during
training. So the root is fundamentally a **critic-calibration** problem.

## 2. Investigation (how we got to the verdict)

**(a) 6-lens design review** of the v1 design (gameplay, ML-arch, data/perf, calibration, red-team,
simplicity) caught three **code-verified blockers** that are now baked into the design below:
- **Fixed-damage moves read 0 threat.** Seismic Toss, Night Shade, Counter, Mirror Coat, Super Fang,
  Endeavor, Dragon Rage, Sonic Boom, OHKO moves all have `basePower=0`, and `gen3_data/moves.py`
  buckets `basePower≤0` as STATUS → dropped from both candidate pools → the feature would price
  Seismic-Toss-Blissey / Counter at **P(KO)=0** (the project's own heuristic bot has this exact bug).
- **A false placement premise.** v1 claimed `their_matchups` was "buried" and proposed re-routing it.
  Verified in `features_extractor.py`: it's *already* on the per-mon-token → both-heads path, and
  `non_matchup_rest` is a mutually-exclusive flat lane. The ~0.002 saliency is an **information**
  problem (the block is uninformative + blank), not placement.
- **The spread→mode classifier no-ops on the flagship losses.** Mantine's Ice Beam OHKO and Suicune's
  Hydro Pump come off **0-SpA spreads**, so a "modal offensive spread" statistic under-prices the
  exact OHKOs the feature exists to flag (~11 top-OU mons affected).

**(b) Zero-retrain cliff-attribution falsifier** (`falsifier_cliff_attribution.py`, model-free).
Question: would a never-blank, prior-aware incoming-damage feature have helped the frozen 122M critic
at the value cliffs that lose games? Method: for each big value drop (`delta_v < −8`) in the loss
corpus, attribute it; the **decisive turning point** = the first cliff per loss battle with
`V_before > 5` then `delta_v < −15`. Result (263 decisive loss turning points):

| bucket | share |
|---|---|
| **Addressable** (optimistic V + big unpriced incoming hit + blank coverage) | **A 46% + SETUP 10% = 56%** |
| decisive TP was a big **incoming hit** | 66% |
| …of those, opponent coverage was **blank** (info genuinely missing) | 86% |
| **Info-present** (coverage revealed, critic ignored anyway — the "dead lane" risk) | 8% |
| **Not addressable** (stall/heal resets, post-faint reprice, already-losing, RNG) | 34% |
| our own Explosion/Self-Destruct | 1.5% |

Robust across thresholds (addressable holds **42–64%**). Measurement is clean (verified: 99% of
big-hit cliffs had the opponent execute a damaging move; ~1% recoil; `our.hp_delta` is a clean
incoming proxy). The E_OTHER bucket genuinely isn't addressable (verified ~75% in sample; it slightly
*over*-counts by swallowing a few Sub-absorbed / −24..−33-band real hits), so **56% is a floor.**

**(c) 5-agent adversarial verification** of the verdict surfaced two corrections that downgrade the
case from a naive GO:
- **The "loss-specific" claim reversed.** Conditional on a cliff, addressable share is *higher* in
  wins (39.9%) than losses (29.6%); losses just have ~2.9× more cliffs. The feature targets a genuine
  *critic info gap*, but it is **not** a loss-specific signature.
- **The losing decision is policy-side.** In **67% of addressable cliffs with a healthy bench switch,
  the policy had already collapsed switch mass to <5% (median 0.006).** A critic-side obs feature
  fixes the *valuation*, not the *action*, without a retrain that also fixes under-switching.

## 3. Go / No-Go

**CONDITIONAL-GO, medium confidence.**

- **Proven (necessary condition):** the critic is tail-blind to unpriced incoming hits on blank
  coverage at ~50% of decisive loss cliffs — robust, measurement-clean, and the "critic ignores even
  when informed" failure is small (8%), which supports the "fixable information gap" reading.
- **Not proven (sufficient conditions):** (a) that a *better action* existed and the policy would take
  it — the binding gap is chronic under-switching; (b) that a *retrained* model would attend to the
  feature (current lane at ~0.002 saliency) — unknowable zero-retrain.
- **The root is critic-calibration**, and the cheap **critic-first** lever (`--clip-range-vf` un-clip,
  optionally a distributional/quantile value head, reward rescale) targets it with **zero obs cost and
  no ARCH retrain** — so it must be tried first.

**Decision / gating sequence:**
1. **[in flight, overnight]** training run with **no vf clipping**.
2. **Re-run `falsifier_cliff_attribution.py`** on a post-fix checkpoint.
3. **If the unpriced-incoming-KO cliffs collapse** → the cheap fix won, the obs feature is **moot**.
   **If a residual persists** → that is the clean GO, and we build the **scoped minimal Phase-1** below.
4. Phase-1 ships behind its own measured gate (§7); **all of Phase-2 is gated on a proven Phase-1
   residual.**

## 4. The reframe + the compute-vs-learn line

**We compute a calibrated *belief about being KO'd*, not the damage.** Precision under hidden info is
impossible and not the goal; calibration is. The human read is "probably OHKOs unless it's the
defensive set."

> Compute what the model can't learn; expose everything else; let the model relate.

- **We compute** (the hard threshold + external usage knowledge): per-defender **damage / P(KO)**
  belief, integrated over the hidden-set prior conditioned on reveals.
- **The model learns** (a soft gate — its strength): how to weight physical vs special given the
  reveals (already in the obs) and the board.

No hand-tuned confidence scalar — the calc uses **best revealed data if present, else prior**
(knownness is implicit), and calibration is measured **sliced by knownness** (revealed-move-count
buckets). This mirrors the existing obs contract (`item [id, known, consumed]`, `spread_known`,
ability `known`).

## 5. Proposed implementation (the settled design)

### 5.1 Multimodal sets — classify per MOVE, not per spread

The phys/spec split is the only axis that changes calc *structure* (which of our stats defends, which
moves apply) → **two explicit, fixed channels.** Setup/bulk/CB are within-mode *magnitude* + observed
boosts. Classify each candidate **move** by category and route it to its channel **regardless of EV
investment** (a 0-SpA Suicune still threatens special damage off base 90 SpA + STAB; a physical
Metagross carrying Psychic populates both channels). The spread only sets the **magnitude**, using the
**offensive-tail** spread (~80th-pct invested) with a **base-stat-anchored fallback.** Regression
cases that must pass: **Mantine, Suicune, Snorlax, Metagross-Psychic.**

### 5.2 Threat math — "best single answer," reusing the existing helper

Per our defender X, opponent attacker A, mode m ∈ {phys, spec}:
```
threat_m(X) = max over candidates_m  [ P(M ∈ A's set) · P(KO | M, mag_m, modifiers, roll) ]
```
- `candidates_m` = mode-m moves threatening X = **revealed (P=1) ∪ top-(4−revealed_count) prior**
  moves by usage. `max` = the attacker's single best answer to X (no joint distribution needed; no
  over-count — the independent *product* inflates P(KO)→1 for broad-coverage mons and is rejected).
- The **spec channel is gated on the mode prior** so a near-pure-physical attacker can't accumulate
  spec P(KO) from incidental coverage. Revealed moves are a **lower bound** — seeing only EQ still
  leaves prior mass on Rock Slide, so a Flying mon stays flagged.

**Build on `opponents.py:164 _estimate_damage_fraction(move, attacker, defender)`** (boost-aware
stats, STAB, ability-aware type-eff, multi-hit, avg roll already done — ~90% of the feature). It must
be extended to fix three correctness gaps:

**(a) Fixed-damage / special-mechanic branch** (keyed off move id, before the multiplicative formula):
Seismic Toss / Night Shade = 100; Super Fang = ⌊HP/2⌋; Dragon Rage = 40; Sonic Boom = 20;
Counter / Mirror Coat = 2× our damage that turn, **axis-inverted** (Counter keys physical, Mirror Coat
special); Sheer Cold / Fissure / Horn Drill = ~0.30 guaranteed-KO. Never let these fall through to
STATUS.

**(b) Complete the modifier set** (all knowable from *our* board → correctness, not uncertainty):
Reflect (×0.5 phys), Light Screen (×0.5 spec), Rain/Sun on Water/Fire BP, Burn ×0.5 phys, and **our
Substitute up → hard short-circuit `P(KO)=0`** (or a separate `breaks_sub` signal). **CORRECTION
(this draft was wrong):** an earlier draft listed *Sandstorm SpD ×1.5 for Rock defenders* — that is a
**gen4+ mechanic**; gen3 sandstorm gives Rock-types **no** SpD boost (the Showdown gen3 mod nulls
`onModifySpD`). It was briefly built then **removed**; do not re-add it. (Explosion/Self-Destruct's
target-Def-halve, by contrast, *is* gen3-correct — `gen ≤ 4` in `sim/battle-actions.ts`.)

**(c) Turn-shifted / residual:** scope **Future Sight / Doom Desire** out of the per-turn channel
(they have `basePower>0` and land 2 turns later); fold **known end-of-turn chip** (Spikes on the
switch-in, Sandstorm, the Toxic counter) into `HP_remaining` for the switch-target calc — this decides
*which bench mon actually survives* (the core Claydol case).

### 5.3 Log-space inductive bias

`KO ⟺ log·power + log·A − log·Def + log·eff + log·STAB + log·roll ≥ log·HP − const` → a **linear
hyperplane in log-feature space.** Feed the precomputed **`log(A/Def)` ratio**, `log(HP_current)`,
`log(base_power)`, scalar effectiveness (already `/4`), a STAB bit. **Normalize logs to ~0 mean** so
they don't swamp the [0,1] neighbours in the shared LayerNorm. **Immunity:** `eff=0 → P(KO)=0,
immune_bit=1`, and the **immune bit gates the channel** (never feed `log(0)`); ability-prior immunity
(Levitate on an unrevealed Claydol) → multiply that candidate's P(KO) by `(1 − P(immune-ability))`,
not a binary zero. (Gen3's `+2`/floor terms break exact log-linearity for *small* hits; negligible for
KO-relevant damage.)

### 5.3b Speed — as a PROBABILITY, not a bit (v1, per owner 2026-06-06)

The opponent's Speed is hidden (nature + EVs vary per set), so a binary "we outspeed" is wrong. Per
our mon, emit **`P(outspeed) = P(their_spe < our_spe) + ½·P(tie)`** ∈ [0,1], marginalised over the
opponent active's Speed **distribution** from the spread priors (each usage spread → a concrete Spe
stat at L100/IV31, weighted by usage), with **observed** boosts + paralysis (gen3 par = ×0.25) folded
in on both sides; our Speed is exact. The probability *is* the uncertainty (≈0.5 = set-dependent,
≈0.95 = near-certain). Feed it as a per-our-mon scalar alongside the P(KO) pair and let the model
relate "fast + OHKO = dead before I act" (don't pre-multiply). Reuses the Phase-0 spread priors.
*v1 approximation gap:* Swift Swim / Chlorophyll ability-speed-doubling in weather (marginalise over
the ability prior) deferred to v2; gen3 has no Choice Scarf, so item-speed is a non-issue.

### 5.4 Feature layout & placement

Scope: opponent **active → our active + 5 bench** = 6 matchups (the switch decision). Per our mon,
lead with **expected-damage-fraction** (unsaturated — full gradient, discriminates 2HKO vs OHKO, the
right primitive for the log-hyperplane) + **one calibrated mode-max P(KO)**, as a **phys/spec pair**,
plus the §5.3 log-stats. Do **not** ship the redundant saturated expected+worst P(KO) pair.

Placement: route the scalars via the **`non_matchup_rest` reactive-block lane** (the precedent is the
trapping-signal bits, which already ride it to both heads). The value proposition is **information
content + never-blank**, not re-routing. This is **retrain-class** → bump `ARCH_SIGNATURE`.

### 5.5 Data plan

Priors already exist in `data/pokemon/gen3_smogon_stats.json` (per species `Moves`/`Spreads`/`Items`
+ `Raw count`). Extend the existing `ability/HP-prior` pipeline (`tools/smogon_stats_downloader →
data/ → gen3_data`):
- Normalisation: `P(M) = Moves[M]/RawCount` (weighted **raw** counts, **not** sum-to-1); mode mass
  from `Spreads`; `P(item) = Items[i]/Σ Items`.
- Coverage floor: `RawCount < ~5000` → base-stat-lean fallback with a wide band.
- **Distribution shift (must do):** the agent never plays the ladder — training/eval/self-play draw
  from ~39 curated `data/teams/`. **Derive/validate priors against that pool**, and run the §7
  calibration fuzz on bridge battles from it.
- Teams-corpus joint-set mining (777 teams, ~21 sets/species avg) is a **v2-only** refinement where a
  species has ≥N real sets.

## 6. Validation — two gates

**Gate 1 — calibration fuzz (model-free, BEFORE any model change), on `data/teams/`-pool battles:**
ground truth = `P(KO is AVAILABLE)` via a bridge counterfactual oracle (not `P(KO occurred)`, which is
confounded by opponent policy); ≥200 decisions/bucket, Wilson bands, SE clustered per battle;
reliability curve **sliced by knownness**, with the GO bar on the **low-knownness** slice;
over-confidence in the 90–100% bucket = fail. Plus the **obs-build benchmark** (<10% calls/encode
regression, hard gate): **pin K before sign-off**, reuse the value-memoized
`effective_multiplier_by_types`, hoist opp-active attacker terms once per decision.

**Gate 2 — post-retrain, hard GO/NO-GO:** the incoming block's saliency must rise **≥10–50×** from
~0.002 **and** the CRITIC_BLINDSPOT turning-point count must drop on a fresh forensics pass. A
calibrated obs feature is **necessary, not sufficient** — the model ignoring a reachable signal is what
actually failed before.

## 7. Phasing

- **Phase 0 — falsifier + priors. [DONE — CONDITIONAL-GO.]** The cliff-attribution falsifier ran
  (§2b); priors-derivation is the only remaining Phase-0 build, and it's not retrain-class.
- **Phase 0.5 — critic-first. [IN FLIGHT: no-vf-clip run, overnight.]** The gating step. After it
  reports, re-run the falsifier on the post-fix checkpoint; cliffs collapse → feature moot; residual →
  GO for Phase 1. (A distributional/quantile value head is the next critic-first lever if un-clipping
  alone is insufficient — head-only, no obs ARCH change.)
- **Phase 1 — the minimal measured obs feature.** §5, built on `_estimate_damage_fraction`, with the
  fixed-damage branch + modifiers + residual-into-HP + per-move classifier + immunity gating + log
  stats, routed via `non_matchup_rest`. Retrain + measure against Gate 2.
- **Phase 2 — gated on a proven Phase-1 residual.** Bespoke 16-roll P(KO), separate expected/worst +
  CB worst-case channel, the speed bit, per-mon-token placement, teams-corpus joint mining.

## 8. Open decisions (deferred until/unless Phase 1 is greenlit)

- Speed/outspeed bit in v1 or v2 (4 lenses wanted it; the revenge-KO cluster is speed-gated — "a KO
  you outspeed ≠ one you don't"). Currently Phase 2.
- CB worst-case channel in v1 or v2 (CB is never revealed → inferred from `prior_CB` + item-known +
  the "used ≥2 distinct moves in one uninterrupted stint = not Choice-locked" tell; note this is a
  *Showdown-sim* fact, not cartridge gen3). Currently Phase 2.
- Whether the critic-first fix alone suffices (decided by the post-overnight re-measure).

## 9. Risks & what this does NOT claim

- **Policy-side binding gap:** the agent under-switches (switch mass collapsed to <0.5% in 2/3 of
  addressable cliffs); a critic-side feature won't change actions without a retrain that also moves the
  policy. This is the central risk and the reason for the conditional verdict.
- **Crutch / over-caution:** a pessimistic belief → too-passive play. Mitigated by the expected channel
  + the Gate-1 calibration. (Current problem is *under*-switching, so mild caution is probably
  net-positive — but calibrate, don't guess.)
- **Baking external knowledge into the obs** (usage priors): justified exception to "provide raw facts,
  let it learn" — the prior isn't derivable in-battle and the KO threshold is the unlearnable
  nonlinearity. Guard: strictly immediate-board KO belief, nothing about opponent *intent*.
- **MCTS / search** is the "true" counterfactual answer (the thesis's lever); complementary, larger
  build, out of scope here.

## 10. Appendix — verified code facts (the load-bearing ones)

- **Live obs dim = 3357** (`Gen3ObservationEncoder.dimension`). `src/agents/observation/CLAUDE.md`
  says 3321 — **stale; fix in the same pass when the feature lands.**
- `their_matchups` lives at obs offset **1612** (`OFFSET_REACTIVE + reactive_layout.their_matchups`),
  144 dims, `[opp_mon × move × our_mon]` effectiveness `/4`. Already consumed via the per-mon token →
  transformer → both heads.
- Fixed-damage moves (`Seismic Toss`, `Counter`, `Mirror Coat`, `Super Fang`, `Endeavor`, …) →
  `basePower=0` → `gen3_data/moves.py:_derive_category` → STATUS → dropped. **Must special-case.**
- Reuse `src/agents/opponents.py:164 _estimate_damage_fraction(move, attacker, defender)` (+
  `_opp_known_max_damage_fraction:201`); it already does boost/STAB/ability-eff/multi-hit/avg-roll.
- Smogon priors source has `Moves` / `Spreads` (`Nature:HP/Atk/Def/SpA/SpD/Spe`) / `Items` / `Raw
  count` per species — sufficient for move/mode/item/CB priors.
- The prober `threats` decode (`engine.ThreatView`, `query analyze … threats`) + `scan` exist for the
  post-fix re-measure (shipped in `70dbbcd`).
