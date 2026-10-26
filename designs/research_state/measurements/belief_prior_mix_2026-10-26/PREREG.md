# The belief prior mix against Kakuna (lever 1 (a) and D0f): PRE-REGISTRATION (2026-10-26)

Written and shipped **before** any counted game of a mixed arm. Why: the architecture-generalisation review
([`../../../model/generalisation_review_2026-10-26.md`](../../../model/generalisation_review_2026-10-26.md), P5 / P10)
found that the opponent-belief deltas MEMORISE the training pool. On unfamiliar opponents the true hidden species'
log-presence is −6.2 / −4.6 / −10.0 (F / H / P72) against the Smogon prior's −2.4 / −2.0 / −3.2, and 63–85 % of the
hidden slots hold the wrong species. Offline, a quarter-weight mixture with the prior keeps every familiar ranking and
repairs most of the unfamiliar calibration. Nobody has asked whether that wins GAMES; D0f (prior only against the full
belief, `endstate/design_diversity_teams_opponents.md` §3.3, H3's kill test) has never run on X26.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170; H = `heldout72_hpfix`; F = `forum72`; U = the three unfamiliar panels H, F and P72; λ = the mixing
weight; Δ = a λ arm's score minus P\*'s at λ = 0.

## 1. The intervention

`agents.model.belief_prior_mix` (shipped with this read's pin): at load time each of the three learned belief legs
(X5's species presences, the move posterior, the item posterior) becomes `(1 − λ)·b + λ·prior` before the hypothesis
tokens, the damage operator, the policy and the critic read it. Inference only; the network's weights are P\*'s.
λ = 0 is P\* unchanged (bit-identical) and λ = 1 is exactly the deltas zeroed (D0f's prior-only model), both pinned
by `belief_prior_mix_test.py`.

**The arms:** λ = 0.25 (P10's offline optimum, the PRIMARY) and λ = 1.0 (D0f). λ = 0 is not re-played (§3).

## 2. The cells

- Metamon's Kakuna ckpt 34, greedy vs greedy, our side on CPU, through `python -m main.reads` (this is the module's
  first live read) and its anchor lane, which plays `main.x35.kakuna` units unchanged: the in-process Rust front end
  on a claimed 95xx port, 20-game role-balanced units with `--team-seed = --seed-base =` the unit seed, counted by
  `main.anchors.results.unit_counts`, on the short lane of the heavy slots.
- **Team sets:** the unfamiliar panels on their SYMMETRIC sha-pinned configs (`kakuna.PANEL_CELLS`: H, F, P72) and
  home72 on the X26 read's config (`kakuna.CONFIG`, sha `e8a064db…`, manifest-checked before every unit).
- **Unit seeds 0–19: 400 games per panel per arm**, X51's own seeds, so every game of an arm has a twin in X51's
  banked P\* rows (the same teams on both sides and the same battle seed).
- **Two reads, run side by side** (one `main.reads` state and one anchor-lane supervisor each, so the two arms take
  their turn on the heavy-slot queue independently): `read_spec_l025.json` (λ = 0.25 + a guard at unit seed 0) and
  `read_spec_l100.json` (λ = 1.0 + a guard at unit seed 10). Each arm is its own anchor group (its own regime: the
  rows say `our_regime = greedy+belief_prior_mix=<λ>`).
- **The model:** P\* (`checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`), a pinned ref; `main.reads` copies it
  (sha-checked, with its sidecars) into each state; `models/` is only read.
- **The playing code** is a detached, bootstrapped checkout of the spec's `pin` (the commit that ships the
  intervention), for every unit of both reads.

## 3. The banked control and its determinism guard

- **λ = 0 is X51's P\* cells, never re-played:** `measurements/x51_plateau_vs_kakuna_2026-10-20/rows.jsonl`
  (sha256 `27a8d7ed…6e28`), cells `pstar_hpfix` / `pstar_forum72` / `pstar_p72` / `pstar_home72`, 400 games each at
  unit seeds 0–19 (`main.reads.rules.banked_units`).
- **The VOID guard:** each read also plays P\* UNCHANGED (no load option) at one unit seed on all four team sets (read
  A seed 0, read B seed 10; 160 games in all). Every guard unit must count and reproduce its banked X51 unit GAME FOR
  GAME (each game's half, index, our team, result and turns). The x26c read showed this harness reproduces X51 exactly
  on the unfamiliar panels (60 / 60 units, `e05c45eeb` against `63f97a57`); home72 runs on a different committed
  config from X51's own and is checked here for the first time.

## 4. Statistics

Per arm, against the banked λ = 0 (`main.reads.rules`, rule `prior_mix_ab`):

- **Δ_U (primary statistic):** the UNIT-PAIRED difference (`paired_unit_diff`): per panel the mean over unit seeds of
  (arm − banked) unit win rates, the three panels equally weighted, normal 95 % interval from the per-panel SEs
  (z = 1.959963985). Units pair by (panel, seed); only counted units enter.
- **Δ_home:** the same on home72 alone.
- Reported beside, no rule: each panel's score with its Wilson interval and paired Δ; the pooled U score
  (`strat_score`) and the unpaired Newcombe-of-strata Δ.
- Rule 8 throughout: a bound within `RULE8_EPS` of its boundary is never a crossing (`clears`).

## 5. Decision rules (applied by `prior_mix_ab`; the verdict is `main.reads record`'s)

- **VOID** iff a guard unit does not count or does not reproduce its banked unit game for game, or an arm counts fewer
  than 90 % of its planned units on any team set (18 of 20).
- Per arm: **GAIN** iff Δ_U's LB > 0; **LOSS** iff Δ_U's UB < 0; else **NOT DETECTED**. home72 is **HARMED** iff
  Δ_home's UB < 0 or Δ̂_home < −3.0 pp.
- **PRIOR-HELPS** iff GAIN and home72 not HARMED; **PRIOR-HELPS-AT-A-COST** iff GAIN and HARMED; **PRIOR-HURTS** iff
  LOSS; else **NOT DETECTED** (reported with its interval, never as "no effect").
- **D0f (the λ = 1 arm only):** DiD = Δ_U − Δ_home, se = √(se_U² + se_home²). **H3 is KILLED as a major cause** iff
  the DiD's UB < +3.0 pp AND Δ_H (heldout72_hpfix alone) is not detected above 0 — the diversity design's registered
  rule, applied as written. Otherwise H3 is NOT KILLED.
- What each verdict sends where: PRIOR-HELPS at λ = 0.25 makes the anchor a standing inference option and lever 1 (b)
  (delta dropout in training) the next build; NOT DETECTED leaves lever 1 (b)/(c) to be judged on its own offline
  guard; PRIOR-HURTS retires (a). D0f's read updates H3's row in the diversity design.

## 6. Power

From X51's banked cells (the unit win-rate SD is 0.109–0.158, about the binomial 0.112) and the paired difference's
SD √(2σ²(1 − ρ)): at ρ = 0.25 / 0.5 the 95 % half-width of Δ_U is 4.1 / 3.3 pp (MDE at 80 % power 5.8 / 4.8 pp),
of Δ_home 5.9 / 4.8 pp, of D0f's DiD 7.2 / 5.8 pp. (The x26c read's unit-paired half-width between two different
checkpoints was 3.7 pp, ρ ≈ 0.4; two λ of one network should pair at least as well.) So this read detects an
unfamiliar effect of about ≥ 5 pp, and H3's kill needs a DiD point estimate near or below −3 pp. 600 games per panel
(the brief's upper budget) would need λ = 0 re-played at seeds 20–29 and shrinks the half-width by only ≈ 18 %; on
this box's queue it is not bought.

## 7. Predictions (registered)

- **λ = 0.25: NOT DETECTED**, Δ_U ≈ +1.5 pp; Δ_home ≈ −1 pp (not HARMED). Reasoning: P10 says the mixture repairs
  calibration but cannot raise the off-pool species RANKING above the prior's 0.28, and the policy was trained on the
  unmixed belief, so the changed inputs are partly off its distribution.
- **λ = 1.0: NOT DETECTED on U** (Δ_U ≈ −1 pp), home72 HARMED (Δ_home ≈ −5 pp: pool identification is worth 0.80 of
  the oracle gap in-pool, W12), DiD ≈ +4 pp, **H3 NOT KILLED** (the pre-era D0f read +4.8 pp [+1.9, +7.8]).

## 8. Schedule and stop

Both supervisors detached (`setsid nohup python -m main.reads run --lane anchor --state <state> --hours 36`), rule
5b's shape (nice 19, cores 5-7,13-15, the short lane), registered stop 36 h each. 84 units per read (80 arm + 4 guard).
A stop before an arm reaches 18 / 20 units on a team set is VOID by §5; it is resumed (`resume`), never decided early.

## 9. Known hazards (recorded before play)

- One checkpoint, one seed of training; the policy never saw a mixed belief (inference-time only).
- The spread and Hidden Power type heads are not mixed (not in D0f's definition).
- `main.reads` has never run end to end on real play; every rough edge of its first use is a FINDING of the read.
- The heavy-slot queue is deep (≥ 6 jobs at registration), so the wall time is set by it, not by play (X51: 18.5 h
  for 8.0 h of play).
