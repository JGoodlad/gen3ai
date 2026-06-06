# Design — Stall / Recovery-Reset Obs + Reward (ai_v5)

> Status: **design — revised after a 4-lens adversarial review (2026-06-05).** No production code.
> Sibling of `design_incoming_damage_obs.md`; this targets the **E_OTHER** residual that the
> incoming-damage feature does *not* fix. Origin: the loss forensics of `run_20260601_193826` @122M
> (`LOSS_ANALYSIS_2026-06-05.md` + `_2026-06-02.md`) + my own extension of the cliff-attribution
> falsifier (`falsifier_recovery_attribution.py`, in this branch) + prober (`exact` tier) drill-down.
> The 4-lens review (gameplay / ML-arch / data-perf / red-team) is folded in; what changed is logged
> in §15.
>
> **Cross-reference note:** the sibling `design_incoming_damage_obs.md` is **now on `main`**
> (commit `336f5f9`); its section numbering differs from an earlier draft, so the "incoming-damage
> §N" cites here are to the **landed** version (fixed-damage → its §5.2; placement → §5.4; data/
> priors → §5.5; speed bit → its §8 open-decisions, currently Phase 2). The two designs should land
> their shared infra together (§10).
>
> **Live obs dim = 3357** (re-confirmed live this pass: `Gen3ObservationEncoder.dimension`=3357 +
> the obs-build benchmark header "obs dim 3357"; `observation/CLAUDE.md`'s 3321 is stale — the
> incoming-damage design owns that fix; I touch no obs code here).

---

## 0. The headline (read this first — it corrects the brief)

I was asked to fix "the second-biggest category of value-cliff losses — the stall/recovery-reset
failures," on the working hypothesis that recovery-resets *dominate* the E_OTHER bucket. **I measured
it instead of trusting the eyeball, and the magnitude claim is wrong:**

| At the **decisive turning point** (first cliff while clearly winning: V>5, ΔV<−15), of the 263 loss TPs | share |
|---|---|
| ADDRESSABLE by incoming-damage (A+C) | **56.3%** |
| D_INFO_PRESENT (incoming, critic ignored) | 8.4% |
| B_SELF_KO | 1.5% |
| **E_OTHER (the incoming feature does NOT fix)** | **33.8%** |
| &nbsp;&nbsp;↳ **RECOVERY_RESET** | **3.0% of TPs (9.0% of E_OTHER)** |
| &nbsp;&nbsp;↳ TEMPO/switch-matchup swing | **30.8% of TPs (91.0% of E_OTHER)** |

So **recovery-reset is NOT the biggest E_OTHER sub-category — TEMPO_SWING is, by ~10×.** And at the
raw-cliff level (all 2093 loss cliffs, ΔV<−8), 53% of E_OTHER is **ALREADY_LOSING** (V<0, downstream
of an earlier blindspot — not a fresh category), 37.6% TEMPO, only 9.3% recovery.

**Following the data, the genuinely-biggest *fixable* residual is not a single new feature.** It
splits into two very different things:

1. **TEMPO_SWING (the big one, ~31% of TPs)** *co-occurs heavily with the incoming-damage feature's
   blank spot*, so it is **a candidate hand-off to that design — to measure, not a confirmed fix.**
   On these TPs we took ~no damage (mean our-HP Δ −1.9%, only 11% lost >15%), ~half are switch-driven,
   and **73% are blank-coverage** (`revealed_frac<0.33`) — i.e. the "just-switched-in threat the
   never-blank incoming belief + the speed bit *would* populate." That is correlation, not proven
   causation: it means TEMPO_SWING is the right thing for the incoming-damage design's never-blank +
   speed-bit (its §8 open-decision, currently Phase 2) to **measure its delta against**, not a second obs block here. I flag it and
   hand it back — it is **not** a blocking residual for recovery-reset.

2. **RECOVERY_RESET (small, ~3% of TPs, but uniquely clean)** is the **only E_OTHER sub-category that
   is (a) orthogonal to incoming-damage, (b) reducible to one missing belief, (c) severe per
   instance (ΔV −15 to −54), and (d) the textbook human failure** ("you can't out-chip a Rest wall").
   It is the right standalone target — **but it is small, so the honest recommendation is to fold it
   into the SAME retrain/ARCH bump as the incoming-damage feature** (shared infra; §10), not to spend
   a retrain on it alone.

This doc designs (2) and is explicit about its limited scope. **It does not oversell.**

---

## 1. Problem — what recovery-reset actually is (measured)

The decisive-turning-point metric (one dramatic single-turn cliff) structurally **undercounts** slow
stall losses, because a recovery wall denies the win condition over *many* chip→heal cycles rather
than one cliff. So I also measured it at the **battle level**:

| Battle-level (extends the falsifier) | LOSS | WIN |
|---|---|---|
| "recovery-stall" battles (≥2 big opp self-heals while we attack) | **66/651 = 10.1%** | 26/350 = 7.4% |
| Wall species healed-against — **Suicune** | **127** | **2** |
| Wall species healed-against — Blissey | 105 | 65 |
| Wall species healed-against — Milotic / Celebi | 41 / 41 | 11 / 14 |

*(Metric definition, to be unambiguous: a "recovery-stall battle" has ≥2 turns where the opp active
self-healed ≥20% while we attacked — it counts **both** losses (we never broke it) and wins (we
eventually did), reproduced by `battle_stall_report()` in the falsifier. The species rows are big-heal
**events**, normalised by the 651 loss / 350 win battle base; the loss-bias of the trace sample
(~10 loss / 5 win saved per opponent) inflates loss counts ≈1.86×, which is dwarfed by the
Suicune 63× ratio below — the signal survives the confound.)*

**The single cleanest signal in the whole investigation:** *Suicune is healed against 127× in losses
and 2× in wins.* Blissey heals appear in **both** wins and losses. The reason is mechanical and
load-bearing for the design:

> **Rest cures status; Softboiled/Recover do not.** Our team's win condition vs a bulky Water is
> Toxic-stall. Against **Blissey-Softboiled** the Toxic clock keeps ticking → we sometimes win.
> Against **Suicune-Rest** the Rest *resets the Toxic clock AND fully heals* → the stall is
> structurally unbreakable for our team, and we lose. **This is the "Rest-resets-a-won-Toxic-stall"
> pathology** the project memory flagged from `LOSS_ANALYSIS_2026-06-02`.

### What the prober shows (exact tier, 8 decisive recovery TPs + worst-25 all-cliffs)

Every recovery TP is the same shape — the critic prices a low-HP recovering wall as near-won, then it
heals and V collapses:

| battle (step/opp/file inv) | our active (moves) | opp | V → next (ΔV) | opp acted |
|---|---|---|---|---|
| 116000024/heuristic2/loss_012 #24 | Milotic 66% BRN (surf,toxic,refresh,recover) | **Suicune 20% TOX(5)** | **+27.6 → −26.4 (−54.0)** | rest +80% |
| 118000001/staller/loss_008 #14 | Milotic 77% (surf,toxic,recover,refresh) | Suicune 26% TOX(4) | +10.4 → −25.8 (−36.2) | rest +61% |
| 118000001/staller_v2/loss_006 #13 | Milotic 76% | Suicune 27% TOX(4) | +9.6 → −25.4 (−34.9) | rest +62% |
| 122000011/heuristic2/loss_008 #13 | Milotic 55% | Suicune 38% TOX(4) | +10.3 → −15.7 (−26.0) | rest +62% |
| 114000008/heuristic2/loss_009 #12 | Blissey 86% (stoss,icebeam,twave,softboiled) | Suicune 26% PAR | +17.7 → −4.6 (−22.4) | rest +74% |
| 120000001/sentinel_0/loss_002 #43 | Celebi 28% PAR | Blissey 7% | +22.0 → +6.5 (−15.6) | softboiled +50% |

The worst case (#24) is the proof of the gap: **Suicune at 20% HP, Toxic counter 5** — it would die
to poison next turn, and we *hold Toxic* — so the critic prices the position **+27.6 (won)**. But
Rest cures the poison and heals to full; the "locked-in" kill the critic priced **never existed.**
In `heuristic2/loss_008` our **entire 6-mon stall team** (TTar/Milotic/Blissey/Skarmory/Claydol/
Dugtrio, all Leftovers) lost to **one** Suicune that ended the game at 21%.

**Diagnosis (what the obs lacks).** The obs *does* carry opp HP, the 7-dim status one-hot, and the
Toxic counter, and the per-turn heal is in the 10-turn TurnDelta history (`opp_target_hp_delta`,
`opp_move_id`). What it **lacks is any forward-looking, action-aligned belief that summarises
"this opponent can recover, can cure its status, and I cannot out-damage that."** This is the I-3
"recovery awareness is *co-location*, not *absence*" gap from `LOSS_ANALYSIS_2026-06-02` — the raw
events are present; the *belief* is not. So the critic prices a low-HP Rest wall as won, and the
policy (switch weighted ~0–8% on every one of these turns) never reaches for a breaker/phazer/
PP-stall/concede line.

**Saliency note (corrected):** my first prober pass mis-read the saliency field (`SaliencyBlock.
mean_abs`, not `value`) and printed zeros — disregard those. There is in any case **no recovery obs
block to attend to today**, so a pre-retrain saliency on it would be zero by definition. The
meaningful measurement is **post-retrain (§9 Gate 3, not Gate 2)**: does the model attend to the new
scalars (`mean_abs` materially > 0).

### Robustness / discrimination (the honest caveats)

- **Threshold sweep (HEAL∈{15,20,30}%, V_MIN∈{0,5,10}, ΔV_MAX∈{−10,−15,−25}):** recovery is
  **6–20% of E_OTHER**, rising to ~20% at the deepest cliffs (ΔV<−25) — i.e. *when it does fire it
  is among the most severe*, but its share of all TPs stays small (2–7%). Insensitive to the HEAL%
  threshold (the heals are +50–90%, far above any cutoff).
- **Win-vs-loss discrimination is WEAK at the TP level** (recovery decisive-TP rate per battle: LOSS
  1% vs WIN 0%) and **only modest at the battle level** (10.1% vs 7.4%). The discrimination lives
  almost entirely in **which wall** (Suicune-Rest), not in "a heal happened." A design that just
  flags "opp healed" would barely separate wins from losses; the **cures-status axis is what
  discriminates** (§5).

---

## 2. The reframe (load-bearing)

**We compute a calibrated belief about whether this opponent's HP can be *taken away faster than it
is restored* — a "breakability" belief — not a heal predictor.** Mirror of the incoming-damage
reframe (P(being KO'd), not the damage): here it is *P(I can break this wall on my current plan)*,
and its dual *"is my status/Toxic win-condition real against a Rest user."* Precision under hidden
sets is impossible and not the goal; calibration sliced by knownness is. The human read is exactly
this: "Suicune at 20% Toxic'd is *not* won — it Rests, and Rest cures the poison."

---

## 3. Goals / non-goals

**Goals**

- A calibrated, **never-blank** belief, for the **opp active wall**, about (a) its recovery rate, (b)
  whether that recovery **cures status** (the Rest discriminator), and (c) whether **our best move
  out-damages it** ("net chip margin / breakability"). Sharpen with reveals; fall back to Smogon move
  priors when unrevealed.
- Compose with — not duplicate — the incoming-damage feature; share its infrastructure and its
  retrain.
- A **reward** complement (windowed/heal-aware futile) so the policy isn't *paid* to chip an
  unbreakable wall for 10+ turns.

**Non-goals**

- A PP-stall / phazer / Taunt *planner* — we expose the economy; the model learns the plan.
- Predicting *whether/when* the opponent chooses to Rest (we give capability, not a behaviour model).
- The TEMPO_SWING bucket (§0 item 1) — handed to the incoming-damage design (never-blank + speed).
- Fixing ALREADY_LOSING cliffs (downstream; out of scope by construction).
- **Per-opp-bench recovery.** v1 is **opp-ACTIVE only** (all 8 measured decisive recovery TPs are
  about the active wall). It will *not* pre-price a benched, not-yet-revealed Rest-wall the opp could
  switch in — that mon reads as a blank until it acts. This is the A3 Phase-2 gap (§5), gated on
  whether the active-only feature solves the measured cliffs.
- A standalone retrain justified by this feature alone (its scope is ~3% of decisive TPs — §10).

---

## 4. The compute-vs-learn line

> Compute what the model can't learn; expose everything else; let the model relate.

- **We compute** (hard nonlinearity + external usage knowledge):
  1. `opp_recovery_rate` — expected HP-fraction restored per turn, integrated over the move-usage
     prior conditioned on reveals (the same "priors first, confirm on reveal" pattern as
     `status_will_land` / ability priors). Includes Leftovers (+1/16) from the item prior + reveal.
  2. `opp_recovery_cures_status` — prior-weighted P(the recovery is **Rest** specifically) — the only
     status-curing heal in gen3, and the measured discriminator (§1).
  3. `net_chip_margin` — `max over our moves of expected_damage_fraction(move, our_active, opp_active)`
     **minus** `opp_recovery_rate`. Negative ⇒ unbreakable by chipping. The hard damage-vs-heal
     threshold the model cannot infer from raw stats. Reuses `opponents._estimate_damage_fraction`
     (§6) **with the fixed-damage branch** (Seismic Toss is our literal Blissey breaker and reads 0
     today — shared blocker with incoming-damage §5.2 / its appendix §10).
- **We expose** (already in the obs — let the model relate): opp HP, opp status + Toxic counter, our
  per-move PP, Leftovers item id/known.
- **The model learns** (its strength): how to weight "unbreakable by chip" against the lines we
  *don't* hand-code — switch to a phazer, Taunt the Rest, PP-stall it out, or concede the matchup —
  and how to fold `cures_status × Toxic-counter` into "this won Toxic-stall is fake."

**No hand-tuned confidence scalar** (same discipline as the incoming-damage design): use real data if
revealed, else the prior; carry a `known` bit; measure calibration *sliced by knownness*, not "at a
confidence level."

---

## 5. The design — a 4-scalar opp-active recovery belief (minimal v1)

Add a small block to the **reactive scalar region** (rides `non_matchup_rest` → transformer global
token **and** both projection heads; the trapping-signals precedent — §7). For the **opp active mon**:

| # | scalar | range | meaning / computation |
|---|---|---|---|
| 1 | `opp_recovery_rate` | [0,1] | the **largest HP fraction the opp active can restore in one move**, prior-weighted: `max over heal candidates of P(m∈set)·heal_amount(m)` (revealed ∪ top-K prior), `+` passive Leftovers `P(item=lefto)·(1/16)`. **`heal_amount` is move-specific and HP-dependent** (§6.1): **Rest → restores to FULL = `(1 − opp_hp_frac)`** (not 0.5!); Recover/Softboiled/MilkDrink/SlackOff → `min(0.5, 1−opp_hp_frac)`; weather-heal (Moonlight/MorningSun/Synthesis) → `0.66` sun · `0.5` clear · `0.25` other weather, ×`(1−hp)` cap; Wish → 0.5 of wisher max (delayed); Pain Split situational (~0.25). |
| 2 | `opp_recovery_cures_status` | [0,1] | prior-weighted **P(Rest ∈ set)** (revealed→1/0). **Rest is the ONLY gen3 HP-recovery move that also cures status** — the Toxic-clock killer and **the discriminating axis** (§1: Suicune-Rest in losses, Blissey-Softboiled in both). |
| 3 | `opp_recovery_known` | {0,1} | 1 iff a heal move is **revealed** (or the item is `known`), else 0 (prior-only). Same routing as `status_will_land_known` / ability `known`. |
| 4 | `net_chip_margin` | [−1,1] | `clip(best_our_expected_damage_fraction − opp_recovery_rate, −1, 1)`. **The breakability belief.** Negative ⇒ our best single hit loses the HP race. (`opp_recovery_rate` is the ONLY place Leftovers enters — see §6.3b double-count guard.) |

**Why these four and not "effective HP":** an "effective/recovery-adjusted HP" scalar (`hp /
(1−uptime)`) blows up to ∞ exactly where it matters (chip ≤ heal) and hides *why*. The 4 components
keep the gradient unsaturated and let the model relate `cures_status` to its own Toxic counter — which
the data says is the real discriminator. The model already holds opp HP and the Toxic counter; it
needs the *rate*, the *cures-status* axis, and the *margin*.

**Why `cures_status` is first-class, not folded into rate:** the measured win/loss split is *Blissey
heals in both, Suicune almost only in losses*. The mechanical difference is purely status-cure. A
rate-only feature would price Blissey-Softboiled and Suicune-Rest identically and **fail to learn the
one distinction that separates our wins from our losses.**

### Alternatives (traded off)

- **A2 — action-aligned `net_chip_margin` per our move (4 dims, request-order).** Tells the policy
  *which* move breaks it (aligns with logits 6+k, like the move-effect block). Stronger for the
  policy; +3 dims. **Deferred to Phase 2** — v1 ships the single best-move margin (cheaper, and the
  measured failure is the *critic* over-valuing, which the scalar already addresses).
- **A3 — per-opp-6-mon recovery (switch decisions).** Tells us *which benched wall* is also a
  Rest-trap, informing "don't switch to chip that one either." Bigger; the measured cliffs are all
  about the **active** wall. **Phase 2**, gated on a residual.
- **A4 — `opp_recovery_pp_norm`** (revealed recovery move's PP/max). The PP-stall-proximity signal —
  a real win condition vs Rest (run it out → Struggle). Weak when unrevealed; **Phase 2.**
- **A5 — reward-only (no obs).** The windowed-futile fix (§8) alone. Rejected as *sufficient*: it
  stops the policy *after* the fact but leaves the **critic mispricing** (the value cliff) in place,
  and the policy still needs the obs to know *when* to switch to a breaker. Kept as a **complement**,
  not a substitute.

---

## 6. The recovery math + the reuse

### 6.1 Recovery rate over the prior

```
opp_recovery_rate(X) = clip( max_{m∈heal_cands}  P(m ∈ set) · heal_amount(m, opp_hp_frac)
                              +  P(item=leftovers)·(1/16),  0, 1)
heal_cands = { revealed heal moves (P=1) } ∪ { top-K prior heal moves by usage }
```
**`max`, not `Σ`** — the wall uses its single best heal, not all of them (mirrors the incoming-damage
"best single answer" §5.2; avoids the same over-count). `heal_amount` is HP-**dependent** and
move-specific (the table in §5 #1): **Rest restores to FULL → `1 − opp_hp_frac`**, the others cap at
0.5. This is the gameplay-lens fix — modelling Rest as a flat 0.5 under-prices it badly (a Rest off
20% restores 0.8, not 0.5).

`P(m∈set)` is from the **new** move-usage prior (does **not** exist yet — Phase 0 builds it, §10):
`P(m∈set) = Moves[m] / RawCount` — a **per-move presence probability** (the share of that species'
sampled sets carrying move *m*). It is **not** sum-to-1: a set has ~4 moves, so `Σ_m P(m) ≈ 2–2.4`
(verified: Suicune `Σ/RawCount = 2.36`, `rest = 382791/1104684 = 0.35`). That is correct and
intended — it's "is this move on the set," not a categorical over moves. Same normalisation the
incoming-damage design §9 documents (shared foot-gun). `is_heal` (`MoveData.is_heal`, already derived
from Showdown `flags.heal`, verified True for rest/recover/softboiled/milkdrink/slackoff/moonlight/
morningsun/synthesis/wish) is the membership test; the **amount** is the §5 table (Showdown exposes no
uniform heal-fraction field).

### 6.2 `cures_status` and the Rest discriminator

```
opp_recovery_cures_status(X) = P(rest ∈ set)          # revealed → 1.0/0.0; else prior usage of "rest"
```
Rest is the *only* HP-recovery move in gen3 that also cures status. RestTalk (Rest + Sleep Talk) is
covered automatically — it still *is* Rest. (Heal Bell / Aromatherapy cure status without HP recovery;
out of scope for this block — they don't reset an HP race, only the status, and are rarer; note them
as a Phase-2 `cures_status_only` candidate if the residual shows team-status-cure losses.)

### 6.3 `net_chip_margin` — reuse `_estimate_damage_fraction`, fix the same gaps

`best_our_expected_damage_fraction = max over our (revealed) moves of
_estimate_damage_fraction(move, our_active, opp_active)` (boost-aware stats, STAB, ability-aware
type-eff, avg roll — already implemented). **Two correctness gaps must be fixed first** (both are
*shared* with the incoming-damage design — do them once):

- **(a) Fixed-damage moves read 0 — a hard Phase-1 dependency.** Verified: `_estimate_damage_fraction`
  (`opponents.py:173-174`) returns `0.0` for `base_power==0`. Seismic Toss/Night Shade (100),
  Super Fang (½ current HP), Dragon Rage (40), Sonic Boom (20) all have `base_power=0`.
  **Seismic Toss is our Blissey's actual breaker in the loss cases** (cases #6/#7, §1) — without the
  branch the feature reads "0 chip / unbreakable" on exactly the mon that *can* (slowly) chip, so
  `net_chip_margin` mis-fires. Concrete fix — a small id-keyed table before the multiplicative path:
  ```python
  FIXED_DAMAGE = {  # gen3, level-100; returned as a fraction of defender CURRENT hp
    "seismictoss": lambda atk, dfn: 100.0,  "nightshade": lambda a, d: 100.0,
    "superfang":   lambda atk, dfn: 0.5,    # already a fraction of current hp
    "dragonrage":  lambda a, d: 40.0,       "sonicboom":  lambda a, d: 20.0,
  }  # → dmg_fraction = fixed_hp / current_hp  (superfang short-circuits to 0.5)
  ```
  **This is the SAME branch the incoming-damage design needs (its §5.2 / appendix §10).** Land it ONCE, in a shared
  PR, with a unit test (Seismic Toss vs Blissey ⇒ ~0.14, not 0). **Phase 1 is BLOCKED on this branch
  being in the codebase** — do not merge the recovery obs without it (see §10 dependency).
- **(b) Modifiers that flip the race** are knowable from our board and belong in the calc: Reflect/
  Light Screen (our screens are in `global_env`), Sandstorm SpD×1.5 for Rock walls, Burn ×0.5
  physical. **v1 ships the type-eff + STAB + roll already in the helper** and treats screens/weather as
  a Phase-2 refinement.
- **(b′) Leftovers double-count guard (concrete).** `_estimate_damage_fraction` computes the fraction
  off `defender.current_hp_fraction` (`opponents.py:186`) and applies **no item/Leftovers logic** —
  verified. So Leftovers enters the belief **exactly once**, via `opp_recovery_rate` (§6.1). The rule:
  `net_chip_margin = best_chip_fraction − opp_recovery_rate`, and the chip calc must **never** add a
  Leftovers term. Unit test: *Blissey + Leftovers, our Seismic Toss → margin = 100/maxhp − (P(rest)·
  deficit + 1/16)*, with the `1/16` appearing **once**.

`net_chip_margin = clip(best_our_expected_damage_fraction − opp_recovery_rate, −1, 1)`. **Sign is the
signal:** ≤0 ⇒ the wall out-heals our best single hit (the unbreakable-by-chip case). This is a per-
*turn* margin, not a multi-turn integral — deliberately: the per-turn sign is what the policy needs,
the model integrates over the horizon itself (γ=0.9999).

### 6.4 Encoding / inductive bias

The four are already in natural ranges ([0,1] / [−1,1]); they sit in the shared LayerNorm with the
other reactive scalars. `net_chip_margin` is the one with a meaningful **sign threshold at 0**, so do
**not** rescale it to [0,1] — keep it signed so "0 = breakeven" reads as a hyperplane the downstream
linear can recover (LayerNorm's per-sample affine shifts it, but the learned `γ/β` + the projection
weights restore an arbitrary threshold; the *sign structure* is what matters, not the literal zero).

**On log-space (claim softened after ML review):** unlike the incoming-damage KO threshold (a product
of logs), these are already HP-fractions, so I *expect* no log transform is needed — the nonlinearity
lives in the *computation* (prior integration, the HP-dependent heal table, the damage calc), not the
representation. But this is an inductive-bias claim, not a proven one. **Phase-1 implementation check
(cheap):** histogram `opp_recovery_rate` and `net_chip_margin` over ~1000 opp-active states from the
`data/teams/sample` pool; if either concentrates in a narrow band (e.g. >90% of margins in
[−0.2, 0.2]), add a `log1p`/scale transform before sign-off. Document the plot.

---

## 7. Placement & routing (code-verified)

Add the 4 scalars to the **reactive scalar region** (`reactive.py`, after the trapping bits, before
the move-effect block / matchup offset). Verified in `features_extractor.py`:

- `non_matchup_rest` = `remaining_part[:, 2·active_ctx : reactive_start + matchup_offset]` (line 314)
  — i.e. *everything in the reactive block before the matchup offset*, plus global env.
- It feeds **both** the transformer global token (`global_token_input`, line 642) **and** both
  projection heads directly (`ProjectionAssembler`, lines 768 + 774). This is the
  **trapping-signals lane** (`trapped`/`maybe_trapped` at `vec[12]`/`vec[13]`) — exactly the
  flat-to-both-heads routing a critic-calibration belief wants. The value path additionally has its
  own `value_pooled` CLS readout, so the critic gets the signal twice.
- Mutual-exclusivity (from the incoming-damage §5.4 finding): a block is **either** on the per-mon
  token path (the matchup matrices) **or** in `non_matchup_rest`, never both. A small opp-active
  belief belongs in `non_matchup_rest`. (A3's per-opp-6-mon variant would instead go on the per-mon
  token path — a Phase-2 architectural choice.)

`REACTIVE_SCALAR_DIM` 14 → 18; the matchup offset (currently 50) and every downstream offset shift
**automatically** (computed from constants via `get_layout()`). **This is retrain-class: bump
`ARCH_SIGNATURE`** in `model_version.py` (current `gen3_move_effects_v1`). The prober's pinned offsets
(`engine_test.py::test_offsets_resolve_matches_layout`) and the obs-layout table in
`observation/CLAUDE.md` must be updated in the same pass (they resolve at runtime, but the pins are
hardcoded values).

**Which head is the primary user (ML-review clarification).** The 4 scalars route to the policy and
value projection heads **identically** (`ProjectionAssembler` lines 768/774) — there is no separate
routing per head. The intended primary consumer is the **critic** (the failure is *value*
over-pricing of a low-HP wall), and the value path gets the signal **twice**: once via the shared
`non_matchup_rest` concat and once via the value-dedicated CLS readout (`value_pooled`, line 732) over
the transformer (whose global token also carries `non_matchup_rest`). The **policy** reads the same
scalars incidentally — useful for picking a breaker/switch, but its decision is dominated by the
team/active pools. We do **not** claim the heads are decoupled *on this signal*; the §9 Gate-3 check
explicitly compares policy-head vs value-head `mean_abs` to see whether the critic is the one using
it.

**Mandatory architecture validation (Phase-1, before retrain):** a unit test that builds an obs with
the 4 new scalars set to known non-zero values, runs `Gen3FeaturesExtractor.forward`, and asserts they
materially affect **both** `pi_features` and `vf_features` (perturb-and-diff) — proving the routing
survives the offset shift and the scalars aren't silently dropped. This is cheap and catches a
mis-wired layout before a multi-hour retrain.

---

## 8. Reward complement — windowed / heal-aware futile (R-1, still live)

**Code-verified, current:** `_compute_futile_attack_penalty` (`reward_manager.py:715`) (a) **returns
0 on the exact Rest turn** (line 729–730: "opponent used Rest; large self-heal is expected") and (b)
only ever inspects a **single turn** (line 742: `opp_hp_delta.sum() >= 0`). So a multi-turn
chip→Rest cycle is **never accounted for net progress** — the policy chips, gets `hp_opp = +2·chip`
on chip turns, the Rest turn's futile penalty is *skipped*, and over the cycle the net is ~0 with a
critic that says "winning." There is **no gradient telling the policy to stop.** (Re-confirmed: in
the decisive cliffs the per-turn reward is already mildly negative because we also Recover — so for
*those* TPs the obs/value lever dominates; the reward lever attacks the **long stall loops**, the
16-Earthquakes-into-Blissey class, that bleed us out across a battle.)

**Fix (retrain-class, no ARCH bump) — concrete:** keep a short rolling window of opp-active HP deltas
and penalise zero-net-progress chipping across the *cycle*, not the turn:
```python
# in the reward manager, per decision (our move was a damaging move):
WINDOW = 3                          # ~one chip→chip→Rest cycle
net = sum(opp_active_hp_delta[-WINDOW:])      # negative = we made progress
if net >= 0 and our_move_was_damaging and not we_switched:
    # gated on the NEW obs belief so we don't tax legitimate trading:
    if opp_recovery_rate > 0.3 or opp_recovery_cures_status > 0.5:
        penalty = FUTILE_ATTACK_PENALTY * min(1.0, WINDOW_turns_chipping / WINDOW)
window resets on our switch / on the opp fainting   # a new matchup starts a fresh cycle
```
- **Window slides** (re-evaluated every turn), **resets on our switch or the opp fainting** (a new
  matchup is a fresh cycle). N=3 by default (one chip→chip→Rest cycle); tune at implementation.
- **Gated on the new obs belief** (`opp_recovery_rate>0.3` or `cures_status>0.5`) so it can't tax
  normal trading where the opp merely Leftovers-ticks — this is why the obs feature and the reward
  fix ship together.
- **Keep it conservative** (the 2026-06-02 doc warns the wasteful-family penalties are an order of
  magnitude below `HP_VALUE=2.0` and otherwise get absorbed): the goal is a *behavioural* nudge to
  switch to a breaker/PP-stall/concede, not a hand-coded "stop chipping."
- Do **not** invert it into a positive "broke the wall" bonus in v1 (reward-hacking risk); a
  forward-progress/closing term is the 2026-06-02 R-4 idea and belongs in the reward-rebalance design.

---

## 9. Validation — gated, model-free first

**Gate 0 — zero-retrain attribution (DONE; `falsifier_recovery_attribution.py`).** Already run:
recovery-reset is **3% of decisive TPs / 9% of E_OTHER / 10% of loss battles**, Suicune-Rest is the
discriminator (127 vs 2), worst ΔV −54. This *bounds the prize* before any spend — and is why §10
recommends folding into the incoming-damage retrain rather than a solo one. **This gate is the
project's GO/NO-GO on scope.**

**Gate 1 — model-free feature-separation + calibration (BEFORE any model change), bridge battles from
the `data/teams/` pool.**
- **Separation (commit a standalone validator — extend `falsifier_recovery_attribution.py`):** on
  ~50–100 recovery-reset traces (V-high-then-crash) vs ~50 *matched win states* (a low-HP wall we
  broke), compute the would-be `net_chip_margin` / `cures_status` and show they **separate the two**
  (losses ⇒ margin≤0 & cures_status high; wins ⇒ margin>0 or cures_status low). **Fail loudly** if
  margin-sign agreement < 80% or prior-slice calibration error > 0.15. If the feature doesn't separate
  the cases it's meant to explain, **stop** (this is the gate that protects against the weak per-TP
  discrimination of §1 — the separation must come from the cures-status×margin axes, not from "a heal
  happened"). Publish the result before any code review.
- **Calibration sliced by knownness:** ground truth = a **bridge counterfactual oracle** — does the
  chip plan actually fail to net-reduce the wall's HP over the cycle (stay-in directly; switch-target
  via the oracle). ≥200 decisions/bucket, Wilson bands, **cluster SE per battle**; reliability curve
  bucketed by **revealed-recovery-move-count**; the GO bar is on the **prior-only (unrevealed)
  slice** (the revealed slice is near-trivially right). **Over-confidence in the high-`recovery_rate`
  bucket = fail** (don't price a Blissey-Softboiled as unbreakable when Toxic still wins — that's the
  `cures_status` axis earning its place).
- **Distribution-shift caveat (shared with incoming-damage §5.5):** the agent never plays the ladder —
  training/eval/self-play draw from the curated `data/teams/sample/` pool (**33** files on disk; the
  incoming-damage doc cites "~39"). **Derive and validate the move-usage prior against that pool**,
  not raw ladder usage. (The broader 777-team `data/teams/others/` corpus is the Phase-2 joint-set-
  mining source, not the v1 prior base.)

**Gate 2 — obs-build benchmark (hard, <10% calls/encode regression).** Run
`obs_build_benchmark.py --turn 25 --reps 400` before/after. **Baseline re-confirmed live this pass**
(after wiring the bridge deps): obs dim **3357**, **≈6.46k calls/encode** (775k calls / 120 reps;
matches the documented ≈6.36k), same tottime ranking (`effective_multiplier_by_types`,
`reactive.encode`, `moves.encode`, `live_view.from_pokemon`). So the gate is runnable and the baseline
holds. The new cost is one `_estimate_damage_fraction` over our ~4 moves vs the opp active + a memoized
prior lookup, per decision — small, but `_estimate_damage_fraction` calls `_stat_estimation` (poke-env
property reads) and `effective_multiplier` (the **object wrapper** — a per-cell pitfall in
`observation/CLAUDE.md`): **hoist the opp-active defender terms once per decision and route through the
value-memoized `effective_multiplier_by_types`, never the object wrapper.** **Pin K (prior candidate
count) before sign-off.** (Requires the pokemon-showdown bridge build — `git submodule update --init`
+ the `dist`/`node_modules` symlinks per the root `CLAUDE.md`; run at implementation time.)

**Gate 3 — post-retrain, hard GO/NO-GO (all three REQUIRED, not just logged):**
1. **Win-rate** rises on the recovery-heavy opponents — `staller`/`staller_v2`, `heuristic2`, and the
   `sentinel_*` mirror — without regressing the rest.
2. **Saliency floor (a blocking gate, not a log line):** the new recovery scalars' `mean_abs` must be
   **≥ 0.5× the mean_abs of the existing reactive scalars**, and **no individual scalar < 0.01** —
   else the model ignores the block and we ship nothing (the incoming-damage lesson: calibrated ≠
   used). Also compare **policy-head vs value-head** `mean_abs` (per §7): the *critic* should be the
   primary user; if only the policy attends, the value mispricing is unaddressed → escalate.
3. **Forensics drop, anchored on the BATTLE-level metric** (the decisive-TP count is a noisy 3% and
   may not move in lockstep — §1): the **Suicune-stall-loss battle rate** (measured 10.1% of losses)
   falls toward the win baseline, and **"close games" (3-2 / 2-1) appear vs stallers** (the 2026-06-02
   success metric). The recovery-reset decisive-TP count is a secondary, confirmatory signal — do not
   gate solely on it.

---

## 10. Phasing — minimal, and folded into incoming-damage

**The governing recommendation:** this feature's measured prize is small (~3% of decisive TPs). A
solo retrain + ARCH bump is **not** cost-justified. **Ship it inside the incoming-damage retrain** —
they share *all* the expensive infrastructure:

| shared component | incoming-damage | recovery |
|---|---|---|
| move-usage prior (`gen3_move_priors.json` + `priors.move(species)`) | candidate moves | heal candidates |
| `_estimate_damage_fraction` + **fixed-damage branch** | their damage to us | our chip to them |
| `non_matchup_rest` flat-to-both-heads lane | the KO scalars | the recovery scalars |
| prior-then-confirmation + `known` bit | revealed coverage | revealed heal move |
| `data/teams/`-validated priors, calibration-by-knownness gate | shared | shared |
| ONE `ARCH_SIGNATURE` bump + ONE retrain | shared | shared |

**Phase 0 — shared infra + this doc's Gate 0/1 (Gate 0 done; rest cheap, NO retrain).** Build the
**new** `gen3_move_priors.json` (does not exist today — `compute_priors.py` only emits ability + HP
priors; add `priors.move(species)`), needed by **both** features. Land the **shared fixed-damage
branch** (§6.3a) in `_estimate_damage_fraction` as its own small PR + unit test. Run the §9 Gate-1
feature-separation/calibration validator and publish it.

**Phase 1 — minimal obs (the 4 §5 scalars, opp-active only) + the §8 windowed-futile reward.**
**Hard dependencies (do not merge the recovery obs without these in the codebase):** the §6.3a
fixed-damage branch, the §6.3b′ Leftovers no-double-count, `gen3_move_priors.json` + `priors.move`,
and the §7 architecture unit test passing. Then Gate 2 (benchmark) + Gate 3 (retrain GO/NO-GO).

**⚠ Confounding cost & attribution (ML-review — read before bundling).** Folding recovery + incoming
into **one** `ARCH_SIGNATURE` bump and **one** retrain is cheaper, but a single retrain that adds two
feature families **and** fixes shared bugs (fixed-damage, Leftovers) means a Gate-3 win-rate lift
**cannot be attributed** to recovery vs incoming vs the bug-fixes. Two honest ways to keep
attribution:
- **(A) Sequential retrains (cleanest, recommended if compute allows):** retrain #1 = incoming-damage
  (+ shared fixed-damage/prior infra), measure its delta; retrain #2 = add the recovery scalars,
  measure *its* delta. Each feature is independently validated. Costs one extra retrain.
- **(B) One bundled retrain + post-hoc isolation:** bump once, but at Gate 3 run **inference-time
  ablation** (zero/mask the 4 recovery scalars on the trained model, re-measure win-rate vs stallers
  → recovery's marginal share) **and** the per-feature saliency split (§9 Gate 3.2). Cheaper, weaker
  causal claim.

This is **Decision §12.1** — the owner picks the compute-vs-attribution trade. Given recovery's small
prize, **(B)** is defensible; **(A)** is correct if the incoming-damage delta itself is in doubt.

**Phase 2 — gated strictly on a measured Phase-1 residual:** action-aligned per-move margin (A2),
per-opp-6-mon recovery for switches (A3), the PP-stall economy (A4), screens/weather in the chip calc
(§6.3b), Heal-Bell/`cures_status_only`, and (shared) a **distributional/quantile value head** — which
targets the downside tail directly with no obs change and helps *both* the recovery and incoming-KO
cliffs (and pairs with un-clipping `--clip-range-vf`, already planned next run given
`clip_fraction_vf≈0.70`). The value head is the natural Phase-2 lever **iff** the obs features prove
necessary-but-insufficient.

---

## 11. Risks & honesty

- **Small prize.** ~3% of decisive TPs / ~10% of loss battles. If win-rate doesn't move at Gate 3,
  the obs block may be dropped while keeping the (cheap) reward fix. The fold-in (§10) caps the
  downside cost to a few obs dims on a retrain we're doing anyway.
- **Over-correction.** Pricing every low-HP wall as unbreakable would make the critic *too*
  pessimistic and could suppress legitimate chipping (Blissey-via-Toxic, where the chip *does* win).
  The `cures_status` axis is the guard, and Gate-1 explicitly fails on high-`recovery_rate`
  over-confidence.
- **Calc fidelity.** `net_chip_margin` inherits `_estimate_damage_fraction`'s approximations and the
  fixed-damage gap; a wrong margin sign is the failure mode. Shared with incoming-damage; pinned by
  Gate 1's oracle.
- **Self-play may erode it for free — but hasn't.** `LOSS_ANALYSIS_2026-06-05` shows the recovery
  cliffs reproduce identically vs the frozen pool (sentinel_0/1) — so the pool isn't teaching it
  away. The obs belief is the lever the pool lacks; but re-bench under live self-play (the prior is
  derived for the curated pool, which the sentinels also draw from).
- **Knownness leakage.** Once a Suicune has Rested once, Rest is revealed and the belief is exact;
  the *hard* case is the first encounter (prior-only) — which is precisely the Gate-1 GO slice.
- **Metric blind spot acknowledged.** The decisive-TP framing undercounts slow stalls (§1) — I
  mitigated with the battle-level metric, but the true win-rate impact is only knowable at Gate 3.

---

## 12. Decisions for the owner (genuine forks)

1. **Fold vs solo, and the attribution fork (recommended: FOLD via §10-B).** Ship the 4 recovery
   scalars inside the incoming-damage ARCH bump/retrain (§10), or treat recovery as its own retrain?
   Given the 3% scope, folding is strongly recommended; a solo retrain is hard to justify. **If you
   fold, pick the attribution path (§10):** (A) sequential retrains for clean per-feature deltas
   (one extra retrain), or (B) one bundled retrain + inference-time ablation + per-feature saliency
   (cheaper, weaker causal claim). Recommend **(B)** unless the incoming-damage delta itself is
   uncertain.
2. **Obs + reward, or reward-only first?** The windowed-futile reward fix (§8) is cheap and
   non-ARCH. One could ship it *first* (next retrain) and add the obs block only if the recovery
   cliffs persist. Trades a faster cheap experiment against a second retrain.
3. **Hand TEMPO_SWING back to incoming-damage?** I recommend yes — TEMPO_SWING (the actual E_OTHER
   majority) **co-occurs heavily** with the never-blank-incoming + speed-bit spot (§0; 73% blank), so
   it is the right thing for that design to **measure its delta against**, not a fresh feature here.
   Owner confirms the incoming-damage design takes it as a measurement target (its speed bit — an §8
   open-decision, currently Phase 2 — + never-blank).
4. **`cures_status` granularity.** v1 collapses it to P(Rest). Add a separate `cures_status_only`
   (Heal Bell/Aromatherapy) now, or defer to Phase 2 on residual? (Recommend defer.)
5. **Value head timing.** Pull the distributional/quantile value head forward to Phase 1 (it helps
   both cliff families) or keep it Phase-2-on-residual? It's the one lever that needs no obs change.

---

## 13. Alternatives considered (rejected)

- **Feed raw priors + reveals, let the model learn the whole inference** — rejected: hands back the
  unlearnable damage-vs-heal threshold + the external move-usage knowledge (the model has had 122M
  steps and hasn't learned "Suicune-Rest ≠ won").
- **One collapsed "breakability" black-box scalar** — rejected: hides the `cures_status` axis the
  data says is the discriminator; brittle.
- **"Effective HP" = hp/(1−uptime)** — rejected (§5): saturates to ∞ exactly where it matters and
  hides the components.
- **Reward-only (windowed-futile, no obs)** — kept as a complement, rejected as *sufficient*: leaves
  the critic mispricing (the value cliff) and gives the policy no forward signal of *when* to switch.
- **MCTS / search at inference** — the true counterfactual (would the wall actually out-heal me);
  complementary, larger build, out of scope (mirrors incoming-damage).

## 14. Provenance

Numbers in §0/§1 from `falsifier_recovery_attribution.py` (this branch; extends
`falsifier_cliff_attribution.py`) over `eval_traces/step_{114000008,116000024,118000001,120000001,
122000011}` (651 loss / 350 win battles, exact-tier snapshots present). The decisive-TP split, the
threshold sweep, and the **battle-level / per-species heal counts** (Suicune 127 vs 2 — reproduced by
`battle_stall_report()`) all print from that one committed script. Prober drill-down via
`main.prober.session.ProbeSession` (`exact` tier, faithful). Obs dim **3357** and the obs-build
baseline (**≈6.46k calls/encode**, unchanged tottime ranking) re-confirmed live this pass via
`obs_build_benchmark.py` (after wiring the bridge deps). Code facts verified in `reactive.py`,
`features_extractor.py` (routing lines 314/642/768/774), `reward_manager.py:715-744` (futile-attack
single-turn + Rest-skip), `opponents.py:164-187` (`_estimate_damage_fraction`: base_power==0→0,
no item logic), `gen3_data/{moves,priors}.py` (`is_heal` present; no `move()` prior),
`constants.py` (`REACTIVE_SCALAR_DIM=14`), `model_version.py` (`ARCH_SIGNATURE=gen3_move_effects_v1`),
and `data/pokemon/gen3_smogon_stats.json` (Suicune Rest 34.6%, Leftovers 57%; `data/teams/sample`=33).
Cross-checked against `LOSS_ANALYSIS_2026-06-05.md` (heuristic2 Rest td −26.3; sentinel_1 Toxic'd-
Suicune Rest) and `_2026-06-02.md` (R-1 windowed-futile; I-3 recovery co-location).

## 15. Review log (4-lens adversarial, 2026-06-05)

Verdicts: gameplay `sound-with-fixes`, red-team `sound-with-fixes`, ML-arch `needs-major-revision`,
data-perf `needs-major-revision`. **Folded in:**
- **Gameplay (load-bearing fix):** Rest restores to **FULL**, not 0.5 — §5 table + §6.1 heal-amount
  reworked (`heal_amount(rest)=1−hp_frac`, `max` not `Σ`). Mechanics confirmed (Rest uniquely cures
  status; 10 PP; Recover/Softboiled=50%/no-cure).
- **ML-arch:** the **confounding/attribution** cost of one bundled ARCH bump → §10 (A) sequential vs
  (B) bundled+ablation, surfaced as Decision §12.1; Gate-3 saliency floor made **blocking** + per-head
  comparison (§9); a mandatory **architecture unit test** (4 scalars reach both heads) added to §7;
  critic-vs-policy routing clarified (§7); log-space claim **softened** to a Phase-1 distribution check
  (§6.4).
- **Data-perf:** `gen3_move_priors.json` is **new infra** (built Phase 0, not extant) — sharpened;
  **fixed-damage branch** given a concrete table + made a **hard Phase-1 dependency** (§6.3a); Leftovers
  double-count guard made concrete with a code cite + unit test (§6.3b′); teams count corrected
  **39 → 33** (§9/§10); move-prior normalization clarified (per-move presence prob, Σ≈2.4, not sum-to-1);
  obs baseline re-run live (§9 Gate 2).
- **Red-team:** TEMPO_SWING reclass **softened** from "is incoming-damage residual" to "co-occurs /
  measure the delta" (§0/§12.3); battle-level metric definition + the loss-bias confound made explicit
  (§1); per-species counts made reproducible from the committed falsifier; Gate sequence for saliency
  clarified (§1).
- **Not changed (false alarm):** "missing incoming-damage sibling" — it was unmerged when the reviewers
  ran; it has **since landed on `main`** (`336f5f9`). Writing it was out of scope (another agent's
  deliverable).

**Post-rebase reconciliation (2026-06-05):** rebased onto `origin/main` after the incoming-damage
design (`336f5f9`), the anchored-ELO subsystem (`85f9a55`), and `--clip-range-vf 'none'` (`dfb580e`)
landed. No textual conflicts (disjoint files). Semantic fix: the landed sibling's section numbering
differs from the pre-merge draft, so every "incoming-damage §N" cite here was remapped to the landed
structure (fixed-damage §5.2, placement §5.4, data/priors §5.5, speed bit §8-open-decision). Note
`--clip-range-vf 'none'` (§10/§11's "un-clipping next run") is now implemented on `main`.
