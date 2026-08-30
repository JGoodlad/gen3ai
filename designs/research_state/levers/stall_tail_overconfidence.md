# Stall-tail over-confidence — the 0.999 tails the clock did not reach

**Status:** 🔬 **OPEN — sequenced behind a data fix, not a design** · **Ledger:** `627ab58` (probe O
registered) · `32c39df` (probe O verdict) · `b63a96f` (the 0.999 mechanism) · `40f3da6` (the harvest
design) · `195ce9f` (harvest landed; reducibility blocked)

One-line claim: *the win-prob head reads a lost-by-construction position as won on ~35% of cap
tails, and the failure is DATA-shaped (cliff-vs-slope × gradient economics × joint selection) rather
than conceptual — so the diet gets the first attempt and architecture only if the diet measurably
fails.*

## Known (cleared the honesty gates)

Record: `measurements/stall_tail_head_reading_2026-08-29.{json,md}` — model-free, recorded values
only, 2,350 clock-era games; every headline reported both run-clustered-bootstrap AND
per-run-differences per the Simpson discipline, and all four significant results agree.

- **P1 CONFIRMED, both halves (`32c39df`).** The **clock fix HELD**: positive-V-at-final-decision on
  cap losses fell **81.2% pre-clock → 22.2% post-clock**, the historical 13/14 hand count reproduces
  at 84.9% (n=93), the break sits **exactly** at the clock boundary, and it has held for seven
  generations. **And the residual is real: 34.8% of cap tails end φ_T ≥ 0.5 on games that lose by
  construction** — 4.3× the regular-loss rate; 4.4% at ≥0.98. Worst specimen: **φ 0.999 held five
  straight decisions into a −30 forfeit, with V = +14.16.** LONG_WIN at 0.986 proves it is not a
  length effect.
- **P2 SPLIT, scored as registered.** Fails by the letter (the composite criterion's "declining"
  half saturates in every class — a criterion defect, noted) and **passes by the substance**
  (φ_T ≤ 0.5: 0.652 vs 0.908, **−0.256 [−0.315, −0.185]**).
- **The clock era is the FIRST where caps are distinctively worse than their own era's regular
  losses** (1.25× vs pre-clock's 0.81×) — the clock fixed the common case and left the pathological
  one exposed.
- **A finding from an EMPTY pre-registered class:** non-capping stall losses have **zero** members —
  in this population the stall pattern IS the cap ending (81.5% of caps have zero-faint tails vs
  0.0% of long non-cap losses).
- **The labels are RIGHT — verified in code before the hypothesis was uttered (`b63a96f`).** Tie and
  cap forfeits are labeled 0.0 with the mask set (`won = 1.0 iff battle1.won is True`), killing the
  masked-draw / selection-label account outright.
- **The three compounding mechanisms of record (`b63a96f`), all DATA-shaped:** (1) **CLIFF vs
  SLOPE** — "φ = 0 at t ≥ 250 regardless of position" is a multiplicative VETO, and a sigmoid over
  mostly-additively-combined features builds a slope; the clock bought 81% → 22% (a huge slope) and
  a slope cannot zero a 0.999 position prior in five turns, hence failures concentrating where the
  position prior is strongest (CAP_TRADE 48% over-confident). (2) **GRADIENT ECONOMICS** — BCE
  optimizes the average and cap games' final turns are ~epsilon of buffer mass, so confident
  wrongness there costs ≈ nothing; MC labels are unbiased, **not influential**, and absolute-but-rare
  rules get learned as tendencies. (3) **JOINT SELECTION** — strong positions finish early, so
  (dominant position × t ≈ 249) barely exists in training; reaching the cap while dominant requires a
  pathologically un-losable opponent, mostly an eval-sentinel phenomenon.
- **The 14× over-exposure figure does NOT rescue this.** `gen13_stall_coverage` measured stall
  trajectories at 3.0% of training decisions vs 0.21% of matched sentinel eval losses — but that
  covers heal-war states, **which the head reads RIGHT (φ 0.146)**. It is not the failing joint.
- **CLEAN-WORLD IMPLICATION, adopted:** the escalation trigger is **NOT fired** and the no-bias
  launch STANDS. The exposure is **CONDITIONAL** — 65% of tails read correctly, so a flat bias would
  tax the correct majority to reach the 35%. This supports the registered head-fix-before-bias
  ordering.
- **Bonus closure:** `gen14_endofrun_runbook` §(c) is CLOSED via its own sanctioned route
  (n = 134 / 23 runs, all three deltas significant).
- **The head has no propagation problem (`40f3da6`)** — MC labels stamp the terminal outcome onto
  every step, so turn-100 of a capped game is labeled 0 directly. This is a **census** problem
  (discrimination mass at time slices), not a signal-travel problem.

## Not-known

- **THE question: is the φ ≥ 0.98 tail REDUCIBLE?** If more mass alone teaches the veto from existing
  features, the defect is diet. If it does not, the defect is REPRESENTATIONAL and factory labels
  will not fix it. The test is an offline probe-decodability read (obs → cap-doom) on the
  over-confident subset **specifically**.
- **It is currently UNDERPOWERED and BLOCKED (`195ce9f`).** The subset is ~47 games across mixed
  architectures; current-arch caps are n = 14. The blocker is named: **the rust `forcelose` gap** —
  cap records carry no terminal before 2026-08-24 (**0/40 before the fix, 8/8 after**; 8 of 48
  current-arch cap battles replayable, n = 3 held-out caps). **Scarcity is TRANSIENT** — every
  post-fix trace carries the terminal, so the rev-4 and verdict batteries supply the data by
  existing.
- Whether the harvest diet (mid-game re-seeding + prioritized selection + slice re-weighting)
  actually moves the 35%, given that the naive pilot regressed.

## Pros

- The mechanism is fully accounted for by three data-shaped causes, each of which the registered
  harvest attacks directly — the cheapest possible first attempt.
- The harvest design's own components are principled and semantics-free: **mid-game re-seeding +
  multi-rollout** turns one bit into a tight per-state win RATE; **Beta-evidence prioritized
  selection** hammers the over-confident 35% automatically; **slice re-weighting** makes late-game
  accuracy worth what it should be to the optimizer.
- If the diet works, the whole clean-world design's weakest dependency is repaired inside the clean
  worldview — outcome units, no hand-coded term.
- The chain closes the POLICY's hardest problem indirectly: harvest → the head learns the slide →
  PBRS-φ meters the terminal badness out along the drag as per-turn shaping → the policy's problem
  reduces to one-step learning. *The value channel carries the time structure so the policy channel
  never has to.*

## Cons

- **Naive fine-tuning is measured DESTRUCTIVE** (`195ce9f`) — the harvest pilot regressed, caught
  only by the untouched control, because the fit set never covered the meter's region.
  `--anchor-coef` defaults ON for exactly this reason.
- The traced cap fraction is quota-sampled and **must NEVER be quoted as a stall-rate baseline** for
  the clean-world escalation trigger.
- A true multiplicative VETO may simply not be learnable from a sum-through-a-squash at any data
  volume, in which case this lever hands off to the (deliberately gated) multiplicative-structure
  programme.
- Everything here is measured on recorded values across mixed architectures; the current-arch cell
  is thin.

## Next test

Strictly ordered, and the order is the point:

1. **Accumulate post-fix cap traces** (they arrive free with the rev-4 / verdict batteries; a
   purpose-built bridge self-play harvest is the cheap accelerator).
2. **The REDUCIBILITY probe on the current arch** — offline probe decodability (obs → cap-doom) on
   the over-confident subset, plus the harvest→fine-tune→`harvest_meter` loop pre/post.
3. **Only then**, if the diet measurably fails, the head-side **learned multiplicative gate** on
   (clock, context) modulating φ — outcome units, inside the clean worldview — which is the
   gated-readout rung of the general multiplicative programme
   ([multiplicative_structure.md](multiplicative_structure.md)).
