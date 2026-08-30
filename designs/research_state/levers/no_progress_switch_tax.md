# The `no_progress_tax` is a de-facto SWITCH TAX — and a candidate cause of under-switching

**Status:** 🔬 **OPEN — the causal arm costs nothing; fixes landed OFF at v106** · **Ledger:**
`bcec45c` (probe M registered) · `1d5a866` (probe M verdict) · `579279d` (probe N dispatched) ·
`cfbc9bf` (probe N verdict) · `132d198` (wave D landed, config v106)

One-line claim: *the single surviving hand-coded BIAS term pays roughly −0.101 reward per decision
against switching, while the win-prob head rates switching significantly BETTER — and it does so
because the toll was designed against +0.35 of counterweights that a later commit deleted.*

## Known (cleared the honesty gates)

Record: `measurements/bias_tax_head_alignment_2026-08-29.{json,md}` (5,035 battles / 147k
decisions, exact model-free reconstruction) and
`measurements/no_progress_tax_review_2026-08-29.md` (every claim with sha and `file:line`).

- **SCOPE COLLAPSE (`1d5a866`).** `--all-shaping-pbrs` has been default-ON since 2026-08-18, so of
  29 registry BIAS members exactly **ONE** is live: production reward = **1 TERMINAL + 7 PBRS + 1
  BIAS (`no_progress_tax`)**. `stall_tax` is zeroed; the heal-war grace is a ProgressClock branch,
  not a reward; the draw penalty is TERMINAL. The remembered anti-stall family has not run in
  months.
- **Method note worth keeping:** the fold→window alignment was **MEASURED, not assumed** — and the
  intuitive alignment was wrong (**0 / 10,442** vs **8,710 / 10,424**). Probe M would have reported
  a confident null on a mis-joined table.
- **P1 (alignment ≥70%) REFUTED.** 45.7% vs a **44.7% matched control** — the tax carries **no
  win-prob information beyond phase and action kind**; every matched-control diff is null. So it is
  *not* annealable scaffolding that a properly-dosed PBRS-φ could subsume.
- **P2 (over-tax) HELD, far larger than predicted, and the decomposition is BEHAVIORAL.**
  **48.5% of charges have Δφ ≥ 0.** **73% of voluntary switches are charged vs 6.7% of moves; 79%
  of all charges land on switches; 36% land on ZERO-AGENCY post-faint replacements.** Implied
  differential **−0.101 reward/decision against switching** while the head rates switching
  **+0.0042 win-prob BETTER (SIG)** — opposite signs. Taxed switches are **+0.0103 better** than
  untaxed ones.
- **P3: the exemptions are mostly RIGHT.** In-grace heals gain +1.6pp; the freeze protects
  dice-ruined turns. The rule's *exemption* logic is not the defect.
- **🚨 The intent verdict is COMPOSITION DRIFT (`cfbc9bf`).** Charging a voluntary switch was
  **designed in writing** (design doc `6a2cab4`: "a pure tempo-pivot that lands nothing pays the
  front-loaded toll once (correctly)") — **but designed inside a reward where a switch also
  collected +0.35 net** of bonuses (`SWITCH_BASE_BONUS` 0.5, se_switch, escape_threat, pivot_*).
  Commit **`928a00b` (2026-06-12) zeroed every counterweight and kept the toll**; the net sign on
  switching flipped **+0.35 → −0.15** and nobody re-derived the term. The manager still carries a
  comment describing the dead world.
- **Three implementation defects, confirmed line by line.** (1) The **SITOUT guard is off by one
  window** — it exempts the KO turn and charges the replacement, and leaves the corpus's costliest
  class (−5.1pp × 13.2% of decisions) exempt. Root cause named and generalisable: *an attribute
  minted for the obs slot where its tense is correct, reused in a different tense — same name, both
  readings true of the same delta, hence untestable.* (2) **No switch can ever satisfy
  `_is_progress`** — the term prices an action KIND, not progress. (3) NEW: the **trapped gate reads
  the upcoming legal too** (same three lines, 2.9%).
- **Per-path intent audit:** seven charge paths INTENDED; the voluntary switch DESIGNED-then-net-
  DRIFTED at `928a00b`; the forced switch **a bug at origin**.
- **Fixes are BUILT and OFF (`132d198`, config v106).** F1 (a new prev-tense field + threading the
  previous legal — restores 36% of charges) and F2b (freeze-not-charge voluntary switches; F2a was
  rejected because it would reintroduce the hand-coded switch heuristic `928a00b` deleted), plus the
  staller per-instance RNG. **Obs-parity honesty: these are retrain-class BECAUSE the clock scalar
  (obs col 1602) IS the charge basis** — the change is confined to that one column, proven by a
  per-column integration test, defaults byte-identical against a `git show`-loaded old clock on all
  15 reference windows. Probe M's discriminator was reproduced as a unit test (0/3 vs 3/3 moving
  columns under the flag).

## Not-known

- **Whether it CAUSES the under-switching pathology.** The hypothesis is registered, not measured:
  the policy switches ~16% of voluntary decisions vs strong humans' ~30% (the faithful
  human-agreement probe; the human half reproduces model-free at 28.96% over 30,146 reconstructed
  ≥1500 decisions). A reward term paying −0.101/decision against switching for a whole generation is
  exactly the shape that produces it — but "exactly the shape that produces it" is a hypothesis.
- Whether removing the tax costs anything the tax was actually buying (the anti-stall intent), now
  that draw-penalty and the ProgressClock carry that job.
- Whether F1 + F2b (intent restoration) or plain OFF is the better endpoint. They are different
  arms and should not be conflated.

## Pros

- **The causal arm is FREE and needs no code: `--no-progress-penalty 0.0`**, with switch rate as the
  registered endpoint. Nothing else on the frontier has this cost profile.
- If it lands, it retires a pathology that has survived every previous framing (representation →
  valuation → commitment/sharpness) — all of which looked at the network and none of which looked at
  the reward.
- The win-prob head served here as an **auditor of the owner's own priors**, with the owner's
  blessing, and found a real defect. That instrument is now proven and reusable on any hand-coded
  term.
- The fixes are already built, flag-gated, revert-verified and byte-identical when off.

## Cons

- **Reward changes are retrain-class**, and deliberately not touched mid-campaign — this cannot be
  A/B'd cheaply on a live run.
- The clean-world arm changes many things at once, so it cannot serve as the causal test; the
  tax-only arm must run separately or the attribution is lost.
- The head's +0.0042 preference for switching is a *win-prob* claim on recorded states, not a
  demonstration that switching more would have won more games — the transfer lesson (τ = 0.17)
  applies to any per-decision claim.
- Under-switching may be over-determined: removing the tax could move the rate without moving
  strength.

## Next test

**The tax-OFF arm** — `--no-progress-penalty 0.0`, everything else matched, with **switch rate** as
the primary endpoint and win rate / stall rate as guards. Sequenced next-era with the other
retrain-class reward items (`132d198`'s F1 / F2b enablement, the staller-RNG fix), post-verdict. It
is also registered as a secondary endpoint of the clean-world A/B, where its reading will be
suggestive rather than causal.
