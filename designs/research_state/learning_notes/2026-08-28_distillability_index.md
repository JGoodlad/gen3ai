# Learning note — the distillability index: what it would mean (2026-08-28)

> ⚠️ **STATUS BANNER:** written while the instrument was being built (agent dispatched same day).
> The plasticity forensics (same day, later) refuted the specific "v8 was rigid" account, but the
> index itself measures a property (absorption-per-collateral) that is hypothesis-neutral — it
> remains the right instrument; only the *expected shape* of its curve is now open.

## The definition

Take a checkpoint. Push a fixed teacher's behavior into it with a standardized micro-distillation
(same teacher, same dose, same states, every time) and measure two curves: how fast the student
comes to agree with the teacher on the teacher's specialty (**absorption**), and how much the
student's behavior elsewhere drifts from what it was (**collateral**). The index = the collateral
paid at a fixed absorption level. One number per checkpoint; low is good.

## What it represents

It measures the model **as a medium, not as a player**. Every metric we track (Elo, piloting win
rate) asks "how good is this network at Pokémon?" The index asks "how good is it at *being
taught*?" — a separate property that can move independently of strength.

## What it would buy

1. **A fold-timing signal** — fold when the curve says the medium is ready, not when the calendar
   does.
2. **Honest compute accounting** — if the index rises through a training run's flat tail, that
   compute is visibly buying an asset, and "when do we stop a generation" becomes a two-currency
   decision.
3. **Cross-generation comparability** — architecture/regularizer choices scored on how early they
   make a trunk teachable.
4. **The big one: measurable ⇒ optimizable.** "Train FOR distillability" becomes a legal
   experiment — consolidation-promoting regularizers could make the high-efficiency fold regime a
   design choice instead of an end-of-generation reward.

## Honest caveats

A micro-probe is not a fold: no PPO running beside it, no team-bias sampling, no 3M-step
timescale — it measures *capacity to absorb*, not a full fold's outcome. Validity rests on
correlating with real fold efficiency (one anchor today: rev-2's 13%). Failure mode: measuring
something learning-rate-flavored instead of geometric — hence collateral in the denominator, and
hence the sanity cells (a fresh-init network must read "absorbs fast, damages everything" or the
instrument is not admitted).
