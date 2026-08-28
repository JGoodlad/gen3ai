# Learning note — plasticity, maturity, and the two currencies (2026-08-28)

> ⚠️ **STATUS BANNER (added same day):** the plasticity forensics measurement
> (`designs/research_state/measurements/plasticity_forensics_v8_vs_gen_2026-08-28.md`) has since
> **REFUTED the application of this note's framing to our two eras**: the v8 parent shows NO
> capacity loss (Lyle capacity_ratio 1.154 vs rev-1's 0.948 — P1 refuted, opposite direction) and
> its teachers renovated the TRUNK MORE than ours (P3 refuted, opposite). The *concepts* below
> (how plasticity is measured, why a coefficient is not rigidity, the two-currencies frame) stand
> as ML background. The *claim* "v8 distilled well because its parent was rigid" does not — see
> the 2026-08-28 ledger entry for the replacement account (teacher differentiation +
> drift-anchoring).

## 1. How do we measure plasticity?

Plasticity is not one number; the field uses proxies, three of which are instrumented in this
repo (`python -m main.capacity`):

- **Trainability (the Lyle probe — the most direct).** How fast do the current weights fit a
  fresh target vs a freshly initialized network fitting the same target? Plastic = as fast as
  fresh; plasticity-lost = slower.
- **Effective rank of representations.** How many independent directions the features actually
  use; collapsing rank is the classic plasticity-loss signature. Caveat from our own arms: rank
  is noisy as a *harm* signal (four arms at rank 12.4–13.2 spanned −7.5pp to +4.4pp).
- **Dormant-unit fraction (ReDo).** Vacuous on our Tanh architecture — a meter that cannot fire
  is not evidence of health.

Plus the **behavioral** proxy that drove the fold week: how much the *function* moves per unit of
training pressure (the transfer gate: fork, train narrow, measure whether general competence
moved).

## 2. How would we know v8 was "mature"?

Direct evidence: ~276M steps, flat learning curve, small per-update policy movement, a plateau
diagnosed as genuine convergence. Retrospective evidence: its forks specialized without losing
general competence. Honest gap at the time of writing: no Lyle probe was ever run on v8-era
checkpoints — and when the forensics ran one (era-pinned), **the v8 parent measured MORE
trainable than rev-1**, which is why the banner above exists. "Converged in loss" and "rigid in
capacity" turned out to be different properties.

## 3. Radical change vs fine-tune — how to tell

Weight distance is nearly useless (symmetries). The measures that work:

- **Function-space:** policy KL between fork and parent on shared states.
- **Representation-space:** CKA (Centered Kernel Alignment) between the two networks' activations
  on the same inputs; or probe-transfer (does a linear probe trained on parent features still
  decode on fork features?). Probe transfers = fine-tune; breaks = renovation.
- **Location:** change concentrated in heads = patch; in trunk/encoders = rebuild.

## 4. Why isn't turning down a coefficient the same as rigidity?

A coefficient scales the MAGNITUDE of change; rigidity controls the DIRECTION. A scalar
multiplies every gradient uniformly — specialization and collateral renovation slow down in the
same proportion. A converged model separates them *by geometry*: at a minimum, old-task gradients
are ≈0 and the loss surface is curved — steep in directions old competence cares about (movement
is punished and pulled back), flat in directions it is indifferent to. New learning gets
channeled into the flat directions: an annex, not a remodel, enforced by walls rather than by a
knob. A startup told to "change more slowly" still remodels; a mature institution builds annexes
because the structure resists.

## 5. The two currencies

Playing strength (Elo — visible, bought early) and substrate maturity / distillability
(invisible in Elo, hypothesized to be bought by the flat tail of training). The owner's framing:
v8's late training didn't make it much better at *playing*; it made it better at *being in a
representational geometry that allowed distillation*. This note originally endorsed that as the
best-fitting story with an honesty flag ("explains everything at once is also what a good
just-so story does") and named the two measurements that would test it — the forensics ran, and
the just-so story lost on two of four predictions. The two-currencies FRAME survives (the
distillability index still measures the second currency); the specific account of where v8's
second-currency wealth came from did not.
