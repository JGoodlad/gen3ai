# Attack type-mismatch (wrong-effectiveness move pick)

**Status:** ✅ confirmed (small) · **Ledger id:** H2

One-line claim: *the policy sometimes confidently fires a resisted/immune move while a super-effective
move sits in the SAME mon's kit — a representation gap, not a reward/critic blunder.*

## Known
- Falsifier-confirmed + independent type-chart cross-check (e.g. Gengar Giga Drain into Zapdos → should
  Ice Punch; +1–2 mon material on the clean cases).
- It's a REPRESENTATION gap: reward + critic are both CORRECT here (dV negative, the opposite signature
  to self-KO) — the policy just mis-reads its own move effectiveness vs the current target.

## Not-known
- After hard decontamination (confident, in a loss, a strictly-better non-immune move existed, not
  forced/Hidden-Power/fixed-damage), the truly-clean core is only ~3–5 decisions out of 11,580 loss
  decisions (~0.04%). So: real, but does fixing it move win-rate measurably? (Likely <1%.)

## Pros
- Cheap, double-confirmed, and the machinery exists (move-effect obs block + type chart in `gen3_data`).
- A clean representation A/B (feature ON vs OFF), distinct class from the reward levers.

## Cons
- Small — a fraction of a percent. A secondary lever, not a primary one.
- Mostly the immune-attack cluster is in WON games (not game-losing) or lacks a 2× alternative.

## Next test
- Add/strengthen a per-move-vs-active effectiveness feature in the action-aligned move-effect obs block;
  A/B the immune/resisted-pick rate per mature loss (should fall toward 0) with no win-rate regression.
