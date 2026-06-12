# Self-KO over-valuation (healthy Explosion)

**Status:** ✅ confirmed, 🛠 fix built (not shipped) · **Ledger id:** H1

One-line claim: *the policy throws away healthy mons to Explosion because the critic over-values the
trade, neutralizing the (correct) negative reward in the PPO advantage.*

## Known (adversarially verified — 3 wrong hypotheses ruled out first)
- ~38% of all Explosion/Self-Destruct selections are at ≥80% HP (incl. turn-1 full-HP Metagross),
  chosen with median policy conf ~0.5 (a LEARNED preference, not exploration tail).
- Reward is CORRECT (−2.7; the finishing-blow mis-credit is already guarded) → not a reward bug.
- Exploration ruled out (conf ~0.5). ① `value_active_readout` exonerated (the no-① baseline explodes
  just as much; the "0 explosions" reading was a script bug).
- **The critic over-values the result: dV +2.9 on a 98%-HP trade → PPO advantage +1.5, 74% ≥0.** That
  positive advantage is why the policy never un-learns it. Root: the symmetric material PBRS prices a
  1-for-1 healthy trade at ~0.

## Not-known
- The retrain A/B result: does `--self-ko-hp-penalty 2.5` drop the healthy-explosion rate (toward the
  human ~0–5%) and lift `win_rate_vs_bots`? (Static pre-check flips the trade advantage negative;
  expect the critic over-valuation to also shrink as the TD target sharpens.)
- True win-rate size: the census says confirmed blunders (incl. this) are only a *few points* of the
  18% gap — so this is real but NOT the whole floor.

## Pros
- Concrete, human-obvious, falsifier-confirmed (+1–1.7 mon). Architecture-light (a reward term).
- Cheap A/B; resume-immutable so a clean fresh-run test.

## Cons
- Only a couple of points of the gap (per the census) — not a silver bullet.
- The fix sharpens the reward, but the *critic* over-valuation is the deeper cause; if it persists post
  reward-fix, a representation fix is the backup.

## Next test
- A fresh run with `--self-ko-hp-penalty 2.5`. Gate: healthy-explosion rate ↓ + `win_rate_vs_bots` ↑,
  no regression. Re-verify with `decision-table <run> --cat selfko` (watch the selfko `dV_med` fall).
