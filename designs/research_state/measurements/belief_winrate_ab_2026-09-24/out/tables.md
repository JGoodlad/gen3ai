# Belief win-rate A/B — generated tables (`scripts/analyze.py report`)

Primary indices OK: **2996** of 3000 complete (dropped 4). Secondary indices OK: 2996. INCONCLUSIVE cells: none.

| cell | attempted | failed | share |
|---|---|---|---|
| pool:learned | 3000 | 0 | 0.00% |
| pool:prior | 3000 | 0 | 0.00% |
| ladder:learned | 3000 | 1 | 0.03% |
| ladder:prior | 3000 | 3 | 0.10% |
| pool:move | 3000 | 0 | 0.00% |
| ladder:move | 3000 | 1 | 0.03% |

## PRIMARY (n = 2996 paired indices)

| cell | WR | 95% CI |
|---|---|---|
| pool:learned | 0.4838 | [0.4658, 0.5017] |
| pool:prior | 0.4321 | [0.4142, 0.4496] |
| ladder:learned | 0.6649 | [0.6479, 0.6819] |
| ladder:prior | 0.6614 | [0.6442, 0.6786] |

| contrast | point (pp) | 95% CI (pp) | read |
|---|---|---|---|
| E_pool = WR_prior - WR_learned (POOL) | -5.2 | [-7.4, -3.0] | NEGATIVE |
| E_ladder = WR_prior - WR_learned (LADDER) | -0.4 | [-2.5, +1.7] | NOT DETECTED |
| DiD = E_ladder - E_pool | +4.8 | [+1.9, +7.8] | POSITIVE |
| WR_learned ladder - pool | +18.1 | [+15.6, +20.5] | POSITIVE |

Discordant-outcome share (index-level, vs LEARNED): pool:prior 37.3%, ladder:prior 34.2%

## SECONDARY (n = 2996 paired indices)

| cell | WR | 95% CI |
|---|---|---|
| pool:learned | 0.4838 | [0.4656, 0.5015] |
| pool:prior | 0.4321 | [0.4146, 0.4499] |
| ladder:learned | 0.6649 | [0.6482, 0.6817] |
| ladder:prior | 0.6614 | [0.6444, 0.6781] |
| pool:move | 0.4282 | [0.4109, 0.4461] |
| ladder:move | 0.6532 | [0.6362, 0.6701] |

| contrast | point (pp) | 95% CI (pp) | read |
|---|---|---|---|
| E^move_pool = WR_move - WR_learned (POOL) | -5.6 | [-7.7, -3.3] | NEGATIVE |
| E^move_ladder = WR_move - WR_learned (LADDER) | -1.2 | [-3.2, +0.9] | NOT DETECTED |
| DiD^move | +4.4 | [+1.3, +7.4] | POSITIVE |

Discordant-outcome share (index-level, vs LEARNED): pool:prior 37.3%, pool:move 38.6%, ladder:prior 34.2%, ladder:move 33.9%

## SIDE READ — greedy-action change rate along the LEARNED trajectories

Sanity (OFF re-forward ≠ committed action): 0 (must be 0).

| column:scope | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=all |
|---|---|---|---|---|---|---|---|---|
| pool:all | — | 18.9% [17.9, 19.9] (n=7966) | 18.8% [17.8, 19.9] (n=14052) | 18.7% [18.0, 19.4] (n=18844) | 17.5% [16.8, 18.2] (n=25658) | 16.5% [16.0, 17.1] (n=34418) | 14.5% [13.9, 15.1] (n=44456) | 16.7% [16.4, 17.0] (n=145394) |
| pool:move | — | 16.4% [15.5, 17.4] (n=7966) | 17.3% [16.4, 18.4] (n=14052) | 16.8% [16.1, 17.5] (n=18844) | 15.8% [15.2, 16.5] (n=25658) | 14.4% [13.8, 14.9] (n=34418) | 11.9% [11.4, 12.5] (n=44456) | 14.6% [14.3, 14.9] (n=145394) |
| ladder:all | — | 18.5% [17.6, 19.5] (n=8282) | 17.7% [16.9, 18.5] (n=14964) | 17.8% [17.1, 18.6] (n=20314) | 17.1% [16.4, 17.8] (n=26795) | 15.7% [15.1, 16.3] (n=35096) | 14.3% [13.7, 14.9] (n=45622) | 16.2% [15.9, 16.5] (n=151073) |
| ladder:move | — | 17.5% [16.6, 18.5] (n=8282) | 16.4% [15.6, 17.2] (n=14964) | 16.4% [15.7, 17.2] (n=20314) | 15.7% [15.0, 16.5] (n=26795) | 14.4% [13.9, 15.0] (n=35096) | 12.6% [12.1, 13.1] (n=45622) | 14.7% [14.4, 15.0] (n=151073) |

## Descriptive

| cell | mean turns | max turns | mean decisions | draws |
|---|---|---|---|---|
| pool:learned | 44.3 | 250 | 48.5 | 5 |
| pool:prior | 41.6 | 250 | 46.0 | 9 |
| ladder:learned | 46.7 | 250 | 50.4 | 8 |
| ladder:prior | 46.5 | 250 | 50.2 | 7 |
| pool:move | 42.5 | 250 | 46.9 | 6 |
| ladder:move | 46.4 | 250 | 50.1 | 10 |

## Failures

* ladder:prior i=1492 timeout: wall>1800.0s
* ladder:prior i=2542 error: UnknownVolatileError: volatile 'typechange' has no gen3 encoding slot — classify it in gen3_effects.py (binary / trap / counter) before training. Known: ['attract', 'bide', 'bind', 'charge', 'clamp', 'confusion', 'curse', 'defensecurl', 'destinybond', 'disable', 'doomdesire', 'encore', 'endure', 'firespin', 'flashfire', 'flinch', 'focusenergy', 'focuspunch', 'followme', 'foresight', 'futuresight',
* ladder:prior i=2345 timeout: wall>1800.0s
* ladder:move i=2542 error: UnknownVolatileError: volatile 'typechange' has no gen3 encoding slot — classify it in gen3_effects.py (binary / trap / counter) before training. Known: ['attract', 'bide', 'bind', 'charge', 'clamp', 'confusion', 'curse', 'defensecurl', 'destinybond', 'disable', 'doomdesire', 'encore', 'endure', 'firespin', 'flashfire', 'flinch', 'focusenergy', 'focuspunch', 'followme', 'foresight', 'futuresight',
* ladder:learned i=1541 error: AssertionError: Error with move solarbeam. Expected self.moves to contain copycat, metronome, mefirst, mirrormove, or assist, or to have the ability dancer. Got {'counter': counter (Move object), 'seismictoss': seismictoss (Move object), 'explosion': explosion (Move object), 'rollout': rollout (Move object)}, ability: clearbody
