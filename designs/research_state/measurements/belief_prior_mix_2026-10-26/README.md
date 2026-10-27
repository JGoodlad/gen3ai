# The belief prior mix against Kakuna — results (2026-10-26)

Registered in [`PREREG.md`](PREREG.md) (shipped `4ee655648`, before any counted game), specs `read_spec_l025.json` /
`read_spec_l100.json` (pin `4ee655648`, shipped `ef6f6bab1`). The first live use of `main.reads`.

Abbreviations: pp = percentage points; P\* = `rb_x26_s2601` @280,000,170; H = `heldout72_hpfix`; F = `forum72`; U =
H, F and P72 pooled with equal weight; Δ = the arm minus X51's banked P\* (λ = 0), unit-paired on the same seeds.

## λ = 0.25 (PRIMARY): NOT DETECTED

`record`: `belief_prior_mix_l025_2026-10-26: lambda=0.25: NOT DETECTED` (rule `prior_mix_ab`; regimes
`arm_l025` `3d60d85393bd49f6`, guard `d44d454737bcba71`). Files: `read_l025/` (`cells.json`, `inputs.json`,
`decision.json`, `record.json`).

| | λ = 0.25 W / n | score [Wilson 95 %] | banked λ = 0 | Δ unit-paired [95 %] |
|---|---|---|---|---|
| H | 195 / 400 | 0.487 [0.439, 0.536] | 199 / 400 | −1.00 pp [−7.58, +5.58] |
| F | 217 / 400 | 0.542 [0.494, 0.591] | 212 / 400 | +1.25 pp [−6.57, +9.07] |
| P72 | 190 / 400 | 0.475 [0.427, 0.524] | 173 / 400 | +4.25 pp [−3.35, +11.85] |
| **U** | | **0.502 [0.473, 0.530]** | 0.487 | **+1.50 pp [−2.74, +5.74]** (Newcombe +1.50 [−2.49, +5.48]) |
| **home72** | 237 / 400 | 0.593 [0.544, 0.640] | 239 / 400 | **−0.50 pp [−5.32, +4.32]**, not HARMED |

- **Not VOID.** The guard (P\* unchanged, unit seed 0, all four team sets) replays X51 GAME FOR GAME (half, index, our
  team, result, turns): 12 / 11 / 6 / 13 wins, equal to X51's. 80 / 80 arm units and 4 / 4 guard units counted.
- **The prediction** (Δ_U ≈ +1.5 pp, NOT DETECTED; home72 ≈ −1 pp) matched.
- **Reading.** Anchoring the belief a quarter of the way to the prior at inference does not detectably change play
  against Kakuna, on unfamiliar teams or on our own. The repaired calibration (P10) is not worth a detectable number
  of games at this sample, through a policy trained on the unmixed belief. An effect below ≈ 5 pp is not excluded.

## λ = 1.0 (D0f, prior only): NOT DETECTED; H3 NOT KILLED

`record`: `belief_prior_mix_l100_2026-10-26: lambda=1: NOT DETECTED, D0f H3 NOT KILLED` (regimes `arm_l100`
`b62e25ee3852fbc5`, guard `d44d454737bcba71`). Files: `read_l100/`.

| | λ = 1 W / n | score [Wilson 95 %] | banked λ = 0 | Δ unit-paired [95 %] |
|---|---|---|---|---|
| H | 197 / 400 | 0.492 [0.444, 0.541] | 199 / 400 | −0.50 pp [−6.65, +5.65] |
| F | 192 / 400 | 0.480 [0.431, 0.529] | 212 / 400 | −5.00 pp [−12.35, +2.35] |
| P72 | 177 / 400 | 0.443 [0.395, 0.491] | 173 / 400 | +1.00 pp [−6.06, +8.06] |
| **U** | | **0.472 [0.444, 0.500]** | 0.487 | **−1.50 pp [−5.47, +2.47]** (Newcombe −1.50 [−5.48, +2.49]) |
| **home72** | 234 / 400 | 0.585 [0.536, 0.632] | 239 / 400 | **−1.25 pp [−6.71, +4.21]**, not HARMED |

- **D0f:** DiD (U − home72) **−0.25 pp [−7.00, +6.50]**; its UB is above +3.0, so **H3 is NOT KILLED** by the
  registered rule, because the interval is wide, not because the read shows an H3 effect (the point estimate is ≈ 0).
- **Not VOID:** the guard (unit seed 10, all four sets) replays X51 game for game; 80 / 80 + 4 / 4 units counted.
- **The prediction was wrong** on home72 (predicted HARMED ≈ −5 pp; read −1.25, not harmed) and on the DiD (predicted
  ≈ +4; read −0.25). Removing the learned deltas entirely costs no detectable strength against Kakuna on our own teams,
  and gains none on unfamiliar ones.
- Held twice by the orchestrator for CPU priority (λ = 0.25 first, then D3's look-1 read); both holds killed only the
  supervisor, and each in-flight unit counted on resume.

## Both arms

Neither anchoring the belief toward the prior (λ = 0.25) nor replacing it with the prior (λ = 1) moves Kakuna games
detectably (U +1.5 / −1.5 pp, home72 −0.5 / −1.25 pp, every interval ≈ ±4–5 pp). The deltas' memorisation is real on
calibration (the review's P5) and invisible in games at this resolution, consistent with X37a putting the Kakuna gap
in PILOTING, not FACING. Lever 1 (b) / (c) (anchoring in training) and D7 stay unranked: their registered trigger,
a game effect of the deltas off-pool, did not fire.

## FINDINGS (the first live use of `main.reads`)

1. **Pairing did not tighten the interval.** The unit-paired half-width of Δ_U is 4.24 pp against the unpaired
   Newcombe's 3.99: a λ arm's games diverge from their twins early and then correlate with them little. The PREREG's
   power assumed ρ = 0.25–0.5 (half-width 3.3–4.1 pp); the read detects ≈ ≥ 6 pp at 80 % power.
2. **`setsid nohup … &`'s `$!` is the setsid wrapper's PID**, gone within seconds, not the supervisor's (the form
   `designs/tools/reads.md` §3 shows). Find the supervisor by its argv.
3. **A model-load option needs its own anchor group.** It changes `our_regime`, and a group holds one regime per
   team set, so each λ is a group (as the specs do).
4. **No banked-control path existed for an anchor rule.** `prior_mix_ab` added `rules.banked_units` (another read's
   committed per-game rows, sha-checked) and per-game identities on every unit row (`assemble.unit_games`).
5. **Holding a read is killing its supervisor by PID.** The in-flight unit runs in its own session, finishes and
   counts on resume; there is no `hold` verb.
6. **One unit attempt (guard forum72) did not count: `peer_clean = False`.** The automatic retry counted (no game lost).
7. **The queue set the wall time**: λ = 0.25's 84 units took 12.6 h.
8. **`record` needs every planned unit played**, so a stop at the rule's `min_unit_frac` cannot be recorded; and the
   spec's `pin` must exist before the spec, so a PREREG and its specs land in two commits.
