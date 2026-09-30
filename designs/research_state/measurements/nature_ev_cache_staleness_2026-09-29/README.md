# How often `Gen3Env` serves stale nature/EV labels (2026-09-29)

This is **step 0** of the TECH_DEBT row *"Stale nature/EV label cache in `Gen3Env`"* (Lane C F-LC-6). It is a measurement only. The fix is batched to the next obs ARCH change.

## The defect

`Gen3Env._nature_ev_map` caches the opponent team's inverted `(species → nature, EVs)` map, and the cache is keyed only by the team's species SET. The cache lives on the env instance across battles. So an env whose next opponent team has the same six species but a different spread on any of them gets the previous team's nature/EV labels for that whole battle.

## The draw (read from the code)

The production opponent side is `matchup.opponent_teams.build` → `Gen3Teambuilder(all_teams)` for the "opponent teams: full pool" matchup. Every opponent kind draws from it. It runs with `--team-pfsp off --team-block-episodes 1`, so it makes one uniform `random.choice(packed_teams)` per battle, per env worker. The rate per battle is therefore the exact pairwise rate over the packed pool: `#{ordered pairs (i, j): same species set, different spread} / N²`.

## Result

| Source | Teams | Species sets (shared by 2+) | P(same set, consecutive) | **P(stale labels) per battle, EXACT** | Simulated through the real draw, 48 envs × 100k (batch-means 95 %) |
|---|---|---|---|---|---|
| **Training pool** (719 texts → 719 packed, 717 distinct) | 719 | 602 (82) | 0.2025 % | **0.0619 %** (320 of 719² ordered pairs) | 0.0653 % [0.0627, 0.0679] |
| **Ladder corpus** (`ladder_corpus` full tier; not a training source yet, X9) | 22,813 | 7,624 (2,541) | 0.1797 % | **0.1753 %** | 0.1773 % [0.1726, 0.1819] |

- **Why the exact value is the pool's answer:** under an independent uniform draw, the pairwise value IS the long-run rate.
- **The simulation only checks that the production draw is that draw.** Over one 20M-draw stream with the same RNG, the real `yield_team()` and a plain `random.choice(packed_teams)` both read **0.06247 %**.
- **Seed scatter:** the default seed's 4.8M-transition run reads a little high: 0.0653 %, and its CI excludes 0.0619 % at about 3σ. Two more seeds (11, 22) read 0.0628 % and 0.0636 %, and both CIs contain it. Three plain uniform-index streams of 4M read 0.0632 / 0.0588 / 0.0599 %. That is seed variation, not a sampler difference.
- **The batch-means interval** (per-env streams) is used because consecutive transitions share a draw. In practice it is about as wide as the naive Wilson interval printed beside it.

**What that means at training scale:** at ~45 trainee decisions per episode (the watchers' `ep_len`), a 75M-step run like N0 plays ~1.7M battles. About **1,000** of them (0.06 %) carry the previous team's nature/EV labels. Only the revealed species whose spreads differ are mislabelled (masked slots carry no label). Those same battles are where Python and the Rust core will disagree at M5's training integration: Rust recomputes per episode.

**The ladder corpus runs ~2.8× the pool's rate (0.175 %),** and the collisions are concentrated there. The top 5 species sets carry 47 % of its stale pairs, the standard Gen-3 OU cores. For example `aerodactyl/gengar/skarmory/swampert/tyranitar/zapdos` has 364 teams and 361 distinct spreads. When X9 moves opponents to ladder teams, the rate roughly triples, still under 0.2 %.

### Worst offenders

**Training pool** (80 of its species sets produce stale pairs; the top 5 = 20.6 % of them):

| Share of stale pairs | Teams | Spreads | Species set |
|---|---|---|---|
| 5.6 % | 5 | 4 | blissey / dugtrio / forretress / gengar / swampert / tyranitar |
| 3.8 % | 4 | 4 | blissey / gengar / skarmory / starmie / swampert / tyranitar |
| 3.8 % | 4 | 4 | aerodactyl / gengar / skarmory / swampert / tyranitar / zapdos |
| 3.8 % | 4 | 4 | blissey / claydol / dugtrio / milotic / skarmory / tyranitar |
| 3.8 % | 4 | 4 | aerodactyl / celebi / magneton / skarmory / swampert / tyranitar |

**Which fields differ.** Across the (stale pair, species) cases in the pool: EVs 1,396; IVs 772; nature 684. Some IV differences come from `fix_gen3_hp_ivs`, which gives each Hidden Power type its IVs. A difference that inverts to the same (nature, EVs) is still counted, so the rate is an **upper bound** on label-visible staleness.

## Reproduce

```bash
export PYTHONPATH=$PYTHONPATH:src
python designs/research_state/measurements/nature_ev_cache_staleness_2026-09-29/measure.py \
    --sim-per-env 100000 --envs 48 --json <out.json>        # ~75 s, CPU; result.txt / result.json here
```

"Spread" = the packed team's per-species `(nature, EVs, IVs, level)`, which is what the sim computes the derived stats from.
