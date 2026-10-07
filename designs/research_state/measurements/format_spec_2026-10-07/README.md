# Format spec measurements — 2026-10-07

The gen3ou FORMAT SPEC (`agents.gen3_data.format_spec`, `designs/endstate/design_format_spec.md`): what the
ladder's rules say about our pools, our priors and the pinned engine. All CPU, deterministic, from the repo root
(`PYTHONPATH=src`).

| file | command | reads |
|---|---|---|
| `team_legality.out` | `python -m main.team_legality` | every pool vs the LADDER's rules: training pool 719 / 719 legal, Smogon sample 32 / 32, promoted 40 / 40, specialist 3 / 3; 8 `others/yak_attack` entries already marked `valid: false` by the acquisition-time Showdown validator are illegal (the same rules); no pool team holds Quick Claw |
| `procedural_legality.out` | `python procedural_legality.py 1000` | the procedural generator (`ou_random_teams.js`, seed 20260924): 19 / 1,000 teams (1.9 %) hold Quick Claw — legal on the pinned validator, illegal on the ladder |
| `format_drift.out` | `python -m main.format_drift check --pinned` then `check` | the pinned engine differs from master by exactly Quick Claw (banlist) and Recycle (One Boost Passer); the committed master snapshot (`c046106c`) matches the spec |
| `prior_mass.out` | `python prior_mass.py` | the mass the held prior filter moves, on the CURRENT committed priors: Quick Claw 0.32 % of all sets (182 species, max 51 % Hypno), Dugtrio / Diglett / Gligar / Electrode Sand Veil–Soundproof 0.01 → 0, Voltorb 0.5 → 0, 8 ability-locked rows dropped; 0 banned moves / teammates / species with mass; model-table floors: 789 move cells, 3 item columns, 20 species |

The master files were fetched at commit `c046106cbe075931b1ff8d8b800ff5be47a85f96` (2026-10-07) and checked
byte-equal to the earlier cache; `src/main/format_drift_snapshot/MANIFEST.json` records each full file's sha256.

## The prior filter as landed (`gen3_format_spec_priors_v1`)

| file | command | reads |
|---|---|---|
| `obs_golden_diff.out` | `python obs_golden_diff.py` | on the six golden observation battles, the filter moves **148 of 991 decisions in exactly 2 columns**: an unrevealed opponent slot's second-ability id (Sand Veil 8 → 0) and its P(top ability) (0.99 → 1.0) — nothing else in the observation |

`prior_mass.out` above was taken on the PRE-filter priors (`5e8d2956`); after `compute_priors.py` regenerates them
every banned entry is gone (`src/agents/gen3_data/format_spec_priors_test.py`). Re-recorded on purpose: the golden
observation fixture, `hypothesis_dex_rows.json` (Diglett, Dugtrio, Voltorb, Electrode, Mr. Mime, Gligar), the Rust
env core's oracle-reveal `off` / `species` digests, the K9 learner golden (both arms, rebuilt buffers) and
`main.h2h`'s off/off digests.
