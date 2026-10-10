# `smogon_stats_downloader` — the priors' acquisition contract

Lifted out of `tools/CLAUDE.md` on 2026-10-10 (always-current). The CONSUMER side — what each prior
file holds and the load-time checks the facade runs — is `src/agents/gen3_data/CLAUDE.md` ("The files
in `data/pokemon/`"); this doc is the PRODUCER side: `smogon_stats_downloader/sync.py` (12 monthly
chaos JSONs → `data/pokemon/gen3_smogon_stats.json`) and `compute_priors.py` (→ the six
`gen3_{ability,hidden_power,move,item,spread,teammate}_priors.json`).

## The denominator W (`gen3_smogon_prior_denominator_v1`, F-X5-41)

A Smogon chaos record mixes ONE unweighted field with rating-WEIGHTED ones — never divide across
them. `Raw count` is the UNWEIGHTED set count; `Abilities` / `Items` / `Spreads` / `Happiness` /
`Moves` add the rating weight once per set (`Moves` once per SLOT, an empty one under `""`), so the
first four share one total W and `Σ Moves = 4 W` (pkmn/stats `stats/src/stats.ts` `updateStats`;
Smogon's own report divides by that W, `reports.ts` L271). The move prior is `Moves[m] / W`
(`compute_priors.weighted_count`); until 2026-10-04 it was `/ Raw count`, deflated ×0.10–0.94 by
species.

- `compute_priors.check_priors` THROWS before any write on an output that breaks its invariant (a
  distribution summing to 1; a move prior summing with the empty-slot mass to exactly 4).
- `weighted_count` THROWS on a record whose weighted fields disagree.
- The facade re-checks at load (`gen3_data.priors`).
- The SPECIES-USAGE marginal is the same W (`gen3_smogon_species_usage_weighted_v1`, F-X5-47): the
  facade's `priors.species_usage()` reads each species' `Σ Abilities` (it was `Raw count`, ×0.19–×1.75
  off by species) and THROWS on a table that is not W. No tool writes it — it is derived at call time
  from `gen3_smogon_stats.json`.

## The 12-month merge keeps W consistent (`sync.py` `_merge_species`)

`Raw count` and every weighted field (`Moves`, `Abilities`, `Items`, `Spreads`, `Happiness`,
`Teammates`) are summed element-wise across months, so the merged W is `Σ_months W_m` and every
identity above holds on the aggregated file (0 of 216 records off, 1.2e-14). Only `usage`,
`Viability Ceiling` and `Checks and Counters` are latest-month (not summable) — never read `usage`
as a count or a 12-month share.

## The FORMAT SPEC filters every prior at acquisition (`gen3_format_spec_priors_v1`)

`designs/endstate/design_format_spec.md` §5.1. `compute_priors.py` imports
`agents.gen3_data.format_spec` (run it with `PYTHONPATH=src`) and drops every banned ability (before
the ability tiers, so the 1 % floor never lands on one), item, move (the legal moves scaled by
`(4 − e) / (4 − e − b)`, a value above 1 THROWS) and teammate species; `check_format_legal` THROWS
before any write if banned mass survives. A species left with no legal ability must be exactly the
spec's `ABILITY_LOCKED_SPECIES`, or the tool THROWS.
