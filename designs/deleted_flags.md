# Deleted flags — the history list the freshness gate reads

**What this file is.** A `CLAUDE.md` sometimes names a `--flag` on purpose *because it no longer
exists* — "`--use-showdown-bridge` is DELETED", "the June `--compile-damage-op` integration was
inference-only". That is good documentation: it stops the next reader re-deriving a plan from a
lever that is gone. But it is indistinguishable, to a scanner, from a flag that was quietly renamed
and never fixed in the prose.

So `src/claude_md_freshness_gate_test.py` fails on any `--flag` in a `CLAUDE.md` that resolves in
no parser in this tree — **unless the flag is listed here with a citation.** This file is the one
escape hatch, and it is a ratchet, not a bin:

* Every row carries a **citation** — the version (`v88`), the signature (`gen3_dead_flag_purge_v1`),
  a date, or a commit sha. The gate fails an uncited row: an unexplained entry is the c-family
  failure the root `CLAUDE.md` records, where an allowlist entry outlived its own fix and then
  misled every reader after it.
* A row must **leave** if its flag comes back. The gate fails a row whose flag is in the live CLI
  surface again — while it is listed, a second deletion of the same flag would go unnoticed.
* A row is **not a fix**. Listing a flag here says the prose is deliberately historical. If the
  prose actually meant to name a live flag, correct the prose instead.

Three categories, because "does not resolve" has three different causes and they call for different
reactions from a reader.

---

## 1. DELETED — the flag and the code behind it are gone

| flag | citation | note |
|---|---|---|
| `--use-showdown-bridge` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L2905) | the deprecated `--use-bridge=node` alias; deleted when `rust` became the default, precisely so the legacy spelling could not silently select the slower impl |
| `--pubval-mode` | v88 `gen3_dead_flag_purge_v1` (CHANGELOG L4078) | the whole public-info value subsystem went with it — `agents.training.pubval`, `PubValHead`, `_pubval_loss`, `data/gen3_pubval.json`. Measured NULL, never ON in production |
| `--pubval-coef` | v88 `gen3_dead_flag_purge_v1` (CHANGELOG L4078) | as above |
| `--damage-matrices-outgoing-all` | v88 `gen3_dead_flag_purge_v1` (CHANGELOG L4073); void-ed again explicitly at v95 (L5247) | a v78 config_only demotion frozen OFF, then deleted outright. The `_outgoing_attacker_matrix` KERNEL survives as `d2`'s engine |
| `--value-clock` | v96 `gen3_critic_route_wave_v1` (CHANGELOG L5511) | one of the seven audited-dead critic routes; dV 0.2169 at 2× against a 0.39 bar |
| `--value-intent` | v96 `gen3_critic_route_wave_v1` (CHANGELOG L5457) | dV 0.156. Its **re-entry condition survives its deletion** — any α/β-critic proposal passes the C4 offline gate first (ledger C6) |
| `--intent-value-reduce` | v96 `gen3_critic_route_wave_v1` (CHANGELOG L5516) | dV 0.3176 at 2× |
| `--opp-belief-latent-coef` | v75 (CHANGELOG L3940) | the SimSiam LATENT belief deleted with `opp_belief_latent`; ~13% of the train step |
| `--seed-quantile-coef` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3980) | the SEED-PRESSURE pair — both cap at ~1-D of k=4, from opposite directions |
| `--seed-vicreg-coef` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3980, L3990) | as above; a training-only coefficient, so any recorded value pops silently |
| `--value-seed-vicreg-coef` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3980) | the v62 spelling of the same coefficient |
| `--film-grad-accum-steps` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L2892) | went with the zarch family's group accumulator and the `film/*` + `zarch/*` TB families |
| `--zarch-film` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3987) | the zarch conditioning family; the LUT arm moved the N=20 ceiling +0.024, CI [-0.016,+0.064] |
| `--zarch-mode` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3987) | as above |
| `--spread-belief-nature-marginalize` | v66; guarded by `gen3_dead_kwarg_tripwire_v1`, 2026-08-17 (CHANGELOG L4682) | the op used to marginalise P(KO) over the nature posterior |
| `--compile-damage-op` | the June 2026 integration, superseded 2026-08-14 by the `--compile-opponents` / `--compile-trainer` split | the inference-only damage-op compile; the surviving split is `--compile-opponents` / `--compile-trainer`. Named in `src/agents/training/CLAUDE.md` only as the precedent for why a CPU backward does not lower |

## 2. DEMOTED — the config field survives, the CLI flag does not

v78 `gen3_flag_surface_p1_v1` introduced the `cli` / `config_only` / `constructor_only` TIER axis:
a settled toggle can lose its CLI entry and keep the recorded field, the version gate and the
constructor kwarg. A document that still spells one of these `--like-a-flag` is describing a
*capability*, not something you can type.

| flag | citation | note |
|---|---|---|
| `--attend-unrevealed-opponents` | v78 `gen3_flag_surface_p1_v1` (CHANGELOG L3996) | demoted to `config_only`, **frozen ON**. Live as `ModelFlag("attend_unrevealed_opponents", True, Tier.CONFIG_ONLY, …)` in `src/agents/model/flag_registry.py` |
| `--damage-refine-rounds` | v70/v71 (the refine loop deleted; CHANGELOG L3915) | the field survives in the registry's requires-graph; there is no CLI entry. `src/main/prober/CLAUDE.md` names it as an axis the production config does not run — true, but it cannot be turned on from a command line either |
| `--threat-unrevealed-outgoing` | v78 (CHANGELOG L3915, L3729) | same shape; `requires threat_refine_outgoing` in the registry, no CLI entry |

## 3. PROPOSED — named as a future shape, never built

| flag | citation | note |
|---|---|---|
| `--trainee-team-prob` | named as future-only since 2026-08-25 in `src/agents/training/CLAUDE.md` § MatchupSpec, which says so verbatim: "the future `--trainee-team-prob` shape — supported, no CLI yet" | the `pin_biased` draw exists; the flag does not |

---

## Deleted PATHS

The path half of the same gate. A `CLAUDE.md` that names a file which no longer exists fails —
unless the file is listed here, which says the sentence is deliberately telling you what used to be
there.

| path | citation | note |
|---|---|---|
| `data/gen3_pubval.json` | v88 `gen3_dead_flag_purge_v1` (CHANGELOG L4078) | the public-info value calibration artifact, deleted with the subsystem. Named as history in `designs/CLAUDE.md` and `src/agents/training/CLAUDE.md` |
