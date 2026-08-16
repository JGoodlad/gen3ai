# design — THE CLEANUP JOURNEY: flags, delivery, diagnostics-on, and what is deliberately spared

**Status:** operational cleanup plan (2026-08-14, owner + assistant session; state refreshed
2026-08-15). **Phase 1 EXECUTED** (v78 `gen3_flag_surface_p1_v1`: the flag registry + its
five-surface validators, the zarch family + seed-pressure pair DELETED, three demotions,
`--use-bridge` defaults rust — §2's decision landed with it). **Phase 3's instrument is BUILT
and validated** (`critic_route_audit.py`: seed/threat/hidden_opp/all_off arms + the v80
`entity_pool` arm; the §3 diagnostics verdict layer landed as `main/prober/awareness.py` —
knew_by_turn/blind_loss + quantile coverage, gen-10 baselines recorded in the gen-11 runbook).
**The Phase-3 critic-route consolidation now has a built successor route**: v80
`gen3_unified_value_readout_v1` (`--value-entity-pool`, opt-in), so a condemnation and its
replacement can land in one config change. This doc records the whole journey so each later
phase starts from a decision record instead of a re-derivation. Owner decisions captured here: **diagnostic heads are
instruments, not levers — they are exempt from strength-based deletion and DEFAULT ON** (§3);
**the training-transport default moves to rust** (§2); the pair-reduction rungs are **spared**
(§7). Companion docs: `design_history_entity.md` (the E9 build), `design_op_tensors.md` (step 3),
`design_tiered_belief.md` (T3), `design_unified_belief.md` (T0).

---

## 0. The principle

A flag has three roles — **SELECT** (a launch-time choice), **RECORD** (explicitness), **GATE**
(resume/compat enforcement) — and the record/gate roles already live in `model_config.json` +
`ModelVersion`, independent of the CLI. So the CLI surface can shrink without losing
explicitness, as long as fields survive. Three explicitness tiers exist in the codebase today
and become policy:

| Tier | Settable via | Recorded/gated? | Precedent |
|---|---|---|---|
| CLI | launch command | ✔ | most current flags |
| **config-only** ("recorded, not settable") | frozen default; field written + version-checked | ✔ | the Phase-1 demotions |
| constructor-only | tests/experiments instantiate directly | ✖ until promoted | `pair_reduce`'s `reduce_how` |

New experiments start constructor-only or config-only and are **promoted** to CLI only when a
real run needs to set them at launch; settled decisions are **demoted**; killed levers are
**deleted with their evidence cited**. One further distinction the owner fixed this session:
**instruments vs levers.** A lever earns its place by strength and dies by a null; an
*instrument* (win-prob, the distributional readout) earns its place by observability — the model
having a positive scalar V the turn before a stall loss is invisible without the distribution —
and its deletion criterion is "does anything still read it", never ELO.

## 1. Phase 1 — registry, dead-module deletions, first demotions (ISSUED)

The issued prompt covers: `flag_registry.py` (one declarative row per toggle; a consistency test
cross-checking the five hand-synced surfaces — argparse, `_resolve`, `ARCH_ARG_KEYS`,
`current_model_version`, the `ModelVersion` field — plus a generated `designs/flag_registry.md`);
deletion of the **zarch family** (7 flags, three standing nulls), the **seed-pressure pair**
(`seed_vicreg`/`seed_quantile`, both measured at ceiling), and the deprecated
`--use-showdown-bridge` alias; three **demotions** establishing the config-only pattern; and a
**decision package** (not a deletion) for pubval, pending the owner's instrument test. Excluded
by name: `win_prob_*`, `value_dist_*`, `value_from_dist` (§3), and anything colliding with the
live gen-9 run's `original_command` on the `--sync-to-main` path (listed, not deleted).

## 2. Phase 1 amendment — the training transport defaults to rust (owner, 2026-08-14)

`--use-bridge` default `off` → **`rust`**: training and eval are serverless by default. `node`
stays as an explicit value (the A/B arm and the parity harness require it); `off` stays as the
websocket mode — which is also the **ladder** path, so nothing about playing on a real server is
affected (the deleted alias was a spelling of `node`, not a capability). The **offline** driver
seams (`SearchSession`, replay/reroll, the prober `--impl`) keep their `node` defaults — those
are per-seam, the node arm is what makes cross-impl parity checkable, and the search-teacher
full-cycle composition on rust remains the one ungated leg.

**Gate before the default lands:** the evidence file is strong (1.41× at `--n-envs 48`, 9 MB vs
~224 MB child, 719/719 pool construction, the seed/forfeit/choose-path parity suites) but the
lock-in reject fix (`a2ae60d`) is one day old and its failure mode was concurrency-rate-
dependent (two launches killed at ~8 min). Run `bridge_session_fuzz_test.py --impl rust` plus a
multi-hour `--n-envs 48` smoke on the current tree, and watch the first hour of the first run on
the new default. Docs updated in the same pass: root `CLAUDE.md`'s "default stays websocket"
prose, `src/utils/bridge/README.md`, the launcher leaf and its run templates.

## 3. Diagnostics DEFAULT ON (owner call-out, 2026-08-14)

The instruments the forensic practice actually uses stop being opt-in:

- **`win_prob_mode` defaults to `read_only`** (head trains on its own params; stop-grad read; no
  trunk shaping, no leak — `shaping` remains the explicit experimental choice).
- **The distributional readout defaults ON in `read_only`** with the standing bins/support;
  **`value_from_dist` stays a separate, explicit choice** — whether the critic's forward
  *consumes* the distribution is a lever; the distribution being *visible* is an instrument.
- **Per-decision trace recording** of the compact readout (q10/q50/q90, `P(return<0)`, `P(return
  ≤ draw-penalty band)`, the win-prob logit) rides `RLPlayer` → the trace summary (the
  `opp_intent` block precedent), so the readouts survive **without loading the model** — they
  escape `ArchDriftError` on archived runs.
- The engine derives the standing verdicts ONCE — `knew_by_turn` (first sustained P(loss)>0.5),
  `lead_time`, `blind_loss`, and the stall signature (scalar V positive while left-tail mass
  grows) — surfaced in `turns`/`/battle` (a per-turn probability strip), `scan`/`triage`
  columns, and a run-level aggregate: *fraction of cap losses tail-aware ≥K turns early* — the
  regression test the deadline clock never had. `calibration` extends to quantile coverage.

Mechanics: both heads are weight-adding, so the defaults are **fresh-run defaults**
(version-gated as today; existing checkpoints unaffected; a resume keeps its recorded values).
Cost is one small MLP each per forward and a handful of floats per trace row.

## 4. Phase 2 — launch by manifest

`--from-config <json> [--override k=v ...]`: the run manifest becomes the primary launch
surface, generated-from and validated-against the registry; `original_command` +
`model_config.json` remain the provenance (the manifest is *more* diffable than a flag string).
The CLI shrinks toward run-shape (`--steps/--model/--device/--n-envs/...`), runtime perf knobs
(never recorded, by design), and the live experiment surface. Consolidations that become
profiles rather than flags: the `exploiter-temp-*` sextet, the eval-worker family's defaults.

## 5. Phase 3 — the post-gen-9 audit deletions

Blocked on the current production run's end-of-run measurements (now **gen-11** — the
pre-registered battery is `designs/research_state/gen11_endofrun_runbook.md`), executed with
the concat-deletion playbook (pre-registered arms, then delete):

1. **Critic-route consolidation** — zero each of seed readout / threat-inject / dist-as-critic
   (`value_from_dist`) / intent-reduce; keep the winner, delete the losers.
   `MultiSeedValueReadout` goes if any successor carries — and the successor is now BUILT
   (v80 `--value-entity-pool`, with its own audit arm). *The dist HEAD is not in scope — §3.*
2. **The 768-dim hidden-opp concat** and **`non_matchup_rest`** (needs a re-home first — no
   pool reads the global token) and the **prev-turn action mask** zero-arm.
3. **OpTensors step 3** — drop the flat render, trim `out_gain` [660]→[138] (retrain-class).
4. **Reward end-state adoption decision** — `--all-shaping-pbrs` + `--stall-pbrs` exist and
   tested; adopting them zeroes the whole BIAS class. A config decision, not code.
5. Registry demotion sweep #2 from whatever the audits settle.

## 6. Structural debt (byte-identical, schedulable anytime)

`ForwardScratch` (the ~20 `last_*` stashes become a per-forward object); `opp_addressable`
single-sourcing on `ExtractorContext` (hoist `(hp>0)|believed`; leave the deliberately
revealed-gated kernels); OpTensors step-2's recompute-dedup half (E4/d3/s3 `pair_in` computed
once); the dual move-order hazard (role encoder's move read made permutation-invariant); doc
residue (the observation leaf's per-block body; the model leaf's accreted narrative; extend the
generated-tables pattern to surviving hand tables).

## 7. The SPARED register (record the why — the allowlist-outliving-its-fix lesson)

| Spared | Why — do not re-derive a deletion from an old list |
|---|---|
| `pairwise_*` edge kernels, all 15 families | owner "KEEP ALL" (2026-08-06); load-bearing per every audit |
| `pair_reduce` rungs, **including R2/R2L/R3** | R0 = byte-identity anchor, R1 live via threat-inject; the G1 null that condemned R2/R3 tested reducers **without a distribution worth weighting** (w-for-α, damage-only cells) — awaiting a fair retest with α; constructor-only, zero cost in production |
| `win_prob` / `value_dist` heads, `value_from_dist` flag | instruments (owner, 2026-08-14) — default ON per §3; deletion criterion is observability |
| node bridge impl + offline node driver seams | the parity arm; per-seam defaults |
| `HiddenOppBeliefPool` | live at k=6 in gen-8/9 by explicit invocation; Phase-3 audit decides |
| pubval | pending the owner's instrument test (Phase-1 decision package) |

## 8. Sequencing

| When | What |
|---|---|
| ~~now~~ DONE | Phase 1 (v78) + §2 rust-default + the §3 verdict layer (awareness/coverage) |
| **gen-11 finishes** | the audit battery (`gen11_endofrun_runbook.md`) → Phase 3 deletions + the v80 successor enable define gen-12's config alongside `h` |
| any quiet window | §6 structural debt (ForwardScratch); Phase 2 manifest launching; the Phase-1b demotions (the registry now exists) |

Standing gates for every phase: full unit suite; byte-identity on the production config for
anything claiming OFF-equivalence; the artifact chain (`delivery_graph` → viewer → `arch_tables`)
regenerated in order and `--check`-green; a bridge smoke; CHANGELOG + generated ARCHITECTURE
tables + the `designs/CLAUDE.md` state row in the same pass.
