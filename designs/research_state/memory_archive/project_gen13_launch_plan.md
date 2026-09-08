---
name: project_gen13_launch_plan
description: "OWNER-AUTHORIZED gen-13: enable the v81-v87 successor stack AND delete the legacy it obsoletes, gated on the gen-12 endofrun battery."
metadata: 
  node_type: memory
  type: project
  originSessionId: 1046b1d6-7bc6-429f-9028-1d78fdc77677
  modified: 2026-08-17T00:00:31.859Z
---

**OWNER PLAN (2026-08-16).** Gen-13 = enable the successor stack + retire the legacy it obsoletes,
in the SAME code change. Worktree only, `/gen3ai-ship` only. Read `designs/ARCHITECTURE.md` and
`designs/research_state/gen12_endofrun_runbook.md` before touching anything. Gen-14 is
pre-registered separately: [[project_gen14_preregistered]].

## ⚠️ v89 AMENDMENT (2026-08-17) — the §3 route audit's vf-tail arms are CONFOUNDED

`gen3_value_pooled_routes_v1` (`1fa4733`) proved the post-assembler vf tail NEVER REACHED the
dist-head critic under `--value-from-dist` (gen-11 AND gen-12): `value_entity_pool.out_proj` and
`intent_value_reduce.proj` were bit-exact ZERO after gen-12's 25M steps. Consequences for this plan:
- **§3 arms measuring vf-tail routes measured a DEAD LIMB** — `entity_pool` and `intent_reduce`
  arms carry nothing BY CONSTRUCTION, and the wave-1 "condemned" routes (seed readout /
  hidden-opp-vf / nmr concat) are also in the dead tail, so "carries nothing" no longer
  distinguishes useless from disconnected. Do NOT read §3's vf-tail conclusions at face value;
  the wave-1 deletions are now licensed as PURE HYGIENE (provably zero-gradient), not by audit.
- **The fix is a PREREQUISITE and is SHIPPED** (`1fa4733`, v89): all five routes inject into
  `value_pooled`; `value_route_gradient_test.py` guards connectivity under both critic
  parameterizations. Gen-13 launches from ≥ this commit or its critic enables are placebo.
- **Gen-13 must be FRESH-INIT at v89** — the migration REFUSES <v89 checkpoints with any route
  flag ON (their projection shapes no longer exist), so gen-12's checkpoint cannot carry weights
  forward with those flags. Production sha baseline moved 3cab191a → 694c1652.

## STEP 0 — the gate that unlocks everything (run FIRST, when gen-12 ends)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.endofrun models/ai_v9_14_gen12_h_entitypool_shaping_0816 \
    --ref models/ai_v9_13_gen11_labelonly_winprob_0815
```
Writes a verdict into `designs/research_state/measurements/`. **Cite its rules, do not re-derive
them** — they were pre-registered before gen-12 had numbers. What it licenses:
- **§1 non-inferiority** — gen-13 proceeds from gen-12's config only if gen-12 PASSES.
- **§2 `h` verdict** — `h` graduates into the standing family string, or reverts to opt-in.
- **§3 route audit** — the ONLY license for the wave-1 deletions, and only if the `entity_pool`
  arm carries what the condemned routes carried.
- **`intent_move_cell` arm (= the G3 verdict)** — whether the v84/v85 mechanic cells ride as
  capability, or only `--intent-threshold`'s p_KO-to-critic half.

## The enable set (all zero-init / bit-identical at init; compile-gated as ONE stack)

```
gen-12's config
  + --history-events                     # v81 H-B event seats (the headline arm)
  + --edge-bias-families ...,h,r         # h per §2; r = H-C reference edges
  + --item-belief                        # v83 (item CE at the default 0.05 coef)
  + --intent-threshold                   # v84 — p_thresh + p_KO to the critic
  + --intent-conditional                 # v85 — Counter/flinch/boom/Protect/MagicCoat/Pursuit
  + --damage-matrices both               # v85's arrival-pko source (stash-only under lean)
  + --op-drop-renders                    # v86 — render tail leaves the forward (bit-identical)
  + --op-believed-lean                   # v86 — de-timid fiction retired (needs --spread-belief, on)
  + --value-entity-pool-full             # v82 — the ONE critic contract (pool already on)
  + --value-clock + --value-intent       # v87 — deadline clock + alpha/beta posteriors to the critic
```
Carry gen-12's standing policy: `--win-prob-mode shaping --win-prob-coef 0.05`,
`--belief-grad-mode shaping` (label_only is CONVICTED — [[project_current_run]]).

**Flags VERIFIED present** @`f92a597`, all `Tier.CLI, Klass.STRUCTURAL` in the registry:
`value_entity_pool` 80, `history_events` 81, `value_entity_pool_full` 82, `item_belief` 83,
`intent_threshold` 84, `intent_conditional` 85, `op_drop_renders`/`op_believed_lean` 86,
`value_clock`/`value_intent` 87; `r` is in `_EDGE_FAMILIES`. **Re-verify at launch** — v83→v87 all
landed within one day.

⚠️ **Verify flags via `agents/model/flag_registry.py`, NEVER by grepping `extractor_arch.py`.**
`dc58b46` made `ARCH_ARG_KEYS` GENERATED from the registry, so a literal grep of that file returns
nothing for EVERY flag, live ones included — I raised exactly that false alarm once. Or just run
`extractor_arch_coverage_test.py`. The registry also makes the old `--t0-species-prior` failure
(parses fine, never reaches the extractor, suite stays green) structurally impossible.

**Pre-launch checks — all exist (verified @f92a597), just run them:** the one-graph compile cell
`extractor_compiles_test::test_intent_threshold_arch_compiles_to_one_graph` (covers the full rider
stack), a `--debug` smoke with the full flag set, the round-trip smoke, and — POST-launch, on the
new traces — `python -m agents.model.mechanic_usage_baseline` against
`measurements/gen12_mechanic_usage_baseline.json`. **Those baselines are the did-it-work bar for
the mechanic cells: Endure 0.0%, Sub 0.9%, Counter 5.6% @ 9.2% prob.**

## Wave-1 deletions — SAME code change, each gated on §3

Only if the audit condemns the route AND pool-full carries its content:
1. **`MultiSeedValueReadout`** + `seed_diagnostics.py` + the `value_seeds/*` TB contract — 256 vf dims.
2. **`--value-threat-inject`** + `value_threat_proj` (CLSPool).
3. **The `hidden_opp_belief` VF half** — 768 vf dims. Pool-full's belief queries are the successor;
   KEEP the pi half unless the audit condemns that too.
4. **The `non_matchup_rest` VF concat** — its unique validated content (the clock) is now
   `--value-clock`; pool-full's global row covers the rest.

**End-state check:** the critic reads `value_pooled` + the entity pool + the zero-init tail parts
and NOTHING else.

**Delete regardless of any verdict (pure dead code, all four targets confirmed still present):**
- the **pubval** subsystem (`pubval_mode` none in production, measured NULL, head never built)
- **`value_active_readout`** (config-only, frozen OFF, superseded twice)
- **OAX** (`_outgoing_attacker_matrix` + `damage_matrices_outgoing_all`, frozen OFF, no consumer)

Each deletion: bump what versioning requires, update `ARCHITECTURE.md` + `CHANGELOG.md` in the
same pass, regenerate delivery graph + viewer + arch tables + flag doc.

## Guardrails
- 🚫 **Do NOT delete the 7x159 TurnDelta frames in gen-13** — that is gen-14's change. It is the one
  NON-zero-init deletion and must not share a generation with the H-B enable
  (`design_history_entity.md` row-2 attribution discipline).
- 🚫 **Never quote a mid-run ELO or delta.** Verdict = `snapshot_ladder/ladder.json` at run END,
  cross-run at matched snapshot COUNT ([[feedback_elo_reading_rules]]).
- Launch through `main.launcher` (worktree-pinned). Forward the runtime knobs explicitly —
  `--compile-opponents [--compile-opponents-preload]`, `--grad-checkpointing`, `--async-rollout`
  per current practice. **They are never inherited on resume.**
- Never `--sync-to-main` an older live run (obs width moves every generation).

## Reporting contract
For every ENABLE: name the flag, its audit arm, and its did-it-work readout. For every DELETION:
name the verdict line that licensed it. Anything the battery leaves AMBIGUOUS is deferred and
LISTED, never decided. **Pre-register gen-13's own end-of-run rules in the runbook BEFORE launch**
(the concat-deletion precedent).

Related: [[project_gen14_preregistered]], [[project_current_run]], [[feedback_elo_reading_rules]],
[[feedback_git_workflow]], [[feedback_no_auto_ship]].
