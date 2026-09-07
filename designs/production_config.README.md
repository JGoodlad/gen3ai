# `production_config.json` — where it came from

`production_config.json` is JSON with no comment syntax and is normally written **verbatim** by
`--sync-config` (sorted keys, never hand-edited), so its provenance cannot live in the file. It
lives here.

| | |
|---|---|
| **Run** | `models/ai_v12_02_winprob_critic/` — the WIN-PROB CRITIC era |
| **Built as** | `models/ai_v9_21_gen17_pfspoff_0820/model_config.json` (gen-17, v97) migrated to the v109 schema, with the **critic family alone** overridden |
| **Critic rows changed from gen-17** | 13 (listed below) · **non-critic rows changed: 0** |
| **Mirror updated** | 2026-09-06 |
| **`config_version` / `arch_signature`** | 109 / `gen3_critic_route_wave_v1` |
| **Previously** | gen-17's config verbatim (v97), from 2026-08-22 |

## Why it was CONSTRUCTED rather than copied from a run

The first arm of this era, `ai_v12_01_winprob_critic`, launched with a 38-flag argv that carried
only the critic block and the hyperparameters — so it recorded a config **44 shared fields** away
from gen-17, of which only 13 were the critic. The other 31 were the whole architecture surface
silently reverting to its OFF defaults: `edge_bias_families` `off`, `entity_topk_seats` 0,
`opp_belief_slots` false, `history_events` false, both intent heads off, every pointer cell off,
four beliefs off, both value routes off. That arm was killed and relaunched as `_02` with gen-17's
full surface plus the critic block.

**Mirroring `_01` would have redefined the production architecture by omission.** So this file is
built the other way round — gen-17's surface, migrated forward, with an explicit override list —
and then VERIFIED against the relaunched arm.

## The override set (the only 13 rows that differ from gen-17)

| key | gen-17 | production | why |
|---|---|---|---|
| `critic` | `shaped` | `winprob` | the mode itself |
| `use_popart` | true | **false** | IMPLIED — a bounded Bernoulli payoff has no scale to track |
| `win_prob_coef` | 0.05 | **1.0** | the flag is REFUSED under this mode; 1.0 is its `_resolve` default |
| `value_dist_mode` | `shaping` | **`none`** | REFUSED (the A2 census: ~15 sites gate on the string, not on the module) |
| `value_dist_bins` | 51 | **0** | " |
| `value_dist_vmin` | −12.0 | **0.0** | " |
| `value_dist_vmax` | 12.0 | **0.0** | " |
| `value_from_dist` | true | **false** | REFUSED |
| `value_tail_weight` | 0.3 | **0.0** | REFUSED → its concrete argparse default |
| `hand_shaping` | true | **false** | REQUIRED (`--no-hand-shaping`) |
| `terminal_indicator` | false | **true** | REQUIRED |
| `victory_value` | 30.0 | **1.0** | REQUIRED — at 1.0 the undiscounted return IS `1{win}` |
| `draw_penalty` | −35.0 | **0.0** | REQUIRED — a [0,1] critic cannot rank a timeout below a loss |

`win_prob_mode` is `shaping` in both, so the mode's third implication changes nothing.
**`gamma` is not a `model_config.json` key at all** — the mode implies 1.0 and the value in force is
recorded in `metadata.json`'s `cli_args`, so it cannot be mirrored here.

**Every other key is gen-17's, byte-for-byte** — including `value_entity_pool`,
`value_entity_pool_full` and `value_threat_inject`, which SURVIVE the swap: they inject additively
into `value_pooled`, which is exactly what the win head reads.

The v97 → v109 migration adds 39 keys that did not exist in gen-17's config (the `cf_*` family, the
`q_winprob_*` family, `policy_grad_coef`, the distillation block, `rank_tripwire`, …). All take the
migration's defaults, and all read OFF or INERT in §6's generated table.

### One flag a hand-built relaunch argv will drop

`--intent-label-bot-weight` has an argparse default of `None` and a `_resolve` default of **1.0**,
while gen-17 ran it at **0.25**. It is part of gen-17's architecture surface and is NOT critic
family, so the relaunch argv must pass `--intent-label-bot-weight 0.25` explicitly or the run will
record 1.0 and diverge from this mirror. (`_01` recorded 1.0 for exactly this reason.)

## What derives from this file

Four things, which would otherwise each carry their own idea of the architecture:

- `designs/ARCHITECTURE.md`'s generated sections (`python -m agents.model.arch_tables`)
- `designs/architecture_graph.dot` + `src/agents/model/delivery_graph_snapshot.json`
- `designs/architecture_viewer.html` (`python -m agents.model.build_arch_viewer`)
- `extractor_compiles_test`, which compiles "the production arch" from it

## Which run is production is a JUDGEMENT

Nothing derivable answers it. `arch_tables_test.test_production_config_matches_newest_run` only
enforces that the mirror agrees with the **newest run in `models/` on every shared field** — it
cannot tell an experiment arm from a generation, and, as `_01` showed, the newest run is not always
the answer. The call is made by a human (or the orchestrator on the owner's direction) and recorded
here.

## Refreshing it

Top-down, in one commit, never partially:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m agents.model.delivery_graph \
    --sync-config models/<run>/model_config.json \
    --dot designs/architecture_graph.dot \
    --json src/agents/model/delivery_graph_snapshot.json
python -m agents.model.build_arch_viewer
python -m agents.model.arch_tables            # ARCHITECTURE.md's generated sections
```

Then update this file's table, re-derive the hand-written prose the new config contradicts, and
append a `designs/CHANGELOG.md` entry naming the switch. Both `--check` gates
(`delivery_graph --check`, `build_arch_viewer --check`) must pass afterwards.

⚠️ **Prefer `--sync-config` — a CONSTRUCTED mirror like this one owes a verification step.** A
hand-built config describes a run nobody has launched until it is checked against one, and every
artifact above then describes a model that may not exist.

## Verification (2026-09-06)

`ai_v12_02_winprob_critic` did not exist when this mirror was built, so it was verified against the
**relaunch argv** instead: gen-17's recorded `original_command`, minus the launcher-owned flags,
minus every flag `--critic winprob` refuses, plus the critic block — parsed by the live
`build_parser()` and resolved through the launch path's own
`resolve_critic_mode` → `desugar_umbrella_flags` → `resolve_config`.

**Result: all 105 argv-settable fields of this mirror equal what that argv resolves to.** The other
23 keys come from the observation LAYOUT or are derived, so no argv can move them: the six embedding
dims, the five `max_*` capacities, `net_arch` / `move_net_hidden` / `role_encoder_hidden` /
`role_token_size` / `projection_dim`, `active_context_dim`, `total_dim`, `arch_signature`,
`config_version`, the config-only `attend_unrevealed_opponents`, and the three `_DERIVED` toggles
(`opp_belief_slots`, `opp_intent`, `opp_intent_grad_mode`, computed from their coefficients).

**Re-verify against the run itself** once `ai_v12_02` writes a `model_config.json` — that is the
authoritative check, and a difference there is a defect in this file, not in the run.

### The argv finding that verification produced

**gen-17's recorded command no longer launches on HEAD**, for a reason unrelated to the critic:
`--distill-team-bias 0.4` with no `--distill-teacher` is now refused
(`main.train.combination_checks`, migrated 2026-09-06). It is not a `model_config.json` key, so it
cannot move this mirror — but a relaunch argv built by copying gen-17's command will be rejected
until it is dropped or paired with a teacher. `python -m main.checkargs` reports it offline.
