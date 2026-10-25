# Training — counterfactual (cf) label grounding: DELETED

Nothing in training produces or consumes counterfactual win-prob labels any more. This doc is a tombstone with the
current truth; the full former text (label cost model, producer throughput, forfeit class, the retention race, the
prefix-sharing speedup) is in
[`designs/research_state/claude_md_archive/training_cf_grounding_deleted_history.md`](../research_state/claude_md_archive/training_cf_grounding_deleted_history.md).

## What exists now

- **The prober's counterfactual views** play out alternatives on the Rust core (`src/utils/rust_env/counterfactual.py`);
  see [`../prober/counterfactual_probes.md`](../prober/counterfactual_probes.md).
- **`MIN_LABELABLE_TURN`** (the sampler's lower turn bound) lives in `src/agents/training/fork_arm.py`, which the
  fork arm uses ([`forks.md`](forks.md) §14). The Rust fork arm needs no record ring.
- **`src/agents/training/stats.py`** kept the shared statistics; `harvest_schema.py` and `winprob_finetune.py` stayed
  (the stall-tail pipeline).
- **Retired checkpoints:** one that recorded a cf coefficient live refuses resume / fork, and one that recorded a
  structural head (evidential Beta, twin win-prob, shadow critic, per-action Q) ON is refused on every load
  (`src/agents/model/model_version/retired_levers.py`).
- **Any future playout label** (X4 / X6) would run on the Rust search driver, not a record ring.

## What was deleted, and when

| when | what | record |
|---|---|---|
| 2026-10-02, deletion pass L4 (owner D1 "delete, port none"; config v134) | the TRAINING half: the `cf_records` ring tap, the label buffer, every consumer term and structural head, the supply guard, team-PFSP, the exploiter ladder | `designs/ops/deletion_pass_manifest.md` §2; every flag in `designs/deleted_flags.md` |
| 2026-10-08, P6 slice 6d-1 (owner: "LGTM for deleting both cf label and search, they need poke-env") | the OFFLINE stack: `cf_producer*`, `cf_audit*`, `cf_q_labels`, `cf_mc_return`, `utils/bridge/counterfactual.py` | `designs/ops/deletion_pass_manifest.md` §8.4; `designs/deleted_flags.md` (the two `python -m agents.training.cf_*` CLI rows) |
| 2026-10-08, P6 slices 5 and 6d-2 | the poke-env prefix-sharing materializer and the Python offline replay (`obs_materializer`) | same manifest |

## Why (the evidence that closed it)

- Only 3 runs ever ingested a cf label; `ai_v12_12_ladder_cflabels` trained 10M steps at a live win-prob coefficient of
  0.5 and received ZERO labels (ledger 2026-09-30, *CfLabelBuffer INODE REUSE*): the producer was a separate program
  nobody started.
- No cf lever moved leaf quality (the cf-labels arm and the leaf battery, 2026-09-12). The win-prob head's defect is
  RESOLUTION, not an optimism offset (ledger 2026-08-22, the G0 bias map).
- Lesson that survives: a supply a coefficient depends on is a declared startup resource, and a checkpoint cadence
  counted in vec-env calls (not env steps) silently starves a step-lagged consumer. The trainer's cadence is now
  `--checkpoint-every-steps` (`designs/ops/training_runbook.md`).
