# measurements/ — the audit outputs behind the numbers we quote

Every measured claim in `designs/ARCHITECTURE.md` and `designs/learning/*.md` should point at a file
here. The reason is narrow and practical: **a number quoted in prose has no expiry date.** The
2026-07-25 op-block ablation was cited as current for two weeks after the config it was measured on
stopped existing. If the number lives in a file that carries its own checkpoint, step, state count
and date, a stale citation is detectable by comparing dates instead of by re-deriving the result.

## How to read a file here

Each `.json` is the probe's **verbatim output** with one added top-level key:

```json
"provenance": { "generation", "run", "checkpoint", "step", "n_states", "date", "producer", "note" }
```

`provenance` was added when the file was co-located into version control (2026-08-08); nothing else
was touched. Files were copied out of gitignored `tmp/` directories and per-run `models/` folders —
neither of which is in the repo, which is exactly why the numbers kept outliving their context.

**A measurement is scoped to the model AND the config that produced it.** A generation is a fresh
lineage behind an `ARCH_SIGNATURE` wall; gen-1 (6 edge families) and gen-3 (15) are not the same
model, and a mid-training read is not an end-of-run read. Cross-generation comparisons are only
sound when the metric is a within-run ratio.

## Index

| File | Gen | Step | n | Date | What |
|---|---|---|---|---|---|
| `gen3_edge_family_audit_9p6M.json` | 3 | 9.6M | 6000 | 2026-08-07 | **Current-config** per-edge-family ablation, all 15 families + `all` / `concat` / `concat_cells` |
| `gen3_op_block_dependence_6k.json` | 3 | 9.6M | 6000 | 2026-08-07 | **Current-config** per-op-sub-block dependence, with a THREAT-state subset and a shuffle control |
| `gen3_op_block_dependence_1p5k.json` | 3 | 9.6M | 1500 | 2026-08-07 | Smaller-n first pass of the same probe (sample-size check) |
| `gen3_oracle_belief_voi.json` | 3 | 9.6M | 2000 | 2026-08-07 | Value-of-information of a partial look-ahead move-belief oracle |
| `gen1_edge_family_audit_9p6M.json` | 1 | 9.6M | 2048 | 2026-08-04 | 6 families, mid-training |
| `gen1_edge_family_audit_40M.json` | 1 | 40M | 4000 | 2026-08-05 | 6 families, end of run |
| `gen1_edge_family_audit_40M_with_concat.json` | 1 | 40M | 4000 | 2026-08-05 | …plus the op head-concat arm |
| `gen2_edge_family_audit_40M.json` | 2 | 40M | 4000 | 2026-08-06 | 11 families, end of run |
| `gen25_edge_family_audit_25M.json` | 2.5 | 25M | 4000 | 2026-08-07 | 15 families, mid-run |
| `gen13_stall_coverage.json` | 13 | 25M | 1349 | 2026-08-17 | **§7 successor** — stall-trajectory share of training decisions vs matched eval losses, + recorded critic sign at the final decision of a timeout loss (model-free; producer script committed beside it) |

## The two headline reads

### 1. Per-op-sub-block dependence — the current-config table

`gen3_op_block_dependence_6k.json`. Method: zero each sub-block as a contiguous slice of the op's
output **at the `ProjectionAssembler` concat only** — the edges, the `prefuse_proj` token injection
and the pointer cells all stay live — then measure masked KL against the policy's own distribution.
So this answers *"what does the HEAD still lean on"*, not *"what does the model use"*.

| Sub-block | Width | KL (all states) | Argmax flips | KL (shuffle control) |
|---|---|---|---|---|
| `FULL_CONCAT` (the ceiling) | 660 | 0.2444 | 23.6% | 0.1818 |
| `in_matrix` (incoming per-move matrix) | 522 | **0.2534** | **24.2%** | 0.1357 |
| `out_active` (outgoing single-active) | 45 | 0.0176 | 5.4% | **0.0254** |
| `INCOMING_all` (per-mon + CB) | 85 | 0.0174 | 6.2% | 0.0158 |
| `in_permon` | 72 | 0.0160 | 6.0% | 0.0130 |
| `in_cb` (Choice-Band tail) | 13 | 0.0013 | 1.4% | 0.0014 |
| `out_status` (status landing) | 8 | 0.0006 | 0.8% | 0.0010 |

**Read the shuffle column before drawing a conclusion.** Shuffling a block across the batch keeps
its marginal statistics and destroys its state-specific content. For `out_active`, `in_cb` and
`out_status` the shuffle arm meets or exceeds the zero arm — those blocks show no dependence this
probe can separate from noise at this n. The clean signal is `in_matrix`: zeroing it costs
essentially the whole concat ceiling, and its shuffle arm is roughly half its zero arm.

**This is a mid-training read (9.6M of 40M).** Edge dependence grew ~3× with training in earlier
generations, so treat the levels as provisional and the ordering as the finding. It has not been
re-run at end of run.

⚠️ **Do not quote the 2026-07-25 P1 table as current.** That measurement (`OUTGOING 65.7%`,
`incoming per-mon 12.7%`, ceiling 0.9385) predates the pointer-native action head, ranked two
sub-blocks that no longer exist in the production config, and is not reproduced by the table above
— its headline (outgoing dominates) is the *opposite* of gen-3's (the incoming matrix dominates).
Its raw output was not archived here because it is not in version control anywhere; only the
derived table in `designs/learning/shortcut_learning_and_feature_delivery.md` survives.

### 2. The concat did not starve the edges — three replications

Zeroing the **whole** edge system (`all`) versus zeroing the op's **head concat**, same run, same
states:

| Run | Step | families | edges off (flips) | concat off (flips) | concat+cells off (flips) |
|---|---|---|---|---|---|
| gen-1 | 40M | 6 | 26.9% | 35.5% | 40.4% |
| gen-2 | 40M | 11 | 31.5% | 33.1% | 42.6% |
| gen-2.5 | 25M | 15 | 14.3% | 31.3% | 41.3% |
| **gen-3** | **9.6M** | **15** | **13.9%** | **23.6%** | **37.8%** |

The concat arm flips more actions than turning the entire edge system off, in every run measured.
Paths compete only when they are substitutes; a bias carries a softmax-normalised **ratio** and the
concat carries an **absolute**, so these two do different jobs. The gen-3 row is mid-training and
its edge column should be expected to rise.

Top families by argmax flips are stable across generations: `d2` then `d1` then `v` — the OUTGOING
damage families — with every incoming and consequence family an order of magnitude below.

## Regenerating

The edge-family audits are reproducible from the repo:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m agents.model.edge_ablation_audit <checkpoint.zip> --states '<eval_traces glob>'
```

**The two gen-3 probes are NOT reproducible from the repo.** `incoming_conditional_probe.py` and
`oracle_belief_voi.py` were written as one-off scripts in a gitignored worktree `tmp/`
(`.claude/worktrees/bridge-cse_01GY32k9rtxziMpwhvp56wVL/tmp/`) and were never committed. Their
method is documented in this file and in each JSON's `provenance.note`, but the exact code is
recoverable only from that worktree — an honest gap, and the reason this directory exists.
