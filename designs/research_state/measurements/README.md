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

## 🚨 Every `kl_mean` / `flip_rate` here predates the mask fix (2026-08-22)

`gen3_audit_mask_recovery_v1`: until 2026-08-22 the ablation probes recovered the legal-action
mask as `logits > -1e8`, but the eval recorder stores **PRE-mask** logits — `inference/player.py`
keeps the `-1e9` offset in a local that never reaches disk. **Measured: 0 of 800+ archived
`states.npz` across every run back to ai_v5 carries a single logit below -1e8**, so the recovery
returned ALL-LEGAL on every row of every audit ever run, and `edge_ablation_audit`'s "zero legal
actions" guard passed vacuously by construction. On a 400-file sample **38.4% of the action space
was wrongly counted legal** (min 18%, max 68%).

**24 files here carry a `kl_mean`, and all of them are pre-fix.** The mask is both the KL's
summation domain and the policy's renormalization domain, so the affected axes move materially.
Re-measured on gen-17 (`ai_v9_21_gen17_pfspoff_0820` final, n=512, identical states both arms):

| axis | before | after | |
|---|---|---|---|
| `all` `kl_mean` | 0.58549 | 0.35647 | −39% |
| `all` `flip_rate` | 34.57% | 21.29% | −38% |
| `d2` `kl_mean` | 0.32574 | 0.19764 | −39% |
| `h` `kl_mean` | 0.00606 | 0.00279 | −54% (largest) |
| `t` `kl_mean` | 0.00500 | 0.00623 | **+25%** |
| `concat_cells` `flip_rate` | 28.71% | 31.05% | **+8%** |
| every `dv_mean` | — | — | **UNCHANGED** |

**It is not a uniform rescale, so RANKINGS change** — `t` and `h` swap places on gen-17. Treat
pre-fix `kl_*` and `flip_rate` as ORDINAL-ONLY within a single file and never compare one to a
post-fix number. **`dv_mean` is unaffected** (the critic delta never touches the mask), so every
|ΔV| reading here — including the gen-13.5 §4 frame-deletion dV comparison — stands as measured.

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
| `stall_tail_head_reading_2026-08-29.{json,md}` | pooled 11→17 + rev-1/R2/R3 vs v6/v7/v8 | all retained | 4869 battles (135 clock-era cap endings) | 2026-08-29 | **PROBE O** (ledger `627ab58`) — what the win-prob head reads over the last 5 decisions before a stall/cap ending. Model-free (recorded `win_probs`/`values`). **13/14 blindness GONE** (81.2%→22.2% positive V within the `ai_v9` lineage, n=612→135, break at the clock boundary); **residual over-confidence REAL** (34.8% of cap tails end φ≥0.5 = 4.3× ordinary losses; 17.0% still in C3's 0.70–0.98 band). Closes `gen14_endofrun_runbook` §(c) via the runbook's own pooled-within-run-difference route. ⚠️ its registered `detect` composite is saturated by its "declining" half and reads NULL — the `φ_T ≤ 0.5` half is the informative one. Four producer scripts committed beside it |
| `gen14_endofrun_battery.json` | 14 | 25M | 12391 | 2026-08-18 | **The §1–§5 end-of-run battery**: §1 INFERIOR (Δ −38.30, paired-refit CI [−55.0,−21.6], post 4× tie-break), §2 frames EXONERATED, §3 two DELETEs, §4 threat KEEP, §5 `r` ALIVE |
| `ai_v9_16_gen14_framedel_v91_0817_endofrun.{json,md}` | 14 | 25M | 12391 | 2026-08-18 | `main.endofrun`'s RAW output. ⚠️ its §1 is the SPARSE orientation fit (INCONCLUSIVE), NOT the verdict — kept so §0's artifact disagreement stays checkable |
| `gen14_route_audit_12391.json` | 14 | 25M | 12391 | 2026-08-18 | Raw critic-route arms behind §2/§3/§4 |
| `gen14_family_audit_12391.json` | 14 | 25M | 12391 | 2026-08-18 | Raw edge-family ablation behind §5 |
| `gen14_pool_refresh_compile_cost.json` | 14 | 4.0M | 2 events | 2026-08-17 | Cost of a self-play pool promotion under `--compile-opponents`: **+77 s recurring (~2.7%)**; the +1095 s first event is a one-time self-play activation, not a promotion |
| `archive_grooming_dryrun_2026-09-06.{json,md}` | — | — | 218 runs | 2026-09-06 | **`--policy standing`** (the DEFAULT): one rule for every CLOSED run, `checkpoints/` + `eval_traces/` only — would free **18.8 GB of 257.1**. **DRY RUN; nothing deleted.** |
| `archive_grooming_tiered_2026-09-06.{json,md}` | — | — | 218 runs | 2026-09-06 | **`--policy tiered`**: graded by ERA and by who still reaches for the run (tier 0 LIVE / 1 REFERENCED / 2 v9+ / 3 v8 / 4 pre-v8 keep-list) **plus the first `snapshots/` rule** — would free **87.9 GB of 257.4 (34.1%)**, incl. 377 legacy ROOT-level checkpoints the standing policy structurally cannot see. **DRY RUN; nothing deleted.** |

⚠️ **The two grooming reports above are BOOKKEEPING, not references.** They name every run in
`models/` by construction, so the tool excludes its own reports (and `fh_lineage*` /
`sidecar_audit*` / `folding_history_*`) when deciding whether a run is still referenced — reading
them as evidence put all 118 non-tier-0 runs in tier 1 on the first tiered pass and graded nothing.
Producer: `archive_grooming_dryrun.py` + `archive_grooming_tiers.py`, both committed here with
81 tests across `archive_grooming_dryrun_test.py` (34) and `archive_grooming_tiered_test.py` (47).
Policy of record: [`../models_retention_policy.md`](../models_retention_policy.md).

⚠️ **The index above is PARTIAL** — it was built when this directory held ~16 records and the
directory now holds ~60. An unlisted file is not an unofficial one; read its `provenance` block (and
its `.md` companion, where one exists) directly. The one window that IS indexed completely is below.

### The 2026-08-26 → 08-30 programme week (complete)

Each row: the record, the ledger entry that scored it, and the finding in one line. Every one of
these had its predictions **registered before the data existed**, and the verdict column says which
registered reading was selected.

| Record | Ledger | Finding |
|---|---|---|
| `plasticity_forensics_v8_vs_gen_2026-08-28.{json,md}` | `181c1d5` | **The parent-RIGIDITY account is NOT SUPPORTED.** P1 & P3 refuted-OPPOSITE: v8's 277M parent Lyle **1.154** (no capacity loss) vs plastic rev-1 **0.948**; v8's teacher deltas TRUNK-heavy 0.47 vs 0.28. Two reframes replace it (teacher differentiation; the drift anchor) |
| `differentiation_vs_breadth_2026-08-28.{json,md}` | `673a694` | **Probe A: FLAT.** Slope **+0.0003 ± 0.0013 (z=0.23)** over a recipe-controlled 2/3/4/9-team ladder ⇒ breadth does not drive differentiation. What moves is fork LENGTH (+0.039 ± 0.016). Also corrects the record: v8's teachers pinned **3/10/10 teams, not 23 each** |
| `drift_anchor_decomposition_2026-08-28.{json,md}` | `d4d551d` | **Probe B: content is REAL** — the ZapDug natural experiment gives DiD **≥+4.0pp** teacher-specific content (a lower bound). And **ALIGNMENT ≠ BENEFIT**: R2-KL absorbed the most teacher shift and finished 4.8pp behind R2-ACTION. Uncovered the `--distill-team-bias`-at-coef-0 confound |
| `dark_knowledge_decomposition_2026-08-28.{json,md}` | `92ad277` | **Probe D: mechanism supported, premise refuted.** Divergence is mode-dominated, not tail-dominated; the right meter is copied tail SHAPE (cosine 1.000/0.916/0.308, monotone with benefit +2.6/+3.4/+7.4pp, **sign-reversed against dose**). **The tail is certifiably NOISE** (inter-fork 0.327 vs no-fork control 0.306) |
| `era_diff_v8_vs_gen_2026-08-28.{json,md}` | `92ad277` | **Probe C: the 2026-08-25 "CORRECTION 3" is FALSE.** `exploiter_bot_fraction` is INERT without `--exploiter-keep-bots`; v8 ran a different opponent regime with run-dir PROOF. Fourth specimen of recorded≠effective, and **this one invalidated a kill** |
| `v8_tail_agreement_2026-08-28.{json,md}` | `2122878` | **Probe E: v8's tails are NOT special.** 0.349 vs gen's 0.344 like-for-like, CI [−0.021, +0.033]; two thirds of the apparent lead was the MASK REGIME and the sign FLIPS under a state-restricted read. The full-KL re-entry path loses its main empirical pillar |
| `per_team_gradient_geometry_2026-08-28.{json,md}` (+ dir) | `c6d420e` | **Probe F: PCGrad has a substrate at fold-INIT (12/36 negative pairs, 0.324 of norm) and NONE at fold-END (0/36).** The conflict crosses the balance↔offense line and MIGRATES out of the trunk. **PC1 carries 0.466–0.519 of energy vs a 0.111 null — domination-by-average is the structural fact** |
| `distillability_index_gen_2026-08-28.{json,md}` | `18c2f86` | Instrument ADMITTED (41 cells / 6.1 CPU-h). **Absorption RISES with training age in all six arms**, including an ancestry-free lineage; collateral's SIGN is set by step size; the zero-content control shows ~79% of mature collateral is **ADAM OVERSHOOT, not content rejection**; a fold does NOT consume distillability |
| `racing_root_selection_2026-08-28.{json,md}` (+ `_zrule`) | `d2a0212` | **Probe I: the middle branch** (1.47× deadline / 1.87–2.40× spend). Separation is **U-SHAPED with an empty middle** (52.2% never separate). 🚨 **And the standing caveat: the battery's own 1 s cell agrees with its large-budget argmax on only 86.1%** — ~1 in 7 historical "searched" decisions was allocator noise |
| `search_triage_policy_2026-08-28.{json,md}` | `79e8b11` | **Probe H: "forced decisions" REFUTED** — search flips **0.694** and no cheap confidence feature separates flips. Flip **COST** is separable: 83% of the dividend in 22.7% of decisions, found only by \|P(win)−0.5\|. Governing context: **search currently NET-LOSES** |
| `critic_bias_split_2026-08-28.{json,md}` (+ `_valuehead`) | `5f98d26` | **Probe G: offset 0.728 / differential 0.272** ⇒ **pair first**; contrastive training SIZED ≤5.7pp, not convicted. **The one-ply WIN-PROB head beats the played action (+0.0219); the scalar V head does not clear zero** — any search must read the win-prob head |
| `defensive_search_first_cell_2026-08-29.{json,md}` | `4cf81fd` | **Search STOPPED LOSING**: mirror **0.4937** vs honest_1s 0.2929, beating playoff_10s at 1/20 the budget on identical seeds. The overrule-rate prediction refuted instructively — budget-limited AT THE FLOOR while banking 77% of its own budget |
| `defensive_search_iter2_2026-08-29.{json,md}` | `35dbc3c` | 🔴 **The mechanism moved EXACTLY to spec and the dividend is ZERO** (0.5003 [0.4803, 0.5203]); overrules 13×. The **winner's curse of a biased instrument** named: statistical separation of a biased reader is not correctness |
| `transfer_coefficient_cell_2026-08-29.{json,md}` (+ `_prereg`) | `deb0bc9` | 🔴 **τ = 0.17 [−0.34, +0.68], EXCLUDES 1.0** over 8,100 games / 4,050 paired units. Checkpoint and the bot half of population removed with the dividend absent ⇒ **COMPOUNDING convicted**. Its falsifier also found the **global-`random` staller defect** |
| `whiff_head_knowledge_2026-08-29.{json,md}` (+ `_decisions.json.gz`) | `bda8382` | 🏆 **Probe L: the head knows 0.964 [0.948, 0.978] of immune whiffs at decision time**, margin 0.049 vs a 0.00062 floor, whiff-SPECIFIC; **1.000 on the FIRST click of a loop**; the policy samples its preference at **p = 0.002**. The "shaping" lever is **structurally refuted** (trunk share 1.02% at cosine −0.133) |
| `bias_tax_head_alignment_2026-08-29.{json,md}` | `1d5a866` | 🚨 **Probe M: the one live BIAS term is a de-facto SWITCH TAX the head refutes.** Alignment 45.7% vs a 44.7% matched control; **48.5% over-tax; 73% of voluntary switches charged vs 6.7% of moves; 36% on zero-agency replacements.** Implied −0.101/decision against switching vs the head's +0.0042 preference FOR it |
| `no_progress_tax_review_2026-08-29.md` | `cfbc9bf` | 📜 **Probe N: COMPOSITION DRIFT** — the switch toll was designed inside a reward where a switch also collected **+0.35**; `928a00b` deleted every counterweight and kept the toll (net **+0.35 → −0.15**). A new defect genre, plus the SITOUT off-by-one confirmed exactly and a third defect found |
| `stall_tail_head_reading_2026-08-29.{json,md}` | `32c39df` | 🟡 **Probe O: the clock fix HELD** (81.2% → 22.2% over seven generations, break exactly at the clock boundary) — **and 34.8% of cap tails still end φ ≥ 0.5 on games that lose by construction** (4.3× ordinary losses). Exposure is CONDITIONAL ⇒ the clean world's no-bias launch stands |

| `global_random_sweep_2026-08-30.md` | *(dispatch S2)* | 🔧 **The staller's global-`random` coin was a GENRE, and the staller was its SMALLEST member.** Census of every process-wide RNG draw in the tree: **4 class-(a) cross-arm couplings** (every player's `choose_random_*` + `DEFAULT_CHOICE_CHANCE`; the TEAM DRAW; `RLPlayer`'s **torch**-global action sample; the self-play pool draw), 6 class-(b), ~45 benign. All four fixed OPT-IN, defaults byte-identical. Measured stake: under the **same fixed sim seed**, unseeded arms played *different games* (84/145 turns vs 212/233, different winners); seeded, identical battle for battle. ⚠️ **The falsifier that found the staller could not have caught the biggest one** — it conditions on zero-overrule units and `random`'s overrule rate is 1.00, so that bot contributed **none**, while the "7 deterministic bots · 0.0000" row was read as covering it |

Two records from the same window that are **method artifacts rather than verdicts**:
`teacher_sharpness_probe.{json,md}` and `playoff_formal_read.{json,md}`. And
`post_paydown_baselines_2026-08-23.{json,md}` holds the per-decision CPU baseline that
`CLAUDE.md`'s trainer-turn table cites.

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
