# Op GIGO 2026-10-07: how often the damage operator's ability / status-rule defect fired on real data

**What was measured.** The production damage operator read an opponent's ability as REVEALED whenever its id was
non-zero. But an unrevealed opponent's `id1` carries the species' top-1 Smogon-prior ability with `known` = 0, so
the op treated that guess as certain. The fix (gen3_op_ability_known_v1) reads the `known` flag and mixes in the
species prior. gen3_op_status_rules_v1 adds the Safeguard, Sleep / Freeze Clause, our-Substitute and Yawn rules to
status landing. This directory measures, on real games, how often each change moves anything.

**Data.** The committed M5 Lane S bank (`designs/research_state/measurements/m5_laneS/bank_v1`): the WHOLE bank,
**580 battles, 42,465 answered decisions (both viewers)**. Each battle is replayed on the Rust core
(`core_events --trackers --obs`) with no poke-env. It is deterministic: two full fresh-init runs gave identical
numbers, and the base-vs-base noise control differs on 0 rows in both runs.

**Two extractors.**
- **FRESH INIT**: built from `production_config.json` kwargs (move-resolution `off`), `torch.manual_seed(0)`.
- **TRAINED**: `/home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1007/final_model.zip`.
  - It is the last snapshot of the newest X5 BLOB-arm run, at 15,047,535 steps (`resolve_model_ref` on the run dir).
  - It was strict-loaded on CPU through `load_checkpoint_strict` (read-only); its kwargs are its own (config v138).
  - It trained at pin `706fa536`, i.e. under the OLD op.

🚨 **Read the model-output counts from the TRAINED column.** The pair-outcome / conditional-threat / intent cells
that carry most of the op's reach deliver through ZERO-INIT projections. At fresh init those projections read
exactly 0, so the channels they carry look unreachable when they are not. (a), (c), the op-block counts and the
direct p_land / damage-cell reads do not depend on the weights, and are identical in both columns.

**The OLD read** is the same observation with THEIR six's ability `known` column set to `ability1_ids > 0`, routed
through the op's single opponent-ability reader `opp_ability_view`. On the committed test's 48 battles this matches
an actual revert of the two helpers: the same 9 contradicted claims.

**Commands** (worktree root, `PYTHONPATH=<worktree>/src`; about 30 s replay + about 300 s op reads each).
`measure.out` holds the two JSON outputs in this order: fresh, then trained (the trained one carries an
`extractor` key).

    scripts/ops/mem_cap.sh 16 timeout 1500 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
        designs/research_state/measurements/op_gigo_2026-10-07/measure.py
    scripts/ops/mem_cap.sh 16 timeout 1800 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
        designs/research_state/measurements/op_gigo_2026-10-07/measure.py \
        --checkpoint /home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1007/final_model.zip

## Results (n = 580 battles / 42,465 rows)

| | FRESH INIT | TRAINED (blob s1007, 15.0M) | of |
|---|---|---|---|
| **(a)** opp ACTIVE has an unrevealed ability with a non-zero top-1 prior id | 12,861 (30.3 %) | same | 42,465 rows |
| · of those, our active has ≥ 1 status move | 4,504 | same | 12,861 |
| **(b)** OLD ≠ NEW in ANY model output (pi / vf / pointer inputs / win-prob / α) | 3,323 (25.8 %) | **7,041 (54.7 %)** | 12,861 (a) rows |
| · OLD ≠ NEW in the op's flat block | 3,323 | 3,323 | 12,861 |
| · material: a status-move p_land or an outgoing KO/high/low cell moves > 0.05 | 344 (2.7 %) | 350 (2.7 %) | 12,861 |
| · · status p_land > 0.05 | 326 | 326 | |
| · · outgoing damage cell > 0.05 | 20 | 25 | |
| · status `known` bit flips (certain → prior estimate) | 2,146 | 2,146 | |
| · OLD ≠ NEW in model outputs on rows OUTSIDE (a) | 0 | **1,169** | 29,604 |
| **(c)** played status moves, executed, their active stayed, their status unchanged first | 1,531 (1,141 landed) | same | |
| · OLD op certain-zero claims (p_land 0, known 1) | 199 | same | 1,531 |
| · · **contradicted: the status LANDED** | **12** (all Toxic into Snorlax, 4 battles) | same | 199 |
| · NEW op certain-zero claims | 181 | same | 1,531 |
| · · contradicted | **0** | same | 181 |

The status-move resolution in (c) is the committed gate's: `src/agents/model/op_status_landing_bridge_integration_test.py`.
It excludes 300 played moves because their active switched first, 237 because our move did not execute, and 20
because their side's status changed before our move.

The two material columns differ (outgoing damage cell 20 vs 25) because `_outgoing_block` reads the forward's
spread belief, which is learned.

**(d) Drivers.**
- (a) rows by species [top-1 prior ability]: Blissey [Natural Cure] 3,613 · Skarmory [Keen Eye] 2,601 · Snorlax
  [Immunity] 2,516 · Magneton [Magnet Pull] 1,181 · Aerodactyl [Rock Head] 890 · Starmie [Natural Cure] 728 · …
- (b) any-change rows:
  - fresh: Snorlax 1,629 · Blissey 576 · Skarmory 331 · …
  - trained: **Blissey [Natural Cure] 2,887** (presumably through its other prior ability, Serene Grace, which scales its secondaries — UNVERIFIED) ·
    Snorlax [Immunity] 1,830 · Magneton 829 · Dugtrio 420 · Skarmory 419 · Aerodactyl 259 · …
- (b) material rows: **Snorlax [Immunity] 334 / 335** · Hariyama [Guts] 6 / 11 · Houndoom [Flash Fire] 4 / 4.
- (c) contradicted: Snorlax [Immunity] Toxic, 12 of 12. Also, 6 OLD certain-zero Toxic claims into Snorlax did not
  land, and the NEW op prices them as possible.
- No Levitate / absorb row appears among the material drivers. Gengar's prior is Levitate at 1.0, so the old and
  new reads coincide.

**(e) The new status rules, each switched off alone.** Op-block changes are 0 for every rule in both columns: the
incoming status channel lives outside the flat block.

| rule | model-output rows changed: FRESH | TRAINED | rows where its condition holds |
|---|---|---|---|
| their Safeguard (outgoing) | 0 | 0 | 0 (no Safeguard line in 580 battles) |
| their active drowsy (our Yawn fails) | 0 | 0 | 0 (no Yawn line in 580 battles) |
| Yawn as a sleep inflictor (the table change) | 0 | 0 | (no Yawn) |
| our Safeguard (incoming) | 0 | 0 | 0 |
| our active drowsy (incoming) | 0 | 0 | 0 |
| incoming Sleep Clause | 0 | **150** | 283 |
| Freeze Clause | 0 | **246** | 482 |
| our Substitute (incoming) | 0 | **259** | 449 |
| REACH control: the op's whole incoming status channel set to 0 | 0 | **33,588 (79.1 %)** | 42,465 |

## Findings

1. **The ability fix matters on real data, almost entirely through Snorlax.** The old read made 12 certain-zero
   Toxic claims that the protocol contradicts, out of 199 certain-zero claims on 1,531 resolved status moves. The
   new op makes 0 contradicted claims out of 181. On a trained model, the old read changes some model output on
   54.7 % of the affected rows (plus 1,169 rows outside them) and moves a p_land or a damage cell by more than 0.05
   on 2.7 %.
2. **CORRECTED: the op's INCOMING status channel does reach the trained model.** Zeroing it changes model outputs
   on 79.1 % of rows, and the incoming rules each change outputs on a material share of the rows where their
   condition holds:

   | rule | rows changed | of rows where the condition holds |
   |---|---|---|
   | Sleep Clause | 150 | 283 |
   | Freeze Clause | 246 | 482 |
   | our Substitute | 259 | 449 |

   The first version of this note said the channel "reaches no model output". That was an artifact of the
   fresh-init extractor: the zero-init projections that carry the channel read exactly 0 at init. It is not true of
   a trained model.
3. **The outgoing Safeguard and Yawn rules, and our-side Safeguard / drowsy, are UNEXERCISED by this bank.** No
   Safeguard or Yawn line appears in 580 battles, so they are verified only by the unit tests
   (`op_status_rules_test.py`), not on real data.
4. **CORRECTED: the opponent's BENCH abilities do reach the trained model.** 1,169 rows outside (a) change under
   the old read. That is through the six-slot reads, e.g. `discrete_outgoing_status` and the pairwise cells. At
   fresh init those reads also showed 0, the same zero-init artifact as finding 2.
5. **Caveat on the trained column.** The checkpoint trained under the OLD op (pin `706fa536`), so its weights
   adapted to the old read. Its counts measure how sensitive a trained model is to the change, not what a model
   trained under the new op would do.
