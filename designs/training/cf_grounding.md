# Training — cf grounding (the TRAINING half DELETED; the OFFLINE stack stays)

**The counterfactual (cf) TRAINING half was deleted 2026-10-02, deletion pass L4** (owner-approved,
decision D1 "delete, port none"; config v134, stamp-only — no `ARCH_SIGNATURE` bump). This file used
to hold its operating detail; the training half is now this one paragraph of history, and what
remains below is the OFFLINE cf stack, which reads OLD runs' `cf_records/` rings and is kept
(design decision D6).

**What the deleted half was.** The record TAP (a count-capped ring of reconstruction records,
`cf_records` / `cf_records_keep`, written by every env worker), the LABEL BUFFER (`cf_label_buffer.py`,
a polled, deduplicated, staleness-bounded buffer over a producer's label files), every CONSUMER term
(`cf_terms.py`: the win-prob grounding term with its binomial / BCE likelihood, the head-only switch,
and the evidential / twin / shadow terms; `q_winprob_terms.py`: the per-action Q terms), the SUPPLY
GUARD (`cf_supply.py`, `cf_supply_callback.py`: the trainer spawned or verified a producer, and a
starved stream was `FATAL_SUPPLY`), the duty-cycle refusal, and the four STRUCTURAL heads those terms
supervised — the evidential Beta head, the twin win-prob heads B/C, the passive shadow critic, and the
per-action Q win-prob head. The `train/cf_*`, `cf/*` and `q_winprob/*` TensorBoard families and the
`grad/cf_*_share` / `grad/q_winprob_share` probes no longer exist. A checkpoint that recorded a
structural head ON is refused on every load; one that recorded a cf coefficient live refuses
resume / fork (`agents/model/model_version/retired_levers.py`). Flag-by-flag citations:
`designs/deleted_flags.md`; the deletion's scope: `designs/ops/deletion_pass_manifest.md`.

**Why it is gone (the headline evidence, kept as history).**

* **Only 3 runs ever ingested a cf label** (ledger 2026-09-30, *CfLabelBuffer INODE REUSE*).
  `ai_v12_12_ladder_cflabels` trained 10M steps at a live win-prob coefficient of 0.5 and received
  **ZERO labels** — the producer was a separate program nobody started, and every signal that could
  have said so (an empty `cf_labels/`, `cf/labels_ingested_total` flat at 0, `train/cf_loss` never
  written) was a scalar nothing thresholded. A supply guard built on 2026-09-30 to make that loud was
  deleted with the rest of the half.
* **No lever was found that moves leaf quality** (the cf-labels arm and the leaf battery,
  2026-09-12). The earlier G0 bias map (ledger 2026-08-22; 2,204 tight-MC
  labels) had found the win-prob head's defect is **RESOLUTION, not an optimism offset** — the
  population-mean predicted-vs-MC gap is |0.05|–|0.07| with a sign that flips with the population
  weighted to, while the true within-decile spread of P(win) is 0.11–0.36, so the per-state error is
  2–6x the aggregate offset. That diagnosis and its meter, `sd_true_excess`, survive in `cf_audit`
  below.
* **The labels that would still be wanted are not the ring's.** X4 / X6 playouts are deferred and,
  when scheduled, would use the Rust search driver, not the `cf_records/` ring.
* **The duty-cycle trap, as history:** the cf label buffer expired a row more than a bound behind the
  live policy while the producer could only stamp the newest checkpoint's step, and SB3's
  `CheckpointCallback.save_freq` counted VEC-ENV CALLS, not env steps — at 48 envs a 6.25% duty cycle
  (measured on `ai_v9_29_rev1_0823`: 6 labels ingested against 255 expired in two hours, every
  counter reading healthy). The trainer's checkpoint cadence is now declared in env steps
  (`--checkpoint-every-steps`, `designs/ops/training_runbook.md`).

**What the Python fork arm still reads (until unit L5).** `agents/training/cf_records.py` is reduced
to the `safe_tag` / `record_key` / `index_records` join helpers; its only readers are the Python fork
arm and the wrapper's decision-time handle capture, both unreachable now (`--fork-fraction > 0` on
`--env-core python` is refused, combination row `fork_python_core_unavailable`) and deleted with L5.
The Rust fork port needs no ring ([`forks.md`](forks.md) §14).

**What stays — the OFFLINE cf stack (design decision D6; nothing in training spawns or consumes it):**

| piece | where | described below |
|---|---|---|
| the counterfactual audit instrument | `agents/training/cf_audit.py` (+ `cf_audit_render.py`, `cf_audit_twin.py`) | § `cf_audit` |
| the label PRODUCER driver (reads an old run's `cf_records/` ring, writes `cf_labels/`) | `agents/training/cf_producer.py` (+ `cf_producer_sampler.py`, `_snapshot.py`, `_labels.py`, and `cf_producer_lock.py` — the single-instance lock + `--parent-pid` binding, the one piece of the deleted supply guard it kept) | § label PRODUCER DRIVER |
| the per-action label arithmetic, the MC-return label | `agents/training/cf_q_labels.py`, `agents/training/cf_mc_return.py` | § PER-ACTION stream |
| prefix-sharing materialization | `agents/training/obs_materializer.py` | § Prefix-sharing |
| the win-prob fine-tune, the harvest schema | `agents/training/winprob_finetune.py`, `agents/training/harvest_schema.py` | their own module docstrings |
| the live critic read | `main/ops/critic_read.py` | its docstring |
| the counterfactual rollout driver | `utils/bridge/counterfactual.py` | [`../../src/utils/bridge/README.md`](../../src/utils/bridge/README.md) |
| the prober's counterfactual views | `main/prober/` | [`../prober/counterfactual_probes.md`](../prober/counterfactual_probes.md) |

---

## `cf_audit` — the counterfactual audit instrument (`cf_audit.py`)

**Three modules, one instrument.** `cf_audit.py` owns the frame, the sampler, the label
schema, the bias map and the CLI; two readouts live beside it because they are the parts that
need nothing else the module knows. **`cf_audit_render.py`** takes a finished bias map and
returns markdown — no statistic, no file, and its two formatting rules are the ABSENT-vs-ZERO
ones (a checkpoint with no evidential head renders a NOTE, a flat width renders `n/a`, because
a row of zeros makes "no head" and "no uncertainty" read identically). **`cf_audit_twin.py`**
holds the twin-head paired read, `twin_resolution_read`, `shadow_read` and `attach_twin_heads`
— it takes labelled rows and, for the last one, a session. Both were extracted on 2026-09-06
(the ratchet's second cut, 1279 → 918 lines, under the 1,000 TARGET), and `cf_audit`
re-imports every name, so `from agents.training.cf_audit import render_markdown` still
resolves. **The extraction-parity golden below is the evidence for both moves** — it calls
them through those re-exports and was not regenerated.

```bash
python -m agents.training.cf_audit models/<run> \
    [--rollouts 8] [--states 200] [--step N] [--checkpoint PATH] [--impl rust] [--out DIR]
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

**Offline and standalone — it trains nothing.** Given a run's bridge-eval traces (the ones with a
`*_reconstruction.json` sibling) and a loadable checkpoint, it manufactures the value labels the
on-policy stream structurally cannot produce: for a sampled decision it plays the **recorded**
action and rolls the rest of the battle out live **R** times with fresh post-divergence dice
against the RELOADED real opponent, and takes the win rate. Training sees one Monte-Carlo sample of
each state's value; this sees R of the same state.

It emits two things, independently useful:

| output | what |
|---|---|
| `<out>/bias_map.json` + `bias_map.md` | predicted win-prob vs tight-MC per stratum, with `sd_true_excess`, battle-clustered CIs, the sampler design and full accounting |
| `<out>/cf_labels/labels_<producer>_<step>.jsonl` | label rows in the **shared v1 schema** — the contract a label reader consumes |

**The meter is `sd_true_excess`, NOT the mean gap.** G0 (2026-08-22) measured the population-mean
predicted−MC gap at |0.05|–|0.07| *with a sign that flips with the population you weight to*, while
the true within-decile spread of P(win) is 0.11–0.36 — the per-state error is 2–6× the aggregate
offset, so the head's defect is **resolution**, not an optimism offset. The estimator subtracts the
R-rollout binomial floor from the observed within-cell variance:

```
Var(MC | cell) = Var(true p) + E[sampling var];   E[p̂(1−p̂)]/(R−1)  is EXACTLY unbiased for p(1−p)/R
sd_true_excess = sqrt(max(0, Var(MC) − E[p̂(1−p̂)]/(R−1)))
```

Subtracting the floor is what makes it a claim about the world rather than about R. **A lever that
merely re-centres the head moves the mean gap and leaves this untouched** — and would be scored a
success by the wrong meter, which is exactly why the meter is stated here.

**The EVIDENTIAL and TWIN-HEAD reads (history of a deleted lever).** `cf_audit` / `cf_audit_twin` still carry
the readers for the evidential Beta head (`width_vs_blur_spearman`, the rank correlation across strata
between the head's epistemic width and the measured `sd_true_excess`, with a battle-bootstrap CI, a FLAT
width scoring `None` rather than 0, and a checkpoint without the head OMITTING the columns rather than
printing zeros) and for the twin win-prob heads / shadow critic (paired per-row proper-score
differences). The heads themselves are DELETED (deletion pass L4) and a checkpoint that recorded one ON
is refused on every load, so those columns cannot be produced on a loadable checkpoint; the reader code
still sits in those modules (the extraction-parity golden below covers it). **UNVERIFIED:** whether a
later pass removes it. The read is best-effort throughout: the audit's products are the labels and the bias
map, so a model that will not load (architecture drift — measured **2026-08-13: 79 of 79 archived
runs**; the tree carried 100 checkpoint-bearing runs as of 2026-08-23 and the 0-of-N was not
re-measured) costs the run only its head columns.

**Label trust before map trust.** The ANCHOR arm runs FIRST: recorded action + recorded dice must
reproduce the recorded battle outcome. Below `--anchor-tolerance` (default 0.9) the tool exits **3**
and writes NO labels — the bias map is still written, marked `label_trust_passed: false`, for
diagnosis only. A factory whose replay is not exact is GIGO, and a map computed from it measures the
bug. Pinned by `cf_audit_integration_test.py`.

**Selection-awareness.** Eval traces over-capture losses (an explicit win/loss quota), so a pooled
gap convicts the critic of the sampler's sins. Every aggregate is computed *within* an outcome
stratum and recombined at the frame's own population shares; every CI is a bootstrap over
**battles**, never states.

**Sampling** is `(confidence decile × battle outcome × turn tercile)` with a declared
`CONVICTION_BOOST` on the high-confidence-from-lost-battles region (the "0.827 class", the
population R1 supervises). The weights, the seed and `SAMPLER_VERSION` are written into every bias
map — a silent priority change is a distribution-shift confound for every downstream readout.

**The shared label schema (v1)** — one JSON object per line; treat it as a contract, version it
rather than editing it in place:

```json
{"schema": 1, "kind": "mc_winprob", "battle": "<record path>", "decision_idx": 12,
 "obs_sha1": "<sha1 of the obs float32 bytes>", "obs_npz": "<states.npz>::obs",
 "obs_inline": null, "label": 0.625, "n_rollouts": 8, "wilson_lo": 0.30, "wilson_hi": 0.86,
 "policy_step": 24000000, "opponent": "heuristic", "created_unix": 1.77e9}
```

`obs_npz` names the array and `decision_idx` selects its ROW; `--inline-obs` swaps that for a
base64 float32 payload when the traces won't travel with the labels. `obs_sha1` is always present
so a reader can verify the row it loaded is the row that was labelled.

**Known coverage bounds, printed in the accounting and never silent:** turn-1 decisions (one per
battle, 3.35% of move decisions) and forced-switch rounds (the re-roll layer anchors at
start-of-turn move rounds).

⚠️ **The two have DIFFERENT standing, and the turn-1 one changed on 2026-08-23.** Forced-switch
rounds are a structural limit of the re-roll anchor. Turn 1 is not: it was a rust `search_driver`
defect (`at_turn_start` compared `BattleState::turn`, which still reads 0 at the pre-commit first
boundary), **fixed by `gen3_search_turn1_open_v1`** — both impls now open turn 1, and node always
could. So turn-1 decisions ARE labelable, and the `turn_1_unopenable` skip key is a retained
misnomer for a **sampler** bound (`cf_producer.MIN_LABELABLE_TURN = 2`), not a capability limit.
Lowering it widens the declared candidate distribution by ~3.35% and is deliberately left as its
own change: missing a label is free, silently re-weighting the sampler is not.

**Cost** is the rollouts, not the materializer: an R=8 label is ~0.9 s at load ~7 and ~2.8 s at load
~25 — *more* load-sensitive than `loadavg/cpus` predicts, so any throughput figure taken beside a
trainer is a lower bound. Prefix sharing (below) does not apply to a rollout-to-end label, which has
one arm; it is the lever for the one-ply counterfactual (`lookahead`) path.

**Tests.** `cf_audit_test.py` (pure: the stratifier, the schema writer, and the EXTRACTION PARITY
GOLDEN — every public readout on one synthetic fixture, JSON-serialised canonically and pinned by
digest plus a dozen named values, captured from the tree BEFORE the statistics moved to `stats.py`
and reproduced byte-for-byte through that move AND the `cf_audit_render` / `cf_audit_twin`
extraction that followed it: 43,075 bytes, sha256 `cf2971d5…`, never regenerated),
`cf_audit_render_test.py` and `cf_audit_twin_test.py` (the tests that moved with their
functions; both import their fixtures FROM `cf_audit_test` rather than copying them, because a
renderer tested against its own hand-built bias map is testing a shape the audit no longer
emits), `stats_test.py` (the estimators themselves — `sd_true_excess`
validated at ZERO true effect AND at a known nonzero one, the clustered bootstrap and its
difference-of-means sibling, Wilson, Spearman) and `cf_audit_integration_test.py` (`sim`: a real

## The label PRODUCER DRIVER (`cf_producer.py`) — the loop from an old run's ring to label files

```bash
nohup nice -n 10 python -m agents.training.cf_producer \
    models/<run> [--rollouts 8] [--top-n 3] [--records-per-cycle 4] \
    [--max-labels-per-hour 2000] [--anchor-every 50] [--impl rust] \
    > models/<run>/cf_producer.log 2>&1 &
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

**Standalone and OFFLINE.** The deleted training half used to ring `<run>/cf_records/` and consume the
label files this process writes; no run writes a ring any more, so the producer only has an OLD run's
`cf_records/` ring to label (37 archived runs held one as of 2026-08-29, and only one of them could
still be loaded by the code of that day). It is operator-run: the trainer neither spawns it
nor checks that it exists. It holds an exclusive `flock` on `<run>/cf_producer.lock` for its lifetime
(`cf_producer_lock.py`; the kernel drops it on any death) and exits **4** if another holds it — two
producers would share one state file and one `seq`; the optional `--parent-pid` binds it to a parent
(`PR_SET_PDEATHSIG(SIGTERM)` plus a ppid check between cycles). Its `--device` defaults to `cpu`.

**Four modules, one factory.** `cf_producer.py` owns the LOOP and everything with state in it —
the cycle, the record ring's reading side, the crash-safe `ProducerState`, the anchor, the rollout
arms, the heartbeat and the CLI. Three pieces that need nothing the loop knows live beside it, the
same shape as `cf_audit`'s split above. **`cf_producer_sampler.py`** holds the DECLARED, VERSIONED
priority (`cf_producer_priority_v1`): `SAMPLER_VERSION` / `PRIORITY_WEIGHTS` / `MIN_LABELABLE_TURN`
and the four pure functions that rank a candidate — `critic_surprise` (the conviction region),
`normalized_entropy`, `priority_score`, and the `is_move_round` filter. Those two CONSTANTS had to
travel with the arithmetic rather than stay behind, because `ProducerState` and `label_row` both
STAMP them and a constant left in the hub would have made the import edge point back up.
**`cf_producer_snapshot.py`** holds WHICH weights are running (`resolve_latest_checkpoint`,
`step_from_checkpoint_name`) and the `Snapshot` that both scores decisions and builds the players
that roll them out — one object because the two must use the SAME weights, and almost everything
non-obvious in it is a `torch.compile` shape/dtype fact (B=1 scoring under a compiled graph,
float32 masks, the two-key warm-up) rather than arithmetic. **`cf_producer_labels.py`** holds the
v1 label ROW and its batch writer — a CONTRACT with its readers (the deleted `cf_label_buffer` was the training one), who know nothing about
this loop — and `OPPONENT_LABEL`, i.e. THE ECOLOGY DECISION itself. The cut was 2026-09-06 (the
ratchet's third pass over the 1,000-2,000 band, **1899 → 1484 lines**; still in the band, because
the loop class alone is ~810 lines and the 1,000 TARGET is unreachable without splitting it).
`cf_producer` re-imports every public name, so `from agents.training.cf_producer import label_row`
still resolves; the private `_warm_the_compiled_graph` is the one name that does not, and its two
tests reach it through the owning module. **The extraction-parity golden below is the evidence.**

#### 🚨 THROUGHPUT — the ~8 s/label was the POLICY FORWARD, not the drivers

Profiled 2026-08-23 on `/tmp` copies of live `ai_v9_29_rev1_0823` ring records against that run's
own checkpoint, beside the live trainer (load 15-25):

| stage | share | note |
|---|---|---|
| **rollouts** (R × `replay_counterfactual`) | **93%** | of which **93% is `choose_move`** |
| `scan_record` (replay + obs materialize) | 0.3% | ~170 ms/record, ONCE |
| `replay_battle` (offline replay driver) | 0.2% | ~35 ms/record, ONCE |
| `score` (the ranking forward) | 0.1% | |
| record parse / label write | <0.1% | |

Inside a rollout: **832 `choose_move` calls at 15.4 ms** against 0.40 ms of scripted-prefix
`embed_battle`, 0.14 ms of `_invert_choice`, and a 9 MB rust child spawn. A B=1 CPU decision
measured **26.3 ms eager → 4.1 ms compiled (6.4×)**; the extractor alone is 21.5 → 3.3 ms, i.e.
~82% of it.

**That contradicts the banked ~162 ms/label cost model, and the correction is the useful part.**
That model was measured on the *materializer* path (one-ply labels, whole-prefix replay per arm),
where cold driver spawns dominate and a warm rust `SearchSession` is 289×. This producer is a
different shape: it replays each record **once** (`scan_record` takes `chunks=` from the single
`replay_battle`) and its labels are **rollouts to end**, which spawn no offline driver at all — they
play a live bridge battle whose cost is policy forwards. So a **warm/persistent search-driver
session buys ~0.2% here, and prefix sharing ~3%** (a scripted prefix decision is 0.4 ms against
4-26 ms for every live one, and cloning the mid-battle state would have to clone two poke-env
players' trackers as well). *When a cost model is carried across a path shape, re-measure before
building to it.*

**What shipped (`gen3_cf_producer_compiled_rollouts_v1`), and the three shape hazards it closed:**

* **`--compile-extractor` now DEFAULTS ON.** It is a fallback, not an opt-in;
  `--no-compile-extractor` is there for when the compile is the suspect (a compile *failure*
  already degrades to eager on its own). It costs ~40 s **once per PROCESS, not per checkpoint**:
  dynamo keys on the CODE object and the weights are graph inputs, so the next refresh reuses the
  graph — measured **1.1 s for an entire second `load_snapshot`** against 39.7 s for the first.
* **`score` forwards ONE ROW AT A TIME under compile.** Dynamo specializes on the batch dimension,
  so a single `B=29` scoring call in front of B=1 rollouts forced a re-trace: measured **79.4 s for
  the first label's 8 rollouts against 3.0 s for the second**. Row-wise costs ~0.12 s/record against
  ~0.04 s batched and keeps exactly one signature alive. Eager snapshots keep the batched forward.
* **The mask is cast to `float32` there too.** A materialized mask is `int8` and a live one is
  `float32`, and a compiled graph guards on DTYPE as hard as on shape — **19.5 s on the first
  scored row** before the cast.
* **The compiled graph is warmed through the LIVE call signature** (`_warm_the_compiled_graph`).
  `maybe_compile_extractor` warms with `{"observation": …}` alone; every real call also carries
  `action_mask`, and a dict's KEY SET is part of the guard — so the compile looked warm and the
  first real decision re-traced for **19.5 s**, charged to whichever record was first. It is now a
  startup cost that announces itself.
* **`--rollout-concurrency` (default 1) is a KNOB, not a win.** Measured a wash over 10 paired
  label-arms (conc=1 mean 3.86 s vs conc=8 4.17 s, no consistent sign): every policy forward *and*
  every protocol parse runs on poke-env's single `POKE_LOOP` thread, so overlapping arms finds
  almost no idle to fill. The arms are independent by construction (own players, own bridge child,
  own post-divergence dice) and one that dies costs that arm alone.

**Measured before/after — a REAL one-cycle run, same 6 records, same checkpoint, same box:**

| | cycle wall | per label (excl. one-time load/compile) |
|---|---|---|
| before (`d78aa81`, eager) | 198.5 s @ load 17 | **10.8 s** |
| after (compiled) | 99.4-102.9 s @ load 20-22 | **3.2 s** |

Interleaved arm-by-arm on the same decisions (the load-fair form): **8.09 s → 1.81 s of rollout
wall per label, 4.5×**. The eager arm reproduces the live producer's own 8.2 s/label, which is what
says the harness measured the right thing. **The label output is unchanged**: identical key set,
the same decisions selected with the same ranking, and all 18 rows were ingested by the (since deleted) `CfLabelBuffer`.
The only difference anywhere is `priority.win_prob` in the **6th decimal** (0.670049 → 0.670050) —
Inductor's arithmetic, the documented max|Δ| ~5e-7, on a field nothing thresholds.

#### The producer/retention race (`records_vanished`)

The training run OWNED `cf_records/` — every env worker pruned it to the newest 512 records (the
deleted `cf-records-keep` flag) — and this process only reads it, so a record can be enumerated and then deleted before it is
opened. Measured on the same run: **176 records lost to `FileNotFoundError` across 67 cycles**,
with "538 pending" against a ring of 512 (the excess is a guaranteed loss by arithmetic). Three
properties, all load-bearing, none of them a change to the ring's semantics:

* **Records are taken NEWEST FIRST.** The ring deletes from the OLD end, so the oldest pending
  record is the one already promised away — and the loop walked exactly that end. Newest-first puts
  the deletion end of the ring at the low-value end of the work queue. (It is independently the
  right sampler order: a newer record came from a policy closer to the one the label supervises.)
* **The batch is READ AT ENUMERATION TIME** (`CfProducer._load_batch`). The window the ring wins is
  enumerate → anchor (a full scripted replay, seconds to minutes) → claim + fsync → open; reading
  immediately collapses it, and everything downstream works from an in-memory record.
* **A vanished file is a COUNTED BENIGN SKIP** — `records_vanished` in the state file, on the
  heartbeat, and one explanatory log line — **never an exception path.** As an exception it landed
  in `skip_reasons` as `error:FileNotFoundError`, indistinguishable from a corrupt record, and on
  the ANCHOR path it reached `anchors_errored`, where an ordinary ring deletion could **exit 3**
  and stop the factory. The remedy was a larger ring cap, which no longer exists: an old ring is fixed, so the producer
should be run over it promptly if it is still being pruned by a live old-code run.

Each cycle: poll `<run>/cf_records/` for unprocessed records → refresh the freshest `checkpoints/`
snapshot (via `latest.txt`, else the highest-stepped zip; its step is stamped on every label) →
replay each record ONCE (which yields the realized outcome, every decision's obs, its mask, its
action index and its committed choice string, via `obs_materializer.scan_record`) → forward the
snapshot over the candidates → label the top `--top-n` by the declared priority → roll each out
`--rollouts` times → write one NEW file per batch to
`<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`.

**"The freshest checkpoint" means BOTH names a resumable checkpoint is written under** — the
periodic `checkpoint_<step>_steps.zip` *and* the FORCED `checkpoint_forced_<step>_<HHMMSS>.zip` that
SIGUSR1 writes (the launcher TUI's `c` key). Reading only the first was not cosmetic: a forced save
was reachable solely through `latest.txt` and then, because its step did not parse,
`resolve_latest_checkpoint`'s key ranked it **below every periodic zip** — so forcing a checkpoint
mid-run walked the producer BACKWARDS onto an older snapshot and it went on stamping that older
step, with every counter on both sides reading healthy. Found by the R1 multi-cycle composition
smoke (2026-08-23), which is the only thing that had ever run a producer across a checkpoint
boundary; pinned by `cf_producer_test::TestCheckpointResolution` (the resolver) and
`cf_producer_integration_test::test_a_new_checkpoint_mid_run_restamps_the_labels`
(the label path as a whole moving forward).

⚠️ **THE ECOLOGY DECISION — read this before quoting any label this producer wrote.**
A training record carries **no opponent identity**. The tap's `__RECON__` frame holds the resolved
seed, both packed teams and the committed choices, and nothing that says *which policy* sat on the
other side — a self-play pool snapshot, one of the nine heuristic bots, or the trainee's own
weights. The label therefore cannot name the opponent it was measured against, and a value claim
that cannot name its population is not a value claim (the G0 rule: *never quote "the critic is
optimistic by X" without naming the population — the sign depends on it*). So v1 makes the
approximation **explicit rather than guessed**: every rollout is played by the **CURRENT snapshot
on BOTH sides, sampling stochastically at temperature 1.0** — the regime the training actor itself
plays in. That matches the ~90% self-play share of the training mixture, and it is wrong in a
KNOWN direction for the rest: on an episode whose opponent was a bot, a weaker opponent is replaced
by a stronger self-like one, so that label is biased LOW. Every row carries
`opponent: "self_current"` — never a bot name it cannot verify — so a reader can always tell a
producer label from a `cf_audit` label, whose opponent IS identified. Closing the approximation
means threading the opponent's identity through the training-side tap; it is not a change to
`cf_producer.py`. **Stochastic is the load-bearing half of the regime**, not a style: a greedy copy
of a net is strictly stronger than a temp-1.0 sample of it, and greedy rollouts biased the prober's
sentinel labels LOW by a measured +0.037 [+0.007, +0.066].

**Which side is the trainee.** A training record names none, so `_trainee_side` answers from the
transport's own invariant: `BridgeSession` seats `env.agent1` — the trainee — on **p1**, always. A
record that DOES name a trainee (an eval sibling handed to this tool) is honoured instead.

🚨 **A rollout that reaches the 250-turn cap is a DRAW AT CAP and scores 0.5** — never a win or a
loss (`gen3_cf_draw_at_cap_v1`, fixed 2026-08-23). Both sides of a rollout stall-forfeit at
`MAX_TURNS`, so at the cap BOTH forfeit and the recorded winner is decided by which `FORCELOSE` the
sim processes first — a fact about ordering, not about the position. **Measured over 16 capped
lines on `node` and `rust` alike, the ordering is not even a coin flip: p1's forfeit is always
processed first, so p1 always loses**, and `_trainee_side` puts the trainee on p1 always. Every
capped rollout therefore used to score a hard 0, biasing tight-MC P(win) labels **DOWNWARD** on
exactly the stall-shaped positions where the cap is reachable — an *upward* bias was guessed when
the class was first noted, and the guess was wrong. A genuine tie went the same way (`outcome ==
"win"` is False for a tie) and is likewise 0.5 now. The count rides out as **`n_capped` beside
`n_rollouts`** on every row (an ADDITION; a reader that reads a fixed key set ignores the rest,
and `schema` stays 1) and as `rollouts_capped` in the state file + the heartbeat — because a 0.5
built from 8 draws-at-cap and a 0.5 built from 4 wins and 4 losses are the same number about
different positions, and no reader can re-derive which afterwards. `wilson_lo`/`wilson_hi` now take
a fractional success total, so with draws in the sample the interval is an approximation that errs
narrow; `n_capped` is what says how much. Detection is exact rather than heuristic —
`replay_counterfactual` returns `capped = finished and turn >= turn_cap_of(both players)`, and
`_handle_stall` forfeits at every decision from the cap turn onward, so a battle can never resolve
normally on or after it. Gated by `cf_producer_test::TestDrawAtCap` (revert-verified),
`counterfactual_test::test_battle_outcome_flags_a_battle_that_ENDED_AT_THE_CAP`, and the `sim`
`cf_producer_integration_test::test_a_rollout_that_reaches_the_TURN_CAP_is_a_draw_on_either_seat`,
which plays the same fixture board from BOTH seats at a forced low cap and requires one label.

⚠️ **Labels written before that fix carry no `n_capped`, and cap-reaching is NOT re-derivable from
them** — the rollouts leave no artifact and a capped 0 is indistinguishable from a played-out 0. On
`ai_v9_29_rev1_0823` (999 rows / 333 source records / 7,992 rollouts as of 2026-08-23) what IS
derivable bounds it: every label sits at a decision turn ≤ **96** (p50 9, p90 25), so a rollout
needs ≥154 further turns to cap; and the cap's base rate in the surrounding training population is
**2 stall events across the 4,097 episodes in the record ring's 5.1-minute window (~0.05%)**, which
puts the expected count in the single digits of ~8,000 rollouts. Treat that as a base-rate estimate,
not a measurement of the corpus.

⚠️ **A record written BEFORE 2026-08-24 by the rust bridge carries no `forcelose` entry at all, so a
scan of its `commands` reads a false 0.** The rust `sim_bridge` pushed `commands` only in
`handle_choose`, while node has always pushed `['forcelose', <side>]` — so under the PRODUCTION
default (`--use-bridge rust`) a forfeited battle's record looked exactly like one that played on.
Two consumers read that field and both were silently wrong: the offline replay path
(`search::feed_recorded_cmd` has a `"forcelose"` arm; `recorded_turn_choices` stops at one) never
reproduced the forfeit, and `record_is_full_replay_anchorable`'s forfeit exclusion was **INERT**
(`anchors_skipped_unanchorable` reading 0 on the live run means the exclusion never fired, not that
there were no forfeits — it is what misled the #34 census). **FIXED at the record writer**
(`handle_forcelose` now pushes the entry, at the site node does), which repairs the record itself
rather than one of its readers. Gate:
`bridge_impl_parity_test::test_a_forfeited_battles_record_logs_the_forcelose_command`, over BOTH
impls — verified failing on the rust arm when the push is reverted. **Records already on disk are
frozen wrong**; a forfeit census over an old rust corpus is not re-derivable from `commands`.

**The sampler is DECLARED and VERSIONED** (`cf_producer_priority_v1`), written into the state file
AND every label row, because a silent priority change is a distribution-shift confound for every
downstream readout (design decision-of-record 3):

| term | what | weight |
|---|---|---|
| `critic_surprise` | `\|P(win\|s) − realized outcome\|` — the **conviction region** G0 measured at +0.23, and the population R1 exists to supervise. A single realized outcome cannot say whether the head was wrong or the dice were (53% of that class was genuinely winning); tight-MC labels are the only instrument that separates them, so they are spent here first | **1.00** |
| `policy_entropy` | the masked action distribution's entropy ÷ `log(n_legal)` — the decisions the policy has not made up its mind about. **Normalized by the support size** so a 2-way coin flip outranks a 9-way near-certainty, which raw entropy inverts | **0.35** |

A tie (the turn cap) scores outcome **0.5**, not a loss — it is uninformative about conviction, not
evidence the head was wrong. A checkpoint with **no win-prob head** has no surprise term at all;
the producer says so once and ranks on entropy alone rather than reading a missing head as a
confident 0.0. Candidates are start-of-turn **move rounds** at turn ≥ 2 only (a forced-switch round
has no valid recorded answer to script; the turn-1 bound is the same one `cf_audit` declares — a
retained SAMPLER choice since `gen3_search_turn1_open_v1` made both impls able to open turn 1, no
longer the driver limitation it was introduced for).

**Crash safety, and what it costs.** A record is claimed in `<run>/cf_producer_state.json` and the
state file is **fsync-replaced BEFORE its rollouts run**. So a crash mid-record loses that record's
labels and can NEVER double-label. That direction is deliberate: the (deleted) buffer deduped on the obs
digest, so a duplicate was survivable — but it is also a silent re-weighting of the declared
sampler, and a record aged out of the ring unprocessed is simply a record that was not labelled.
Missing a label is free; mis-weighting the sampler is not. Pinned on the ORDER (the state file must
already be durable when `process_record` raises), not argued.

**The anchor rule, inherited from `cf_audit`.** At startup and every `--anchor-every` records, one
record is replayed FULLY SCRIPTED through the live bridge (`divergence_turn=None` — the correctness
oracle) and must reproduce the winner the offline replay driver reports. On failure the producer
**exits 3 and writes nothing further**: a factory whose replay is not exact is GIGO, and every
label after it would be a measurement of the bug. This anchor is *stronger* than `cf_audit`'s —
nothing is played by a policy, so a MISMATCH is unambiguously a defect rather than a die roll. An
anchor that CRASHED counts as a FAILURE, never a pass.

⚠️ **But a CRASH and a MISMATCH must not print the same sentence, and until 2026-08-23 they did.**
The two refusals reach the same exit 3 and have opposite diagnoses: a mismatch says the replay is
inexact; an exception (a wedged bridge child, a transport error, a contention `ProgressTimeout`)
never returned a verdict, so it says nothing about exactness — it refuses because an anchor that
did not complete has certified nothing. `main` printed the MISMATCH text for both, and that turned
ONE flaky `cf_producer_integration_test` failure into an investigation of a replay-exactness gap
that did not exist. They are now counted apart (`anchors_errored` in the state file and the
heartbeat, beside `anchors_run`/`anchors_reproduced` — the split `cf_audit` has always had) and
rendered apart by the pure `anchor_refusal_message`, which appends `describe_contention()` when the
exception was a timeout. Same rule as everywhere else in this tree: **a timeout is never a semantic
outcome**, and a message must never assert a cause the code has not established. Pins:
`cf_producer_test::TestAnchorRefusal::test_an_anchor_{ERROR_is_counted_and_reported_apart_from_a_MISMATCH,TIMEOUT_self_diagnoses_instead_of_accusing_the_replay}`.

#### ⚠️ The FORFEIT class — the one thing a live scripted replay cannot adjudicate

**Root-caused 2026-08-23**, from the single intermittent `ANCHOR REFUSED` the R1 composition test
hit. `record_is_full_replay_anchorable` now EXCLUDES it, visibly and by count
(`anchors_skipped_unanchorable`); it is a declared coverage bound of the oracle, in the same family
as `cf_audit`'s turn-1 and forced-switch bounds — **not** a retry, and never a second attempt at the
same record.

- **The mechanism.** A battle that reaches `StallConfig.threshold` (= `MAX_TURNS`, 250) is ended by
  ONE side forfeiting, which the bridge logs as `['forcelose', <side>]` in `record.commands`.
  `install_scripted_prefix` builds each side's script as `[c for (s, c) in commands if s == side]`,
  so `'forcelose'` matches NEITHER side and is dropped — the scripted replay has no way to reproduce
  the recorded forfeit. Instead **both** players re-derive one from their own `_handle_stall` at
  turn ≥ 250, and whichever `FORCELOSE` the bridge processes first loses. In the recording only ONE
  side could forfeit at all (in training, the trainee; in the composition test, the
  `RecordingFuzzPlayer` against a plain poke-env `RandomPlayer` that has no stall handling), so the
  replay can hand the win to the side that actually LOST — the exact `scripted full replay → win,
  record says <opponent>` signature the ledger recorded.
- **MEASURED. 1037 fresh battles played and rung exactly as `_play_and_ring` does, `--impl node`:
  4 refusals, 0 errors (0.39%). ALL FOUR were forfeit-terminated records — 4 of the 16 that reached
  250 turns (25%) — and 0 of the 1021 non-forfeit records refused** (95% upper bound 0.29% for any
  other class). The per-forfeit flip rate is itself a race, so treat 25% as order-of-magnitude and
  possibly load-dependent: the four splits were 0/8 in one batch and 2/2 in another. Re-anchoring one refusing record refused **7/12 and 8/12** across two batches, so it
  is a RACE, not a property of the record; every non-forfeit record re-anchored **40/40 identical**.
  Rebuilding the anchor with the OPPONENT's stall threshold unreachable — mirroring the recording —
  made that same record **12/12 correct**. That is the mechanism proof.
- **It is a faithfulness LIMIT, not a defect the anchor can report.** The offline replay driver gets
  it right every time because it replays the ORDERED command log including the `forcelose`; two
  poke-env players driven concurrently cannot reproduce that ordering.
- ⚠️ **Lowering the stall threshold to force the class did NOT reproduce it** — 384 battles at
  threshold 25, 381 of them forfeit-terminated, **0 refusals** — and reading that as a clearance is
  what nearly closed this investigation early. A turn-25 board is not a turn-250 Struggle endgame;
  making a rare event common changed the thing that decides it. Force the *condition*, then confirm
  on the real one.
- 🔴 **TASK, not fixed: the ROLLOUT path inherits the same asymmetry.** A label's rollouts play both
  sides with `RLPlayer`s that both stall-forfeit, whereas the recorded training battle had only the
  trainee forfeiting — so a rollout reaching the 250-turn cap can be scored a WIN purely because the
  opponent's forfeit landed first, biasing the labels of long games upward. The anchor exclusion does
  not touch it.

**And the whole path is NODE-ONLY** — the composition test passes with `POKESIM_SIM_BRIDGE_BIN` and
`POKESIM_SEARCH_DRIVER_BIN` pointed at nonexistent files, so no `src/rust_sim` binary, stale or
otherwise, participates in it. The stale-main-binary trap is not in play here.

**A second, stricter check ships with it, and it is honest about never having fired.**
`replay_counterfactual` now returns `script_exhausted` — the sides that ran OUT of recorded commands
and finished on the live policy, which a `divergence_turn=None` full replay can only do after
diverging — and the anchor refuses on it even when the winner happens to match. It exists because
the fallback policy is random, so a script desync only flips the WINNER about half the time. It did
NOT catch the forfeit class above (that race consumes no script at all), and it was **empty on every
one of 274 instrumented healthy replays**, so it costs a correct run nothing. Pin:
`cf_producer_test::TestAnchorRefusal::test_a_full_replay_that_RAN_OUT_of_script_fails_the_anchor_even_on_a_matching_winner`.

⚠️ **ONE latent desync found by inspection and NOT reproducible** (`install_scripted_prefix`,
`utils/bridge/counterfactual.py`): when the mask is empty the scripted player returns
`choose_default_move()` **without popping the script**, but the live player it is replaying DID
emit that `/choose default` and the bridge DID record it as a command — so the script would sit one
entry ahead for the rest of the battle. Unreachable in this fuzz (0 `default` commands in 356
gen3ou records, so the branch never fires) and left alone deliberately rather than "fixed" blind.

**Observability.** A separate process has no TensorBoard, so it prints one **heartbeat line per
cycle** and keeps `<run>/cf_producer_state.json` human-readable (indented; sampler + weights +
totals + the last heartbeat + skip reasons):

```
[cf_producer] cycle 2 | snapshot step 29,867,520 | records 1 pending / 3 done | labels 2
              (+6 total, 6/h) | anchor 1/1 | PRODUCING | load 23.6 | 9.2s
```

The trainer-side half of the contract (the `cf/*` scalars, where `cf/labels_ingested_total` going flat
was what a dead producer looked like) is DELETED with the training half.

**Two guards on running beside a live trainer** (still how the producer behaves beside an old-code run).
`--max-labels-per-hour` (default 2000, a sliding
one-hour window) keeps it a sidecar. `--stale-checkpoint-minutes` (default 90) **pauses production**
when no NEW checkpoint has appeared for that long — the trainer is probably gone, and a producer
grinding against a frozen snapshot either burns the box filling a label buffer whose rows would expire, or
teaches the current policy an ancestor's values. It keeps WATCHING (a restarted trainer resumes it)
and announces itself exactly once in each direction. `--lag-warn-steps` (default 150 000, the deleted
buffer's staleness bound) warns once when the snapshot in hand falls that far behind the
newest checkpoint.

**`obs_materializer.scan_record`** is the new read primitive under it. An eval trace ships its obs
and action indices in `states.npz`; a training record ships **neither** — only the seed, the teams
and the committed choice strings — so the only route to a training decision's observation is to
replay the one-sided protocol AND recover the action history by inverting those choices through the
real mapper. `scan_record` does both in ONE replay and returns `RecordDecision(index, turn, action,
choice, mask, obs)` rows. It shares `_InvertingReplayPlayer` with `infer_action_indices` (which
stays track-only), and both go through one `_encode_or_track` step so the two replay players cannot
drift on the one operation where drift would silently change an obs rather than fail.
`scan_record(capture_choices=True)` additionally fills each row's `.choices` with the FULL legal
action → sim-choice-string map at that decision (`gen3_cf_q_labels_v1`, below) — asked for INSIDE
the replay it already runs, because asking afterwards costs a `materialize_from_record` prefix
replay per labelled decision. OFF by default and byte-identical off.

#### The PER-ACTION stream (`--q-labels`, `gen3_cf_q_labels_v1`) — the supply side of the (deleted) Q head

The per-action Q win-prob head (deleted, deletion pass L4; `QWinProbHead` survives only as the scorer
class the ride-along A head is built from) shipped as a **trained consumer of a stream nobody wrote**:
mode `none`, both coefficients 0, and a producer that emitted no `q_labels`. This stream closed that:
the same tight-MC rollout, once per **legal action**, on the **same dice**. It is `--no-q-labels` by default
and byte-identical off — including the dice, whose salt now routes through `cf_q_labels.q_arm_salt`
but is verbatim the string `_rollout` always used (pinned by a test, since a change there would make
every existing label file incomparable).

| flag | default | what |
|---|---|---|
| `--q-labels` / `--no-q-labels` | **OFF** | emit `q_labels` + `taken_action` on each swept row |
| `--q-top-n N` | 1 | how many of a record's `--top-n` labelled decisions get swept |
| `--q-rollouts R` | 0 = follow `--rollouts` | R per SIBLING arm |
| `--q-max-actions K` | 0 = every legal action | cap the arms per swept decision |

**THE PAIRING IS THE POINT, and it is asserted rather than remembered.** The sweep's product is a
RANKING ("is Rock Slide better than Earthquake here?"), and at R=8 the per-arm standard error is
~0.18 — so on independent dice a 0.1 gap between two siblings is invisible. Every arm therefore
takes `cf_q_labels.q_arm_seeds`, whose salt is a function of the DECISION and carries **no action
term**; `assert_paired_dice` adjudicates at the seam on the seeds each arm ACTUALLY received (never
re-derived — a check that recomputes its own input proves only that one function is deterministic).
⚠️ The pairing covers the SIM DICE only: both sides are a stochastic snapshot at temperature 1.0 and
`Categorical.sample` draws from torch's global RNG, so the policy draws are an unpaired residual. It
biases nothing (both arms draw the same policy) and cannot be closed by seeding — the arms diverge
immediately and stop drawing the same NUMBER of samples.

**The recorded action's arm is FREE, and its q-label is an IDENTITY.** The row's own `label` IS the
recorded action's counterfactual label — same salt, same R, same substituted choice — so at
`--q-rollouts == --rollouts` it is lifted verbatim and `q_labels[recorded] == label` exactly (pinned
in the unit tests AND in the `sim` composition test). At a DIFFERENT R it is re-rolled instead,
because an anchor measured over more arms than the siblings it anchors makes `q[recorded] −
q[other]` a comparison between two sample sizes.

**The selection rule is DECLARED (`cf_q_sweep_v1`, stamped on every row) for the same reason
`SAMPLER_VERSION` is.** Recorded action first; the rest in a **deterministic decision-keyed
shuffle**, truncated by `--q-max-actions`. Both obvious orders are wrong here: descending policy
probability rebuilds the on-policy starvation the head exists to escape (probe L: median p=0.002 on
the better-ranked alternative), and action index is a systematic preference for SWITCHES, since the
space is `[switch x6, move x4, struggle]` and a prefix of it is all switches. `K=0` sweeps
everything and has no bias to declare.

🚨 **COST MULTIPLIES BY THE ARM COUNT — meter it, do not estimate it.** A swept decision costs R
rollouts per legal action instead of R total. `--max-labels-per-hour` therefore counts every
per-action arm it actually ROLLS (the reused recorded arm costs nothing, so it does not), which
keeps the cap a **cost** cap rather than letting the sweep silently multiply a sidecar's box load by
its arm count. The state file and the heartbeat carry `q_rows`,
`q_entries_total`, `q_arms_rolled`, `q_arms_reused`, `q_rollouts_total`, `q_wall_seconds` and a
separate `q_skip_reasons` (a lost ARM is not a lost RECORD, so it never touches `records_skipped`);
`(q_arms_rolled + q_arms_reused) / q_rows` **is** the measured multiplier. A sweep that exhausts the
throttle mid-decision ships a SHORT block rather than a broken one — every entry in it is a real
measurement and a reader masks the rest.

**MEASURED 2026-08-29** — 90 `cf_records` of `ai_v9_72_R3SELF_0828` against **its own v107
checkpoint** (of the 37 archived runs holding `cf_records`, the only one current code can still
load), CPU, `--impl rust`, `nice -n 15` beside a live trainer at load ~27-33, compiled extractor at
9.3×, `--rollouts 4 --top-n 1 --q-top-n 1 --q-rollouts 4 --q-max-actions 0`:

| producer | | the (deleted) `CfLabelBuffer`'s read of the same rows | |
|---|---|---|---|
| records / skipped | 90 / **0** | ingested / skipped | 90 / **0** (0 field skips) |
| **arms per row** | **7.70** (3-9) | `cf/q_label_coverage` | **1.0000** |
| arms rolled / free | 603 / 90 | `cf/q_labels_per_row` | **7.70** |
| **throughput** | **1.98 s/entry** | labelled (s,a) cells | **693 / 990 = 70.0%** |
| `q[recorded] == label` | **90/90** exactly | sweep wall / cycle | 1 375 s of 1 619 s |

Folded through the real loss kernel on that batch, `q_masked_binomial_nll` reads **0.693147 = log 2
to six places** at a zero-init head (the P = 0.5 prior it must be) and 0.4218 fitted, with the
gradient on every UNSWEPT cell **exactly 0.0** — the masked form's whole safety property, measured
rather than asserted. So the cost reads as **~7.7× a plain label**: ~15 s of sweep per row against
~2 s for the row's own. Full record: `designs/CHANGELOG.md` → *The PER-ACTION LABEL FACTORY*.

**The arithmetic lives in `cf_q_labels.py`, not in the producer** — pure, so the pairing rule, the
selection rule and the wire shape are testable without a simulator. `q_labels` is a **LIST OF
OBJECTS** each naming its own action index (never parallel arrays — the rule the deleted `cf_label_buffer`
wrote down), it rides the SAME row as the per-state label (the deleted buffer deduped on the obs digest, so a
second row for one state would collide), and it is **additive-optional at schema v1**: the sweep may
never bump `schema`, which was a REFUSAL gate on the deleted buffer, so a v2 row would have been unreadable by every existing
trainer. An arm whose rollouts ALL failed is OMITTED rather than shipped at `n_rollouts: 0`, because
the deleted consumer built its mask from PRESENCE and a zero-evidence entry would mask ON a cell whose
target is the `0.0` fallback — a confident loss for an action nobody measured. `taken_action`
travels with `q_labels` — the reader-facing name for the index the row already carried as
`recorded_action`, and deliberately not given its own flag, so nothing is offered the
on-policy-only regime the stream exists to escape.

**Tests.** `cf_q_labels_test.py` (pure: the salt has no action term and is byte-identical to the
historical one, a smaller R is a PREFIX of a larger one, `assert_paired_dice` raises on divergent
AND on merely shorter lists, the recorded action always survives a cap, a capped sweep does not
prefer switches — measured over 400 decisions, with the index-ordered rule pinned as the
counterfactual it fails — and the wire shape incl. the zero-evidence drop).
`cf_producer_test.py` (pure: the state file's claim-before-work order and its bounded processed
set, the producer/retention race — `TestProducerRetentionRace` deletes a record mid-cycle and
asserts a counted skip, newest-first order, that a preloaded record survives its file, and that a
vanished anchor record is not an anchor FAILURE — the throttle
and its sliding window, the stale-trainer pause + resume, the anchor's refusal / cadence /
crash-is-a-failure, the q-sweep's pairing / content / budget knobs / cost meter / schema,
and that every help string renders), plus `TestRolloutArms` (arms aggregate to the same label
sequential or overlapped, one dead arm costs that arm alone and the arms after it still run, every
arm gets its own players and its own dice, all-arms-dead is a skip rather than a phantom 0.0).
**Three sibling files hold the tests that moved with the functions they cover** and reach every
subject through `cf_producer`'s re-exports as `P.<name>`, which is what proves the move changed
nothing a caller can see: `cf_producer_sampler_test.py` (the priority arithmetic incl. the entropy
normalization and the tie rule, the move-round filter), `cf_producer_snapshot_test.py` (checkpoint
resolution incl. a newer FORCED save outranking an older periodic one, plus the THROUGHPUT
contract — a compiled snapshot scores one row at a time, an eager one keeps the batched forward,
the mask reaches the graph as `float32` either way, the chunking changes not a single number, and
the warm-up forwards BOTH obs keys and survives raising), and `cf_producer_labels_test.py` (the v1
key set, the ecology field on every row, the optional `mc_return` / `q_labels` streams, and the
writer's never-reuse-a-name rule).

**The EXTRACTION-PARITY GOLDEN stays in `cf_producer_test.py`**, beside the fixtures, and it is
what made the cut an extraction rather than a rewrite: every public entry point of the module —
the four sampler functions, checkpoint resolution, `Snapshot.score` over a recording stub, the
state file's whole round trip, the label row across its optional streams, the batch file, the
outcome scalars, the anchor predicates and both refusal TEXTS, and the CLI's declared defaults —
run on one synthetic fixture, canonically JSON-serialised and pinned by sha256
(`121fe201…`, a 7,278-byte blob), captured BEFORE the move and reproduced byte-for-byte after it.
Two fields are deliberately excluded and each says why in place: the row's `created_unix` (a
`time.time()` default) and the anchor's TIMEOUT tail (it appends `describe_contention()`, which
reads the live load average — only the self-diagnosing prefix is pinned). A dozen named values sit
beside the digest so a failure names the entry point instead of a hash. The parser's defaults are
IN the blob on purpose: a silently changed default is exactly the class it exists to catch, so a
new flag legitimately moves it and the regeneration note says so.
**The deliverable is `cf_producer_integration_test.py` (`sim`)**: a REAL bridge battle → its
reconstruction record laid down in a ring directory in the shape the (deleted) TRAINING tap wrote
(**`trainee_username` stripped**) → ONE REAL producer cycle → assertions on the label files it wrote:
every row in the shared v1 schema, digests verifying, correct `policy_step`, and — the strongest
assertion in the file — that the obs the producer *materialized* is **bit-identical** to the obs the
LIVE player encoded, which is the only thing that proves the inverted action history did not desync
the encoder's trackers. (Until deletion pass L4 the same test fed the files to the real
`CfLabelBuffer`; both halves of that two-process contract had unit tests when its last two contract
bugs shipped, and neither ever ran the other half's real output, which is why the composition test
existed. The consumer half is deleted, so the producer's rows are now asserted directly.)

The PER-ACTION stream is covered at both altitudes. Unit (`cf_producer_test.py`): OFF leaves the row's
key set and its DICE byte-identical; the sibling arms demonstrably receive one seed list; a producer
that derives seeds per action RAISES (the regression expressed as the bug); the check reads the base
arm's OBSERVED seeds; `q_labels[recorded] == label` and the recorded arm is not rolled twice; each
budget knob bites; the cost meter round-trips through the state file; and each per-action label counts
against the throttle. `sim`: `test_the_PER_ACTION_stream_composes_ring_to_labels` runs the whole thing
on a real battle and asserts the wire shape of the per-action block, and
`test_scan_record_recovers_the_FULL_choice_map_at_every_decision` checks the capture against the one
entry known independently — the map's value at the RECOVERED action index must be the string the
side actually committed, since a wrong map is a silently MISLABELLED action rather than an error.

## Prefix-sharing materialization (`obs_materializer.materialize_branches`)

K counterfactual arms of one decision share an identical prefix, and the materializer used to
replay it from turn 1 for **every** arm — the measured bottleneck of the counterfactual label path
(`arm_ms = 4.78 + 0.853·turn`, of which prefix replay is `2.53 + 0.855·turn`; the branched turn is
~0.5 ms and the obs encode ~1.8 ms). `materialize_branches` replays the prefix once, snapshots the
player's whole battle/tracker state at the branch decision, and restores it per arm.

- **Contract: exactly equivalent to per-arm `materialize_decisions`, bit-for-bit.** Measured on 6
  gen-17 eval battles / 59 decisions / 452 arms: **59/59 byte-identical**, **15.4 → 5.3 ms per arm
  (2.91×)**, rising with the branch turn (3.7–3.9× at turn 26–28) because the part it removes is the
  part that is linear in the turn. Gate: `obs_materializer_branch_integration_test.py`, which
  compares EVERY arm rather than a sample.
- The clone SHARES append-only immutable records (`BattleEvent`, `BattleContext`) instead of copying
  them — a **contract, not an inference**, and the reason the gate compares every arm: a broken
  contract shows up as arm 2+ reading history arm 1 mutated.
- **The per-arm RESTORE is serialized ONCE and rebuilt per arm, not deep-copied per arm**
  (`_PlayerSnapshot._freeze`, 2026-08-23). Once the prefix is shared, `restore` becomes the single
  largest cost in the loop: measured on a live search-dividend oracle decision it was **3.69 ms of
  the materializer's 6.45 ms per arm — 57% of it**, because a restore is three `deepcopy`
  traversals of the battle graph and deepcopy re-walks and re-dispatches every node every time.
  Pickling each master once at snapshot time and `loads`-ing per arm measures **1.98 → 0.22 ms
  (9.1×)** on the same graph against a one-off 0.66 ms to freeze. Equivalence rests on three
  things: **three separate blobs** (one per structure, reproducing the three independent memos —
  a single blob would ALIAS the 12 objects reachable from both `battles` and `trackers`); **pins
  honoured via `persistent_id`**, so a `Logger` / `MappingProxyType` / immutable record comes back
  as itself; and `GenData` added to the pin set, because it declares itself a singleton with
  `__deepcopy__` and pickle honours no such hook. A graph that will not pickle **falls back to
  deepcopy and says so once on stderr** — a 9× regression nothing mentions is the failure shape
  this tree keeps eating. Gates: the every-arm bit-identity test above, plus
  `obs_materializer_test.py` for the graph contract.
- `lookahead` uses it for its whole `(candidate × seed)` sweep.

