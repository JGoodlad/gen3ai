# Training — cf grounding

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.** *Prefix-sharing materialization* was added
in the **2026-09-08** second pass.

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
| `<out>/cf_labels/labels_<producer>_<step>.jsonl` | label rows in the **shared v1 schema** — the contract a training-side consumer reads |

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

**The EVIDENTIAL read — the pre-registered meter for `--cf-evidential`, and the reader it was
missing.** The Beta head reads the same `value_pooled` as the scalar one, so it **cannot remove** the
blur G0 measured; the only success available to it is *confessing* it — wide exactly where the states
behind a confidence bin disagree. So the meter is not the loss but
**`width_vs_blur_spearman`**: the rank correlation, ACROSS STRATA, between the head's mean epistemic
width and the measured `sd_true_excess`. When the audited checkpoint carries a `cf_evid_head`,
`cf_audit` forwards it over the labelled states and the resolution table gains `evid_width_mean` /
`evid_precision_mean` columns beside each decile's `sd_true_excess`.

- **Rank, not Pearson** — the claim is an ordering ("wider where blurrier") and the two quantities
  are not on a common scale (a Beta's std vs the within-cell sd of an R-rollout mean).
- **The CI is a bootstrap over BATTLES**, and each draw rebuilds the strata from scratch through the
  same `resolution_cells` the point estimate uses. A draw that loses a thin decile to the minimum-n
  floor is dropped and reported as `draws_usable` — a CI whose resamples ran different arithmetic
  from its point estimate is a CI of nothing.
- **A FLAT width scores `None`, never 0.** "Wide everywhere" and "width unrelated to blur" are the
  same null in outcome but different findings in diagnosis, and the flat-to-1-ulp case (a weighted
  average of a constant) otherwise falls through to a `corrcoef` that divides by ~1e-17 and reports
  a confident correlation of float noise.
- **A checkpoint without the head OMITS the columns** and prints a one-line note. Zeros would render
  "this run has no head" identically to "this head claims no uncertainty". The read is
  **best-effort** throughout: the audit's products are the labels and the bias map, so a model that
  will not load (architecture drift — measured **2026-08-13: 79 of 79 archived runs**; the tree
  carries 100 checkpoint-bearing runs as of 2026-08-23 and the 0-of-N has not been re-measured)
  costs the run its evidential columns and nothing else. `accounting.evidential_scored` says how
  many states were scored.
- Reads the head through `ProbeSession.probe_model()` → `ProbeModel.cf_evidential_batch()`. That
  method exists because the extractor forward **never calls the head**, so unlike `win_prob_at` there
  is no stash to read: it forwards the extractor and applies the head to `stash.value_pooled`
  itself — the same thing `_cf_evidential_term` does, which is what makes the offline number
  comparable with the live `cf/evid_*` scalars.

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
so a consumer can verify the row it loaded is the row that was labelled.

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
bridge battle it plays itself, run end to end at R=2 — including the anchor refusal).
## Counterfactual win-prob grounding (`--cf-records` / `--cf-winprob-coef`, `gen3_cf_label_plumbing_v1`)

The **trainer-side plumbing** for `designs/ai_v10/design_counterfactual_value_grounding.md` — its gate
**G3**, which is explicitly "tap + buffer + flags at coefficient zero, byte-identity gated". Rung **R1**
only: tight Monte-Carlo P(win) labels, delivered to the **win-prob head**. The label PRODUCER is a
separate, out-of-process program (`cf_producer.py`, § *The label PRODUCER DRIVER* below);
**nothing in this section produces a label**, and the two halves share only a file format.

**Why the win-prob head and why head-only first.** The G0 bias map (ledger 2026-08-22; 2,204 tight-MC
labels) found the head's defect is **RESOLUTION, not an optimism offset** — population-mean gaps are
|0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36, 80–95% of it real
state-to-state variance. Only tight-MC labels carry that within-bin separation; a single realized
outcome (what the on-policy BCE eats today) structurally cannot. The head is MC-native, so R1 needs no
route change and owes no C4 gate. `--cf-head-only` defaults **TRUE** because the safe stage comes first:
the term trains the head's own params and provably cannot perturb the trunk.

**The four pieces:**

- **The record TAP (`cf_records.py`, `--cf-records`, default OFF).** The bridge emits a `__RECON__`
  reconstruction record at the end of every episode; **training discards it** (`BridgeSession` keeps a
  single overwritten slot), which is precisely why a label producer cannot reach a training decision.
  `--cf-records` threads a `recon_sink` callable into `attach_bridge_transport`, and each env worker
  writes the record into `<run_dir>/cf_records/` as a **count-capped ring** (`--cf-records-keep`,
  default 512). Crash-safe (`.tmp` + `os.replace`), filenames sort chronologically
  (`<time_ns>_<pid>_<tag>_reconstruction.json`) so the prune needs no `stat`, and the cap is **GLOBAL** —
  every worker prunes the shared dir and a lost delete race is swallowed, which is what keeps the bound
  across `n_envs` AND across launcher restarts. **The cap only bounds the directory because the `.tmp`
  is bounded too**: `prune` matches on `RECON_SUFFIX`, so a `<...>_reconstruction.json.tmp` is invisible
  to it — a failed write therefore unlinks its own tmp, and `prune` additionally sweeps tmps OLDER than
  the oldest kept record (a crash between `open` and `os.replace` cannot unlink its own; a tmp being
  filled right now is newer than every record on disk, so the sweep can never race a writer). Without
  that, the full disk this module promises to survive leaked one file per episode per worker, forever.
  **The automatic prune is THROTTLED to one write in `prune_every` (16).** It is a full `readdir`
  running on the bridge reader's coroutine — the path every env step waits behind — and pruning per
  write paid that scan ~512 times to delete ~512 files. The price is a bounded transient overshoot
  (≤ `prune_every` unpruned writes per live writer) and it degrades gracefully **because the cap is
  global**: every writer sweeps the WHOLE directory, so one worker's next sweep collects every other
  worker's backlog, and a process that dies mid-backlog has its leftovers collected by the next
  one's first sweep. Bound: `keep + prune_every·n_writers` transiently, `keep` again the moment any
  writer sweeps. `prune()` itself is unthrottled (a caller that wants the cap now can have it).
  The artifact shape is byte-for-byte the one
  `reconstruction._write_artifact` writes, so `ReconstructionRecord.load()` reads a ring file directly.
  A write failure warns once and is swallowed — a full disk must not crash a run. `--cf-records`
  without `--use-bridge` is REFUSED (a websocket run emits no such frame; the flag would be a silent
  no-op).
- **The LABEL BUFFER (`cf_label_buffer.py`).** Watches `<run_dir>/cf_labels/labels_*.jsonl`, remembering a
  per-file byte OFFSET so an appending producer is read incrementally and a partial trailing line waits
  for the next poll instead of counting as malformed. **The offset is keyed on `(name, inode)`, and the
  map is pruned to the files still on disk** — a producer that DELETES and RECREATES `labels_x.jsonl`
  (an in-place rotation) gets a new inode, and keying on the name alone made the buffer seek past the
  new file's first `offset` bytes and drop those rows with no counter and no warning. "Never a silent
  accept" has a mirror: never a silent DROP. Schema v1 is in the module docstring; obs resolve
  `obs_inline` > `obs_npz` > skip. **Everything unexpected is a COUNTED skip, never a crash and never a
  silent accept**: unknown `schema`, unknown `kind`, malformed JSON, out-of-range label, unresolvable
  obs, an obs whose width ≠ this run's, and an `obs_sha1` that disagrees with its own bytes (the GIGO
  guard — it warns loudly once). `obs_npz` resolves `<path>::<key>` and **`decision_idx` selects the
  ROW** of a 2-D array (which is what `cf_audit` emits by default, one battle's whole obs matrix per
  row) through a small per-file LRU, so N rows of a battle open the archive once instead of N times.
  FIFO at `capacity`, and **staleness expiry** at `--cf-label-lag-steps`
  (default 150 000 ≈ one production PPO iteration): `age == bound` survives, `age == bound + 1` does not,
  enforced at ingest AND on every poll. `0` disables expiry.
- **The label-QUALITY trio (task #28), landed before the coefficient ever goes live.** At coefficient
  zero none of these costs anything; the moment the term is on, each is a silent change to what the
  critic is taught.
  - **DEDUP on the obs digest, keep-NEWEST.** A producer that re-labels a decision it already shipped
    (an overlapping cycle, a re-run over the same trace tree, a truncate-and-rewrite) would give that
    one state N× the weight of every other — a change to the sampler's declared distribution with no
    flag and no counter, which design decision-of-record 3 forbids. The resident row is REPLACED, not
    appended beside. Keep-newest because a fresher label is a strictly better estimate of the same
    state (measured under a policy closer to the consumer, and carrying more evidence if R grew), and
    the replacement re-enters at the FIFO tail rather than inheriting the old row's position.
    `cf/labels_replaced_total`. Measured before: a 5-row file rewritten in place left fill **6**.
  - **SYMMETRIC staleness** — the bound is on `abs(current_step − policy_step)`. A crash-restart
    resumes from the last checkpoint, so `num_timesteps` moves BACKWARDS while the label files still
    carry pre-crash steps; under a one-sided test those rows are **immortal** and quietly become the
    whole buffer. Live tell, measured: `cf/label_age_steps_p50` reading **−4,999,000**. Future rows
    expire like stale ones, are counted separately (`cf/labels_future_total`) and trip a one-time
    loud warning naming the cause — a negative age is a diagnosis, not noise.
  - **The ObservationDebugger is SUPPRESSED around the CF forward** (`--no-compile-trainer` runs, the
    only ones that still have it). The CF rows are recorded FOREIGN states — other episodes, other
    policy steps, read off disk — and the debugger's premise is "this is the board we are about to
    act on"; it was being handed 256 replayed rows per minibatch and reporting their integrity
    against the live env's expectations. `Gen3FeaturesExtractor.suppress_observation_debugger()` is a
    context manager that restores on the way out (including on an exception) — deliberately NOT
    `disable_observation_debugger()`, which is permanent and is the compile path's trade.
- **The LOSS (`instrumented_ppo._cf_winprob_term`).** Per minibatch — the `_td_aux_term` / search-teacher /
  OPD shape, and for the same reason: the labelled states are recorded PAST decisions, absent from this
  rollout, so they cannot ride `rollout_data`, and a once-per-`train()` fold would make the coefficient
  mean something different from every other aux. `_cf_sample_and_forward` samples up to `CF_SAMPLE_SIZE`
  (256) rows and runs ONE extractor forward (`{"observation": …}` is the only key the model reads); the
  term applies the win-prob head to `stash.value_pooled` — **detaching iff `cf_head_only`**. It
  The forward runs under **`no_grad` unless something downstream actually wants the graph** — the
  condition is computed exactly (`cf_head_only` OR a dead `cf_winprob_coef`), not assumed, because the
  one arm that needs it is `--no-cf-head-only` with a live coefficient, and silently dropping the
  graph there would turn the trunk-open A/B into two copies of head-only. Both heads still train
  their own params either way: `head(value_pooled)` is applied OUTSIDE the context, which is pinned
  on the parameter update rather than argued. It
  deliberately does NOT read `last_win_prob_logits`: that stash is produced under the extractor's own
  `win_prob_mode`, which governs the ON-POLICY win-prob BCE; this term's trunk exposure is a separate
  decision, and re-applying the head makes the two independent by construction. It CLOBBERS the
  minibatch's extractor stashes, so it is folded beside `_td_aux_term`, after every loss that reads one.
  The **evidential term (below) shares that ONE sample and that ONE forward** — two samples would pay
  twice for the block's whole cost and would make the two terms disagree about which states they scored.
- **The scalars.** `cf/*` is **producer liveness and is published whenever a buffer exists**, even if not
  one label ever arrived — `cf/buffer_fill`, `cf/label_age_steps_p50`, `cf/labels_ingested_total`,
  `cf/labels_expired_total`, `cf/labels_future_total`, `cf/labels_replaced_total`,
  `cf/labels_skipped_total`, plus `cf/rows_sampled` (rows the fold actually CONSUMED this `train()`,
  summed over minibatches — residency and throughput are different questions, and only the second
  goes to zero when a producer dies while its last labels are still resident). That is deliberate: an empty buffer that does not
  announce itself is this tree's oldest failure mode (the search-teacher's silent starvation), and a flat
  `labels_ingested_total` is unambiguous evidence the producer stopped, which reads completely differently
  from a rising `labels_expired_total` (a producer that is running but lagging). `train/cf_loss` +
  `train/cf_grad_share` are the TERM, only when it folded; `cf_grad_share` is lifted from the
  grad-balance probe's shared denominator (so it is comparable with `grad/policy_share`) and reads
  **exactly 0.0 under `--cf-head-only`** — that is its verification, not a defect.

**Flag class — the `td_aux_coef` class** (`gen3_cf_coef_provenance_v1`, config **v100**). All four are
**training-only** — no forward, no weight shape, not in `agents/model/flag_registry.py` (which declares
EXTRACTOR toggles, and none of these builds a module), and **never in `check_compatible`**: a frozen
eval/pool/distill opponent runs no loss at all, so gating a loss coefficient there would be a false
rejection that breaks league play. But they ARE `ModelVersion` fields, recorded for provenance and
**read back on a flagless resume** via `_resolve`.

> ⚠️ **They were the `--opd-coef` class until 2026-08-22, and the failure that bought the promotion is
> invisible by construction.** An R1 arm resumed without re-typing `--cf-winprob-coef 1.0` kept
> training, kept logging, and simply stopped applying the term it was launched to measure — no error,
> no FATAL, just a metric that goes quiet. It was strictly worse than a symmetric loss, because the
> three STRUCTURAL cf flags (`--cf-evidential` v98, `--cf-twin-heads` / `--cf-shadow-critic` v99) were
> already recorded and GATED, so a flagless resume kept the HEAD and dropped the COEFFICIENT that
> drives it. "The launcher forwards every non-launcher flag verbatim" was the old mitigation, and it
> only ever covered a launcher-managed resume — never a bare `train_rl_agent.py --model …`.
>
> The same pass found the enabling defect underneath: `--cf-evidential` / `--cf-twin-heads` /
> `--cf-shadow-critic` each HAD a `_resolve` line and each had an argparse `default=False`, and
> `_resolve` only fires on `None` — so the line was dead and the presence test that checks for it
> passed anyway. `flag_registry_test.test_cli_flags_argparse_default_is_none` is now the gate for the
> reachability half; it found three more live flags in the same state (`value_threat_inject`, ON in
> the gen-17 production config, and `opp_intent_coef`, which `opp_intent` is DERIVED from — both would
> have made a flagless resume of PRODUCTION FATAL at `check_compatible`).

`--cf-winprob-coef > 0` REQUIRES `--win-prob-mode read_only|shaping` — `none` does not build a
`WinProbHead`, so a live coefficient would fold nothing for a whole run; the parser refuses it, and the
loss independently no-ops if the head is somehow absent.

### The LIKELIHOOD: `--cf-label-likelihood {binomial,bce}` (default **`binomial`**, `gen3_cf_binomial_likelihood_v1`)

The label schema carries `label` **and** `n_rollouts`, so the row's win COUNT is recoverable —
`w = round(label · n_rollouts)` — and the flat BCE was throwing that away. A 0.75 label from 4
rollouts and a 0.75 from 16 are the same number carrying **four times the evidence**; scoring them
identically is a modelling error, not a weighting preference.

```
w = round(label·n)            NLL_i = −[ w_i·log q_i + (n_i − w_i)·log(1 − q_i) ]
term = Σ NLL_i / Σ n_i        (mean NLL per ROLLOUT)
```

- **`binomial` is the DEFAULT**, and that is a deliberate break with the usual "new option defaults
  to old behaviour" rule: `--cf-winprob-coef` has never been live in a production run, so there is
  no trained behaviour to preserve and nothing to be compatible with. `bce` stays as the explicit
  A/B arm.
- **The normalization is `Σ NLL / Σ n`**, not `Σ NLL` and not `/mean(n)`. Two properties buy it: a
  producer that changes its R does not silently change the effective coefficient, and **at `n ≡ 1`
  it reduces EXACTLY to the mean BCE** the flat path computes (a one-rollout label is already 0 or
  1, so the round is the identity and `Σn = B`). That exact agreement is pinned bit-for-bit, which
  is what makes `binomial` a strict generalisation rather than a different objective.
- Computed through `softplus` (`−log σ(z) = softplus(−z)`), stable where `log(sigmoid(·))`
  underflows. A row whose producer omitted `n_rollouts` parses as 0 and is clamped to **one**
  observation — never a divide-by-zero, never a silently dropped row.
- Training-only, the `td_aux_coef` class: no forward, no weight shape, never gated — but recorded
  (config v100) and **read back on a flagless resume**.
- `cf/n_rollouts_mean` rides beside `cf/loss` — under the binomial likelihood the loss is per
  rollout, so a producer that quietly changed R would otherwise move the loss with no visible cause.

### The EVIDENTIAL Beta head: `--cf-evidential` + `--cf-evidential-coef` / `--cf-evidential-reg` (`gen3_cf_evidential_head_v1`, v98)

**What it is for, and what it is NOT for.** G0 convicted the win-prob head of **RESOLUTION**: the
population-mean gaps are |0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36.
A point estimate cannot represent that spread at all. This head reads the same `value_pooled` and
therefore **cannot remove the blur** — it has no information the scalar head lacks. What it can do
is **CONFESS** it: emit a Beta whose width is large exactly where the states behind a confidence bin
disagree. A confessed width is actionable (the factory's priority sampler can label the states the
critic knows it cannot separate; the awareness stack can read it); a point estimate that is silently
wrong is not.

- **`CfEvidentialHead` (`agents/model/aux_value_heads.py`)** — the `WinProbHead` bottleneck widened
  from 1 logit to 2, mapped by `softplus(·) + 1` so **α, β ≥ 1**: the Beta stays UNIMODAL (α<1 puts
  mass at an endpoint, turning "uncertain" into "certain of both extremes") and the uniform
  `Beta(1,1)` is exactly reachable, so maximum ignorance is a representable state.
- **The loss is the Beta-Binomial MARGINAL likelihood** of the row's counts — `p` integrated out,
  not plugged in: `NLL = −[log B(α+w, β+n−w) − log B(α, β)]` (lgamma-based; `log C(n,w)` is dropped
  as a constant in α,β). That is the correct evidential objective for count data, and it does two
  things at once: pulls the mean toward `w/n` AND grows the precision `α+β` only as far as
  consistency across states supports. Normalized by `Σn` like the scalar term, so the two
  coefficients are in the same units (nats per rollout). Checked against
  `scipy.stats.betabinom.logpmf`, not against a re-derivation of itself.
- **`--cf-evidential-reg` (default 1e-3) is the standard evidential-overconfidence guard**:
  `KL(Beta(α,β) ‖ Beta(1,1))`, closed form via digamma/lgamma, exactly 0 at the reachable floor. It
  rides INSIDE the coefficient, so coefficient zero kills the regularizer too. Nothing in the
  likelihood bounds `α+β` on locally-consistent data, and an inflated precision makes the width —
  the entire product — meaningless.
- **ALWAYS DETACHED, with no mode to change that.** Unlike `win_prob_mode` / `value_dist_mode` there
  is no read_only/shaping split: the head feeds nothing forward, so letting it shape the trunk would
  be a training change with no consumer to justify it. `train/cf_evidential_grad_share` reads
  **exactly 0.0 by construction** — published so the contract is a live measurement, not a docstring.
- **It is not called by the extractor forward at all** (the training-side term applies it to the
  stashed `value_pooled`), and it is built **LAST** in `Gen3FeaturesExtractor.__init__`. So OFF is
  byte-identical AND **ON-at-coefficient-0 is BIT-identical in pi/vf** — a stronger claim than the
  two precedents make, and one that depends on the build order: a module inserted mid-constructor
  shifts the init RNG stream for everything after it.
- **Metrics `cf/evid_*`**: `nll`, `reg`, `alpha_mean`, `precision_mean` (α+β — the claimed
  evidence), `epistemic_std_mean` (**the headline**), `pred_mean`, `n`; plus
  `train/cf_evidential_loss` and `train/cf_evidential_grad_share`. Read `nll` and `precision_mean`
  together: a falling NLL with a runaway precision is the head buying its loss with certainty it has
  not earned. A per-decision `(α, β)` stash lands on `fe.last_cf_evidential` for a future trace
  capture; **the npz capture itself is NOT wired** (deliberately deferred). ⚠️ Note when picking that
  up: the stash is written **only by the train loop**, so wiring it through `RLPlayer` would capture
  nothing — the extractor forward never calls the head, so an honest per-decision capture has to
  CALL it at record time (as `ProbeModel.cf_evidential_batch` does) and add an npz key.
- 🔒 **THE PRE-REGISTERED READ, for the experiment that has not run yet:** the predicted Beta's
  width should **CORRELATE with the measured `sd_true_excess` per stratum** (the `cf_audit` bias
  map's meter). Wide everywhere and wide nowhere are the same null. A falling `nll` with a flat
  width-vs-`sd_true_excess` correlation is the standing learns≠helps kill, not a result.
  **That correlation now has a reader**: `cf_audit`'s `width_vs_blur_spearman` (§ *The EVIDENTIAL
  read* above) computes it with a battle-clustered bootstrap CI, so the meter is an instrument
  rather than an intention.

**Flag class — the split, and why.** `--cf-evidential` is **STRUCTURAL** and IS in
`agents/model/flag_registry.py` (v98, `cli`/`structural`): it is a `Gen3FeaturesExtractor`
constructor kwarg that builds a MODULE, which is exactly the registry's declared scope, and the
`win_prob_mode` / `value_dist_mode` precedent. It gets a `ModelVersion` field, a `check_compatible`
bool compare, a `MODEL_CONFIG_VERSION` bump to **98** with a migration defaulting pre-v98 configs
OFF, and a `snapshot.current_model_version` keyword (so a frozen eval/pool opponent's gate sees it).
**No `ARCH_SIGNATURE` bump** — optional side head, obs family unchanged, the value_dist precedent.
The gate matters more here than usual: because the head is never called by the forward, a mismatched
resume produces **no shape error anywhere**, so `check_compatible` is the only thing standing between
a flipped flag and a run that silently supervises a freshly-random head for good. The two
**coefficients** are training-only (the `td_aux_coef` class): deliberately NOT in the registry — they
are loss weights set on the model, never reaching the extractor — but RECORDED on `ModelVersion` and
`_resolve`-inherited since config **v100**, so a flagless resume cannot keep this head and drop the
coefficient that supervises it.

`--cf-evidential-coef > 0` REQUIRES `--cf-evidential`, refused at the CLI. Unlike the win-prob case
the head cannot be added later to rescue a live coefficient: it is a state_dict change, so the
mistake would cost a whole run AND FATAL the resume that tried to fix it. The `cf_labels/` directory
is created when **either** consumer is live, so an evidential-only run is not silently starved.

**Gates.** `instrumented_ppo_test.py` pins the byte-identity that G3 is: a POPULATED buffer at
`cf_winprob_coef=0` yields the same parameter update as no buffer at all (the fold is gated on the
COEFFICIENT, not the buffer), and so does a live coef with no head. The two `cf_head_only` halves are
measured on the parameter update rather than asserted about a detach call — head-only moves the head and
leaves the trunk bit-identical; `--no-cf-head-only` moves the trunk. The same file pins the binomial
likelihood's exact properties as pure-function facts (`binomial == bce` bit-for-bit at `n≡1`; the
gradient ratio is exactly `n₂/n₁`; per-rollout normalization; `w` recovery; the `n=0` degradation) and
the evidential fold's three (ON-at-coef-0 byte-identical with the head in the optimizer; a live
coefficient reaching ONLY `cf_evid_head` — trunk AND win_head bit-identical; one shared sample and one
shared forward for both terms, counted). `agents/model/cf_evidential_head_test.py` holds the head's
maths (scipy cross-check, the hand-computed uniform-Beta anchor, `KL(Beta(1,1)‖Beta(1,1)) == 0`, the
regularizer actually moving α,β toward 1, the 1/√12 std anchor), the BIT-identity of ON's pi/vf, that
the forward never calls it, and the v98 gate + both migration legs.
`cf_label_buffer_test.py` covers FIFO, the exact expiry boundary (past AND future, both inclusive),
incremental polling, the partial-line case, every skip counter, dedup keep-newest + the
rewrite-converges case, the `obs_npz` row index and its per-file cache bound, the ring's
cap/atomicity/race-tolerance, the prune throttle's declared overshoot bound, **the launcher-restart
cap across sequential processes** (the one G3 sub-claim that used to stand on construction alone),
and that `batch_tensors` carries the rollout COUNT rather than just the ratio. The CF forward's two
guards are pinned in `instrumented_ppo_test.py` on the *stashed tensor* and the *parameter update*
rather than on a `with` statement: no graph under head-only, a graph in the trunk-open arm, both
heads still receiving their own gradients under `no_grad`, and the debugger suppressed-then-restored
(including on an exception). `main/cf_flags_test.py` covers
the defaults, both `--no-` spellings, the three new refusals and `checkargs`. End-to-end: a
`--debug --steps 10000` CPU smoke with fixture labels built from REAL episode obs.

### The TWIN HEADS + the SHADOW CRITIC (`--cf-twin-heads` / `--cf-shadow-critic`, `gen3_cf_twin_heads_v1`, v99)

**The owner-authorized amendment to the SIGNED R1 pre-registration** (ledger 2026-08-22 evening,
"Three owner sign-offs" item 3). It changes what the arm's primary comparison *is*, so read this
before reading the runbook's §2.

**The problem it solves.** R1 as signed compared two RUNS — an arm with `--cf-winprob-coef` and a
control without. Two runs differ in every random draw they ever make, and the primary meter carries
a MEASURED floor of ~39% of its own variance (`tmp/hidden_info_floor_report.md`). So a cross-run
difference has to clear noise the design cannot control, and a null would be uninterpretable.

**The design: three win-prob heads on ONE trunk, differing ONLY in their label stream.**

| head | module | trained by | isolates |
|---|---|---|---|
| **A** (control) | `win_head` — the EXISTING head, untouched | the on-policy single-outcome BCE, at `win_prob_coef` | — |
| **B** (coverage) | `cf_twin_head_b` | A's loss **+** the cf-labelled states with **SINGLE-OUTCOME** labels (n≡1) | **B−A = coverage/prioritization** |
| **C** (treatment) | `cf_twin_head_c` | A's loss **+** the same states with **TIGHT-MC** labels (n=R) | **C−B = pure variance reduction** |

That factorial is the mechanism split. `C−A` remains the original R1 claim; the amendment's value is
that it now decomposes. Because all three read the same `value_pooled` on the same rows in the same
minibatch, the trunk, the states, the seeds and the hidden-information floor are **identical by
construction**, not matched by design.

- **B and C are `WinProbHead` — the same class and capacity as A.** A difference of architectures
  would be a second explanation for every difference of scores, and nothing downstream would say so.
- **Head-only ALWAYS in v1.** Both twins read a DETACHED `value_pooled` in *every* term they take,
  including the on-policy mirror. So this measures the **LABEL effect on a trunk that is frozen with
  respect to them**; trunk exposure and policy transfer stay CROSS-RUN questions (runbook §0a,
  unamended). `train/cf_twin_grad_share` reads exactly 0.0 — published so the contract is a live
  measurement.
- **The mirror rides `win_prob_coef`, not `cf_twin_coef`.** All three heads must carry a
  bit-identical copy of the control objective, or B−A would confound "extra states" with "a
  different base objective".
- **B and C pull EQUALLY HARD.** `_cf_binomial_nll` normalizes by `Σn`, so a row's gradient is
  `(q − target)/B` whatever its n. B's n≡1 rows and C's n=R rows therefore differ only in the
  TARGET — which is what makes C−B a read of label PRECISION rather than of effective learning rate.
- ⚠️ **`cf/twin_b_coverage` is the FIRST thing to read.** A producer shipping no `outcome_label`
  trains B on nothing; B then equals A, the pre-registered C−B contrast silently becomes C−A, and
  every other counter reads healthy. That is the one way this arm produces a confident wrong answer.
  B's fold is skipped rather than trained on a zero-filled absent label, and the scalar says so.

**The SHADOW CRITIC** is the other half and a different job: a passive `ShadowValueHead` trained on
**`mc_return`** labels — the mean realized **shaped return** over the producer's rollouts, in the
units the live critic V actually predicts. It **never computes an advantage, never enters GAE,
feeds nothing forward, and reads `value_pooled.detach()` unconditionally** (the `pubval` structural
precedent). Swapping the live critic for an MC-grounded one is critic SURGERY and owes the C4
offline gate; this head is the **staged promotion path** that earns or refuses that gate without
risking a run.

- **The frame.** Under PopArt the head's raw output IS the normalized value and the target is
  `popart.normalize(mc_return)` — `_value_distill_mse`'s handling, for its reason (the coefficient
  stays scale-comparable with the value loss). Every reported metric is DE-normalized to real
  shaped-return units, which is the only frame a reader can interpret.
- 🔒 **THE METER is `cf/shadow_shadow_vs_live_v`** — the SIGNED real-unit mean of (shadow − live V)
  on the same states, with the live V taken off the *same* forward through `policy._critic_value`
  (never a hand-rolled `value_net` call, which under `--value-from-dist` reads a head the run does
  not use). A shadow sitting systematically BELOW the live critic is a live critic that is
  optimistic about the states the factory samples, **measured against ground truth rather than
  argued from a calibration curve**. `cf/shadow_live_v_vs_label` is its direct half; read them
  together, because the shadow is itself a fitted head and can be wrong too.

**The LABEL SCHEMA decision, and why it is not a version bump.** The three streams ride ONE row
(`outcome_label`, `mc_return` + `mc_return_n`, `reward_sha1` as additive-optional v1 fields) rather
than arriving as separate `kind`s. Two reasons, the first decisive: **`CfLabelBuffer` dedups on the
obs digest**, so a second row for the same state would collide and one would silently replace the
other. And one-row-per-state makes "heads B and C saw identical states" *structural* rather than
hoped-for. `schema` stays **1** because it is a REFUSAL gate — a consumer skips every row whose
version it does not know — so bumping it would make a new producer's output unreadable by an
existing trainer, which is the opposite of backward compatible. Old consumers ignore the new keys;
new consumers supervise nothing extra when they are absent.

**`mc_return` carries a REWARD DIGEST and is REFUSED on a mismatch.** A shaped return is a fact
about a board *under a reward composition*, so a return measured under a different `RewardConfig`
is a measurement of a **different value function**, not a noisier sample of ours — and there is no
shape error or range violation that would catch it. `reward_config_digest(config)` (a stable sha1
over every `RewardConfig` field) is stamped by the producer and handed to the buffer by the
trainer; a mismatch drops the **field** (never the row — its win-prob labels are still good), counts
`cf/labels_mc_return_rejected_total`, and warns once by name. The digest is only passed when
`--cf-shadow-coef > 0`: a run with no shadow head must not reject rows over a field it does not read.

**The producer side** (`cf_producer.py`): `outcome_label` is free (it already computes the recorded
outcome for the critic-surprise term). `mc_return` needs the server-free reward path —
`agents/training/cf_mc_return.py` wraps `RewardTracker`, keeps the per-turn rewards *in order*, and
folds them with `--gamma`. Two non-obvious facts live there: **`RewardTracker` accumulates an
UNDISCOUNTED total** (a return is `Σγᵏr` from a particular state, so the rewards must be captured
per turn), and **the divergence turn's own move is SCRIPTED**, so a tracker hooked only into the
live `choose_move` would begin at T+1 and its return would be missing `r_T` and carry an extra
factor of γ — against the very state the label is FOR. That is why `install_scripted_prefix` grew an
`on_scripted_decision` hook (default None, byte-identical): it REPORTS, and the producer's closure
decides. The reward config is read from the run's own `metadata.json` `cli_args` through the SAME
`RewardConfig.from_args` the trainer uses; when it cannot be read the default is used and the fact
is printed LOUDLY, because the digest will then simply not match and the trainer will say so.

**⚠️ TWO SEAM BUGS shipped in the first version of the `mc_return` path and were caught by
adversarial review, not by the tests.** Both produced plausible-looking labels; keep them in mind
before moving either seam.

1. **`action_to_order` is NOT a valid recording seam.** It looks ideal (the commit point; it raises
   `StaleDecisionError` on a superseded attempt) — but `counterfactual._invert_choice` calls it in a
   **LOOP over every legal index** to recover a recorded choice's action number, on every scripted
   decision of the prefix. Recording there fired 6-9 times per scripted turn with actions that were
   never played, each advancing the STATEFUL reward function. The seams are `_predict_best_action`
   (caches the committed `(idx, mask)`) + the player's own `choose_move` (the once-per-decision
   boundary; it must be wrapped BEFORE `install_scripted_prefix`, which captures it as its live
   delegate) + `_battle_finished_callback` (the terminal reward).
2. **The hook must `arm_at_next()` AND `note()`, in that order.** Arming alone left the first LIVE
   decision at T+1 as the armed one, so `r_T` was dropped and every label was `G(s_{T+1})` against an
   obs row for `s_T` — biased by whatever happened on the divergence turn (a KO there is the largest
   single shaping term), i.e. **correlated with the state and shaped exactly like a real signal**.

Both are pinned in `cf_mc_return_test.py`, the second with an explicit negative control showing the
buggy shape, because neither is visible in any scalar. Note what did NOT catch them: the
bridge-backed composition test asserted only that an `mc_return` was PRESENT. **A composition test
that checks presence rather than value is a presence test.**

**Two counters, not one, and the distinction is the same one twice.** `cf/labels_skipped_total` is
the ROW-level GIGO meter and must keep partitioning the input with `labels_ingested_total`; an
optional FIELD that is malformed or out of range ACCEPTS the row and counts into
`cf/labels_field_skipped_total`, and a reward-digest refusal counts into
`cf/labels_mc_return_rejected_total`. Folding any of these into the first would make "is the
producer feeding me garbage" climb at the ingestion rate on a buffer refusing nothing.

**The discount comes from the RewardConfig, not from a flag.** `reward_config_digest` hashes every
field including `gamma`, so folding the return at `cfg.gamma` puts the discount under the same GIGO
guard as the reward. `--gamma` survives only as an explicit override, and its help says what that
costs: a mistyped value ships returns folded against a different value function with the digest
still matching and every liveness counter reading healthy.

**⚠️ The ONE coupling head-only does NOT remove: the global gradient CLIP.**
`clip_grad_norm_` scales every gradient by `max_norm / total_norm` over ALL parameters, so any term
with a non-zero gradient anywhere perturbs the policy and value updates in the last bits. It is
tiny at a sane coefficient and it is shared by every aux this tree runs — but it is not zero, and an
arm claiming a bit-identical trunk must know which of the two mechanisms it is claiming.
`instrumented_ppo_test.py::test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP`
pins the pair: with the clip active the updates differ, with it raised out of the way they are
bit-identical. A genuine gradient leak would survive both.

**Flag class.** `--cf-twin-heads` and `--cf-shadow-critic` are **STRUCTURAL**, in
`agents/model/flag_registry.py` (v99, `cli`/`structural`), with `ModelVersion` fields, bool compares
in `check_compatible`, a `MODEL_CONFIG_VERSION` bump to **99** with a setdefault-False migration,
`snapshot.current_model_version` keywords, and **no `ARCH_SIGNATURE` bump** (optional side heads,
obs family unchanged) — the `cf_evidential` precedent exactly, and the gate matters for its reason:
the forward never calls these heads, so `check_compatible` is the ONLY thing that can catch a
flipped flag. `--cf-twin-coef` / `--cf-shadow-coef` are training-only (the `td_aux_coef`
class), deliberately not in the registry but recorded and **read back on a flagless resume** since
config **v100** — a within-run paired comparison whose coefficient silently zeroed on restart would
report B−A ≈ C−B ≈ 0 and look like a null result.

Refusals, all at the CLI: `--cf-twin-coef > 0` requires `--cf-twin-heads`; `--cf-shadow-coef > 0`
requires `--cf-shadow-critic` (both are state_dict changes and cannot be added mid-run to rescue a
live coefficient); and **`--cf-twin-heads` requires `--win-prob-mode read_only|shaping`**, because
the twins mirror head A's loss and `none` builds no head A — the arm's control arm would silently
not exist.

**The AUDIT read** — `cf_audit` gained `attach_twin_heads` (one more batched forward, same
best-effort contract as `attach_evidential`) and two blocks:

- 🔒 **`twin_paired` is the amended PRIMARY.** Per row, `brier = (pred − mc)²` and
  `abs_err = |pred − mc|` for each head, **differenced across heads on the same row**, with a
  battle-clustered bootstrap CI on the difference. Two properties buy it over `sd_true_excess` here:
  the hidden-information floor **cancels exactly** (it is a property of the STATE, identical in
  every arm — the amended §2 argued it cancels at matched *step*; twins strengthen that to matched
  *state*), and no stratification means no selection correction is owed. **SIGN: these are ERROR
  scores, so a NEGATIVE difference means the first-named head is better.**
  ⚠️ A near-zero contrast with a near-zero `mean_abs_pred_diff` is a **coverage/dosage** reading,
  not the pre-registered null — the label streams did not separate the heads and there is nothing
  to decompose yet.
- **`twin_resolution`** is the G0 continuity link: each head's own `sd_true_excess` binned by its
  own prediction. Its cells are **UNWEIGHTED** and the block says so in its own `weighting` field —
  the population re-weighting is unavailable for B and C, because the eval frame carries only head
  A's predictions, so their decile membership over the whole frame is unknown. Absolute levels here
  are NOT comparable with the bias map's `population_weighted_sd_true_excess`.
- **`shadow`** carries `shadow_vs_live_v` (signed, battle-clustered) and `shadow_vs_live_v_abs`.

**Gates.** `agents/model/cf_twin_heads_test.py` holds the heads' contracts (the shadow's UNBOUNDED
range — a sigmoid creeping in would clamp every label while the MSE fell; the twins' identical
architecture; their INDEPENDENT init, so `cf/twin_b_vs_c_abs` at step 0 is not reading its own
initialization; the BIT-identity of ON's pi/vf for each flag and both together; that the forward
never calls any of them; the v99 gate on both flags and both migration legs; the registry rows; the
`current_model_version` threading). `instrumented_ppo_test.py` pins the routing and the isolation:
coefficient-zero byte-identity for each half, a live coefficient reaching ONLY its own heads (with
the clip raised — see above), **the ROUTING pin** (B's loss equals the binomial NLL of the OUTCOME
and demonstrably NOT of the tight-MC label, with the two set to opposite extremes), B's n≡1
weighting, B's skip-and-count when no row carries an outcome, the mirror's coefficient and its
detach, the shadow's PopArt frame and masking, and that all FOUR cf terms share ONE sample and ONE
forward. `cf_label_buffer_test.py` covers both schema directions (an old row still ingests; a new
row carries both streams), the out-of-range field skip that keeps the row, the reward-digest
refusal and its counter, the coverage scalars, and the masks in `batch_tensors`.
`cf_mc_return_test.py` pins the oldest-first discount and the deliberate one-decision arming delay.
`cf_audit_test.py` pins the sign convention, the honest null, the refusal to compare heads it does
not have, and the shadow block's signedness. `main/cf_flags_test.py` covers the defaults, both
negation forms, the four refusals and `checkargs`. **The composition** is
`cf_producer_integration_test.py` (`sim`): a real bridge battle → the ring → one producer cycle →
the REAL `CfLabelBuffer`, now additionally asserting every row carries a valid `outcome_label`,
that at least one carries an `mc_return` with its digest, that the buffer keeps them, and that a
buffer configured with a FOREIGN digest refuses the `mc_return` while keeping the row. A second
test in that file covers the **MULTI-CYCLE** seam the first one holds fixed: a checkpoint lands
between cycles, the producer reloads it and RE-STAMPS the rows, and the real buffer holds the two
vintages at their two different ages — plus a poisoned row (`obs_sha1` disagreeing with its own
bytes) costing exactly itself beside the good ones. That is the leg the 2026-08-23 R1 composition
smoke found a live defect in.

### The PER-ACTION Q WIN-PROB HEAD (`--q-winprob-mode` + `--q-winprob-coef` / `--q-winprob-onpolicy-coef`, `gen3_q_winprob_head_v1`, v107)

**The problem, stated as a cost.** Every value readout this tree owns evaluates a STATE, so a
per-action win probability is not a read — it is eleven simulator re-rolls plus eleven forwards,
because the successors have to be manufactured first. That is exactly why probe L's ranking "is not
a quantity the network computes" (it is the head composed with a simulator, and PPO performs no such
composition). `QWinProbHead` amortizes the composition: one forward, eleven `P(win|s,a)`, scored
from the pointer head's own action tokens. The architecture half is in
`src/agents/model/CLAUDE.md` → `QWinProbHead`; this section is the training half.

🚨 **THE STARVATION TRAP — read this before setting either coefficient.** On-policy data labels
exactly ONE action per state, and probe L measured the policy sampling its own better-ranked
alternative at a median **p = 0.002**. A Q head trained on that stream is untrained precisely on the
never-tried moves — i.e. **confidently wrong on the entire set a per-action readout would ever be
consulted about**, because the shared scorer generalizes the taken-action signal onto the unvisited
columns with nothing to correct it. The head's primary labels are therefore COUNTERFACTUAL, from the
same R1 factory the rest of this block feeds on (ledger 229e9f1).

**THE LABEL CONTRACT** is an ADDITIVE-OPTIONAL extension of the existing v1 row — the schema version
deliberately does NOT move, for the reason stated at `cf_label_buffer`: `schema` is a REFUSAL gate,
so bumping it would make a new producer's output unreadable by an existing trainer.

```json
"q_labels": [{"action": 7, "label": 0.62, "n_rollouts": 16}, ...],   // per-ACTION counterfactual
"taken_action": 7                                                    // for the weak fallback only
```

`q_labels` is a **list of objects, never parallel arrays**. Three same-length lists can be written in
the wrong order by a producer and read as valid by the consumer; a per-action object cannot. Each
entry names its own index in the ACTION SPACE (`[switch x6, move x4, struggle]`) — the same index the
policy's logits, the action mask and the Q head's column `a` use. A malformed entry is a counted
FIELD skip (`q_labels_*`), so the row survives with its three other label streams intact: a producer
bug in one stream must not cost the trainer the rest. Duplicate actions collapse keep-LAST (two
entries for one action are the producer contradicting itself; summing them would invent evidence).

**THE LOSS** is `q_masked_binomial_nll` — the scalar cf term's likelihood restricted to the labelled
cells, normalized by `Σ(mask·n)`. Two invariances come out of that normalizer and both are pinned:
the coefficient's meaning is independent of the producer's R (an R=16 label pulls exactly 4x an R=4
one — that IS the likelihood of the data, not an emphasis choice) **and** of the minibatch's label
DENSITY. A whole-grid `Σn` would make the term shrink as coverage fell, which is the opposite of what
a starving factory should do to a loss. At full coverage the function equals
`cf_terms.cf_binomial_nll` EXACTLY, which is what makes "the same likelihood, restricted" a fact
rather than an analogy. **An unlabelled cell contributes zero to numerator and denominator alike** —
never a zero target, which is indistinguishable from a confident "this action loses".

**TWO COEFFICIENTS, AND THE SPLIT IS THE POINT.** `--q-winprob-coef` weights the counterfactual
stream. `--q-winprob-onpolicy-coef` weights the WEAK fallback — the recorded battle's realized
outcome as a single-sample label for the ONE action that was taken, at `n ≡ 1` so its per-row
gradient magnitude matches a counterfactual row's and only the TARGET differs. It defaults to **0.0**
and should usually stay there; it exists so a starved-factory run has something to show, not as a
substitute. Separate coefficients so the two can never be confused in a run's provenance, and its
metrics carry an `onpolicy_` prefix so its numbers can never be read as the grounded stream's.

**BOTH ARE HEAD-ONLY, STRUCTURALLY.** The head's inputs are detached inside the EXTRACTOR forward
(`q_winprob_mode` has no `shaping` value), so no coefficient can route a gradient into the trunk and
`grad/q_winprob_share` reads exactly 0.0 by construction — the verification, not a defect.

**⚠️ Both folds RE-APPLY the head rather than reading `last_q_winprob_logits`, and the reason is not
the same as `_cf_winprob_term`'s.** That term re-applies its head because `win_prob_mode` governs a
different decision than `cf_head_only`. This one does it because `cf_sample_and_forward` runs under
`th.no_grad()` whenever nothing downstream needs a graph — a condition computed from the *scalar*
term's settings, which know nothing about this one — so a term folded from that stash would train
**exactly nothing** while every metric looked healthy. It reads the pointer stash from the same
forward and RAISES on a batch-size disagreement (the `_critic_value` stale-stash precedent):
structurally impossible, therefore loud rather than degrading.

**METRICS — `q_winprob/*`**, its own prefix so a per-ACTION number can never be read as the per-state
win head's.

| key | read |
|---|---|
| `label_coverage` / `labels_per_row` | **FIRST.** Coverage is "is the factory running"; **`labels_per_row` is "is it running at more than one action per state"** — i.e. the number that separates a real counterfactual stream from the on-policy trickle this head exists to avoid |
| `loss` · `abs_err` · `bias` | the fit on labelled cells |
| `pred_spread` vs `label_spread` | **the discriminating pair.** A head that learned nothing per-ACTION still scores well on `abs_err` by predicting each state's mean; a `pred_spread` far below `label_spread` is a head that amortized the VALUE and not the SEARCH. Computed only over rows with ≥2 labelled actions, since a one-action row's spread is 0 by construction |
| `onpolicy_*` | the weak fallback's, and not evidence about the grounded stream |
| `train/q_winprob_loss` | **ABSENT, never 0.0, when the fold starves** — a defaulted zero is a perfect score for a head that trained on nothing |

**THE OFFLINE METER (E5 step 5)** is `python -m main.q_amortization <run_dir>`: the head's per-action
row against the prober's own one-ply `lookahead` sweep — Spearman, top-1 agreement, and the
**amortization residual**. Shrinking ⇒ the AlphaZero ratchet (search's value has migrated into the
net; search must deepen to add anything); stubbornly large on a class of states ⇒ those states
genuinely need live search, a triage signal for the ladder time manager. Two caveats live in the
script and belong here too: it is a **PREDICTIVE** meter and says nothing about whether the policy
plays better (iteration 2's lesson — keep it distinct from the behavioral dividend), and its ground
truth is **itself a model read**, since `lookahead` scores each re-rolled successor with the same
checkpoint's critic. `--self-check` runs the init-state sanity (zero-init ⇒ P = 0.5 everywhere ⇒ a
total tie) with no checkpoint, no traces and no simulator; it is gated in the suite.

**Status: LATENT.** Mode `none`, both coefficients 0.0, and **the producer does not yet emit
`q_labels`** — that is the next piece and the one that decides whether any of this measures
anything.

### The label PRODUCER DRIVER (`cf_producer.py`) — the piece that runs the loop

```bash
nohup nice -n 10 python -m agents.training.cf_producer \
    models/<run> [--rollouts 8] [--top-n 3] [--records-per-cycle 4] \
    [--max-labels-per-hour 2000] [--anchor-every 50] [--impl rust] \
    > models/<run>/cf_producer.log 2>&1 &
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

The tap rings records; the buffer consumes label rows; **this walks one to the other.** It is a
long-lived **standalone sidecar run beside a live trainer** — the `snapshot_ladder` /
`bot_matchup_matrix` pattern — and deliberately NOT auto-spawned by the trainer: producer and
consumer share only a file format (that is `cf_label_buffer`'s whole premise), and a producer the
trainer owned would make a label-path failure a *training* failure.

**Four modules, one factory.** `cf_producer.py` owns the LOOP and everything with state in it —
the cycle, the record ring's consumer side, the crash-safe `ProducerState`, the anchor, the rollout
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
v1 label ROW and its batch writer — a CONTRACT with `cf_label_buffer`, which knows nothing about
this loop — and `OPPONENT_LABEL`, i.e. THE ECOLOGY DECISION itself. The cut was 2026-09-06 (the
ratchet's third pass over the 1,000-2,000 band, **1899 → 1484 lines**; still in the band, because
the loop class alone is ~810 lines and the 1,000 TARGET is unreachable without splitting it).
`cf_producer` re-imports every public name, so `from agents.training.cf_producer import label_row`
still resolves; the private `_warm_the_compiled_graph` is the one name that does not, and its two
tests reach it through the owning module. **The extraction-parity golden below is the evidence.**

#### 🚨 THE DUTY CYCLE — the number that decides whether ANY of this works

The producer can only stamp a label with the step of the newest `checkpoints/` zip, and the buffer
expires a row more than `--cf-label-lag-steps` behind the live policy. **Those two flags define a
fraction, and until 2026-08-23 nobody computed it:**

```
duty cycle = --cf-label-lag-steps / (env steps between checkpoints)
```

The denominator is the trap. SB3's `CheckpointCallback.save_freq` **counts VEC-ENV CALLS, not env
steps** — one `_on_step` per `vec_env.step()`, which advances `n_envs` envs at once — and it was a
bare hardcoded `50000` in `main/train/callbacks.py`, read as "50k steps" by everyone including the
R1 design. At `--n-envs 48` it is **2,400,000 env steps** against a 150,000-step bound: a **6.25%**
duty cycle. Measured on the live `ai_v9_29_rev1_0823`: **6 labels ingested against 255 expired in
two hours**, with every counter on both sides reading healthy — the producer was producing, the
buffer was expiring, and neither knew the other's number.

Two things close the class:

* **`--checkpoint-every-steps <env_steps>`** (trainer) sets the cadence in the unit a reader means.
  Default `None` = the historical `50000` vec-calls, byte for byte, so a flagless resume is
  unchanged; a value is converted back by ceil-division (`main.train.constants`).
* **The launch REFUSES a duty cycle under 25%** and PRINTS it when healthy. With `--cf-records` on
  and a live `--cf-twin-coef` / `--cf-winprob-coef`, `main/train/config.py` computes it, names all
  three numbers plus both remedies, and exits `FATAL_CONFIG` (not `parser.error` — a restart would
  hit the identical config, so the launcher must give up rather than loop). `--debug` prints and
  is exempt. *A quantity nobody computes is how this shipped, so it is now printed on every launch
  that has both halves on.*

At the production shape that is `--checkpoint-every-steps 150000 --n-envs 48` → 3125 vec-calls →
a 100% duty cycle.

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
the same decisions selected with the same ranking, and the real `CfLabelBuffer` ingests 18/18 rows.
The only difference anywhere is `priority.win_prob` in the **6th decimal** (0.670049 → 0.670050) —
Inductor's arithmetic, the documented max|Δ| ~5e-7, on a field nothing thresholds.

#### The producer/retention race (`records_vanished`)

The trainer OWNS `cf_records/` — every env worker prunes it to the newest `--cf-records-keep` (512)
— and this process only reads it, so a record can be enumerated and then deleted before it is
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
  and stop the factory. The remedy is a larger `--cf-records-keep`, which a restart can raise.

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
`cf_producer_integration_test::test_a_new_checkpoint_mid_run_restamps_the_labels_and_the_buffer_takes_both`
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
`n_rollouts`** on every row (an ADDITION; the buffer reads a fixed key set and ignores the rest,
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
labels and can NEVER double-label. That direction is deliberate: the buffer dedups on the obs
digest, so a duplicate is survivable — but it is also a silent re-weighting of the declared
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

The trainer-side half of the contract is the `cf/*` scalars — `cf/labels_ingested_total` going flat
is what a dead producer looks like from over there (see the R1 runbook's launch-window table).

**Two guards on running beside a live trainer.** `--max-labels-per-hour` (default 2000, a sliding
one-hour window) keeps it a sidecar. `--stale-checkpoint-minutes` (default 90) **pauses production**
when no NEW checkpoint has appeared for that long — the trainer is probably gone, and a producer
grinding against a frozen snapshot either burns the box filling a buffer whose rows will expire, or
teaches the current policy an ancestor's values. It keeps WATCHING (a restarted trainer resumes it)
and announces itself exactly once in each direction. `--lag-warn-steps` (default 150 000, matching
the buffer's `DEFAULT_LAG_BOUND`) warns once when the snapshot in hand falls that far behind the
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

#### The PER-ACTION stream (`--q-labels`, `gen3_cf_q_labels_v1`) — the supply side of the Q head

The v107 `QWinProbHead` (above) shipped as a **trained consumer of a stream nobody wrote**: mode
`none`, both coefficients 0, and a producer that emitted no `q_labels`. This closes that: the same
tight-MC rollout, once per **legal action**, on the **same dice**. It is `--no-q-labels` by default
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
measurement and the consumer masks the rest.

**MEASURED 2026-08-29** — 90 `cf_records` of `ai_v9_72_R3SELF_0828` against **its own v107
checkpoint** (of the 37 archived runs holding `cf_records`, the only one current code can still
load), CPU, `--impl rust`, `nice -n 15` beside a live trainer at load ~27-33, compiled extractor at
9.3×, `--rollouts 4 --top-n 1 --q-top-n 1 --q-rollouts 4 --q-max-actions 0`:

| producer | | consumer (the REAL `CfLabelBuffer`) | |
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
OBJECTS** each naming its own action index (never parallel arrays — see `cf_label_buffer`'s
docstring), it rides the SAME row as the per-state label (the buffer dedups on the obs digest, so a
second row for one state would collide), and it is **additive-optional at schema v1**: the sweep may
never bump `schema`, which is a REFUSAL gate, so a v2 row would be unreadable by every existing
trainer. An arm whose rollouts ALL failed is OMITTED rather than shipped at `n_rollouts: 0`, because
the consumer builds its mask from PRESENCE and a zero-evidence entry would mask ON a cell whose
target is the `0.0` fallback — a confident loss for an action nobody measured. `taken_action`
travels with `q_labels` — the consumer-facing name for the index the row already carried as
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
crash-is-a-failure, the q-sweep's pairing / content / budget knobs / cost meter / schema round
trip, and that every help string renders), plus `TestRolloutArms` (arms aggregate to the same label
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
**The deliverable is `cf_producer_integration_test.py` (`sim`)**: a REAL bridge
battle → the REAL `CfRecordRing` in the TRAINING tap's shape (**`trainee_username` stripped**) →
ONE REAL producer cycle → the REAL `CfLabelBuffer`, asserting every row INGESTED with **zero skips**,
digests verifying, correct `policy_step`, and — the strongest assertion in the file — that the obs
the producer *materialized* is **bit-identical** to the obs the LIVE player encoded, which is the
only thing that proves the inverted action history did not desync the encoder's trackers. Both
halves of a two-process contract had unit tests when the last two contract bugs shipped; neither
test ever ran the other half's real output, which is why this file runs the composition.

The PER-ACTION stream is covered at both altitudes for the same reason. Unit
(`cf_producer_test.py`): OFF leaves the row's key set and its DICE byte-identical; the sibling arms
demonstrably receive one seed list; a producer that derives seeds per action RAISES (the regression
expressed as the bug); the check reads the base arm's OBSERVED seeds; `q_labels[recorded] == label`
and the recorded arm is not rolled twice; each budget knob bites; the cost meter round-trips through
the state file; each per-action label counts against the throttle; and the schema round-trips
through the real `CfLabelBuffer` — including a **deliberately shuffled** list reading identically
(the object-not-arrays property demonstrated, not asserted), a malformed entry costing the FIELD and
not the row's other three streams, and an OLD row still reading on the NEW consumer.
`sim`: `test_the_PER_ACTION_stream_composes_ring_to_buffer` runs the whole thing on a real battle
into the real buffer's per-action columns, and
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

