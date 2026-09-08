# Training — stall tail harvest

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## The STALL-TAIL HARVEST + head-repair pipeline (`main.harvest` → `winprob_finetune` → `main.harvest_meter`)

An **offline, three-stage pipeline** that manufactures win-probability labels for the population
probe O convicted, fits the win-prob head on them with the trunk frozen, and measures the result
against a battle-level holdout. It is the ai_v12 head-repair backbone and it writes nothing into
`models/` — artifacts land under `utils.paths.harvest_dir()` (repo-root `harvest/`, gitignored,
`$GEN3AI_HARVEST_DIR` overrides).

```bash
export PYTHONPATH=$PYTHONPATH:src
# 1. HARVEST — mine late-game states, score them with the subject, label by re-seed multi-rollout
python -m main.harvest --subject models/<run>/final_model.zip \
    [--states 300] [--rollouts 32] [--min-turn 60] [--workers 2] [--inline-obs] [--dry-run]
# 2. FIT — head-only, trunk frozen, binomial NLL, turn-slice re-weighted
python -m agents.training.winprob_finetune <harvest_dir> --subject models/<run>/final_model.zip
# 3. METER — probe O's battery, PAIRED, pre vs post, on the held-out battles
python -m main.harvest_meter <harvest_dir> --head <finetune_out>/head_best.pt
```

**Why it exists.** Probe O measured the head ending above 0.5 on **34.8%** of the tails of games it
loses by construction. Ledger `b63a96f` banked the account: the labels are correct, and all three
mechanisms are DATA-shaped — BCE optimizes the average and cap-game final turns are epsilon of the
buffer; strong-position-at-turn-249 barely exists in training at all. So the head's problem is a
**census** problem (discrimination mass at the late time-slices), not a propagation one — MC labels
already stamp the terminal onto every step. This pipeline manufactures the missing mass.

### The label schema is a CONTRACT (`agents/training/harvest_schema.py`)

Gzipped JSONL, eleven pinned fields plus an obs locator triple; `validate_row` runs on every write
AND every read. It is deliberately **separate from `cf_audit`'s v1 schema**, which is a single-run
bias-map contract with a live consumer (the trainer's label ring): the harvest spans many runs'
traces scored by ONE subject, carries the selection `priority` that drew each row, and pays a
binomial likelihood over `k`/`n`. Widening `cf_audit`'s schema in place would make every existing
reader tolerate columns absent from every row it has ever seen.

`load_obs` is the ONE resolver both sides call, and it verifies `obs_sha1`. That is not ceremonial:
`cf_audit` shipped a real bug where `obs_npz` rows ignored `decision_idx` and **every** default-mode
label was rejected as architecture drift, because both halves of the contract had tests and neither
ran the other's real output.

### Three measured facts that shaped the sampler — each one changed the design

1. **The subject must be RE-SCORED, always.** A trace's recorded `win_probs` came from whichever
   checkpoint played it. Measured: the subject's re-scored φ differs from the recorded column by up
   to **0.135 on that subject's own traces**. `phi_head` is a fresh forward (~2.9 ms/state batched);
   the rollouts likewise run the subject as trainee via `ProbeSession(ckpt_override=…)`, which is
   what makes `k/n` an estimate of *the subject's* value rather than of the run that recorded it.

2. **Priority alone fills the sample with doomed states.** A 40-state draw came from 4 battles, all
   losing. A head fit on nothing but doomed late states has a trivial way to score perfectly — say 0
   at every late turn — which is a BIAS, not a repair, and it would wreck the thing the head already
   does RIGHT (probe O: `LONG_WIN` reads **0.986** at 128 median turns; the failure is a right tail
   on the doomed side, not a length effect). Hence `--drag-frac` (0.6).

3. **Capping the doomed share was NOT enough.** With the remainder also ranked by priority, a
   300-state draw took **1** state from a won battle: the gap term is `|phi − realized|`, and a
   correctly-read win scores ~0 on it, so the control stratum was selected out of existence by the
   very rule that makes the doomed stratum good. Hence `--general-win-frac` (0.5). The meter's
   long-win control is only an honest test because of this.

### 🚨 THE FAILED PILOT — sample the REGION THE METER READS, or the fit gets worse in both directions

The most important thing this pipeline knows, and it was bought by running it. Pilot 1 (200 states
/ 41 battles / 6,281 rollouts, priority-ranked with no tail stratum) produced a head that was
**worse on every population**, and the long-win control is what convicted it:

| population | metric | pre | post | paired diff, CI95 | |
|---|---|---:|---:|---|---|
| held-out long losses (n=86) | `detect_le05` | 0.977 | 0.302 | −0.674 [−0.767, −0.570] | **SIG** |
| held-out long losses | `phi_T` | 0.070 | 0.607 | +0.537 [+0.492, +0.580] | **SIG** |
| LONG-WIN control (n=40) | `phi_T` | 0.943 | 0.567 | −0.376 [−0.447, −0.299] | **SIG** |

Both directions collapsed toward ≈0.6 — the head lost its dynamic range and became nearly
constant. **This is not "late means lost"** (that would have moved the two populations in opposite
directions); it is regression to the *sample* mean, and the cause is a measured distribution
mismatch:

| | fit set (harvest) | eval set (meter) |
|---|---|---|
| turn range | **60–152**, p50 90 | **96–239**, p50 118 |
| mean MC label | **0.621** — states the subject WINS | ~0 — doomed by construction |

**29.3% of the meter's eval turns were above the harvest's maximum turn.** The head was fit on
mid-game positions it wins 62% of the time, never shown a losing tail, and then asked about one.
That is extrapolation, and extrapolating from a 62%-win sample onto a 0%-win population lands on
the sample mean.

The fix is `--tail-frac` (default 0.5): a reserved share of the doomed budget goes to a battle's
last `TAIL_K = 5` decisions — `TAIL_K` matches the meter's `K_TAIL` and probe O's K **on purpose**,
because that is the region being scored. Fitting the same REGION on DIFFERENT battles is
generalization, not leakage: the battles themselves are held out by the producer. `--tail-frac 0`
ablates it back to the pilot-1 behaviour.

**The durable lesson, which generalizes past this pipeline: a label factory that never samples the
region its meter scores is extrapolating, and no amount of label quality fixes it.** Pilot 1's
labels were excellent — 31.4 adjudicated rollouts per state, 1.86% timeouts, a 0.052 noise floor.
They were labels for the wrong states. **And note what caught it**: every stall-tail metric alone
would have read this as "detection got worse", which is ambiguous with a bad fit; it was the
untouched LONG-WIN control moving the *other* way that identified the collapse as loss of dynamic
range. That is what the control is for, and it is why the meter refuses to report without one.

### 🚨 The cap-record blocker — 8 of 48, and it is the KNOWN rust `forcelose` gap

A 250-turn game ends by FORFEIT (`Gen3Env.action_to_order` returns `ForfeitBattleOrder` at
`MAX_TURNS`), logged as a `["forcelose", side]` command. **A record without it never terminates, and
the offline replay driver refuses it** — *"replayed all N commands but battle has not ended (turn
250)"*, asserted by BOTH impls in the `replay` verb that `materialize_from_record` depends on. So
every model-scored offline path is blocked on such a record, not just this one.

Measured archive-wide: **689 cap records, 543 (79%) carry the forfeit.** Within the current-arch
corpus it is **8 of 48**, and the split is *exactly* the documented boundary — the rust `sim_bridge`
pushed `commands` only in `handle_choose` until **2026-08-24** (§ *The FORFEIT class* above):

| runs | caps | replayable |
|---|---:|---:|
| `…_0819` … `…_0824` (pre-fix) | 40 | **0** |
| `…_0825` … `…_0828` (post-fix) | 8 | **8** |

So the scarcity is **transient** — every post-fix run adds replayable caps — and it is why the
doomed-tail population is caps **plus long losses** rather than caps alone (`Candidate.meter_class`).
That widening is faithful, not convenient: probe O's own framing is a heavy right tail on the doomed
side, and caps are where it concentrates (4.3× the ordinary-loss rate), not the only place it lives.
The two classes are stratified separately everywhere and the meter never pools them, because the
head reads them very differently (probe O: `detect_le05` 0.652 on caps vs 0.94–0.95 on long losses).

**The missing forfeit is NOT synthesized.** Appending `["forcelose", trainee_side]` in memory would
make all 40 replay, and the battle is a LOSS at exactly 250 turns so a trainee forfeit is
overwhelmingly likely. It is refused because "overwhelmingly likely" is not a basis on which a
LABEL FACTORY may manufacture an ending — a record may lack its terminal because the *opponent*
forfeited, and inventing our own loss would fabricate the very outcome being measured. Skipped,
counted as `cap_record_unterminated`, published in the manifest.

### Holdout is decided by the PRODUCER, before a single label is bought

The split is battle-level, computed from `--seed`, written to `holdout.json`, and the sampler
**refuses to draw a candidate from a held-out battle** — exclusion happens before ranking, so no
later slicing can readmit one. Leakage is unrepresentable rather than forbidden. It is **stratified
by class**: with 8 cap battles, an unstratified 35% draw can easily take 0 or all 8, and either way
one arm of the meter loses the class the exercise is about.

### A timeout is its own bucket

`n_rollouts` counts ADJUDICATED rollouts — one that reached a terminal, i.e. win, loss **or TIE**;
everything else (`unfinished`) lands in `provenance.n_timeout` and is excluded from both numerator
and denominator. Folding a timeout in would make a busy box read as a losing position — the error
that once let a starved parity run report 39/40 timeouts as a clean PASS. The cap forfeit itself
adjudicates normally (it is recorded as a LOSS), so it needs no special case.

⚠️ **A TIE is a semantic outcome and belongs in the DENOMINATOR — it went into the timeout bucket
until the R1 adversarial review.** `counterfactual._battle_outcome` emits four values (win / loss /
tie / unfinished) and `label_one` computed `n = win + loss`, so a tie was filed as a timeout, which
is none of the three things this section says that bucket counts. Every dropped rollout is a
NON-win, so `k/n` overstated P(win) on exactly the states where a game can end even — and the
owner's clean-world ruling says the same thing in the reward's language (a draw scores `-victory`,
i.e. as a loss). Ties are now adjudicated and counted separately in `provenance.n_tie`, because a
denominator that silently absorbed a second outcome class is one nobody can audit.

### ⚠️ TWO different `--holdout-frac` flags, and conflating them would look like leakage

They are disjoint by construction and neither is the other:

| flag | splits | for |
|---|---|---|
| `main.harvest --holdout-frac` | **battles**, out of the doomed-tail population, BEFORE any label is bought | the METER's test set. These battles contribute zero training states. |
| `winprob_finetune --holdout-frac` | rows of the ALREADY-harvested set, by `battle_tag` | the fit's own train/val split, for epoch selection |

The fit's val split is drawn only from battles the harvest already chose, so it can never touch a
meter-held-out battle — the producer excluded those before ranking. Both are battle-level; the
consumer's `split_by_battle` calls `assert_battle_disjoint` on its own result before returning it.

### The fit is head-only, and that is STRUCTURAL

`winprob_finetune` runs in two phases: a no-grad forward of the frozen trunk caching `value_pooled`
(the `WinProbHead`'s only input, via `ProbeModel._value_pooled_batch` so the numbers stay comparable
with live `cf/*` scalars), then an Adam fit over `head.parameters()` alone. The trunk is not frozen,
it is **absent** from phase 2 — and `_assert_head_only` raises if any param group holds anything
else. Loss is the binomial NLL `k·softplus(−z) + (n−k)·softplus(z)`, slice-weighted and normalized
by `Σ w·n`, asserted exactly equal to the live trainer's `cf_terms.cf_binomial_nll` and to mean BCE
at `n ≡ 1`. Slice weights are inverse-frequency over declared edges (`SLICE_EDGES = (60, 80, 100,
130, 170, 250)`, `SLICE_VERSION`), rescaled to mean 1 so the learning rate means the same thing
across datasets. **Best epoch is chosen by the PLAIN val NLL**, not the re-weighted one — the
re-weighting is an optimizer device, and scoring with it would make the meter agree with the device
by construction.

### The ANCHOR (`--anchor-coef`, default **0.3**) — what makes the fit SAFE

The second thing the pilots bought, and the reason a flagless run is now non-destructive. A harvest
is a **prioritized** sample — selected precisely where the head is wrong — so its label mean sits
far from the population's, and a 6-parameter head fit on it with nothing holding it back collapses
toward that sample mean. Measured, on held-out battles, with the long-WIN control as the
falsification:

| | fit-set label mean | long-loss `detect_le05` | LONG-WIN control `phi_T` | |
|---|---:|---|---|---|
| pilot 1, no anchor | 0.621 | −0.674 [−0.767, −0.570] | **−0.376** [−0.447, −0.299] | REGRESSION |
| pilot 2 (tail stratum), no anchor | 0.427 | −0.326 [−0.430, −0.221] | **−0.165** [−0.212, −0.119] | REGRESSION, half the dose |
| pilot 2, **anchor 0.3** | 0.427 | **±0.000** | **−0.033** [−0.045, −0.023] | **control HOLDS** |

The damage scaled with the sample-mean offset across two independent runs — which is what
identifies the mechanism as selection bias rather than a bad hyper-parameter.

`anchor_coef * mean((z − z0)^2)`, where `z0` is the SUBJECT's logit on the same cached
`value_pooled`, captured **before** the resume branch can touch the head (taking it afterwards
changes the objective mid-run and makes a resumed fit diverge — which is exactly how the bug was
caught, by `test_resume_reproduces_the_uninterrupted_run_bitwise`). It is a per-EXAMPLE penalty,
not a weight penalty: the quantity that must not drift is the head's FUNCTION on the real state
distribution, and an L2 on six parameters says nothing about that.

**Dose.** 1.0 and 3.0 both stop at best-epoch 0 — the anchor dominates and the fit never moves. So
0.3 is the largest dose that still lets the labels speak, and it is the default because shipping
0.0 would ship a setting measured to be destructive. `--anchor-coef 0` opts out and reproduces the
pilots exactly.

⚠️ **At 0.3 the fit is SAFE but not yet BENEFICIAL**: every categorical metric is unchanged and only
`phi_T` moves — the right way on cap endings (−0.109, n=3) and slightly wrong on long losses
(+0.024). The pilot demonstrates the pipeline and its guard rails; **it does not answer the
reducibility question**, which needs more labels than 359 states across two runs.

### The meter is PAIRED, and its control is the whole falsification

`harvest_meter` re-scores each held-out battle's last **K = 5** decisions through both heads and
reports probe O's metrics — `detect_le05` (the substantive criterion), `detect` (as REGISTERED,
whose "declining" half probe O showed saturates in every class), `miss`, `overconf`, `c3band` — with
a bootstrap over BATTLES, never states.

Two properties that are not incidental. **"Pre" is the SUBJECT's reading, not the trace's**, so
these numbers are legitimately not comparable to probe O's published levels. And the **LONG-WIN
CONTROL** (long won battles touched by neither arm) is reported beside the detection rate because a
head that learned "late means lost" scores *perfectly* on every stall metric while being strictly
worse than the head it replaced. `verdict_lines` names that outcome **FAILED RUN** in those words
rather than leaving a reader to notice. `graft_head` likewise **refuses** a graft that changed
nothing — a silent no-op would make the post arm a bitwise duplicate of the pre arm and every metric
would read a perfect, perfectly confident null.

**Tests** `main/harvest_test.py` (39, unmarked, 0.35 s) — schema round-trip, the `decision_idx`
indexing contract and its digest refusal, the cap-record skip, class separation, holdout
stratification/determinism/disjointness, the per-battle cap, the outcome-balance property that
mechanism 3 above exists for, priority monotonicity + the no-evidential-head fallback, CRN pairing
at the `replay_counterfactual` seam, the timeout bucket, and the meter's statistics including the
saturating registered criterion and the FAILED-RUN verdict. Plus
`agents/training/winprob_finetune_test.py` (34, 3.7 s) for the consumer.
