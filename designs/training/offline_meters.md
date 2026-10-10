# Training — shared statistics and the offline probes

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## `stats.py` — the package's SHARED small-sample statistics

**`agents/training/stats.py` is where a stateless estimator lives once the second consumer exists.**
It holds `wilson_ci`, `spearman`, `cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci`,
`sd_true_excess` and the `MIN_CELL_N` / `MIN_SUBCELL_N` sample-size floors below which that
spread is not reported — pure NumPy in, floats out: no labels, no battles, no checkpoints, no
filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a helper that
has to know what a *decision* or a *bias map* is belongs beside the instrument that owns the
concept. The floors are the one pair of CONSTANTS admitted, and for the estimator's own reason:
`cf_audit.resolution_cells` and `cf_audit_twin.twin_resolution_read` bin different things and
must refuse at the same n, and two copies of a floor is two thresholds that drift.
They were lifted out of `cf_audit.py` on 2026-09-06 (the file-size ratchet's first cut of
the 1,000–2,000 band, 1439 → 1279 lines), which imports them straight back, so
`from agents.training.cf_audit import wilson_ci` — which `cf_producer` and the tests did — still
resolved (both modules, and `cf_audit_twin`, were deleted in P6 slice 6d-1; the functions live on here for
`best_response_gap` and `value_sidecar_read`). The arithmetic is unchanged, and that is EVIDENCE rather than a promise: the parity golden
above was captured before the move and reproduced byte-for-byte after it.

⚠️ **Three near-siblings elsewhere in the tree are deliberately NOT merged into it**, and the module
docstring carries the reasons so nobody "de-duplicates" a shipped instrument's output by accident.
`scaffolding.py`'s `spearman_rho` and its own `cluster_bootstrap_ci` use the **NaN** refusal
convention (TensorBoard drops NaN, so a degenerate slice left a GAP in the live
`train/scaffolding_gauge` curve, retired in P11d; the convention stays with the offline readers) where this module returns `None` for a JSON report, and its
bootstrap is strictly more general — it resamples ROW INDICES and evaluates an arbitrary `stat_fn`,
which is what lets `reliability_table` compose with it. `winprob_finetune.label_noise_variance`
subtracts the same `p̂(1−p̂)/(n−1)` identity but PER ROW with a heterogeneous `n`.
`main/q_amortization.spearman` was the one true duplicate; it went with the E5 step-5 amortization meter (deletion pass L4), so `stats.py` now has no near-duplicate of `spearman` left in the tree. `hodge.py` and `elo.py` carry no general-purpose
statistics at all; everything in them is bound to the rating model.

## `replay_imputation_probe` — DELETED (T27 / P0 dead-code removal, 2026-10-06)

The own-side imputation meter (`agents/training/replay_imputation_probe.py` + its 37 unit tests) is gone: a one-off
measurement ("how far would our observation move if the only thing we knew about our OWN side were what a public
Showdown replay had shown by now?") that ran once, 2026-08-24 (20 battles / 2,640 decisions; the result is recorded in
full in `designs/research_state/metamon_replay_feasibility.md`), had no importer and no caller, and
read the Python battle layer's raw `Pokemon` objects, which the poke-env retirement (T27) is removing. The error was
structurally confined to the our-team block: `moves` carried almost all of it (relL2 0.56 early, 0.36 late), `items` was
~free in gen3ou (0.036), and `spread` was a flat floor (~0.27) because no battle progress ever reveals an EV spread.
Recover the code from the repository history (the last commit that has the file) if the question returns; on the Rust
reader it would be re-asked as a reader-level meter, not as a mutation of Python objects.
Record: `designs/ops/deletion_pass_manifest.md` §8.

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### The POLICY DRIFT meter — refining or adopting a new strategy? (`policy_drift.py`, `python -m main.policy_drift`)

🚨 **A DESCRIPTOR, not a test.** It extends the churn probe (`churn_probe.py`: masked KL between two
checkpoints on a FROZEN probe-state set — collected since P6 of the poke-env retirement by the checkpoint's greedy
self-play on the RUST env core, `utils.rust_env.fixture_battles`; a set frozen before then came from poke-env
bridge battles, a different sample of the same distribution) into a per-snapshot SERIES. Each snapshot is compared with
(a) the PREVIOUS one, (b) the latest one at least `--back-steps` (10M) earlier, (c) a fixed ANCHOR (the
first snapshot recorded — pool snapshots exist only once self-play is seeded — or `--anchor <zip>`).
Per reference it records the masked KL (mean and median), the greedy FLIP RATE bucketed by the OLDER
policy's top-1 − top-2 margin (`<0.1 | 0.1-0.3 | >0.3`; single-legal states excluded), and the
ACTION-MIX shares on "choice" states (both a switch and a move legal): switch / attack / status / setup
/ hazard / recovery / phazing / self_ko, classed from the dex (`agents.gen3_data.moves`) through the
REQUEST-order `active_req_moves` block. Self-KO is a curated id set (Explosion / Selfdestruct / Memento),
because the dex carries no self-faint flag. **Cycling?** means the current policy is closer to an older
reference than the previous snapshot was. The verdict thresholds (`SHIFT_ABS`, `CONF_FLIP_RATE`,
`CYCLE_*`) are reading aids, not calibrated against a null.

**CONDITIONAL class rates are the primary action read** (`policy_drift_cond.py`). The overall share
("hazard 2%") confounds LIKING a class with how often it is AVAILABLE, so each class is also reported
among the probe states (≥ 2 legal actions) where it was LEGAL: the greedy rate, its n, a Wilson 95%
interval, the mean probability MASS on the class (a lean before the argmax flips), and the delta vs
prev / back / anchor with a PAIRED bootstrap 95% interval (same states, seeded). For hazard / recovery
/ setup a USEFUL grain is added from obs facts read through the layout, each rule verified in
deps/pokemon-showdown: hazard = opp Spikes < 3; recovery = HP < 100% (Wish: none pending — it is useful
at full HP; Rest: not asleep; Swallow: stockpiled); setup = a raised stat < +6 (Belly Drum: HP > 50%;
non-Ghost Curse assumed). The verdict names a class by its conditional change (useful grain where
defined) **only when that interval excludes 0** and |Δ| ≥ `SHIFT_ABS`; the overall shares stay as a
secondary line. Rows written before the block existed get it from `backfill <run>` (from the cached
`probs/`, no model run) into a `cond.jsonl` sidecar — `rows.jsonl` is never rewritten; `watch` runs
the backfill at startup.

- `collect <run>` freezes a probe set once per lineage, from the latest pool snapshot (or the latest
  checkpoint, with a warning to RECOLLECT once the pool seeds). `watch <run>` is DETACHABLE and
  RESUMABLE. It writes one fsynced row per snapshot to `~/gen3ai_archive/policy_drift/<run>/rows.jsonl`
  (`$GEN3AI_ARCHIVE_DIR` overrides) and never writes under `models/`, which it REFUSES. It also caches
  each snapshot's action probabilities in `probs/`, so a snapshot the sliding pool window has pruned
  still serves as a 10M-back reference. `report <run>` prints the KL/flip table, then the conditional
  tables (rates, plain-legal rates for the gated classes, Δ vs the long reference with intervals).
- 🚨 **The probe set is pinned by sha256 in `meta.json`**, and a watch on a different one is REFUSED.
  To recollect, start a NEW out dir. Comparisons across two probe sets are not comparable.
- `--source snapshots` (the default) follows the PROMOTION-gated pool, so a run that stops being
  promoted stops producing rows. `--source both` adds the periodic checkpoints. Note that the default
  anchor is then the first CHECKPOINT, which may predate self-play.

### `stats.py` — the package's SHARED small-sample statistics

**Where a stateless estimator lives once the second consumer exists** — `wilson_ci`, `spearman`,
`cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci`, `sd_true_excess` and the `MIN_CELL_N` /
`MIN_SUBCELL_N` floors. Pure NumPy in, floats out: no labels, no battles, no checkpoints, no
filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a
helper that has to know what a *decision* or a *bias map* is belongs beside the instrument that owns
the concept. ⚠️ **Two near-siblings elsewhere are deliberately NOT merged into it** (the
NaN-refusal pair in `scaffolding.py` and `winprob_finetune.label_noise_variance`) — the reasons are in the module docstring and in
[`designs/training/offline_meters.md`](offline_meters.md), so nobody
"de-duplicates" a shipped instrument's output by accident.
