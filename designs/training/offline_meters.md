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
