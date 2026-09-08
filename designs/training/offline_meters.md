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
`from agents.training.cf_audit import wilson_ci` — which `cf_producer` and the tests do — still
resolves. The arithmetic is unchanged, and that is EVIDENCE rather than a promise: the parity golden
above was captured before the move and reproduced byte-for-byte after it.

⚠️ **Three near-siblings elsewhere in the tree are deliberately NOT merged into it**, and the module
docstring carries the reasons so nobody "de-duplicates" a shipped instrument's output by accident.
`scaffolding.py`'s `spearman_rho` and its own `cluster_bootstrap_ci` use the **NaN** refusal
convention (TensorBoard drops NaN, so a degenerate slice leaves a GAP in the live
`train/scaffolding_gauge` curve) where this module returns `None` for a JSON report, and its
bootstrap is strictly more general — it resamples ROW INDICES and evaluates an arbitrary `stat_fn`,
which is what lets `reliability_table` compose with it. `winprob_finetune.label_noise_variance`
subtracts the same `p̂(1−p̂)/(n−1)` identity but PER ROW with a heterogeneous `n`.
`main/q_amortization.spearman` is the one true duplicate — same shape, same `None` convention, an
exact `std() == 0` flatness test instead of the relative-tolerance one here — and moving its call
site is a behaviour change (a *near*-flat row starts refusing instead of reporting float noise)
that wants its own pass and its own evidence. `hodge.py` and `elo.py` carry no general-purpose
statistics at all; everything in them is bound to the rating model.

## `replay_imputation_probe` — the own-side imputation meter (`replay_imputation_probe.py`)

A **meter, not a lever**: it answers "how far would our observation move if the only thing we knew
about our OWN side were what a public Showdown replay had shown by now?" — the #1 risk in
`designs/research_state/metamon_replay_feasibility.md` §2.7, which Metamon cannot quantify because
they have no ground truth and we can because we own a simulator. There is **no transcoder and no
`|request|` synthesis here**; those are that memo's Gap 1 and deliberately out of scope.

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/agents/training/replay_imputation_probe.py 20 [--impl rust] [--json out.json]
```

**Mechanism.** It plays reproducible bridge battles (the `record_fixture_battle` recipe — pinned
teams, a per-player RNG, a fixed sim seed) with a seeded-random policy that decides on the TRUE
obs, and at every decision: encodes truth cold (`assembler=None`), snapshots our own mons,
overwrites their **not-yet-revealed-by-then** moves / item / EVs+nature with the top Smogon-prior
candidate for the species, bumps `Gen3Battle._state_epoch` so the `live_view` memo rebuilds,
encodes again, restores, and bumps again. Truth is re-encoded a THIRD time after the restore and
required to be bit-identical — the meter mutates the live battle, so a leaked restore would make
every later decision's "truth" a previous decision's imputation, and that is gated rather than
believed.

**Reveal tracking models a REPLAY, not poke-env** (`track_own_reveals`, pure over raw protocol
lines, 37 unit tests in `replay_imputation_probe_test.py`): moves on use, item on activation,
spreads never. The sharp edges each have a test — Sleep Talk's callee IS in the user's set,
Metronome's and Mirror Move's are NOT, Struggle is never a set move, Knock Off on the opponent
does not disclose ours, and the species comes from the switch DETAILS (this pool carries localized
nicknames, the same trap `search_dividend.determinize` documents).

**Result (2026-08-24, 20 battles / 2,640 decisions), recorded in full in the memo.** The error is
structurally confined to the our-team block — opp_team, active context, global env, pair history
and the H-B event window are *exactly* zero everywhere, being opponent- or log-derived. Inside it,
`moves` carries almost all of it (relL2 0.56 early → 0.36 late), `items` is ~free in gen3ou
(0.036), and **`spread` is a flat floor that never decays** (0.268 / 0.268 / 0.256) because no
battle progress ever reveals an EV spread. The early-game confound is confirmed at ~2.7× (whole-obs
relL2 0.364 at turns 1–5 vs 0.136 at 16+). ⚠️ The `reactive.active_req_moves` row it prints is an
ARTIFACT of holding the request at truth and is marked `*` in the output — read its direction,
never its size.

