# Hidden-info KNOCK-IN FORKS — measurements

Registration: [`designs/endstate/design_hidden_info_binding.md`](../../../endstate/design_hidden_info_binding.md) §8
(the training axis), §8.4 (the crosses) and §8.5 (the rules). Run by the Training session.

**As played** (the arms, the pin and the deviations from §8.3's argv, each an orchestrator decision, 2026-10-14):

- 15 forks of `rb_es_end2_s2001/final_model.zip` (the closing test's adopted end state, 15,047,835 steps) to
  18,047,835 at P_hib `64a695dc` (the data-landed main commit: the learnset pre-evolution fix), seeds 3101–3103 ×
  arms `moves` (`--oracle-reveal ki_moves` = `moves:seen@0.95`), `items` (`ki_items`), `spreads` (`ki_spreads`),
  `species` (`species:unseen@0.95`, mass-matched) and the NO-CHANGE control `ctrl`, interleaved per seed.
- The levers are OBSERVATION reveals (`--oracle-reveal` + `--diagnostic-fork-consent`), not §8.1's belief-input
  injection (never built; `design_information_levers.md` §6).
- **No wrong-moves placebo** (decision (a)): the training observation refuses a `~placebo` source, so §8.5's K-VALID
  placebo clause and K-L-WORTH's Δ_L − Δ_WRONG are UNREADABLE in this batch.
- **The control-vs-parent (v8) read is CONFOUNDED** (decision (c)) by the learnset data change (the parent trained
  before it) plus the +3M steps.
- `--trunk-precision bf16` on every arm (the bf16 strength A/B's PASS, `../bf16_strength_ab_2026-10-12/`); the
  parent's recorded recipe TYPED on every fork (a fork does not inherit the cli_args recipe: untyped it falls to the
  parser defaults — 32 envs, no self-play, …).
- `--behaviour-check warn` uniformly (Amendment 2 of the closing test, carried over): the K9(b) tie-share clause warns,
  |d log π| < 1e-4 is held BY READ at each fork's bank. The `moves` arm's tie share sits above the 0.15 ceiling
  (median ≈ 0.16 on seeds 3101 and 3102 against 0.005–0.036 on every other arm): the moves reveal raises it.

- [`read_hib.py`](read_hib.py) — §8.5's rules CODED BEFORE ANY CELL IS PLAYED: `main.h2h.cross.cross_stat` at n = 3
  (df 4), the two-sided 95 % interval X5 §7.4 reports, rule 8 at ±0.05 pp; the per-lever verdict
  (`WORTH_FIRST_CLAUSE_ONLY` / `NULL` / `NOT_DETECTED`; K-L-WORTH itself is not declarable, decision (a)), the moves
  EXTENSION trigger, K-RANK on the paired contrast (items and spreads registered; species a declared descriptive
  extension), the v8 read descriptive and stamped. `python read_hib.py rehearse` (from a checkout's `src`) exercises
  every verdict branch and every rule-8 boundary on constructed matrices.

The cross artifacts (plans, shas, cells, matrices, the verdict) are added here when the crosses are read.
