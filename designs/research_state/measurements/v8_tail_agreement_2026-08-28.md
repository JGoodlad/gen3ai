# Inter-fork tail agreement — does a CONSOLIDATED era's teacher fleet share tail structure? (2026-08-28)

**Question (probe E).** Probe D found the current era's teacher tails carry no fork-specific
content: inter-fork tail cosine **0.327** against a fork-vs-no-fork-control **0.306**, excess
`+0.021` — drift, not dark knowledge. The surviving hope for the recorded full-KL re-entry path
was that this is a property of *this* era's under-differentiated fleet, and that a consolidated
parent's exploiters would show real shared tail structure. v8 is the only consolidated era we
have three teachers from. This probe measures its inter-fork tail agreement.

**Answer: v8's teachers agree about their tails at the SAME level as the current era's.** Pairwise
tail cosine **0.357 / 0.330 / 0.361, mean 0.349** — against the gen era's 0.327, and against
**0.344** once the gen era is put on v8's own all-legal mask footing, which is the only honest
comparison. The bootstrap difference is `+0.005`, CI `[−0.021, +0.033]`. Under the brief's other
construction (restrict to argmax-agreeing states) v8 comes out **lower** than the gen era, 0.384
vs 0.401.

**Registered reading selected: "≈ gen's level ⇒ the full-KL re-entry path loses its main
empirical pillar."** The bar for the other reading was ≈0.5+; nothing here reaches it, and the
highest v8 number produced by any construction at any checkpoint is 0.460.

⚠ **This is a weaker instrument than probe D's, by one structural fact: there is NO v8 no-fork
control and there never was one.** Probe D could subtract an ancestry-drift baseline; here we can
only compare a raw agreement against another era's raw agreement. See *Caveats*.

All numbers are pure NumPy over logit dumps that already existed on disk. CPU-only, `nice -n 15`,
BLAS pinned to 1, `models/` untouched, no model loaded.

---

## Provenance and data path

**Data path 1 — existing dumps. No recompute, no era worktree.** Every logit array needed was
already on disk from the plasticity forensics run (`/tmp/plast/`): a shared 3000-state set per era
plus per-model `*__logits.npy` for the parent and every fork.

| | v8 (consolidated) | gen (current, v9) |
|---|---|---|
| parent | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` (277.18M steps) | `ai_v9_29_rev1_0823/final_model.zip` (25.07M steps) |
| teachers | `ai_v8_06_semistall_3team`, `ai_v8_09_pool10`, `ai_v8_13_defensive10` | `ai_v9_53..57_R2F5a..e` |
| **no-fork control** | **MISSING — never run** | `ai_v9_58_R2CTRL` (`--distill-coef 0.0`, no pins) |
| states | `states_v8.npz`, n=3000, obs 2992 | `states_v9.npz`, n=3000, obs 2501 |
| fork length | Δ3.13–3.15M (`_m` checkpoints) | Δ3.05M |
| action mask | **none recorded → reconstructed ALL-LEGAL, 11/11** | real, mean **6.72/11** |
| pairs | **3** | 10 |

**Acid test — the re-implementation reproduces probe D.** Running this probe's construction on the
gen era returns COORD tail inter-fork **0.3269** (published 0.327), fork-vs-CTRL **0.3058**
(0.306), excess **+0.0211** (+0.021), MODE **0.3817 / 0.3391** (0.382 / 0.339). Same construction,
independently coded, four figures agreeing to 3 dp. That is what licenses putting the v8 number
beside the published one.

**Two constructions, both reported, because the brief and its comparison target name different
ones.** The number probe E must be scored against (0.327) came from a *coordinate* partition; the
brief's wording describes a *state* partition. They are not the same measurement, so both are run
on both eras.

| tag | signal per fork | notes |
|---|---|---|
| **COORD** | `(p_teacher − p_parent)` with the two decision coordinates (each side's argmax) zeroed | probe D's M1b — **this is what 0.327 / 0.306 are**; primary |
| **STATE-shared** | the full `(p_teacher − p_parent)` vector, restricted to states where **both** forks' argmax agrees with the parent's | the brief's wording, "over the shared state set" |
| STATE-own | same, each fork zeroed on **its own** flip states | sub-variant, reported for completeness |

---

## The measurement

### Primary — COORD tail cosine, v8 teachers at matched fork length

| pair | tail cosine |
|---|---|
| semistall3 ↔ pool10 | **0.3565** |
| semistall3 ↔ defensive10 | **0.3297** |
| pool10 ↔ defensive10 | **0.3611** |
| **mean** | **0.349** |

Range 0.330–0.361 — **tight**, so the "unstable across pairs ⇒ AMBIGUOUS" branch of the registered
reading does not apply. The three pairs say one thing.

### Beside the gen era

| era / footing | inter-fork **TAIL** | fork ↔ **CONTROL** | excess | inter-fork MODE |
|---|---|---|---|---|
| **v8, matched length** (all-legal, 11 slots) | **0.349** | **— no control exists —** | **unmeasurable** | 0.303 |
| gen, true mask (6.72 slots) — *probe D's published row* | **0.327** | 0.306 | **+0.021** | 0.382 |
| **gen, ALL-LEGAL era-symmetric arm** (11 slots) | **0.344** | 0.330 | +0.014 | 0.385 |
| *(robustness)* v8 at FINAL checkpoints, length-unmatched | 0.366 | — | — | 0.337 |

**The like-with-like comparison is row 1 against row 3: 0.349 vs 0.344.** Two thirds of the
apparent v8 lead over the published 0.327 is the mask regime, not the era — moving the gen era
alone onto all-legal footing carries it 0.327 → 0.344 with no change of models.

### Under the brief's own construction

| era / footing | STATE-shared | STATE-own |
|---|---|---|
| **v8, matched length** | **0.384** (pairs 0.375 / 0.375 / 0.402) | 0.298 |
| gen, true mask | 0.390 | 0.291 |
| gen, all-legal | **0.401** | 0.287 |
| *(gen control, all-legal)* | 0.377 | 0.273 |
| *(robustness)* v8 finals | 0.460 | 0.313 |

Under the construction the brief actually specifies, **v8 is below the gen era**, and below it on
both gen footings. The two constructions disagree on the *sign* of a ~0.01–0.02 difference, which
is itself the finding: the era difference is smaller than the choice of how to measure it.

### Cluster bootstrap on the era difference

Resampled over `src_file` (states come from trace files, many decisions per file; an i.i.d.
bootstrap would understate the error). B = 2000, eras independent — v8 n=132 clusters, gen n=229.

| quantity | point | 95% CI |
|---|---|---|
| v8 COORD tail | 0.349 | [0.328, 0.372] |
| gen COORD tail, all-legal | 0.344 | [0.327, 0.362] |
| gen COORD tail, true mask | 0.327 | [0.313, 0.341] |
| **v8 − gen (all-legal, like-for-like)** | **+0.005** | **[−0.021, +0.033]** |
| v8 − gen (true mask, mask-confounded) | +0.022 | [−0.003, +0.048] |
| **v8 − gen, STATE-shared (all-legal)** | **−0.017** | **[−0.068, +0.029]** |

Both like-for-like differences straddle zero, and the CI's upper edge (+0.033) puts the ceiling on
v8's advantage at ~0.38 — nowhere near the registered 0.5+ bar.

---

## Magnitudes (the caveat the brief asked to carry, made numeric)

Cosines are scale-free; the raw divergences are not, and v8's are much smaller.

Fork-only means (the control is excluded from every row).

| era | mean KL(teacher‖parent) | mean tail-signal ‖·‖₂ | mean argmax agreement |
|---|---|---|---|
| **v8, matched length** | **0.179** | 6.90 | 0.657 |
| gen, all-legal | **0.445** | 9.22 | 0.656 |
| gen, true mask | 0.362 | 8.31 | 0.730 |
| v8, finals (Δ ≈ 10M) | 0.362 | 8.74 | 0.557 |

v8's teachers diverge from their parent **2.5× less** than the gen era's at matched length
(0.179 vs 0.445), and their tail signal is **0.75×** the gen era's in norm — v8's teachers were
not richer in the tail, they were quieter everywhere. Note the argmax agreement is essentially
**identical** across the two eras once the mask is matched (0.657 vs 0.656), so the tail cosines
are being computed over comparably sized agreeing sets.

**One ordering that is genuinely v8-specific and cuts against the hypothesis.** In the gen era the
mode half is the more shared half (mode 0.385 > tail 0.344). In v8 it inverts: tail **0.349** >
mode **0.303**. v8's three teachers agree with each other *less* about where their decisions moved
than about how their tails wandered. Read against a dark-knowledge story that predicts a *richer*
tail on top of differentiated decisions, that is the wrong shape: v8's forks differentiate their
decisions less, not their tails more.

**A cross-era heuristic, offered as a heuristic only.** At the shared all-legal footing, v8's
inter-fork tail cosine (**0.349**) sits barely above the *gen era's fork-vs-no-teacher-control*
cosine (**0.330**). That is the closest thing to a baseline v8 has, and v8 clears it by 0.019 —
about the size of the excess probe D called "nothing". It is a cross-era substitution and cannot
be treated as a test.

---

## Verdict

| Registered reading | Selected? |
|---|---|
| v8 markedly above gen's 0.327 (≈0.5+) ⇒ consolidation-restores-dark-knowledge gains real support | **NO** — max v8 value across every construction and checkpoint is 0.460 (finals, STATE-shared, length-unmatched); the primary is 0.349 |
| **≈ gen's level ⇒ the full-KL re-entry path loses its main empirical pillar** | **YES** — 0.349 vs 0.344 like-for-like, diff +0.005 CI [−0.021, +0.033]; the brief's own construction puts v8 *below* gen |
| In between / unstable across pairs ⇒ AMBIGUOUS | no — the three pairs span 0.330–0.361, tight |

**What this does and does not settle.** It removes the *positive* case for full-distribution KL
re-entry on a consolidated parent: the one era we can look at does not show the shared tail
structure that would justify it. It does **not** prove a consolidated era's tails are noise —
proving that needs the excess over a no-fork control, and v8 has none. The honest statement is
that the re-entry path is now supported by *no* measurement rather than contradicted by one, which
is a downgrade from "supported by an untested plausibility" and should be scored as such.

**The entry condition probe D proposed survives intact and is now more clearly the binding
constraint.** `tail specificity excess` = inter-fork tail cosine − fork-vs-control tail cosine is
still the number that says whether a teacher's distribution is worth more than its argmax. Probe E
could not compute it for the one era it most wanted to. **A no-fork control is now the blocking
input for a third program** — the plasticity forensics named it MISSING, probe D needed it, and
probe E is where its absence actually costs a verdict.

---

## Caveats

1. **No v8 no-fork control, and one cannot be manufactured.** `ai_v8_04` has no plain
   continuation; producing one now would mean training a 277M-step-parent fork on the v8-era
   codebase. Every v8 cosine here is a raw agreement, so the v8 column can be compared *across
   eras* but never converted into a specificity excess.
2. **Three forks = three pairs.** The gen era's 0.327 pools ten. The bootstrap CI reflects state
   sampling, not fork sampling; a fork-level bootstrap on n=3 would be meaningless and was not run.
3. **Mask regime.** v8 traces record no `action_mask`; the reconstruction is all-legal over 11
   slots against the gen era's real 6.72. Handled by reporting the gen era on both footings, and
   the effect is large enough to matter (0.327 → 0.344 from the mask alone) — which is exactly why
   the headline comparison is the all-legal one and the published 0.327 is *not* the right target
   to compare against directly.
4. **Magnitude asymmetry could, if anything, favour v8.** Cosine between two signals each carrying
   a fixed-size idiosyncratic component is attenuated when the shared component is smaller; v8's
   total divergence is 2.5× smaller. So v8's 0.349 may be a slight *under*-read. The direction is
   noted honestly, and it is not remotely enough to reach 0.5.
5. **Residual era asymmetry is irreducible.** obs 2992 vs 2501, a different module set (v8 has no
   `opp_intent` / `opp_belief` / `pair_outcome` / `cf_*`), parent 277M vs 25M steps, fork length
   matched only to 1.03×, and the gen forks ran `--grad-accum-steps 8` against different arms'
   settings. No amount of masking symmetry removes these.
6. **Checkpoint choice matters at the margin.** Matched-length is primary (it is the comparison the
   forensics established); v8 finals run ~10M steps and score higher on every construction
   (COORD 0.366, STATE-shared 0.460) — more training, more shared drift, and no longer matched to
   the gen fleet. Reported as robustness, not as the headline. Even the finals row does not reach
   the 0.5 bar.
7. **Dump caveat, inherited.** `plast_forward.py` called `get_distribution(obs)` without
   `action_masks`, so the stored log-probs are normalised over all 11 slots; the mask is
   re-applied here, which is the training-faithful choice (`_distill_loss` masks both sides before
   the softmax). Identical treatment in both eras.

## Reproduction

`tmp/v8_tail_agreement.py` (the four era blocks and both constructions) and
`tmp/v8_tail_bootstrap.py` (the cluster bootstrap) in this worktree; every raw number, including
all per-pair cosines and per-fork magnitudes, is in the sibling `.json`. Inputs are read-only
from `/tmp/plast/` (the plasticity forensics' state sets and logit dumps) — no model was loaded
and no era worktree was pinned.
