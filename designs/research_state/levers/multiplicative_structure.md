# General multiplicative structure — gated activations, MI junctions, extremal pooling

**Status:** 🔲 **REGISTERED, deliberately UNBUILT — every rung must be earned by a measured gap** ·
**Ledger:** `00c5a11` (the gate-structure family, owner insight) · `07e9a54` (**owner ruling: no
hand-picked aggregates**) · `c0619a3` (asymmetric gating ruling) · `b63a96f` (the mechanism it
targets)

One-line claim: *rare VETOES — "t ≥ 250 regardless of position", "my last Roar mon is gone" — are
products, and a network of sums through squashes learns them as slopes; the fix is to supply the
MULTIPLICATION generically and let data supply the decision.*

## Known (cleared the honesty gates)

- **The target defect is measured, not hypothesised.** The 0.999 stall tails (34.8% of cap tails end
  φ ≥ 0.5 on lost-by-construction games) have a mechanism whose first component is exactly this:
  a multiplicative veto that a sigmoid over additively-combined features can only approximate as a
  slope (`b63a96f`). The clock bought 81% → 22% — a huge slope — and a slope cannot zero a 0.999
  position prior in five turns.
- **The deficiency is NOT expressiveness — it is SAMPLE ECONOMICS.** Attention is already bilinear,
  FiLM already exists here, and the damage op is this very lesson applied at the feature level
  ("precompute every nonlinearity of two numbers IN the op", "ship the MARGIN"). Rare vetoes on thin
  slices hit all three of the 0.999 mechanisms at once.
- **🚨 OWNER RULING (`07e9a54`): NO hand-picked aggregates.** The originally-registered
  coverage/answer-count feature family (`00c5a11` — per THEIR mon, the count of OUR living mons that
  both survive its best hit and threaten back; living-phazer count; living-resist count per revealed
  threat axis) is **DEMOTED to probe-side instruments only** — legal for measuring what the model
  knows, **never shipped as obs**. The distinction of record: *the damage op ships GAME RULES (exact
  physics); answer-counts are OUR judgment of what matters — a step over the line.*
- **The general programme, three levels, zero chosen semantics (`07e9a54`):**
  1. **GATED ACTIVATIONS** — the towers are plain Tanh (sums through squashes ⇒ slopes); the GLU
     family (SwiGLU / GeGLU, the modern transformer FFN default) gives every layer an elementwise
     learned-gate PRODUCT. The canonical general multiplicative primitive (Shazeer 2020; Jayakumar
     et al. 2020, *Multiplicative Interactions*), and the family **strictly enlarges the
     easily-learnable class** — which is where the cliff problem lives.
  2. **MI JUNCTIONS** — low-rank bilinear / FiLM-style layers where streams meet (context
     injections, `value_pooled` → heads), including the **GATED READOUT** that subsumes the
     clock-gate generically.
  3. **EXTREMAL POOLING** — min / softmin / max channels beside the attention and mean pools:
     averages wash out worst cases *by construction*. The operator set is a VERB, not a noun.
  Depth is noted as the blunt alternative — capacity without a change of inductive bias.
- **Asymmetric / decoupled gating is LEGAL and standard (`c0619a3`).** f(content) ⊗ g(context) with
  an attention-pooled (CLS) conditioner is FiLM / cross-attention conditioning; symmetric GLU is the
  special case; diffusion's text-gates-image is the canonical decoupled example. Decoupling buys a
  FACTORIZATION bias (shared basis × per-context relevance).
- **…but the TEAM-dial application collides with three of this project's own nulls (`c0619a3`).**
  LUT free per-team codes **+0.024 n.s.** ("do NOT climb to LoRA/MoE"); count-dominates-conditioning
  (**+0.077 SIG** vs **+0.027 n.s.**); the FiLM SNR analysis found **2/3 of conditioning energy in
  ONE shared direction**. The network voted for mostly-shared competence, gates FRAGMENT statistical
  strength across 719 thin slices, and the trunk already attends over both teams. **Standing bar
  re-affirmed: a team dial ships only after the gated quantity is shown to PREDICT performance — the
  exact bar the LUT arm failed.**
- **Where the construction legitimately debuts: the SURVIVAL-CONDITIONED READOUT** — φ = f(position)
  ⊗ g(CLS over temporal context) — aimed at the one MEASURED multiplicative defect (the 0.999
  tails), semantics-free, slotted as the gated-readout rung.

## Not-known

- **Whether data alone suffices.** This is the gating question and it is not yet answered: the
  reducibility probe is blocked on post-fix cap traces (see
  [stall_tail_overconfidence.md](stall_tail_overconfidence.md)).
- Whether SwiGLU/MI helps *anything else* here, or only the veto class. Nothing has been A/B'd.
- Whether extremal pooling is even reachable — it is the third rung and conditioned on
  aggregate-level failures surviving both earlier legs.
- Inductor / `torch.compile` behavior under a changed activation family is **unverified** and is a
  named prerequisite of any arch arm.

## Pros

- It is the *general* answer to a class this project keeps re-encountering — the stall veto and the
  bait/pivot coverage pathology are the same object seen from two sides.
- Every level is standard, published machinery with no Pokémon semantics baked in — it satisfies
  "provide facts, don't bake priors" by construction, which the answer-count family did not.
- The gated readout is a small, local change aimed at one measured defect, not a trunk rebuild.

## Cons

- **Retrain-class and arch-version-bumping** — the most expensive category of change available.
- The project's own conditioning nulls are direct evidence that added gating structure does not
  automatically pay here; the LUT/FiLM history is a warning, not background.
- Building it before the data leg would violate the sequencing rule that this family was registered
  *under*, and would make a null uninterpretable (data-starved vs architecture-adequate).

## Next test — the sequence IS the test (`07e9a54`)

1. **(i) Data leg:** harvest + factory over-representation (Beta-head priority — also
   semantics-free) + the reducibility probe **on the CURRENT architecture**. *Data may suffice.*
2. **(ii) Arch leg, only if (i) fails:** a **SwiGLU / MI A/B at 5M pre-test scale** — arch-version,
   retrain-class, Inductor re-verification required.
3. **(iii) Extremal pooling**, only if aggregate-level failures survive both.

**Every step general, every step earned by a measured gap.**
