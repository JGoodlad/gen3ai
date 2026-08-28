# Learning note — distillation-friendliness: what the literature says (2026-08-28)

> ⚠️ **STATUS BANNER:** written hours before the plasticity forensics reported. The forensics
> refuted the "our v8 parent was rigid" premise (v8 parent MORE trainable than rev-1; its
> teachers renovated the trunk MORE) — so the sections below describing what mature bases buy
> should be read as *the literature's claims*, which our v8-vs-gen pair does NOT cleanly
> instantiate. The strategy catalogue (§5) is unaffected: those tools work by *constructing* the
> desired property rather than assuming we have it.

**Framing:** there is no literature called "make your model distillable." Our question sits at
the intersection of four literatures plus LLM practice; none asks it exactly.

## 1. The stability–plasticity dilemma — biology's answer is our flywheel

Grossberg's dilemma (plasticity to acquire, stability to retain, one substrate struggles to do
both) and its most influential answer: **Complementary Learning Systems** (McClelland, McNaughton
& O'Reilly 1995) — the hippocampus learns fast and episodically; during sleep it replays into the
neocortex, which learns slowly and integrates without overwriting. Exploiters = hippocampus,
generalist = cortex, the fold = sleep replay. The theory's warning maps onto our week:
consolidation works *because cortex learns slowly* — replaying into a fast-learning cortex causes
interference. A 1995 prediction of our 13%.

## 2. Plasticity-loss literature — useful, but pointed the other way

Lyle et al. (capacity loss), Dohare et al. (continual backprop, Nature 2024), Sokar et al.
(dormant neurons), Nikishin et al. (primacy bias), Ash & Adams (warm-starting hurts): all treat
plasticity loss as a **disease** to cure. We want the *structure* of maturity without the
disease — their toolbox (resets, recycling, shrink-and-perturb) is the inverse of our need, and
our own audit found those levers null on this model anyway.

## 3. Model merging / task arithmetic — the strongest published "annex" evidence

**Task arithmetic** (Ilharco et al. 2022): on mature pretrained bases, fine-tune deltas ("task
vectors") can be literally ADDED in weight space and the skills compose. Model soups (Wortsman
et al. 2022) average whole fine-tunes and it works. Why: fine-tunes of a converged base stay in
the same basin (**linear mode connectivity**, Frankle et al. 2020) — rigidity forces every
fine-tune to express itself as a small, near-additive, mutually compatible delta. Every LoRA
marketplace is this fact commercialized. Documented failure mode: merging fine-tunes of a young
or heavily diverged base produces garbage.

Sharp near-free prediction for us: F5a's task vector added onto rev-1 should fail; a v8 teacher's
onto the v8 parent should roughly work. (Post-forensics: the weight-delta profiles complicate the
v8 half — its teachers moved the trunk substantially — so this probe is now *more* interesting,
not less: if v8 task vectors still add despite trunk-heavy deltas, basin-sharing matters more
than delta location.)

## 4. Continual learning — the toolbox for SYNTHETIC maturity

- **Anchor important weights:** EWC (Kirkpatrick et al. 2017) — Fisher-weighted penalties =
  manufactured rigidity. Variants: Synaptic Intelligence, MAS.
- **Project the gradients:** Orthogonal Gradient Descent; Gradient Projection Memory (Saha et al.
  2021) — new learning projected into the orthogonal complement of the subspace existing
  competence uses. The annex enforced by linear algebra. (Distinct from PCGrad, which mediates
  two LIVE gradients; this protects a FIXED subspace.)
- **Distill from yourself as the anchor:** Learning without Forgetting (Li & Hoiem 2016) — while
  learning the new task, pull outputs toward your own pre-update outputs on old-task inputs. For
  us: fold = teacher CE on-slice + self CE off-slice. Targets exactly our bar 2, nearly free in
  existing plumbing.

## 5. The strategy catalogue

- **A. Constrain the teacher** (annex-shaped content at the source): LoRA-style exploiters on a
  frozen trunk — cannot renovate by construction, mergeable by construction. Softer: a KL-leash
  (off-slice penalty tying the exploiter to the parent) — turns the transfer gate from a
  measurement into a training objective. Cost: possibly weaker exploiters.
- **B. Protect the student** (synthetic maturity at fold time): EWC / gradient projection / the
  LwF self-anchor.
- **C. Schedule it** (the CLS answer): fold when the substrate is ready — the distillability
  index as the timing signal; two-timescale variants (fold into a fast copy, slow-merge the copy)
  exist if needed.
- **D. Give annexes somewhere to live** (architectural): Progressive Neural Networks (Rusu et al.
  2016), adapter slots, Mixture-of-Experts routing. Biggest promise, biggest build.
- **E. Fix the state distribution** (orthogonal to geometry): on-policy distillation (GKD,
  Agarwal et al. 2023; the DAgger lineage) — already ours; the remaining gap is REACHABILITY of
  the teacher's deep lines → teacher-guided episode starts (our reconstruction layer can
  materialize mid-battle states; a real build).

## 6. Ranking for this project + honesty

By evidence-per-cost: (1) LwF self-anchor off-slice — nearly free, targets bar 2; (2) the
task-vector probe — near-free diagnostic; (3) exploiter KL-leash; (4) EWC/projection on the fold;
(5) LoRA exploiters — the big structural bet.

Honesty: the merging literature's evidence is overwhelmingly from supervised fine-tunes of very
large converged bases. Whether its picture survives RL fine-tuning on a 25M-step game trunk is
what we are testing, not what we may assume. "Distillability as a training objective" is not a
published line; strategy A and the index are extrapolations, and the day's two agents are the
first data on whether they extrapolate. (The first agent's answer, hours later: not the way the
just-so story said — see the forensics record.)
