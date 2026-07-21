# Regularization and noise in PPO — why dropout breaks it, why FiLM doesn't

**TL;DR:** PPO doesn't forbid randomness — it forbids *unmodeled* randomness. The importance
ratio π_new(a|s)/π_old(a|s) is a measurement that only means anything if the network is a fixed
function between rollout and update; dropout makes every forward a different sub-network, so the
ratio reads mask noise as "policy change," the clip mis-fires, and the value bootstrap inherits
the same corruption. FiLM is a *deterministic* function of the observation — the ratio machinery
treats its weights like any other weights. And the two address opposite diseases: dropout is
subtractive noise against **overfitting** (memorization); our problem was **under-fitting the
conditional structure** (per-team lessons averaged away by interference), which wants additive
deterministic structure, not noise. The rule: **randomness is welcome in the environment and in
the action distribution, and forbidden inside the function between them.**

## Intuitive level

PPO's safety mechanism is an instrument: each update epoch it re-asks the network's opinion of
the actions the old policy took, and clips the update when the opinion has moved too far. The
instrument assumes same input → same opinion. Dropout turns the network into a slot machine —
a different sub-network answers every time — so the instrument registers change where no learning
happened, and spends its trust region on noise. You haven't regularized the policy; you've
blinded the thing that keeps policy updates safe.

The policy's own action *sampling* is different: that randomness is part of the modeled
distribution — the log-probs are known exactly — so the accounting is exact. The existence proof
that "unmodeled" is the real issue: *consistent dropout* for policy gradients (freeze the mask at
rollout, reuse it in updates) fixes dropout by removing exactly the unmodeled part — at which
point it is essentially a deterministic network again.

## Why FiLM helped where dropout would hurt

Two axes, FiLM on the opposite end of both:

1. **Deterministic vs stochastic.** FiLM (`gen3_zarch_film_v1`) is a plain function of the team
   in the obs: same team → same z → same γ/β → same logits, all epochs. Nothing PPO measures is
   corrupted. (Determinism is also load-bearing for the verification stack — OFF-byte-identical
   toggles, CRN-anchored search, frozen-opponent parity — none survive dropout.)
2. **Noise-that-removes vs structure-that-adds.** Dropout shrinks effective capacity during
   training to prevent co-adaptation/memorization. Our diagnosed disease (the amortization gap,
   see [[amortization_gap_and_conditioning]]) was the opposite: the shared head *averaged*
   per-team lessons into a generalist mush. The fix wanted a dedicated, deterministic,
   team-conditioned storage channel — not noise. Dropout on an interference problem is treating
   anemia with bloodletting.

## Where dropout's *job* actually lives in PPO (our flags)

The supervised-learning role ("don't commit too hard to the current data") is served by
mechanisms placed where PPO's accounting allows:

- **The clip range** (`--clip-range`) — the RL analog of overfitting is over-updating on the
  latest rollouts (policy collapse); clipping is that regularizer, built into the core objective.
- **The entropy bonus** (`--ent-coef`, the state-conditioned `--defensive-entropy-boost`) —
  regularizes the *policy distribution* (keeps modeled randomness from collapsing to premature
  determinism). The closest spiritual sibling of dropout, living in the objective where it is
  differentiable and accounted.
- **Deterministic representation regularizers** — PopArt (`--use-popart`, value-scale
  conditioning), the z_arch VICReg variance floor + recon BCE (`--zarch-vicreg-coef` /
  `--zarch-recon-coef`, anti-collapse). Dropout's "keep representations healthy" job, done via
  losses instead of noise.
- **Environment-side randomness — always legal.** Stochastic pool opponents at temperature,
  PFSP/team sampling, battle RNG: randomness upstream of the policy's input diversifies the data
  distribution; the ratios never see it.

## Synthesis

Dropout and FiLM look like siblings ("bolt something onto the net"), but they differ on exactly
the axis PPO cares about: whether the addition is inside or outside the measurement. Dropout
injects unmodeled stochasticity into the very quantity PPO must measure precisely; FiLM adds a
deterministic conditional pathway PPO treats as ordinary weights. And they target opposite
diseases — memorization vs interference. When reaching for regularization in this codebase, ask
first *where the randomness will live*: environment (fine), action distribution (fine, entropy
keeps it alive), or the function in between (forbidden).

## See also
- [[amortization_gap_and_conditioning]] — the interference/storage diagnosis FiLM addresses, and
  why "adds capacity" decomposes into representational / optimization / gradient-coherence claims
- [[objective_richness_and_representation]] — the simplicity-bias backbone (why the shared head
  under-fits conditional structure by default)
- `src/agents/training/CLAUDE.md` — the live flag surfaces (clip range, entropy, PopArt, zarch aux)
