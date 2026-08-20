# Continual learning and forgetting — the tick's quiet risk

*Why neural networks forget, the three fix families, and which of them the flywheel already is.
Grounded in the retention measurements this project has actually taken.*

## 1. Intuitive: networks learn by overwriting

A network has one set of weights and every gradient step bends ALL of them toward the current
batch. Learn task B after task A and B's gradients freely repurpose the weights A depended on —
nothing marks them as load-bearing. That is catastrophic forgetting, and it is not a capacity
problem (the network could store both) but an OPTIMIZATION problem: nothing in SGD says "keep
what you had."

The flywheel lives with this on both clock edges. The TICK distills K teachers into the
generalist while RL keeps training — new skills overwriting old ones, and each teacher's slice
overwriting the others'. The TOCK forks an exploiter and specializes it hard — the fork's
generalist competence erodes while the exploit sharpens (measured: the double-sided recipe exists
precisely because a naive exploiter's defense rotted).

## 2. The three fix families — and which ones we already run

**Rehearsal (replay old data).** The simplest and empirically strongest family: keep interleaving
samples of the old task. **The flywheel's teacher data IS rehearsal** — supervised distillation
targets are replayable, so a slice's restoring force is a data-mixing dial rather than a shrinking
share of on-policy experience. This is the structural reason the fold has no 1/N wall (ai_v10 §9)
while pure multi-team RL does, and the leaky-bucket refresh (a decayed slice's teacher returns for
one revolution) is SCHEDULED rehearsal. Distillation is a particularly good rehearsal medium
because soft targets rehearse the FUNCTION, not just the answers — the dark-knowledge point.

**Regularization (protect important weights).** EWC and kin: estimate each weight's importance to
old tasks (Fisher information) and penalize moving the important ones. We do not run this, and
the honest reason is that rehearsal dominates when old-task data is cheap — and ours is free.
Worth knowing for the one place data is NOT replayable: the RL self-play skill itself, where the
only "old data" is a frozen snapshot. There, the pool of past selves acts as an implicit
regularizer (you keep being tested against what you used to be).

**Architectural (separate parameters per task).** Adapters, LoRA, FiLM — give each task its own
small parameter set so interference is impossible by construction. This project has TWO measured
nulls against it in its own domain: the zarch/FiLM conditioning family moved the N=20 ceiling
+0.024 CI [−0.016,+0.064], and the orthogonal 2×2 showed team COUNT dominating conditioning. The
sharper finding underneath: FiLM gradient ANALYSIS showed per-team gradients nearly orthogonal at
300M — the interference these methods exist to prevent was not the binding constraint here.

## 3. The measured anchors (what forgetting actually looks like in this project)

- **Retention ablation**: after teacher retirement, distilled per-team piloting equilibrates at
  0.645 against a 0.438 floor — **~76% of the distilled skill sticks with no life support**.
  Distillation here is bootstrapping, not maintenance. That number is the empirical
  stability-plasticity setting of THIS trunk and should be re-measured as ticks get distill-heavy.
- **Headroom capture ~93%** (D1): the fold transfers nearly everything the teacher had, so the
  loss channel to watch is post-fold DECAY, not fold inefficiency.
- **Gradient conflict instruments exist**: the cosine audit across shaping heads (all orthogonal
  except one harmless −0.088) is the same measurement PCGrad-style methods act on. If tick-era
  distill losses ever conflict pairwise, that audit is the detector and gradient surgery is the
  named intervention — measure first, per the house rule.

## 4. The stability-plasticity frame for the tick-tock

Every continual learner trades stability (keep old skills) against plasticity (learn new ones).
The tick-tock makes the trade EXPLICIT and periodic instead of implicit and continuous: tocks are
maximum-plasticity episodes (fork, specialize, let the generalist skills rot — safely, in a copy),
ticks are stability-weighted episodes (distill + RL with the promotion gate as the stability
audit). The promotion gate's non-inferiority bar is, in this vocabulary, a catastrophic-forgetting
detector at the whole-policy level; per-slice piloting decay is the same detector at task
granularity.

**The question you can answer after this note:** *piloting on an old slice decays two revolutions
after its teacher retired — capacity, interference, or drift?* Interference predicts the decay
correlates with WHICH new slices were folded (check gradient cosines between the slices'
distill losses); capacity predicts uniform decay as total folded skill grows (the H_capacity
signature, and the one that argues for a bigger trunk); drift predicts decay tracking the RL
steps between refreshes regardless of what was folded (argues for more rehearsal, not more
network). Three causes, three different cures, one measurement plan.

Related: the flywheel design (`../ai_v10/design_flywheel_tick_tock.md`) §4-§5; the retention and
capture measurements live in the memory/ledger record cited there.
