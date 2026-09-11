# PRE-REGISTRATION — written 2026-09-10, BEFORE any N-curve number existed

Timestamped by its own commit. The scripts were written, the extraction had run (732,626 states /
23,891 battles on `ctrl10M`; 716,727 / 23,877 on `ctrl10M_b`), and the fits were launched; nothing
had been read.

## The question (the owner's)

Would the win-prob head improve if the policy moved SLOWER — i.e. if non-stationarity were removed
so the head could accumulate data on a fixed target?

## The three readings and their decisions

| | the curve | what it would mean | the treatment |
|---|---|---|---|
| **(i)** | the TERMINAL-label meters RISE with N toward the conditional target's level | non-stationarity / discard is the binding constraint | a slower policy, a value replay, or a stationary window — and the curve says how much per doubling |
| **(ii)** | they stay FLAT from 1k to ~21k while the CONDITIONAL target conditions at every N | the target's NOISE SHARE is the constraint; more stationary data does not help | target-side re-pricing (cf labels, opponent-stratified loss weighting) — the owner's hypothesis is REFUTED for this regime |
| **(iii)** | they rise only past ~10k battles | a data threshold exists but is far above a rollout's supply | quantify the stationary window an online head would need against 48 envs x 2048 steps ~ 3,300 episodes |

Guard, checked FIRST (the head refit's rule): if `cond_oracle` — the conditional target emitted
verbatim — does not itself reach a turn-1–3 ratio whose CI lower bound clears 0.80, the (b) column
is INCONCLUSIVE and is never folded into the reading.

## The prediction

**(ii), with one axis dissenting.** Stated in full so it can be wrong in public:

1. **The turn-1–3 spread ratio of `mlp_term` will be FLAT in N.** The delta between N = 1,000 and
   N ~ 21,000 will straddle zero, and the per-doubling slope will be indistinguishable from 0.
   Reason: the head refit's mechanism is arithmetic and scale-free — only 14.4 % of the terminal
   label's variance lies between cells on this substrate, so a head minimising BCE buys resolution
   from the board and its own team at every n. Nothing about 23x the data changes the price.
2. **The turn-1 opponent-CLASS AUC of `mlp_term` will be FLAT and near its permutation null** at
   every N. (On CTRL the head refit found this axis dead in every condition; the reason it gave —
   a cycle-dominated conditional target — cannot apply here, so this is a genuinely new test of it.)
3. **DISSENT: the turn-1 own-team win-rate R² of `mlp_term` WILL RISE with N.** It already
   recovered 0.053 (CTRL) / 0.318 (A) from 951 battles of pure 0/1 labels, so it is not at a floor;
   the own team is the cheap axis and more data should buy more of it. I expect a detected rise,
   and I expect it to remain well below the conditional refit's 0.467 and `value_pooled`'s 0.628.
   **If the spread ratio and the class AUC move with it, the reading is (i) and I am wrong.**
4. **`mlp_cond` will be DETECTED above `online` on the spread ratio at EVERY N, including 1,000**,
   and will itself rise with N (its label gets better as the factors get more battles).
5. **The MORE-OPTIMISATION control at the largest N will move nothing** on the opponent axis. The
   head refit already eliminated the optimisation-history hazard from the other side (a fine-tune
   of the online weights lands where a scratch fit lands); this closes the remaining "not enough
   passes" reading at 23x the scale.
6. **Brier will improve monotonically with N for `mlp_term`** while the opponent meters do not —
   that dissociation, if it appears, is the mechanism visible in one row: the head is spending the
   extra data on the axis the loss pays for.

## What would change my mind, quantitatively

A per-doubling rise of the turn-1–3 spread ratio of **+0.03 or more** with a CI clear of zero,
sustained over the four doublings 1k → 16k, extrapolates to the ratio reaching the conditional
refit's level within ~4 further doublings (~340k battles) — which IS reachable inside an online
run's lifetime, and would make "slow the policy / replay a stationary window" a real lever rather
than a ruled-out class. That is the number to look for.
