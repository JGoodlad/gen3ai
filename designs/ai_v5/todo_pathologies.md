# AI v5 — Reward & Obs Pathologies to Re-Evaluate Under Self-Play

A focused TODO carried over from the **ai_v4 pathology-hunting phase**. The forensic loss
analysis of `models/run_20260531_182804` (eval steps 17M/18M/19M, 9 fixed-bot opponents) found a
cluster of reward/obs gaps. **One reward fix landed; the rest were deliberately deferred to be
re-evaluated and tuned *with* self-play, not before it.** This doc is the checklist for that
re-evaluation.

Full evidence: `models/run_20260531_182804/LOSS_ANALYSIS_2026-06-02.md` (gitignored, run-local).
Living register: `designs/design_pathologies.md`. Related v5 docs:
`design_reward_annealing.md` (the anneal that interacts with most of these),
`design_selfplay_preflight.md` (the flip checklist).

---

## Why defer to self-play (the sequencing rationale)

The dominant pathology was **all-or-nothing 6-0 play** (135/135 wins sweep, 247/248 losses are
full wipes; no close games) — a *training-regime* problem (8 fixed bots, offense-heavy dense
reward, γ=0.9999), **not** primarily a reward-term bug. Two consequences:

1. **Much of the dense shaping exists to compensate for fixed-bot training.** Terms tuned against
   those bots may not transfer, and self-play changes *which* pathologies even dominate. Tuning the
   reward rebalance now would aim at a target self-play replaces.
2. **It collides with the reward anneal.** `design_reward_annealing.md` drives the strategic
   shaping family (attack/switch/field) → 0 while keeping outcome proxies (HP/faint/win_loss) and
   flooring the anti-degenerate taxes. So **do not over-invest in tuning a shaping term that the
   anneal is scheduled to zero out** — re-evaluate these *after* self-play + the anneal are live,
   and ask the sharper question: once self-play supplies the strategic signal and shaping is
   annealing away, do the outcome proxies + floored taxes suffice?

The same logic parked the **P1b damage-magnitude obs feature**: a symmetric, meaningful
expected-damage + uncertainty signal is only well-posed once the opponent distribution is the
self-play pool.

---

## What already landed (ai_v4 close-out)

- **P2 — explosion/faint reward fix** (`reward_manager.py`, shipped): `finishing_blow` suppressed
  on self-faint; flat asymmetric `FAINT_MATERIAL_PENALTY=0.75` on `faint_ours` only. A healthy
  1-for-1 Explosion trade went **+0.5 → −0.75**; low-for-healthy stays positive. This was the one
  clear, regime-independent fix (a 1-for-1 healthy trade should never be net-positive regardless of
  opponent), so it shipped standalone.

---

## Re-evaluate once `--self-play` is on (and the anneal is running)

Method: re-run the forensic loss pass (per-opponent deep-dive + code-verification, as in the v4
analysis) against **self-play eval traces**; for each item below decide *persists / changed /
dissolved*, then tune against the self-play distribution.

| # | Item | Type | What to check / do |
|---|------|------|--------------------|
| **A** | `FAINT_MATERIAL_PENALTY=0.75` (the P2 fix) | REWARD (tune) | Magnitude is an **untuned judgment call**. Confirm it doesn't over-discourage *legitimate* sacrifices (sac-for-safe-switch-in is correct OU play), and that the asymmetry (~−3.75 extra over a 6-0 loss vs ~−0.75 in a win) doesn't double-count badly with `win_loss=−30`. Note it's in the **anneal's "anti-degenerate / outcome" tier** — keep, don't anneal to 0. |
| **B** | `futile_attack` is **single-turn only** | REWARD | Fires only when opp net-gains HP *on the same turn* (`opp_hp_delta.sum() ≥ 0`), so a chip→Recover/Softboiled **2-turn cycle never trips it**. This is the #1 reward reason it **can't break recovering walls** (the hardest matchups: V2 stallers, bulky Waters). Make it **windowed/heal-aware** (net opp HP over the recovery cycle). Check whether self-play even surfaces this (do mirror walls stall the same way?). |
| **C** | `switch_base` flat **+0.5 for any voluntary switch** | REWARD | Rewards the *act*, not its *value* — pays +0.5 even on a switch into an 86% SE hit. Gate on realized matchup/HP improvement, or fold into `pivot_damage`. **Caveat:** it's also the exploration subsidy that *raises* switch rate (which we want more of) — don't over-gate. **Also slated to anneal → 0** (switch tier), so possibly moot post-anneal. |
| **D** | Wasteful-family magnitudes **~10× too small** | REWARD | `futile_attack −0.05`, `status_wasted −0.3`, `futile_setup −0.3` (cap-only), `setup_low_hp −0.10` (<40% HP), `matchup_penalty −0.15`/turn — all dwarfed by `HP_VALUE=2.0`/bar; the policy demonstrably absorbs them. Rescale up / escalate faster / extend **setup discipline mid-stack** (penalize Calm Mind/DD while not progressing / down material / last-mon). **But:** most of these are in the **anneal's strategic-shaping tier (→ 0)** — re-evaluate whether they're even needed once self-play + anneal are live, rather than rescaling them now. |
| **E** | **No positive forward-progress / closing term** | REWARD | Every offense reward is per-HP-chip; nothing rewards *converting* — breaking a heal-stalemate or securing the last opponent mon. Consider a small term; watch the anneal interaction (a "win-probability" value head is the v6/MCTS goal, so prefer outcome-aligned signals over more shaping). |
| **F** | **P1b — action-aligned damage-magnitude obs** | INFO (retrain-class) | The head can't rank its own moves by damage or sense an OHKO (only base-power + type-mult are action-aligned). v0 sketch: 12-dim **`[is_physical, is_special, our_offensive_stat]`** per move slot, **pre-item** so Choice Band stays emergent, **no opponent priors** (let the model learn species bulk/spreads). Design the symmetric (our→them / them→us) + uncertainty version **only once self-play defines the opponent distribution.** See `feedback_provide_vs_learn` principle. ARCH bump + obs-build benchmark gate. **→ PARTIALLY LANDED (`impl_step4_incoming_damage_obs.md`, `gen3_incoming_damage_v1`):** the **incoming / them→us** direction shipped as a calibrated KO **belief** (P(KO)+expected-chip+P(outspeed) per our mon, prior-aware — the design deliberately *uses* opponent priors here because their set is hidden; CB stays emergent/deferred). **Still open:** the **our→them action-aligned** direction (let the policy head rank its own moves by damage) and Gate-2 efficacy. |

---

## Exit / done criteria

This doc is closed when, **after self-play + the reward anneal are live**, the forensic pass shows:
close games appear (3-2 / 2-1 outcomes), voluntary-switch rate rises, the can't-break-walls
texture (B) is gone, and the V2 setup/staller win rates close the gap to their V1 versions — with
each item above marked *persists-and-fixed*, *dissolved-by-self-play*, or *annealed-away*.
